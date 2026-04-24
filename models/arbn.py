import numpy as np

from scipy.linalg import cholesky, solve_triangular
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    ),
    "tanh": lambda x: np.tanh(x),
    "leaky_relu": lambda x: np.maximum(0.01 * x, x),
    "swish": lambda x: x * (1 / (1 + np.exp(-x))),
    "mish": lambda x: x * np.tanh(np.log1p(np.exp(x))),
    "hard_sigmoid": lambda x: np.clip(0.2 * x + 0.5, 0, 1),
    "hard_swish": lambda x: x * np.clip(0.2 * x + 0.5, 0, 1),
}


DEFAULT_DTYPE = np.float32


def _resolve_optional_param(value, default):
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return default
    return value


class NodeGenerator:
    """Node generator for Broad Learning System."""

    def __init__(self, activation="relu", whiten=False, orthogonalize_output=False):
        self.Wlist = []
        self.blist = []
        self.whiten = whiten
        self.orthogonalize_output = orthogonalize_output
        self.output_transform_list = []
        self.spW = None
        self._cached_transform_params = None
        self.activation = activation

    @staticmethod
    def orth(W):
        if W.shape[0] < W.shape[1]:
            return W
        Q, _ = np.linalg.qr(np.asarray(W, dtype=np.float64), mode="reduced")
        return Q[:, : W.shape[1]].astype(W.dtype, copy=False)

    @staticmethod
    def fit_orthogonal_output_transform(H):
        if H.shape[0] < H.shape[1]:
            raise ValueError(
                "Enhancement group output cannot be orthogonalized when "
                "num_samples < group_width."
            )
        H64 = np.asarray(H, dtype=np.float64)
        # QR provides a numerically stable Gram-Schmidt equivalent.
        _, R = np.linalg.qr(H64, mode="reduced")
        identity = np.eye(R.shape[0], dtype=np.float64)
        try:
            transform = solve_triangular(R, identity, lower=False)
        except Exception:
            transform = np.linalg.pinv(R)
        transformed = H64 @ transform
        return (
            transformed.astype(DEFAULT_DTYPE, copy=False),
            transform.astype(DEFAULT_DTYPE, copy=False),
        )

    def generate_nodes(self, data, feature_size, times):
        self.Wlist = []
        self.blist = []
        self.output_transform_list = []
        self._cached_transform_params = None
        input_dim = data.shape[1]
        dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else DEFAULT_DTYPE

        outputs = []
        for _ in range(int(times)):
            W = np.random.uniform(-1, 1, size=(input_dim, feature_size)).astype(
                dtype,
                copy=False,
            )
            if self.whiten:
                W = self.orth(W)
            b = np.random.uniform(-0.5, 0.5, size=(feature_size,)).astype(
                dtype,
                copy=False,
            )
            self.Wlist.append(W)
            self.blist.append(b)
            if self.orthogonalize_output:
                H = ACTIVATIONS[self.activation](data @ W + b)
                H, transform = self.fit_orthogonal_output_transform(H)
                self.output_transform_list.append(transform)
                outputs.append(H)

        if self.orthogonalize_output:
            return np.hstack(outputs).astype(DEFAULT_DTYPE, copy=False)

        return self.transform(data)

    def transform(self, X):
        if self.spW is not None and not self.orthogonalize_output:
            return ACTIVATIONS[self.activation](X @ self.spW)

        if not self.Wlist:
            return np.zeros((X.shape[0], 0), dtype=DEFAULT_DTYPE)

        if not self.orthogonalize_output:
            if self._cached_transform_params is None:
                W_big = np.hstack(self.Wlist)
                b_big = np.concatenate(self.blist, axis=0)
                self._cached_transform_params = (W_big, b_big)
            else:
                W_big, b_big = self._cached_transform_params
            return ACTIVATIONS[self.activation](X @ W_big + b_big).astype(
                DEFAULT_DTYPE,
                copy=False,
            )

        outputs = []
        for idx, (W, b) in enumerate(zip(self.Wlist, self.blist)):
            H = ACTIVATIONS[self.activation](X @ W + b)
            if idx >= len(self.output_transform_list):
                raise RuntimeError(
                    "Orthogonal output transforms are missing. Refit the ARBN model."
                )
            transform = np.asarray(self.output_transform_list[idx], dtype=np.float64)
            H = np.asarray(H, dtype=np.float64) @ transform
            outputs.append(H.astype(DEFAULT_DTYPE, copy=False))
        return np.hstack(outputs).astype(DEFAULT_DTYPE, copy=False)

    def update(self, otherW, otherb, other_transforms=None):
        self.Wlist += otherW
        self.blist += otherb
        if self.orthogonalize_output:
            if other_transforms is None or len(other_transforms) != len(otherW):
                raise ValueError(
                    "Orthogonalized node updates must provide matching output transforms."
                )
            self.output_transform_list += other_transforms
        self._cached_transform_params = None


class ARBN(BaseEstimator):
    """Adaptive Re-weighting Broad Network (ARBN)."""

    def __init__(
        self,
        feature_times=10,
        enhance_times=10,
        n_classes=10,
        mapping_function="linear",
        enhance_function="tanh",
        feature_size="auto",
        reg=0.01,
        sig=0.01,
        use_sparse=False,
        cls_num_list=None,
        adaptive_reg=True,
        class_weight_beta=0.5,
    ):
        self.feature_times = feature_times
        self.enhance_times = enhance_times
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.mapping_function = mapping_function
        self.enhance_function = enhance_function
        self.reg = reg
        self.sig = sig
        self.use_sparse = use_sparse
        self.cls_num_list = cls_num_list
        self.adaptive_reg = adaptive_reg
        self.class_weight_beta = float(_resolve_optional_param(class_weight_beta, 0.5))

        if self.cls_num_list is not None:
            cls_num_array = np.asarray(self.cls_num_list, dtype=np.float64)
            if cls_num_array.shape[0] != int(self.n_classes):
                raise ValueError(
                    "Length of cls_num_list must match n_classes for ARBN weighting."
                )
            if np.any(cls_num_array <= 0):
                raise ValueError("cls_num_list must contain strictly positive counts.")
            self.class_weights = np.power(1.0 / cls_num_array, self.class_weight_beta).astype(
                DEFAULT_DTYPE,
                copy=False,
            )
        else:
            self.class_weights = np.ones(int(self.n_classes), dtype=DEFAULT_DTYPE)

        self.mapping_generator = NodeGenerator(mapping_function)
        self.enhance_generator = NodeGenerator(
            enhance_function,
            orthogonalize_output=True,
        )

        self.W = None
        self.is_fitted = False
        self._mapping_nodes = None

    def _encode_targets(self, y, y_labels):
        if y.ndim == 1:
            return np.eye(int(self.n_classes), dtype=DEFAULT_DTYPE)[y_labels]

        y_encoded = np.asarray(y, dtype=DEFAULT_DTYPE)
        if y_encoded.shape[1] != int(self.n_classes):
            raise ValueError(
                "One-hot targets must have the same second dimension as n_classes."
            )
        return y_encoded

    def compute_pinv(self, A):
        A64 = np.asarray(A, dtype=np.float64)
        n, m = A64.shape

        if n < m:
            M = A64 @ A64.T + float(self.reg) * np.eye(n, dtype=np.float64)
            try:
                L = cholesky(M, lower=True)
                identity = np.eye(n, dtype=np.float64)
                inv_M = solve_triangular(L, identity, lower=True)
                inv_M = solve_triangular(L.T, inv_M, lower=False)
                result = A64.T @ inv_M
            except Exception:
                result = A64.T @ np.linalg.pinv(M)
        else:
            M = A64.T @ A64 + float(self.reg) * np.eye(m, dtype=np.float64)
            try:
                L = cholesky(M, lower=True)
                result = solve_triangular(L, A64.T, lower=True)
                result = solve_triangular(L.T, result, lower=False)
            except Exception:
                result = np.linalg.solve(M, A64.T)

        return result.astype(DEFAULT_DTYPE, copy=False)

    def _solve_weighted_ridge(self, A, B, sample_weights=None):
        A64 = np.asarray(A, dtype=np.float64)
        B64 = np.asarray(B, dtype=np.float64)
        hidden_dim = A64.shape[1]

        if sample_weights is None:
            lhs = A64.T @ A64
            rhs = A64.T @ B64
        else:
            weights64 = np.asarray(sample_weights, dtype=np.float64)
            if weights64.shape[0] != A64.shape[0]:
                raise ValueError("sample_weights length must match the number of samples.")
            weighted_A = weights64[:, None] * A64
            lhs = A64.T @ weighted_A
            rhs = A64.T @ (weights64[:, None] * B64)

        lhs = lhs + float(self.reg) * np.eye(
            hidden_dim,
            dtype=np.float64,
        )

        try:
            L = cholesky(lhs, lower=True)
            y = solve_triangular(L, rhs, lower=True)
            solution = solve_triangular(L.T, y, lower=False)
        except Exception:
            solution = np.linalg.solve(lhs, rhs)

        return solution.astype(DEFAULT_DTYPE, copy=False)

    def ridge_solve(self, A, B):
        return self._solve_weighted_ridge(A, B, sample_weights=None)

    def ridge_solve_adaptive(self, A, B, y_labels=None):
        if not self.adaptive_reg or y_labels is None or self.cls_num_list is None:
            return self.ridge_solve(A, B)
        y_labels = np.asarray(y_labels, dtype=np.int64)
        sample_weights = self.class_weights[y_labels]
        return self._solve_weighted_ridge(A, B, sample_weights=sample_weights)

    def fit(self, X, y):
        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        y = np.asarray(y)

        if self.is_fitted:
            self.reset()

        y_labels = y.astype(np.int64, copy=False) if y.ndim == 1 else np.argmax(y, axis=1)
        y_labels = np.asarray(y_labels, dtype=np.int64)

        if self.feature_size == "auto":
            self.feature_size = int(X.shape[1])
        feature_size = int(self.feature_size)

        mapping_nodes = self.mapping_generator.generate_nodes(
            X,
            feature_size,
            int(self.feature_times),
        )
        self._mapping_nodes = mapping_nodes

        if self.use_sparse:
            pinvX = self.compute_pinv(X)
            self.mapping_generator.spW = pinvX @ mapping_nodes

        enhance_nodes = self.enhance_generator.generate_nodes(
            mapping_nodes,
            feature_size,
            int(self.enhance_times),
        )

        A = np.hstack((mapping_nodes, enhance_nodes)).astype(DEFAULT_DTYPE, copy=False)
        y_encoded = self._encode_targets(y, y_labels)
        self.W = self.ridge_solve_adaptive(A, y_encoded, y_labels)
        self.is_fitted = True
        return self

    def add_enhancement_nodes(self, X, y, num_nodes=5):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        y = np.asarray(y)
        y_labels = y.astype(np.int64, copy=False) if y.ndim == 1 else np.argmax(y, axis=1)
        y_labels = np.asarray(y_labels, dtype=np.int64)
        y_encoded = self._encode_targets(y, y_labels)

        mapping_nodes = self._mapping_nodes
        if mapping_nodes is None or mapping_nodes.shape[0] != X.shape[0]:
            mapping_nodes = self.mapping_generator.transform(X)
            self._mapping_nodes = mapping_nodes

        current_enhance = self.enhance_generator.transform(mapping_nodes)
        num_nodes = int(num_nodes)
        if num_nodes <= 0:
            return self
        if mapping_nodes.shape[0] < num_nodes:
            raise ValueError(
                "num_nodes must not exceed the number of samples when "
                "enhancement outputs are orthogonalized."
            )

        input_dim = mapping_nodes.shape[1]
        W_new = np.random.uniform(-1, 1, size=(input_dim, num_nodes)).astype(
            DEFAULT_DTYPE,
            copy=False,
        )
        b_new = np.random.uniform(-0.5, 0.5, size=(num_nodes,)).astype(
            DEFAULT_DTYPE,
            copy=False,
        )
        new_nodes = ACTIVATIONS[self.enhance_generator.activation](mapping_nodes @ W_new + b_new)
        new_nodes, new_transform = self.enhance_generator.fit_orthogonal_output_transform(
            new_nodes
        )

        self.enhance_generator.update([W_new], [b_new], [new_transform])

        A = np.hstack((mapping_nodes, current_enhance, new_nodes)).astype(
            DEFAULT_DTYPE,
            copy=False,
        )
        self.W = self.ridge_solve_adaptive(A, y_encoded, y_labels)
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        mapping_nodes = self.mapping_generator.transform(X)
        enhance_nodes = self.enhance_generator.transform(mapping_nodes)
        A = np.hstack((mapping_nodes, enhance_nodes)).astype(DEFAULT_DTYPE, copy=False)
        logits = A @ self.W

        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def reset(self):
        self.W = None
        self.mapping_generator = NodeGenerator(self.mapping_function)
        self.enhance_generator = NodeGenerator(
            self.enhance_function,
            orthogonalize_output=True,
        )
        self.is_fitted = False
        self._mapping_nodes = None

    def evaluate_imbalanced(self, X, y, average="macro"):
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average=average)
        recall = recall_score(y, y_pred, average=average)
        f1 = f1_score(y, y_pred, average=average)

        if self.n_classes == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class="ovr")

        print("Classification Report:")
        print(classification_report(y, y_pred))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }


BLS = ARBN  # deprecated, kept for pickle compatibility
