"""Adaptive Re-weighting Broad Network (ARBN).

Extension of BLS that counteracts long-tailed class distributions by
injecting class-aware sample weights directly into the closed-form
ridge solver.
"""

import logging
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from .node_generator import (
    DEFAULT_DTYPE,
    ACTIVATIONS,
    NodeGenerator,
)

logger = logging.getLogger(__name__)


class ARBN(BaseEstimator):
    """Adaptive Re-weighting Broad Network.

    Parameters
    ----------
    feature_times, enhance_times, n_classes, mapping_function,
    enhance_function, feature_size, reg, use_sparse :
        Identical to :class:`BLS`.
    cls_num_list : list of int, optional
        Per-class sample counts for weight computation.
    adaptive_reg : bool
        Enable adaptive sample weighting (ARBN mode).
    class_weight_beta : float
        Exponent in ``w_k = (1 / |D_k|)^beta`` (0 = uniform, 1 = inverse freq).
    """

    def __init__(
        self,
        feature_times=10,
        enhance_times=10,
        n_classes=10,
        mapping_function="linear",
        enhance_function="tanh",
        feature_size="auto",
        reg=0.01,
        use_sparse=False,
        cls_num_list=None,
        adaptive_reg=True,
        class_weight_beta=0.5,
        **kwargs,
    ):
        self.feature_times = feature_times
        self.enhance_times = enhance_times
        self.n_classes = n_classes
        self.mapping_function = mapping_function
        self.enhance_function = enhance_function
        self.feature_size = feature_size
        self.reg = reg
        self.use_sparse = use_sparse
        self.cls_num_list = cls_num_list
        self.adaptive_reg = adaptive_reg
        self.class_weight_beta = float(
            class_weight_beta if class_weight_beta is not None else 0.5
        )

        self._extra_kwargs = kwargs

        # Compute class weights
        if self.cls_num_list is not None:
            cls_arr = np.asarray(self.cls_num_list, dtype=np.float64)
            if cls_arr.shape[0] != int(self.n_classes):
                raise ValueError(
                    "Length of cls_num_list must match n_classes."
                )
            if np.any(cls_arr <= 0):
                raise ValueError(
                    "cls_num_list must contain strictly positive counts."
                )
            self.class_weights = np.power(
                1.0 / cls_arr, self.class_weight_beta
            ).astype(DEFAULT_DTYPE, copy=False)
        else:
            self.class_weights = np.ones(int(self.n_classes), dtype=DEFAULT_DTYPE)

        # Generators
        self.mapping_generator = NodeGenerator(mapping_function)
        self.enhance_generator = NodeGenerator(
            enhance_function,
            orthogonalize_output=True,
        )

        self.W = None
        self.is_fitted = False
        self._mapping_nodes = None

    # ------------------------------------------------------------------
    #  Solver
    # ------------------------------------------------------------------

    def _encoded_targets(self, y, y_labels):
        if y.ndim == 1:
            return np.eye(int(self.n_classes), dtype=DEFAULT_DTYPE)[y_labels]
        y_enc = np.asarray(y, dtype=DEFAULT_DTYPE)
        if y_enc.shape[1] != int(self.n_classes):
            raise ValueError("One-hot targets must have n_classes columns.")
        return y_enc

    def _weighted_ridge_solve(self, A, B, sample_weights=None):
        """Solve  (A^T W A + λI) W* = A^T W B."""
        A64 = np.asarray(A, dtype=np.float64)
        B64 = np.asarray(B, dtype=np.float64)
        hidden_dim = A64.shape[1]
        lam = float(self.reg)

        if sample_weights is None:
            lhs = A64.T @ A64
            rhs = A64.T @ B64
        else:
            w64 = np.asarray(sample_weights, dtype=np.float64)
            if w64.shape[0] != A64.shape[0]:
                raise ValueError(
                    "sample_weights length must match number of samples."
                )
            lhs = A64.T @ (w64[:, None] * A64)
            rhs = A64.T @ (w64[:, None] * B64)

        lhs = lhs + lam * np.eye(hidden_dim, dtype=np.float64)

        try:
            L = cholesky(lhs, lower=True)
            y = solve_triangular(L, rhs, lower=True)
            solution = solve_triangular(L.T, y, lower=False)
        except Exception:
            solution = np.linalg.solve(lhs, rhs)

        return solution.astype(DEFAULT_DTYPE, copy=False)

    def ridge_solve(self, A, B):
        """Standard unweighted ridge solve."""
        return self._weighted_ridge_solve(A, B)

    def ridge_solve_adaptive(self, A, B, y_labels=None):
        """Adaptively weighted ridge solve (ARBN core)."""
        if not self.adaptive_reg or y_labels is None or self.cls_num_list is None:
            return self.ridge_solve(A, B)
        y_labels = np.asarray(y_labels, dtype=np.int64)
        if np.any(y_labels < 0) or np.any(y_labels >= int(self.n_classes)):
            raise ValueError(
                "y_labels must be in [0, n_classes).  "
                f"Got values outside [0, {self.n_classes})."
            )
        return self._weighted_ridge_solve(
            A, B, sample_weights=self.class_weights[y_labels]
        )

    def _compute_pinv(self, A):
        """Ridge pseudo-inverse (for sparse shortcut)."""
        A64 = np.asarray(A, dtype=np.float64)
        n, m = A64.shape
        lam = float(self.reg)

        if n < m:
            M = A64 @ A64.T + lam * np.eye(n, dtype=np.float64)
            try:
                L = cholesky(M, lower=True)
                I_n = np.eye(n, dtype=np.float64)
                inv_tri = solve_triangular(L, I_n, lower=True)
                inv_M = solve_triangular(L.T, inv_tri, lower=False)
                result = A64.T @ inv_M
            except Exception:
                result = A64.T @ np.linalg.pinv(M)
        else:
            M = A64.T @ A64 + lam * np.eye(m, dtype=np.float64)
            try:
                L = cholesky(M, lower=True)
                result = solve_triangular(L, A64.T, lower=True)
                result = solve_triangular(L.T, result, lower=False)
            except Exception:
                result = np.linalg.solve(M, A64.T)

        return result.astype(DEFAULT_DTYPE, copy=False)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train the ARBN model."""
        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        y = np.asarray(y)
        if X.shape[0] == 0 or y.size == 0:
            raise ValueError("Empty input data passed to fit().")

        # --- input validation (7.1) ---
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2, got %d" % self.n_classes)

        if self.is_fitted:
            self.reset()

        y_labels = (
            y.astype(np.int64, copy=False)
            if y.ndim == 1
            else np.argmax(y, axis=1)
        )
        y_labels = np.asarray(y_labels, dtype=np.int64)

        if str(self.feature_size) == "auto":
            self.feature_size = int(X.shape[1])
        feat_size = int(self.feature_size)
        if feat_size <= 0:
            raise ValueError("feature_size must be a positive integer, got %s"
                             % self.feature_size)

        mapping_nodes = self.mapping_generator.generate_nodes(
            X, feat_size, int(self.feature_times)
        )
        self._mapping_nodes = mapping_nodes

        if self.use_sparse:
            pinvX = self._compute_pinv(X)
            self.mapping_generator.spW = pinvX @ mapping_nodes

        enhance_nodes = self.enhance_generator.generate_nodes(
            mapping_nodes, feat_size, int(self.enhance_times)
        )

        A = np.hstack((mapping_nodes, enhance_nodes)).astype(DEFAULT_DTYPE, copy=False)
        Y = self._encoded_targets(y, y_labels)

        # Store Cholesky factor for efficient incremental updates (2.2)
        A64 = np.asarray(A, dtype=np.float64)
        M = A64.T @ A64 + float(self.reg) * np.eye(A64.shape[1], dtype=np.float64)
        self._chol_L = cholesky(M, lower=True)

        self.W = self.ridge_solve_adaptive(A, Y, y_labels)
        self.is_fitted = True
        return self

    def _incremental_ridge_update(self, A_old, H_new, Y, chol_L):
        """Efficiently update ridge solution when appending columns H_new.

        Uses the Schur-complement / block-matrix inversion approach.
        Only a k×k matrix (k = cols of H_new) needs to be factorized.
        """

        A64 = np.asarray(A_old, dtype=np.float64)
        H64 = np.asarray(H_new, dtype=np.float64)
        Y64 = np.asarray(Y, dtype=np.float64)
        lam = float(self.reg)

        m_old = A64.shape[1]
        k = H64.shape[1]

        # Solve  V = (A_old^T A_old + λI)⁻¹ (A_old^T H_new)
        ATH = A64.T @ H64
        V = cho_solve((chol_L, True), ATH)

        # Schur complement
        HTH = H64.T @ H64
        S = HTH + lam * np.eye(k, dtype=np.float64) - H64.T @ (A64 @ V)

        try:
            L_s = cholesky(S, lower=True)
        except Exception:
            W2 = np.linalg.solve(S, H64.T @ (Y64 - A64 @ self.W))
            W1 = self.W.astype(np.float64) - V @ W2
            return np.vstack([W1, W2]).astype(DEFAULT_DTYPE, copy=False)

        residual = Y64 - A64 @ np.asarray(self.W, dtype=np.float64)
        r = H64.T @ residual
        W2 = solve_triangular(L_s.T, solve_triangular(L_s, r, lower=True), lower=False)
        W1 = np.asarray(self.W, dtype=np.float64) - V @ W2

        return np.vstack([W1, W2]).astype(DEFAULT_DTYPE, copy=False)

    def add_enhancement_nodes(self, X, y, num_nodes=5):
        """Add new enhancement nodes incrementally.

        Uses an efficient Schur-complement update when a stored Cholesky
        factor is available; falls back to full re-solve otherwise.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        y = np.asarray(y)

        num_nodes = int(num_nodes)
        if num_nodes <= 0:
            return self

        y_labels = (
            y.astype(np.int64, copy=False)
            if y.ndim == 1
            else np.argmax(y, axis=1)
        )
        y_labels = np.asarray(y_labels, dtype=np.int64)
        Y = self._encoded_targets(y, y_labels)

        mapping_nodes = self._mapping_nodes
        if mapping_nodes is None or mapping_nodes.shape[0] != X.shape[0]:
            mapping_nodes = self.mapping_generator.transform(X)
            self._mapping_nodes = mapping_nodes

        if mapping_nodes.shape[0] < num_nodes:
            raise ValueError(
                "num_nodes must not exceed the number of samples when "
                "enhancement outputs are orthogonalized."
            )

        current_enhance = self.enhance_generator.transform(mapping_nodes)
        input_dim = mapping_nodes.shape[1]

        W_new = np.random.uniform(-1.0, 1.0, size=(input_dim, num_nodes)).astype(
            DEFAULT_DTYPE, copy=False
        )
        b_new = np.random.uniform(-0.5, 0.5, size=(num_nodes,)).astype(
            DEFAULT_DTYPE, copy=False
        )

        new_nodes = ACTIVATIONS[self.enhance_generator.activation](
            mapping_nodes @ W_new + b_new
        )
        new_nodes, new_transform = self.enhance_generator.fit_orthogonal_output_transform(
            new_nodes
        )

        self.enhance_generator.update([W_new], [b_new], [new_transform])

        # ---- incremental vs full re-solve (2.2) ----
        A_old = np.hstack((mapping_nodes, current_enhance)).astype(
            DEFAULT_DTYPE, copy=False
        )
        if hasattr(self, "_chol_L") and self._chol_L is not None:
            # Use Schur-complement incremental update
            self.W = self._incremental_ridge_update(A_old, new_nodes, Y, self._chol_L)
            # Update stored Cholesky factor for future incremental calls
            A_new = np.hstack((A_old, new_nodes))
            A_new64 = np.asarray(A_new, dtype=np.float64)
            M = A_new64.T @ A_new64 + float(self.reg) * np.eye(
                A_new64.shape[1], dtype=np.float64
            )
            self._chol_L = cholesky(M, lower=True)
        else:
            # Fallback — full re-solve
            A = np.hstack((A_old, new_nodes)).astype(DEFAULT_DTYPE, copy=False)
            self.W = self.ridge_solve_adaptive(A, Y, y_labels)

        return self

    def predict_proba(self, X):
        """Softmax class probabilities."""
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
        """Hard class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)

    def reset(self):
        """Reset model state for re-training."""
        self.W = None
        self.mapping_generator = NodeGenerator(self.mapping_function)
        self.enhance_generator = NodeGenerator(
            self.enhance_function,
            orthogonalize_output=True,
        )
        self.is_fitted = False
        self._mapping_nodes = None
        self._chol_L = None

    def evaluate_imbalanced(self, X, y, average="macro"):
        """Compute accuracy, precision, recall, F1, and AUC."""
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

        unique_labels = np.unique(y)
        n_unique = len(unique_labels)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average=average, zero_division=0)
        recall = recall_score(y, y_pred, average=average, zero_division=0)
        f1 = f1_score(y, y_pred, average=average, zero_division=0)

        # --- 7.2: Safe AUC for edge cases (single-class, etc.) ---
        try:
            if n_unique < 2:
                auc = float("nan")  # AUC undefined for < 2 classes
            elif self.n_classes == 2:
                auc = float(roc_auc_score(y, y_proba[:, 1]))
            else:
                auc = float(roc_auc_score(y, y_proba, multi_class="ovr"))
        except ValueError:
            auc = float("nan")

        logger.info("Classification Report:\n%s", classification_report(y, y_pred))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }


# Deprecated alias kept for pickle backward-compatibility only.
# New code should import BLS from models.bls, not from models.arbn.
BLS = ARBN
