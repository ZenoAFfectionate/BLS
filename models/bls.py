"""Standard Broad Learning System (BLS) implementation.

Reference
---------
Chen, C. L. P., & Liu, Z. (2017). Broad Learning System:
An Effective and Efficient Incremental Learning System Without the
Need for Deep Architecture. IEEE TNNLS.
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


class BLS(BaseEstimator):
    """Broad Learning System.

    Constructs a wide hidden representation via random feature mappings
    and orthogonally-constrained enhancement nodes, then learns the
    linear readout layer in closed-form via ridge regression.

    Parameters
    ----------
    feature_times : int
        Number of feature-node groups.
    enhance_times : int
        Number of enhancement-node groups.  Each group is orthogonalized
        (Gram-Schmidt).
    n_classes : int
        Number of target classes.
    mapping_function : str
        Activation for feature nodes (default ``"linear"``).
    enhance_function : str
        Activation for enhancement nodes (default ``"tanh"``).
    feature_size : int or ``"auto"``
        Width per feature-node group.  ``"auto"`` uses ``input_dim``.
    reg : float
        L2 regularization strength (ridge parameter lambda).
    use_sparse : bool
        If True, use a sparse autoencoder shortcut for feature nodes.
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

        # Store any extra kwargs (e.g. sig) for serialization compat
        self._extra_kwargs = kwargs

        self.mapping_generator = NodeGenerator(mapping_function)
        self.enhance_generator = NodeGenerator(
            enhance_function,
            orthogonalize_output=True,
        )

        self.W = None
        self.is_fitted = False
        self._mapping_nodes = None

    # ------------------------------------------------------------------
    #  Solver helpers
    # ------------------------------------------------------------------

    def _ridge_solve(self, A, B):
        """Solve  (A^T A + λI) W = A^T B  via Cholesky (dual/primal)."""
        A64 = np.asarray(A, dtype=np.float64)
        B64 = np.asarray(B, dtype=np.float64)
        n, m = A64.shape
        lam = float(self.reg)

        if n < m:  # dual form  →  (A A^T + λI)^{-1} A B
            M = A64 @ A64.T + lam * np.eye(n, dtype=np.float64)
            try:
                L = cholesky(M, lower=True)
                y = solve_triangular(L, B64, lower=True)
                y = solve_triangular(L.T, y, lower=False)
                result = A64.T @ y
            except Exception:
                logger.debug("Cholesky failed – falling back to SVD (dual).")
                U, s, Vt = np.linalg.svd(A64, full_matrices=False)
                s_reg = s / (s ** 2 + lam)
                result = Vt.T @ (s_reg[:, None] * (U.T @ B64))
        else:         # primal form →  (A^T A + λI)^{-1} A^T B
            M = A64.T @ A64 + lam * np.eye(m, dtype=np.float64)
            try:
                L = cholesky(M, lower=True)
                ATB = A64.T @ B64
                y = solve_triangular(L, ATB, lower=True)
                result = solve_triangular(L.T, y, lower=False)
            except Exception:
                logger.debug("Cholesky failed – falling back to SVD (primal).")
                U, s, Vt = np.linalg.svd(A64, full_matrices=False)
                s_reg = s / (s ** 2 + lam)
                result = Vt.T @ (s_reg[:, None] * (U.T @ B64))

        return result.astype(DEFAULT_DTYPE, copy=False)

    def _compute_pinv(self, A):
        """Ridge pseudo-inverse: (A^T A + λI)^{-1} A^T."""
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

    def _encode_targets(self, y):
        """Convert integer labels to one-hot."""
        y = np.asarray(y)
        if y.ndim == 1:
            return np.eye(int(self.n_classes), dtype=DEFAULT_DTYPE)[y.astype(np.int64)]
        y_enc = np.asarray(y, dtype=DEFAULT_DTYPE)
        if y_enc.shape[1] != int(self.n_classes):
            raise ValueError("One-hot targets must have n_classes columns.")
        return y_enc

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train the BLS model."""
        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        y = np.asarray(y)
        if X.shape[0] == 0 or y.size == 0:
            raise ValueError("Empty input data passed to fit().")

        # --- input validation (7.1) ---
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2, got %d" % self.n_classes)
        feat_size = (
            int(X.shape[1]) if str(self.feature_size) == "auto"
            else int(self.feature_size)
        )
        if feat_size <= 0:
            raise ValueError("feature_size must be a positive integer, got %s"
                             % self.feature_size)

        if self.is_fitted:
            self.reset()

        y_labels = np.asarray(y, dtype=np.int64)
        if y_labels.ndim != 1:
            y_labels = np.argmax(y_labels, axis=1)

        if str(self.feature_size) == "auto":
            self.feature_size = int(X.shape[1])
        feat_size = int(self.feature_size)

        # 1. Feature / mapping nodes
        mapping_nodes = self.mapping_generator.generate_nodes(
            X, feat_size, int(self.feature_times)
        )
        self._mapping_nodes = mapping_nodes

        # Optional sparse shortcut
        if self.use_sparse:
            pinvX = self._compute_pinv(X)
            self.mapping_generator.spW = pinvX @ mapping_nodes

        # 2. Enhancement nodes (orthogonalized per group)
        enhance_nodes = self.enhance_generator.generate_nodes(
            mapping_nodes, feat_size, int(self.enhance_times)
        )

        # 3. Concatenate & solve
        A = np.hstack((mapping_nodes, enhance_nodes)).astype(DEFAULT_DTYPE, copy=False)
        Y = self._encode_targets(y_labels)

        # Store Cholesky factor for efficient incremental updates (2.2)
        A64 = np.asarray(A, dtype=np.float64)
        M = A64.T @ A64 + float(self.reg) * np.eye(A64.shape[1], dtype=np.float64)
        self._chol_L = cholesky(M, lower=True)
        self._A_for_incremental = A  # for incremental update reference
        self._Y_for_incremental = Y

        self.W = self._ridge_solve(A, Y)
        self.is_fitted = True
        return self

    def _incremental_ridge_update(self, A_old, H_new, Y, chol_L):
        """Efficiently update ridge solution when appending columns H_new.

        Uses the Schur-complement / block-matrix inversion approach to
        avoid a full O(m³) re-solve.  Only a k×k matrix (k = cols of
        H_new) needs to be factorized, and the stored Cholesky factor of
        A_old^T A_old + λI is reused.

        Complexity: O(N·m·k + m²·k + k³)  vs  O((m+k)³) for full re-solve.
        """

        A64 = np.asarray(A_old, dtype=np.float64)
        H64 = np.asarray(H_new, dtype=np.float64)
        Y64 = np.asarray(Y, dtype=np.float64)
        lam = float(self.reg)

        m_old = A64.shape[1]
        k = H64.shape[1]

        # Solve  V = (A_old^T A_old + λI)⁻¹ (A_old^T H_new)   via stored Cholesky
        ATH = A64.T @ H64  # m_old × k
        V = cho_solve((chol_L, True), ATH)  # m_old × k

        #  Schur complement  S = H_new^T H_new + λI - H_new^T A_old V
        HTH = H64.T @ H64  # k × k
        S = HTH + lam * np.eye(k, dtype=np.float64) - H64.T @ (A64 @ V)
        try:
            L_s = cholesky(S, lower=True)
        except Exception:
            # Fallback — nearly singular S → direct solve
            W2 = np.linalg.solve(S, H64.T @ (Y64 - A64 @ self.W))
            W1 = self.W.astype(np.float64) - V @ W2
            return np.vstack([W1, W2]).astype(DEFAULT_DTYPE, copy=False)

        # W₂ = S⁻¹ H_new^T (Y - A_old W_old)
        residual = Y64 - A64 @ np.asarray(self.W, dtype=np.float64)
        r = H64.T @ residual
        W2 = solve_triangular(L_s.T, solve_triangular(L_s, r, lower=True), lower=False)

        # W₁ = W_old - V W₂
        W1 = np.asarray(self.W, dtype=np.float64) - V @ W2

        return np.vstack([W1, W2]).astype(DEFAULT_DTYPE, copy=False)

    def add_enhancement_nodes(self, X, y, num_nodes=5):
        """Add new enhancement nodes incrementally.

        Uses an efficient Schur-complement update (instead of full
        re-solve) when a stored Cholesky factor is available.
        Falls back to full re-solve for the first incremental call.
        """
        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        num_nodes = int(num_nodes)
        if num_nodes <= 0:
            return self

        y_labels = np.asarray(y, dtype=np.int64)
        if y_labels.ndim != 1:
            y_labels = np.argmax(y_labels, axis=1)
        Y = self._encode_targets(y_labels)

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
            self.W = self._ridge_solve(A, Y)

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
        self._A_for_incremental = None
        self._Y_for_incremental = None
