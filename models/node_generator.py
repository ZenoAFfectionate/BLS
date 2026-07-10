"""Shared NodeGenerator and activation functions for BLS models."""

import warnings
import numpy as np
from scipy.linalg import solve_triangular

# ---------------------------------------------------------------------------
# Numerically stable activation functions
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "linear":      lambda x: x,
    "relu":        lambda x: np.maximum(0, x),
    "sigmoid":     lambda x: np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    ),
    "tanh":        lambda x: np.tanh(x),
    "leaky_relu":  lambda x: np.maximum(0.01 * x, x),
    "swish":       lambda x: x * (1.0 / (1.0 + np.exp(-x))),
    "mish":        lambda x: x * np.tanh(np.log1p(np.exp(x))),
    "hard_sigmoid": lambda x: np.clip(0.2 * x + 0.5, 0.0, 1.0),
    "hard_swish":  lambda x: x * np.clip(0.2 * x + 0.5, 0.0, 1.0),
}

DEFAULT_DTYPE = np.float32


# ---------------------------------------------------------------------------
# NodeGenerator
# ---------------------------------------------------------------------------

class NodeGenerator:
    """Random node generator for Broad Learning System.

    Generates random feature / enhancement nodes with optional weight
    whitening (QR) and output-group orthogonalization (Gram-Schmidt
    equivalent via QR + triangular solve).

    Parameters
    ----------
    activation : str
        Key into ``ACTIVATIONS``.  Default ``"relu"``.
    whiten : bool
        If True, QR-orthogonalize each random weight matrix.
    orthogonalize_output : bool
        If True, apply Gram-Schmidt orthogonalization to each output group
        so that ``group.T @ group == I``.
    """

    def __init__(self, activation="relu", whiten=False, orthogonalize_output=False):
        self.activation = activation
        self.whiten = whiten
        self.orthogonalize_output = orthogonalize_output

        self.Wlist = []
        self.blist = []
        self.output_transform_list = []
        self.spW = None
        self._cached_transform_params = None

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def orth(W):
        """QR-orthogonalize columns of W (reduced).

        Silently returns W unchanged when rows < cols (cannot orthogonalize).
        """
        if W.shape[0] < W.shape[1]:
            warnings.warn(
                "NodeGenerator.orth: rows < cols, skipping orthogonalization."
            )
            return W
        Q, _ = np.linalg.qr(np.asarray(W, dtype=np.float64), mode="reduced")
        return Q[:, : W.shape[1]].astype(W.dtype, copy=False)

    @staticmethod
    def fit_orthogonal_output_transform(H):
        """Apply Gram-Schmidt orthogonalization to output group H.

        Returns ``(H_orth, transform)`` so that ``H_orth.T @ H_orth == I``
        and ``H_orth == H @ transform``.
        """
        if H.shape[0] < H.shape[1]:
            raise ValueError(
                "Enhancement group cannot be orthogonalized when "
                "num_samples < group_width."
            )
        H64 = np.asarray(H, dtype=np.float64)
        _, R = np.linalg.qr(H64, mode="reduced")
        I_mat = np.eye(R.shape[0], dtype=np.float64)
        try:
            transform = solve_triangular(R, I_mat, lower=False)
        except Exception:
            transform = np.linalg.pinv(R)
        transformed = H64 @ transform
        return (
            transformed.astype(DEFAULT_DTYPE, copy=False),
            transform.astype(np.float64, copy=False),
        )

    # ---- core API ---------------------------------------------------------

    def generate_nodes(self, data, feature_size, times):
        """Generate ``times`` node groups, each of width ``feature_size``."""
        if data.size == 0:
            raise ValueError("Empty data passed to generate_nodes.")
        if feature_size <= 0:
            raise ValueError(
                "feature_size must be a positive integer, got %s" % feature_size
            )
        if int(times) <= 0:
            raise ValueError(
                "times must be a positive integer, got %s" % times
            )
        self.Wlist = []
        self.blist = []
        self.output_transform_list = []
        self._cached_transform_params = None

        input_dim = data.shape[1]
        dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else DEFAULT_DTYPE
        ts = int(times)

        # --- 6.1: Hoist orthogonalize_output branch out of loop ---
        if self.orthogonalize_output:
            outputs = []
            for _ in range(ts):
                W = np.random.uniform(-1.0, 1.0, size=(input_dim, feature_size)).astype(
                    dtype, copy=False
                )
                if self.whiten:
                    W = self.orth(W)
                b = np.random.uniform(-0.5, 0.5, size=(feature_size,)).astype(
                    dtype, copy=False
                )
                self.Wlist.append(W)
                self.blist.append(b)

                H = ACTIVATIONS[self.activation](data @ W + b)
                H, transform = self.fit_orthogonal_output_transform(H)
                self.output_transform_list.append(transform)
                outputs.append(H)
            return np.hstack(outputs).astype(DEFAULT_DTYPE, copy=False)
        else:
            for _ in range(ts):
                W = np.random.uniform(-1.0, 1.0, size=(input_dim, feature_size)).astype(
                    dtype, copy=False
                )
                if self.whiten:
                    W = self.orth(W)
                b = np.random.uniform(-0.5, 0.5, size=(feature_size,)).astype(
                    dtype, copy=False
                )
                self.Wlist.append(W)
                self.blist.append(b)
            return self.transform(data)

    def transform(self, X):
        """Apply the generated nodes to new data X."""
        # sparse shortcut
        if self.spW is not None and not self.orthogonalize_output:
            return ACTIVATIONS[self.activation](X @ self.spW)

        # no nodes generated yet
        if not self.Wlist:
            return np.zeros((X.shape[0], 0), dtype=DEFAULT_DTYPE)

        # ---- non-orthogonalized path (cached concatenation) ---------------
        if not self.orthogonalize_output:
            if self._cached_transform_params is None:
                W_big = np.hstack(self.Wlist)
                b_big = np.concatenate(self.blist, axis=0)
                self._cached_transform_params = (W_big, b_big)
            else:
                W_big, b_big = self._cached_transform_params
            return ACTIVATIONS[self.activation](X @ W_big + b_big).astype(
                DEFAULT_DTYPE, copy=False
            )

        # ---- orthogonalized path (per-group) ------------------------------
        outputs = []
        for idx, (W, b) in enumerate(zip(self.Wlist, self.blist)):
            H = ACTIVATIONS[self.activation](X @ W + b)
            if idx >= len(self.output_transform_list):
                raise RuntimeError(
                    "Orthogonal output transforms are missing. Refit the model."
                )
            transform = self.output_transform_list[idx]  # already float64
            H = np.asarray(H, dtype=np.float64) @ transform
            outputs.append(H.astype(DEFAULT_DTYPE, copy=False))
        return np.hstack(outputs).astype(DEFAULT_DTYPE, copy=False)

    def update(self, otherW, otherb, other_transforms=None):
        """Append new node groups (for incremental learning)."""
        self.Wlist += otherW
        self.blist += otherb
        if self.orthogonalize_output:
            if other_transforms is None or len(other_transforms) != len(otherW):
                raise ValueError(
                    "Orthogonalized node updates must provide matching "
                    "output transforms."
                )
            self.output_transform_list += other_transforms
        self._cached_transform_params = None
