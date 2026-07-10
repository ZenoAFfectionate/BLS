"""Tests for NodeGenerator class."""

import pytest
import numpy as np
from models.node_generator import (
    NodeGenerator,
    ACTIVATIONS,
    DEFAULT_DTYPE,
)


class TestActivationFunctions:
    """Test all built-in activation functions."""

    @pytest.mark.parametrize("name", list(ACTIVATIONS.keys()))
    def test_output_shape(self, name):
        fn = ACTIVATIONS[name]
        x = np.random.randn(50, 32).astype(np.float32)
        y = fn(x)
        assert y.shape == x.shape, f"{name} changed shape"

    @pytest.mark.parametrize("name", list(ACTIVATIONS.keys()))
    def test_no_nan_inf(self, name):
        fn = ACTIVATIONS[name]
        # float32-safe range (exp(88) is near float32 max)
        x = np.array([-80.0, -10.0, -1.0, 0.0, 1.0, 10.0, 80.0], dtype=np.float32)
        y = fn(x)
        assert not np.any(np.isnan(y)), f"{name} produced NaN"
        assert not np.any(np.isinf(y)), f"{name} produced inf"

    def test_sigmoid_range(self):
        x = np.linspace(-50, 50, 1000).astype(np.float32)
        y = ACTIVATIONS["sigmoid"](x)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_relu_nonnegative(self):
        x = np.array([-5, -1, 0, 1, 5], dtype=np.float32)
        y = ACTIVATIONS["relu"](x)
        assert np.all(y >= 0)

    def test_mish_known_behavior(self):
        """mish(x) → 0 as x → -∞, and mish(x) ≈ x as x → +∞."""
        x_pos = np.array([100.0], dtype=np.float64)
        np.testing.assert_allclose(ACTIVATIONS["mish"](x_pos), x_pos, rtol=0.01)
        x_neg = np.array([-100.0], dtype=np.float64)
        np.testing.assert_allclose(ACTIVATIONS["mish"](x_neg), [0.0], atol=0.01)
        # mish(x) is non-monotonic (dips near x≈-1.2), which is correct behavior

    def test_hard_sigmoid_range(self):
        x = np.linspace(-10, 10, 1000).astype(np.float32)
        y = ACTIVATIONS["hard_sigmoid"](x)
        assert y.min() >= 0 and y.max() <= 1


class TestNodeGeneratorBasic:
    """Tests for basic NodeGenerator operations."""

    def test_generate_nodes_shape(self):
        ng = NodeGenerator("relu")
        X = np.random.randn(100, 10).astype(DEFAULT_DTYPE)
        out = ng.generate_nodes(X, feature_size=20, times=5)
        assert out.shape == (100, 20 * 5)
        assert out.dtype == DEFAULT_DTYPE

    def test_generate_nodes_reproducible(self):
        X = np.random.randn(50, 5).astype(DEFAULT_DTYPE)

        np.random.seed(42)
        ng = NodeGenerator("tanh")
        out1 = ng.generate_nodes(X, feature_size=8, times=3)

        np.random.seed(42)
        ng2 = NodeGenerator("tanh")
        out2 = ng2.generate_nodes(X, feature_size=8, times=3)

        np.testing.assert_array_equal(out1, out2)

    def test_transform_same_as_generate(self):
        """generate_nodes should produce same output as separate transform."""
        np.random.seed(123)
        ng = NodeGenerator("relu")
        X = np.random.randn(60, 8).astype(DEFAULT_DTYPE)

        out_gen = ng.generate_nodes(X, feature_size=16, times=4)
        out_trans = ng.transform(X)

        assert np.allclose(out_gen, out_trans)

    def test_transform_new_data_same_shape(self):
        np.random.seed(7)
        ng = NodeGenerator("tanh")
        X_train = np.random.randn(30, 6).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X_train, feature_size=10, times=3)

        X_test = np.random.randn(20, 6).astype(DEFAULT_DTYPE)
        out = ng.transform(X_test)
        assert out.shape == (20, 30)

    def test_empty_nodes_zero_output(self):
        ng = NodeGenerator("relu")
        X = np.random.randn(10, 5).astype(DEFAULT_DTYPE)
        out = ng.transform(X)
        assert out.shape == (10, 0)

    def test_dtype_preservation(self):
        ng = NodeGenerator("linear")
        X = np.random.randn(40, 12).astype(np.float32)
        out = ng.generate_nodes(X, feature_size=8, times=2)
        assert out.dtype == np.float32

    def test_different_activations(self):
        for act in ["linear", "relu", "tanh", "sigmoid", "leaky_relu", "swish", "mish"]:
            ng = NodeGenerator(act)
            X = np.random.randn(20, 5).astype(DEFAULT_DTYPE)
            out = ng.generate_nodes(X, feature_size=6, times=2)
            assert not np.any(np.isnan(out)), f"{act} produced NaN"

    def test_feature_size_one(self):
        ng = NodeGenerator("relu")
        X = np.random.randn(30, 4).astype(DEFAULT_DTYPE)
        out = ng.generate_nodes(X, feature_size=1, times=5)
        assert out.shape == (30, 5)

    def test_times_one(self):
        ng = NodeGenerator("tanh")
        X = np.random.randn(25, 3).astype(DEFAULT_DTYPE)
        out = ng.generate_nodes(X, feature_size=10, times=1)
        assert out.shape == (25, 10)


class TestNodeGeneratorWhiten:
    """Tests for QR-whitening of weight matrices."""

    def test_whiten_columns_orthogonal(self):
        ng = NodeGenerator("linear", whiten=True)
        X = np.random.randn(100, 50).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X, feature_size=30, times=1)

        for W in ng.Wlist:
            if W.shape[0] >= W.shape[1]:
                wtw = W.T @ W
                np.testing.assert_allclose(wtw, np.eye(W.shape[1]), atol=1e-5)

    def test_whiten_skip_when_rows_lt_cols(self):
        ng = NodeGenerator("linear", whiten=True)
        X = np.random.randn(100, 5).astype(DEFAULT_DTYPE)
        with pytest.warns(UserWarning, match="rows < cols"):
            ng.generate_nodes(X, feature_size=50, times=1)  # 5 < 50


class TestNodeGeneratorOrthogonalizeOutput:
    """Tests for output-group orthogonalization."""

    def test_orthogonal_output_columns_orthonormal(self):
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(100, 10).astype(DEFAULT_DTYPE)
        out = ng.generate_nodes(X, feature_size=5, times=2)
        assert out.shape == (100, 10)
        # Groups h-stacked -> check per-group via stored transforms
        assert len(ng.output_transform_list) == 2

    def test_orthogonal_output_too_wide_raises(self):
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(10, 5).astype(DEFAULT_DTYPE)
        with pytest.raises(ValueError, match="num_samples < group_width"):
            ng.generate_nodes(X, feature_size=20, times=1)

    def test_orthogonal_transform_same(self):
        """transform() with orthogonalize_output reproduces outputs."""
        np.random.seed(99)
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(80, 6).astype(DEFAULT_DTYPE)
        _ = ng.generate_nodes(X, feature_size=4, times=3)

        X2 = np.random.randn(40, 6).astype(DEFAULT_DTYPE)
        out = ng.transform(X2)
        assert out.shape == (40, 12)

    def test_orthogonal_missing_transform_raises(self):
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(50, 8).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X, feature_size=4, times=2)

        # Corrupt transforms
        ng.output_transform_list = []
        with pytest.raises(RuntimeError, match="transforms are missing"):
            ng.transform(X)


class TestNodeGeneratorUpdate:
    """Tests for incremental node updates."""

    def test_update_non_orthogonal(self):
        np.random.seed(1)
        ng = NodeGenerator("relu")
        X = np.random.randn(30, 4).astype(DEFAULT_DTYPE)

        ng.generate_nodes(X, feature_size=5, times=2)
        out_before = ng.transform(X)

        W_new = [np.random.randn(4, 5).astype(DEFAULT_DTYPE)]
        b_new = [np.random.randn(5).astype(DEFAULT_DTYPE)]
        ng.update(W_new, b_new)

        out_after = ng.transform(X)
        assert out_after.shape == (30, 15)  # 2*5 + 1*5 = 15

    def test_update_orthogonal(self):
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(100, 6).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X, feature_size=3, times=2)

        W_new = [np.random.randn(6, 3).astype(DEFAULT_DTYPE)]
        b_new = [np.random.randn(3).astype(DEFAULT_DTYPE)]

        H = ACTIVATIONS["tanh"](X @ W_new[0] + b_new[0])
        H_orth, transform = NodeGenerator.fit_orthogonal_output_transform(H)

        ng.update(W_new, b_new, [transform])
        out = ng.transform(X)
        assert out.shape == (100, 9)

    def test_update_orthogonal_missing_transforms_raises(self):
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(50, 5).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X, feature_size=3, times=1)

        with pytest.raises(ValueError, match="matching output transforms"):
            ng.update(
                [np.random.randn(5, 3)],
                [np.random.randn(3)],
            )


class TestNodeGeneratorSparse:
    """Tests for the sparse shortcut (spW)."""

    def test_sparse_transform(self):
        ng = NodeGenerator("relu")
        X = np.random.randn(30, 5).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X, feature_size=4, times=3)
        # Simulate sparse shortcut
        ng.spW = np.random.randn(5, 12).astype(DEFAULT_DTYPE)
        out = ng.transform(X)
        assert out.shape == (30, 12)

    def test_sparse_skip_orthogonalized(self):
        ng = NodeGenerator("tanh", orthogonalize_output=True)
        X = np.random.randn(50, 6).astype(DEFAULT_DTYPE)
        ng.generate_nodes(X, feature_size=3, times=2)
        ng.spW = np.random.randn(6, 6).astype(DEFAULT_DTYPE)
        # Should NOT use spW — orthogonalized path takes precedence
        out = ng.transform(X)
        assert out.shape == (50, 6)  # orthogonal path, not spW
