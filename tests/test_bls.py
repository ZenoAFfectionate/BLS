"""Tests for the BLS model (models/bls.py)."""

import pytest
import numpy as np
from models.bls import BLS


@pytest.fixture
def simple_data():
    """2-class linearly separable data in 2D."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    return X, y


@pytest.fixture
def multi_class_data():
    """3-class data in 4D."""
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 4).astype(np.float32)
    y = np.random.randint(0, 3, size=n).astype(np.int64)
    return X, y


@pytest.fixture
def tiny_data():
    """Very small dataset for edge-case testing."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    return X, y


# -----------------------------------------------------------------------
#  Basic fit / predict
# -----------------------------------------------------------------------

class TestBLSFitPredict:
    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=5)
        out = model.fit(X, y)
        assert out is model

    def test_is_fitted_flag(self, simple_data):
        X, y = simple_data
        model = BLS(n_classes=2)
        assert not model.is_fitted
        model.fit(X, y)
        assert model.is_fitted

    def test_predict_shape(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert preds.dtype == np.int64

    def test_predict_proba_shape_and_range(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_high_accuracy_on_linear_separable(self, simple_data):
        X, y = simple_data
        model = BLS(
            feature_times=5,
            enhance_times=5,
            n_classes=2,
            feature_size=16,
            reg=0.001,
        )
        model.fit(X, y)
        acc = np.mean(model.predict(X) == y)
        assert acc > 0.85, f"Expected > 85% on separable data, got {acc:.1%}"


# -----------------------------------------------------------------------
#  Multi-class
# -----------------------------------------------------------------------

class TestBLSMultiClass:
    def test_multi_class_fit_predict(self, multi_class_data):
        X, y = multi_class_data
        model = BLS(
            feature_times=3, enhance_times=3, n_classes=3, feature_size=16, reg=0.01
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})
        assert preds.shape[0] == X.shape[0]

    def test_multi_class_proba_shape(self, multi_class_data):
        X, y = multi_class_data
        model = BLS(feature_times=3, enhance_times=3, n_classes=3, feature_size=10)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)


# -----------------------------------------------------------------------
#  Feature size = "auto"
# -----------------------------------------------------------------------

class TestBLSAutoFeatureSize:
    def test_auto_feature_size(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=1, n_classes=2, feature_size="auto")
        model.fit(X, y)
        assert model.feature_size == X.shape[1]

    def test_explicit_feature_size(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=1, n_classes=2, feature_size=20)
        model.fit(X, y)
        assert model.feature_size == 20


# -----------------------------------------------------------------------
#  Reset & re-fit
# -----------------------------------------------------------------------

class TestBLSReset:
    def test_reset_clears_state(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)
        assert model.is_fitted
        model.reset()
        assert not model.is_fitted
        assert model.W is None
        assert model._mapping_nodes is None

    def test_refit_after_reset(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)
        acc1 = np.mean(model.predict(X) == y)
        model.fit(X, y)  # triggers reset internally
        acc2 = np.mean(model.predict(X) == y)
        assert acc1 >= 0 and acc2 >= 0  # both should work


# -----------------------------------------------------------------------
#  Incremental learning
# -----------------------------------------------------------------------

class TestBLSIncremental:
    def test_add_enhancement_nodes(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)

        initial_groups = len(model.enhance_generator.Wlist)
        model.add_enhancement_nodes(X, y, num_nodes=5)
        updated_groups = len(model.enhance_generator.Wlist)
        assert updated_groups == initial_groups + 1  # one new group

    def test_add_zero_nodes_noop(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)
        n_before = len(model.enhance_generator.Wlist)
        model.add_enhancement_nodes(X, y, num_nodes=0)
        n_after = len(model.enhance_generator.Wlist)
        assert n_before == n_after

    def test_add_nodes_must_be_fitted(self, simple_data):
        X, y = simple_data
        model = BLS(n_classes=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.add_enhancement_nodes(X, y, num_nodes=3)

    def test_add_too_many_nodes_raises(self, tiny_data):
        X, y = tiny_data
        model = BLS(feature_times=1, enhance_times=1, n_classes=2, feature_size=2)
        model.fit(X, y)
        with pytest.raises(ValueError, match="num_nodes must not exceed"):
            model.add_enhancement_nodes(X, y, num_nodes=10)


# -----------------------------------------------------------------------
#  Error handling
# -----------------------------------------------------------------------

class TestBLSErrors:
    def test_predict_before_fit(self):
        model = BLS(n_classes=2)
        X = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_predict_proba_before_fit(self):
        model = BLS(n_classes=2)
        X = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)

    def test_onehot_wrong_shape(self, simple_data):
        X, _ = simple_data
        model = BLS(n_classes=2)
        # 3-class one-hot for a 2-class model → argmax gives label 2 → IndexError
        bad_y = np.eye(3, dtype=np.float32)[
            np.random.randint(0, 3, size=X.shape[0])
        ]
        with pytest.raises((ValueError, IndexError)):
            model.fit(X, bad_y)


# -----------------------------------------------------------------------
#  Regularization & solver
# -----------------------------------------------------------------------

class TestBLSSolver:
    def test_regularization_effect(self, simple_data):
        X, y = simple_data
        # High reg → large W norm penalty
        np.random.seed(42)
        m_high = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=10, reg=1e3)
        m_high.fit(X, y)

        np.random.seed(42)
        m_low = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=10, reg=1e-6)
        m_low.fit(X, y)

        assert np.linalg.norm(m_high.W) < np.linalg.norm(m_low.W)

    def test_solution_is_deterministic(self, simple_data):
        X, y = simple_data
        np.random.seed(123)
        m1 = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        m1.fit(X, y)
        w1 = m1.W.copy()

        np.random.seed(123)
        m2 = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        m2.fit(X, y)

        np.testing.assert_allclose(w1, m2.W)


# -----------------------------------------------------------------------
#  Sparse mode
# -----------------------------------------------------------------------

class TestBLSSparse:
    def test_sparse_mode(self, simple_data):
        X, y = simple_data
        model = BLS(
            feature_times=2, enhance_times=2, n_classes=2, feature_size=8,
            use_sparse=True,
        )
        model.fit(X, y)
        assert model.mapping_generator.spW is not None
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)


# -----------------------------------------------------------------------
#  Edge cases
# -----------------------------------------------------------------------

class TestBLSEdgeCases:
    def test_data_unchanged_by_predict(self, simple_data):
        X, y = simple_data
        model = BLS(feature_times=2, enhance_times=2, n_classes=2, feature_size=8)
        model.fit(X, y)
        X_copy = X.copy()
        _ = model.predict(X)
        np.testing.assert_array_equal(X, X_copy)

    def test_single_sample(self):
        X = np.random.randn(1, 5).astype(np.float32)
        y = np.array([1], dtype=np.int64)
        # feature_size must be <= n_samples for orthogonalized enhancement
        model = BLS(feature_times=2, enhance_times=1, n_classes=2, feature_size=1)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (1, 2)
