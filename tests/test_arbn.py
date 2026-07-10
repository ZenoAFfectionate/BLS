"""Tests for the ARBN model (models/arbn.py)."""

import pytest
import numpy as np
from models.arbn import ARBN


# -----------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def balanced_data():
    """Balanced 2-class data."""
    np.random.seed(42)
    X = np.random.randn(200, 4).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    np.random.shuffle(y)
    return X, y


@pytest.fixture
def imbalanced_data():
    """Imbalanced 3-class data (100, 30, 10)."""
    np.random.seed(42)
    n = 140
    X = np.random.randn(n, 4).astype(np.float32)
    y = np.array(
        [0] * 100 + [1] * 30 + [2] * 10, dtype=np.int64
    )
    return X, y


@pytest.fixture
def cls_num_list():
    return [100, 30, 10]


# -----------------------------------------------------------------------
#  Basic fit / predict
# -----------------------------------------------------------------------

class TestARBNFitPredict:
    def test_fit_balanced(self, balanced_data):
        X, y = balanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=2, feature_size=8,
            cls_num_list=None,
        )
        model.fit(X, y)
        assert model.is_fitted

    def test_fit_imbalanced(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=cls_num_list, class_weight_beta=0.5,
        )
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_proba_shape(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=10,
            cls_num_list=cls_num_list,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


# -----------------------------------------------------------------------
#  Class weights
# -----------------------------------------------------------------------

class TestARBNWeights:
    def test_beta_zero_uniform_weights(self, cls_num_list):
        model = ARBN(
            n_classes=3, cls_num_list=cls_num_list, class_weight_beta=0.0,
        )
        np.testing.assert_allclose(model.class_weights, np.ones(3))

    def test_beta_one_inverse_frequency(self, cls_num_list):
        model = ARBN(
            n_classes=3, cls_num_list=cls_num_list, class_weight_beta=1.0,
        )
        # w_k = 1/|D_k|
        expected = np.array([1 / 100, 1 / 30, 1 / 10], dtype=np.float32)
        np.testing.assert_allclose(model.class_weights, expected)

    def test_cls_num_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match n_classes"):
            ARBN(n_classes=3, cls_num_list=[100, 30])

    def test_zero_count_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ARBN(n_classes=2, cls_num_list=[100, 0])

    def test_no_cls_num_list_uniform(self):
        model = ARBN(n_classes=5)
        np.testing.assert_allclose(model.class_weights, np.ones(5))


# -----------------------------------------------------------------------
#  Adaptive re-weighting
# -----------------------------------------------------------------------

class TestARBNAdaptive:
    def test_adaptive_disabled_falls_back(self, imbalanced_data):
        """With adaptive_reg=False, should behave like unweighted."""
        X, y = imbalanced_data
        np.random.seed(42)
        m1 = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=[100, 30, 10], adaptive_reg=False,
        )
        m1.fit(X, y)

        np.random.seed(42)
        m2 = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=None, adaptive_reg=True,
        )
        m2.fit(X, y)

        np.testing.assert_allclose(m1.W, m2.W)

    def test_adaptive_enabled_different_weights(self, imbalanced_data):
        """With adaptive weights, different beta → different W."""
        X, y = imbalanced_data
        np.random.seed(42)
        m_low = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=[100, 30, 10], class_weight_beta=0.2,
        )
        m_low.fit(X, y)

        np.random.seed(42)
        m_high = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=[100, 30, 10], class_weight_beta=0.8,
        )
        m_high.fit(X, y)

        assert not np.allclose(m_low.W, m_high.W)


# -----------------------------------------------------------------------
#  Incremental learning
# -----------------------------------------------------------------------

class TestARBNIncremental:
    def test_add_enhancement_nodes(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=cls_num_list,
        )
        model.fit(X, y)

        initial_groups = len(model.enhance_generator.Wlist)
        model.add_enhancement_nodes(X, y, num_nodes=4)
        assert len(model.enhance_generator.Wlist) == initial_groups + 1

    def test_add_nodes_maintains_proba_shape(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=cls_num_list,
        )
        model.fit(X, y)
        model.add_enhancement_nodes(X, y, num_nodes=3)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)


# -----------------------------------------------------------------------
#  Evaluate imbalanced
# -----------------------------------------------------------------------

class TestARBNEvaluate:
    def test_evaluate_imbalanced_returns_all_keys(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=cls_num_list,
        )
        model.fit(X, y)
        metrics = model.evaluate_imbalanced(X, y)
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in metrics

    def test_evaluate_imbalanced_binary(self):
        """Test evaluate_imbalanced with 2 classes."""
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=2, feature_size=8,
            cls_num_list=[50, 50],
        )
        model.fit(X, y)
        metrics = model.evaluate_imbalanced(X, y)
        assert "auc" in metrics


# -----------------------------------------------------------------------
#  Reset
# -----------------------------------------------------------------------

class TestARBNReset:
    def test_reset(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=cls_num_list,
        )
        model.fit(X, y)
        model.reset()
        assert not model.is_fitted
        assert model.W is None


# -----------------------------------------------------------------------
#  Edge cases
# -----------------------------------------------------------------------

class TestARBNEdgeCases:
    def test_class_weight_beta_none_defaults(self):
        """class_weight_beta=None should default to 0.5"""
        model = ARBN(
            n_classes=3, cls_num_list=[10, 10, 10], class_weight_beta=None,
        )
        assert model.class_weight_beta == 0.5

    def test_auto_feature_size(self, imbalanced_data, cls_num_list):
        X, y = imbalanced_data
        model = ARBN(
            feature_times=2, enhance_times=1, n_classes=3,
            feature_size="auto", cls_num_list=cls_num_list,
        )
        model.fit(X, y)
        assert model.feature_size == X.shape[1]
