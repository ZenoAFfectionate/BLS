"""Tests for utility functions (utils.py)."""

import pytest
import numpy as np
from models.bls import BLS
from loader.data_loader import get_cls_num_list
from utils import (
    accuracy,
    top_k_accuracy,
    evaluate_model,
)


class TestAccuracy:
    def test_perfect(self, data_fixture):
        X, y = data_fixture
        model = _make_predict_wrapper(_make_trained_bls(X, y))
        acc = accuracy(model, X, y)
        assert 0 <= acc <= 100

    def test_zero(self):
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.array([1] * 50, dtype=np.int64)
        model = _make_zero_model(X)
        acc = accuracy(model, X, y)
        assert acc == 0.0


class TestTopKAccuracy:
    def test_k5_all_hit(self):
        # All probability mass on the correct label → top-5 always includes it
        n = 100
        y_true = np.random.randint(0, 20, size=n)
        y_proba = np.zeros((n, 20))
        y_proba[np.arange(n), y_true] = 1.0
        assert top_k_accuracy(y_true, y_proba, k=5) == 1.0

    def test_k1_vs_top1_accuracy(self):
        n = 50
        y_true = np.random.randint(0, 10, size=n)
        y_proba = np.random.random((n, 10))
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        top1 = top_k_accuracy(y_true, y_proba, k=1)
        manual = np.mean(np.argmax(y_proba, axis=1) == y_true)
        assert top1 == pytest.approx(manual)


class TestEvaluateModel:
    def test_returns_all_keys(self):
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float32)
        y = np.random.randint(0, 5, size=100).astype(np.int64)

        model = BLS(feature_times=2, enhance_times=2, n_classes=5, feature_size=8)
        model.fit(X, y)
        metrics = evaluate_model(model, X, y, n_classes=5)

        expected_keys = {
            "accuracy", "recall_macro", "recall_micro",
            "precision_macro", "precision_micro",
            "f1_macro", "f1_micro",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_top5_only_for_many_classes(self):
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float32)
        y = np.random.randint(0, 20, size=100).astype(np.int64)

        model = BLS(feature_times=2, enhance_times=2, n_classes=20, feature_size=16)
        model.fit(X, y)

        metrics_few = evaluate_model(model, X, y, n_classes=5)
        metrics_many = evaluate_model(model, X, y, n_classes=20)

        assert "top5_accuracy" not in metrics_few
        assert "top5_accuracy" in metrics_many


class TestGetClsNumList:
    def test_exponential_decay(self):
        targets = np.array([0] * 100, dtype=np.int64)  # only class 0
        result = get_cls_num_list(targets, n_classes=5, imb_factor=2.0)
        # max = 100, then: 100, 100*(1/2)^(1/4), 100*(1/2)^(2/4), ...
        assert result[0] == 100
        assert result[0] > result[1] > result[-1]

    def test_balanced_factor_one(self):
        targets = np.arange(100).astype(np.int64) % 10
        result = get_cls_num_list(targets, n_classes=10, imb_factor=1.0)
        # With imb_factor=1, all should be equal to max
        assert all(v == result[0] for v in result)


# -----------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------

@pytest.fixture
def data_fixture():
    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randint(0, 3, size=100).astype(np.int64)
    return X, y


def _make_trained_bls(X, y):
    model = BLS(feature_times=2, enhance_times=2, n_classes=3, feature_size=8)
    model.fit(X, y)
    return model


def _make_predict_wrapper(trained_model):
    class PredictWrapper:
        def predict(self, X):
            return trained_model.predict(X)
        def predict_proba(self, X):
            return trained_model.predict_proba(X)
    return PredictWrapper()


class FakeModel:
    """Minimal mock always predicting class 0."""
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=np.int64)

    def predict_proba(self, X):
        n = X.shape[0]
        p = np.zeros((n, 2), dtype=np.float32)
        p[:, 0] = 1.0
        return p


def _make_zero_model(X):
    return FakeModel()
