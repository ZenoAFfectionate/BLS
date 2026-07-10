"""Tests for model serialization (pickle save/load)."""

import os
import tempfile
import pytest
import numpy as np
from models.bls import BLS
from models.arbn import ARBN
from loader.model_loader import load_model, store_model


@pytest.fixture
def data():
    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randint(0, 3, size=100).astype(np.int64)
    return X, y


class TestBLSSerialization:
    def test_save_load_identical_predictions(self, data):
        X, y = data
        np.random.seed(7)
        model = BLS(feature_times=3, enhance_times=3, n_classes=3, feature_size=10)
        model.fit(X, y)

        proba_orig = model.predict_proba(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            store_model(model, path)
            loaded = load_model(BLS, path)
            proba_loaded = loaded.predict_proba(X)
            np.testing.assert_allclose(proba_orig, proba_loaded)
        finally:
            os.unlink(path)

    def test_save_load_is_fitted(self, data):
        X, y = data
        np.random.seed(3)
        model = BLS(feature_times=2, enhance_times=2, n_classes=3, feature_size=8)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            store_model(model, path)
            loaded = load_model(BLS, path)
            assert loaded.is_fitted
            _ = loaded.predict(X)
        finally:
            os.unlink(path)


class TestARBNSerialization:
    def test_save_load_identical_predictions(self, data):
        X, y = data
        np.random.seed(11)
        model = ARBN(
            feature_times=3, enhance_times=3, n_classes=3, feature_size=10,
            cls_num_list=[40, 30, 30], class_weight_beta=0.5,
        )
        model.fit(X, y)

        proba_orig = model.predict_proba(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            store_model(model, path)
            loaded = load_model(ARBN, path)
            proba_loaded = loaded.predict_proba(X)
            np.testing.assert_allclose(proba_orig, proba_loaded)
        finally:
            os.unlink(path)
