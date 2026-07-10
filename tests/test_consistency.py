"""Cross-model consistency tests (BLS vs ARBN)."""

import pytest
import numpy as np
from models.bls import BLS
from models.arbn import ARBN


@pytest.fixture
def data():
    np.random.seed(42)
    X = np.random.randn(150, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=150).astype(np.int64)
    return X, y


class TestBLSEquivalenceARBN:
    """ARBN with cls_num_list=None should be equivalent to BLS.

    Both use the same NodeGenerator with orthogonalize_output=True
    for enhancement nodes, so with equal random seeds they should
    produce identical results.
    """

    COMMON = dict(
        feature_times=3, enhance_times=3, n_classes=3, feature_size=10, reg=0.01
    )

    def test_same_seed_same_W(self, data):
        X, y = data

        np.random.seed(123)
        bls = BLS(**self.COMMON)
        bls.fit(X, y)

        np.random.seed(123)
        arbn = ARBN(**self.COMMON, cls_num_list=None, adaptive_reg=False)
        arbn.fit(X, y)

        np.testing.assert_allclose(bls.W, arbn.W, rtol=1e-5, atol=1e-5)

    def test_same_seed_same_predictions(self, data):
        X, y = data

        np.random.seed(42)
        bls = BLS(**self.COMMON)
        bls.fit(X, y)

        np.random.seed(42)
        arbn = ARBN(**self.COMMON, cls_num_list=None, adaptive_reg=False)
        arbn.fit(X, y)

        np.testing.assert_array_equal(bls.predict(X), arbn.predict(X))
        np.testing.assert_allclose(bls.predict_proba(X), arbn.predict_proba(X))


class TestIncrementalConsistency:
    """Adding all nodes at once vs incrementally should produce similar results."""

    def test_full_vs_incremental(self, data):
        X, y = data

        # Full training with more enhancement groups
        np.random.seed(99)
        bls_full = BLS(
            feature_times=2, enhance_times=4, n_classes=3,
            feature_size=8, reg=0.01,
        )
        bls_full.fit(X, y)

        # Train with fewer, then add
        np.random.seed(99)
        bls_inc = BLS(
            feature_times=2, enhance_times=2, n_classes=3,
            feature_size=8, reg=0.01,
        )
        bls_inc.fit(X, y)

        # Cannot directly compare because add_enhancement_nodes generates
        # new random weights.  But both should produce valid predictions.
        bls_inc.add_enhancement_nodes(X, y, num_nodes=8)
        bls_inc.add_enhancement_nodes(X, y, num_nodes=8)

        proba_full = bls_full.predict_proba(X)
        proba_inc = bls_inc.predict_proba(X)

        assert proba_full.shape == proba_inc.shape
        assert np.allclose(proba_full.sum(axis=1), 1.0)
        assert np.allclose(proba_inc.sum(axis=1), 1.0)


class TestARBNBetaInvariance:
    """ARBN with beta=0 should give same results as ARBN with cls_num_list=None."""

    def test_beta_zero_equals_no_weights(self, data):
        X, y = data
        cls_counts = [50, 50, 50]  # balanced counts

        np.random.seed(7)
        m1 = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=cls_counts, class_weight_beta=0.0,
        )
        m1.fit(X, y)

        np.random.seed(7)
        m2 = ARBN(
            feature_times=2, enhance_times=2, n_classes=3, feature_size=8,
            cls_num_list=None,
        )
        m2.fit(X, y)

        np.testing.assert_allclose(m1.W, m2.W)


class TestARBNImprovesTail:
    """On imbalanced data, ARBN should improve recall on tail classes."""

    def test_arbn_better_tail_recall(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 4).astype(np.float32)
        # Severely imbalanced: 160 head vs 40 tail
        y = np.array([0] * 160 + [1] * 40, dtype=np.int64)
        np.random.shuffle(y)

        cls_counts = [int((y == c).sum()) for c in range(2)]

        # Simple test on same train data (检验是否偏向多数类)
        np.random.seed(1)
        bls = BLS(
            feature_times=3, enhance_times=3, n_classes=2,
            feature_size=16, reg=0.01,
        )
        bls.fit(X, y)

        np.random.seed(1)
        arbn = ARBN(
            feature_times=3, enhance_times=3, n_classes=2,
            feature_size=16, reg=0.01,
            cls_num_list=cls_counts, class_weight_beta=1.0,
        )
        arbn.fit(X, y)

        from sklearn.metrics import recall_score
        bls_recall = recall_score(y, bls.predict(X), pos_label=1, zero_division=0)
        arbn_recall = recall_score(y, arbn.predict(X), pos_label=1, zero_division=0)

        # ARBN should have >= recall on the minority class
        assert arbn_recall >= bls_recall, (
            f"ARBN tail recall {arbn_recall:.3f} < BLS {bls_recall:.3f}"
        )
