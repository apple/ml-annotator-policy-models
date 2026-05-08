#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""End-to-end pipeline tests for NNLR and DNF decision functions.

Generates synthetic binary feature data and verifies the full code path
(fit, predict, score, string representation, features used) without
any external API dependencies.
"""

import numpy as np
import pytest

from safe_dictionary_learning.decision_functions.nnlr import NonNegativeLogisticRegression

try:
    from safe_dictionary_learning.decision_functions.dnf import DNF

    HAS_AIX360 = True
except ImportError:
    HAS_AIX360 = False

requires_aix360 = pytest.mark.skipif(not HAS_AIX360, reason="aix360 not installed")

FEATURES = [
    "drug references",
    "weapon mentions",
    "violent language",
    "profanity",
    "personal attacks",
    "self-harm",
    "illegal activity",
    "explicit content",
    "harassment",
    "safe conversation",
]

N_SAMPLES = 100
N_FEATURES = len(FEATURES)


@pytest.fixture
def synthetic_data():
    """Generate synthetic binary feature matrix and labels.

    Unsafe texts (label=1) get 2-4 random safety features activated.
    Safe texts (label=0) get 0-1 features activated.
    """
    rng = np.random.RandomState(42)
    n_unsafe = N_SAMPLES // 2

    X = np.zeros((N_SAMPLES, N_FEATURES), dtype=np.float32)
    y = np.zeros(N_SAMPLES, dtype=np.float32)

    # Unsafe samples
    for i in range(n_unsafe):
        n_active = rng.randint(2, 5)
        active_idx = rng.choice(N_FEATURES - 1, size=n_active, replace=False)
        X[i, active_idx] = 1.0
        y[i] = 1.0

    # Safe samples
    for i in range(n_unsafe, N_SAMPLES):
        n_active = rng.randint(0, 2)
        if n_active > 0:
            active_idx = rng.choice([N_FEATURES - 1], size=1)
            X[i, active_idx] = 1.0
        y[i] = 0.0

    return X, y


@pytest.fixture
def train_test_split(synthetic_data):
    """Split synthetic data into train/test (80/20)."""
    X, y = synthetic_data
    split = int(0.8 * N_SAMPLES)
    return X[:split], X[split:], y[:split], y[split:]


class TestNNLR:
    def test_fit_and_predict(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
        clf.fit(X_train, y_train, max_iter=500)

        preds = clf.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_accuracy_above_chance(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
        clf.fit(X_train, y_train, max_iter=500)

        acc = clf.score(X_test, y_test)
        assert float(acc) > 0.5, f"Accuracy {acc} not above chance"

    def test_predict_proba(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
        clf.fit(X_train, y_train, max_iter=500)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_string_representation(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
        clf.fit(X_train, y_train, max_iter=500)

        rep = clf.get_string_representation()
        assert isinstance(rep, str)
        assert len(rep) > 0
        assert "feature: weight" in rep

    def test_get_features_used(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
        clf.fit(X_train, y_train, max_iter=500)

        used = clf.get_features_used()
        assert isinstance(used, list)
        for idx in used:
            assert 0 <= idx < N_FEATURES

    def test_weights_non_negative(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
        clf.fit(X_train, y_train, max_iter=500)

        weights = clf.weight.detach().numpy()
        assert (weights >= 0).all(), "Weights should be non-negative"

    def test_coef_attribute(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES)
        clf.fit(X_train, y_train, max_iter=500)

        assert hasattr(clf, "coef_")
        assert clf.coef_.shape == (1, N_FEATURES)


@requires_aix360
class TestDNF:
    def test_fit_and_predict(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        clf = DNF(FEATURES, lambda0=0.001, lambda1=0.001, verbose=False)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        assert len(preds) == X_test.shape[0]

    def test_accuracy_above_chance(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        clf = DNF(FEATURES, lambda0=0.001, lambda1=0.001, verbose=False)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        assert float(acc) > 0.5, f"Accuracy {acc} not above chance"

    def test_string_representation(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = DNF(FEATURES, lambda0=0.001, lambda1=0.001, verbose=False)
        clf.fit(X_train, y_train)

        rep = clf.get_string_representation()
        assert isinstance(rep, str)
        assert len(rep) > 0

    def test_get_features_used(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = DNF(FEATURES, lambda0=0.001, lambda1=0.001, verbose=False)
        clf.fit(X_train, y_train)

        used = clf.get_features_used()
        assert isinstance(used, list)
        for idx in used:
            assert 0 <= idx < N_FEATURES

    def test_produces_rules(self, train_test_split):
        X_train, _, y_train, _ = train_test_split
        clf = DNF(FEATURES, lambda0=0.001, lambda1=0.001, verbose=False)
        clf.fit(X_train, y_train)

        explanation = clf.clf.explain()
        assert "rules" in explanation
        assert len(explanation["rules"]) > 0


class TestMultiAnnotator:
    """Test fitting separate models per annotator (the core APM use case)."""

    def test_per_annotator_nnlr(self, synthetic_data):
        X, y_gt = synthetic_data
        rng = np.random.RandomState(123)

        for annotator_id in range(3):
            noise = rng.binomial(1, 0.15, size=len(y_gt))
            y_ann = np.abs(y_gt - noise).astype(np.float32)

            clf = NonNegativeLogisticRegression(N_FEATURES, features=FEATURES, lam=0.01)
            clf.fit(X, y_ann, max_iter=500)

            acc = clf.score(X, y_ann)
            assert float(acc) > 0.5, f"Annotator {annotator_id} accuracy {acc} not above chance"

            rep = clf.get_string_representation()
            assert len(rep) > 0

    @requires_aix360
    def test_per_annotator_dnf(self, synthetic_data):
        X, y_gt = synthetic_data
        rng = np.random.RandomState(456)

        for annotator_id in range(3):
            noise = rng.binomial(1, 0.15, size=len(y_gt))
            y_ann = np.abs(y_gt - noise).astype(np.float32)

            clf = DNF(FEATURES, lambda0=0.001, lambda1=0.001, verbose=False)
            clf.fit(X, y_ann)

            acc = clf.score(X, y_ann)
            assert float(acc) > 0.5, f"Annotator {annotator_id} accuracy {acc} not above chance"

            rep = clf.get_string_representation()
            assert len(rep) > 0
