"""Tests for statistical threshold computation (Phase 3, TDD)."""

from __future__ import annotations

import numpy as np

from anomaly_explainer.config import DETECTION
from anomaly_explainer.detection.threshold import (
    Threshold,
    fit_threshold,
    percentile_threshold,
    sigma_threshold,
)


def test_sigma_threshold_matches_formula():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = 2.0
    expected = errors.mean() + k * errors.std()
    assert np.isclose(sigma_threshold(errors, k), expected)


def test_percentile_threshold_matches_numpy():
    rng = np.random.default_rng(0)
    errors = rng.random(1000)
    assert np.isclose(percentile_threshold(errors, 99.0), np.percentile(errors, 99.0))


def test_fit_threshold_returns_threshold_object():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = fit_threshold(errors, method="sigma")
    assert isinstance(t, Threshold)
    assert t.method == "sigma"
    assert t.value > errors.mean()


def test_fit_threshold_percentile_method():
    rng = np.random.default_rng(1)
    errors = rng.random(500)
    t = fit_threshold(errors, method="percentile")
    assert t.method == "percentile"
    assert np.isclose(t.value, np.percentile(errors, DETECTION.threshold_percentile))


def test_fit_threshold_rejects_unknown_method():
    errors = np.array([1.0, 2.0, 3.0])
    try:
        fit_threshold(errors, method="bogus")
    except ValueError as e:
        assert "method" in str(e).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown method")


def test_fit_threshold_rejects_empty_errors():
    try:
        fit_threshold(np.array([]), method="sigma")
    except ValueError as e:
        assert "non-empty" in str(e).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty errors")


def test_threshold_above_most_normal_errors():
    # A sigma threshold should sit above the bulk of the normal distribution.
    rng = np.random.default_rng(2)
    errors = np.abs(rng.normal(0, 1, 10000))
    t = fit_threshold(errors, method="sigma")
    assert (errors < t.value).mean() > 0.9
