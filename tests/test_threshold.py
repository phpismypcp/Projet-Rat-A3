"""Tests for statistical threshold computation (Phase 3) and persistence (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_explainer.config import DETECTION
from anomaly_explainer.detection.threshold import (
    Threshold,
    fit_threshold,
    load_threshold,
    percentile_threshold,
    save_threshold,
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


def test_fit_threshold_defaults_to_configured_method():
    """The method lives in config, not in a function default."""
    errors = np.abs(np.random.default_rng(4).normal(0, 1, 500))
    assert fit_threshold(errors).method == DETECTION.threshold_method


def test_fit_threshold_records_its_parameter():
    """The tuned value alone is not reproducible — record k / the percentile."""
    errors = np.abs(np.random.default_rng(3).normal(0, 1, 500))

    sigma = fit_threshold(errors, method="sigma")
    assert sigma.parameter == DETECTION.threshold_sigma_k

    pct = fit_threshold(errors, method="percentile")
    assert pct.parameter == DETECTION.threshold_percentile


# --- persistence (Phase 4) --------------------------------------------------

def test_save_and_load_threshold_round_trip(tmp_path):
    original = Threshold(value=1.2634, method="percentile", parameter=99.9)
    save_threshold(original, tmp_path)
    assert (tmp_path / "threshold.json").exists()

    assert load_threshold(tmp_path) == original


def test_save_threshold_creates_missing_directory(tmp_path):
    target = tmp_path / "artifacts"
    save_threshold(Threshold(value=1.0, method="sigma", parameter=3.0), target)
    assert load_threshold(target).value == 1.0


def test_load_threshold_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="threshold.json"):
        load_threshold(tmp_path)
