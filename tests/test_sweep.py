"""Tests for the threshold sweep (Phase 4, TDD).

The sweep is the tool that fixes the miscalibrated default threshold: it walks a
range of candidate thresholds and reports the precision/recall trade-off at each,
so an operating point can be chosen on evidence rather than by assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_explainer.evaluation.sweep import (
    DEFAULT_PERCENTILES,
    SweepPoint,
    SweepResult,
    sweep_percentiles,
)


@pytest.fixture
def separable_errors():
    """Normal errors near zero, fraud errors clearly higher.

    900 normals in [0, 1), 100 frauds in [2, 3) — separable, so a well-chosen
    threshold reaches F1 = 1.0 while a badly-chosen one does not.
    """
    rng = np.random.default_rng(0)
    normal = rng.uniform(0.0, 1.0, 900)
    fraud = rng.uniform(2.0, 3.0, 100)
    errors = np.concatenate([normal, fraud])
    y_true = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
    return normal, errors, y_true


def test_sweep_returns_one_point_per_percentile(separable_errors):
    normal, errors, y_true = separable_errors
    result = sweep_percentiles(normal, y_true, errors, percentiles=(90.0, 95.0, 99.0))

    assert isinstance(result, SweepResult)
    assert len(result.points) == 3
    assert all(isinstance(p, SweepPoint) for p in result.points)
    assert [p.percentile for p in result.points] == [90.0, 95.0, 99.0]


def test_thresholds_increase_with_percentile(separable_errors):
    normal, errors, y_true = separable_errors
    result = sweep_percentiles(normal, y_true, errors)

    values = [p.threshold for p in result.points]
    assert values == sorted(values)


def test_higher_threshold_trades_recall_for_precision(separable_errors):
    """The central trade-off: raising the threshold must not increase recall."""
    normal, errors, y_true = separable_errors
    result = sweep_percentiles(normal, y_true, errors)

    recalls = [p.metrics.recall for p in result.points]
    assert recalls == sorted(recalls, reverse=True)


def test_best_by_f1_picks_the_maximum(separable_errors):
    normal, errors, y_true = separable_errors
    result = sweep_percentiles(normal, y_true, errors)

    best = result.best_by_f1()
    assert best.metrics.f1 == max(p.metrics.f1 for p in result.points)
    # Data is separable, so a near-perfect operating point must exist.
    assert best.metrics.f1 > 0.9


def test_best_at_min_precision_maximises_recall_under_constraint(separable_errors):
    normal, errors, y_true = separable_errors
    result = sweep_percentiles(normal, y_true, errors)

    best = result.best_at_min_precision(0.9)
    eligible = [p for p in result.points if p.metrics.precision >= 0.9]
    assert best.metrics.precision >= 0.9
    assert best.metrics.recall == max(p.metrics.recall for p in eligible)


def test_best_at_min_precision_raises_when_unattainable(separable_errors):
    normal, errors, y_true = separable_errors
    with pytest.raises(ValueError, match="no threshold"):
        result = sweep_percentiles(normal, y_true, errors)
        result.best_at_min_precision(1.01)


def test_default_percentiles_focus_on_the_tail():
    """Fraud is 0.17% of data, so useful thresholds live in the far tail."""
    assert min(DEFAULT_PERCENTILES) >= 95.0
    assert max(DEFAULT_PERCENTILES) < 100.0
    assert list(DEFAULT_PERCENTILES) == sorted(DEFAULT_PERCENTILES)


def test_sweep_rejects_empty_percentiles(separable_errors):
    normal, errors, y_true = separable_errors
    with pytest.raises(ValueError, match="non-empty"):
        sweep_percentiles(normal, y_true, errors, percentiles=())


def test_sweep_rejects_out_of_range_percentiles(separable_errors):
    normal, errors, y_true = separable_errors
    with pytest.raises(ValueError, match="between 0 and 100"):
        sweep_percentiles(normal, y_true, errors, percentiles=(101.0,))


def test_sweep_rejects_empty_normal_errors(separable_errors):
    _normal, errors, y_true = separable_errors
    with pytest.raises(ValueError, match="non-empty"):
        sweep_percentiles(np.array([]), y_true, errors)


def test_sweep_result_serializes_to_dict(separable_errors):
    import json

    normal, errors, y_true = separable_errors
    result = sweep_percentiles(normal, y_true, errors, percentiles=(99.0, 99.5))
    payload = result.to_dict()

    assert json.dumps(payload)  # must not raise
    assert len(payload["points"]) == 2
    assert "f1" in payload["points"][0]["metrics"]
