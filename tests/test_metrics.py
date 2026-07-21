"""Tests for evaluation metrics (Phase 4, TDD).

Metrics are hand-computed on tiny explicit arrays so the expected values can be
verified by reading the test, not by trusting the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_explainer.evaluation.metrics import (
    ClassificationMetrics,
    EvaluationReport,
    classification_metrics,
    classify,
    evaluate,
    ranking_metrics,
)


# --- classify ---------------------------------------------------------------

def test_classify_flags_errors_strictly_above_threshold():
    errors = np.array([0.1, 0.5, 1.0, 2.0])
    flags = classify(errors, threshold=1.0)
    # 1.0 is NOT above 1.0 — strict comparison, matching AnomalyDetector.
    assert flags.tolist() == [False, False, False, True]


def test_classify_does_not_mutate_input():
    errors = np.array([0.1, 2.0])
    before = errors.copy()
    classify(errors, threshold=1.0)
    assert np.array_equal(errors, before)


# --- confusion counts & derived metrics -------------------------------------

def test_classification_metrics_counts_confusion_matrix():
    #            TP    FP    TN    FN
    y_true = np.array([1, 0, 0, 1])
    errors = np.array([2.0, 2.0, 0.0, 0.0])
    m = classification_metrics(y_true, errors, threshold=1.0)

    assert (m.true_positives, m.false_positives) == (1, 1)
    assert (m.true_negatives, m.false_negatives) == (1, 1)


def test_precision_recall_f1_match_definitions():
    # 3 TP, 1 FP, 2 FN  ->  P = 3/4, R = 3/5
    m = ClassificationMetrics(
        true_positives=3, false_positives=1, true_negatives=10, false_negatives=2
    )
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.6)
    assert m.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_metrics_are_zero_when_nothing_is_flagged():
    """No predictions must give 0.0, not a ZeroDivisionError."""
    m = ClassificationMetrics(
        true_positives=0, false_positives=0, true_negatives=5, false_negatives=3
    )
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_perfect_separation_gives_f1_of_one():
    y_true = np.array([0, 0, 1, 1])
    errors = np.array([0.1, 0.2, 5.0, 6.0])
    m = classification_metrics(y_true, errors, threshold=1.0)
    assert m.f1 == pytest.approx(1.0)


# --- ranking metrics --------------------------------------------------------

def test_ranking_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    errors = np.array([0.1, 0.2, 5.0, 6.0])
    r = ranking_metrics(y_true, errors)
    assert r.roc_auc == pytest.approx(1.0)
    assert r.pr_auc == pytest.approx(1.0)


def test_ranking_metrics_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        ranking_metrics(np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]))


# --- input validation at the boundary ---------------------------------------

def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        classification_metrics(np.array([0, 1]), np.array([1.0]), threshold=0.5)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        classification_metrics(np.array([]), np.array([]), threshold=0.5)


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0 or 1"):
        classification_metrics(np.array([0, 2]), np.array([1.0, 2.0]), threshold=0.5)


# --- full report ------------------------------------------------------------

def test_evaluate_builds_complete_report():
    y_true = np.array([0, 0, 0, 1, 1])
    errors = np.array([0.1, 0.2, 0.3, 5.0, 6.0])
    report = evaluate(y_true, errors, threshold=1.0)

    assert isinstance(report, EvaluationReport)
    assert report.threshold == 1.0
    assert report.n_samples == 5
    assert report.n_positives == 2
    assert report.metrics.f1 == pytest.approx(1.0)
    assert report.ranking.roc_auc == pytest.approx(1.0)


def test_report_serializes_to_plain_dict():
    """The report is written to JSON, so it must be dict-serializable."""
    import json

    y_true = np.array([0, 0, 1, 1])
    errors = np.array([0.1, 0.2, 5.0, 6.0])
    payload = evaluate(y_true, errors, threshold=1.0).to_dict()

    assert json.dumps(payload)  # must not raise
    assert payload["metrics"]["precision"] == pytest.approx(1.0)
    assert payload["threshold"] == 1.0
