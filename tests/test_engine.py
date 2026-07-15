"""Tests for the anomaly detection engine (Phase 3, TDD)."""

from __future__ import annotations

import dataclasses

import numpy as np

from anomaly_explainer.config import DATASET, MODEL
from anomaly_explainer.detection.engine import (
    AnomalyDetector,
    DetectionOutput,
    top_contributors,
)
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder


def _detector(threshold: float):
    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    model = TransformerAutoEncoder(cfg)
    model.eval()
    return AnomalyDetector(
        model=model, threshold=threshold, feature_names=DATASET.feature_cols
    )


def _x(n=10):
    cfg_features = len(DATASET.feature_cols)
    return np.random.randn(n, cfg_features).astype(np.float32)


def test_detect_returns_output_with_shapes():
    det = _detector(threshold=1.0)
    x = _x(10)
    out = det.detect(x)
    assert isinstance(out, DetectionOutput)
    assert out.errors.shape == (10,)
    assert out.per_feature_errors.shape == (10, len(DATASET.feature_cols))
    assert out.is_anomaly.shape == (10,)
    assert out.severity.shape == (10,)


def test_is_anomaly_matches_threshold():
    det = _detector(threshold=1.0)
    x = _x(10)
    out = det.detect(x)
    assert np.array_equal(out.is_anomaly, out.errors > det.threshold)


def test_severity_is_error_over_threshold():
    det = _detector(threshold=2.0)
    x = _x(10)
    out = det.detect(x)
    assert np.allclose(out.severity, out.errors / 2.0)


def test_severity_gt_one_iff_anomaly():
    det = _detector(threshold=1.0)
    out = det.detect(_x(20))
    assert np.array_equal(out.severity > 1.0, out.is_anomaly)


def test_high_threshold_flags_nothing():
    det = _detector(threshold=1e9)
    out = det.detect(_x(10))
    assert not out.is_anomaly.any()


def test_zero_threshold_flags_everything():
    det = _detector(threshold=0.0)
    out = det.detect(_x(10))
    assert out.is_anomaly.all()


def test_top_contributors_sorted_and_named():
    det = _detector(threshold=1.0)
    out = det.detect(_x(5))
    top = top_contributors(out, index=0, k=3)
    assert len(top) == 3
    names = [name for name, _ in top]
    values = [val for _, val in top]
    assert all(n in DATASET.feature_cols for n in names)
    assert values == sorted(values, reverse=True)  # descending by error


def test_detect_does_not_mutate_input():
    det = _detector(threshold=1.0)
    x = _x(6)
    before = x.copy()
    _ = det.detect(x)
    assert np.array_equal(x, before)
