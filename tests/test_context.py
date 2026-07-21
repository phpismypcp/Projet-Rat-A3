"""Tests for building the LLM explanation context (Phase 5, TDD).

The context is the evidence bundle the spec requires the LLM to reason over:
original values, the model's reconstruction, per-attribute error, and the
severity score — with values returned in **real units**, not z-scores, so an
explanation can say "Amount = 1809.68" rather than "Amount = 4.7".
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from anomaly_explainer.config import DATASET, MODEL
from anomaly_explainer.detection.engine import AnomalyDetector
from anomaly_explainer.explain.context import (
    AttributeDeviation,
    ExplanationContext,
    build_context,
)
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder

N_FEATURES = len(DATASET.feature_cols)


@pytest.fixture
def fitted_scaler():
    rng = np.random.default_rng(0)
    raw = rng.normal(loc=100.0, scale=25.0, size=(200, N_FEATURES))
    return StandardScaler().fit(raw)


@pytest.fixture
def detection(fitted_scaler):
    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    model = TransformerAutoEncoder(cfg)
    model.eval()
    detector = AnomalyDetector(
        model=model, threshold=0.5, feature_names=DATASET.feature_cols
    )
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, N_FEATURES)).astype(np.float32)
    return detector.detect(x), x, fitted_scaler


def test_build_context_returns_expected_type(detection):
    out, x, scaler = detection
    ctx = build_context(out, index=0, x_scaled=x, scaler=scaler, top_k=5)
    assert isinstance(ctx, ExplanationContext)
    assert all(isinstance(a, AttributeDeviation) for a in ctx.attributes)


def test_context_carries_the_detection_verdict(detection):
    out, x, scaler = detection
    ctx = build_context(out, index=2, x_scaled=x, scaler=scaler, top_k=5)

    assert ctx.severity == pytest.approx(float(out.severity[2]))
    assert ctx.error == pytest.approx(float(out.errors[2]))
    assert ctx.threshold == pytest.approx(out.threshold)
    assert ctx.is_anomaly == bool(out.is_anomaly[2])


def test_attributes_are_top_k_sorted_by_error(detection):
    out, x, scaler = detection
    ctx = build_context(out, index=0, x_scaled=x, scaler=scaler, top_k=6)

    assert len(ctx.attributes) == 6
    errors = [a.error for a in ctx.attributes]
    assert errors == sorted(errors, reverse=True)
    # they must be the genuinely largest errors of that row
    assert errors[0] == pytest.approx(float(out.per_feature_errors[0].max()))


def test_values_are_returned_in_original_units(detection):
    """This is the point of the module: no z-scores reach the LLM."""
    out, x, scaler = detection
    ctx = build_context(out, index=1, x_scaled=x, scaler=scaler, top_k=N_FEATURES)

    expected = scaler.inverse_transform(x[1].reshape(1, -1))[0]
    by_name = {a.name: a.original for a in ctx.attributes}
    for pos, name in enumerate(DATASET.feature_cols):
        assert by_name[name] == pytest.approx(expected[pos], rel=1e-4)


def test_reconstructed_values_are_also_in_original_units(detection):
    out, x, scaler = detection
    ctx = build_context(out, index=1, x_scaled=x, scaler=scaler, top_k=N_FEATURES)

    expected = scaler.inverse_transform(out.reconstructions[1].reshape(1, -1))[0]
    by_name = {a.name: a.reconstructed for a in ctx.attributes}
    for pos, name in enumerate(DATASET.feature_cols):
        assert by_name[name] == pytest.approx(expected[pos], rel=1e-4)


def test_top_k_is_clamped_to_feature_count(detection):
    out, x, scaler = detection
    ctx = build_context(out, index=0, x_scaled=x, scaler=scaler, top_k=999)
    assert len(ctx.attributes) == N_FEATURES


def test_transaction_id_is_carried_when_given(detection):
    out, x, scaler = detection
    ctx = build_context(
        out, index=0, x_scaled=x, scaler=scaler, top_k=3, transaction_id=4242
    )
    assert ctx.transaction_id == 4242


def test_rejects_out_of_range_index(detection):
    out, x, scaler = detection
    with pytest.raises(IndexError, match="out of range"):
        build_context(out, index=99, x_scaled=x, scaler=scaler, top_k=3)


def test_rejects_non_positive_top_k(detection):
    out, x, scaler = detection
    with pytest.raises(ValueError, match="top_k"):
        build_context(out, index=0, x_scaled=x, scaler=scaler, top_k=0)


def test_context_serializes_to_dict(detection):
    import json

    out, x, scaler = detection
    payload = build_context(out, index=0, x_scaled=x, scaler=scaler, top_k=3).to_dict()
    assert json.dumps(payload)
    assert len(payload["attributes"]) == 3
