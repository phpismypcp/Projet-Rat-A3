"""Tests for reconstruction-error computation (Phase 3, TDD)."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from anomaly_explainer.config import MODEL
from anomaly_explainer.detection.reconstruction import (
    per_feature_errors,
    reconstruct,
    reconstruction_errors,
)
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder


def _model():
    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    m = TransformerAutoEncoder(cfg)
    m.eval()
    return m, cfg


def test_per_feature_errors_shape():
    model, cfg = _model()
    x = np.random.randn(7, cfg.n_features).astype(np.float32)
    pfe = per_feature_errors(model, x)
    assert pfe.shape == (7, cfg.n_features)


def test_errors_are_non_negative():
    model, cfg = _model()
    x = np.random.randn(7, cfg.n_features).astype(np.float32)
    assert (per_feature_errors(model, x) >= 0).all()


def test_total_error_is_mean_of_per_feature():
    model, cfg = _model()
    x = np.random.randn(5, cfg.n_features).astype(np.float32)
    total = reconstruction_errors(model, x)
    pfe = per_feature_errors(model, x)
    assert total.shape == (5,)
    assert np.allclose(total, pfe.mean(axis=1), atol=1e-6)


def test_errors_are_deterministic():
    model, cfg = _model()
    x = np.random.randn(4, cfg.n_features).astype(np.float32)
    assert np.allclose(reconstruction_errors(model, x), reconstruction_errors(model, x))


def test_batching_matches_single_pass():
    model, cfg = _model()
    x = np.random.randn(100, cfg.n_features).astype(np.float32)
    big = reconstruction_errors(model, x, batch_size=1000)
    small = reconstruction_errors(model, x, batch_size=8)
    assert np.allclose(big, small, atol=1e-6)


def test_empty_input_returns_empty_arrays():
    model, cfg = _model()
    x = np.empty((0, cfg.n_features), dtype=np.float32)
    assert per_feature_errors(model, x).shape == (0, cfg.n_features)
    assert reconstruction_errors(model, x).shape == (0,)


def test_input_not_mutated():
    model, cfg = _model()
    x = np.random.randn(4, cfg.n_features).astype(np.float32)
    before = x.copy()
    _ = per_feature_errors(model, x)
    assert np.array_equal(x, before)


# --- reconstructed values (Phase 5) -----------------------------------------
# The spec requires the LLM to see the model's reconstruction, not just the
# error derived from it, so the raw rebuilt values must be obtainable.

def test_reconstruct_returns_same_shape_as_input():
    model, cfg = _model()
    x = np.random.randn(6, cfg.n_features).astype(np.float32)
    assert reconstruct(model, x).shape == (6, cfg.n_features)


def test_per_feature_errors_are_squared_difference_from_reconstruction():
    """The two functions must agree — errors are derived from the rebuild."""
    model, cfg = _model()
    x = np.random.randn(9, cfg.n_features).astype(np.float32)
    recon = reconstruct(model, x)
    assert np.allclose(per_feature_errors(model, x), (x - recon) ** 2, atol=1e-6)


def test_reconstruct_batching_matches_single_pass():
    model, cfg = _model()
    x = np.random.randn(50, cfg.n_features).astype(np.float32)
    assert np.allclose(
        reconstruct(model, x, batch_size=1000),
        reconstruct(model, x, batch_size=7),
        atol=1e-6,
    )


def test_reconstruct_empty_input():
    model, cfg = _model()
    x = np.empty((0, cfg.n_features), dtype=np.float32)
    assert reconstruct(model, x).shape == (0, cfg.n_features)


def test_reconstruct_does_not_mutate_input():
    model, cfg = _model()
    x = np.random.randn(4, cfg.n_features).astype(np.float32)
    before = x.copy()
    _ = reconstruct(model, x)
    assert np.array_equal(x, before)
