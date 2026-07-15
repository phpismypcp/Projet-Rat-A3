"""Tests for the Transformer auto-encoder architecture (Phase 2, TDD)."""

from __future__ import annotations

import dataclasses

import torch

from anomaly_explainer.config import MODEL
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder


def _small_model():
    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    return TransformerAutoEncoder(cfg), cfg


def test_forward_output_shape_matches_input():
    model, cfg = _small_model()
    x = torch.randn(5, cfg.n_features)
    out = model(x)
    assert out.shape == x.shape


def test_encode_returns_latent_dim():
    model, cfg = _small_model()
    x = torch.randn(5, cfg.n_features)
    z = model.encode(x)
    assert z.shape == (5, cfg.latent_dim)


def test_forward_is_deterministic_in_eval():
    model, cfg = _small_model()
    model.eval()
    x = torch.randn(3, cfg.n_features)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)


def test_parameters_update_after_optimizer_step():
    model, cfg = _small_model()
    x = torch.randn(8, cfg.n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, x)
    opt.zero_grad()
    loss.backward()
    opt.step()

    after = list(model.parameters())
    # At least one parameter tensor changed.
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
