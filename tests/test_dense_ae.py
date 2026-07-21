"""Tests for the plain dense auto-encoder baseline (TDD).

This model exists to answer one question the spec's choice of a Transformer
invites: *did attention actually buy us anything?* It must therefore be a fair
comparison — same bottleneck, same training loop, same data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from anomaly_explainer.config import DENSE_MODEL, DenseModelConfig
from anomaly_explainer.model.dense_ae import DenseAutoEncoder, count_parameters


def test_output_has_same_shape_as_input():
    model = DenseAutoEncoder(DENSE_MODEL).eval()
    x = torch.randn(8, DENSE_MODEL.n_features)
    assert model(x).shape == x.shape


def test_encode_produces_the_configured_latent_size():
    """The bottleneck is what makes it an auto-encoder — it must be enforced."""
    model = DenseAutoEncoder(DENSE_MODEL).eval()
    z = model.encode(torch.randn(5, DENSE_MODEL.n_features))
    assert z.shape == (5, DENSE_MODEL.latent_dim)


def test_forward_is_deterministic_in_eval_mode():
    model = DenseAutoEncoder(DENSE_MODEL).eval()
    x = torch.randn(4, DENSE_MODEL.n_features)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_dropout_makes_training_mode_stochastic():
    cfg = dataclasses.replace(DENSE_MODEL, dropout=0.5)
    model = DenseAutoEncoder(cfg).train()
    x = torch.randn(64, cfg.n_features)
    assert not torch.allclose(model(x), model(x))


def test_wider_hidden_layers_mean_more_parameters():
    small = DenseAutoEncoder(dataclasses.replace(DENSE_MODEL, hidden_dims=(32, 16)))
    large = DenseAutoEncoder(dataclasses.replace(DENSE_MODEL, hidden_dims=(512, 256)))
    assert count_parameters(large) > count_parameters(small)


def test_accepts_a_single_hidden_layer():
    cfg = dataclasses.replace(DENSE_MODEL, hidden_dims=(32,))
    model = DenseAutoEncoder(cfg).eval()
    assert model(torch.randn(3, cfg.n_features)).shape == (3, cfg.n_features)


def test_rejects_empty_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims"):
        DenseAutoEncoder(dataclasses.replace(DENSE_MODEL, hidden_dims=()))


def test_gradients_flow_to_every_parameter():
    """A layer that never receives gradient would silently not train."""
    model = DenseAutoEncoder(DENSE_MODEL).train()
    loss = ((model(torch.randn(16, DENSE_MODEL.n_features))) ** 2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_count_parameters_matches_manual_sum():
    model = DenseAutoEncoder(DENSE_MODEL)
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


def test_config_defaults_share_the_transformer_bottleneck():
    """Equal latent size is what makes the comparison meaningful."""
    from anomaly_explainer.config import MODEL

    assert DENSE_MODEL.latent_dim == MODEL.latent_dim
    assert DENSE_MODEL.n_features == MODEL.n_features
    assert isinstance(DENSE_MODEL, DenseModelConfig)


def test_does_not_mutate_input():
    model = DenseAutoEncoder(DENSE_MODEL).eval()
    x = torch.randn(4, DENSE_MODEL.n_features)
    before = x.clone()
    with torch.no_grad():
        model(x)
    assert torch.equal(x, before)


def test_works_with_numpy_pipeline_dtype():
    """Reconstruction utilities feed float32 arrays through torch.from_numpy."""
    model = DenseAutoEncoder(DENSE_MODEL).eval()
    x = torch.from_numpy(np.random.randn(6, DENSE_MODEL.n_features).astype(np.float32))
    with torch.no_grad():
        assert model(x).shape == x.shape
