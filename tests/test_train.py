"""Tests for the auto-encoder training loop (Phase 2, TDD)."""

from __future__ import annotations

import dataclasses

from anomaly_explainer.config import MODEL, TRAIN
from anomaly_explainer.data.preprocess import prepare_data
from anomaly_explainer.model.train import train_autoencoder


def _fast_configs():
    model_cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    train_cfg = dataclasses.replace(TRAIN, epochs=8, batch_size=32)
    return model_cfg, train_cfg


def test_train_returns_model_and_history(synthetic_df):
    prepared = prepare_data(synthetic_df)
    model_cfg, train_cfg = _fast_configs()
    model, history = train_autoencoder(prepared, model_cfg, train_cfg)
    assert model is not None
    assert "train_loss" in history and "val_loss" in history
    assert len(history["train_loss"]) >= 1


def test_training_reduces_reconstruction_loss(synthetic_df):
    prepared = prepare_data(synthetic_df)
    model_cfg, train_cfg = _fast_configs()
    _, history = train_autoencoder(prepared, model_cfg, train_cfg)
    # Loss at the end should be lower than at the start (model is learning).
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_training_is_reproducible(synthetic_df):
    prepared = prepare_data(synthetic_df)
    model_cfg, train_cfg = _fast_configs()
    _, h1 = train_autoencoder(prepared, model_cfg, train_cfg)
    _, h2 = train_autoencoder(prepared, model_cfg, train_cfg)
    assert h1["train_loss"] == h2["train_loss"]
