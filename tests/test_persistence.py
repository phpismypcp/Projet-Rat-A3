"""Tests for saving/loading model + scaler artifacts (Phase 2, TDD)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from anomaly_explainer.config import MODEL
from anomaly_explainer.data.preprocess import prepare_data
from anomaly_explainer.model.persistence import load_artifacts, save_artifacts
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder


def test_save_load_roundtrip_reconstruction_identical(synthetic_df, tmp_path):
    prepared = prepare_data(synthetic_df)
    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    model = TransformerAutoEncoder(cfg)
    model.eval()

    x = torch.from_numpy(prepared.X_val[:4])
    with torch.no_grad():
        before = model(x)

    save_artifacts(model, prepared.scaler, cfg, tmp_path)
    loaded_model, loaded_scaler, loaded_cfg = load_artifacts(tmp_path)
    loaded_model.eval()

    with torch.no_grad():
        after = loaded_model(x)

    assert torch.allclose(before, after, atol=1e-6)
    assert loaded_cfg == cfg
    assert loaded_scaler.mean_.shape == prepared.scaler.mean_.shape


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path / "nope")
