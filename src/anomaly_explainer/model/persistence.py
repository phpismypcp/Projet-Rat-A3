"""Save and load model weights, the fitted scaler, and the model config.

Artifacts are written as three files in a directory:
- ``model.pt``    : the auto-encoder's ``state_dict``
- ``scaler.pkl``  : the fitted ``StandardScaler``
- ``config.json`` : the ``ModelConfig`` used to build the network

Keeping the config alongside the weights means we can always rebuild the exact
architecture before loading the weights.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from anomaly_explainer.config import ModelConfig
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder

MODEL_FILE = "model.pt"
SCALER_FILE = "scaler.pkl"
CONFIG_FILE = "config.json"


def save_artifacts(
    model: TransformerAutoEncoder,
    scaler: StandardScaler,
    config: ModelConfig,
    directory: str | Path,
) -> Path:
    """Persist ``model``, ``scaler`` and ``config`` into ``directory``."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out / MODEL_FILE)
    with open(out / SCALER_FILE, "wb") as fh:
        pickle.dump(scaler, fh)
    with open(out / CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(config), fh, indent=2)
    return out


def load_artifacts(
    directory: str | Path,
) -> tuple[TransformerAutoEncoder, StandardScaler, ModelConfig]:
    """Rebuild the model from saved config + weights, and load the scaler.

    Raises:
        FileNotFoundError: if the directory or any artifact is missing.
    """
    src = Path(directory)
    if not src.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {src}")

    for name in (MODEL_FILE, SCALER_FILE, CONFIG_FILE):
        if not (src / name).exists():
            raise FileNotFoundError(f"Missing artifact: {src / name}")

    with open(src / CONFIG_FILE, encoding="utf-8") as fh:
        config = ModelConfig(**json.load(fh))

    model = TransformerAutoEncoder(config)
    model.load_state_dict(torch.load(src / MODEL_FILE, map_location="cpu"))
    model.eval()

    with open(src / SCALER_FILE, "rb") as fh:
        scaler = pickle.load(fh)

    return model, scaler, config
