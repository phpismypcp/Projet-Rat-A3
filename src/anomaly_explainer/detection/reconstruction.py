"""Compute reconstruction errors from the auto-encoder.

The auto-encoder rebuilds each transaction; the squared difference between the
input and its reconstruction is the *reconstruction error*. A large error means
the model could not reproduce the transaction — the signal for an anomaly.

Two granularities:
- **per-feature error** (n, n_features): which attributes were badly rebuilt —
  used by the LLM explainer to name the responsible attributes.
- **total error** (n,): the mean across features — used for thresholding.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

_DEFAULT_BATCH = 4096


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_feature_errors(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = _DEFAULT_BATCH,
) -> np.ndarray:
    """Return squared error per feature, shape ``(n, n_features)``.

    Runs in eval mode without gradients, batched to bound memory on the full
    dataset. Does not mutate ``x``.
    """
    dev = device or _resolve_device()
    model = model.to(dev)
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size]))
            chunk = chunk.to(dev)
            recon = model(chunk)
            sq = (chunk - recon) ** 2
            outputs.append(sq.cpu().numpy())

    if not outputs:  # empty input
        return np.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def reconstruction_errors(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = _DEFAULT_BATCH,
) -> np.ndarray:
    """Return the total (mean-over-features) error per sample, shape ``(n,)``."""
    pfe = per_feature_errors(model, x, device=device, batch_size=batch_size)
    if pfe.size == 0:
        return np.empty((0,), dtype=np.float32)
    return pfe.mean(axis=1).astype(np.float32)
