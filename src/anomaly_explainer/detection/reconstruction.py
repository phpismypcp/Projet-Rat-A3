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


def reconstruct(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = _DEFAULT_BATCH,
) -> np.ndarray:
    """Return the model's rebuilt values, shape ``(n, n_features)``.

    The explainer needs the reconstruction itself, not only the error derived
    from it: "the model expected an amount near 88 but saw 1809" is far more
    actionable for an analyst than "the error on Amount was 12.1".

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
            recon = model(chunk.to(dev))
            outputs.append(recon.cpu().numpy())

    if not outputs:  # empty input
        return np.empty((0, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def per_feature_errors(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = _DEFAULT_BATCH,
) -> np.ndarray:
    """Return squared error per feature, shape ``(n, n_features)``.

    Derived from :func:`reconstruct` so the errors and the rebuilt values shown
    to the analyst can never disagree. Does not mutate ``x``.
    """
    recon = reconstruct(model, x, device=device, batch_size=batch_size)
    if recon.size == 0:
        return recon
    return ((np.asarray(x, dtype=np.float32) - recon) ** 2).astype(np.float32)


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
