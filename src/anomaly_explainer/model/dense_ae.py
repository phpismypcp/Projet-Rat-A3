"""Plain fully-connected auto-encoder — the baseline for the Transformer.

The project spec requires a Transformer auto-encoder because attention should
capture interactions between features. That is a *claim*, and a claim needs a
control: without a plain auto-encoder to compare against, a reported PR-AUC has
nothing to be better than.

This model is deliberately ordinary — stacked ``Linear`` layers down to the same
latent size and back up. It is trained by the same loop, on the same normal-only
data, with the same seed, so a difference in results points at the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from anomaly_explainer.config import DENSE_MODEL, DenseModelConfig


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters — used to report capacity alongside scores."""
    return sum(p.numel() for p in model.parameters())


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    """Stack Linear+GELU+Dropout blocks, leaving the final layer bare.

    The last layer is left without activation so the encoder can emit any latent
    value and the decoder can reproduce standardized features (which are signed).
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DenseAutoEncoder(nn.Module):
    """Ordinary auto-encoder: ``n_features -> hidden... -> latent -> ... -> n_features``."""

    def __init__(self, config: DenseModelConfig = DENSE_MODEL) -> None:
        super().__init__()
        if not config.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer width")

        self.config = config
        hidden = list(config.hidden_dims)

        self.encoder = _mlp(
            [config.n_features, *hidden, config.latent_dim], config.dropout
        )
        self.decoder = _mlp(
            [config.latent_dim, *reversed(hidden), config.n_features], config.dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
