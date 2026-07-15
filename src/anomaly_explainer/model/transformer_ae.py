"""Transformer Auto-Encoder for tabular anomaly detection.

Why a Transformer here
----------------------
Each transaction has 30 features. We treat **every feature as a token** and let
multi-head self-attention learn how features relate to one another (e.g. a large
``Amount`` combined with an unusual ``Time`` and an odd ``V14``). This captures
non-linear interactions a plain dense auto-encoder would miss, giving cleaner
reconstructions — and therefore sharper anomaly detection.

Flow: values -> per-feature embedding (tokenizer) -> Transformer encoder ->
compact latent vector -> expand -> Transformer decoder -> reconstruct values.

A transaction the model reconstructs poorly (high error) is anomalous.
"""

from __future__ import annotations

import torch
from torch import nn

from anomaly_explainer.config import MODEL, ModelConfig


class FeatureTokenizer(nn.Module):
    """Turn each scalar feature into a ``d_model``-dim token.

    Uses a per-feature weight/bias plus a learned feature embedding, so the model
    can tell features apart (analogous to positional embeddings in NLP).
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.feature_embedding = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.feature_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_model)
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        return tokens + self.feature_embedding


def _encoder_stack(cfg: ModelConfig, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.d_model * 4,
        dropout=cfg.dropout,
        batch_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class TransformerAutoEncoder(nn.Module):
    """Attention-based auto-encoder producing a same-shape reconstruction."""

    def __init__(self, config: ModelConfig = MODEL) -> None:
        super().__init__()
        self.config = config
        n, d = config.n_features, config.d_model

        # Encoder: tokenize -> attention -> compact latent
        self.tokenizer = FeatureTokenizer(n, d)
        self.encoder = _encoder_stack(config, config.n_encoder_layers)
        self.to_latent = nn.Linear(n * d, config.latent_dim)

        # Decoder: latent -> tokens -> attention -> per-feature scalar
        self.from_latent = nn.Linear(config.latent_dim, n * d)
        self.decoder = _encoder_stack(config, config.n_decoder_layers)
        self.output_head = nn.Linear(d, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)                    # (B, n, d)
        encoded = self.encoder(tokens)                # (B, n, d)
        flat = encoded.flatten(start_dim=1)           # (B, n*d)
        return self.to_latent(flat)                   # (B, latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        n, d = self.config.n_features, self.config.d_model
        tokens = self.from_latent(z).view(-1, n, d)   # (B, n, d)
        decoded = self.decoder(tokens)                # (B, n, d)
        return self.output_head(decoded).squeeze(-1)  # (B, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
