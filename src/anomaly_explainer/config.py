"""Central configuration for the Anomaly Explainer project.

All tunable values live here so that no magic numbers are scattered across the
codebase. Values are grouped into immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = DATA_DIR / "artifacts"
RAW_CSV: Path = DATA_DIR / "creditcard.csv"


# --- Dataset schema ---------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Schema of the Kaggle Credit Card Fraud dataset."""

    time_col: str = "Time"
    amount_col: str = "Amount"
    label_col: str = "Class"
    # V1..V28 PCA components
    pca_cols: tuple[str, ...] = field(
        default_factory=lambda: tuple(f"V{i}" for i in range(1, 29))
    )
    n_expected_rows: int = 284_807

    @property
    def feature_cols(self) -> tuple[str, ...]:
        """Model input columns (everything except the label)."""
        return (self.time_col, *self.pca_cols, self.amount_col)


# --- Model / training -------------------------------------------------------

# These defaults suit a GPU (or a fast machine): training auto-detects CUDA. On a
# CPU-only box, shrink d_model / layers / epochs here to keep runtime reasonable.
# config.py is the single place to tune the model — nothing is hardcoded elsewhere.
@dataclass(frozen=True)
class ModelConfig:
    n_features: int = 30            # Time + V1..V28 + Amount
    d_model: int = 64              # per-feature embedding dim
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    latent_dim: int = 16
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    early_stopping_patience: int = 5
    random_seed: int = 42


# --- Detection --------------------------------------------------------------

@dataclass(frozen=True)
class DetectionConfig:
    # Threshold = mean + k * std of reconstruction error on normal validation set.
    # A percentile-based alternative is also supported by the threshold module.
    threshold_sigma_k: float = 3.0
    threshold_percentile: float = 99.5


# --- LLM explainer (Ollama) -------------------------------------------------

@dataclass(frozen=True)
class ExplainerConfig:
    ollama_host: str = "http://localhost:11434"
    model_name: str = "llama3.2:3b"
    request_timeout_s: int = 120
    top_k_attributes: int = 6      # how many most-deviant features to surface


DATASET = DatasetConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
DETECTION = DetectionConfig()
EXPLAINER = ExplainerConfig()
