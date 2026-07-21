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
    """How the continuous reconstruction error becomes a yes/no decision.

    ``percentile`` is the default, and that choice is empirical. The normal
    reconstruction errors are strongly right-skewed (mean ~0.044 vs median
    ~0.007, std ~3.19), so the textbook ``mean + 3*sigma`` rule lands at ~9.6 —
    far out in the tail, catching only 5% of frauds. The 99.9th percentile sits
    at ~1.26 and was the F1-optimal point of the Phase 4 sweep on validation
    (precision 0.80, recall 0.69). ``sigma`` is kept for comparison only.
    """

    threshold_method: str = "percentile"
    threshold_sigma_k: float = 3.0
    threshold_percentile: float = 99.9


# --- LLM explainer (Ollama) -------------------------------------------------

@dataclass(frozen=True)
class ExplainerConfig:
    """Local LLM settings.

    Measured on this CPU-only machine with ``llama3.2:3b`` (Q4_K_M, ~10 tok/s):
    a warm explanation takes 6-9 s, a cold one ~19 s because the 2 GB model has
    to be read back into RAM.

    ``keep_alive`` is why that matters. Ollama's default is 5 minutes, so a pause
    during a live demo silently unloads the model and the next explanation takes
    three times as long. Pinning it to 30 minutes keeps the demo responsive.

    ``json_format`` asks Ollama to constrain decoding to valid JSON, which makes
    the structured explanation parseable instead of best-effort scraped prose.
    """

    ollama_host: str = "http://localhost:11434"
    model_name: str = "llama3.2:3b"
    request_timeout_s: int = 120   # generous: covers a cold model load
    keep_alive: str = "30m"
    json_format: bool = True
    temperature: float = 0.2       # low: explanations should be stable, not creative
    top_k_attributes: int = 6      # how many most-deviant features to surface
    # The deliverable and its defense are in French, so analyst-facing text is
    # French by default. Switch to "en" here if an English demo is needed.
    language: str = "fr"


DATASET = DatasetConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
DETECTION = DetectionConfig()
EXPLAINER = ExplainerConfig()
