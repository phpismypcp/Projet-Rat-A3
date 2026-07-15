"""Load the raw Credit Card Fraud CSV and validate its schema.

Validation happens at this system boundary so downstream modules can trust the
data (fail-fast on anything unexpected).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anomaly_explainer.config import DATASET, RAW_CSV, DatasetConfig


def validate_schema(df: pd.DataFrame, dataset: DatasetConfig = DATASET) -> None:
    """Raise ``ValueError`` if the frame does not match the expected schema.

    Checks required columns are present and that there are no null values in
    the feature or label columns.
    """
    required = (*dataset.feature_cols, dataset.label_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    null_counts = df[list(required)].isnull().sum()
    offending = null_counts[null_counts > 0]
    if not offending.empty:
        raise ValueError(f"CSV contains null values in columns: {dict(offending)}")


def load_raw(
    path: str | Path | None = None,
    *,
    dataset: DatasetConfig = DATASET,
    validate: bool = True,
) -> pd.DataFrame:
    """Read the raw dataset from ``path`` (defaults to the configured CSV).

    Args:
        path: CSV location; falls back to :data:`RAW_CSV`.
        dataset: schema configuration.
        validate: run :func:`validate_schema` after loading.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if the schema is invalid (when ``validate`` is True).
    """
    csv_path = Path(path) if path is not None else RAW_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if validate:
        validate_schema(df, dataset)
    return df
