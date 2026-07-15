"""Tests for the raw-data loader and schema validation (Phase 1, TDD)."""

from __future__ import annotations

import pandas as pd
import pytest

from anomaly_explainer.config import DATASET
from anomaly_explainer.data.loader import load_raw, validate_schema


def test_load_raw_returns_dataframe_with_expected_columns(synthetic_csv):
    df = load_raw(synthetic_csv)
    assert isinstance(df, pd.DataFrame)
    for col in (*DATASET.feature_cols, DATASET.label_col):
        assert col in df.columns


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.csv")


def test_validate_schema_passes_on_good_frame(synthetic_df):
    # Should not raise.
    validate_schema(synthetic_df)


def test_validate_schema_raises_on_missing_columns(synthetic_df):
    broken = synthetic_df.drop(columns=[DATASET.amount_col])
    with pytest.raises(ValueError, match="missing"):
        validate_schema(broken)


def test_validate_schema_raises_on_null_values(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, DATASET.amount_col] = None
    with pytest.raises(ValueError, match="null"):
        validate_schema(broken)
