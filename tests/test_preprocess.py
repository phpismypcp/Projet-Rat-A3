"""Tests for preprocessing: split, scaling, train/val/test preparation (TDD)."""

from __future__ import annotations

import numpy as np

from anomaly_explainer.config import DATASET, TRAIN
from anomaly_explainer.data.preprocess import (
    PreparedData,
    prepare_data,
    split_normal_fraud,
)


def test_split_normal_fraud_separates_by_class(synthetic_df):
    normal, fraud = split_normal_fraud(synthetic_df)
    assert (normal[DATASET.label_col] == 0).all()
    assert (fraud[DATASET.label_col] == 1).all()
    assert len(normal) + len(fraud) == len(synthetic_df)


def test_split_normal_fraud_does_not_mutate_input(synthetic_df):
    before = synthetic_df.copy(deep=True)
    _ = split_normal_fraud(synthetic_df)
    # original frame untouched (immutability principle)
    assert synthetic_df.equals(before)


def test_prepare_data_returns_prepared_data(synthetic_df):
    prepared = prepare_data(synthetic_df)
    assert isinstance(prepared, PreparedData)


def test_prepare_train_contains_only_normal(synthetic_df):
    prepared = prepare_data(synthetic_df)
    # Training set is normal-only by construction, so there is no label array
    # for it and its size matches ~70% of the normal rows.
    normal, _ = split_normal_fraud(synthetic_df)
    expected_train = round(len(normal) * (1 - TRAIN.val_fraction - TRAIN.test_fraction))
    assert abs(prepared.X_train.shape[0] - expected_train) <= 1


def test_prepare_val_and_test_contain_fraud(synthetic_df):
    prepared = prepare_data(synthetic_df)
    assert prepared.y_val.sum() > 0
    assert prepared.y_test.sum() > 0


def test_prepare_feature_dimension_matches_config(synthetic_df):
    prepared = prepare_data(synthetic_df)
    assert prepared.X_train.shape[1] == len(DATASET.feature_cols)
    assert prepared.feature_names == DATASET.feature_cols


def test_scaler_fit_on_train_only_gives_zero_mean_train(synthetic_df):
    prepared = prepare_data(synthetic_df)
    # After standardization fit on train, train columns have ~zero mean.
    assert np.allclose(prepared.X_train.mean(axis=0), 0.0, atol=1e-6)


def test_prepare_data_is_reproducible(synthetic_df):
    a = prepare_data(synthetic_df)
    b = prepare_data(synthetic_df)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.X_test, b.X_test)


def test_prepare_data_does_not_mutate_input(synthetic_df):
    before = synthetic_df.copy(deep=True)
    _ = prepare_data(synthetic_df)
    assert synthetic_df.equals(before)
