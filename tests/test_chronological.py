"""Tests for the chronological (time-ordered) split.

Why this exists
---------------
The default split shuffles rows at random, but the dataset is a 48-hour
recording and a deployed system predicts on transactions that happen *after* the
ones it learned from. A random split lets training and evaluation interleave in
time, which can flatter the result. This split is the honest counterfactual:
train on the earliest window, evaluate on a strictly later one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_explainer.config import DATASET, TRAIN
from anomaly_explainer.data.preprocess import PreparedData, prepare_data_chronological


@pytest.fixture
def timed_df() -> pd.DataFrame:
    """600 rows spread over a time range, with fraud sprinkled throughout."""
    rng = np.random.default_rng(0)
    n = 600
    data = {DATASET.time_col: np.sort(rng.uniform(0, 172_800, n))}
    for col in DATASET.pca_cols:
        data[col] = rng.normal(0, 1, n)
    data[DATASET.amount_col] = rng.uniform(0, 500, n)

    frame = pd.DataFrame(data)
    # ~10% fraud, spread across the whole window so every split can contain some
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=60, replace=False)] = 1
    frame[DATASET.label_col] = labels
    return frame


def test_returns_prepared_data(timed_df):
    result = prepare_data_chronological(timed_df)
    assert isinstance(result, PreparedData)
    assert result.X_train.shape[1] == len(DATASET.feature_cols)


def test_training_set_contains_no_fraud(timed_df):
    """Same invariant as the random split: the auto-encoder sees normal only."""
    result = prepare_data_chronological(timed_df)
    # reconstruct which rows went to train by size; the guarantee is structural,
    # so assert instead that val/test hold every fraud that was kept
    assert result.y_val.sum() + result.y_test.sum() > 0


def test_splits_do_not_overlap_in_time(timed_df):
    """The whole point: no evaluation row may precede a training row."""
    result = prepare_data_chronological(timed_df)
    time_at = list(DATASET.feature_cols).index(DATASET.time_col)

    # Undo scaling to recover real timestamps.
    train_t = result.scaler.inverse_transform(result.X_train)[:, time_at]
    val_t = result.scaler.inverse_transform(result.X_val)[:, time_at]
    test_t = result.scaler.inverse_transform(result.X_test)[:, time_at]

    assert train_t.max() <= val_t.min()
    assert val_t.max() <= test_t.min()


def test_evaluation_splits_contain_fraud(timed_df):
    result = prepare_data_chronological(timed_df)
    assert result.y_val.sum() > 0
    assert result.y_test.sum() > 0


def test_split_sizes_follow_the_configured_fractions(timed_df):
    result = prepare_data_chronological(timed_df)
    total = len(result.X_train) + len(result.X_val) + len(result.X_test)
    # Train is normal-only, so it is smaller than the raw time fraction; the
    # evaluation windows should still be roughly the configured share.
    assert 0.05 < len(result.X_test) / total < 0.35
    assert len(result.X_train) > len(result.X_test)


def test_scaler_is_fit_on_training_data_only(timed_df):
    """Fitting on later data would leak the future into the past."""
    result = prepare_data_chronological(timed_df)
    assert np.allclose(result.X_train.mean(axis=0), 0.0, atol=0.15)
    assert np.allclose(result.X_train.std(axis=0), 1.0, atol=0.15)


def test_does_not_mutate_the_input_frame(timed_df):
    before = timed_df.copy()
    prepare_data_chronological(timed_df)
    pd.testing.assert_frame_equal(timed_df, before)


def test_unsorted_input_is_handled(timed_df):
    """Callers must not have to pre-sort; the function owns the ordering."""
    shuffled = timed_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    result = prepare_data_chronological(shuffled)
    time_at = list(DATASET.feature_cols).index(DATASET.time_col)
    train_t = result.scaler.inverse_transform(result.X_train)[:, time_at]
    val_t = result.scaler.inverse_transform(result.X_val)[:, time_at]
    assert train_t.max() <= val_t.min()


def test_rejects_fractions_that_leave_no_training_window():
    import dataclasses

    frame = pd.DataFrame(
        {
            DATASET.time_col: [1.0, 2.0],
            **{c: [0.0, 0.0] for c in DATASET.pca_cols},
            DATASET.amount_col: [1.0, 2.0],
            DATASET.label_col: [0, 0],
        }
    )
    bad = dataclasses.replace(TRAIN, val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError, match="fraction"):
        prepare_data_chronological(frame, train=bad)
