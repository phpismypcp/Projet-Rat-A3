"""Preprocessing: split normal vs fraud, scale features, build train/val/test.

Key design decisions
--------------------
* **Train on normal only.** The auto-encoder must learn the structure of normal
  transactions, so the training set contains no fraud. Fraud rows are reserved
  for validation and test, where they are used to tune the threshold and measure
  performance.
* **Fit the scaler on the training set only.** Standardizing with statistics
  taken from the whole dataset would leak information from val/test into
  training. We fit on train and merely *apply* the transform elsewhere.
* **Immutability.** Functions never modify their input frame; they return new
  objects (see the project coding-style rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from anomaly_explainer.config import DATASET, TRAIN, DatasetConfig, TrainConfig


@dataclass(frozen=True)
class PreparedData:
    """Model-ready arrays plus the fitted scaler.

    ``X_train`` is normal-only (hence has no label array). ``X_val``/``X_test``
    mix normal and fraud rows, with ``y_val``/``y_test`` giving the true labels
    (1 = fraud) used only for threshold tuning and evaluation.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    scaler: StandardScaler


def split_normal_fraud(
    df: pd.DataFrame, dataset: DatasetConfig = DATASET
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(normal_df, fraud_df)`` without mutating ``df``."""
    is_fraud = df[dataset.label_col] == 1
    normal = df.loc[~is_fraud].copy()
    fraud = df.loc[is_fraud].copy()
    return normal, fraud


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    """Fit and return a new ``StandardScaler`` on the training features."""
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def _features(df: pd.DataFrame, dataset: DatasetConfig) -> np.ndarray:
    return df[list(dataset.feature_cols)].to_numpy(dtype=np.float32)


def prepare_data_chronological(
    df: pd.DataFrame,
    dataset: DatasetConfig = DATASET,
    train: TrainConfig = TRAIN,
) -> PreparedData:
    """Split by time instead of at random: train on the past, evaluate on the future.

    The default :func:`prepare_data` shuffles rows, so training and evaluation
    interleave across the dataset's 48-hour window. A deployed detector never has
    that luxury — it scores transactions that occur *after* everything it learned
    from. This split reproduces those conditions and is therefore the honest
    reference point for what performance would look like in production.

    Rows are cut at time quantiles rather than by position, so transactions
    sharing a timestamp always land in the same split and no evaluation row can
    precede a training row. Fraud rows inside the training window are discarded:
    they cannot join the normal-only training set, and moving them into a later
    split would break the very ordering this function exists to preserve.

    Raises:
        ValueError: if the configured fractions leave no training window.
    """
    eval_fraction = train.val_fraction + train.test_fraction
    if not 0 < eval_fraction < 1:
        raise ValueError(
            f"val_fraction + test_fraction must be a fraction in (0, 1), "
            f"got {eval_fraction}"
        )

    ordered = df.sort_values(dataset.time_col, kind="mergesort")
    times = ordered[dataset.time_col]
    train_end = float(times.quantile(1.0 - eval_fraction))
    val_end = float(times.quantile(1.0 - train.test_fraction))

    is_train_window = ordered[dataset.time_col] <= train_end
    is_val_window = (ordered[dataset.time_col] > train_end) & (
        ordered[dataset.time_col] <= val_end
    )
    is_test_window = ordered[dataset.time_col] > val_end
    is_normal = ordered[dataset.label_col] == 0

    train_df = ordered.loc[is_train_window & is_normal]
    val_df = ordered.loc[is_val_window]
    test_df = ordered.loc[is_test_window]

    x_train = _features(train_df, dataset)
    scaler = fit_scaler(x_train)

    return PreparedData(
        X_train=scaler.transform(x_train).astype(np.float32),
        X_val=scaler.transform(_features(val_df, dataset)).astype(np.float32),
        y_val=val_df[dataset.label_col].to_numpy(dtype=np.int64),
        X_test=scaler.transform(_features(test_df, dataset)).astype(np.float32),
        y_test=test_df[dataset.label_col].to_numpy(dtype=np.int64),
        feature_names=dataset.feature_cols,
        scaler=scaler,
    )


def prepare_data(
    df: pd.DataFrame,
    dataset: DatasetConfig = DATASET,
    train: TrainConfig = TRAIN,
) -> PreparedData:
    """Build normal-only train plus mixed val/test sets, scaled consistently.

    Normal rows are split train/val/test by the configured fractions. Fraud rows
    are split evenly between val and test. The scaler is fit on the normal
    training features only, then applied to every split.
    """
    normal, fraud = split_normal_fraud(df, dataset)

    val_test_fraction = train.val_fraction + train.test_fraction
    # Split normal rows: first carve out (val + test), then split that in two.
    normal_train, normal_valtest = train_test_split(
        normal,
        test_size=val_test_fraction,
        random_state=train.random_seed,
        shuffle=True,
    )
    rel_test = train.test_fraction / val_test_fraction
    normal_val, normal_test = train_test_split(
        normal_valtest,
        test_size=rel_test,
        random_state=train.random_seed,
        shuffle=True,
    )

    # Split fraud rows evenly between val and test.
    fraud_val, fraud_test = train_test_split(
        fraud,
        test_size=0.5,
        random_state=train.random_seed,
        shuffle=True,
    )

    val_df = pd.concat([normal_val, fraud_val], ignore_index=True)
    test_df = pd.concat([normal_test, fraud_test], ignore_index=True)

    x_train = _features(normal_train, dataset)
    scaler = fit_scaler(x_train)

    x_train = scaler.transform(x_train).astype(np.float32)
    x_val = scaler.transform(_features(val_df, dataset)).astype(np.float32)
    x_test = scaler.transform(_features(test_df, dataset)).astype(np.float32)

    y_val = val_df[dataset.label_col].to_numpy(dtype=np.int64)
    y_test = test_df[dataset.label_col].to_numpy(dtype=np.int64)

    return PreparedData(
        X_train=x_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        feature_names=dataset.feature_cols,
        scaler=scaler,
    )
