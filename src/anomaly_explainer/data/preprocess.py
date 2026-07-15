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
