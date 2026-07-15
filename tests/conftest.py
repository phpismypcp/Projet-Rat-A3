"""Shared pytest fixtures.

We build a tiny synthetic dataset that mirrors the real Kaggle schema so unit
tests stay fast and never depend on the 150 MB CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_explainer.config import DATASET


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A small frame with the exact columns of the credit-card dataset.

    200 normal rows + 20 fraud rows. Fraud rows are shifted far from the normal
    distribution so preprocessing/detection logic has a clear signal to test.
    """
    rng = np.random.default_rng(0)
    n_normal, n_fraud = 200, 20

    def block(n: int, loc: float) -> dict:
        data = {DATASET.time_col: rng.uniform(0, 1e5, n)}
        for col in DATASET.pca_cols:
            data[col] = rng.normal(loc, 1.0, n)
        data[DATASET.amount_col] = rng.uniform(0, 500, n)
        return data

    normal = pd.DataFrame(block(n_normal, loc=0.0))
    normal[DATASET.label_col] = 0
    fraud = pd.DataFrame(block(n_fraud, loc=8.0))  # far from normal
    fraud[DATASET.label_col] = 1

    df = pd.concat([normal, fraud], ignore_index=True)
    # shuffle so order carries no information
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df) -> str:
    path = tmp_path / "creditcard.csv"
    synthetic_df.to_csv(path, index=False)
    return str(path)
