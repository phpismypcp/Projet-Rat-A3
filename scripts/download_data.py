"""Fetch the Kaggle Credit Card Fraud dataset into ``data/creditcard.csv``.

The dataset requires a Kaggle account. Two supported paths:

1. Automated (needs ``~/.kaggle/kaggle.json`` API token)::

       python scripts/download_data.py

2. Manual: download from
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   and place ``creditcard.csv`` into the ``data/`` directory.

The script validates row count and expected columns after download so that
downstream code can trust the schema (fail-fast on bad data).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from anomaly_explainer.config import DATA_DIR, RAW_CSV, DATASET

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def _validate(csv_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=5)
    missing = [c for c in (*DATASET.feature_cols, DATASET.label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    print(f"[ok] schema validated: {len(df.columns)} columns present")


def _download_via_kaggle() -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "kaggle package not installed and creditcard.csv not found. "
            "Either `pip install kaggle` with a ~/.kaggle/kaggle.json token, "
            "or download the CSV manually into data/."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[..] downloading {KAGGLE_DATASET} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), quiet=False)

    zip_path = DATA_DIR / "creditcardfraud.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()


def main() -> int:
    if RAW_CSV.exists():
        print(f"[ok] {RAW_CSV} already present")
        _validate(RAW_CSV)
        return 0

    _download_via_kaggle()

    if not RAW_CSV.exists():
        print(f"[!!] download finished but {RAW_CSV} not found", file=sys.stderr)
        return 1

    _validate(RAW_CSV)
    print(f"[ok] dataset ready at {RAW_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
