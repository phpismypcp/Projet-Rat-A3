"""Train the Transformer auto-encoder and save artifacts.

Usage::

    PYTHONPATH=src python scripts/train_model.py

Loads the real dataset, prepares normal-only training data, trains with early
stopping, prints the loss history, and writes model + scaler + config into
``data/artifacts/``.
"""

from __future__ import annotations

import time

from anomaly_explainer.config import ARTIFACTS_DIR, MODEL, TRAIN
from anomaly_explainer.data.loader import load_raw
from anomaly_explainer.data.preprocess import prepare_data
from anomaly_explainer.model.persistence import save_artifacts
from anomaly_explainer.model.train import resolve_device, train_autoencoder


def main() -> int:
    print("[..] loading dataset ...")
    df = load_raw()
    prepared = prepare_data(df)
    print(
        f"[ok] train={prepared.X_train.shape[0]:,} (normal-only) | "
        f"val={prepared.X_val.shape[0]:,} | test={prepared.X_test.shape[0]:,}"
    )

    print(f"[..] device: {resolve_device()}")
    print(f"[..] training up to {TRAIN.epochs} epochs (early stopping) ...")
    start = time.perf_counter()

    def log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
        elapsed = time.perf_counter() - start
        print(
            f"    epoch {epoch:2d}  train={train_loss:.6f}  val={val_loss:.6f}"
            f"  ({elapsed:.0f}s elapsed)",
            flush=True,
        )

    model, _history = train_autoencoder(prepared, MODEL, TRAIN, on_epoch_end=log_epoch)

    out = save_artifacts(model, prepared.scaler, MODEL, ARTIFACTS_DIR)
    print(f"[ok] artifacts saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
