"""How much does the random split flatter our results?

Usage::

    PYTHONPATH=src python scripts/chronological_check.py --arch dense
    PYTHONPATH=src python scripts/chronological_check.py --arch transformer

The reported metrics come from a **random** train/val/test split, but the dataset
is a 48-hour recording and a deployed detector only ever scores transactions that
happen *after* the ones it learned from. A random split lets training and
evaluation interleave in time, which can make results look better than
production would.

This script trains the same architecture twice — once on the random split, once
on a strictly time-ordered one — holding the seed, the training loop, the latent
size and the thresholding rule constant, so the difference isolates the split.

**Read the fraud counts before the metrics.** The chronological split keeps far
fewer evaluation frauds (the ones inside the training window must be discarded),
so its numbers carry much wider error bars. A drop here is suggestive, not proof.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from anomaly_explainer.config import ARTIFACTS_DIR, DENSE_MODEL, DETECTION, MODEL, TRAIN
from anomaly_explainer.data.loader import load_raw
from anomaly_explainer.data.preprocess import (
    PreparedData,
    prepare_data,
    prepare_data_chronological,
)
from anomaly_explainer.detection.reconstruction import reconstruction_errors
from anomaly_explainer.detection.threshold import percentile_threshold
from anomaly_explainer.evaluation.metrics import evaluate
from anomaly_explainer.model.dense_ae import DenseAutoEncoder
from anomaly_explainer.model.train import train_autoencoder
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder

REPORT_FILE = "chronological_check.json"


def _factory(arch: str):
    if arch == "dense":
        return lambda: DenseAutoEncoder(dataclasses.replace(DENSE_MODEL, hidden_dims=(512, 256)))
    return lambda: TransformerAutoEncoder(MODEL)


def _run(prepared: PreparedData, arch: str, label: str, epochs: int) -> dict:
    print(f"\n[..] {label}: training {arch} "
          f"(train={len(prepared.X_train):,}, test frauds={int(prepared.y_test.sum())}) ...")
    start = time.perf_counter()

    def log(epoch: int, train_loss: float, val_loss: float) -> None:
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}",
                  flush=True)

    model, history = train_autoencoder(
        prepared,
        train_config=dataclasses.replace(TRAIN, epochs=epochs),
        on_epoch_end=log,
        model_factory=_factory(arch),
    )
    elapsed = time.perf_counter() - start
    n_epochs = len(history["train_loss"])
    print(f"[ok] {label}: {n_epochs} epochs in {elapsed:.0f}s "
          f"({'converged' if n_epochs < epochs else 'HIT EPOCH CAP'})")

    val_errors = reconstruction_errors(model, prepared.X_val)
    test_errors = reconstruction_errors(model, prepared.X_test)
    threshold = percentile_threshold(
        val_errors[prepared.y_val == 0], DETECTION.threshold_percentile
    )
    report = evaluate(prepared.y_test, test_errors, threshold)

    return {
        "split": label,
        "architecture": arch,
        "epochs": n_epochs,
        "converged": n_epochs < epochs,
        "train_rows": int(len(prepared.X_train)),
        "val_frauds": int(prepared.y_val.sum()),
        **report.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["dense", "transformer"], default="dense")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    df = load_raw()
    rows = [
        _run(prepare_data(df), args.arch, "random (reported)", args.epochs),
        _run(prepare_data_chronological(df), args.arch, "chronological", args.epochs),
    ]

    print(f"\n{'split':<20} {'test frauds':>12} {'PR-AUC':>8} {'ROC-AUC':>8} "
          f"{'prec':>6} {'recall':>7} {'F1':>6} {'FP':>5}")
    print("-" * 78)
    for r in rows:
        m, k = r["metrics"], r["ranking"]
        print(f"{r['split']:<20} {r['n_positives']:>12} {k['pr_auc']:>8.4f} "
              f"{k['roc_auc']:>8.4f} {m['precision']:>6.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>6.3f} {m['false_positives']:>5}")

    delta = rows[1]["ranking"]["pr_auc"] - rows[0]["ranking"]["pr_auc"]
    print(f"\nPR-AUC change moving to a time-ordered split: {delta:+.4f}")
    if delta < -0.05:
        print("=> the random split MATERIALLY flatters the reported numbers.")
    elif delta < 0:
        print("=> the random split flatters the numbers slightly.")
    else:
        print("=> no evidence the random split flatters the numbers.")
    print(f"\nCAVEAT: chronological keeps only {rows[1]['n_positives']} test frauds "
          f"vs {rows[0]['n_positives']} — wide error bars, treat as indicative.")

    out = Path(ARTIFACTS_DIR) / REPORT_FILE
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"results": rows}, fh, indent=2)
    print(f"[ok] report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
