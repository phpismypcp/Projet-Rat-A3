"""Does attention actually help? Transformer auto-encoder vs a plain one.

Usage::

    PYTHONPATH=src python scripts/compare_baseline.py

The spec mandates a Transformer auto-encoder on the grounds that attention
captures interactions between features. That is a claim, and this script is its
control: it trains plain fully-connected auto-encoders and scores them with the
**identical** procedure used for the Transformer.

Held constant across every model: the data splits, the seed, the training loop
(optimizer, batching, early stopping), the latent size, and the thresholding
rule (99.9th percentile of normal validation errors). The only thing that varies
is the architecture — so a difference in score is attributable to it.

Two baselines are trained, because "the Transformer won" is a weak claim if it
simply had more capacity:

* **matched bottleneck** — same latent size, modest width (far fewer parameters);
* **matched capacity** — widened until its parameter count is comparable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from anomaly_explainer.config import (
    ARTIFACTS_DIR,
    DENSE_MODEL,
    DETECTION,
    TRAIN,
)
from anomaly_explainer.data.loader import load_raw
from anomaly_explainer.data.preprocess import PreparedData, prepare_data
from anomaly_explainer.detection.reconstruction import reconstruction_errors
from anomaly_explainer.detection.threshold import percentile_threshold
from anomaly_explainer.evaluation.metrics import evaluate
from anomaly_explainer.model.dense_ae import DenseAutoEncoder, count_parameters
from anomaly_explainer.model.persistence import load_artifacts
from anomaly_explainer.model.train import train_autoencoder

REPORT_FILE = "baseline_comparison.json"


def _score(model, prepared: PreparedData, label: str) -> dict:
    """Threshold on normal validation errors, then evaluate on the test split."""
    val_errors = reconstruction_errors(model, prepared.X_val)
    test_errors = reconstruction_errors(model, prepared.X_test)

    threshold = percentile_threshold(
        val_errors[prepared.y_val == 0], DETECTION.threshold_percentile
    )
    report = evaluate(prepared.y_test, test_errors, threshold)

    return {
        "model": label,
        "parameters": count_parameters(model),
        "threshold": threshold,
        **report.to_dict(),
    }


def _train_dense(
    prepared: PreparedData, hidden: tuple[int, ...], label: str, epochs: int
) -> dict:
    config = dataclasses.replace(DENSE_MODEL, hidden_dims=hidden)
    train_config = dataclasses.replace(TRAIN, epochs=epochs)
    print(f"\n[..] training dense baseline {label} hidden={hidden} "
          f"(max {epochs} epochs) ...")
    start = time.perf_counter()

    def log(epoch: int, train_loss: float, val_loss: float) -> None:
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}",
                  flush=True)

    model, history = train_autoencoder(
        prepared,
        train_config=train_config,
        on_epoch_end=log,
        model_factory=lambda: DenseAutoEncoder(config),
    )
    elapsed = time.perf_counter() - start
    n_epochs = len(history["train_loss"])
    converged = n_epochs < epochs  # early stopping fired, so it plateaued
    print(f"[ok] {label}: {n_epochs} epochs in {elapsed:.0f}s "
          f"({'converged' if converged else 'HIT EPOCH CAP — may be under-trained'})")

    result = _score(model, prepared, label)
    result["train_seconds"] = round(elapsed, 1)
    result["epochs"] = n_epochs
    result["converged"] = converged
    result["final_val_loss"] = history["val_loss"][-1]
    return result


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'model':<28} {'params':>9} {'PR-AUC':>8} {'ROC-AUC':>8} "
          f"{'prec':>6} {'recall':>7} {'F1':>6} {'FP':>5}")
    print("-" * 84)
    for r in rows:
        m = r["metrics"]
        print(f"{r['model']:<28} {r['parameters']:>9,} {r['ranking']['pr_auc']:>8.4f} "
              f"{r['ranking']['roc_auc']:>8.4f} {m['precision']:>6.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>6.3f} {m['false_positives']:>5}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAIN.epochs,
        help="epoch cap for the baselines; raise it so early stopping, not the "
        "cap, ends training — otherwise a baseline can look weak merely because "
        "it was cut off.",
    )
    args = parser.parse_args()

    print("[..] loading data and the trained Transformer ...")
    prepared = prepare_data(load_raw())
    transformer, _scaler, _cfg = load_artifacts(ARTIFACTS_DIR)

    rows = [_score(transformer, prepared, "Transformer AE (spec)")]
    print(f"[ok] transformer: {rows[0]['parameters']:,} parameters")

    rows.append(
        _train_dense(prepared, (64, 32), "Dense AE (matched bottleneck)", args.epochs)
    )
    rows.append(
        _train_dense(prepared, (512, 256), "Dense AE (matched capacity)", args.epochs)
    )

    _print_table(rows)

    best = max(rows, key=lambda r: r["ranking"]["pr_auc"])
    transformer_row = rows[0]
    delta = transformer_row["ranking"]["pr_auc"] - max(
        r["ranking"]["pr_auc"] for r in rows[1:]
    )
    print(f"\nBest PR-AUC: {best['model']}")
    print(f"Transformer minus best dense baseline: {delta:+.4f} PR-AUC")
    if delta > 0:
        print("=> attention helps on this dataset.")
    else:
        print("=> attention does NOT help here; the plain auto-encoder matches or wins.")

    out = Path(ARTIFACTS_DIR) / REPORT_FILE
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"results": rows}, fh, indent=2)
    print(f"\n[ok] report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
