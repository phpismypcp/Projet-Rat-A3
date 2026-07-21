"""Tune the detection threshold and evaluate the detector.

Usage::

    PYTHONPATH=src python scripts/evaluate.py

Methodology (kept honest on purpose):

1. Sweep candidate thresholds on the **validation** split and pick an operating
   point (F1-optimal by default, or the highest-recall point meeting a precision
   floor via ``--min-precision``).
2. Persist that threshold to ``data/artifacts/threshold.json`` so the detector,
   the explainer and the UI all reuse the exact same value.
3. Report the chosen point **once** on the untouched **test** split. The test set
   is never used to choose anything, so its numbers are an unbiased estimate.

Also writes a JSON report and a precision-recall curve for the defense.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly_explainer.config import ARTIFACTS_DIR, DETECTION
from anomaly_explainer.data.loader import load_raw
from anomaly_explainer.data.preprocess import prepare_data
from anomaly_explainer.detection.reconstruction import reconstruction_errors
from anomaly_explainer.detection.threshold import Threshold, save_threshold
from anomaly_explainer.evaluation.metrics import evaluate
from anomaly_explainer.evaluation.sweep import DEFAULT_PERCENTILES, sweep_percentiles
from anomaly_explainer.model.persistence import load_artifacts

REPORT_FILE = "evaluation_report.json"
CURVE_FILE = "pr_curve.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Select the highest-recall threshold meeting this precision floor "
        "instead of the F1-optimal one (e.g. 0.85 to cap analyst workload).",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory holding model.pt / scaler.pkl / config.json.",
    )
    return parser.parse_args()


def _print_sweep_table(result) -> None:
    print(f"\n{'pctile':>7} {'threshold':>10} {'TP':>5} {'FP':>6} {'FN':>5} "
          f"{'prec':>6} {'recall':>7} {'F1':>6}")
    print("-" * 60)
    for p in result.points:
        m = p.metrics
        print(f"{p.percentile:>7} {p.threshold:>10.4f} {m.true_positives:>5} "
              f"{m.false_positives:>6} {m.false_negatives:>5} "
              f"{m.precision:>6.3f} {m.recall:>7.3f} {m.f1:>6.3f}")


def _write_pr_curve(y_true: np.ndarray, errors: np.ndarray, path: Path) -> bool:
    """Save a precision-recall curve. Returns False if plotting is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay
    except ImportError as exc:  # pragma: no cover - optional viz dependency
        print(f"[!!] skipping PR curve, matplotlib unavailable: {exc}")
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, errors, ax=ax)
    ax.set_title("Precision-Recall — reconstruction error as fraud score (test set)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    args = _parse_args()

    print("[..] loading model artifacts and dataset ...")
    model, _scaler, _cfg = load_artifacts(args.artifacts)
    prepared = prepare_data(load_raw())

    print("[..] computing reconstruction errors ...")
    val_errors = reconstruction_errors(model, prepared.X_val)
    test_errors = reconstruction_errors(model, prepared.X_test)
    val_normal_errors = val_errors[prepared.y_val == 0]

    print(
        f"[ok] normal error distribution (val): "
        f"mean={val_normal_errors.mean():.4f} median={np.median(val_normal_errors):.4f} "
        f"std={val_normal_errors.std():.4f}"
    )

    # --- 1. sweep on validation ------------------------------------------
    result = sweep_percentiles(
        val_normal_errors, prepared.y_val, val_errors, DEFAULT_PERCENTILES
    )
    _print_sweep_table(result)

    if args.min_precision is not None:
        chosen = result.best_at_min_precision(args.min_precision)
        criterion = f"max recall at precision >= {args.min_precision}"
    else:
        chosen = result.best_by_f1()
        criterion = "max F1"

    print(f"\n[ok] selected ({criterion}): percentile={chosen.percentile} "
          f"threshold={chosen.threshold:.4f}")

    # --- 2. persist the chosen threshold ---------------------------------
    threshold = Threshold(
        value=chosen.threshold, method="percentile", parameter=chosen.percentile
    )
    saved = save_threshold(threshold, args.artifacts)
    print(f"[ok] threshold persisted to {saved}")

    # --- 3. report once on the untouched test split ----------------------
    report = evaluate(prepared.y_test, test_errors, threshold.value)
    m, r = report.metrics, report.ranking

    print("\n=== TEST SET (held out, used only here) ===")
    print(f"  samples          : {report.n_samples:,} ({report.n_positives} frauds)")
    print(f"  threshold        : {report.threshold:.4f}")
    print(f"  TP / FP / FN     : {m.true_positives} / {m.false_positives} / "
          f"{m.false_negatives}")
    print(f"  precision        : {m.precision:.3f}")
    print(f"  recall           : {m.recall:.3f}")
    print(f"  F1               : {m.f1:.3f}")
    print(f"  PR-AUC           : {r.pr_auc:.4f}   (baseline "
          f"{report.n_positives / report.n_samples:.5f})")
    print(f"  ROC-AUC          : {r.roc_auc:.4f}")

    # Comparison against the old sigma default, to justify the config change.
    sigma_value = float(
        val_normal_errors.mean() + DETECTION.threshold_sigma_k * val_normal_errors.std()
    )
    sigma_report = evaluate(prepared.y_test, test_errors, sigma_value)
    print(f"\n  [for comparison] sigma k={DETECTION.threshold_sigma_k} "
          f"threshold={sigma_value:.4f} -> recall {sigma_report.metrics.recall:.3f}, "
          f"F1 {sigma_report.metrics.f1:.3f}")

    payload = {
        "selection_criterion": criterion,
        "threshold": {
            "value": threshold.value,
            "method": threshold.method,
            "parameter": threshold.parameter,
        },
        "validation_sweep": result.to_dict(),
        "test": report.to_dict(),
        "sigma_baseline": sigma_report.to_dict(),
    }
    report_path = Path(args.artifacts) / REPORT_FILE
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[ok] report written to {report_path}")

    curve_path = Path(args.artifacts) / CURVE_FILE
    if _write_pr_curve(prepared.y_test, test_errors, curve_path):
        print(f"[ok] PR curve written to {curve_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
