"""Detect and explain a single transaction end to end.

Usage::

    PYTHONPATH=src python scripts/explain_transaction.py              # worst fraud
    PYTHONPATH=src python scripts/explain_transaction.py --index 42
    PYTHONPATH=src python scripts/explain_transaction.py --false-positive
    PYTHONPATH=src python scripts/explain_transaction.py --no-llm     # force fallback

Runs the whole pipeline against the real trained model and the tuned threshold:
preprocessing -> auto-encoder -> detection -> local LLM explanation. Useful as a
smoke test and as the live demo for the defense.
"""

from __future__ import annotations

import argparse

import numpy as np

from anomaly_explainer.config import ARTIFACTS_DIR, EXPLAINER
from anomaly_explainer.data.loader import load_raw
from anomaly_explainer.data.preprocess import prepare_data
from anomaly_explainer.detection.engine import AnomalyDetector
from anomaly_explainer.detection.threshold import load_threshold
from anomaly_explainer.explain.context import build_context
from anomaly_explainer.explain.explainer import explain, fallback_explanation
from anomaly_explainer.explain.ollama_client import OllamaClient
from anomaly_explainer.model.persistence import load_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, help="explain this row of the test set")
    group.add_argument(
        "--false-positive",
        action="store_true",
        help="explain the worst false positive (a normal transaction we flagged)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="skip Ollama, show the fallback"
    )
    return parser.parse_args()


def _pick_index(args, out, y_test) -> tuple[int, str]:
    """Choose which transaction to explain, and describe why."""
    if args.index is not None:
        if not 0 <= args.index < len(y_test):
            raise SystemExit(f"--index must be in [0, {len(y_test)})")
        return args.index, "requested"

    if args.false_positive:
        mask = out.is_anomaly & (y_test == 0)
        if not mask.any():
            raise SystemExit("no false positives to show")
        candidates = np.flatnonzero(mask)
        return int(candidates[np.argmax(out.severity[candidates])]), "worst false positive"

    mask = out.is_anomaly & (y_test == 1)
    if not mask.any():
        raise SystemExit("no detected frauds to show")
    candidates = np.flatnonzero(mask)
    return int(candidates[np.argmax(out.severity[candidates])]), "most severe true fraud"


def main() -> int:
    args = _parse_args()

    print("[..] loading model, threshold and data ...")
    model, scaler, _cfg = load_artifacts(ARTIFACTS_DIR)
    threshold = load_threshold(ARTIFACTS_DIR)
    prepared = prepare_data(load_raw())

    detector = AnomalyDetector(
        model=model,
        threshold=threshold.value,
        feature_names=prepared.feature_names,
    )
    out = detector.detect(prepared.X_test)
    index, why = _pick_index(args, out, prepared.y_test)

    truth = "FRAUD" if prepared.y_test[index] == 1 else "normal"
    print(
        f"[ok] threshold {threshold.value:.4f} "
        f"({threshold.method} @ {threshold.parameter})"
    )
    print(f"[ok] explaining test row {index} — {why}; ground truth: {truth}")

    context = build_context(
        out,
        index,
        x_scaled=prepared.X_test,
        scaler=scaler,
        top_k=EXPLAINER.top_k_attributes,
        transaction_id=index,
    )

    print("\n--- detection ---")
    print(f"  anomaly  : {context.is_anomaly}")
    print(f"  error    : {context.error:.4f} (threshold {context.threshold:.4f})")
    print(f"  severity : {context.severity:.2f}")
    print("  top attributes (observed vs expected):")
    for a in context.attributes:
        print(f"    {a.name:<8} {a.original:>12.2f} vs {a.reconstructed:>10.2f}"
              f"   (err {a.error:.3g})")

    if args.no_llm:
        result = fallback_explanation(context)
    else:
        client = OllamaClient()
        if not client.is_available():
            print(f"\n[!!] {EXPLAINER.model_name} unavailable — using fallback. "
                  f"Try: ollama pull {EXPLAINER.model_name}")
        print("\n[..] asking the local LLM ...")
        result = explain(context, client=client)

    print(f"\n--- explanation (source: {result.source}) ---")
    print(f"  risk level : {result.risk_level}")
    print(f"  attributes : {', '.join(result.attributes) or '-'}")
    print(f"\n  {result.interpretation}\n")
    print("  next steps:")
    for s in result.suggestions:
        print(f"    - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
