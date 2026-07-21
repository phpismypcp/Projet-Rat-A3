"""Composition root for the analyst dashboard.

Everything the UI needs to *know* lives here; the Streamlit file only decides how
to *draw* it. Two reasons: Streamlit code is awkward to unit-test, and keeping
the logic here means the dashboard is verified by the same suite as the rest of
the project rather than by clicking around.

The service deliberately tolerates ``labels is None``. Ground truth exists only
because this is a benchmark dataset — a real deployment has no ``Class`` column,
and an interface that collapses without it would not be reusable in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from anomaly_explainer.config import ARTIFACTS_DIR, DATASET, EXPLAINER
from anomaly_explainer.data.loader import load_raw
from anomaly_explainer.data.preprocess import prepare_data
from anomaly_explainer.detection.engine import AnomalyDetector, DetectionOutput
from anomaly_explainer.detection.threshold import Threshold, load_threshold
from anomaly_explainer.explain.context import ExplanationContext, build_context
from anomaly_explainer.model.persistence import load_artifacts


@dataclass(frozen=True)
class ServiceSummary:
    """Headline numbers for the dashboard KPI row.

    Label-dependent fields are ``None`` when no ground truth is available.
    """

    n_transactions: int
    n_flagged: int
    threshold: float
    n_frauds: int | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


@dataclass(frozen=True)
class DetectionService:
    """Detection results plus the lookups the dashboard performs on them."""

    x_scaled: np.ndarray
    scaler: StandardScaler
    detector: AnomalyDetector
    output: DetectionOutput
    threshold: Threshold
    labels: np.ndarray | None = None

    # --- aggregate ----------------------------------------------------------

    def summary(self) -> ServiceSummary:
        """Counts and, when labels exist, the quality of the current threshold."""
        flagged = self.output.is_anomaly
        base = ServiceSummary(
            n_transactions=int(flagged.size),
            n_flagged=int(flagged.sum()),
            threshold=float(self.threshold.value),
        )
        if self.labels is None:
            return base

        # Reuse the evaluation module so the dashboard cannot disagree with the
        # numbers reported in the written evaluation.
        from anomaly_explainer.evaluation.metrics import classification_metrics

        m = classification_metrics(self.labels, self.output.errors, self.threshold.value)
        return ServiceSummary(
            n_transactions=base.n_transactions,
            n_flagged=base.n_flagged,
            threshold=base.threshold,
            n_frauds=int((self.labels == 1).sum()),
            true_positives=m.true_positives,
            false_positives=m.false_positives,
            false_negatives=m.false_negatives,
            precision=m.precision,
            recall=m.recall,
            f1=m.f1,
        )

    # --- listing ------------------------------------------------------------

    def _original_units(self) -> np.ndarray:
        """The whole matrix back in real units (Amount in currency, not z-scores)."""
        return self.scaler.inverse_transform(
            np.asarray(self.x_scaled, dtype=np.float64)
        )

    def table(self, *, only_anomalies: bool = True, limit: int = 200) -> pd.DataFrame:
        """Transactions ranked by severity, most suspicious first.

        Ranking by severity is the point of the queue: it puts the transactions
        most worth an analyst's attention at the top.
        """
        order = np.argsort(self.output.severity)[::-1]
        if only_anomalies:
            order = order[self.output.is_anomaly[order]]
        order = order[:limit]

        originals = self._original_units()
        cols = list(DATASET.feature_cols)
        amount_at = cols.index(DATASET.amount_col)
        time_at = cols.index(DATASET.time_col)

        data = {
            "index": order.astype(int),
            "severity": self.output.severity[order],
            "error": self.output.errors[order],
            "amount": originals[order, amount_at],
            "time": originals[order, time_at],
            "flagged": self.output.is_anomaly[order],
        }
        if self.labels is not None:
            data["actual"] = self.labels[order]

        return pd.DataFrame(data).reset_index(drop=True)

    # --- per transaction ----------------------------------------------------

    def context(
        self, index: int, top_k: int = EXPLAINER.top_k_attributes
    ) -> ExplanationContext:
        """Evidence bundle for one transaction, ready for the explainer."""
        return build_context(
            self.output,
            index,
            x_scaled=self.x_scaled,
            scaler=self.scaler,
            top_k=top_k,
            transaction_id=index,
        )

    def deviation_frame(
        self, index: int, top_k: int = EXPLAINER.top_k_attributes
    ) -> pd.DataFrame:
        """Observed vs expected for the worst-reconstructed attributes.

        This is the chart that carries the explanation: the analyst sees *where*
        the transaction departed from what the model considered normal.
        """
        ctx = self.context(index, top_k=top_k)
        observed = np.array([a.original for a in ctx.attributes])
        expected = np.array([a.reconstructed for a in ctx.attributes])
        errors = np.array([a.error for a in ctx.attributes])

        # Signed deviation in standard deviations. Raw units cannot be compared
        # across attributes (Amount is in currency, V17 is a PCA component), but
        # the scaled gap can: error is the squared scaled gap, so its square root
        # is the magnitude and the raw difference supplies the direction —
        # "higher than expected" vs "lower" is what an analyst acts on.
        gap_z = np.sign(observed - expected) * np.sqrt(errors)

        return pd.DataFrame(
            {
                "attribute": [a.name for a in ctx.attributes],
                "observed": observed,
                "expected": expected,
                "error": errors,
                "gap_z": gap_z,
            }
        )

    def actual_label(self, index: int) -> int | None:
        """Ground truth for a row, or ``None`` when unlabelled."""
        return None if self.labels is None else int(self.labels[index])


def load_service(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    csv_path: str | Path | None = None,
    *,
    split: str = "test",
) -> DetectionService:
    """Assemble a :class:`DetectionService` from saved artifacts and the dataset.

    Loads the trained model, the fitted scaler and the **tuned** threshold, then
    scores the requested split. The threshold is read from disk rather than
    recomputed so the dashboard shows exactly the operating point that was
    evaluated.

    Args:
        split: ``"test"`` (default, the held-out split) or ``"val"``.

    Raises:
        FileNotFoundError: if the model or threshold artifacts are missing.
        ValueError: if ``split`` is unknown.
    """
    if split not in ("test", "val"):
        raise ValueError(f"unknown split {split!r}; expected 'test' or 'val'")

    model, scaler, _config = load_artifacts(artifacts_dir)
    threshold = load_threshold(artifacts_dir)
    prepared = prepare_data(load_raw(csv_path))

    x = prepared.X_test if split == "test" else prepared.X_val
    y = prepared.y_test if split == "test" else prepared.y_val

    detector = AnomalyDetector(
        model=model, threshold=threshold.value, feature_names=prepared.feature_names
    )
    return DetectionService(
        x_scaled=x,
        scaler=scaler,
        detector=detector,
        output=detector.detect(x),
        threshold=threshold,
        labels=y,
    )
