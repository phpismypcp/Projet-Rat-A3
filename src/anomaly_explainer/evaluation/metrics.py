"""Evaluation metrics for anomaly detection.

Why not accuracy
----------------
Fraud is ~0.17% of the data, so a model that predicts "normal" for everything
scores 99.83% accuracy while catching zero frauds. Accuracy is meaningless here.
What matters is the behaviour on the rare positive class:

* **precision** — of the transactions we flagged, how many were really fraud
  (drives analyst workload / false-positive cost);
* **recall** — of the real frauds, how many we caught (drives fraud losses);
* **F1** — their harmonic mean, a single number balancing the two;
* **PR-AUC** — threshold-independent quality on the rare class. This is the
  honest headline metric for imbalanced problems; ROC-AUC looks flatteringly
  high because true negatives dominate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels/scores at the boundary and return them as clean arrays.

    Raises:
        ValueError: on length mismatch, empty input, or non-binary labels.
    """
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=np.float64)

    if labels.shape[0] != values.shape[0]:
        raise ValueError(
            f"y_true and scores must have the same length "
            f"({labels.shape[0]} vs {values.shape[0]})"
        )
    if labels.size == 0:
        raise ValueError("y_true and scores must be non-empty")

    unique = np.unique(labels)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"y_true must contain only 0 or 1, found: {unique.tolist()}")

    return labels.astype(np.int64), values


@dataclass(frozen=True)
class ClassificationMetrics:
    """Confusion-matrix counts at one threshold, with derived rates.

    Precision/recall/F1 are computed on demand rather than stored, so the counts
    remain the single source of truth and cannot drift out of sync.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Share of flagged transactions that were really fraud. 0.0 if none flagged."""
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Share of real frauds that were caught. 0.0 if there are no frauds."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall. 0.0 when both are 0."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class RankingMetrics:
    """Threshold-independent metrics — how well the error *ranks* frauds first."""

    pr_auc: float
    roc_auc: float

    def to_dict(self) -> dict:
        return {"pr_auc": self.pr_auc, "roc_auc": self.roc_auc}


@dataclass(frozen=True)
class EvaluationReport:
    """Everything needed to judge one operating point."""

    threshold: float
    metrics: ClassificationMetrics
    ranking: RankingMetrics
    n_samples: int
    n_positives: int

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "metrics": self.metrics.to_dict(),
            "ranking": self.ranking.to_dict(),
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
        }


def classify(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Flag transactions whose error is strictly above ``threshold``.

    Strict ``>`` matches :class:`~anomaly_explainer.detection.engine.AnomalyDetector`
    so evaluation cannot disagree with the detector it is measuring.
    Does not mutate ``errors``.
    """
    return np.asarray(errors, dtype=np.float64) > threshold


def classification_metrics(
    y_true: np.ndarray, errors: np.ndarray, threshold: float
) -> ClassificationMetrics:
    """Confusion counts obtained by thresholding ``errors``."""
    labels, values = _validate(y_true, errors)
    predicted = classify(values, threshold)
    actual = labels == 1

    return ClassificationMetrics(
        true_positives=int((predicted & actual).sum()),
        false_positives=int((predicted & ~actual).sum()),
        true_negatives=int((~predicted & ~actual).sum()),
        false_negatives=int((~predicted & actual).sum()),
    )


def ranking_metrics(y_true: np.ndarray, errors: np.ndarray) -> RankingMetrics:
    """PR-AUC and ROC-AUC of the raw error as a fraud score.

    Raises:
        ValueError: if ``y_true`` does not contain both classes — the areas are
            undefined with a single class present.
    """
    labels, values = _validate(y_true, errors)
    if len(np.unique(labels)) < 2:
        raise ValueError("y_true must contain both classes to compute PR/ROC AUC")

    return RankingMetrics(
        pr_auc=float(average_precision_score(labels, values)),
        roc_auc=float(roc_auc_score(labels, values)),
    )


def evaluate(
    y_true: np.ndarray, errors: np.ndarray, threshold: float
) -> EvaluationReport:
    """Full report: confusion metrics at ``threshold`` plus ranking metrics."""
    labels, values = _validate(y_true, errors)
    return EvaluationReport(
        threshold=float(threshold),
        metrics=classification_metrics(labels, values, threshold),
        ranking=ranking_metrics(labels, values),
        n_samples=int(labels.size),
        n_positives=int(labels.sum()),
    )
