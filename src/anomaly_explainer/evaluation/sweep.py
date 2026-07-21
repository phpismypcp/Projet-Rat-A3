"""Threshold sweep: choose the operating point on evidence.

The threshold is the single most consequential knob in the system, and it cannot
be picked by assumption. On this dataset the reconstruction-error distribution of
normal transactions is heavily right-skewed (mean is ~6x the median), so the
textbook ``mean + 3*sigma`` rule lands far out in the tail and catches almost
nothing. Sweeping percentiles of the *normal* error distribution and reading off
the precision/recall trade-off is the reliable way to choose.

The sweep is always fit on the **validation** split; the chosen point is then
reported once on the untouched **test** split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from anomaly_explainer.evaluation.metrics import (
    ClassificationMetrics,
    classification_metrics,
)

# Fraud is ~0.17% of transactions, so anything below the 95th percentile flags
# far too much to be useful. The interesting trade-offs live in the far tail.
DEFAULT_PERCENTILES: tuple[float, ...] = (
    95.0, 98.0, 99.0, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95,
)


@dataclass(frozen=True)
class SweepPoint:
    """One candidate operating point."""

    percentile: float
    threshold: float
    metrics: ClassificationMetrics

    def to_dict(self) -> dict:
        return {
            "percentile": self.percentile,
            "threshold": self.threshold,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class SweepResult:
    """All candidate points, with selectors for choosing between them."""

    points: tuple[SweepPoint, ...]

    def best_by_f1(self) -> SweepPoint:
        """The point with the highest F1 — the balanced default choice."""
        return max(self.points, key=lambda p: p.metrics.f1)

    def best_at_min_precision(self, min_precision: float) -> SweepPoint:
        """Highest-recall point whose precision meets ``min_precision``.

        Use this when analyst review capacity, not F1, sets the budget: it
        catches as much fraud as possible while keeping false alarms tolerable.

        Raises:
            ValueError: if no candidate reaches ``min_precision``.
        """
        eligible = [p for p in self.points if p.metrics.precision >= min_precision]
        if not eligible:
            raise ValueError(
                f"no threshold in the sweep reaches precision >= {min_precision}; "
                f"best available is {max(p.metrics.precision for p in self.points):.3f}"
            )
        return max(eligible, key=lambda p: p.metrics.recall)

    def to_dict(self) -> dict:
        return {"points": [p.to_dict() for p in self.points]}


def _validate_percentiles(percentiles: Sequence[float]) -> tuple[float, ...]:
    if len(percentiles) == 0:
        raise ValueError("percentiles must be non-empty")
    for p in percentiles:
        if not 0 < p < 100:
            raise ValueError(f"percentiles must be between 0 and 100, got {p}")
    return tuple(float(p) for p in percentiles)


def sweep_percentiles(
    normal_errors: np.ndarray,
    y_true: np.ndarray,
    errors: np.ndarray,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> SweepResult:
    """Evaluate every percentile threshold of the normal error distribution.

    Args:
        normal_errors: errors of known-normal rows — defines the thresholds.
        y_true: true labels of the evaluation set (1 = fraud).
        errors: reconstruction errors of the evaluation set.
        percentiles: candidate percentiles, ascending.

    Returns:
        A :class:`SweepResult` with one point per percentile, in input order.
    """
    candidates = _validate_percentiles(percentiles)
    baseline = np.asarray(normal_errors, dtype=np.float64)
    if baseline.size == 0:
        raise ValueError("normal_errors must be non-empty to derive thresholds")

    points = tuple(
        SweepPoint(
            percentile=p,
            threshold=float(np.percentile(baseline, p)),
            metrics=classification_metrics(
                y_true, errors, float(np.percentile(baseline, p))
            ),
        )
        for p in candidates
    )
    return SweepResult(points=points)
