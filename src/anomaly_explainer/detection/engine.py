"""Detection engine: turn reconstruction errors into decisions.

Given a trained model and a fitted threshold, classify each transaction as
normal or anomalous, attach a **severity score** (how far above the threshold),
and expose which attributes contributed most to the error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from anomaly_explainer.detection.reconstruction import per_feature_errors


@dataclass(frozen=True)
class DetectionOutput:
    """Batch detection result.

    Attributes:
        errors: total reconstruction error per sample, shape ``(n,)``.
        per_feature_errors: squared error per feature, shape ``(n, n_features)``.
        is_anomaly: boolean flag per sample (``error > threshold``).
        severity: ``error / threshold`` — ``> 1`` exactly when anomalous.
        threshold: the threshold value used.
        feature_names: feature order matching ``per_feature_errors`` columns.
    """

    errors: np.ndarray
    per_feature_errors: np.ndarray
    is_anomaly: np.ndarray
    severity: np.ndarray
    threshold: float
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class AnomalyDetector:
    """Wraps a trained model and a threshold to score transactions."""

    model: nn.Module
    threshold: float
    feature_names: tuple[str, ...]
    device: torch.device | None = None
    batch_size: int = 4096

    def detect(self, x: np.ndarray) -> DetectionOutput:
        """Score scaled feature matrix ``x`` (shape ``(n, n_features)``)."""
        pfe = per_feature_errors(
            self.model, x, device=self.device, batch_size=self.batch_size
        )
        errors = pfe.mean(axis=1).astype(np.float32)
        is_anomaly = errors > self.threshold
        # Guard against a zero threshold when computing the ratio.
        denom = self.threshold if self.threshold != 0 else np.finfo(np.float32).tiny
        severity = (errors / denom).astype(np.float32)

        return DetectionOutput(
            errors=errors,
            per_feature_errors=pfe,
            is_anomaly=is_anomaly,
            severity=severity,
            threshold=self.threshold,
            feature_names=self.feature_names,
        )


def top_contributors(
    output: DetectionOutput, index: int, k: int
) -> list[tuple[str, float]]:
    """Return the ``k`` features with the largest error for sample ``index``.

    Sorted by error descending. Used by the LLM explainer to name the attributes
    most responsible for an anomaly.
    """
    row = output.per_feature_errors[index]
    order = np.argsort(row)[::-1][:k]
    return [(output.feature_names[i], float(row[i])) for i in order]
