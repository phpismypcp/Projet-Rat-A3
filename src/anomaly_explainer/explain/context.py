"""Build the evidence bundle handed to the LLM for one transaction.

The project spec lists exactly what the LLM must be shown for a flagged
transaction: the original values, the model's reconstruction, the per-attribute
error, and the severity score. This module assembles those four things for a
single row and — importantly — **inverse-transforms the values back to real
units**.

That last step matters more than it looks. The detector works on standardized
features, so a raw value reaching the LLM would read ``Amount = 4.7`` (a
z-score). An analyst needs ``Amount = 1809.68``. Explaining a transaction in
units nobody can act on would defeat the purpose of the explainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from anomaly_explainer.detection.engine import DetectionOutput


@dataclass(frozen=True)
class AttributeDeviation:
    """One feature's contribution to an anomaly, in original units."""

    name: str
    original: float
    reconstructed: float
    error: float

    @property
    def gap(self) -> float:
        """Signed difference the analyst actually reads: observed - expected."""
        return self.original - self.reconstructed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "original": self.original,
            "reconstructed": self.reconstructed,
            "error": self.error,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class ExplanationContext:
    """Everything the LLM needs to explain one transaction."""

    severity: float
    error: float
    threshold: float
    is_anomaly: bool
    attributes: tuple[AttributeDeviation, ...]
    transaction_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "severity": self.severity,
            "error": self.error,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "attributes": [a.to_dict() for a in self.attributes],
        }


def build_context(
    output: DetectionOutput,
    index: int,
    *,
    x_scaled: np.ndarray,
    scaler: StandardScaler,
    top_k: int,
    transaction_id: int | None = None,
) -> ExplanationContext:
    """Assemble the context for row ``index`` of a detection result.

    Args:
        output: the batch detection result.
        index: which row to explain.
        x_scaled: the scaled matrix that was passed to ``detect``.
        scaler: the fitted scaler, used to return values in original units.
        top_k: how many of the most-deviant attributes to include; clamped to
            the number of features.
        transaction_id: optional identifier to carry into the explanation.

    Raises:
        IndexError: if ``index`` is outside ``output``.
        ValueError: if ``top_k`` is not positive.
    """
    n_rows = output.per_feature_errors.shape[0]
    if not 0 <= index < n_rows:
        raise IndexError(f"index {index} out of range for {n_rows} detected rows")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    names = output.feature_names
    k = min(top_k, len(names))

    # Back to real units: inverse_transform expects a 2-D batch, so reshape.
    original = scaler.inverse_transform(
        np.asarray(x_scaled[index], dtype=np.float64).reshape(1, -1)
    )[0]
    rebuilt = scaler.inverse_transform(
        np.asarray(output.reconstructions[index], dtype=np.float64).reshape(1, -1)
    )[0]

    row_errors = output.per_feature_errors[index]
    ranked = np.argsort(row_errors)[::-1][:k]

    attributes = tuple(
        AttributeDeviation(
            name=names[pos],
            original=float(original[pos]),
            reconstructed=float(rebuilt[pos]),
            error=float(row_errors[pos]),
        )
        for pos in ranked
    )

    return ExplanationContext(
        severity=float(output.severity[index]),
        error=float(output.errors[index]),
        threshold=float(output.threshold),
        is_anomaly=bool(output.is_anomaly[index]),
        attributes=attributes,
        transaction_id=transaction_id,
    )
