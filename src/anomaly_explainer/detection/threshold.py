"""Statistical threshold on reconstruction error.

A transaction is anomalous when its reconstruction error exceeds a threshold
computed from the distribution of errors on **normal** data. Two strategies:

- ``sigma``: ``mean + k * std`` — flags errors that are k standard deviations
  above the normal average.
- ``percentile``: the p-th percentile of normal errors — flags the top
  ``100 - p``% most poorly reconstructed transactions.

Tuning this threshold is the main lever for trading off false positives vs
missed frauds (see the evaluation phase).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anomaly_explainer.config import DETECTION, DetectionConfig


@dataclass(frozen=True)
class Threshold:
    """A fitted threshold value plus the method used to compute it."""

    value: float
    method: str


def sigma_threshold(normal_errors: np.ndarray, k: float) -> float:
    """``mean + k * std`` of the normal error distribution."""
    errors = np.asarray(normal_errors, dtype=np.float64)
    return float(errors.mean() + k * errors.std())


def percentile_threshold(normal_errors: np.ndarray, percentile: float) -> float:
    """The ``percentile``-th percentile of the normal error distribution."""
    errors = np.asarray(normal_errors, dtype=np.float64)
    return float(np.percentile(errors, percentile))


def fit_threshold(
    normal_errors: np.ndarray,
    config: DetectionConfig = DETECTION,
    method: str = "sigma",
) -> Threshold:
    """Fit a :class:`Threshold` on normal reconstruction errors.

    Args:
        normal_errors: reconstruction errors for known-normal transactions.
        method: ``"sigma"`` or ``"percentile"``.

    Raises:
        ValueError: if ``normal_errors`` is empty or ``method`` is unknown.
    """
    errors = np.asarray(normal_errors, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("normal_errors must be non-empty to fit a threshold")

    if method == "sigma":
        value = sigma_threshold(errors, config.threshold_sigma_k)
    elif method == "percentile":
        value = percentile_threshold(errors, config.threshold_percentile)
    else:
        raise ValueError(f"unknown threshold method: {method!r} (use sigma|percentile)")

    return Threshold(value=value, method=method)
