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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anomaly_explainer.config import DETECTION, DetectionConfig

THRESHOLD_FILE = "threshold.json"


@dataclass(frozen=True)
class Threshold:
    """A fitted threshold value plus how it was computed.

    ``parameter`` records the input that produced ``value`` (``k`` for sigma, the
    percentile for percentile), so a saved threshold is reproducible rather than
    an unexplained number.
    """

    value: float
    method: str
    parameter: float | None = None


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
    method: str | None = None,
) -> Threshold:
    """Fit a :class:`Threshold` on normal reconstruction errors.

    Args:
        normal_errors: reconstruction errors for known-normal transactions.
        method: ``"sigma"`` or ``"percentile"``; defaults to
            ``config.threshold_method`` so the choice lives in one place.

    Raises:
        ValueError: if ``normal_errors`` is empty or ``method`` is unknown.
    """
    errors = np.asarray(normal_errors, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("normal_errors must be non-empty to fit a threshold")

    method = method if method is not None else config.threshold_method

    if method == "sigma":
        parameter = config.threshold_sigma_k
        value = sigma_threshold(errors, parameter)
    elif method == "percentile":
        parameter = config.threshold_percentile
        value = percentile_threshold(errors, parameter)
    else:
        raise ValueError(f"unknown threshold method: {method!r} (use sigma|percentile)")

    return Threshold(value=value, method=method, parameter=parameter)


# --- persistence ------------------------------------------------------------
# The tuned threshold is an artifact in its own right: the UI and the explainer
# must reuse the exact value chosen during evaluation, never re-derive their own.

def save_threshold(threshold: Threshold, directory: str | Path) -> Path:
    """Write ``threshold`` to ``directory/threshold.json``, creating the dir."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "value": threshold.value,
        "method": threshold.method,
        "parameter": threshold.parameter,
    }
    with open(out / THRESHOLD_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out / THRESHOLD_FILE


def load_threshold(directory: str | Path) -> Threshold:
    """Read a :class:`Threshold` back from ``directory``.

    Raises:
        FileNotFoundError: if ``threshold.json`` is absent — meaning evaluation
            has not been run yet.
    """
    path = Path(directory) / THRESHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {THRESHOLD_FILE}: {path}. Run scripts/evaluate.py to tune "
            "and persist the detection threshold."
        )

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    return Threshold(
        value=float(payload["value"]),
        method=str(payload["method"]),
        parameter=payload.get("parameter"),
    )
