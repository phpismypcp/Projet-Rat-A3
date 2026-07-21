"""Tests for the dashboard service layer (Phase 6, TDD).

Streamlit code is awkward to test, so all the logic the dashboard needs lives
here instead and the UI file stays thin presentation. This is what makes the
interface verifiable rather than merely demo-able.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from anomaly_explainer.config import DATASET, MODEL
from anomaly_explainer.detection.engine import AnomalyDetector
from anomaly_explainer.detection.threshold import Threshold
from anomaly_explainer.model.transformer_ae import TransformerAutoEncoder
from anomaly_explainer.service import DetectionService

N_FEATURES = len(DATASET.feature_cols)


@pytest.fixture
def service() -> DetectionService:
    """A service over synthetic data with a known fraud/normal structure."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(60, N_FEATURES))
    fraud = rng.normal(9.0, 1.0, size=(10, N_FEATURES))  # far from normal
    x = np.vstack([normal, fraud]).astype(np.float32)
    y = np.concatenate([np.zeros(60, dtype=np.int64), np.ones(10, dtype=np.int64)])

    scaler = StandardScaler().fit(rng.normal(50.0, 10.0, size=(200, N_FEATURES)))

    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    model = TransformerAutoEncoder(cfg)
    model.eval()

    detector = AnomalyDetector(
        model=model, threshold=1.0, feature_names=DATASET.feature_cols
    )
    return DetectionService(
        x_scaled=x,
        labels=y,
        scaler=scaler,
        detector=detector,
        output=detector.detect(x),
        threshold=Threshold(value=1.0, method="percentile", parameter=99.9),
    )


# --- summary ----------------------------------------------------------------

def test_summary_counts_the_dataset(service):
    s = service.summary()
    assert s.n_transactions == 70
    assert s.n_frauds == 10
    assert s.threshold == 1.0


def test_summary_confusion_counts_are_consistent(service):
    s = service.summary()
    assert s.n_flagged == s.true_positives + s.false_positives
    assert s.n_frauds == s.true_positives + s.false_negatives
    assert 0.0 <= s.precision <= 1.0
    assert 0.0 <= s.recall <= 1.0


def test_summary_works_without_labels(service):
    """In production there are no labels — the dashboard must still work."""
    unlabelled = dataclasses.replace(service, labels=None)
    s = unlabelled.summary()
    assert s.n_transactions == 70
    assert s.n_frauds is None
    assert s.precision is None


# --- transaction table ------------------------------------------------------

def test_table_is_sorted_by_severity_descending(service):
    df = service.table(only_anomalies=False)
    assert list(df["severity"]) == sorted(df["severity"], reverse=True)


def test_table_can_filter_to_anomalies_only(service):
    df = service.table(only_anomalies=True)
    assert len(df) == int(service.output.is_anomaly.sum())
    assert df["flagged"].all()


def test_table_respects_the_limit(service):
    assert len(service.table(only_anomalies=False, limit=5)) == 5


def test_table_reports_amount_in_original_units(service):
    """A z-scored amount is useless to an analyst."""
    df = service.table(only_anomalies=False, limit=3)
    position = list(DATASET.feature_cols).index(DATASET.amount_col)
    for _, row in df.iterrows():
        expected = service.scaler.inverse_transform(
            service.x_scaled[int(row["index"])].reshape(1, -1)
        )[0][position]
        assert row["amount"] == pytest.approx(expected, rel=1e-4)


def test_table_includes_ground_truth_when_available(service):
    df = service.table(only_anomalies=False, limit=5)
    assert "actual" in df.columns


def test_table_omits_ground_truth_without_labels(service):
    unlabelled = dataclasses.replace(service, labels=None)
    assert "actual" not in unlabelled.table(only_anomalies=False, limit=5).columns


def test_table_index_column_addresses_the_original_row(service):
    df = service.table(only_anomalies=True, limit=1)
    index = int(df.iloc[0]["index"])
    assert service.output.severity[index] == pytest.approx(df.iloc[0]["severity"])


# --- per-transaction detail -------------------------------------------------

def test_context_matches_the_requested_row(service):
    ctx = service.context(3, top_k=5)
    assert ctx.transaction_id == 3
    assert len(ctx.attributes) == 5
    assert ctx.severity == pytest.approx(float(service.output.severity[3]))


def test_context_rejects_a_bad_index(service):
    with pytest.raises(IndexError):
        service.context(999, top_k=5)


def test_deviation_frame_pairs_observed_with_expected(service):
    """The chart needs observed vs expected side by side, in real units."""
    df = service.deviation_frame(3, top_k=4)
    assert len(df) == 4
    assert {"attribute", "observed", "expected", "error"} <= set(df.columns)
    assert list(df["error"]) == sorted(df["error"], reverse=True)


def test_gap_z_magnitude_is_the_root_of_the_error(service):
    """Raw units are not comparable across attributes; the scaled gap is."""
    df = service.deviation_frame(3, top_k=6)
    assert np.allclose(np.abs(df["gap_z"]), np.sqrt(df["error"]), atol=1e-5)


def test_gap_z_sign_says_higher_or_lower_than_expected(service):
    """Direction is what an analyst acts on, so it must survive the transform."""
    df = service.deviation_frame(3, top_k=6)
    expected_sign = np.sign(df["observed"] - df["expected"])
    assert np.array_equal(np.sign(df["gap_z"]), expected_sign)


def test_actual_label_lookup(service):
    assert service.actual_label(0) == 0
    assert service.actual_label(65) == 1
    assert dataclasses.replace(service, labels=None).actual_label(0) is None


# --- loading from disk (integration) ----------------------------------------

@pytest.fixture
def artifacts_on_disk(tmp_path):
    """Save a real (untrained) model, scaler and threshold, as the app expects."""
    from anomaly_explainer.detection.threshold import save_threshold
    from anomaly_explainer.model.persistence import save_artifacts

    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    model = TransformerAutoEncoder(cfg)
    scaler = StandardScaler().fit(
        np.random.default_rng(0).normal(size=(50, N_FEATURES))
    )
    save_artifacts(model, scaler, cfg, tmp_path)
    save_threshold(Threshold(value=0.9, method="percentile", parameter=99.9), tmp_path)
    return tmp_path


def test_load_service_builds_a_working_service(artifacts_on_disk, synthetic_csv):
    from anomaly_explainer.service import load_service

    service = load_service(artifacts_on_disk, synthetic_csv)
    summary = service.summary()

    assert summary.n_transactions > 0
    assert summary.threshold == 0.9
    assert service.table(only_anomalies=False, limit=3).shape[0] <= 3


def test_load_service_can_use_the_validation_split(artifacts_on_disk, synthetic_csv):
    from anomaly_explainer.service import load_service

    service = load_service(artifacts_on_disk, synthetic_csv, split="val")
    assert service.summary().n_transactions > 0


def test_load_service_rejects_an_unknown_split(artifacts_on_disk, synthetic_csv):
    from anomaly_explainer.service import load_service

    with pytest.raises(ValueError, match="split"):
        load_service(artifacts_on_disk, synthetic_csv, split="train")


def test_load_service_reports_missing_threshold(tmp_path, synthetic_csv):
    """Running the dashboard before evaluate.py must fail with a clear message."""
    from anomaly_explainer.model.persistence import save_artifacts
    from anomaly_explainer.service import load_service

    cfg = dataclasses.replace(MODEL, d_model=16, n_heads=2, latent_dim=8)
    save_artifacts(
        TransformerAutoEncoder(cfg),
        StandardScaler().fit(np.random.default_rng(0).normal(size=(50, N_FEATURES))),
        cfg,
        tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="threshold.json"):
        load_service(tmp_path, synthetic_csv)
