"""Headless smoke test for the Streamlit dashboard (Phase 6).

Why this exists
---------------
The dashboard once "worked" — the server answered HTTP 200 and reported healthy
— while the script itself crashed on a bad import. Streamlit renders
client-side, so a healthy port proves only that the server booted, not that the
app runs. This test executes the script and fails on any uncaught exception.

It needs the real artifacts and the 150 MB CSV, so it is opt-in rather than part
of the fast feedback loop::

    RUN_APP_TEST=1 PYTHONPATH=src pytest tests/test_streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anomaly_explainer.config import ARTIFACTS_DIR, RAW_CSV

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_APP_TEST") != "1",
        reason="opt-in: set RUN_APP_TEST=1 (loads the full dataset, ~1 min)",
    ),
    pytest.mark.skipif(
        not (ARTIFACTS_DIR / "threshold.json").exists() or not RAW_CSV.exists(),
        reason="requires trained artifacts and the dataset",
    ),
]


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_app_renders_its_title(app):
    assert any("fraudes bancaires" in t.value for t in app.title)


def test_app_shows_the_detection_kpis(app):
    labels = [m.label for m in app.metric]
    for expected in ("Transactions", "Signalées", "Précision", "Rappel"):
        assert expected in labels


def test_app_preselects_a_transaction_and_explains_it(app):
    """The detail panel must render for the top-ranked transaction."""
    assert any("Transaction #" in s.value for s in app.subheader)
    assert any("risque" in m.value for m in app.markdown)


def test_app_renders_the_queue_and_the_detail_table(app):
    assert len(app.dataframe) >= 1
