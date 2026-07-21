"""Tests for the Ollama HTTP client (Phase 5, TDD).

All network calls are mocked — the suite must never depend on a running Ollama.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from anomaly_explainer.config import EXPLAINER
from anomaly_explainer.explain.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
)

MODULE = "anomaly_explainer.explain.ollama_client.requests"


def _response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_generate_returns_the_model_text():
    with patch(MODULE) as req:
        req.post.return_value = _response({"response": "hello"})
        assert OllamaClient().generate("prompt") == "hello"


def test_generate_posts_the_configured_payload():
    with patch(MODULE) as req:
        req.post.return_value = _response({"response": "ok"})
        OllamaClient().generate("my prompt")

        url = req.post.call_args.args[0]
        body = req.post.call_args.kwargs["json"]
        assert url == f"{EXPLAINER.ollama_host}/api/generate"
        assert body["model"] == EXPLAINER.model_name
        assert body["prompt"] == "my prompt"
        assert body["stream"] is False
        assert body["keep_alive"] == EXPLAINER.keep_alive
        assert body["format"] == "json"
        assert body["options"]["temperature"] == EXPLAINER.temperature
        assert req.post.call_args.kwargs["timeout"] == EXPLAINER.request_timeout_s


def test_connection_error_becomes_ollama_unavailable():
    with patch(MODULE) as req:
        req.exceptions = requests.exceptions
        req.post.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(OllamaUnavailable, match="ollama"):
            OllamaClient().generate("p")


def test_timeout_is_reported_as_a_timeout():
    with patch(MODULE) as req:
        req.exceptions = requests.exceptions
        req.post.side_effect = requests.exceptions.Timeout("too slow")
        with pytest.raises(OllamaError, match="timed out"):
            OllamaClient().generate("p")


def test_http_error_is_wrapped():
    with patch(MODULE) as req:
        req.exceptions = requests.exceptions
        bad = _response({}, status=500)
        bad.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        req.post.return_value = bad
        with pytest.raises(OllamaError):
            OllamaClient().generate("p")


def test_missing_response_field_is_an_error():
    with patch(MODULE) as req:
        req.exceptions = requests.exceptions
        req.post.return_value = _response({"unexpected": "shape"})
        with pytest.raises(OllamaError, match="response"):
            OllamaClient().generate("p")


def test_non_json_body_is_wrapped():
    with patch(MODULE) as req:
        req.exceptions = requests.exceptions
        bad = _response({})
        bad.json.side_effect = ValueError("no json")
        req.post.return_value = bad
        with pytest.raises(OllamaError, match="invalid JSON"):
            OllamaClient().generate("p")


def test_is_available_true_when_model_is_installed():
    with patch(MODULE) as req:
        req.get.return_value = _response(
            {"models": [{"name": EXPLAINER.model_name}]}
        )
        assert OllamaClient().is_available() is True


def test_is_available_false_when_model_missing():
    """The exact failure we hit in phase 0: server up, no model pulled."""
    with patch(MODULE) as req:
        req.get.return_value = _response({"models": []})
        assert OllamaClient().is_available() is False


def test_is_available_false_when_server_is_down():
    with patch(MODULE) as req:
        req.exceptions = requests.exceptions
        req.get.side_effect = requests.exceptions.ConnectionError("refused")
        assert OllamaClient().is_available() is False
