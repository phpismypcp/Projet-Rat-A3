"""Thin HTTP client for a local Ollama server.

Everything runs on ``localhost``: no transaction data ever leaves the machine,
which is the whole reason the spec chose a local LLM over a cloud API.

The client's job is to fail *clearly*. A dead server, a missing model and a slow
generation are three different problems with three different fixes, so they get
three different messages instead of one opaque stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from anomaly_explainer.config import EXPLAINER, ExplainerConfig


class OllamaError(RuntimeError):
    """Any failure while talking to Ollama."""


class OllamaUnavailable(OllamaError):
    """The server could not be reached at all."""


@dataclass(frozen=True)
class OllamaClient:
    """Calls the Ollama generate API with the configured model."""

    config: ExplainerConfig = EXPLAINER

    def is_available(self) -> bool:
        """True if the server responds *and* the configured model is installed.

        Both halves matter: during Phase 0 the server was running happily while
        ``/api/tags`` was empty, which would have failed only at generation time.
        """
        try:
            resp = requests.get(f"{self.config.ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            installed = {m.get("name", "") for m in resp.json().get("models", [])}
        except (requests.exceptions.RequestException, ValueError):
            return False

        wanted = self.config.model_name
        base = wanted.split(":")[0]
        return any(name == wanted or name.startswith(f"{base}:") for name in installed)

    def generate(self, prompt: str) -> str:
        """Send ``prompt`` to the model and return the raw text response.

        Raises:
            OllamaUnavailable: the server could not be reached.
            OllamaError: timeout, HTTP error, or an unexpected response shape.
        """
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {"temperature": self.config.temperature},
        }
        if self.config.json_format:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.config.ollama_host}/api/generate",
                json=payload,
                timeout=self.config.request_timeout_s,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailable(
                f"cannot reach ollama at {self.config.ollama_host} — is the service "
                f"running? (`systemctl status ollama`)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"ollama timed out after {self.config.request_timeout_s}s; a cold "
                f"model load on CPU can be slow, try raising request_timeout_s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaError(f"ollama returned invalid JSON: {exc}") from exc

        if "response" not in body:
            raise OllamaError(
                f"ollama reply has no 'response' field (keys: {sorted(body)}); "
                f"is {self.config.model_name!r} installed? (`ollama list`)"
            )
        return body["response"]
