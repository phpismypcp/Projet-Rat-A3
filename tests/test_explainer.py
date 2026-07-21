"""Tests for the LLM explainer orchestration (Phase 5, TDD).

The explainer must never be the reason the dashboard breaks: whatever the LLM
returns — or fails to return — the analyst still gets a usable explanation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from anomaly_explainer.explain.context import AttributeDeviation, ExplanationContext
from anomaly_explainer.explain.explainer import (
    RISK_LEVELS,
    Explanation,
    explain,
    fallback_explanation,
)
from anomaly_explainer.explain.ollama_client import OllamaError, OllamaUnavailable


@pytest.fixture
def context() -> ExplanationContext:
    return ExplanationContext(
        severity=4.2,
        error=5.31,
        threshold=1.2634,
        is_anomaly=True,
        transaction_id=7,
        attributes=(
            AttributeDeviation("V14", -9.42, -0.51, 79.4),
            AttributeDeviation("Amount", 1809.68, 88.21, 12.1),
        ),
    )


def _client(text: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = text
    return client


VALID = json.dumps(
    {
        "attributes": ["V14", "Amount"],
        "interpretation": "Montant très supérieur à l'attendu.",
        "risk_level": "high",
        "suggestions": ["Contacter le porteur", "Vérifier l'historique"],
    }
)


# --- happy path -------------------------------------------------------------

def test_parses_a_valid_llm_response(context):
    result = explain(context, client=_client(VALID))

    assert isinstance(result, Explanation)
    assert result.source == "llm"
    assert result.attributes == ("V14", "Amount")
    assert result.risk_level == "high"
    assert len(result.suggestions) == 2


def test_risk_level_is_normalised(context):
    payload = json.loads(VALID) | {"risk_level": "  HIGH "}
    result = explain(context, client=_client(json.dumps(payload)))
    assert result.risk_level == "high"


def test_scalar_lists_are_coerced(context):
    """Small models often return a bare string where a list was requested."""
    payload = json.loads(VALID) | {"attributes": "V14", "suggestions": "Call client"}
    result = explain(context, client=_client(json.dumps(payload)))
    assert result.attributes == ("V14",)
    assert result.suggestions == ("Call client",)


# --- degradation ------------------------------------------------------------

def test_falls_back_when_ollama_is_unavailable(context):
    client = MagicMock()
    client.generate.side_effect = OllamaUnavailable("down")
    result = explain(context, client=client)
    assert result.source == "fallback"
    assert result.interpretation


def test_falls_back_on_any_ollama_error(context):
    client = MagicMock()
    client.generate.side_effect = OllamaError("timeout")
    assert explain(context, client=client).source == "fallback"


def test_falls_back_on_malformed_json(context):
    assert explain(context, client=_client("not json at all")).source == "fallback"


def test_falls_back_when_required_keys_are_missing(context):
    partial = json.dumps({"interpretation": "hmm"})
    assert explain(context, client=_client(partial)).source == "fallback"


def test_falls_back_when_response_is_a_json_list(context):
    assert explain(context, client=_client('["a", "b"]')).source == "fallback"


def test_invalid_risk_level_is_derived_from_severity(context):
    payload = json.loads(VALID) | {"risk_level": "catastrophic"}
    result = explain(context, client=_client(json.dumps(payload)))
    # severity 4.2 is well above threshold -> high
    assert result.risk_level == "high"
    assert result.source == "llm"  # the rest of the answer is still usable


# --- fallback quality -------------------------------------------------------

def test_fallback_names_the_responsible_attributes(context):
    result = fallback_explanation(context)
    assert result.source == "fallback"
    assert "V14" in result.attributes


def test_fallback_reports_real_numbers(context):
    """A fallback that says nothing concrete is worse than useless."""
    result = fallback_explanation(context)
    assert "1809.68" in result.interpretation or "1809.68" in " ".join(result.suggestions)


def test_fallback_risk_scales_with_severity(context):
    import dataclasses

    low = fallback_explanation(dataclasses.replace(context, severity=0.3, is_anomaly=False))
    high = fallback_explanation(dataclasses.replace(context, severity=9.0))
    assert low.risk_level == "low"
    assert high.risk_level == "high"
    assert RISK_LEVELS.index(high.risk_level) > RISK_LEVELS.index(low.risk_level)


def test_fallback_always_offers_next_steps(context):
    assert len(fallback_explanation(context).suggestions) >= 1


def test_medium_risk_between_the_bands(context):
    import dataclasses

    result = fallback_explanation(dataclasses.replace(context, severity=2.0))
    assert result.risk_level == "medium"


def test_fallback_in_english(context):
    import dataclasses

    from anomaly_explainer.config import EXPLAINER as cfg

    english = dataclasses.replace(cfg, language="en")
    result = fallback_explanation(context, english)
    assert "anomalous" in result.interpretation
    assert "1809.68" in result.interpretation  # the interpretable attribute
    assert len(result.suggestions) >= 1


def test_empty_interpretation_falls_back(context):
    payload = json.loads(VALID) | {"interpretation": "   "}
    assert explain(context, client=_client(json.dumps(payload))).source == "fallback"


def test_non_list_non_string_fields_become_empty(context):
    payload = json.loads(VALID) | {"attributes": 42, "suggestions": None}
    result = explain(context, client=_client(json.dumps(payload)))
    assert result.attributes == ()
    assert result.suggestions == ()


def test_fallback_without_attributes_still_works():
    """A context with no attributes must not crash the fallback."""
    bare = ExplanationContext(
        severity=0.5, error=0.1, threshold=0.2, is_anomaly=False, attributes=()
    )
    result = fallback_explanation(bare)
    assert result.interpretation
    assert result.attributes == ()


def test_explanation_serializes_to_dict(context):
    payload = explain(context, client=_client(VALID)).to_dict()
    assert json.dumps(payload)
    assert payload["risk_level"] == "high"
    assert payload["source"] == "llm"
