"""Tests for LLM prompt construction (Phase 5, TDD).

The prompt must carry the four inputs the spec requires (original values,
reconstruction, per-attribute error, severity) and ask for the four outputs it
requires (responsible attributes, interpretation, risk level, suggestions).
"""

from __future__ import annotations

import pytest

from anomaly_explainer.config import EXPLAINER
from anomaly_explainer.explain.context import AttributeDeviation, ExplanationContext
from anomaly_explainer.explain.prompt import REQUIRED_KEYS, build_prompt


@pytest.fixture
def context() -> ExplanationContext:
    return ExplanationContext(
        severity=4.2,
        error=5.31,
        threshold=1.2634,
        is_anomaly=True,
        transaction_id=1234,
        attributes=(
            AttributeDeviation("V14", -9.42, -0.51, 79.4),
            AttributeDeviation("Amount", 1809.68, 88.21, 12.1),
        ),
    )


def test_prompt_contains_the_verdict_numbers(context):
    prompt = build_prompt(context)
    assert "4.2" in prompt          # severity
    assert "5.31" in prompt         # error
    assert "1.2634" in prompt       # threshold


def test_prompt_contains_original_and_reconstructed_values(context):
    """The spec requires both — the gap is what makes an explanation concrete."""
    prompt = build_prompt(context)
    assert "V14" in prompt and "Amount" in prompt
    assert "1809.68" in prompt      # original
    assert "88.21" in prompt        # reconstructed
    assert "79.4" in prompt         # per-attribute error


def test_prompt_requests_every_required_output_key(context):
    prompt = build_prompt(context)
    for key in REQUIRED_KEYS:
        assert key in prompt, f"prompt does not ask for {key!r}"


def test_prompt_is_deterministic(context):
    assert build_prompt(context) == build_prompt(context)


def test_prompt_states_the_transaction_is_anomalous(context):
    assert "anomal" in build_prompt(context).lower()


def test_prompt_handles_a_normal_transaction(context):
    import dataclasses

    normal = dataclasses.replace(context, is_anomaly=False, severity=0.4)
    prompt = build_prompt(normal)
    assert prompt  # must not raise
    assert "0.4" in prompt


def test_prompt_defaults_to_configured_language(context):
    """The deliverable is French, so explanations must default to French."""
    assert EXPLAINER.language == "fr"
    assert "français" in build_prompt(context).lower()


def test_prompt_language_is_overridable(context):
    prompt = build_prompt(context, language="en")
    assert "english" in prompt.lower()


def test_prompt_rejects_unknown_language(context):
    with pytest.raises(ValueError, match="language"):
        build_prompt(context, language="klingon")


def test_prompt_omits_missing_transaction_id(context):
    import dataclasses

    anonymous = dataclasses.replace(context, transaction_id=None)
    assert build_prompt(anonymous)  # must not raise or print "None"
    assert "None" not in build_prompt(anonymous)
