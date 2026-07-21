"""Orchestrate: context -> prompt -> local LLM -> structured explanation.

Robustness is the design goal here. A 3B model running on CPU will occasionally
return prose instead of JSON, a bare string where a list was asked for, or an
invented risk level — and Ollama may not be running at all. None of that should
put an error page in front of an analyst, so every failure degrades to a
deterministic template built from the same evidence.

The result always carries ``source`` (``"llm"`` or ``"fallback"``) so the UI can
be honest about where the words came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from anomaly_explainer.config import DATASET, EXPLAINER, ExplainerConfig
from anomaly_explainer.explain.context import AttributeDeviation, ExplanationContext
from anomaly_explainer.explain.ollama_client import OllamaClient, OllamaError
from anomaly_explainer.explain.prompt import REQUIRED_KEYS, build_prompt

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")

# Severity is error/threshold, so 1.0 is exactly the detection boundary.
_MEDIUM_SEVERITY = 1.0
_HIGH_SEVERITY = 3.0


@dataclass(frozen=True)
class Explanation:
    """The analyst-facing explanation, matching the spec's four outputs."""

    attributes: tuple[str, ...]
    interpretation: str
    risk_level: str
    suggestions: tuple[str, ...]
    source: str  # "llm" or "fallback"

    def to_dict(self) -> dict:
        return {
            "attributes": list(self.attributes),
            "interpretation": self.interpretation,
            "risk_level": self.risk_level,
            "suggestions": list(self.suggestions),
            "source": self.source,
        }


def _risk_from_severity(severity: float) -> str:
    """Derive a risk level from how far past the threshold the error sits."""
    if severity < _MEDIUM_SEVERITY:
        return "low"
    if severity < _HIGH_SEVERITY:
        return "medium"
    return "high"


def _interpretable(context: ExplanationContext) -> AttributeDeviation | None:
    """The most deviant attribute a human can actually reason about.

    ``V1``..``V28`` are anonymised PCA components, so "V14 was -9.42" means
    nothing to an analyst. ``Amount`` and ``Time`` are real quantities, so when
    one of them is among the offenders it is worth calling out by name.
    """
    readable = {DATASET.amount_col, DATASET.time_col}
    return next((a for a in context.attributes if a.name in readable), None)


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce a model's answer into a tuple of strings.

    Small models frequently return a bare string where a list was requested, so
    accept both rather than discarding an otherwise good answer.
    """
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def _parse(raw: str, context: ExplanationContext) -> Explanation:
    """Parse the model's JSON reply.

    Raises:
        ValueError: if the reply is not a JSON object carrying the required keys.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"reply is missing required keys: {missing}")

    interpretation = str(payload["interpretation"]).strip()
    if not interpretation:
        raise ValueError("reply has an empty interpretation")

    risk = str(payload["risk_level"]).strip().lower()
    if risk not in RISK_LEVELS:
        # The model invented a level; keep the rest of its answer and fall back
        # to the number we actually measured.
        risk = _risk_from_severity(context.severity)

    return Explanation(
        attributes=_as_str_tuple(payload["attributes"]),
        interpretation=interpretation,
        risk_level=risk,
        suggestions=_as_str_tuple(payload["suggestions"]),
        source="llm",
    )


def fallback_explanation(
    context: ExplanationContext, config: ExplainerConfig = EXPLAINER
) -> Explanation:
    """Build a deterministic explanation from the evidence alone.

    Used whenever the LLM cannot be reached or cannot be parsed. It states the
    same facts in plain language — no interpretation invented, but never empty.
    """
    french = config.language == "fr"
    names = tuple(a.name for a in context.attributes)
    top = context.attributes[0] if context.attributes else None
    readable = _interpretable(context)

    if french:
        verdict = "anormale" if context.is_anomaly else "normale"
        interpretation = (
            f"Transaction jugée {verdict} : erreur de reconstruction "
            f"{context.error:.4g} pour un seuil de {context.threshold:.4g} "
            f"(sévérité {context.severity:.2f}). "
        )
        if top is not None:
            interpretation += (
                f"L'attribut le plus mal reconstruit est {top.name} : "
                f"valeur observée {top.original:.2f}, valeur attendue par le "
                f"modèle {top.reconstructed:.2f}."
            )
        if readable is not None and readable is not top:
            interpretation += (
                f" En clair, {readable.name} vaut {readable.original:.2f} alors "
                f"que le modèle attendait {readable.reconstructed:.2f}."
            )
        suggestions = [
            f"Examiner en priorité les attributs {', '.join(names[:3])}.",
            "Comparer cette transaction à l'historique récent du même porteur.",
            "Explication générée sans le LLM (mode dégradé) — vérifier le "
            "modèle local avec `ollama list` si ce n'était pas volontaire.",
        ]
    else:
        verdict = "anomalous" if context.is_anomaly else "normal"
        interpretation = (
            f"Transaction judged {verdict}: reconstruction error "
            f"{context.error:.4g} against a threshold of {context.threshold:.4g} "
            f"(severity {context.severity:.2f}). "
        )
        if top is not None:
            interpretation += (
                f"The worst-reconstructed attribute is {top.name}: observed "
                f"{top.original:.2f}, model expected {top.reconstructed:.2f}."
            )
        if readable is not None and readable is not top:
            interpretation += (
                f" In plain terms, {readable.name} is {readable.original:.2f} "
                f"where the model expected {readable.reconstructed:.2f}."
            )
        suggestions = [
            f"Review attributes {', '.join(names[:3])} first.",
            "Compare against the cardholder's recent history.",
            "Generated without the LLM (degraded mode) — check `ollama list` if "
            "that was not intentional.",
        ]

    return Explanation(
        attributes=names,
        interpretation=interpretation,
        risk_level=_risk_from_severity(context.severity),
        suggestions=tuple(suggestions),
        source="fallback",
    )


def explain(
    context: ExplanationContext,
    client: OllamaClient | None = None,
    config: ExplainerConfig = EXPLAINER,
) -> Explanation:
    """Explain one transaction, degrading to a template if the LLM fails.

    Never raises for an LLM-side problem: the analyst always gets an answer.
    """
    llm = client if client is not None else OllamaClient(config)

    try:
        raw = llm.generate(build_prompt(context, config))
        return _parse(raw, context)
    except (OllamaError, ValueError, TypeError, json.JSONDecodeError):
        return fallback_explanation(context, config)
