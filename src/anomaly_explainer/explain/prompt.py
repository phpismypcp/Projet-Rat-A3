"""Turn an :class:`ExplanationContext` into a prompt for the local LLM.

Design notes
------------
* **Evidence, not conclusions.** The prompt states what the detector measured
  and asks the model to interpret it. It never tells the model "this is fraud",
  because an explanation that just restates our verdict adds nothing.
* **Honesty about the features.** ``V1``..``V28`` are anonymised PCA components
  — nobody, including the LLM, knows what they mean. The prompt says so, to
  discourage the model from inventing confident stories about ``V14``.
* **Strict JSON contract.** The model is asked for exactly four keys, matching
  the spec's required outputs, so the response can be parsed rather than scraped.
"""

from __future__ import annotations

from anomaly_explainer.config import EXPLAINER, ExplainerConfig
from anomaly_explainer.explain.context import ExplanationContext

# The four outputs the project spec requires from the explainer.
REQUIRED_KEYS: tuple[str, ...] = (
    "attributes",
    "interpretation",
    "risk_level",
    "suggestions",
)

SUPPORTED_LANGUAGES: tuple[str, ...] = ("fr", "en")

_LANGUAGE_INSTRUCTION = {
    "fr": "Réponds en français.",
    "en": "Answer in English.",
}

_PREAMBLE = {
    "fr": (
        "Tu assistes un analyste anti-fraude bancaire. Un auto-encodeur Transformer, "
        "entraîné uniquement sur des transactions normales, a analysé la transaction "
        "ci-dessous. Plus la reconstruction est mauvaise, plus le comportement "
        "s'écarte de la normale."
    ),
    "en": (
        "You assist a bank fraud analyst. A Transformer auto-encoder, trained only on "
        "normal transactions, analysed the transaction below. The worse the "
        "reconstruction, the further the behaviour departs from normal."
    ),
}

_FEATURE_CAVEAT = {
    "fr": (
        "Note : V1 à V28 sont des composantes PCA anonymisées — leur signification "
        "métier est inconnue. Ne leur invente pas de sens ; raisonne sur l'ampleur "
        "des écarts. Time et Amount sont en unités réelles."
    ),
    "en": (
        "Note: V1 to V28 are anonymised PCA components — their business meaning is "
        "unknown. Do not invent one; reason about the size of the deviations. Time "
        "and Amount are in real units."
    ),
}

_VERDICT = {
    "fr": {True: "ANOMALIE détectée", False: "transaction jugée normale"},
    "en": {True: "ANOMALY detected", False: "transaction considered normal"},
}

_SCHEMA = {
    "fr": (
        'Réponds UNIQUEMENT avec un objet JSON contenant exactement ces clés :\n'
        '  "attributes"     : liste des attributs responsables (chaînes)\n'
        '  "interpretation" : explication du comportement suspect (chaîne)\n'
        '  "risk_level"     : "low", "medium" ou "high"\n'
        '  "suggestions"    : pistes d\'analyse pour un humain (liste de chaînes)'
    ),
    "en": (
        'Reply ONLY with a JSON object containing exactly these keys:\n'
        '  "attributes"     : list of responsible attributes (strings)\n'
        '  "interpretation" : explanation of the suspicious behaviour (string)\n'
        '  "risk_level"     : "low", "medium" or "high"\n'
        '  "suggestions"    : analysis next steps for a human (list of strings)'
    ),
}

_TABLE_HEADER = {
    "fr": "Attributs les plus mal reconstruits (attribut | observé | attendu | écart) :",
    "en": "Worst-reconstructed attributes (attribute | observed | expected | error):",
}


def build_prompt(
    context: ExplanationContext,
    config: ExplainerConfig = EXPLAINER,
    language: str | None = None,
) -> str:
    """Render the prompt for one transaction.

    Args:
        context: the evidence bundle for the transaction.
        config: explainer settings (supplies the default language).
        language: override the language; one of :data:`SUPPORTED_LANGUAGES`.

    Raises:
        ValueError: if ``language`` is not supported.
    """
    lang = language if language is not None else config.language
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported language {lang!r}; expected one of {SUPPORTED_LANGUAGES}"
        )

    rows = "\n".join(
        f"  {a.name} | {a.original:.2f} | {a.reconstructed:.2f} | {a.error:.4g}"
        for a in context.attributes
    )

    lines = [_PREAMBLE[lang], ""]
    if context.transaction_id is not None:
        lines.append(f"Transaction #{context.transaction_id}")

    lines += [
        f"Verdict: {_VERDICT[lang][context.is_anomaly]}",
        f"Severity: {context.severity:g} (anomalous above 1.0)",
        f"Reconstruction error: {context.error:g} (threshold {context.threshold:g})",
        "",
        _TABLE_HEADER[lang],
        rows,
        "",
        _FEATURE_CAVEAT[lang],
        "",
        _SCHEMA[lang],
        _LANGUAGE_INSTRUCTION[lang],
    ]
    return "\n".join(lines)
