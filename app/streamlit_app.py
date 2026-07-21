"""Analyst dashboard — detect, inspect and justify each flagged transaction.

Usage::

    PYTHONPATH=src streamlit run app/streamlit_app.py

This file is presentation only; every lookup it performs lives in
``anomaly_explainer.service`` where it is unit-tested.

Caching strategy, driven by measurements rather than guesswork:

* loading the model and scoring 42,894 transactions takes seconds and must
  happen **once** per session -> ``st.cache_resource``;
* an LLM explanation costs **6-9 s on CPU**, so each one is cached by
  transaction id and rendered behind a spinner — clicking back to a transaction
  already explained is instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` without PYTHONPATH gymnastics.
# The app directory is added explicitly too: `streamlit run` happens to inject it,
# but nothing else does (headless test harnesses included), so relying on that
# leaves the sibling `charts` import working only under one launch path.
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"
for _path in (str(SRC_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from charts import (  # noqa: E402  (import after sys.path setup)
    RISK_COLORS,
    RISK_ICONS,
    deviation_chart,
    severity_chart,
)

from anomaly_explainer.config import ARTIFACTS_DIR, EXPLAINER  # noqa: E402
from anomaly_explainer.explain.explainer import explain  # noqa: E402
from anomaly_explainer.explain.ollama_client import OllamaClient  # noqa: E402
from anomaly_explainer.service import load_service  # noqa: E402

st.set_page_config(
    page_title="Anomaly Explainer — détection de fraude",
    page_icon="🔎",
    layout="wide",
)


# --- data loading -----------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle et scoring des transactions…")
def _service(split: str):
    return load_service(ARTIFACTS_DIR, split=split)


@st.cache_resource
def _ollama_ready() -> bool:
    return OllamaClient().is_available()


@st.cache_data(show_spinner=False)
def _explanation(split: str, index: int) -> dict:
    """Explain one transaction. Cached: each costs 6-9 s on CPU."""
    context = _service(split).context(index, top_k=EXPLAINER.top_k_attributes)
    return explain(context).to_dict()


# --- small render helpers ---------------------------------------------------

def _risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, RISK_COLORS["medium"])
    icon = RISK_ICONS.get(level, "▲")
    # Icon + text, never colour alone.
    return (
        f"<span style='background:{color};color:#fff;padding:3px 12px;"
        f"border-radius:12px;font-weight:600;font-size:0.9rem'>"
        f"{icon}&nbsp; risque {level}</span>"
    )


def _render_sidebar():
    st.sidebar.title("🔎 Anomaly Explainer")
    st.sidebar.caption(
        "Auto-encodeur Transformer + LLM local. Aucune donnée ne quitte la machine."
    )
    split = st.sidebar.radio(
        "Jeu de données",
        options=["test", "val"],
        format_func=lambda s: "Test (jeu de référence)" if s == "test" else "Validation",
        help="Le seuil a été calibré sur la validation ; le test n'a servi qu'une fois.",
    )
    only_anomalies = st.sidebar.checkbox("Uniquement les transactions signalées", True)
    limit = st.sidebar.slider("Nombre de lignes affichées", 10, 500, 100, step=10)

    st.sidebar.divider()
    if _ollama_ready():
        st.sidebar.success(f"LLM local prêt — {EXPLAINER.model_name}")
    else:
        st.sidebar.warning(
            f"{EXPLAINER.model_name} indisponible. Les explications passeront en "
            f"mode dégradé.\n\n`ollama pull {EXPLAINER.model_name}`"
        )
    return split, only_anomalies, limit


def _render_kpis(summary) -> None:
    cols = st.columns(5)
    cols[0].metric("Transactions", f"{summary.n_transactions:,}".replace(",", " "))
    cols[1].metric("Signalées", f"{summary.n_flagged:,}".replace(",", " "))

    if summary.precision is None:
        cols[2].metric("Fraudes réelles", "—", help="Aucun label disponible")
        cols[3].metric("Précision", "—")
        cols[4].metric("Rappel", "—")
        return

    cols[2].metric(
        "Fraudes détectées",
        f"{summary.true_positives} / {summary.n_frauds}",
        help="Vrais positifs sur le nombre total de fraudes",
    )
    cols[3].metric(
        "Précision",
        f"{summary.precision:.1%}",
        help=f"{summary.false_positives} fausses alertes",
    )
    cols[4].metric("Rappel", f"{summary.recall:.1%}", help=f"F1 = {summary.f1:.3f}")


def _render_detail(service, split: str, index: int) -> None:
    context = service.context(index, top_k=EXPLAINER.top_k_attributes)
    actual = service.actual_label(index)

    verdict = "🚨 ANOMALIE" if context.is_anomaly else "✅ Normale"
    st.subheader(f"Transaction #{index} — {verdict}")

    if actual is not None:
        truth = "fraude confirmée" if actual == 1 else "transaction légitime"
        if context.is_anomaly and actual == 1:
            st.success(f"Vérité terrain : {truth} — détection correcte.")
        elif context.is_anomaly and actual == 0:
            st.warning(f"Vérité terrain : {truth} — **fausse alerte**.")
        elif actual == 1:
            st.error(f"Vérité terrain : {truth} — **fraude manquée**.")
        else:
            st.info(f"Vérité terrain : {truth}.")

    left, right = st.columns([1, 1])
    with left:
        st.metric("Sévérité", f"{context.severity:.2f}×",
                  help="Erreur de reconstruction rapportée au seuil ; > 1 = anomalie")
    with right:
        st.metric("Erreur de reconstruction", f"{context.error:.3f}",
                  help=f"Seuil : {context.threshold:.4f}")

    st.altair_chart(severity_chart(context.severity), width="stretch")

    st.markdown("#### Attributs responsables")
    st.caption(
        "Écart entre la valeur observée et celle que le modèle attendait, exprimé "
        "en écarts-types pour rester comparable d'un attribut à l'autre."
    )
    frame = service.deviation_frame(index, top_k=EXPLAINER.top_k_attributes)
    st.altair_chart(deviation_chart(frame), width="stretch")

    with st.expander("Valeurs exactes (unités d'origine)"):
        st.dataframe(
            frame.rename(
                columns={
                    "attribute": "Attribut",
                    "observed": "Observé",
                    "expected": "Attendu par le modèle",
                    "error": "Erreur",
                    "gap_z": "Écart (σ)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    # --- explanation --------------------------------------------------------
    st.markdown("#### Explication")
    with st.spinner("Génération de l'explication par le LLM local (6-9 s)…"):
        result = _explanation(split, index)

    st.markdown(_risk_badge(result["risk_level"]), unsafe_allow_html=True)
    st.write("")
    st.write(result["interpretation"])

    if result["attributes"]:
        st.markdown("**Attributs mis en cause :** " + ", ".join(result["attributes"]))
    if result["suggestions"]:
        st.markdown("**Pistes d'analyse :**")
        for suggestion in result["suggestions"]:
            st.markdown(f"- {suggestion}")

    if result["source"] == "fallback":
        st.caption(
            "⚠️ Explication générée sans le LLM (mode dégradé) — texte déterministe "
            "construit à partir des mêmes mesures."
        )
    else:
        st.caption(f"Généré localement par {EXPLAINER.model_name} via Ollama.")


# --- page -------------------------------------------------------------------

def main() -> None:
    split, only_anomalies, limit = _render_sidebar()

    st.title("Détection et explication des fraudes bancaires")

    try:
        service = _service(split)
    except FileNotFoundError as exc:
        st.error(
            f"Artefacts manquants : {exc}\n\n"
            "Lancer d'abord l'entraînement puis la calibration :\n\n"
            "```\npython scripts/train_model.py\npython scripts/evaluate.py\n```"
        )
        return

    summary = service.summary()
    _render_kpis(summary)
    st.caption(
        f"Seuil : {summary.threshold:.4f} "
        f"({service.threshold.method} @ {service.threshold.parameter}ᵉ percentile)"
    )
    st.divider()

    table_col, detail_col = st.columns([1, 1.3], gap="large")

    with table_col:
        st.markdown("#### File d'attente analyste")
        st.caption("Triée par sévérité décroissante — le plus suspect en premier.")
        frame = service.table(only_anomalies=only_anomalies, limit=limit)

        if frame.empty:
            st.info("Aucune transaction ne correspond au filtre.")
            return

        display = frame.rename(
            columns={
                "index": "N°",
                "severity": "Sévérité",
                "error": "Erreur",
                "amount": "Montant",
                "flagged": "Signalée",
                "actual": "Fraude réelle",
            }
        ).drop(columns=["time"])

        event = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=560,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Sévérité": st.column_config.NumberColumn(format="%.2f"),
                "Erreur": st.column_config.NumberColumn(format="%.3f"),
                "Montant": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        rows = event.selection["rows"] if event and event.selection else []
        selected = int(frame.iloc[rows[0]]["index"]) if rows else int(frame.iloc[0]["index"])

    with detail_col:
        _render_detail(service, split, selected)


if __name__ == "__main__":
    main()
