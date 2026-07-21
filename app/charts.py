"""Altair chart builders for the analyst dashboard.

Altair ships with Streamlit, so this adds no dependency.

Two deliberate design decisions:

* **One axis, scaled units.** Plotting raw values across attributes would be
  meaningless — ``Amount`` is in the thousands while a PCA component sits near
  zero, so a shared axis would render everything except ``Amount`` invisible.
  The chart therefore plots the deviation in *standard deviations*, the one
  quantity comparable across attributes, and the exact figures stay in the table
  beside it.
* **Diverging, not sequential.** "Higher than expected" and "lower than
  expected" are opposite states, not more/less of one thing, so the encoding is
  a two-hue diverging pair around a neutral zero rather than a single ramp.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# Diverging pair (blue <-> red) plus recessive chrome, from the validated
# reference palette. Chosen to stay legible on both light and dark surfaces.
ABOVE_EXPECTED = "#e34948"   # observed higher than the model expected
BELOW_EXPECTED = "#3987e5"   # observed lower than the model expected
AXIS_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

RISK_COLORS = {           # status palette — always paired with an icon + label
    "low": "#0ca30c",
    "medium": "#fab219",
    "high": "#d03b3b",
}
RISK_ICONS = {"low": "●", "medium": "▲", "high": "■"}


def deviation_chart(frame: pd.DataFrame, height: int = 240) -> alt.Chart:
    """Horizontal diverging bars: how far each attribute sat from expectation.

    Args:
        frame: output of ``DetectionService.deviation_frame`` — needs
            ``attribute``, ``gap_z``, ``observed``, ``expected``, ``error``.
    """
    data = frame.assign(
        direction=lambda d: d["gap_z"].apply(
            lambda v: "Au-dessus de l'attendu" if v >= 0 else "En dessous de l'attendu"
        )
    )

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadius=4, height=14)
        .encode(
            x=alt.X(
                "gap_z:Q",
                title="Écart par rapport à l'attendu (écarts-types)",
                axis=alt.Axis(
                    grid=True, gridColor=GRIDLINE, labelColor=AXIS_INK,
                    titleColor=AXIS_INK, domainColor=BASELINE, tickColor=BASELINE,
                ),
            ),
            y=alt.Y(
                "attribute:N",
                title=None,
                sort=alt.EncodingSortField(field="error", order="descending"),
                axis=alt.Axis(labelColor=AXIS_INK, domainColor=BASELINE, ticks=False),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Au-dessus de l'attendu", "En dessous de l'attendu"],
                    range=[ABOVE_EXPECTED, BELOW_EXPECTED],
                ),
                legend=alt.Legend(title=None, orient="top", labelColor=AXIS_INK),
            ),
            tooltip=[
                alt.Tooltip("attribute:N", title="Attribut"),
                alt.Tooltip("observed:Q", title="Observé", format=",.2f"),
                alt.Tooltip("expected:Q", title="Attendu", format=",.2f"),
                alt.Tooltip("gap_z:Q", title="Écart (σ)", format="+,.2f"),
                alt.Tooltip("error:Q", title="Erreur", format=",.3g"),
            ],
        )
    )

    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=BASELINE, strokeWidth=2)
        .encode(x="x:Q")
    )

    return (
        (zero + bars)
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=12, titleFontSize=11)
    )


def severity_chart(severity: float, height: int = 90) -> alt.Chart:
    """A one-bar scale placing this transaction's severity against the threshold.

    Severity is ``error / threshold``, so 1.0 *is* the decision boundary. Drawing
    that boundary explicitly is what makes the number readable at a glance.
    """
    capped = min(float(severity), 10.0)  # keep extreme outliers from flattening the bar
    data = pd.DataFrame({"label": ["sévérité"], "value": [capped]})

    bar = (
        alt.Chart(data)
        .mark_bar(cornerRadius=4, height=22, color=ABOVE_EXPECTED)
        .encode(
            x=alt.X(
                "value:Q",
                scale=alt.Scale(domain=[0, 10]),
                title="× le seuil  (1,0 = limite de détection ; échelle tronquée à 10)",
                axis=alt.Axis(
                    grid=True, gridColor=GRIDLINE, labelColor=AXIS_INK,
                    titleColor=AXIS_INK, domainColor=BASELINE, tickColor=BASELINE,
                ),
            ),
            y=alt.Y("label:N", title=None, axis=None),
            tooltip=[alt.Tooltip("value:Q", title="Sévérité", format=",.2f")],
        )
    )

    boundary = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(color=AXIS_INK, strokeWidth=2, strokeDash=[4, 3])
        .encode(x="x:Q")
    )

    return (
        (bar + boundary)
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=11)
    )
