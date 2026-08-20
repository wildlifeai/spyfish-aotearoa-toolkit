"""Plotly conventions, in one place.

Nearly every chart in the report repeated the same three or four calls: apply
`PLOT_LAYOUT`, set a height, move the legend somewhere that does not overlap
the plot, push bar labels outside without clipping them. Repeated, they drifted
— legends ended up in three different places and two charts reversed their own
key — and each new chart copied whichever neighbour it was pasted next to.

These helpers are deliberately thin. They take a figure and return it, so they
compose with anything Plotly does, and none of them decide what a chart *is*:
the chart modules still choose the marks, the colours and the ordering. What
lives here is only the part that should look the same everywhere.

`theme.py` holds the values (palette, `PLOT_LAYOUT`); this holds the moves.
"""

import plotly.graph_objects as go
from theme import PLOT_LAYOUT

# Where a legend goes, given that the plot area must not shrink to make room.
#
# "top" sits above the plot, reading left to right like a heading; "bottom"
# below it, for a chart whose title already explains the split. Both are
# horizontal: a vertical legend on the right steals width from every bar in the
# chart, and these charts are usually in a half-width column already.
_LEGENDS = {
    "top": dict(orientation="h", y=1.12, x=0, title_text=""),
    "bottom": dict(orientation="h", y=-0.25, x=0, title_text=""),
    # Plotly reverses legend order on stacked charts, so the key reads bottom-up
    # against a bar that stacks top-down. `traceorder` puts it back.
    "stack": dict(orientation="h", y=-0.35, x=0, title_text="", traceorder="normal"),
}


def protection_dashes(statuses) -> dict:
    """Line style per protection status: solid inside a reserve, dotted outside.

    One convention, one place. Any chart drawing a line per protection status
    reads it from here, so "solid = protected" holds across the report instead
    of Plotly assigning dash styles in whatever order the values happened to
    arrive — which is what made two charts disagree about which line was the
    reserve.

    Partial protection gets a dash: it is neither, and drawing it as one of the
    two would claim something the status does not say. The grouping is
    `protection_group`, the same config-driven definition every comparison
    uses, so a status cannot be solid on one chart and dotted on the next.
    """
    import pandas as pd
    from ecology_data import OTHER_PROTECTION, PROTECTED, protection_group

    groups = protection_group(pd.Series(list(statuses)))
    dashes = {PROTECTED: "solid", OTHER_PROTECTION: "dash"}
    return {status: dashes.get(group, "dot") for status, group in zip(statuses, groups)}


def group_colors() -> dict:
    """One colour pair for the Protected/Unprotected comparison.

    Three charts used to carry three different hex pairs for the same
    comparison. These are the theme's own anchors — the strongest-protection
    blue and the no-protection orange from `PROTECTION_COLORS` — so a chart
    coloured by group and one coloured by exact status agree about which end
    of the spectrum each side sits on. The partial group gets the neutral
    grey: it is neither side.
    """
    from ecology_data import OTHER_PROTECTION, PROTECTED, UNPROTECTED
    from theme import NEUTRAL, PROTECTION_COLORS

    return {
        PROTECTED: PROTECTION_COLORS["Type I MPA (Marine Reserve)"],
        UNPROTECTED: PROTECTION_COLORS["No protection"],
        OTHER_PROTECTION: NEUTRAL,
    }


def group_dashes() -> dict:
    """The group-level dash convention: solid = protected, dot = not.

    The per-status version is `protection_dashes` above; this one keys on the
    already-bucketed group labels for charts drawing one line per side.
    """
    from ecology_data import OTHER_PROTECTION, PROTECTED, UNPROTECTED

    return {PROTECTED: "solid", OTHER_PROTECTION: "dash", UNPROTECTED: "dot"}


def source_coverage_note(df, label: str = "these numbers") -> None:
    """One line saying what the chart above rests on, by annotation source.

    Every rate in the report divides by deployments, and a deployment can have
    been read by a model, by volunteers, or by an expert. "This reserve is
    diverse" and "an expert reviewed this reserve" produce the same bar, and
    nothing on screen distinguished them.

    Deliberately a caption rather than a second chart: it is context for the
    chart above, and a stacked bar of its own would compete with the thing it
    is explaining.
    """
    import streamlit as st
    from ecology_data import source_bucket

    if df.empty or "annotated_by" not in df.columns:
        return
    per_source = (
        df.assign(_source=source_bucket(df["annotated_by"]))
        .groupby("_source")["drop_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    if per_source.empty:
        return
    total = int(df["drop_id"].nunique())
    parts = ", ".join(
        f"**{int(n)}** {source.lower()}" for source, n in per_source.items()
    )
    st.caption(
        f"Behind {label}: {total:,} deployments, read by {parts}. Sources are "
        "not equally thorough, and they do not cover the same places, so a "
        "difference between two bars can be a difference in who looked."
    )


def style(
    fig: go.Figure,
    *,
    height: int | None = None,
    legend: str | dict | bool | None = None,
    **layout,
) -> go.Figure:
    """Apply the report's shared layout, and a legend position by name.

    `legend` is one of "top", "bottom", "stack", `False` to hide it, a Plotly
    legend dict for a chart that needs its own placement, or None to leave it
    alone. Anything else passes straight through to `update_layout`, so a chart
    that needs something unusual still can.
    """
    extra = dict(layout)
    if height is not None:
        extra["height"] = height
    if legend is False:
        extra["showlegend"] = False
    elif isinstance(legend, dict):
        # A chart with its own legend placement passes the dict straight
        # through. Named positions are the shorthand, not the only way in.
        extra["legend"] = legend
    elif legend:
        extra["legend"] = _LEGENDS[legend]

    # Merged, not splatted side by side. Two reasons, both of which bit:
    #
    # * `update_layout(**PLOT_LAYOUT, margin=...)` is a duplicate keyword and
    #   raises, so a chart wanting its own margin could not use the shared
    #   layout at all — which is how charts ended up not using it.
    # * `xaxis={"ticksuffix": "%"}` REPLACES the whole shared `xaxis` dict, so a
    #   chart asking for a tick suffix silently lost `showgrid: False` and drew
    #   gridlines nothing else has. One level of depth fixes that: the caller's
    #   keys win, the rest of the shared dict survives.
    merged = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in PLOT_LAYOUT.items()
    }
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    fig.update_layout(**merged)
    return fig


def year_axis(fig: go.Figure) -> go.Figure:
    """Year ticks as plain full numbers on the x-axis.

    Year columns are ints, so Plotly treats the axis as continuous: on a wide
    span it thins the ticks to every fifth or tenth year (a 15-year chart read
    as holding two years), and its locale formatting can render 2024 as
    "2,024". One tick per year, bare digits.
    """
    fig.update_xaxes(dtick=1, tickformat="d")
    return fig


def top_n_slider(
    label: str, n_items: int, default: int, key: str, help: str | None = None
) -> int:
    """The "Show top N" slider, with the bounds every chart was hand-rolling.

    Floor of 5 so the chart is never a single bar, ceiling at the item count,
    default clamped into range — four charts wrote this triple with small
    inconsistencies; one definition keeps the sliders behaving alike.
    """
    import streamlit as st

    return st.slider(
        label, 5, max(5, n_items), min(default, n_items), key=key, help=help
    )
