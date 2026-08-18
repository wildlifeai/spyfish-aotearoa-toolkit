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
    two would claim something the status does not say.
    """
    from .data import protection_rank

    return {
        status: ("solid", "dash", "dot", "dot")[protection_rank(status)]
        for status in statuses
    }


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


def outside_labels(fig: go.Figure, **traces) -> go.Figure:
    """Bar labels outside the bar, and not clipped by the axis.

    `cliponaxis=False` is the half of this that is easy to forget: without it
    the label on the longest bar — the one most worth reading — is the one the
    plot area cuts off.
    """
    fig.update_traces(textposition="outside", cliponaxis=False, **traces)
    return fig


def no_axis_titles(fig: go.Figure, x: bool = True, y: bool = True) -> go.Figure:
    """Drop axis titles that only repeat the section heading above the chart."""
    if x:
        fig.update_xaxes(title=None)
    if y:
        fig.update_yaxes(title=None)
    return fig
