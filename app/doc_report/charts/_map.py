"""Shared pieces for the site maps.

Three charts draw sites on a map: the Sites view map, the MPA populations map
and the click-to-filter experiment. They answer different questions, so their
bubbles MEAN different things (effort, abundance, a selection target) — that
stays with each chart. What lives here is what must not drift apart:

* the coordinates gate — one privacy explanation, not three copies,
* the site skeleton — "which sites exist on a map" is the same question
  everywhere: every site in the current selection that has coordinates,
  whether or not anything was recorded there,
* the scatter base — protection colouring on the shared carto style, so the
  maps read as the same map wearing different data.

The MPA populations map keeps its own drawing (log-scale bubbles, zero-dot
traces, camera uirevision) and takes only the gate and the skeleton from here.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from theme import protection_color_map

from ..charting import style

# One wording for why the map is locked, shared by every gated map. The
# explanation IS the policy, so two copies drifting apart would mean two
# different policies on screen.
LOCKED_NOTE = (
    "🔒 The map is hidden. Enable **Show map** in the sidebar and enter the "
    "password to view site positions.\n\n"
    "Site coordinates are withheld because BUV sites are baited and placed "
    "where fish aggregate: a map of them joined to abundance would function "
    "as a fishing guide, and for rare species as a poaching aid."
)


def gate_notice(frame: pd.DataFrame, show_coords: bool) -> bool:
    """The two ways a map can be unavailable. Returns True when clear to draw.

    Renders the locked note or the missing-coordinates warning itself, so a
    caller is a single `if gate_notice(...)` around the drawing.
    """
    if not show_coords:
        st.info(LOCKED_NOTE)
        return False
    if "latitude" not in frame.columns:
        st.warning(
            "Coordinates are not in the database yet, run `--ingest` to "
            "populate them."
        )
        return False
    return True


def site_skeleton(
    df_context: pd.DataFrame,
    effort=None,
    cols: tuple = ("site_name", "region"),
) -> pd.DataFrame:
    """One row per site with coordinates, whatever was recorded there.

    Built from the context frame (all page filters applied) so every map
    shows the same set of sites for a given selection — a site nobody has
    annotated is still a site that was surveyed, and leaving it off makes
    coverage look tidier than it is.

    `effort` is a per-site count to merge on: a named Series indexed by
    site_id or a DataFrame carrying a `site_id` column. Sites absent from it
    are kept at 0, not dropped.
    """
    keep = ["site_id", *cols, "protection_status", "latitude", "longitude"]
    sites = df_context.dropna(subset=["latitude", "longitude"]).drop_duplicates(
        "site_id"
    )[keep]
    if effort is not None:
        if isinstance(effort, pd.Series):
            effort = effort.reset_index()
        value_cols = [c for c in effort.columns if c != "site_id"]
        sites = sites.merge(effort, on="site_id", how="left")
        sites[value_cols] = sites[value_cols].fillna(0)
    return sites


def site_scatter(
    sites: pd.DataFrame,
    *,
    size: str,
    hover_name: str,
    hover_data: dict | None = None,
    custom_data: list | None = None,
    size_max: int = 22,
    zoom: float = 4,
    height: int = 520,
    legend_above: bool = False,
) -> px.scatter_map:
    """The shared scatter base: protection colours on the carto style.

    What bubble size MEANS belongs to the caller; this owns only what must
    look identical across maps — colouring, base map, margins and legend
    placement. `legend_above` puts the legend over the map, for charts where
    it would otherwise collide with the CARTO attribution overlay.
    """
    fig = px.scatter_map(
        sites,
        lat="latitude",
        lon="longitude",
        size=size,
        color="protection_status",
        color_discrete_map=protection_color_map(
            sorted(sites["protection_status"].dropna().unique())
        ),
        hover_name=hover_name,
        hover_data=hover_data,
        custom_data=custom_data,
        size_max=size_max,
        zoom=zoom,
        map_style="carto-positron",
    )
    style(
        fig,
        height=height,
        margin={"l": 0, "r": 0, "t": 30 if legend_above else 0, "b": 0},
        legend=(
            {"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0, "title": None}
            if legend_above
            else {"orientation": "h", "y": -0.05, "title_text": ""}
        ),
    )
    return fig
