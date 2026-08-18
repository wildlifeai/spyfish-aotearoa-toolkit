"""Chart layer for the Sites view.

One function per chart, taking the frame it draws. `sites.py` owns the view:
which questions, in what order, from which frame.

First occupant is the site leaderboard, ported from the retired Experiments page.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from theme import protection_color_map

from ..charting import style
from ..data import real_species
from ..layout import section

# ── Ported from the retired Experiments page ──────────────────────────────
# Widget keys keep their `rep_`
# prefix, which dates from when both pages rendered these and Streamlit scoped
# keys per page.


def render_site_leaderboard(df: pd.DataFrame) -> None:
    # Built here from the frame rather than threaded through a context dict, as
    # it was on the Experiments page, so a status keeps the same colour without
    # a context to carry it.
    prot_cmap = protection_color_map(sorted(df["protection_status"].dropna().unique()))
    section("Leaderboard")
    st.caption(
        "Which sites have the most fish, or the most variety? Colour = protection."
    )

    df = df[df["site_id"].notna() & df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    # Richness first, and therefore the default. It is the one measure here
    # that is not distorted by a single schooling species: a site with one
    # cloud of sweep tops both MaxN measures while holding less of a community
    # than a site with eight species in ones and twos.
    metric = st.radio(
        "Metric",
        ["Species richness", "Mean MaxN per deployment", "Sum of MaxN"],
        horizontal=True,
        key="rep_leader_metric",
    )

    if metric == "Sum of MaxN":
        board = (
            df.groupby(["site_id", "site_name", "protection_status"], dropna=False)[
                "maxn"
            ]
            .sum()
            .reset_index()
            .rename(columns={"maxn": "value"})
        )
        ylabel = "Total MaxN (all species)"
    elif metric == "Mean MaxN per deployment":
        per_dep = (
            df.groupby(
                ["site_id", "site_name", "protection_status", "drop_id"], dropna=False
            )["maxn"]
            .sum()
            .reset_index()
        )
        board = (
            per_dep.groupby(
                ["site_id", "site_name", "protection_status"], dropna=False
            )["maxn"]
            .mean()
            .reset_index()
            .rename(columns={"maxn": "value"})
        )
        board["value"] = board["value"].round(2)
        ylabel = "Mean total MaxN per deployment"
    else:
        # Species count, so the unidentified bucket comes out first: it is N
        # unknown species under one label, and counting it as one would put a
        # site with eight named species level with one holding seven plus a
        # blur. The two MaxN metrics above keep it, because an unidentified
        # fish is still a fish.
        board = (
            real_species(df)
            .groupby(["site_id", "site_name", "protection_status"], dropna=False)[
                "scientific_name"
            ]
            .nunique()
            .reset_index()
            .rename(columns={"scientific_name": "value"})
        )
        ylabel = "Unique species observed"

    top_n = st.slider(
        "Show top N sites",
        5,
        max(5, len(board)),
        min(30, len(board)),
        key="rep_leader_topn",
    )
    board = board.nlargest(top_n, "value").sort_values("value", ascending=True)

    fig = px.bar(
        board,
        x="value",
        y="site_id",
        color="protection_status",
        color_discrete_map=prot_cmap,
        orientation="h",
        labels={"value": ylabel, "site_id": "Site", "protection_status": "Protection"},
        height=max(350, len(board) * 26 + 100),
    )
    style(
        fig,
        legend_title_text="Protection status",
        yaxis_title=None,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)

    n_deps = (
        df.groupby("site_id", dropna=False)["drop_id"].nunique().rename("n_deployments")
    )
    board = board.merge(n_deps, on="site_id", how="left")
    with st.expander("Table"):
        st.dataframe(board.sort_values("value", ascending=False), hide_index=True)


def render_site_map(df_context, effort_view, show_coords: bool) -> None:
    """Where the sites are, sized by effort and coloured by protection.

    Deliberately simpler than the MPA view's map, which sizes bubbles by a
    chosen species' abundance. This one answers "where has the surveying
    happened", which is the Sites question, and needs no species picker.

    Gated on the map password like every other view of coordinates: BUV sites
    are baited and placed where fish aggregate, so positions joined to
    abundance read as a fishing guide, and for rare species as a poaching aid.
    """
    section("Site map")

    if not show_coords:
        st.info(
            "🔒 The map is hidden. Enable **Show map** in the sidebar and enter "
            "the password to view site positions.\n\n"
            "Site coordinates are withheld because BUV sites are baited and "
            "placed where fish aggregate: a map of them joined to abundance "
            "would function as a fishing guide, and for rare species as a "
            "poaching aid."
        )
        return
    if "latitude" not in df_context.columns:
        st.warning(
            "Coordinates are not in the database yet, run `--ingest` to "
            "populate them."
        )
        return

    # Effort, not sightings: a site nobody has annotated is still a site that
    # was surveyed, and leaving it off the map would make the coverage look
    # tidier than it is.
    per_site = effort_view.groupby("site_id")["drop_id"].nunique().rename("deployments")
    sites = (
        df_context.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_id")[
            [
                "site_id",
                "site_name",
                "region",
                "protection_status",
                "latitude",
                "longitude",
            ]
        ]
        .merge(per_site, on="site_id", how="left")
    )
    sites["deployments"] = sites["deployments"].fillna(0)
    if sites.empty:
        st.info("No sites in this selection carry coordinates.")
        return

    fig = px.scatter_map(
        sites,
        lat="latitude",
        lon="longitude",
        size="deployments",
        color="protection_status",
        color_discrete_map=protection_color_map(
            sorted(sites["protection_status"].dropna().unique())
        ),
        hover_name="site_name",
        hover_data={
            "site_id": True,
            "deployments": True,
            "region": True,
            "latitude": False,
            "longitude": False,
        },
        size_max=22,
        zoom=4,
        height=520,
    )
    fig.update_layout(
        map_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "y": -0.05, "title_text": ""},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(sites):,} sites with coordinates, sized by deployments surveyed. "
        "Positions are the recorded site coordinates, not where each drop "
        "actually landed."
    )
