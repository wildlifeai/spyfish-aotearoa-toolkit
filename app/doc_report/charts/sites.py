"""Chart layer for the Sites view.

One function per chart, taking the frame it draws. `sites.py` owns the view:
which questions, in what order, from which frame.

First occupant is the site leaderboard, ported from the retired Experiments page.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from ecology_data import load_sites
from theme import protection_color_map

from ..charting import style, top_n_slider
from ..data import effort_per, real_species
from ..layout import section
from ._map import gate_notice, site_scatter, site_skeleton

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
    st.caption("Which sites hold the most variety? Colour = protection.")

    df = df[df["site_id"].notna() & df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    # Richness only. The Sum-of-MaxN and Mean-MaxN-per-deployment metrics that
    # used to sit beside it summed MaxN across species, which the MPA view
    # rightly refuses ("adding a snapper to a rock lobster produces a number
    # with no meaning") — one cloud of sweep topped both while holding less of
    # a community than eight species in ones and twos. Per-species abundance
    # ranking lives on the Species view, where each species keeps its own
    # scale.
    #
    # Species count, so the unidentified bucket comes out first: it is N
    # unknown species under one label, and counting it as one would put a
    # site with eight named species level with one holding seven plus a blur.
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

    top_n = top_n_slider("Show top N sites", len(board), 30, "rep_leader_topn")
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

    if not gate_notice(df_context, show_coords):
        return

    # Effort, not sightings: the skeleton keeps every site in the selection,
    # so a site nobody has annotated still appears, sized by its deployments.
    sites = site_skeleton(df_context, effort_per(effort_view, "site_id", "deployments"))
    if sites.empty:
        st.info("No sites in this selection carry coordinates.")
        return

    fig = site_scatter(
        sites,
        size="deployments",
        hover_name="site_name",
        hover_data={
            "site_id": True,
            "deployments": True,
            "region": True,
            "latitude": False,
            "longitude": False,
        },
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(sites):,} sites with coordinates, sized by deployments surveyed. "
        "Positions are the recorded site coordinates, not where each drop "
        "actually landed."
    )


# ── Kept as an example, not as a finding ─────────────────────────────────────
#
# The one experiment with no home elsewhere: it demonstrates a Streamlit
# TECHNIQUE (native selection: click a chart or a table row and the script
# reruns with what you clicked) rather than answering a question about fish.
#
# It is here to be judged. If clicking a site to filter the rest of a view is
# useful, the pattern belongs on Sites and MPA properly; if not, this goes.
# Nothing else in the report depends on it.


def render_click_to_filter(df: pd.DataFrame, show_coords: bool) -> None:
    """Click a site on the map or in the table to filter everything below.

    A trial of Streamlit's native selection. `on_select="rerun"` turns a chart
    or a dataframe into an input: the script reruns and the returned event
    carries what was clicked.

    Two things to know about the API, both of which bite quietly:

    * `event.selection.rows` are POSITIONAL indices into the frame that was
      passed, not index labels, so `iloc` is right and `loc` silently returns
      the wrong rows on any sorted frame.
    * Selection lives under the widget key, so it is cleared whenever the widget
      is not rendered, navigating away and back loses it. Anything that must
      survive is mirrored into a plain session key, as the site list is here.
    """
    section("Click to filter")
    st.caption(
        "Click sites on the map or rows in the table. Both feed the same "
        "selection, and everything below responds. Selecting nothing shows "
        "everything."
    )

    # Same gate as the reporting map: real site coordinates joined to abundance
    # are a ranked fishing map, so they stay locked until MAP_PASSWORD is
    # entered. The experiment is about the selection API, not the map, so it is
    # not worth a coordinate leak on an ungated page. The gate WIDGET is drawn
    # once by the Sites view (a keyed widget can only render once per run, a
    # second `render_map_gate()` here crashed the page with a duplicate-key
    # error), so this takes its result instead.
    if not show_coords:
        st.info(
            "🔒 This experiment plots real site coordinates, so it needs the "
            "map password, unlock it in the sidebar."
        )
        return
    sites_geo = load_sites(include_coords=True)
    df = df[df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    # No summed-MaxN column: it added snapper to sweep, and the two counts
    # here (effort and variety) are the ones a site can honestly be ranked by.
    per_site = (
        df.groupby("site_id")
        .agg(
            Deployments=("drop_id", "nunique"),
            Species=("display_name", "nunique"),
        )
        .reset_index()
        .merge(
            sites_geo[
                ["site_id", "site_name", "protection_status", "latitude", "longitude"]
            ],
            on="site_id",
            how="left",
        )
    )
    mapped = per_site.dropna(subset=["latitude", "longitude"])

    # The selection lives here, not in the widgets, so a click on either input
    # updates the same list and neither clears the other's contribution.
    st.session_state.setdefault("click_filter_sites", [])

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("**Sites**, click a marker")
        if mapped.empty:
            st.info("No sites have coordinates.")
        else:
            fig = site_scatter(
                mapped,
                size="Deployments",
                hover_name="site_id",
                custom_data=["site_id"],
                size_max=26,
                zoom=4.2,
                height=430,
                legend_above=True,
            )
            event = st.plotly_chart(
                fig,
                on_select="rerun",
                key="rep_click_map",
                selection_mode=("points", "box", "lasso"),
            )
            picked = [
                point["customdata"][0]
                for point in (event.selection.points if event else [])
                if point.get("customdata")
            ]
            if picked:
                st.session_state["click_filter_sites"] = sorted(set(picked))

    with right:
        st.markdown("**Or click rows**")
        table = per_site[["site_id", "Deployments", "Species"]]
        event = st.dataframe(
            table.sort_values("Deployments", ascending=False),
            hide_index=True,
            width="stretch",
            height=380,
            on_select="rerun",
            selection_mode="multi-row",
            key="rep_click_table",
        )
        rows = event.selection.rows if event else []
        if rows:
            ordered = table.sort_values("Deployments", ascending=False)
            # iloc, not loc: selection.rows are positions in the frame as
            # displayed, and this one is sorted.
            st.session_state["click_filter_sites"] = sorted(
                ordered.iloc[rows]["site_id"]
            )

    selected = st.session_state["click_filter_sites"]
    if selected:
        head, clear = st.columns([4, 1])
        head.success(
            f"Filtered to **{len(selected)}** site(s): "
            + ", ".join(selected[:8])
            + ("…" if len(selected) > 8 else "")
        )
        if clear.button("Clear selection"):
            st.session_state["click_filter_sites"] = []
            st.rerun()
    else:
        st.info("Nothing selected, showing every site.")

    scoped = df[df["site_id"].isin(selected)] if selected else df

    st.divider()
    st.markdown("**Species at the selected sites**")
    species = (
        scoped.groupby("display_name")
        .agg(MaxN=("maxn", "sum"), Deployments=("drop_id", "nunique"))
        .reset_index()
        .sort_values("MaxN", ascending=False)
        .head(15)
    )
    if species.empty:
        st.warning("No species recorded at the selected sites.")
        return
    fig = px.bar(
        species.iloc[::-1],
        x="MaxN",
        y="display_name",
        orientation="h",
        text="MaxN",
        hover_data=["Deployments"],
    )
    fig.update_traces(marker_color="#1E6FB4", textposition="outside", cliponaxis=False)
    style(
        fig,
        height=max(320, 26 * len(species)),
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="rep_click_species")
