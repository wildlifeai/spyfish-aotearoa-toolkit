"""Chart layer for the MPA views.

Everything the Reporting MPA view draws. `mpa.py` decides which questions the
page asks and in what order; these functions take frames and render one chart
or one panel each.

`render_mpa_populations` is the big one, and it is a panel rather than a chart:
it owns the indicator-species picker, the per-MPA trend and the gated site map,
which only make sense read together. Splitting it further would separate a
control from the thing it controls.

The map inside it is coordinate-gated. `show_coords` arrives from the caller
and is never assumed: BUV sites are baited and placed where fish aggregate, so
a map of them joined to abundance functions as a fishing guide.
"""

import math
import sys
from pathlib import Path

# The shared `utils` / `ecology_data` modules live in app/, which is not on
# sys.path when Streamlit runs a page from a subfolder. parents[2] is app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from ecology_data import (  # noqa: E402
    OTHER_PROTECTION,
    PROTECTED,
    UNPROTECTED,
)
from theme import (  # noqa: E402
    GRID_COLOR,
    GRID_ON_WHITE,
    NEUTRAL,
    PLOT_BG_WHITE,
    SINGLE_SERIES,
    protection_color_map,
    protection_sort_key,
)

from spyfish.config.wrapper import config  # noqa: E402

from ..charting import (  # noqa: E402
    focus_inside_outside,
    group_colors,
    group_dashes,
    protection_dashes,
    source_coverage_note,
    style,
    year_axis,
)
from ..data import (  # noqa: E402
    effort_per,
    real_species,
    richness_rollup,
)
from ..layout import section  # noqa: E402
from ..mpa_data import (  # noqa: E402
    DECLINING,
    INSUFFICIENT,
    NOT_RECORDED,
    RECOVERING,
    RICHNESS_METRIC,
    SELECTED_METRIC,
    STABLE,
    bubble_reference,
    composition_shares,
    explode_habitat,
    inside_share,
    linear_trends,
    protection_trend,
    reserve_trend_rows,
    reserve_trend_series,
    site_values,
    species_choices,
    species_peaks,
    split_sides,
    substrate_tables,
)
from ._map import gate_notice, site_skeleton  # noqa: E402

# The two sides of a reserve boundary, named once so the aggregation, the
# table and the line dashes cannot disagree about what they are called. The
# dash and colour conventions come from `charting`, shared with the Species
# view; the ratio line is its own metric and gets its own entry.
INSIDE = PROTECTED
OUTSIDE = UNPROTECTED
PARTIAL = OTHER_PROTECTION
SIDE_DASHES = group_dashes()
from ..site_data import _fit_zoom, split_reserve_rows  # noqa: E402


def render_protection_status(df) -> None:
    """Deployments per protection class, strongest protection first.

    Horizontal, in a half-width column. The class names are long ("Type I MPA
    (Marine Reserve)"), and on an x axis they either angle over into each other
    or eat the plot height; on the y axis they read straight across at full
    size. `category_orders` on a horizontal bar maps top-down, so listing the
    classes strongest-first puts the strongest protection at the top.

    Only rendered embedded beside the By MPA table, so it draws a bold line
    header rather than a section heading: it is one half of a row, not a
    section of its own, and a full heading inside a column read as a section
    nested in a section.
    """
    st.markdown("**Deployments by protection status**")

    prot = (
        df.groupby("protection_status")["drop_id"]
        .nunique()
        .reset_index(name="deployments")
    )
    prot["_order"] = prot["protection_status"].map(protection_sort_key)
    prot = prot.sort_values("_order")

    fig_prot = px.bar(
        prot,
        x="deployments",
        y="protection_status",
        orientation="h",
        color="protection_status",
        color_discrete_map=protection_color_map(prot["protection_status"]),
        category_orders={"protection_status": prot["protection_status"].tolist()},
        text="deployments",
        height=340,
    )
    fig_prot.update_traces(
        marker_cornerradius=4, textposition="outside", cliponaxis=False
    )
    style(fig_prot, legend=False)
    fig_prot.update_xaxes(title="Deployments", showgrid=True, gridcolor=GRID_COLOR)
    fig_prot.update_yaxes(title=None, showgrid=False)
    st.plotly_chart(fig_prot)


def render_by_mpa(
    df: pd.DataFrame, effort_view: pd.DataFrame, protection_source: pd.DataFrame
) -> None:
    """Per-MPA rollup: sites, deployments analysed, species and richness."""
    # ── Per-reserve rollup ────────────────────────────────────────────────────────

    st.caption(
        "A site that sits between two MPAs is counted under **both**, so these "
        "rows do not sum to the totals above, the question here is what each "
        "MPA holds, not how the programme divides up."
    )

    # The shared rollup (`data.richness_rollup`), the same function the Sites
    # view's per-site table uses. Species per deployment, not mean MaxN: the
    # old column summed MaxN across species before dividing, exactly the
    # snapper-plus-rock-lobster number the caption below the populations
    # panel refuses.
    #
    # Both frames are exploded first, so a site between two MPAs counts
    # under each. Effort comes from the deployments, not the sightings, so
    # an MPA where nothing was seen still shows its analysed count.
    reserve_rows = richness_rollup(
        split_reserve_rows(df),
        split_reserve_rows(effort_view),
        by="reserve",
        extra_aggs={"sites": ("site_id", "nunique")},
    )

    # Protection class beside the table, not another ranking chart: the table
    # already carries `species_per_dep`, so ranking it again would say the same
    # thing twice. Protection is the one thing about an area the table cannot
    # show.
    res_left, res_right = st.columns([1, 1])
    with res_left:
        st.markdown("**Per MPA**")
        st.dataframe(
            reserve_rows,
            width="stretch",
            hide_index=True,
            height=340,
            column_config={
                "reserve": st.column_config.TextColumn(
                    "MPA", width="large", pinned=True
                ),
                "sites": st.column_config.NumberColumn("Sites"),
                "analysed": st.column_config.NumberColumn("Analysed"),
                "species": st.column_config.NumberColumn("Species"),
                "species_per_dep": st.column_config.NumberColumn(
                    "Species/dep", format="%.1f"
                ),
            },
            column_order=["reserve", "sites", "analysed", "species", "species_per_dep"],
        )
        st.caption(
            "**Analysed** is how many deployments each average uses. An MPA "
            "with two analysed drops is far less certain than one with forty."
        )
    with res_right:
        render_protection_status(protection_source)


# Diameters are computed in pixels rather than left to Plotly, for
# two reasons.
#
# 1. Absolute scale. The transform below depends only on `size_ref_max`,
#    which is fixed across species, so a given value is always the same
#    circle. Plotly Express rescales to whatever is currently plotted.
# 2. Log spacing. Area-proportional sizing (diameter ∝ √value) crushes the
#    bottom of the range: 1, 2 and 3 land within ~3px of each other while
#    a single unit near the top is worth a quarter of a pixel. BUV counts
#    live mostly at the low end, so that is where the resolution is needed.
#
# The trade-off is explicit: with a log scale, bubble AREA is no longer
# proportional to the count, so areas must not be read as ratios. The
# caption says so, and hover carries the real number.
_MAX_PX = 34
_MIN_PX = 5


def _bubble_px(values: pd.Series, size_ref_max: float) -> pd.Series:
    """Bubble diameter in pixels, log-scaled against `size_ref_max`."""
    log_scale = _MAX_PX / math.log1p(size_ref_max) if size_ref_max > 0 else _MAX_PX
    return (np.log1p(values.clip(lower=0)) * log_scale).clip(lower=_MIN_PX)


def render_mpa_populations(
    df_context: pd.DataFrame,
    effort_view: pd.DataFrame,
    show_coords: bool,
    *,
    year_range=None,
    reserves=None,
    regions=None,
    protections=None,
    species=None,
) -> None:
    """Species and diversity across the MPAs: picker, trend, and the site map."""
    # The shared species filter narrows what the panel's PICKER OFFERS, and
    # nothing else. It used to narrow `df_context` itself, which quietly
    # broke three things downstream, because that frame is also the panel's
    # "everything in scope" reference: the map's site list lost every site
    # that had never recorded the picked species (so a real absence became
    # an absent dot), the richness mode counted only the picked species,
    # and the bubble size reference rescaled with the selection. The chart
    # data is narrowed later by `map_sci`, which is the one species the map
    # is drawing anyway.
    picker_src = (
        df_context[df_context["display_name"].isin(species)] if species else df_context
    )
    df = df_context
    # Deployments analysed per site, used by the map and the trend. Computed
    # here rather than passed in: the Site detail table on the Sites page builds
    # the same frame from the same `effort_view`, and one definition in two
    # places is safer than a shared argument that either caller could change.
    site_effort = effort_per(effort_view, "site_id")
    # ── Map (gated) ───────────────────────────────────────────────────────────────

    st.divider()
    section("MPA populations")
    st.caption(
        "**Each species is kept separate. MaxN is never summed across species.** "
        "Adding a snapper to a rock lobster produces a number with no meaning: they "
        "differ in size, ecological role and how readily a baited camera sees them. "
        "Tick species to compare them side by side, each on its own scale."
    )

    # Indicator species first, then the long tail (`mpa_data.species_choices`).
    sci_to_display, indicator_present, other_present = species_choices(picker_src)

    # One measure at a time on the map. MaxN is never summed across species, so a
    # multi-select would either lie (by adding them) or need several maps. Diversity
    # indices are the legitimate way to reduce many species to one number per site.
    radio_order = indicator_present + other_present
    if not radio_order:
        # `return`, not `st.stop()`: this panel sits mid-page on the Report
        # home, and stopping the script here would also drop every section
        # rendered after it.
        st.info("No species available in this selection.")
        return

    OTHER = "Other species…"

    mode_col, pick_col = st.columns([1, 3])
    with mode_col:
        map_mode = st.radio(
            "Show", ["Single species", "Species richness"], key="map_mode"
        )
    with pick_col:
        if map_mode == "Single species":
            # Indicator species sit in the visible radio; everything else lives one
            # click away behind "Other species…", so the common case stays a single
            # click without hiding the long tail.
            choice = st.radio(
                "Species",
                indicator_present + ([OTHER] if other_present else []),
                format_func=lambda s: s if s == OTHER else sci_to_display[s],
                horizontal=True,
                key="map_species_radio",
            )
            if choice == OTHER:
                with st.expander("Other species", expanded=True):
                    map_sci = st.selectbox(
                        "Choose a species",
                        other_present,
                        format_func=lambda s: sci_to_display[s],
                        key="map_other_species",
                    )
            else:
                map_sci = choice
            measure_label = f"Mean MaxN, {sci_to_display[map_sci]}"
        else:
            map_sci = None
            measure_label = "Species richness"
            st.caption(
                "Species recorded per site, across all real species; the "
                "generic `fish` class is excluded because it is not one "
                "species."
            )

    # ── Indicator species over time, protected vs unprotected ────────────────────

    trend_subject = sci_to_display[map_sci] if map_sci else measure_label
    st.markdown(f"#### {trend_subject} over time, by protection")
    # Per-status is the default: which regime a line belongs to is the first
    # question readers ask of this chart. The binary inside-vs-outside stays
    # for the report's central protected-against-unprotected comparison.
    by_status = (
        st.radio(
            "Lines",
            ["Each protection status", "Inside vs outside"],
            horizontal=True,
            key="map_trend_lines",
            help="**Inside vs outside** is the report's central comparison: "
            "protected against unprotected, partial regimes excluded from "
            "both. **Each protection status** keeps every recorded regime as "
            "its own line, solid means protected, dashed partial, dotted "
            "unprotected.",
        )
        == "Each protection status"
    )
    if by_status:
        st.caption(
            "One line per recorded protection status, opening on the MPA "
            "and no-protection lines; the other regimes are greyed out in "
            "the legend, click one to add its line. Deployments whose site "
            "has no protection metadata are grouped as 'Not recorded'. Lines "
            "resting on few deployments swing hard, hover for the number "
            "behind each point."
        )
    else:
        st.caption(
            "Protected covers "
            + ", ".join(config.protected_statuses)
            + ". Unprotected covers "
            + ", ".join(config.unprotected_statuses)
            + ". Partial regimes (Mataitai, Taiapure, seafloor-only, Type II "
            "MPA) are a third group, 'Other': they restrict some fishing "
            "without clearly protecting these species, so they are left out "
            "of both lines rather than folded into either. Deployments whose "
            "protection status is not recorded are left out too."
        )

    # Which line each deployment belongs to and the year × protection grid
    # both live in `mpa_data.protection_trend`.
    trend, classes, trend_effort = protection_trend(
        df_context, effort_view, map_sci, by_status
    )
    y_title = "Mean MaxN per analysed deployment" if map_sci else "Species richness"

    if trend.empty or trend["value"].notna().sum() == 0:
        st.info("No dated observations for this selection.")
    else:
        fig_trend = px.line(
            trend.sort_values("survey_year"),
            x="survey_year",
            y="value",
            color="protection",
            markers=True,
            # Dotted for unprotected everywhere in the report, as on the Species
            # view: colour alone carries the comparison only for readers who see
            # it, and these two lines are the report's central claim. The
            # per-status grouping reads dash and colour from the shared helpers
            # so "solid = protected" still holds.
            line_dash="protection",
            line_dash_map=(protection_dashes(classes) if by_status else group_dashes()),
            color_discrete_map=(
                protection_color_map(classes) if by_status else group_colors()
            ),
            hover_data={"analysed": True},
            height=400,
        )
        # Connected across unsurveyed years: surveys run on a 1–2 year cadence,
        # so a gap year is the rhythm of the programme, not missed monitoring.
        # The markers still show exactly which years were surveyed, and hover
        # carries the deployment count behind each point.
        fig_trend.update_traces(line={"width": 2}, marker={"size": 9}, connectgaps=True)
        # In per-status mode only the reserve and no-protection lines start
        # selected; partial and unrecorded regimes sit greyed out in the
        # legend. A no-op for the grouped mode, whose two lines both survive.
        focus_inside_outside(fig_trend)
        # Autorange ignores greyed-out traces, so without a pinned range the
        # axis would shrink to the visible lines and hide that the 'Not
        # recorded' line (the legacy imports) reaches further back.
        fig_trend.update_xaxes(
            range=[
                int(trend["survey_year"].min()) - 0.5,
                int(trend["survey_year"].max()) + 0.5,
            ]
        )
        style(
            fig_trend,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "title": None},
        )
        fig_trend.update_yaxes(title=y_title)
        fig_trend.update_xaxes(title=None)
        year_axis(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption(
            "Every year in the range is on the axis. Markers are the years "
            "actually surveyed for that protection class, the line between "
            "them bridges unsurveyed years (surveys run on a 1–2 year "
            "cadence), it does not mean continuous monitoring. Hover for the "
            "number of deployments behind each point."
        )
        # Same note as the Species view: say what the greyed 'Not recorded'
        # line holds, so an axis reaching further back than the visible lines
        # is explained rather than looking broken.
        if by_status:
            nr = trend_effort[trend_effort["protection"] == NOT_RECORDED]
            nr_years = nr["survey_year"].dropna()
            if not nr.empty and not nr_years.empty:
                st.info(
                    f"**{nr['drop_id'].nunique():,} deployments have no "
                    "protection status recorded**, their sites are not in "
                    "the SharePoint sites list. Most are legacy imports, "
                    f"spanning {int(nr_years.min())}–{int(nr_years.max())}; "
                    "in the earliest years they are the only data, which is "
                    "why the axis reaches further back than the visible "
                    "lines. Click 'Not recorded' in the legend to see them."
                )

    st.divider()

    # Headed whether or not it is unlocked, so the chip has somewhere to land
    # and a locked map reads as a section that exists rather than as a stray
    # padlock at the foot of the page.
    section("Site map")

    if gate_notice(df, show_coords):
        # Name the subject on the map itself, by the time you have scrolled here the
        # radio is off-screen, and an unlabelled bubble map says nothing about which
        # animal it describes.
        st.markdown(f"#### {trend_subject} by site")
        # Site skeleton: every site in the selection that has coordinates,
        # whether or not the chosen species was seen there. Built from
        # `df_context`, which carries every filter EXCEPT species (see the
        # note at the top of this function), so neither the radio nor the
        # sidebar species filter changes which sites exist, only how big
        # their bubbles are.
        site_geo = site_skeleton(df_context, site_effort, cols=("region",))

        mapped = site_values(df_context, site_geo, map_sci)
        # Absolute size reference, fixed across the species radio, see
        # `mpa_data.bubble_reference`.
        size_ref_max = bubble_reference(df_context, site_effort, mapped, map_sci)

        if mapped.empty:
            st.warning(
                "No sites in this selection have coordinates recorded. Note that a "
                "stored 0 is treated as 'not recorded' rather than plotted at 0°,0°."
            )
        else:
            # Zeros are drawn as their own trace of fixed small dots rather than being
            # squeezed onto the bubble scale. A proportional size of 0 renders as
            # nothing, and the site vanishes, which reads as "never surveyed" when it
            # actually means "surveyed, none found". Those absences are data, and on a
            # reserve-effect map they are half the argument.
            present_sites = mapped[mapped["value"] > 0]
            zero_sites = mapped[mapped["value"] <= 0]

            fig_map = go.Figure()
            colour_map = protection_color_map(mapped["protection_status"])
            # One trace per protection status in the WHOLE site set, not just the
            # statuses the selected species happens to occur in. A species that
            # appears in three statuses instead of five would otherwise change
            # the trace count, and a figure whose structure changes is re-plotted
            # from scratch rather than reused, which throws away uirevision and
            # with it the user's pan and zoom. Empty traces also keep the legend
            # from gaining and losing entries as the species changes.
            for status in sorted(
                site_geo["protection_status"].dropna().unique(),
                key=protection_sort_key,
            ):
                grp = present_sites[present_sites["protection_status"] == status]
                fig_map.add_trace(
                    go.Scattermap(
                        lat=grp["latitude"],
                        lon=grp["longitude"],
                        mode="markers",
                        name=status,
                        marker={
                            # Already in pixels, sizemode is "diameter" by default.
                            "size": _bubble_px(grp["value"], size_ref_max),
                            "color": colour_map.get(status, NEUTRAL),
                            "opacity": 0.85,
                        },
                        customdata=grp[
                            ["site_id", "region", "analysed", "value"]
                        ].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>%{customdata[1]}"
                            "<br>analysed: %{customdata[2]}"
                            f"<br>{measure_label}: "
                            + "%{customdata[3]:.2f}<extra></extra>"
                        ),
                    )
                )

            # Added even when empty, for the same reason as the loop above: a
            # trace that appears and disappears changes the figure's structure
            # and costs the user their pan and zoom.
            #
            # Zeros keep their protection colour, whether an empty site sits
            # inside or outside a reserve is exactly the comparison this map is
            # for, and greying them out would discard it. Size alone (a fixed
            # small dot) carries "none recorded". No legend entry: the colours are
            # already explained by the bubble legend, and a second set of the same
            # colours would just be noise.
            zero_colours = (
                zero_sites["protection_status"]
                .map(protection_color_map(zero_sites["protection_status"]))
                .fillna(NEUTRAL)
                if not zero_sites.empty
                else []
            )
            fig_map.add_trace(
                go.Scattermap(
                    lat=zero_sites["latitude"],
                    lon=zero_sites["longitude"],
                    mode="markers",
                    marker={"size": 3, "color": list(zero_colours), "opacity": 0.9},
                    showlegend=False,
                    customdata=zero_sites[
                        ["site_id", "region", "analysed", "protection_status"]
                    ].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}"
                        "<br>%{customdata[3]}"
                        "<br>analysed: %{customdata[2]}"
                        "<br><b>none recorded</b><extra></extra>"
                    ),
                )
            )
            # The camera fits the sites the FILTERS leave, not the species
            # selection. Picking an MPA should zoom to it; switching species
            # inside that MPA should not move the map at all, because the same
            # sites are still on screen.
            #
            # `uirevision` is keyed on the filter selection to make that split
            # work: an unchanged key tells Plotly to keep whatever pan and zoom
            # the user has, so the camera below is ignored. Changing a filter
            # changes the key, and the new camera is applied.
            camera_lat = float(site_geo["latitude"].mean())
            camera_lon = float(site_geo["longitude"].mean())
            lat_span = float(site_geo["latitude"].max() - site_geo["latitude"].min())
            lon_span = float(site_geo["longitude"].max() - site_geo["longitude"].min())
            camera_zoom = _fit_zoom(lat_span, lon_span)
            # Anything that changes which sites are plotted belongs in the key.
            camera_key = "sites-map|" + "|".join(
                [
                    ",".join(sorted(reserves or [])),
                    ",".join(sorted(regions or [])),
                    ",".join(sorted(protections or [])),
                    str(year_range),
                ]
            )

            style(
                fig_map,
                height=620,
                # Legend sits above the map: at the bottom it collided with the
                # CARTO / OpenStreetMap attribution overlay.
                margin={"l": 0, "r": 0, "t": 34, "b": 0},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.01,
                    "x": 0,
                    "title": None,
                },
                map={
                    "style": "carto-positron",
                    "center": {"lat": camera_lat, "lon": camera_lon},
                    "zoom": camera_zoom,
                    # Must be set on the map subplot as well: the layout-level
                    # value does not reach a MapLibre subplot, so pan and zoom
                    # were discarded every time the figure was rebuilt for a
                    # different species.
                    "uirevision": camera_key,
                },
                uirevision=camera_key,
            )

            scale_note = (
                " Bubbles are on a **log scale**, so the difference between 1 and 2 is "
                "as visible as the difference between 20 and 25, most BUV counts sit "
                "at the low end. Areas are therefore not proportional to the count; "
                "hover for the real number. The scale is fixed across species, so the "
                "same value is always the same circle."
            )
            zero_note = (
                f" **{len(zero_sites):,} small dots** are sites that were analysed but "
                "recorded none, they keep their protection colour, so an empty site "
                "inside an MPA is still distinguishable from an empty one outside."
                if not zero_sites.empty
                else ""
            )
            # Caption goes ABOVE the map. The CARTO / OpenStreetMap attribution is an
            # HTML overlay owned by the map library, anchored to the bottom of the
            # container rather than to Plotly's plot area, so no amount of bottom
            # margin moves it off text placed underneath. Putting the text above the
            # map sidesteps the collision entirely, and reads better anyway: the
            # encoding is explained before the reader has to interpret it.
            if map_sci:
                st.caption(
                    f"{len(mapped):,} sites. Bubble size is **{measure_label}**, total "
                    "MaxN at the site divided by every deployment analysed there, not "
                    "only those where the species appeared. A site surveyed once with "
                    "one snapper would otherwise match one surveyed forty times "
                    "averaging one snapper; the first is a single observation, the "
                    "second an established population." + scale_note + zero_note
                )
            else:
                st.caption(
                    f"{len(mapped):,} sites. Bubble size is **{measure_label}**, "
                    "computed from summed MaxN per species at each site."
                    + scale_note
                    + zero_note
                )

            st.plotly_chart(fig_map, use_container_width=True, key="sites_map")


# ── Ported from the Experiments page ──────────────────────────────────────
#
# Ported from the old Experiments page, which has since been retired, so
# these are now the only copies. The `rep_` prefix on the widget keys is a
# leftover of that move: harmless, and renaming keys would reset anyone's
# saved selections for no gain.


def render_community_composition(df: pd.DataFrame) -> None:
    section("Composition")
    st.caption(
        "Top-N species relative abundance per MPA. Shows which species "
        "dominate where, colour patterns reveal community similarities and "
        "differences across MPAs."
    )

    df = df[df["scientific_name"].notna() & df["site_id"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    top_n = st.slider("Top N species per MPA", 3, 12, 6, key="rep_comm_top")

    plot_df, reserve_order = composition_shares(df, top_n)

    fig = px.bar(
        plot_df,
        x="pct",
        y="reserve_code",
        color="display_name",
        orientation="h",
        category_orders={"reserve_code": reserve_order},
        labels={
            "pct": "% of MaxN within MPA",
            "reserve_code": "MPA",
            "display_name": "Species",
        },
        height=max(320, len(reserve_order) * 32 + 80),
    )
    style(
        fig,
        barmode="stack",
        xaxis={"ticksuffix": "%", "range": [0, 100]},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Composition table"):
        sorted_df = plot_df.sort_values(
            ["reserve_code", "pct"], ascending=[True, False]
        )
        st.dataframe(
            sorted_df,
            hide_index=True,
            column_config={
                "reserve_code": st.column_config.TextColumn("MPA", pinned=True)
            },
        )

    source_coverage_note(df, "this composition")


def render_reserve_trends(
    df: pd.DataFrame,
    species_filter: list | None = None,
    df_all: pd.DataFrame | None = None,
) -> None:
    """Per-MPA year-on-year trends.

    `df` follows every sidebar filter including Species; `df_all` is the
    species-UNfiltered twin. Richness reads `df_all`: "how many species per
    deployment" must not shrink to 1 because snapper is ticked in the
    sidebar. The Selected-species metric reads `df`, which is the filter
    doing its job.
    """
    section("Trends")
    st.badge("In development", color="red")
    st.caption(
        "Experimental. For each MPA: how is the chosen metric changing "
        "year-on-year? Lines show the metric over time; the table fits a "
        "linear trend per MPA and labels each as "
        "**↑ Recovering / → Stable / ↓ Declining**."
    )

    df = reserve_trend_rows(df)
    df_all = reserve_trend_rows(df_all) if df_all is not None else df
    if df.empty and df_all.empty:
        st.warning("No data after filters.")
        return

    # Two metrics only. Total-MaxN-summed and the inside/outside ratio were
    # removed on review (2026-10-02): summing MaxN across species mixes
    # animals that cannot be added, and the ratio collapsed too much into
    # one number to be read safely.
    metric_choice = st.radio(
        "Metric",
        [RICHNESS_METRIC, SELECTED_METRIC],
        horizontal=False,
        key="rep_trends_metric",
    )
    if metric_choice == RICHNESS_METRIC:
        st.caption(
            "Richness counts the species recorded in a deployment, across "
            "ALL species (the sidebar Species filter does not narrow this "
            "metric). Two caveats: some older annotation sets did not "
            "record every species present, and ML-only deployments can "
            "miss very rare species."
        )

    indicator_selected = None
    if metric_choice == SELECTED_METRIC:
        all_species = sorted(df["display_name"].unique())
        if species_filter:
            # The sidebar Species filter already narrowed the frame; this
            # metric follows it rather than carrying a second picker.
            indicator_selected = all_species
        else:
            # Filter empty: snapper, the report-wide default.
            indicator_selected = [s for s in all_species if "Pagrus auratus" in s]
        if indicator_selected:
            st.caption(
                "Species: **"
                + ", ".join(indicator_selected[:6])
                + ("…" if len(indicator_selected) > 6 else "")
                + "**, change with the sidebar Species filter."
            )

    min_years = st.slider(
        "Minimum years of data per MPA",
        2,
        8,
        3,
        key="rep_trends_min_years",
        help="MPAs with fewer years are shown but not labelled with a trend direction.",
    )

    # ── Compute per-(reserve, year) metric ───────────────────────────────────
    # Both metrics read the species-UNfiltered frame: the denominator is
    # every deployment in the cell, and the sidebar species filter would
    # otherwise drop the deployments that did not record the picked species
    # (pinning the mean to "among those that had it"). The selected-species
    # metric narrows to its species inside the aggregation instead.
    frame = df_all

    # Which side of the boundary each deployment sits on, and why one reserve
    # gets two lines: see `mpa_data.split_sides`.
    frame, partial = split_sides(frame)
    if partial:
        st.caption(
            f"{partial:,} annotation rows are left out: a partial or unclear "
            "protection regime is neither the MPA nor its control."
        )
    if frame.empty:
        st.warning("No deployments sit clearly inside or outside an MPA.")
        return

    if metric_choice == RICHNESS_METRIC:
        series = reserve_trend_series(frame, RICHNESS_METRIC)
        y_title = "Mean species per deployment"
    else:  # Selected species MaxN
        if not indicator_selected:
            st.info(
                "No snapper in this selection. Pick species with the "
                "sidebar Species filter to use this metric."
            )
            return
        series = reserve_trend_series(frame, SELECTED_METRIC, indicator_selected)
        y_title = f"Mean MaxN per deployment ({len(indicator_selected)} species)"

    if series.empty:
        st.warning("No data after applying the metric.")
        return

    # ── Linear trend per reserve ────────────────────────────────────────────
    trends_df = linear_trends(series, min_years)

    # ── Line chart ──────────────────────────────────────────────────────────
    fig = px.line(
        series.sort_values(["reserve_code", "Side", "survey_year"]),
        x="survey_year",
        y="metric",
        color="reserve_code",
        line_dash="Side",
        line_dash_map=SIDE_DASHES,
        markers=True,
        hover_data={"Side": True, "metric": ":.2f"},
        labels={
            "survey_year": "Survey year",
            "metric": y_title,
            "reserve_code": "MPA",
        },
        height=480,
    )
    style(
        fig,
        legend_title_text="MPA",
        xaxis={"dtick": 1},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor=PLOT_BG_WHITE,
    )
    fig.update_xaxes(gridcolor=GRID_ON_WHITE)
    fig.update_yaxes(gridcolor=GRID_ON_WHITE)

    year_axis(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Honest caveat ───────────────────────────────────────────────────────
    st.caption(
        "Trend lines fit a simple linear regression per MPA; 3+ years of data "
        "required for a direction label. Real recovery often takes 5–10+ years and "
        "isn't linear, treat short-term trends as hints, not conclusions."
    )

    # ── Summary metrics ─────────────────────────────────────────────────────
    n_up = int((trends_df["direction"] == RECOVERING).sum())
    n_down = int((trends_df["direction"] == DECLINING).sum())
    n_flat = int((trends_df["direction"] == STABLE).sum())
    n_insuff = int((trends_df["direction"] == INSUFFICIENT).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("↑ Recovering", n_up)
    c2.metric("→ Stable", n_flat)
    c3.metric("↓ Declining", n_down)
    c4.metric("Insufficient data", n_insuff)

    st.markdown("**Per-MPA trend table**")
    st.dataframe(
        trends_df.rename(
            columns={
                "n_years": "Years",
                "slope": "Slope (/yr)",
                "direction": "Trend",
                "first_year": "First year",
                "latest_year": "Latest year",
                "first_metric": "First value",
                "latest_metric": "Latest value",
            }
        ),
        hide_index=True,
    )


def render_inside_share(df) -> None:
    """How much of each reserve's surveying happened inside the boundary.

    A BUV survey of a reserve samples both sides: inside to measure protection,
    outside as the control. This says how that effort actually split, per
    reserve, which is the sampling balance every inside-versus-outside number
    on this page rests on.

    Worth its own section because the imbalance is large and invisible
    elsewhere. It was found through a bug: the reserve-trend chart labelled
    each reserve by whichever side carried more annotation rows, which made a
    43%-inside reserve read "No protection" and a 58%-inside one read "Marine
    Reserve". The label is gone, but the imbalance behind it is real and worth
    showing rather than hiding.
    """
    section("Inside vs outside")
    st.caption(
        "Deployments inside the MPA boundary against those outside it, per "
        "MPA. An MPA effect is a comparison between the two, so an "
        "MPA sampled almost entirely on one side supports a much weaker "
        "comparison than the bar length alone suggests."
    )

    if df.empty or "reserve_code" not in df.columns:
        st.info("No annotations in this selection.")
        return

    # All three groups here, unlike the comparisons (`mpa_data.inside_share`).
    per_reserve = inside_share(df)
    if per_reserve.empty:
        st.info("No annotated deployments in this selection.")
        return

    long = (
        per_reserve[[INSIDE, OUTSIDE, PARTIAL]]
        .reset_index()
        .melt(id_vars="reserve_code", var_name="Side", value_name="Deployments")
    )
    fig = px.bar(
        long,
        x="Deployments",
        y="reserve_code",
        color="Side",
        orientation="h",
        barmode="stack",
        category_orders={
            "reserve_code": per_reserve.index.tolist(),
            "Side": [INSIDE, OUTSIDE, PARTIAL],
        },
        color_discrete_map=group_colors(),
        text="Deployments",
        height=max(280, 26 * len(per_reserve) + 120),
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    style(fig, legend="top", xaxis_title="Annotated deployments")
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="mpa_inside_share")

    table = per_reserve.reset_index().rename(
        columns={
            "reserve_code": "MPA",
            INSIDE: "Protected",
            OUTSIDE: "Unprotected",
            PARTIAL: "Other",
            "total": "Total",
            "share_inside": "Share inside",
        }
    )
    table["Share inside"] = table["Share inside"] * 100
    with st.expander("The numbers behind this"):
        st.dataframe(
            table[
                [
                    "MPA",
                    "Protected",
                    "Unprotected",
                    "Other",
                    "Total",
                    "Share inside",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "MPA": st.column_config.TextColumn("MPA", pinned=True),
                "Share inside": st.column_config.ProgressColumn(
                    "Share inside",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                    help="Annotated deployments inside the boundary, as a share "
                    "of the MPA's annotated deployments.",
                ),
            },
        )


def render_depth(dep: pd.DataFrame, ann: pd.DataFrame) -> None:
    """Deployment depth: the depths surveyed per MPA, and where each species
    was recorded within them.

    Depth is the DEPLOYMENT's depth (`DepthDeployment` from the metadata),
    not the animal's position in the water column: a fish "at 20 m" here is a
    fish recorded by a camera sitting at 20 m.
    """
    st.divider()
    section("Depth")

    depth_dep = dep.assign(depth=pd.to_numeric(dep["depth"], errors="coerce"))
    depth_dep = depth_dep[depth_dep["depth"].notna()]
    if depth_dep.empty:
        st.info(
            "No deployment depths recorded. Depth comes from "
            "`DepthDeployment` in the deployment metadata; re-run ingest "
            "once it is filled in."
        )
        return

    st.caption(
        f"{len(depth_dep):,} of {len(dep):,} deployments carry a depth. "
        "Species sit where the cameras sat: a species shown mostly at 20 m "
        "first of all says the surveying happened at 20 m, so read the right "
        "chart against the surveyed depths on the left."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Surveyed depth per MPA**")
        # The DropID reserve code (KSF, TUK, ...) rather than the full MPA
        # name: the names ate half the chart width as y labels. The full name
        # rides along in the hover.
        by_mpa = depth_dep[depth_dep["reserve_code"] != ""]
        if by_mpa.empty:
            st.info("No deployments with both a depth and an MPA.")
        else:
            order = (
                by_mpa.groupby("reserve_code")["depth"]
                .median()
                .sort_values()
                .index.tolist()
            )
            fig = px.box(
                by_mpa,
                x="depth",
                y="reserve_code",
                category_orders={"reserve_code": order},
                hover_data={"link_to_marine_reserve": True},
            )
            fig.update_traces(marker=dict(size=3), line=dict(width=1.5))
            style(fig, height=max(260, 32 * len(order) + 90), showlegend=False)
            fig.update_xaxes(title="Depth (m)")
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, key="depth_per_mpa")

    with right:
        st.markdown("**Species by deployment depth**")
        # One row per (drop, species): whether the species was recorded on
        # that deployment at all, then the deployment's depth. `real_species`
        # keeps the unidentified bucket out; absence records drop with the
        # notna filter.
        seen = real_species(ann[ann["scientific_name"].notna()])
        seen = seen.drop_duplicates(["drop_id", "scientific_name"]).merge(
            depth_dep[["drop_id", "depth"]], on="drop_id", how="inner"
        )
        # Only species with enough deployments for a distribution to mean
        # anything; a box over two points reads as data it does not have.
        drops_per_species = seen.groupby("display_name")["drop_id"].nunique()
        keep = drops_per_species[drops_per_species >= 5]
        seen = seen[seen["display_name"].isin(keep.index)]
        if seen.empty:
            st.info(
                "No species has five or more depth-carrying deployments "
                "under the current filters."
            )
        else:
            top = keep.sort_values(ascending=False).head(15).index
            seen = seen[seen["display_name"].isin(top)]
            order = (
                seen.groupby("display_name")["depth"]
                .median()
                .sort_values()
                .index.tolist()
            )
            fig = px.box(
                seen,
                x="depth",
                y="display_name",
                category_orders={"display_name": order},
            )
            fig.update_traces(marker=dict(size=3), line=dict(width=1.5))
            style(fig, height=max(260, 32 * len(order) + 90), showlegend=False)
            fig.update_xaxes(title="Depth (m)")
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, key="depth_per_species")
            st.caption(
                "Species with five or more depth-carrying deployments, up "
                "to the 15 most-recorded, shallowest median first."
            )


def render_habitat(
    dep: pd.DataFrame, ann: pd.DataFrame, ann_all: pd.DataFrame | None = None
) -> None:
    """Species against the NZMHCS habitat classes. Under development.

    Reads the `nzmhcs_abiotic` / `nzmhcs_biotic` columns the pipeline
    normalises at ingest (short codes and long spellings merged to one
    label per class). Coverage is sparse, only some surveys recorded
    habitat, so every number names its denominator.

    A deployment listing several classes counts under each: the question
    is "what does this class hold", not a partition of effort.
    """
    section("Habitat")
    st.badge("In development", color="red")

    abiotic = explode_habitat(dep, "nzmhcs_abiotic")
    biotic = explode_habitat(dep, "nzmhcs_biotic")
    if abiotic.empty and biotic.empty:
        st.info(
            "No deployments in this selection carry habitat metadata. "
            "On an older database, run the pipeline's `--ingest` once to "
            "populate the NZMHCS columns."
        )
        return

    # Coverage first, in numbers: habitat was only recorded by some surveys,
    # and every chart below rests on this slice, not on the whole selection.
    n_ab = int(abiotic["drop_id"].nunique())
    n_bi = int(biotic["drop_id"].nunique())
    st.caption(
        f"Habitat comes from the deployment metadata's NZMHCS fields, which "
        f"only some surveys filled in: **{n_ab:,} of the "
        f"{dep['drop_id'].nunique():,} deployments in this selection** carry "
        f"substrate data ({n_bi:,} biotic). Everything below is about those "
        "deployments only."
    )

    left, right = st.columns(2)
    for col, frame, title, key in (
        (left, abiotic, "Substrate (abiotic)", "hab_abiotic"),
        (right, biotic, "Biotic habitat", "hab_biotic"),
    ):
        with col:
            st.markdown(f"**{title}**")
            if frame.empty:
                st.caption("No data in this selection.")
                continue
            counts = (
                frame.groupby("habitat")["drop_id"]
                .nunique()
                .sort_values()
                .reset_index(name="deployments")
            )
            fig = px.bar(
                counts,
                x="deployments",
                y="habitat",
                orientation="h",
                text="deployments",
                height=max(240, 24 * len(counts) + 60),
            )
            fig.update_traces(
                marker_color=SINGLE_SERIES, textposition="outside", cliponaxis=False
            )
            style(fig, legend=False)
            fig.update_yaxes(title=None)
            fig.update_xaxes(title="Deployments with habitat recorded")
            st.plotly_chart(fig, key=key)

    # ── Who turns up on what ─────────────────────────────────────────────────
    # Detection rate and abundance per substrate class, for the most-recorded
    # species in the current selection (the sidebar Species filter narrows
    # `ann`, so picking species narrows these columns too). Peak per (drop,
    # species) first, so a species seen in many intervals of one video is one
    # deployment with one MaxN, same as everywhere else in the report.
    #
    # Denominators ("annotated deps" per class) come from the
    # species-UNfiltered frame, or a sidebar species pick would shrink them
    # to the deployments recording that species and every rate would read
    # 100%. The numerators stay on the filtered peaks, which is the filter
    # choosing which species columns appear.
    denom_ann = ann_all if ann_all is not None else ann
    joined = abiotic.merge(species_peaks(ann), on="drop_id", how="inner")
    if joined.empty:
        st.caption(
            "None of the habitat-carrying deployments are annotated under "
            "the current filters, so the species breakdown is empty."
        )
        return

    det_df, maxn_df, sp_cols = substrate_tables(abiotic, joined, denom_ann)
    if det_df.empty:
        st.caption(
            "No substrate class has five or more annotated deployments under "
            "the current filters."
        )
        return

    # Heatmap shading via a pandas Styler: `st.dataframe` renders its cell
    # backgrounds, and `background_gradient` picks black or white text per
    # cell for contrast, so the tables stay readable in either app theme.
    st.markdown(
        "**Detection rate by substrate**, % of each class's annotated "
        "deployments recording the species"
    )
    st.dataframe(
        det_df.style.background_gradient(
            cmap="Blues", subset=sp_cols, vmin=0, vmax=100
        ).format("{:.0f}%", subset=sp_cols),
        hide_index=True,
        width="stretch",
        column_config={
            "Substrate": st.column_config.TextColumn("Substrate", pinned=True),
        },
    )

    st.markdown(
        "**How many, when seen**, mean MaxN across the deployments in the "
        "class that recorded the species"
    )
    # One green scale across the whole matrix, same convention as the
    # sandbox: schooling species dominate it, so read down a column.
    vmax = pd.to_numeric(maxn_df[sp_cols].stack(), errors="coerce").max()
    st.dataframe(
        maxn_df.style.background_gradient(
            cmap="Greens", subset=sp_cols, vmin=0, vmax=max(float(vmax or 1), 1)
        ).format("{:.1f}", subset=sp_cols),
        hide_index=True,
        width="stretch",
        column_config={
            "Substrate": st.column_config.TextColumn("Substrate", pinned=True),
        },
    )
    st.caption(
        "Classes with at least five annotated deployments; columns are the "
        "most-recorded species under the current filters. A 0.0 in the MaxN "
        "table means never recorded on that substrate. Small denominators "
        "and uneven surveys blur both tables, read them as coverage, not "
        "as a finding."
    )
