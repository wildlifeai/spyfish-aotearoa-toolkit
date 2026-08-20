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
    pielou,
    protection_group,
    shannon,
    simpson,
)
from theme import (  # noqa: E402
    NEUTRAL,
    protection_color_map,
    protection_sort_key,
)

from spyfish.config.wrapper import config  # noqa: E402

from ..charting import (  # noqa: E402
    group_colors,
    group_dashes,
    protection_dashes,
    source_coverage_note,
    style,
    year_axis,
)
from ..data import effort_per, real_species  # noqa: E402
from ..layout import section  # noqa: E402
from ._map import gate_notice, site_skeleton  # noqa: E402

# The two sides of a reserve boundary, named once so the aggregation, the
# table and the line dashes cannot disagree about what they are called. The
# dash and colour conventions come from `charting`, shared with the Species
# view; the ratio line is its own metric and gets its own entry.
INSIDE = PROTECTED
OUTSIDE = UNPROTECTED
PARTIAL = OTHER_PROTECTION
SIDE_DASHES = {**group_dashes(), "Inside ÷ outside": "solid"}
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
    fig_prot.update_xaxes(
        title="Deployments", showgrid=True, gridcolor="rgba(128,128,128,0.2)"
    )
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
        "reserve holds, not how the programme divides up."
    )

    # Species per deployment, not mean MaxN: the old column summed MaxN across
    # species before dividing, exactly the snapper-plus-rock-lobster number the
    # caption below the populations panel refuses. Richness counts only real
    # species (the generic and unidentified classes are N unknowns wearing one
    # label), summed per deployment then divided by the analysed count, so a
    # deployment where nothing was identified still weighs the mean down.
    split_df = real_species(split_reserve_rows(df))
    reserve_rows = (
        split_df.groupby("reserve")
        .agg(sites=("site_id", "nunique"), species=("scientific_name", "nunique"))
        .reset_index()
    )
    rich_sum = (
        split_df.groupby(["reserve", "drop_id"])["scientific_name"]
        .nunique()
        .groupby("reserve")
        .sum()
        .reset_index(name="rich_sum")
    )
    # Effort per reserve comes from the deployments table, not from the sightings, so
    # a reserve where nothing was seen still shows its analysed count rather than
    # vanishing from the table.
    reserve_effort = effort_per(split_reserve_rows(effort_view), "reserve")
    reserve_rows = (
        reserve_rows.merge(rich_sum, on="reserve", how="left")
        .merge(reserve_effort, on="reserve", how="outer")
        .fillna({"sites": 0, "species": 0, "rich_sum": 0, "analysed": 0})
    )
    reserve_rows["species_per_dep"] = reserve_rows["rich_sum"] / reserve_rows[
        "analysed"
    ].clip(lower=1)
    reserve_rows = reserve_rows.sort_values("species_per_dep", ascending=False)

    # Protection class beside the table, not another ranking chart: the table
    # already carries `species_per_dep`, so ranking it again would say the same
    # thing twice. Protection is the one thing about an area the table cannot
    # show.
    res_left, res_right = st.columns([1, 1])
    with res_left:
        st.markdown("**Per reserve**")
        st.dataframe(
            reserve_rows,
            width="stretch",
            hide_index=True,
            height=340,
            column_config={
                "reserve": st.column_config.TextColumn("Marine reserve", width="large"),
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
            "**Analysed** is the deployment count each mean rests on. A reserve "
            "with two analysed drops is far less certain than one with forty."
        )
    with res_right:
        render_protection_status(protection_source)


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
    # The shared species filter narrows what the panel's own picker offers,
    # so picking snapper in the sidebar makes this panel show snapper rather
    # than ignoring the selection. Applied to the sightings only; the effort
    # denominators are per site, not per species, and stay whole.
    if species:
        df_context = df_context[df_context["display_name"].isin(species)]
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

    # Indicator species come from config (reporting.indicator_species), intersected
    # with what is present so the default is never empty. `real_species` also
    # keeps the unidentified bucket out of the picker: it is N unknown species
    # wearing one label, not a species to map.
    present = real_species(df_context)
    sci_to_display = (
        present.dropna(subset=["display_name"])
        .drop_duplicates("scientific_name")
        .set_index("scientific_name")["display_name"]
        .to_dict()
    )
    indicator_present = [s for s in config.indicator_species if s in sci_to_display]
    other_present = sorted(s for s in sci_to_display if s not in indicator_present)

    # One measure at a time on the map. MaxN is never summed across species, so a
    # multi-select would either lie (by adding them) or need several maps. Diversity
    # indices are the legitimate way to reduce many species to one number per site.
    radio_order = indicator_present + other_present
    if not radio_order:
        st.info("No species available in this selection.")
        st.stop()

    DIVERSITY_MEASURES = {
        "Species richness": lambda counts: len(counts),
        "Shannon diversity": shannon,
        "Simpson diversity": simpson,
        "Pielou evenness": pielou,
    }

    OTHER = "Other species…"

    mode_col, pick_col = st.columns([1, 3])
    with mode_col:
        map_mode = st.radio("Show", ["Single species", "Biodiversity"], key="map_mode")
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
            diversity_name = st.radio(
                "Metric", list(DIVERSITY_MEASURES), horizontal=True, key="map_div"
            )
            measure_label = diversity_name

    st.caption(
        "Diversity indices are computed per site across all real species, the "
        "generic `fish` class is excluded, since it would otherwise register as one "
        "very abundant species and flatten every evenness score."
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
            "its own line — solid means protected, dashed partial, dotted "
            "unprotected.",
        )
        == "Each protection status"
    )
    if by_status:
        st.caption(
            "One line per recorded protection status. Deployments whose "
            "status is not recorded are grouped as 'Not recorded'. Lines "
            "resting on few deployments swing hard — hover for the number "
            "behind each point."
        )

        def protection_class(status):
            return status.fillna("Not recorded")

    else:
        st.caption(
            "Protected covers "
            + ", ".join(config.protected_statuses)
            + ". Partial regimes (Mataitai, Taiapure, seafloor-only) count as "
            "unprotected because they do not restrict the take of these "
            "species. Deployments whose protection status is not recorded are "
            "left out of both lines."
        )
        # The shared bucketing (config.protected_statuses; unknown → None), so
        # this trend and the Species view's comparison classify every
        # deployment the same way.
        protection_class = protection_group

    trend_effort = effort_view.copy()
    trend_effort["protection"] = protection_class(trend_effort["protection_status"])
    effort_by_year = effort_per(
        trend_effort.dropna(subset=["survey_year"]), ["survey_year", "protection"]
    )

    trend_src = df_context.dropna(subset=["survey_year"]).copy()
    trend_src["protection"] = protection_class(trend_src["protection_status"])

    if map_sci:
        subject = trend_src[trend_src["scientific_name"] == map_sci]
        measured = (
            subject.groupby(["survey_year", "protection"])["maxn"]
            .sum()
            .reset_index(name="total")
        )
        y_title = "Mean MaxN per analysed deployment"
    else:
        real = real_species(trend_src)
        per_year = real.groupby(["survey_year", "protection", "scientific_name"])[
            "maxn"
        ].sum()
        fn = DIVERSITY_MEASURES[diversity_name]
        measured = (
            per_year.groupby(["survey_year", "protection"])
            .apply(lambda s: fn(list(s)))
            .reset_index(name="value")
        )
        y_title = diversity_name

    # A complete year × protection grid, so years with no survey show as a gap in the
    # line rather than being closed up, a compressed axis silently implies continuous
    # monitoring that did not happen.
    years_present = sorted(effort_by_year["survey_year"].dropna().unique())
    # Which lines the grid holds: the two sides, or every status with either a
    # measurement or effort — a status surveyed but never sighted still gets a
    # line of real zeros rather than vanishing.
    classes = (
        sorted(
            set(measured["protection"].dropna())
            | set(effort_by_year["protection"].dropna()),
            key=protection_sort_key,
        )
        if by_status
        else ["Protected", "Unprotected"]
    )
    if years_present:
        full_years = range(int(min(years_present)), int(max(years_present)) + 1)
        grid = pd.MultiIndex.from_product(
            [list(full_years), classes],
            names=["survey_year", "protection"],
        ).to_frame(index=False)
        trend = grid.merge(
            measured, on=["survey_year", "protection"], how="left"
        ).merge(effort_by_year, on=["survey_year", "protection"], how="left")
        if map_sci:
            # Only years that were actually surveyed get a value; a surveyed year with
            # no sighting is a real zero, an unsurveyed year stays NaN and breaks the line.
            trend["value"] = (trend["total"].fillna(0) / trend["analysed"]).where(
                trend["analysed"].notna()
            )
        else:
            trend["value"] = trend["value"].where(trend["analysed"].notna())
    else:
        trend = pd.DataFrame()

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
            "actually surveyed for that protection class — the line between "
            "them bridges unsurveyed years (surveys run on a 1–2 year "
            "cadence), it does not mean continuous monitoring. Hover for the "
            "number of deployments behind each point."
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
        # Site skeleton: every site in the selection that has coordinates, whether or
        # not the chosen species was seen there. Built from df_context (all filters
        # except species) so switching the radio never changes which sites exist,
        # only how big their bubbles are.
        site_geo = site_skeleton(df_context, site_effort, cols=("region",))

        if map_sci:
            per_site = (
                df_context[df_context["scientific_name"] == map_sci]
                .groupby("site_id")["maxn"]
                .sum()
                .reset_index(name="total_maxn")
            )
            mapped = site_geo.merge(per_site, on="site_id", how="left")
            mapped["total_maxn"] = mapped["total_maxn"].fillna(0)
            mapped["value"] = mapped["total_maxn"] / mapped["analysed"].clip(lower=1)
        else:
            # Diversity is computed from summed MaxN per species at each site, with
            # the generic and unidentified classes dropped first.
            real = real_species(df_context)
            counts = real.groupby(["site_id", "scientific_name"])["maxn"].sum()
            fn = DIVERSITY_MEASURES[diversity_name]
            scores = (
                counts.groupby("site_id")
                .apply(lambda s: fn(list(s)))
                .reset_index(name="value")
            )
            mapped = site_geo.merge(scores, on="site_id", how="left")
            mapped["value"] = mapped["value"].fillna(0)

        # Absolute size reference, so a mean MaxN of 13 is the same circle whichever
        # species is showing. Plotly Express rescales to whatever the current maximum
        # happens to be, which silently makes a rare species look as abundant as a
        # common one. The reference is the largest per-site value across ALL species
        # (or all sites, for a diversity metric) in the current filter selection, it
        # changes when the filters change, but never when the radio changes.
        if map_sci:
            all_species_per_site = (
                df_context.groupby(["site_id", "scientific_name"])["maxn"]
                .sum()
                .reset_index()
                .merge(site_effort, on="site_id", how="left")
            )
            all_species_per_site["per_dep"] = all_species_per_site["maxn"] / (
                all_species_per_site["analysed"].fillna(1).clip(lower=1)
            )
            size_ref_max = float(all_species_per_site["per_dep"].max() or 1.0)
        else:
            size_ref_max = float(mapped["value"].max() or 1.0)

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

            # Diameters are computed here in pixels rather than left to Plotly, for
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
            MAX_PX = 34
            MIN_PX = 5
            log_scale = (
                MAX_PX / math.log1p(size_ref_max) if size_ref_max > 0 else MAX_PX
            )

            def bubble_px(values):
                return (np.log1p(values.clip(lower=0)) * log_scale).clip(lower=MIN_PX)

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
                            "size": bubble_px(grp["value"]),
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
                "inside a reserve is still distinguishable from an empty one outside."
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
# Copied, not moved: the Experiments page still holds its own copy, so the
# two can be read against each other before either is retired. Widget keys
# carry a `rep_` prefix here, since Streamlit scopes keys per page and the
# originals keep the unprefixed names.


def render_diversity(df: pd.DataFrame) -> None:
    # Species counts only: the unidentified bucket is N unknown species
    # under one label, so counting it as one both understates richness and
    # puts a meaningless row in the matrix.
    df = real_species(df)
    # The Experiments page built this once for every chart on it and passed it
    # through its context. Here each chart makes its own from the frame it was
    # given, from the same shared `protection_color_map`, so a status is the
    # same colour wherever it appears without a context to thread through.
    prot_cmap = protection_color_map(sorted(df["protection_status"].dropna().unique()))
    section("Diversity")
    st.caption(
        "**Shannon (H)** rewards evenness, a reserve with 10 evenly-distributed species "
        "scores higher than one with 30 species dominated by one. "
        "**Simpson (1−D)** = probability that two random observations are different species. "
        "**Pielou / evenness (J')** = Shannon normalised by ln(species count); 1 = perfectly even, near 0 = one species dominates."
    )

    df = df[
        df["scientific_name"].notna()
        & df["reserve_code"].notna()
        & (df["reserve_code"] != "")
    ].copy()
    if df.empty:
        st.warning("No data after filters.")
        return

    tot = df.groupby(["reserve_code", "display_name"])["maxn"].sum().reset_index()

    rows = []
    for reserve, grp in tot.groupby("reserve_code"):
        counts = grp["maxn"].values
        if counts.sum() == 0:
            continue
        n_spp = int(grp["display_name"].nunique())
        h = shannon(counts)
        rows.append(
            {
                "Reserve": reserve,
                "Species": n_spp,
                "Total MaxN": int(counts.sum()),
                "Shannon (H)": round(h, 3),
                "Simpson (1-D)": round(simpson(counts), 3),
                "Pielou / evenness": round(pielou(counts), 3),
            }
        )
    if not rows:
        st.warning("Not enough data to compute diversity.")
        return

    div_df = pd.DataFrame(rows)

    # Dominant protection status per reserve, for colouring
    reserve_prot = (
        df.groupby("reserve_code")["protection_status"]
        .agg(lambda x: x.value_counts().index[0] if len(x) else "unknown")
        .rename("protection_status")
    )
    div_df = div_df.merge(reserve_prot, left_on="Reserve", right_index=True, how="left")
    div_df["protection_status"] = div_df["protection_status"].fillna("unknown")

    metric_choice = st.radio(
        "Diversity index",
        ["Shannon (H)", "Simpson (1-D)", "Pielou / evenness"],
        horizontal=True,
        key="rep_div_metric",
    )

    fig = px.bar(
        div_df.sort_values(metric_choice, ascending=True),
        x=metric_choice,
        y="Reserve",
        color="protection_status",
        color_discrete_map=prot_cmap,
        orientation="h",
        hover_data={
            "Species": True,
            "Total MaxN": True,
            "Shannon (H)": ":.3f",
            "Simpson (1-D)": ":.3f",
        },
        labels={"protection_status": "Protection"},
        height=max(350, len(div_df) * 32 + 100),
    )
    style(
        fig,
        legend_title_text="Protection status",
        yaxis_title=None,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full table"):
        st.dataframe(
            div_df.sort_values(metric_choice, ascending=False), hide_index=True
        )


def render_community_composition(df: pd.DataFrame) -> None:
    section("Composition")
    st.caption(
        "Top-N species relative abundance per reserve. Shows which species "
        "dominate where, colour patterns reveal community similarities and "
        "differences across reserves."
    )

    df = df[df["scientific_name"].notna() & df["site_id"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    top_n = st.slider("Top N species per reserve", 3, 12, 6, key="rep_comm_top")

    tot = df.groupby(["reserve_code", "display_name"])["maxn"].sum().reset_index()
    reserve_totals = tot.groupby("reserve_code")["maxn"].sum().rename("reserve_total")
    tot = tot.merge(reserve_totals, on="reserve_code")
    tot["pct"] = (tot["maxn"] / tot["reserve_total"] * 100).round(1)

    keep_rows = []
    for reserve, grp in tot.groupby("reserve_code"):
        top = grp.nlargest(top_n, "pct")
        other_pct = grp["pct"].sum() - top["pct"].sum()
        keep_rows.append(top)
        if other_pct > 0:
            keep_rows.append(
                pd.DataFrame(
                    [
                        {
                            "reserve_code": reserve,
                            "display_name": "Other",
                            "maxn": 0,
                            "reserve_total": 0,
                            "pct": round(other_pct, 1),
                        }
                    ]
                )
            )
    plot_df = pd.concat(keep_rows, ignore_index=True)

    reserve_order = reserve_totals.sort_values(ascending=False).index.tolist()

    fig = px.bar(
        plot_df,
        x="pct",
        y="reserve_code",
        color="display_name",
        orientation="h",
        category_orders={"reserve_code": reserve_order},
        labels={
            "pct": "% of MaxN within reserve",
            "reserve_code": "Reserve",
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
        st.dataframe(sorted_df, hide_index=True)

    source_coverage_note(df, "this composition")


def render_reserve_trends(df: pd.DataFrame) -> None:
    section("Trends")
    st.caption(
        "For each reserve: how is the chosen metric changing year-on-year? "
        "Lines show the metric over time; the table fits a linear trend per reserve "
        "and labels each as **↑ Recovering / → Stable / ↓ Declining**."
    )

    df = df[
        df["scientific_name"].notna()
        & df["survey_year"].notna()
        & df["reserve_code"].notna()
        & (df["reserve_code"] != "")
    ].copy()
    if df.empty:
        st.warning("No data after filters.")
        return

    # Richness first, and therefore the default: total MaxN sums across
    # species, so one schooling species dominates it — kept as an explicit
    # "all species summed" option for the within-reserve time comparison,
    # where the species mix partially cancels, but not as the headline.
    metric_choice = st.radio(
        "Metric",
        [
            "Species richness per deployment",
            "Total MaxN per deployment (all species summed)",
            "Reserve effect ratio (inside ÷ outside)",
            "Flagship species MaxN",
        ],
        horizontal=False,
        key="rep_trends_metric",
    )

    flagship_selected = None
    if metric_choice == "Flagship species MaxN":
        all_species = sorted(df["display_name"].unique())
        # Default to Snapper + Blue cod if present
        defaults = [
            s for s in all_species if "Pagrus auratus" in s or "Parapercis colias" in s
        ]
        flagship_selected = st.multiselect(
            "Flagship species (empty = falls back to all-species mean)",
            options=all_species,
            default=defaults,
            key="rep_trends_flagship",
        )

    min_years = st.slider(
        "Minimum years of data per reserve",
        2,
        8,
        3,
        key="rep_trends_min_years",
        help="Reserves with fewer years are shown but not labelled with a trend direction.",
    )

    df["survey_year"] = df["survey_year"].astype(int)

    # ── Compute per-(reserve, year) metric ───────────────────────────────────
    def _agg_total_maxn_per_dep(grp):
        """Per (reserve, year): total MaxN per drop, then mean across drops."""
        per_drop = grp.groupby("drop_id")["maxn"].sum()
        return per_drop.mean()

    def _agg_richness_per_dep(grp):
        per_drop = grp.groupby("drop_id")["scientific_name"].nunique()
        return per_drop.mean()

    def _agg_flagship_maxn(grp, species_list):
        sub = grp[grp["display_name"].isin(species_list)]
        if sub.empty:
            return 0.0
        per_drop = sub.groupby("drop_id")["maxn"].sum()
        return per_drop.mean()

    # Which side of the boundary each deployment sits on. Computed for every
    # metric, not just the ratio: a reserve is surveyed inside AND outside, and
    # collapsing both into one line per reserve was the bug this replaced. The
    # old code labelled the whole reserve by whichever side happened to carry
    # more annotation rows, so AHE (43% inside) read "No protection" while SLI
    # (58% inside) read "Marine Reserve" — the same kind of place, labelled
    # oppositely, and liable to flip when the year filter moved.
    df = df.assign(_side=protection_group(df["protection_status"]))
    partial = int((df["_side"] == PARTIAL).sum())
    df = df[df["_side"].isin([INSIDE, OUTSIDE])]
    if partial:
        st.caption(
            f"{partial:,} annotation rows are left out: a partial or unclear "
            "protection regime is neither the reserve nor its control."
        )
    if df.empty:
        st.warning("No deployments sit clearly inside or outside a reserve.")
        return

    if metric_choice == "Species richness per deployment":
        series = (
            df.groupby(["reserve_code", "survey_year", "_side"])
            .apply(_agg_richness_per_dep, include_groups=False)
            .reset_index(name="metric")
        )
        y_title = "Mean species per deployment"
    elif metric_choice == "Total MaxN per deployment (all species summed)":
        series = (
            df.groupby(["reserve_code", "survey_year", "_side"])
            .apply(_agg_total_maxn_per_dep, include_groups=False)
            .reset_index(name="metric")
        )
        y_title = "Mean total MaxN per deployment (all species)"
    elif metric_choice == "Reserve effect ratio (inside ÷ outside)":
        by_side = (
            df.groupby(["reserve_code", "survey_year", "_side"])
            .apply(_agg_total_maxn_per_dep, include_groups=False)
            .unstack("_side")
        )
        if INSIDE not in by_side.columns or OUTSIDE not in by_side.columns:
            st.warning(
                "Need both inside-reserve and outside-reserve deployments "
                "to compute this ratio."
            )
            return
        by_side["metric"] = by_side[INSIDE] / by_side[OUTSIDE].replace(0, np.nan)
        series = by_side["metric"].reset_index()
        series = series.dropna(subset=["metric"])
        y_title = "MaxN ratio (inside ÷ outside)"
    else:  # Flagship species MaxN
        if not flagship_selected:
            st.info("Pick at least one flagship species.")
            return
        series = (
            df.groupby(["reserve_code", "survey_year", "_side"])
            .apply(
                lambda g: _agg_flagship_maxn(g, flagship_selected),
                include_groups=False,
            )
            .reset_index(name="metric")
        )
        y_title = f"Mean flagship MaxN per deployment ({len(flagship_selected)} spp)"

    if series.empty:
        st.warning("No data after applying the metric.")
        return

    # The ratio metric already compares the two sides, so it has one line per
    # reserve; every other metric has one per side.
    if "_side" not in series.columns:
        series["_side"] = "Inside ÷ outside"
    series = series.rename(columns={"_side": "Side"})

    # ── Linear trend per reserve ────────────────────────────────────────────
    trend_rows = []
    for (reserve, side), grp in series.groupby(["reserve_code", "Side"]):
        grp = grp.dropna(subset=["metric"]).sort_values("survey_year")
        n_years = grp["survey_year"].nunique()
        prot = side

        if n_years < min_years:
            first_year = int(grp["survey_year"].min()) if not grp.empty else None
            latest_year = int(grp["survey_year"].max()) if not grp.empty else None
            first_metric = (
                round(float(grp["metric"].iloc[0]), 2) if not grp.empty else None
            )
            latest_metric = (
                round(float(grp["metric"].iloc[-1]), 2) if not grp.empty else None
            )
            trend_rows.append(
                {
                    "Reserve": reserve,
                    "Side": prot,
                    "n_years": int(n_years),
                    "slope": np.nan,
                    "direction": "insufficient data",
                    "first_year": first_year,
                    "latest_year": latest_year,
                    "first_metric": first_metric,
                    "latest_metric": latest_metric,
                }
            )
            continue

        slope, _ = np.polyfit(grp["survey_year"].values, grp["metric"].values, 1)
        # Stability threshold: 5% of the mean of the metric (per-year basis)
        eps = 0.05 * abs(grp["metric"].mean()) if grp["metric"].mean() else 0
        if slope > eps:
            direction = "↑ Recovering"
        elif slope < -eps:
            direction = "↓ Declining"
        else:
            direction = "→ Stable"

        trend_rows.append(
            {
                "Reserve": reserve,
                "Side": prot,
                "n_years": int(n_years),
                "slope": round(float(slope), 3),
                "direction": direction,
                "first_year": int(grp["survey_year"].min()),
                "latest_year": int(grp["survey_year"].max()),
                "first_metric": round(float(grp["metric"].iloc[0]), 2),
                "latest_metric": round(float(grp["metric"].iloc[-1]), 2),
            }
        )

    trends_df = pd.DataFrame(trend_rows).sort_values(
        "slope", ascending=False, na_position="last"
    )

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
            "reserve_code": "Reserve",
        },
        height=480,
    )
    style(
        fig,
        legend_title_text="Reserve",
        xaxis={"dtick": 1},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    fig.update_yaxes(gridcolor="#EEEEEE")

    if metric_choice == "Reserve effect ratio (inside ÷ outside)":
        # Reference line at ratio = 1 (parity between inside and outside)
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="#888",
            line_width=1,
            annotation_text="parity",
            annotation_position="right",
        )

    year_axis(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Honest caveat ───────────────────────────────────────────────────────
    st.caption(
        "Trend lines fit a simple linear regression per reserve; 3+ years of data "
        "required for a direction label. Real recovery often takes 5–10+ years and "
        "isn't linear, treat short-term trends as hints, not conclusions."
    )

    # ── Summary metrics ─────────────────────────────────────────────────────
    n_up = int((trends_df["direction"] == "↑ Recovering").sum())
    n_down = int((trends_df["direction"] == "↓ Declining").sum())
    n_flat = int((trends_df["direction"] == "→ Stable").sum())
    n_insuff = int((trends_df["direction"] == "insufficient data").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("↑ Recovering", n_up)
    c2.metric("→ Stable", n_flat)
    c3.metric("↓ Declining", n_down)
    c4.metric("Insufficient data", n_insuff)

    st.markdown("**Per-reserve trend table**")
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
        "Deployments inside the reserve boundary against those outside it, per "
        "reserve. A reserve effect is a comparison between the two, so a "
        "reserve sampled almost entirely on one side supports a much weaker "
        "comparison than the bar length alone suggests."
    )

    if df.empty or "reserve_code" not in df.columns:
        st.info("No annotations in this selection.")
        return

    # All three groups here, unlike the comparisons: this chart describes how
    # the surveying was spread, and the partial-regime deployments are part of
    # that picture even though no comparison can use them.
    sided = df.assign(Side=protection_group(df["protection_status"])).dropna(
        subset=["Side"]
    )
    per_reserve = (
        sided.groupby(["reserve_code", "Side"])["drop_id"]
        .nunique()
        .unstack("Side")
        .fillna(0)
    )
    for column in (INSIDE, OUTSIDE, PARTIAL):
        if column not in per_reserve.columns:
            per_reserve[column] = 0
    per_reserve["total"] = (
        per_reserve[INSIDE] + per_reserve[OUTSIDE] + per_reserve[PARTIAL]
    )
    per_reserve = per_reserve[per_reserve["total"] > 0]
    if per_reserve.empty:
        st.info("No annotated deployments in this selection.")
        return
    per_reserve["share_inside"] = per_reserve[INSIDE] / per_reserve["total"]
    per_reserve = per_reserve.sort_values("share_inside")

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
            "reserve_code": "Reserve",
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
                    "Reserve",
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
                "Share inside": st.column_config.ProgressColumn(
                    "Share inside",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                    help="Annotated deployments inside the boundary, as a share "
                    "of the reserve's annotated deployments.",
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
