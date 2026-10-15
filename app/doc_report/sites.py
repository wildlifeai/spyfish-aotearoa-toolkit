"""Sites tab for the DOC reporting page.

The site-level view: species abundance, per-MPA and per-site rollups,
protection-status breakdown, the gated map, and indicator species over time.
Reached through the reporting nav (`?view=Sites`).

Everything the view needs arrives through `render(ctx)`, nothing is read
from module scope, so a second caller cannot change what the first sees.
"""

import sys
from pathlib import Path

# The shared `utils` / `ecology_data` modules live in app/, which is not on
# sys.path when Streamlit runs a page from a subfolder. parents[1] is app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from ecology_data import (  # noqa: E402
    add_drop_id_columns,
    join_site_metadata,
    load_effort,
    load_sites,
)

from .charts.sites import (  # noqa: E402
    render_click_to_filter,
    render_site_leaderboard,
    render_site_map,
)
from .charts.species import render_yearly_trend  # noqa: E402
from .data import experiments_frame, richness_rollup  # noqa: E402
from .layout import chips, section  # noqa: E402
from .site_data import (  # noqa: E402
    apply_filters,
    best_per_drop_species,
    load_site_view,
    render_map_gate,
    split_reserves,
)


def render(ctx: dict | None = None) -> None:
    """Render the whole Sites view.

    Every filter, species included, arrives via `ctx` from the shared
    sidebar block. Community metrics here (the KPIs, the per-site richness
    table, the leaderboard) read the species-UNfiltered frames; only the
    per-species year trend follows the Species filter.
    """
    # Above this view's own filters, so the strip sits directly under the
    # report-wide filter band on every view rather than after a row of widgets
    # on this one.
    chips(
        [
            "Site detail",
            "Site map",
            "Leaderboard",
            "Year trend",
            "Click to filter",
        ]
    )

    # ── Filters ───────────────────────────────────────────────────────────────────

    # This view renders no map, so it never needs coordinates, but the gate is
    # honoured anyway (locked by default) so the frame in this view's cache
    # entry only ever carries lat/lon in a session that has unlocked them. The
    # unlock control itself lives on the MPA view, next to the map.
    # The unlock control, in the sidebar, so the map at the foot of this view
    # can be opened without leaving for the MPA view to do it.
    show_coords = render_map_gate()

    df_all = load_site_view(show_coords)

    # Every surveyed deployment, enriched the same way, so effort can be sliced by the
    # same site/protection/region attributes as the sightings.
    effort_all = join_site_metadata(
        add_drop_id_columns(
            load_effort(bool((ctx or {}).get("exclude_bad_deployments")))
        ),
        load_sites(show_coords),
    )

    if df_all.empty:
        st.warning("No annotation data found. Has the pipeline run yet?")
        st.stop()

    df_all = best_per_drop_species(df_all)

    # Every filter comes from the shared sidebar block, species included.
    # This view used to carry its own species multiselect; it was removed
    # (2026-10-02) when the shared Species filter came back, so there is one
    # control rather than two fighting each other.
    year_range = (ctx or {}).get("years")
    reserves = (ctx or {}).get("reserves") or []
    regions = (ctx or {}).get("regions") or []
    protections = (ctx or {}).get("protections") or []

    _filters = dict(
        years=year_range, regions=regions, reserves=reserves, protections=protections
    )

    # Everything except the species filter. "Deployments where nothing was seen" has
    # to be measured against this, not against `df`, with a species selected, `df`
    # would report every drop lacking THAT species as blank, which is a different and
    # much larger number.
    df_context = apply_filters(df_all, **_filters)
    effort_view = apply_filters(effort_all, **_filters)

    species = (ctx or {}).get("species") or []

    if df_context.empty:
        st.warning("No sites match these filters.")
        st.stop()

    # ── Headline counts ───────────────────────────────────────────────────────────
    #
    # From `df_context`, the species-UNfiltered frame: "how many sites, how
    # many species" describes the selection, and ticking one species in the
    # sidebar must not answer "1 species".
    kpis = st.columns(4)
    kpis[0].metric("Deployments", f"{df_context['drop_id'].nunique():,}")
    kpis[1].metric("Sites", f"{df_context['site_id'].nunique():,}")
    kpis[2].metric(
        "MPAs", f"{len(split_reserves(df_context['link_to_marine_reserve'])):,}"
    )
    kpis[3].metric("Species", f"{df_context['scientific_name'].nunique():,}")

    st.divider()

    # ── Species totals ────────────────────────────────────────────────────────────

    # "Species abundance" moved to the Species view: it reports on species,
    # not on sites, and every other section here is keyed by site.

    # ── Per-site breakdown ────────────────────────────────────────────────────────

    section("Site detail")

    # The shared rollup (`data.richness_rollup`), the same function the MPA
    # view's per-area table uses, so "species per deployment" cannot mean
    # two things in one report. Species per deployment, not mean MaxN: the
    # old column summed MaxN across species, the number the MPA view
    # refuses. The frame is the species-UNfiltered one, since richness is a
    # community metric: with the sidebar filter on one species every site
    # would otherwise read "1 species".
    site_rows = richness_rollup(
        df_context,
        effort_view,
        by="site_id",
        attrs=("region", "protection_status"),
    )

    st.dataframe(
        site_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "site_id": st.column_config.TextColumn("SiteID", pinned=True),
            "region": st.column_config.TextColumn("Region"),
            "protection_status": st.column_config.TextColumn("Protection status"),
            "analysed": st.column_config.NumberColumn("Analysed"),
            "species": st.column_config.NumberColumn("Species"),
            "species_per_dep": st.column_config.NumberColumn(
                "Species/dep", format="%.1f"
            ),
        },
        column_order=[
            "site_id",
            "region",
            "protection_status",
            "analysed",
            "species",
            "species_per_dep",
        ],
    )

    st.download_button(
        "Download site summary (CSV)",
        data=site_rows.to_csv(index=False).encode("utf-8"),
        file_name="site_summary.csv",
        mime="text/csv",
    )

    # "Deployments by protection status" is on the MPA view: everything here is
    # per-site, while that counts deployments by the protection class of the
    # area, which is an MPA question.

    # ── Ported from the Experiments page ─────────────────────────────────────
    #
    # Reads the annotations in the shape that page works in, one row per
    # (deployment, source, species) with `maxn`, built by `experiments_frame`.
    ann = (ctx or {}).get("annotations")
    exp = None
    if ann is not None and not ann.empty:
        st.divider()
        exp = experiments_frame(ann)
        # The leaderboard ranks sites by VARIETY, so it reads the
        # species-unfiltered frame; with the sidebar filter on one species
        # every site scored 1 and the chart was a wall of identical bars.
        ann_all = (ctx or {}).get("annotations_all_species")
        exp_all = (
            experiments_frame(ann_all)
            if ann_all is not None and not ann_all.empty
            else exp
        )
        render_site_leaderboard(exp_all)
        st.divider()
        # One line per site over the years: a per-site question, which is
        # why it sits here rather than on Species where it was first put.
        # Per-species by nature, so it follows the sidebar Species filter.
        render_yearly_trend(exp, species_filter=species)

    st.divider()
    render_site_map(df_context, effort_view, show_coords)

    # An example of a technique rather than a finding: click a site, everything
    # below filters. Last on the page because it is here to be judged, not to
    # be relied on. See the note above `render_click_to_filter`. Skipped when
    # there are no annotations, since `exp` is what it filters. `show_coords`
    # is passed down because the gate toggle is a keyed widget already rendered
    # above — drawing it a second time crashes the page.
    if exp is not None:
        st.divider()
        # "Species at the selected sites" is a community question: what is
        # at these sites, not just the picked species.
        render_click_to_filter(exp_all, show_coords)
