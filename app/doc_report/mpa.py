"""Reporting - MPA: the marine protected areas themselves.

Three sections, all per-area rather than per-site, which is why they are not on
the Sites view:

* deployments by protection class,
* the per-MPA rollup (sites, effort, species, mean MaxN),
* MPA populations, a species or diversity measure across the areas, its trend
  inside against outside protection, and the site map.

The last two were moved here from `sites.py` unchanged. The loading and
filtering they share with the Sites view lives in `site_data`, so there is one
definition of what a filtered site row is and neither view imports the other.
"""

import sys
from pathlib import Path

# The shared `utils` / `ecology_data` modules live in app/, which is not on
# sys.path when Streamlit runs a page from a subfolder. parents[1] is app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

# The drawing. This module decides what is asked and in what order; how each
# answer looks is one import away, so neither concern is edited by accident
# while working on the other.
from .charts.mpa import (  # noqa: E402
    render_by_mpa,
    render_community_composition,
    render_diversity,
    render_inside_share,
    render_mpa_populations,
    render_reserve_trends,
)
from .data import experiments_frame  # noqa: E402
from .layout import chips  # noqa: E402

# Site-level loading and filtering, shared with the Sites view. In `site_data`
# rather than in either view, so neither view imports the other.
from .site_data import filtered_site_frames, render_map_gate  # noqa: E402


def render(ctx: dict) -> None:
    # Chips first, above the blurb: the strip is chrome that belongs with the
    # filters directly above it, and a line of description between the two
    # split one band into two.
    chips(
        [
            "MPA populations",
            "Site map",
            "Inside vs outside",
            "Diversity",
            "Composition",
            "Trends",
        ]
    )

    st.caption(
        "The marine protected areas: how much protection each area carries, "
        "what each one holds, and how populations differ inside and out."
    )

    dep = ctx["deployments"]
    if dep.empty:
        st.info("No deployments match the current filters.")
        return

    # The unlock control for the site map at the bottom of this view. Rendered
    # first (in the sidebar) so `load_context` below already knows whether this
    # session may hold coordinates.
    render_map_gate()

    site_ctx = load_context(ctx)
    if site_ctx["df_context"].empty:
        st.info("No annotation data for the current filters.")
        return

    render_by_mpa(site_ctx["df_context"], site_ctx["effort_view"], dep)
    render_mpa_populations(
        site_ctx["df_context"],
        site_ctx["effort_view"],
        site_ctx["show_coords"],
        year_range=site_ctx["years"],
        reserves=site_ctx["reserves"],
        regions=site_ctx["regions"],
        protections=site_ctx["protections"],
    )

    # ── Ported from the Experiments page ─────────────────────────────────────
    #
    # These three read the annotations rather than the site frames above, in
    # the shape the Experiments page works in: one row per (deployment, source,
    # species) with `maxn`. `experiments_frame` builds it from the report's
    # already-filtered annotations, so the charts came across unchanged and
    # answer to the report's own filters.
    #
    ann = ctx["annotations"]
    if ann.empty:
        return
    exp = experiments_frame(ann)
    for chart in (
        render_inside_share,
        render_diversity,
        render_community_composition,
        render_reserve_trends,
    ):
        st.divider()
        chart(exp)


def load_context(ctx: dict | None = None) -> dict:
    """The shared site frames. Kept as a name here because the view reads
    better for it; the work is in `site_data`, so the Species view can ask
    for the same frames without importing this module."""
    return filtered_site_frames(ctx)
