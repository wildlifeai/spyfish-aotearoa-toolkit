"""Species view for the DOC reporting page.

What lives where, and whether protection makes a difference.

The comparison this view exists for is **protected against unprotected**. It is
the DOC equivalent of the farm-versus-control design: two sets of deployments
surveyed the same way, differing in one thing, so the difference between them is
attributable rather than merely observed.

This module is the view: what the reader is asked, in what order, and which
frame each question is answered from. The drawing is in `charts/species.py`
and the aggregation in `data.py`, so a change to how a chart looks and a change
to what the page says are never the same edit.

MaxN handling: `max_interval` in the annotations table is the count in ONE time
interval, and there is a row per interval. A species' MaxN for a deployment is
the peak across those intervals, never the sum. That peak is taken once, by
`data.species_maxn`, and every frame below inherits it.
"""

import streamlit as st
from ecology_data import (
    PROTECTED,
    UNPROTECTED,
    load_common_names,
    pielou,
    protection_group,
    shannon,
)

from .charts.species import (
    render_cooccurrence,
    render_detection_rate,
    render_freq_abundance,
    render_protection_boxes,
    render_reserve_effect,
    render_species_abundance,
    render_species_accumulation,
    render_species_over_time,
)
from .data import experiments_frame, species_maxn
from .layout import chips, section
from .site_data import filtered_site_frames


def render(ctx: dict) -> None:
    """Render the Species view from the shared context."""
    ann = ctx["annotations"]
    if ann.empty:
        st.warning("No annotations match the current filters.")
        return

    chips(
        [
            "Species abundance",
            "Protected vs unprotected",
            "Which species",
            "MPA effect by species",
            "Species over time",
            "Detection rate",
            "Frequency vs abundance",
            "Co-occurrence",
            "Accumulation",
        ]
    )

    per_species = species_maxn(ann)
    meta = ann[
        ["drop_id", "protection_status", "survey_year", "site_id"]
    ].drop_duplicates("drop_id")
    per_species = per_species.merge(meta, on="drop_id", how="left")
    per_species["Group"] = protection_group(per_species["protection_status"])

    # dropna=False, or deployments with unclassified protection or no parseable
    # date silently vanish from every KPI above the comparison, and the
    # "left out" caption below can never fire because the unclassified rows it
    # counts were already gone.
    per_dep = (
        per_species.groupby(["drop_id", "Group", "survey_year"], dropna=False)
        .agg(abundance=("maxn", "sum"), richness=("scientific_name", "nunique"))
        .reset_index()
    )

    n_deps = per_dep["drop_id"].nunique()
    kpis = st.columns(4)
    kpis[0].metric(
        "Species",
        f"{ann['scientific_name'].nunique():,}",
        help="Distinct species names across every annotated deployment in the "
        "current filters.",
    )
    kpis[1].metric(
        "Deployments",
        f"{n_deps:,}",
        help="Deployments carrying at least one annotation. Deployments with "
        "none are not counted, because a deployment nobody has looked at "
        "is not the same as one where nothing was seen.",
    )
    kpis[2].metric(
        "Mean abundance",
        f"{per_dep['abundance'].mean():.1f}",
        help="Mean total MaxN per deployment: each species' peak count, summed "
        "across species, averaged over deployments. MaxN is the most "
        "individuals visible in a single frame, so it never double-counts "
        "a fish that leaves and returns.",
    )
    kpis[3].metric(
        "Mean richness",
        f"{per_dep['richness'].mean():.1f}",
        help="Mean species recorded per deployment. Rises with how long and how "
        "carefully a deployment was watched, so compare like with like.",
    )

    # Moved off the Sites view: mean MaxN and frequency are statements about
    # species. Reads the site frames rather than `ann`, because the denominator
    # is every surveyed deployment — including the ones where nothing was seen,
    # which is exactly what `ann` cannot show.
    site = filtered_site_frames(ctx)
    if not site["df_context"].empty:
        st.divider()
        render_species_abundance(
            site["df_context"], site["df_context"], site["effort_view"]
        )

    st.divider()

    # ── Protected against unprotected ────────────────────────────────────────
    section("Protected vs unprotected")
    comparable = per_dep[per_dep["Group"].notna()]
    unclassified = per_dep["Group"].isna().sum()

    if comparable["Group"].nunique() < 2:
        st.info(
            "Both a protected and an unprotected group are needed to compare. "
            "Widen the filters, or check that `protection_status` is populated."
        )
    else:
        st.caption(
            "One dot per deployment, so the spread is visible rather than just "
            "the average. The box covers the middle half of deployments and the "
            "line inside it is the median. Two deployments differing only in "
            "protection is the comparison that makes a difference attributable."
        )
        if unclassified:
            st.caption(
                f"{unclassified:,} deployments are left out: their "
                "`protection_status` does not clearly say protected or not, and "
                "guessing would invent the result."
            )

        render_protection_boxes(comparable)

        summary = (
            comparable.groupby("Group")
            .agg(
                Deployments=("drop_id", "nunique"),
                **{
                    "Median abundance": ("abundance", "median"),
                    "Mean abundance": ("abundance", "mean"),
                    "Median richness": ("richness", "median"),
                },
            )
            .round(1)
            .reset_index()
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(
            "A difference here is worth following up, not quoting. Deployments "
            "are not evenly spread across sites, years or effort, so a gap can "
            "come from where and when the surveying happened as much as from "
            "protection."
        )

    st.divider()

    # ── Community ────────────────────────────────────────────────────────────
    section("Which species")
    left, right = st.columns([1.3, 1])

    with left:
        st.caption(
            "**Seen in** is frequency of occurrence: the share of deployments "
            "recording the species at all. More robust than a total, which can "
            "come from a single lucky deployment."
        )
        table = (
            per_species.groupby("scientific_name")
            .agg(
                best=("maxn", "max"), seen=("drop_id", "nunique"), total=("maxn", "sum")
            )
            .reset_index()
            .rename(columns={"scientific_name": "Species"})
        )
        table["Best drop"] = table["best"].astype(int)
        table["Seen in"] = table["seen"].astype(str) + f" / {n_deps}"
        table["Frequency"] = table["seen"] / max(n_deps, 1) * 100
        st.dataframe(
            table.sort_values("Frequency", ascending=False)[
                ["Species", "Best drop", "Seen in", "Frequency"]
            ],
            hide_index=True,
            width="stretch",
            height=360,
            column_config={
                "Best drop": st.column_config.NumberColumn(
                    "Best drop",
                    help="Highest MaxN this species reached in any single "
                    "deployment.",
                ),
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                    help="Share of annotated deployments recording this " "species.",
                ),
            },
        )

    with right:
        st.caption(
            "Diversity across all deployments in the selection. **Shannon** "
            "rises with both the number of species and how evenly they are "
            "spread; its ceiling depends on the species pool, so it is not "
            "comparable to another survey. **Pielou** divides Shannon by its "
            "own maximum, so it is bounded 0-1 and is comparable."
        )
        for group in (PROTECTED, UNPROTECTED):
            rows = per_species[per_species["Group"] == group]
            if rows.empty:
                continue
            totals = rows.groupby("scientific_name")["maxn"].sum()
            st.markdown(f"**{group}**")
            cols = st.columns(3)
            cols[0].metric("Species", f"{len(totals):,}")
            cols[1].metric("Shannon", f"{shannon(totals):.2f}")
            cols[2].metric("Pielou", f"{pielou(totals):.2f}")

    st.divider()
    render_reserve_effect(per_species, load_common_names())

    st.divider()
    render_species_over_time(per_species, load_common_names())

    # ── Ported from the Experiments page ─────────────────────────────────────
    #
    # These five were written against that page's frame: one row per
    # (deployment, source, species) carrying `maxn`. `experiments_frame` builds
    # exactly that from the report's already-filtered annotations, so the
    # charts came across unchanged and answer to the report's own filters.
    #
    exp = experiments_frame(ann)
    for chart in (
        render_detection_rate,
        render_freq_abundance,
        render_cooccurrence,
        render_species_accumulation,
    ):
        st.divider()
        chart(exp)
