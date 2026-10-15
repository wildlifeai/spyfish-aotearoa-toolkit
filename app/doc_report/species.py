"""Species view for the DOC reporting page.

What lives where, and whether protection makes a difference.

The comparison this view exists for is **protected against unprotected**. It is
the DOC equivalent of the farm-versus-control design: two sets of deployments
surveyed the same way, differing in one thing, so the difference between them is
attributable rather than merely observed.

This module is the view: what the reader is asked, in what order, and which
frame each question is answered from. The drawing is in `charts/species.py`
(protection, time) and `charts/species_community.py`, and the aggregation in
`data.py`, so a change to how a chart looks and a change to what the page
says are never the same edit.

MaxN handling: `max_interval` in the annotations table is the count in ONE time
interval, and there is a row per interval. A species' MaxN for a deployment is
the peak across those intervals, never the sum. That peak is taken once, by
`data.species_maxn`, and every frame below inherits it.
"""

import streamlit as st
from ecology_data import (
    OTHER_PROTECTION,
    PROTECTED,
    UNPROTECTED,
    load_common_names,
    protection_group,
)

from .charts.species import (
    render_protection_boxes,
    render_protection_effect_ci,
    render_reserve_effect,
    render_species_over_time,
)
from .charts.species_community import (
    render_cooccurrence,
    render_detection_rate,
    render_freq_abundance,
    render_species_abundance,
    render_species_accumulation,
    render_species_depth,
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
            "Depth",
        ]
    )

    per_species = species_maxn(ann)
    meta = ann[
        ["drop_id", "protection_status", "survey_year", "site_id", "reserve_code"]
    ].drop_duplicates("drop_id")
    per_species = per_species.merge(meta, on="drop_id", how="left")
    per_species["Group"] = protection_group(per_species["protection_status"])

    # The species-UNfiltered twin, for everything that describes the whole
    # community: KPIs, richness, the protected-vs-unprotected panel. Ticking
    # snapper in the sidebar must not turn "species per deployment" into 1.
    # The per-species sections (over time, by site, the effect slope) keep
    # the filtered frame, which is the filter doing its job.
    ann_all = ctx["annotations_all_species"]
    per_species_all = species_maxn(ann_all)
    # `reserve_code` (the DropID prefix) rides along so the protected-vs-
    # unprotected comparison can be made WITHIN each MPA rather than pooled
    # across places that are not sampled alike.
    meta_all = ann_all[
        ["drop_id", "protection_status", "survey_year", "site_id", "reserve_code"]
    ].drop_duplicates("drop_id")
    per_species_all = per_species_all.merge(meta_all, on="drop_id", how="left")
    per_species_all["Group"] = protection_group(per_species_all["protection_status"])

    # dropna=False, or deployments with unclassified protection or no parseable
    # date silently vanish from every KPI above the comparison, and the
    # "left out" caption below can never fire because the unclassified rows it
    # counts were already gone.
    per_dep = (
        per_species_all.groupby(
            ["drop_id", "Group", "survey_year", "reserve_code"], dropna=False
        )
        .agg(abundance=("maxn", "sum"), richness=("scientific_name", "nunique"))
        .reset_index()
    )

    n_deps = per_dep["drop_id"].nunique()
    kpis = st.columns(4)
    kpis[0].metric(
        "Species",
        f"{ann_all['scientific_name'].nunique():,}",
        help="Distinct species names across every annotated deployment in the "
        "current filters. Not narrowed by the sidebar Species filter.",
    )
    kpis[1].metric(
        "Deployments",
        f"{n_deps:,}",
        help="Deployments carrying at least one annotation. A reviewed "
        "deployment where nothing was seen carries an absence record, so "
        "it counts; a deployment nobody has looked at has no rows and "
        "does not.",
    )
    kpis[2].metric(
        "Mean abundance",
        f"{per_dep['abundance'].mean():.1f}",
        help="Mean total MaxN per deployment: each species' peak count, summed "
        "across species, averaged over deployments. A reviewed deployment "
        "where nothing was seen counts as zero. MaxN is the most "
        "individuals visible in a single frame, so it never double-counts "
        "a fish that leaves and returns.",
    )
    kpis[3].metric(
        "Mean richness",
        f"{per_dep['richness'].mean():.1f}",
        help="Mean species recorded per deployment, counting a reviewed-empty "
        "deployment as zero. Rises with how long and how carefully a "
        "deployment was watched, so compare like with like.",
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
    # The comparison is protected against unprotected, so it uses only those
    # two. Partial regimes are their own group and are counted out loud below
    # rather than being folded into whichever side would have taken them.
    comparable = per_dep[per_dep["Group"].isin([PROTECTED, UNPROTECTED])]
    partial = int((per_dep["Group"] == OTHER_PROTECTION).sum())
    unknown = int(per_dep["Group"].isna().sum())

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
        if partial or unknown:
            left_out = []
            if partial:
                left_out.append(
                    f"**{partial:,}** sit in a partial or unclear regime "
                    "(Type II MPA, taiapure, mataitai, fisheries closure, "
                    "seafloor protection)"
                )
            if unknown:
                left_out.append(f"**{unknown:,}** have no protection status recorded")
            st.caption(
                "Left out of this comparison: "
                + "; ".join(left_out)
                + ". They restrict some fishing without clearly protecting the "
                "species counted here, so putting them on either side would "
                "answer the question with deployments that cannot answer it. "
                "They are still in every other chart on this page."
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

        # The means above always differ. This says whether each difference
        # survives resampling, per indicator species rather than as one
        # cross-species total, and measured inside each MPA rather than
        # pooled across places that are not sampled alike.
        render_protection_effect_ci(comparable, per_species_all, load_common_names())

    st.divider()

    # ── Community ────────────────────────────────────────────────────────────
    section("Which species")
    left, right = st.columns([1.3, 1])

    with left:
        st.caption(
            "**Seen in** is frequency of occurrence: the share of deployments "
            "recording the species at all. Harder to skew than a total, which can "
            "come from a single lucky deployment."
        )
        # Grouped on display_name ("Common name (Scientific name)"), not the
        # raw scientific name: the report shows common names everywhere, and
        # display_name falls back to the scientific name where no common name
        # is known, so nothing is lost.
        species_col = (
            "display_name"
            if "display_name" in per_species.columns
            else "scientific_name"
        )
        table = (
            per_species.groupby(species_col)
            .agg(
                best=("maxn", "max"), seen=("drop_id", "nunique"), total=("maxn", "sum")
            )
            .reset_index()
            .rename(columns={species_col: "Species"})
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
                "Species": st.column_config.TextColumn("Species", pinned=True),
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
        # Diversity indices (Shannon/Pielou) were removed on review
        # (2026-10-02): species counts say the same thing in plain language.
        st.caption(
            "Species recorded on each side of the comparison, across all "
            "deployments in the selection."
        )
        for group in (PROTECTED, UNPROTECTED):
            rows = per_species_all[per_species_all["Group"] == group]
            if rows.empty:
                continue
            totals = rows.groupby("scientific_name")["maxn"].sum()
            st.markdown(f"**{group}**")
            cols = st.columns(2)
            cols[0].metric("Species", f"{len(totals):,}")
            cols[1].metric("Deployments", f"{rows['drop_id'].nunique():,}")

    st.divider()
    # `per_dep` carries the deployment universe: the effect per species is a
    # mean over EVERY classified deployment, counting the ones that did not
    # record the species as zeros, so it needs more than the rows where the
    # species appears. `per_species_all` so the chart is not narrowed to one
    # species by the sidebar filter, which would leave a single dot.
    render_reserve_effect(per_species_all, load_common_names(), per_dep)

    st.divider()
    render_species_over_time(
        per_species,
        load_common_names(),
        species_filter=ctx.get("species"),
        per_species_all=per_species_all,
        # With an MPA picked, the site heatmaps show all of its sites
        # rather than a sample.
        mpa_selected=bool(ctx.get("reserves")),
    )

    # ── Ported from the Experiments page ─────────────────────────────────────
    #
    # These five were written against that page's frame: one row per
    # (deployment, source, species) carrying `maxn`. `experiments_frame` builds
    # exactly that from the report's already-filtered annotations, so the
    # charts came across unchanged and answer to the report's own filters.
    #
    # Detection rate follows the Species filter (rates of the species you
    # picked); the community charts (strategies, co-occurrence, discovery
    # curve) read the unfiltered frame, since they describe the whole
    # community whatever is ticked in the sidebar.
    exp = experiments_frame(ann)
    exp_all = experiments_frame(ctx["annotations_all_species"])
    st.divider()
    render_detection_rate(exp, df_all=exp_all)
    for chart in (
        render_freq_abundance,
        render_cooccurrence,
        render_species_accumulation,
    ):
        st.divider()
        chart(exp_all)

    # Needs the deployments frame as well (depth lives there, not on the
    # annotations), so it sits outside the single-frame loop above.
    st.divider()
    render_species_depth(exp, ctx["deployments"])
