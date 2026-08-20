"""Surveys view for the DOC reporting page.

The PowerBI Surveys tab: how much surveying has happened, when, where, and how
far each survey has got through the pipeline.

A survey is `{Reserve}_{YYYYMMDD}_BUV`, derived from the DropID, because the
pipeline database has no survey table, so this shows IDs. Human survey names
("Banks Peninsula BUV summer 2016/17") live only in `BUV Survey Metadata.csv`
on S3, wiring them in is a possible future enhancement.
"""

import plotly.express as px
import streamlit as st

from .charting import style
from .charts.deployments import add_completion_flags
from .charts.surveys import render_annotation_depth, render_bad_per_survey
from .data import split_reserves
from .layout import chips, section


def render(ctx: dict) -> None:
    """Render the Surveys view from the shared context."""
    dep = ctx["deployments"]
    if dep.empty:
        st.warning("No deployments match the current filters.")
        return

    chips(
        [
            "Survey activity",
            "Surveys",
            "Annotation depth",
            "Bad deployments",
        ]
    )

    dated = dep[dep["survey_year"].notna()]

    # ── Headline counts ──────────────────────────────────────────────────────
    # Widest thing first: MPAs contain sites, sites hold deployments, and
    # surveys are the visits. Reading left to right now goes from place to
    # activity rather than jumping between the two.
    kpis = st.columns(5)
    kpis[0].metric(
        "MPAs",
        f"{len(split_reserves(dep['link_to_marine_reserve'])):,}",
        help="Distinct marine protected areas the surveyed sites link to. "
        "A site between two areas counts under both.",
    )
    kpis[1].metric("Sites", f"{dep['site_id'].nunique():,}")
    kpis[2].metric("Surveys", f"{dep['survey_id'].nunique():,}")
    kpis[3].metric("Deployments", f"{dep['drop_id'].nunique():,}")
    if not dated.empty:
        kpis[4].metric(
            "Survey years",
            f"{int(dated['survey_year'].min())}–{int(dated['survey_year'].max())}",
        )

    # Per-survey averages, copied from the Deployment Management page's Survey
    # Overview. They answer a different question from the counts above: not how
    # much there is, but how big a typical visit is and how much of it comes
    # back annotated.
    flagged = add_completion_flags(dep)
    n_surveys = max(1, dep["survey_id"].nunique())

    kpis = st.columns(5)
    kpis[0].metric(
        "Deployments per survey",
        f"{len(dep) / n_surveys:.1f}",
        help="Mean. A survey is one visit to an MPA.",
    )
    kpis[1].metric(
        "Annotations per survey",
        f"{flagged['total_annotations'].sum() / n_surveys:,.0f}",
        help="Mean across ML, citsci and expert annotations combined.",
    )
    kpis[2].metric(
        "Total annotations",
        f"{int(flagged['total_annotations'].sum()):,}",
    )
    kpis[3].metric(
        "Complete",
        f"{flagged['complete'].mean() * 100:.1f}%",
        help="Deployments where expert or reporting status is complete, or "
        "every section was explicitly skipped. Read from the status "
        "columns, unlike the annotation counts elsewhere on this page.",
    )
    kpis[4].metric(
        "Bad deployments",
        f"{int(dep['is_bad_deployment'].fillna(0).astype(bool).sum()):,}",
    )

    st.caption(
        "Counts are of deployments held in the pipeline database. "
        f"{(dep['ingest_status'] == 'ok').sum():,} passed ingest; "
        f"{(dep['ingest_status'] != 'ok').sum():,} were excluded or hold "
        "validation errors and never reach a processing stage."
    )

    st.divider()

    # ── Activity over time ───────────────────────────────────────────────────
    section("Survey activity")
    st.caption(
        "Counts what is in the database, not what was planned. Deployments per "
        "year is on Report home, split usable against bad, and on Operations "
        "home, split by ingest status."
    )

    if dated.empty:
        st.info("No deployments have a parseable date in their DropID.")
        return

    per_year = (
        dated.groupby(dated["survey_year"].astype(int))
        .agg(Deployments=("drop_id", "nunique"), Surveys=("survey_id", "nunique"))
        .reset_index()
        .rename(columns={"survey_year": "Year"})
    )

    # Surveys per year only. Deployments per year belongs to the two views that
    # already draw it; this is the one view that counts surveys.
    st.markdown("**Surveys per year**")
    fig = px.bar(per_year, x="Year", y="Surveys", text="Surveys")
    fig.update_traces(marker_color="#1E6FB4", textposition="outside", cliponaxis=False)
    style(fig, height=210)
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, key="surveys_per_year")

    st.divider()

    # ── Survey table ─────────────────────────────────────────────────────────
    section("Surveys")
    st.caption(
        "One row per survey. **Analysed** counts deployments carrying any "
        "annotation, so a survey can be fully ingested and not yet analysed. "
        "**ML / CitSci / Expert done** break that down by source, as a share "
        "of the survey. They overlap, because a deployment can carry "
        "annotations from more than one. "
        "**Bad** counts deployments that went wrong in the field: they are not "
        "recoverable by fixing data and will never be annotated."
    )

    has_annotation = (
        dep[["ml_annotations", "citsci_annotations", "expert_annotations"]]
        .fillna(0)
        .gt(0)
        .any(axis=1)
    )
    # Per-source presence, the same measure the Analysed column uses. Counted
    # from the annotation columns rather than the section status columns: an
    # annotation is proof the deployment reached that stage, and the statuses
    # drift from the data. The archived Programme Health table counts the
    # statuses instead, which is why the two can disagree by a deployment or
    # two, see the open question on the reporting-numbers panel.
    per_source = {
        source: dep[f"{source}_annotations"].fillna(0) > 0
        for source in ("ml", "citsci", "expert")
    }

    table = (
        dep.assign(
            _analysed=has_annotation,
            **{f"_{source}": flag for source, flag in per_source.items()},
        )
        .groupby("survey_id")
        .agg(
            Year=("survey_year", "first"),
            Deployments=("drop_id", "nunique"),
            Sites=("site_id", "nunique"),
            Ingested=("ingest_status", lambda s: (s == "ok").sum()),
            Analysed=("_analysed", "sum"),
            ML=("_ml", "sum"),
            CitSci=("_citsci", "sum"),
            Expert=("_expert", "sum"),
            **{"Video present": ("video_presence", lambda s: (s == "present").sum())},
            Bad=("is_bad_deployment", lambda s: int(s.fillna(0).astype(bool).sum())),
        )
        .reset_index()
        .rename(columns={"survey_id": "Survey"})
        .sort_values("Year", ascending=False)
    )
    table["Year"] = table["Year"].astype("Int64")
    # Percentages are stored 0-100, not 0-1. `ProgressColumn`'s `format` is
    # a printf applied to the raw value, so "%.0f%%" on a 0-1 fraction
    # prints "0%" for everything under half and "1%" for a full bar.
    table["Analysed %"] = (table["Analysed"] / table["Deployments"] * 100).fillna(0)
    # Share rather than a count, so a 3-deployment survey losing all 3 is not
    # buried under a 500-deployment survey losing 20.
    table["Bad %"] = (table["Bad"] / table["Deployments"] * 100).fillna(0)
    # Per-source columns are shares, drawn as bars, the way the archived
    # Programme Health table drew them. Share of the survey rather than the raw
    # count: the archived version scaled its bars to the largest survey, which
    # made the bar length a comparison of survey sizes rather than of how far
    # each survey has got. The counts are still in the Analysed column and in
    # the hover.
    for source in ("ML", "CitSci", "Expert"):
        table[f"{source} done"] = (table[source] / table["Deployments"] * 100).fillna(0)

    table = table.drop(columns=["ML", "CitSci", "Expert"])

    # Explicit order: the counts first, then how far each survey has got, with
    # the headline "Analysed" last so the per-source bars build up to it rather
    # than repeating it.
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        column_order=[
            "Survey",
            "Year",
            "Sites",
            "Deployments",
            "Ingested",
            "Video present",
            "Bad",
            "Bad %",
            "ML done",
            "CitSci done",
            "Expert done",
            "Analysed",
            "Analysed %",
        ],
        column_config={
            "Survey": st.column_config.TextColumn(
                "Survey",
                help="Derived from the DropID as {Reserve}_{YYYYMMDD}_BUV. "
                "Survey names live in BUV Survey Metadata.csv on S3, not "
                "in the pipeline database.",
            ),
            "Deployments": st.column_config.NumberColumn(
                "Deployments", help="Distinct DropIDs recorded for this survey."
            ),
            "Ingested": st.column_config.NumberColumn(
                "Ingested",
                help="Deployments with ingest_status = 'ok'. Only these are "
                "picked up by processing stages.",
            ),
            "Analysed": st.column_config.NumberColumn(
                "Analysed",
                help="Deployments carrying at least one annotation from any "
                "source (ML, citsci or expert).",
            ),
            "ML done": st.column_config.ProgressColumn(
                "ML done",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="Share of this survey's deployments carrying ML " "annotations.",
            ),
            "CitSci done": st.column_config.ProgressColumn(
                "CitSci done",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="Share carrying Zooniverse volunteer annotations.",
            ),
            "Expert done": st.column_config.ProgressColumn(
                "Expert done",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="Share carrying expert annotations, legacy or from "
                "BIIGLE. This is the number the reporting rests on.",
            ),
            "Video present": st.column_config.NumberColumn(
                "Video present",
                help="Deployments whose footage is present in S3 "
                "(video_presence = 'present').",
            ),
            "Analysed %": st.column_config.ProgressColumn(
                "Analysed %",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="Share of this survey's deployments that have been " "annotated.",
            ),
            "Bad": st.column_config.NumberColumn(
                "Bad",
                help="Deployments flagged as bad. These went wrong in the "
                "field and are not recoverable by fixing data, so they "
                "will never be annotated.",
            ),
            "Bad %": st.column_config.ProgressColumn(
                "Bad %",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="Share of this survey's deployments flagged bad. A high "
                "share points at something that went wrong on the day, "
                "not at the pipeline.",
            ),
        },
    )

    st.download_button(
        "Download survey summary (CSV)",
        table.to_csv(index=False).encode(),
        file_name="spyfish_survey_summary.csv",
        mime="text/csv",
    )

    st.divider()

    # Side by side: both are per-survey and both answer "which surveys need
    # attention", one from the annotation end and one from the field end. Half
    # the width each keeps them comparable without scrolling between them.
    left, right = st.columns(2)
    with left:
        render_annotation_depth(dep)
    with right:
        render_bad_per_survey(dep)


DONE_BANDS = {"Expert", "CitSci", "ML only"}
