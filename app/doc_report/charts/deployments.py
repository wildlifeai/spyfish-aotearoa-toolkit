"""Deployment-level charts: the funnel, the per-year bars, and the browsers.

Drawn by Report home, Operations home, Operations - Deployments and Reporting -
Annotations. Living here rather than in any one of those is what stops a view
importing another view to borrow a chart.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from theme import INGEST_STATUS_COLORS
from utils import CACHE_TTL_SECONDS

from spyfish.config.base import (
    CitSciStatus,
    ExpertStatus,
    IngestStatus,
    MlStatus,
    ReportingStatus,
    VideoPresence,
)
from spyfish.config.wrapper import config

from ..charting import style
from ..layout import section


def _stage_flags(dep: pd.DataFrame) -> pd.DataFrame:
    """Per-deployment booleans for each stage, measured by data presence.

    Presence of annotations rather than the status columns. A drop carrying ML
    annotations must have got through ML whatever its status column says, and
    section statuses drift from the annotation data. Ingest problems found after
    the fact do not unwind progress already made.
    """
    out = dep.copy()
    out["ok"] = out["ingest_status"] == IngestStatus.OK
    out["has_video"] = out["video_presence"] == VideoPresence.PRESENT
    for source in ("ml", "citsci", "expert"):
        out[f"{source}_done"] = out[f"{source}_annotations"].fillna(0) > 0
    return out


def render_funnel(df: pd.DataFrame, compact: bool = False) -> None:
    """From every deployment held down to expert annotation.

    Shared with the reporting home page so both draw the same chart from the
    same code. `compact` drops the captions for the home page, where the funnel
    sits beside other charts rather than being the subject.

    Each step is a subset of the one above, enforced, not implied, which is
    why the three annotation tiers are intersected with "Video present" rather
    than counted over all deployments. Without the intersection a deployment
    annotated and later archived counts in "ML annotated" but not in "Video
    present", and a lower tier can overtake the one above it. "Video present"
    is a proxy for "fully in play" (the strict bound is "not a bad
    deployment"), and the annotated-then-archived work it hides is real, so
    the caption below reports it instead of letting it silently vanish.

    The first two tiers are losses rather than progress, and they are separate
    steps because they are separate problems: a bad deployment went wrong in
    the field, while an ingest issue is a metadata problem that can be fixed
    to put the deployment back in play.
    """
    section("Pipeline funnel")
    if not compact:
        st.caption(
            "Stages are counted by data presence, not by status column, so a "
            "drop carrying annotations from a source must have got through the "
            "earlier steps for that source whatever its status says."
        )

    bad = df["is_bad_deployment"].fillna(0).astype(bool)
    usable = ~bad
    ingested = usable & df["ok"]
    with_video = ingested & df["has_video"]

    stages = pd.DataFrame(
        {
            "Stage": [
                "All deployments",
                "Not a bad deployment",
                "Passed ingest",
                "Video present",
                "ML annotated",
                "CitSci annotated",
                "Expert annotated",
            ],
            "Count": [
                len(df),
                int(usable.sum()),
                int(ingested.sum()),
                int(with_video.sum()),
                int((with_video & df["ml_done"]).sum()),
                int((with_video & df["citsci_done"]).sum()),
                int((with_video & df["expert_done"]).sum()),
            ],
        }
    )
    fig = go.Figure(
        go.Funnel(
            y=stages["Stage"],
            x=stages["Count"],
            textinfo="value+percent initial",
            marker_color=[
                "#90A4AE",
                "#D9603B",
                "#E8A33D",
                "#42A5F5",
                "#009E73",
                "#E69F00",
                "#0072B2",
            ],
        )
    )
    style(fig, height=380 if compact else 440)
    st.plotly_chart(fig, key=f"funnel_{'home' if compact else 'pipeline'}")

    if not compact:
        st.caption(
            f"**{int(bad.sum()):,}** deployments are flagged bad and cannot be "
            f"recovered. A further **{int((usable & ~df['ok']).sum()):,}** fail "
            f"ingest on their metadata and are recoverable by fixing the "
            f"record. Of what remains, "
            f"**{int((ingested & ~df['has_video']).sum()):,}** have no footage "
            f"in S3."
        )
        annotated = df["ml_done"] | df["citsci_done"] | df["expert_done"]
        outside = int((annotated & ~with_video).sum())
        if outside:
            st.caption(
                f"**{outside:,} annotated deployments are not counted in the "
                "annotation tiers**: their video has since been archived or "
                "removed, or their record failed ingest. The work exists, "
                "the funnel only counts deployments still fully in play. The "
                "Deployments view breaks these out."
            )


# How a year's bar can be split. One chart, two questions:
#
# * `usable_bad`, how much of a year's effort came through usable. Two
#   colours, for Reporting, where the failure mode is not the point.
# * `ingest_status`, the same bars with the lost share broken out by what went
#   wrong, for Operations, where someone has to decide whether a loss is
#   fixable. A validation error can be corrected upstream; an excluded
#   deployment cannot. Lifted from the archived Programme Health page.
#
# Each entry carries everything that differs between the two, so the drawing
# code below has no branches in it and the bars, totals and hover text cannot
# drift apart between the two pages.
#
# `min_label_pct` is the share below which a segment's label is dropped: there
# is no room to print inside a sliver, and the total above the bar plus the
# hover still carry the number. The two-way split keeps every label, since with
# two segments there is nothing to crowd.
SPLITS = {
    "usable_bad": {
        "title": "Deployments per year",
        "column": lambda df: df["is_bad_deployment"]
        .fillna(0)
        .astype(bool)
        .map({False: "ok", True: "bad"}),
        "order": ["ok", "bad"],
        "labels": {"ok": "Usable", "bad": "Bad deployment"},
        "colors": {"ok": "#2E8B57", "bad": "#D9603B"},
        "min_label_pct": 0,
    },
    "ingest_status": {
        "title": "Deployments per year, by ingest status",
        "column": lambda df: df["ingest_status"].fillna("removed"),
        # Usable at the bottom, then by how fixable the loss is: a validation
        # or metadata error sits directly on the usable share because it can be
        # corrected upstream and the deployment put back in play. An excluded
        # deployment went wrong in the field and never comes back, so it caps
        # the bar.
        "order": ["ok", "validation_error", "metadata_error", "excluded", "removed"],
        "labels": {
            "ok": "Usable",
            "excluded": "Bad deployment (excluded)",
            "validation_error": "Validation error",
            "metadata_error": "Metadata error",
            "removed": "Removed",
        },
        # The shared status palette, so a status is the same colour wherever it
        # appears in the app.
        "colors": INGEST_STATUS_COLORS,
        "min_label_pct": 5,
    },
}


def render_deployments_per_year(
    dated: pd.DataFrame, key: str, split: str = "usable_bad"
) -> None:
    """Deployments per year, stacked by whichever split `split` names.

    Stacked rather than several charts: the question is what share of a year's
    effort came through usable, and a share is only readable when the parts sit
    in the same bar. Segment labels carry the share and the total sits above, so
    the chart can be read without hovering.

    See `SPLITS` for what each split shows and why there are two.
    """
    spec = SPLITS[split]
    st.markdown(f"**{spec['title']}**")

    per_year = (
        dated.assign(_split=spec["column"](dated))
        .groupby([dated["survey_year"].astype(int), "_split"])["drop_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    # A category absent from the data still gets a column, so the legend does
    # not change shape as the filters move.
    for category in spec["order"]:
        if category not in per_year.columns:
            per_year[category] = 0
    per_year = per_year[spec["order"]]
    totals = per_year.sum(axis=1)
    years = per_year.index.tolist()

    fig = go.Figure()
    # Share on the segments, count only on the total above the bar. The counts
    # duplicated what the total already says, and two lines of text in a thin
    # segment pushed the labels out of their own bar.
    for category in spec["order"]:
        counts = per_year[category]
        if counts.sum() == 0:
            continue
        pct = (counts / totals * 100).round(0)
        label = spec["labels"][category]
        fig.add_bar(
            x=years,
            y=counts,
            name=label,
            marker_color=spec["colors"][category],
            # Blank rather than 0 where there are none, so the years that went
            # fine are not littered with zeros.
            text=[
                f"{v:.0f}%" if n and v >= spec["min_label_pct"] else ""
                for n, v in zip(counts, pct)
            ],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white"),
            customdata=pct,
            hovertemplate=f"%{{x}}<br>{label}: "
            "%{y} (%{customdata:.0f}%)<extra></extra>",
        )
    for year, total in zip(years, totals):
        fig.add_annotation(
            x=year,
            y=total,
            text=f"<b>{total}</b>",
            showarrow=False,
            yshift=11,
        )
    style(
        fig,
        height=270,
        barmode="stack",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        legend=dict(orientation="h", y=1.12, x=0, title_text="", traceorder="normal"),
    )
    st.plotly_chart(fig, key=key)


# Section status columns in pipeline order, each with its statuses in the order
# the state machine moves through them: the queue first, then the end states.
#
# These strings are internal schema, not config, the state-machine logic
# depends on the exact values, so they live in code beside the status classes
# they mirror. Taken from `spyfish.config.base` rather than typed out, so a
# renamed status is a change in one place.
SECTION_STATUS_ORDER = [
    (
        "ingest_status",
        "Ingest",
        [
            IngestStatus.OK,
            IngestStatus.VALIDATION_ERROR,
            IngestStatus.METADATA_ERROR,
            IngestStatus.EXCLUDED,
            IngestStatus.REMOVED,
        ],
    ),
    (
        "ml_status",
        "ML",
        [
            MlStatus.PENDING,
            MlStatus.READY,
            MlStatus.RUNNING,
            MlStatus.COMPLETE,
            MlStatus.ERROR,
            MlStatus.SKIPPED,
        ],
    ),
    (
        "citsci_status",
        "CitSci",
        [
            CitSciStatus.PENDING,
            CitSciStatus.CLIPS_UPLOADED,
            CitSciStatus.COMPLETE,
            CitSciStatus.ERROR,
            CitSciStatus.SKIPPED,
        ],
    ),
    (
        "expert_status",
        "Expert",
        [
            ExpertStatus.PENDING,
            ExpertStatus.UPLOADED,
            ExpertStatus.COMPLETE,
            ExpertStatus.ERROR,
            ExpertStatus.SKIPPED,
        ],
    ),
    (
        "reporting_status",
        "Reporting",
        [
            ReportingStatus.PENDING,
            ReportingStatus.COMPLETE,
            ReportingStatus.ERROR,
        ],
    ),
]

_PRESENCE_DISPLAY = {
    VideoPresence.PRESENT: "Present",
    VideoPresence.ARCHIVED: "Archived",
    VideoPresence.ABSENT: "Absent",
    VideoPresence.NO_VIDEO_BAD_DEP: "No video (bad dep.)",
}


def add_completion_flags(dep: pd.DataFrame) -> pd.DataFrame:
    """`complete`, `total_annotations` and `video_status`, as the page defines them.

    "Complete" means: expert or reporting complete, or every section explicitly
    skipped. Note this reads the **status columns**, unlike the rest of the
    report, which counts annotations, see the open question on the
    reporting-numbers panel.
    """
    out = dep.copy()
    out["complete"] = (
        (out["expert_status"] == ExpertStatus.COMPLETE)
        | (out["reporting_status"] == ReportingStatus.COMPLETE)
        | (
            (out["ml_status"] == MlStatus.SKIPPED)
            & (out["citsci_status"] == CitSciStatus.SKIPPED)
            & (out["expert_status"] == ExpertStatus.SKIPPED)
        )
    )
    counts = out[["expert_annotations", "ml_annotations", "citsci_annotations"]]
    out["total_annotations"] = counts.fillna(0).astype(int).sum(axis=1)
    out["video_status"] = out["video_presence"].map(_PRESENCE_DISPLAY).fillna("Unknown")
    out["needs_action"] = ~(
        out["complete"] | out["is_bad_deployment"].fillna(0).astype(bool)
    )
    return out


def render_section_progress(dep: pd.DataFrame) -> None:
    """Where deployments sit in each of the five section state machines.

    A table, as the Deployment Management page had it. It was tried as a stacked
    bar first: five sections with up to six statuses each is thirty segments,
    most of them slivers, and the labels do not fit, the shape said nothing the
    numbers did not say better.

    Rows are in conceptual order, not by size: sections in pipeline order, and
    statuses within a section in the order the state machine moves through them,
    so reading down a section is reading the queue from "not started" to "done".
    Sorting by count instead put `complete` next to `pending` whenever the two
    happened to be similar in size.
    """
    section("Section progress")
    st.caption(
        "Each pipeline section's deployments, by status. Read from the "
        "**status columns**, not from annotation counts, so this is the only "
        "place a deployment part-way through a stage, `ml_running`, "
        "`expert_uploaded`, is visible. Percentages are of all deployments in "
        "the current filter, so each section sums to 100%."
    )

    total = len(dep)
    rows = []
    for column, label, order in SECTION_STATUS_ORDER:
        if column not in dep.columns:
            continue
        counts = dep[column].value_counts()
        # Known statuses first, in state-machine order, then anything the code
        # does not know about, a status added upstream should show up as
        # unexpected rather than silently vanish from the table.
        statuses = order + [s for s in counts.index if s not in order]
        for status in statuses:
            count = int(counts.get(status, 0))
            if not count:
                continue
            rows.append(
                {
                    "Section": label,
                    "Status": status,
                    "Count": count,
                    "%": round(count / total * 100, 1) if total else 0,
                }
            )

    if not rows:
        st.info("No deployments to display yet.")
        return

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        height=min(560, 38 * len(rows) + 40),
        column_config={
            "Section": st.column_config.TextColumn("Section", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Count": st.column_config.NumberColumn("Deployments", width="small"),
            "%": st.column_config.ProgressColumn(
                "% of deployments",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )


def render_deployment_browser(dep: pd.DataFrame) -> None:
    """Filter down to specific deployments and see their status across sections."""
    section("Find a deployment")
    st.caption(
        "Status filters are on top of the year and MPA filters at the top of "
        "the page."
    )

    flagged = add_completion_flags(dep)

    s1, s2, s3 = st.columns(3)
    with s1:
        ml_filter = st.multiselect(
            "ML status",
            sorted(flagged["ml_status"].dropna().unique()),
            key="ops_dep_ml",
        )
    with s2:
        citsci_filter = st.multiselect(
            "CitSci status",
            sorted(flagged["citsci_status"].dropna().unique()),
            key="ops_dep_citsci",
        )
    with s3:
        expert_filter = st.multiselect(
            "Expert status",
            sorted(flagged["expert_status"].dropna().unique()),
            key="ops_dep_expert",
        )

    c1, c2 = st.columns(2)
    with c1:
        survey_filter = st.multiselect(
            "Survey",
            sorted(flagged["survey_id"].dropna().unique()),
            key="ops_dep_survey",
        )
    with c2:
        complete_filter = st.selectbox(
            "Completion",
            ["All", "Complete", "Action required"],
            key="ops_dep_complete",
        )

    filtered = flagged
    if ml_filter:
        filtered = filtered[filtered["ml_status"].isin(ml_filter)]
    if citsci_filter:
        filtered = filtered[filtered["citsci_status"].isin(citsci_filter)]
    if expert_filter:
        filtered = filtered[filtered["expert_status"].isin(expert_filter)]
    if survey_filter:
        filtered = filtered[filtered["survey_id"].isin(survey_filter)]
    if complete_filter == "Complete":
        filtered = filtered[filtered["complete"]]
    elif complete_filter == "Action required":
        filtered = filtered[~filtered["complete"]]

    st.markdown(f"**{len(filtered):,} deployments**")
    if filtered.empty:
        st.success("No deployments in this category.")
        return

    show_annotations = st.checkbox(
        "Show annotation counts", key="ops_dep_show_ann", value=False
    )
    columns = [
        "drop_id",
        "survey_id",
        "is_bad_deployment",
        "ml_status",
        "citsci_status",
        "expert_status",
        "video_status",
        "complete",
    ]
    if show_annotations:
        columns += ["ml_annotations", "citsci_annotations", "expert_annotations"]
    columns = [c for c in columns if c in filtered.columns]

    st.dataframe(
        filtered[columns],
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "drop_id": st.column_config.TextColumn("DropID"),
            "survey_id": st.column_config.TextColumn("SurveyID"),
            "complete": st.column_config.CheckboxColumn("Complete", width="small"),
            "is_bad_deployment": st.column_config.CheckboxColumn("Bad", width="small"),
            "video_status": st.column_config.TextColumn("Video", width="small"),
            "ml_status": st.column_config.TextColumn("ML", width="small"),
            "citsci_status": st.column_config.TextColumn("CitSci", width="small"),
            "expert_status": st.column_config.TextColumn("Biigle", width="small"),
            "ml_annotations": st.column_config.NumberColumn("ML", width="small"),
            "citsci_annotations": st.column_config.NumberColumn(
                "CitSci", width="small"
            ),
            "expert_annotations": st.column_config.NumberColumn(
                "Expert", width="small"
            ),
        },
    )


def render_annotation_detail(dep: pd.DataFrame) -> None:
    """Every annotation record held for one deployment, from any source."""
    section("One deployment")
    st.caption(
        "Straight from the annotations database, one row per species and "
        "source, not a summary."
    )

    from spyfish.database.annotation_manager import AnnotationDatabaseManager

    # Only deployments that hold annotations: listing all ~2,400 makes almost
    # every pick land on "no annotations found". The per-source counts are
    # maintained by `sync_annotation_counts`, the same signal the stage flags
    # read.
    annotated = dep[
        dep[["ml_annotations", "citsci_annotations", "expert_annotations"]]
        .fillna(0)
        .sum(axis=1)
        > 0
    ]
    selected = st.selectbox(
        "DropID",
        ["None"] + sorted(annotated["drop_id"].dropna().unique().tolist()),
        key="ops_dep_detail",
        help="Only deployments with at least one annotation are listed.",
    )
    if selected == "None":
        return

    maxn = AnnotationDatabaseManager().get_maxn_summary(drop_id=selected)
    if maxn.empty:
        st.info(f"No annotations found for {selected}.")
        return
    # A null species is an absence record — the source reviewed the footage
    # and saw nothing. Label it rather than showing an empty cell.
    maxn["scientific_name"] = maxn["scientific_name"].fillna("(nothing seen)")

    st.dataframe(
        maxn[
            [
                "scientific_name",
                "annotated_by",
                "maxn",
                "time_of_max",
                "confidence_agreement",
                "external_id",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "scientific_name": st.column_config.TextColumn("Species"),
            "annotated_by": st.column_config.TextColumn("Source", width="small"),
            "maxn": st.column_config.NumberColumn("Peak MaxN", width="small"),
            "time_of_max": st.column_config.TextColumn("Time of MaxN", width="small"),
            "confidence_agreement": st.column_config.NumberColumn(
                "Confidence", format="%.2f", width="small"
            ),
            "external_id": st.column_config.TextColumn(
                "Provenance",
                help=(
                    "Tracks which source produced each annotation. "
                    "ML rows: model name (e.g. 'species_20260429_081503'). "
                    "Expert rows: BIIGLE annotation_id, the unique ID of the "
                    "bbox in BIIGLE; visible in the BIIGLE UI on each "
                    "annotation's detail panel. "
                    "CitSci rows: empty (Zooniverse classifications aren't "
                    "individually addressable post-aggregation)."
                ),
            ),
        },
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def _all_annotations_csv() -> bytes | None:
    from spyfish.database.annotation_manager import AnnotationDatabaseManager

    df = AnnotationDatabaseManager().get_all_annotations_export_df()
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")


def render_exports(dep: pd.DataFrame) -> None:
    """The two CSV exports the team actually uses."""
    section("Export")

    flagged = add_completion_flags(dep)
    left, right = st.columns(2)
    with left:
        # Built from a SELECT * frame, so it MUST go through strip_sensitive()
        # or it hands out deployment coordinates to anyone who can load the
        # page.
        needs_action = config.strip_sensitive(flagged[~flagged["complete"]])
        st.download_button(
            "Download needs-action deployments",
            data=needs_action.to_csv(index=False).encode("utf-8"),
            file_name="deployments_needs_action.csv",
            mime="text/csv",
            help="Deployments not yet complete, with site coordinates removed.",
        )
    with right:
        annotations_csv = _all_annotations_csv()
        if annotations_csv:
            st.download_button(
                "Download all annotations",
                data=annotations_csv,
                file_name="spyfish_all_annotations.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.button(
                "Download all annotations",
                help="No annotations available yet.",
                disabled=True,
            )
