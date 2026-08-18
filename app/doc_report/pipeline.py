"""Pipeline view for the DOC reporting page.

How far deployments have got through processing, and which surveys are losing
deployments on the way.

This is the Operations landing view. It carries the pipeline funnel, which
Report home also draws, from `render_funnel` here, so the two can never
disagree about how many deployments got where, alongside the per-year split by
ingest status.

The metric row deliberately holds only what the funnel does NOT answer. Every
tier of the funnel is a count, so repeating those above it would be the same
number twice on one screen.

Ported from the Programme Health page. The panels that page carried which are
already covered elsewhere in the report were not brought across:

* video presence by year -> Deployments view
* survey summary table   -> Surveys view
* protection status      -> MPA view, where protection belongs
* annotation depth       -> Surveys view, under the table reporting the same
  thing
"""

import streamlit as st

from spyfish.config.base import VideoPresence

from .charts.deployments import (
    _stage_flags,
    add_completion_flags,
    render_deployments_per_year,
    render_funnel,
    render_section_progress,
)
from .layout import chips


def render(ctx: dict) -> None:
    """Render the Pipeline view from the shared context."""
    dep = ctx["deployments"]
    if dep.empty:
        st.warning("No deployments match the current filters.")
        return

    chips(["Pipeline funnel", "Section progress"])

    df = _stage_flags(dep)
    presence = df["video_presence"]

    # Only what the funnel below does NOT already answer. The funnel counts all
    # deployments, the not-bad and ingested subsets, video present, and each
    # annotation source, so none of those are repeated here. What is left is the
    # programme's shape (surveys, sites) and the footage states the funnel has
    # no tier for.
    kpis = st.columns(5)
    kpis[0].metric("Surveys", f"{df['survey_id'].nunique():,}")
    kpis[1].metric("Sites", f"{df['site_id'].nunique():,}")
    kpis[2].metric(
        "Video archived",
        f"{int((presence == VideoPresence.ARCHIVED).sum()):,}",
        help="In Glacier. Retrievable, but not without a restore step first, "
        "which is why these do not count as present.",
    )
    kpis[3].metric(
        "No video",
        f"{int(presence.isin([VideoPresence.ABSENT, VideoPresence.NO_VIDEO_BAD_DEP]).sum()):,}",
        help="Nothing in S3: either the footage never arrived, or the "
        "deployment was bad and there is nothing to look for.",
    )
    kpis[4].metric(
        "Not annotated",
        f"{int((~(df['ml_done'] | df['citsci_done'] | df['expert_done'])).sum()):,}",
        help="No annotation from any source yet. Includes deployments that "
        "never can be, see the first two steps of the funnel.",
    )

    # The Deployment Management page's headline row, brought over. "Complete"
    # and "Action required" read the section STATUS columns, unlike everything
    # else here, which counts annotations, so they answer "what does the
    # pipeline think" rather than "what data exists", and the two can disagree.
    flagged = add_completion_flags(dep)
    kpis = st.columns(4)
    kpis[0].metric("Total", f"{len(dep):,}")
    kpis[1].metric(
        "Action required",
        f"{int((~flagged['complete']).sum()):,}",
        help="Not yet complete by the status columns. Includes deployments "
        "that never can be completed, bad or excluded ones.",
    )
    kpis[2].metric(
        "Videos",
        f"{int((presence == VideoPresence.PRESENT).sum()):,}",
        help="Footage present in S3, ready to process without a restore.",
    )
    kpis[3].metric(
        "Complete",
        f"{int(flagged['complete'].sum()):,}",
        help="Expert or reporting status is complete, or every section was "
        "explicitly skipped. Read from the status columns, not from "
        "annotation counts.",
    )

    st.divider()

    dated = dep[dep["survey_year"].notna()]
    left, right = st.columns(2)
    with left:
        # Report home shows the same bars split usable/bad. Here the lost share
        # is broken out by ingest status, because Operations is where someone
        # decides whether a loss is fixable, a validation error can be
        # corrected upstream, an excluded deployment cannot.
        if dated.empty:
            st.info("No deployments have a parseable date in their DropID.")
        else:
            render_deployments_per_year(
                dated, key="ops_per_year_status", split="ingest_status"
            )
    with right:
        # The same funnel Report home draws, from the same function, so the two
        # cannot disagree about how many deployments got where. Not compact
        # here: this is the page where the caption explaining what is
        # recoverable and what is not belongs.
        render_funnel(df)

    st.divider()

    # Copied from the Deployment Management page. The funnel above counts
    # annotations; this counts the section status columns, so it is the only
    # view where a deployment part-way through a stage, ml_running,
    # expert_uploaded, is visible at all.
    render_section_progress(dep)
