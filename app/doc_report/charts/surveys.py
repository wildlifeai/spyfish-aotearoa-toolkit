"""Per-survey charts: how far each survey has been annotated, and what it lost.

Drawn side by side on Operations - Surveys. Both answer "which surveys need
attention", one from the annotation end and one from the field end.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from spyfish.config.base import IngestStatus

from ..charting import style
from ..layout import section

# One definition of "done", shared with the funnel, so the same deployment
# cannot be annotated in one chart and pending in the other.
from .deployments import _stage_flags


def render_bad_per_survey(dep: pd.DataFrame) -> None:
    """Share of each survey's deployments that went wrong in the field.

    The archived Programme Health chart, as it was: bar height is the share,
    colour depth is how many deployments that share represents. The two
    encodings answer different halves of the question, a survey at 50% matters
    much more when it is 50% of forty than 50% of two, and colour keeps the
    second one on the same bar rather than in a second chart.

    Surveys are grouped by MPA and then ordered by date within it, so a reserve
    that keeps losing deployments shows up as a run of tall bars rather than
    being scattered across the axis. Surveys with no bad deployments are left
    out: they are the majority and would push the interesting bars into a thin
    strip.
    """
    section("Bad deployments")
    st.caption(
        "Bad deployments went wrong in the field and are not recoverable. "
        "Height is the share of the survey lost; colour depth is the count "
        "behind that share."
    )

    per_survey = (
        dep.assign(_bad=dep["is_bad_deployment"].fillna(0).astype(bool))
        .groupby("survey_id")
        .agg(total=("drop_id", "nunique"), bad=("_bad", "sum"))
        .reset_index()
    )
    per_survey["bad_pct"] = (per_survey["bad"] / per_survey["total"] * 100).round(1)
    bad_surveys = per_survey[per_survey["bad"] > 0].copy()

    if bad_surveys.empty:
        st.success("No bad deployments recorded in the current selection.")
        return

    # SurveyID is RESERVE_YYYYMMDD_BUV, so both the reserve and the date come
    # out of it without another join.
    sid_parts = bad_surveys["survey_id"].str.split("_", expand=True)
    bad_surveys["reserve"] = sid_parts[0]
    bad_surveys["survey_date"] = pd.to_datetime(
        sid_parts[1], format="%Y%m%d", errors="coerce"
    )
    bad_surveys = bad_surveys.sort_values(["reserve", "survey_date"])

    fig = px.bar(
        bad_surveys,
        x="survey_id",
        y="bad_pct",
        color="bad",
        color_continuous_scale="Oranges",
        text="bad_pct",
        hover_data={
            "total": True,
            "bad": True,
            "survey_date": "|%Y-%m-%d",
            "reserve": True,
            "survey_id": False,
        },
        labels={
            "survey_id": "Survey",
            "bad_pct": "% bad deployments",
            "bad": "Count",
            "reserve": "Reserve",
        },
        category_orders={"survey_id": bad_surveys["survey_id"].tolist()},
        height=340,
    )
    fig.update_traces(
        textposition="outside", texttemplate="%{text:.1f}%", cliponaxis=False
    )
    # Surveys above this share warrant a closer look.
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="#EF5350",
        line_width=1,
        annotation_text="10% threshold",
        annotation_position="right",
        annotation_font_color="#EF5350",
    )
    style(
        fig,
        xaxis_tickangle=-40,
        coloraxis_colorbar_title="Bad count",
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="% bad deployments", ticksuffix="%")
    st.plotly_chart(fig, key="surveys_bad_per_survey")


# Bands run done-to-do inside a single 100%-wide bar: annotation first, then
# outstanding work. One bar totalling 100% is what makes "this survey is
# finished" readable at a glance, which a diverging layout loses.
# "CitSci", not "ML + CitSci": the band means "citsci annotated it" and is
# assigned on `citsci_done` alone. Legacy Zooniverse backfills exist with no ML
# pass at all, so naming the band after ML would claim work that never ran.
BAND_LABELS = {
    "expert": "Expert",
    "citsci": "CitSci",
    "ml": "ML only",
    "pending": "Not annotated yet",
    "bad": "Bad deployment",
    "ingest": "Ingest issue",
}


BAND_COLOURS = {
    "Expert": "#0072B2",
    "CitSci": "#E69F00",
    "ML only": "#009E73",
    "Not annotated yet": "#E0E0E0",
    "Bad deployment": "#8C2F1F",
    "Ingest issue": "#E08E7B",
}


# Left to right: work done, then work outstanding, ending with the least
# recoverable. An ingest issue can be fixed and the deployment put back in play;
# a bad deployment cannot, so it sits at the far end.
BAND_ORDER = [
    "Expert",
    "CitSci",
    "ML only",
    "Not annotated yet",
    "Ingest issue",
    "Bad deployment",
]


def render_annotation_depth(dep: pd.DataFrame) -> None:
    """One 100%-wide bar per survey: work done left, work outstanding right."""
    section("Annotation depth")
    st.caption(
        "One bar per survey, always 100% wide: **annotation first, then the "
        "work still outstanding.** Each deployment lands in exactly one band, "
        "so the bands are shares of the same survey and cannot double-count. A "
        "survey is finished when the bar is entirely annotation colours."
    )
    st.caption(
        "The two problem bands are different jobs. A **bad deployment** went "
        "wrong in the field and is not recoverable by fixing data. An **ingest "
        "issue** is a metadata problem: the footage may be fine, and fixing the "
        "record puts the deployment back in play. Neither can be annotated "
        "until resolved, which is why they are not left as plain backlog."
    )

    df = _stage_flags(dep)
    bad = df["is_bad_deployment"].fillna(0).astype(bool)

    # Exclusive, applied in order so the last assignment wins. Annotations are
    # applied last on purpose: 23 deployments here were annotated and only later
    # flagged bad or excluded, and letting the problem bands win would erase
    # that finished work from the chart.
    # `excluded` and `is_bad_deployment` are the same 353 deployments here, so
    # the ingest status is what separates the two problems: `excluded` means the
    # deployment itself was bad, `validation_error` means its record is.
    df["band"] = "pending"
    df.loc[df["ingest_status"] == IngestStatus.VALIDATION_ERROR, "band"] = "ingest"
    df.loc[bad | (df["ingest_status"] == IngestStatus.EXCLUDED), "band"] = "bad"
    df.loc[df["ml_done"], "band"] = "ml"
    df.loc[df["citsci_done"], "band"] = "citsci"
    df.loc[df["expert_done"], "band"] = "expert"

    counts = df.groupby(["survey_id", "band"]).size().reset_index(name="count")
    totals = df.groupby("survey_id").size().rename("Deployments")
    counts = counts.merge(totals, on="survey_id")
    counts["pct"] = (counts["count"] / counts["Deployments"] * 100).round(1)
    counts["Band"] = counts["band"].map(BAND_LABELS)
    # Most recent survey at the top, alphabetical within a year.
    #
    # Chronological beats sorting by completeness here: a reader scanning for
    # "how are we doing lately" wants the newest surveys first, and the
    # completeness is already visible in the bar itself.
    #
    # Plotly stacks categories bottom-to-top, so the array runs oldest first,
    # and the within-year order is reversed so it reads A to Z downwards.
    # Newest survey at the top, alphabetical within a date. Sorted on the full
    # survey date rather than the year so two surveys in the same year still
    # order.
    #
    # The array is newest-first: for a horizontal px.bar, category_orders maps
    # to the y-axis top-down, not bottom-up. Verified in the browser rather than
    # assumed, because it is the opposite of what a stacked bar chart does.
    order = pd.DataFrame({"survey_id": totals.index})
    order["date"] = pd.to_datetime(
        order["survey_id"].str.split("_").str[1], format="%Y%m%d", errors="coerce"
    )
    order = order.sort_values(
        ["date", "survey_id"], ascending=[False, True], na_position="last"
    )

    fig = px.bar(
        counts,
        y="survey_id",
        x="pct",
        color="Band",
        orientation="h",
        hover_data={"count": True, "Deployments": True},
        category_orders={"Band": BAND_ORDER, "survey_id": list(order["survey_id"])},
        color_discrete_map=BAND_COLOURS,
    )
    style(
        fig,
        barmode="stack",
        height=max(340, 22 * len(totals)),
        xaxis_range=[0, 100],
        legend=dict(orientation="h", y=1.02, x=0, title_text=""),
        bargap=0.2,
    )
    fig.update_xaxes(
        title="% of deployments, annotation first, outstanding work after",
        ticksuffix="%",
    )
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="surveys_depth")
