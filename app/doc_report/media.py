"""Media view for the DOC reporting page.

Where the footage is. Every downstream stage needs a video in S3, so this
answers "what can we still process?" before anyone asks why a survey has no
annotations.

`video_presence` is its own column, separate from the pipeline statuses. It
tracks S3 file state, not progress.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import NEUTRAL

from .charting import style
from .charts.deployments import render_deployment_browser, render_exports
from .layout import chips, section

# Best to worst, and fixed so both charts stack the same way. "absent" sits
# above "present" because it is the actionable state, and below
# "no_video_bad_dep" because that one is not a footage problem at all.
STATE_ORDER = ["present", "archived", "absent", "no_video_bad_dep", "unknown"]

# Archived splits by whether the deployment is already annotated, because the
# two mean different things operationally: one is a restore backlog, the other
# is footage that can stay in cold storage because the work is done.
DETAIL_ORDER = [
    "Present",
    "Archived, annotated",
    "Archived, not annotated",
    "Absent",
    "No footage expected",
    "Unknown",
]
DETAIL_COLOURS = {
    "Present": "#26A69A",
    "Archived, annotated": "#7FB77E",
    "Archived, not annotated": "#FF9800",
    "Absent": "#EF5350",
    "No footage expected": NEUTRAL,
    "Unknown": "#BDBDBD",
}


def _detail_state(presence: pd.Series, annotated: pd.Series) -> pd.Series:
    """Footage state, with archived split by whether annotations already exist."""
    labels = presence.map(
        {
            "present": "Present",
            "absent": "Absent",
            "no_video_bad_dep": "No footage expected",
        }
    )
    archived = presence == "archived"
    labels = labels.mask(archived & annotated, "Archived, annotated")
    labels = labels.mask(archived & ~annotated, "Archived, not annotated")
    return labels.fillna("Unknown")


PRESENCE_MEANING = {
    "present": "In S3 standard storage. No restore needed.",
    "archived": "In cold storage. Retrievable, but not immediately.",
    "absent": "Expected but not found. This is the actionable state.",
    "no_video_bad_dep": "No footage expected. The deployment itself was bad.",
}


def render(ctx: dict) -> None:
    """Render the Media view from the shared context."""
    dep = ctx["deployments"]
    if dep.empty:
        st.warning("No deployments match the current filters.")
        return

    chips(["Footage funnel", "Coverage by year", "Find a deployment", "Export"])

    presence = dep["video_presence"].fillna("unknown")
    counts = presence.value_counts()

    present = int(counts.get("present", 0))
    archived = int(counts.get("archived", 0))
    absent = int(counts.get("absent", 0))
    no_video = int(counts.get("no_video_bad_dep", 0))
    # Deployments that never had footage are not a coverage gap, they are a
    # deployment that went wrong. Counting them in the denominator understates
    # coverage.
    expected = len(dep) - no_video

    kpis = st.columns(4)
    kpis[0].metric("Deployments", f"{len(dep):,}")
    kpis[1].metric(
        "In standard storage",
        f"{present / expected:.0%}" if expected else "—",
        help=f"Footage in S3 standard storage, so it can be processed without "
        f"a restore: {present:,} of {expected:,} "
        f"deployments that should have footage. The {no_video:,} recorded "
        f"as no-video-bad-deployment are excluded from the denominator, "
        f"because no footage was ever expected for them.",
    )
    kpis[2].metric(
        "Accessible",
        f"{(present + archived) / expected:.0%}" if expected else "—",
        help=f"Footage that exists at all, readable now or not: {present:,} "
        f"present plus {archived:,} archived, over {expected:,} expected. "
        f"Archived footage is in cold storage. It is recoverable with a "
        f"restore, but cannot be processed until then.",
    )
    kpis[3].metric(
        "Missing",
        f"{absent:,}",
        help="Footage expected but not found in S3. Unlike archived, a restore "
        "will not bring it back. These are the deployments blocking their "
        "surveys.",
    )

    st.caption(
        "`video_presence` records S3 file state, not pipeline progress. A "
        "deployment can be fully ingested and still have no footage, which is "
        "why it is tracked separately."
    )

    st.divider()

    left, right = st.columns([1, 1.4])

    with left:
        section(
            "Footage funnel",
            help="  \n".join(
                f"**{state}**, {meaning}" for state, meaning in PRESENCE_MEANING.items()
            ),
        )
        st.caption(
            "Each step is a subset of the one above, so the drops are the "
            "losses. Deployments that never had footage are removed first: "
            "they are not a coverage gap, they are a deployment that went wrong."
        )

        # Expert only, not "any source". The question this funnel answers is
        # what can be reported onwards, and only expert annotation is final:
        # expert wins wherever sources disagree, so a deployment carrying only
        # ML or citsci output is still waiting for review. It also means an
        # archived deployment with only ML output still needs restoring.
        anno = dep["expert_annotations"].fillna(0) > 0
        expected_mask = presence != "no_video_bad_dep"
        exists_mask = presence.isin(["present", "archived"])
        ready_mask = presence == "present"

        # Losses first: what has no footage to talk about. Three tiers, each a
        # strict subset, so the drops are the two ways footage goes missing.
        stages = pd.DataFrame(
            {
                "Stage": ["All deployments", "Footage expected", "Footage exists"],
                "Count": [len(dep), int(expected_mask.sum()), int(exists_mask.sum())],
            }
        )
        fig = go.Figure(
            go.Funnel(
                y=stages["Stage"],
                x=stages["Count"],
                textinfo="value+percent initial",
                marker_color=["#90A4AE", "#42A5F5", "#26A69A"],
            )
        )
        style(fig, height=230)
        st.plotly_chart(fig, key="media_funnel")
        st.caption(
            f"**{no_video:,}** never had footage. **{absent:,}** are missing "
            f"and a restore will not bring them back."
        )

        # Everything with footage is in exactly ONE of four states: annotated or
        # not, accessible now or archived. That is a single classification, not
        # a chain, so it is a bar rather than more funnel tiers, a funnel can
        # only narrow, and these four do not nest.
        arch = presence == "archived"
        with_footage = int(exists_mask.sum())
        # Furthest from done to done: archived and unannotated needs a restore
        # AND review, standard-but-unannotated needs only review, annotated in
        # standard storage is finished, and annotated-archived is finished and
        # can stay in cold storage. Reading left to right is work remaining.
        #
        # "Standard storage" rather than "ready now": it names where the footage
        # IS, which is the same kind of fact as "archived". "Ready" described
        # what could be done with it, mixing two ideas in one axis.
        states = [
            ("Not annotated, archived", arch & ~anno, "#D9603B"),
            ("Not annotated, standard storage", ready_mask & ~anno, "#FF9800"),
            ("Annotated, standard storage", ready_mask & anno, "#0072B2"),
            ("Annotated, archived", arch & anno, "#7FB77E"),
        ]
        composition = pd.DataFrame(
            {
                "State": [name for name, _, _ in states],
                "Deployments": [int(mask.sum()) for _, mask, _ in states],
            }
        )
        composition["Share"] = composition["Deployments"] / max(with_footage, 1)

        st.markdown(f"**The {with_footage:,} deployments that have footage**")
        fig = go.Figure()
        for (name, _, colour), row in zip(states, composition.itertuples()):
            fig.add_bar(
                x=[row.Deployments],
                y=["Footage"],
                orientation="h",
                name=name,
                marker_color=colour,
                text=[f"{row.Deployments:,}<br>{row.Share:.0%}"],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white"),
                hovertemplate=f"{name}<br>%{{x:,}} deployments<extra></extra>",
            )
        style(
            fig,
            barmode="stack",
            height=200,
            uniformtext_minsize=10,
            uniformtext_mode="hide",
            # Plotly reverses legend order on stacked charts by default, so
            # the key read backwards against the bar it explains.
            legend=dict(
                orientation="h", y=-0.35, x=0, title_text="", traceorder="normal"
            ),
        )
        fig.update_yaxes(visible=False)
        fig.update_xaxes(visible=False)
        st.plotly_chart(fig, key="media_states")
        st.caption(
            "**Annotated** means expert annotated, the reportable end state: "
            "expert wins wherever sources disagree, so ML or citsci output "
            "alone is still awaiting review. **Archived** footage is in cold "
            "storage, so it needs restoring before anyone can work on it, "
            "unless it is already annotated, in which case it can stay there."
        )

    with right:
        section(
            "Coverage by year",
            help="Same categories as the funnel, with archived split by "
            "whether the deployment already carries annotations.",
        )
        dated = dep[dep["survey_year"].notna()].copy()
        if dated.empty:
            st.info("No deployments have a parseable date in their DropID.")
        else:
            dated_anno = (
                dated[["ml_annotations", "citsci_annotations", "expert_annotations"]]
                .fillna(0)
                .gt(0)
                .any(axis=1)
            )
            dated["State"] = _detail_state(
                dated["video_presence"].fillna("unknown"), dated_anno
            )
            by_year = (
                dated.groupby([dated["survey_year"].astype(int), "State"])
                .size()
                .reset_index(name="Deployments")
                .rename(columns={"survey_year": "Year"})
            )
            fig = px.bar(
                by_year,
                x="Year",
                y="Deployments",
                color="State",
                color_discrete_map=DETAIL_COLOURS,
                category_orders={"State": DETAIL_ORDER},
            )
            style(
                fig,
                height=340,
                legend=dict(orientation="h", y=1.12, x=0, title_text=""),
            )
            st.plotly_chart(fig, key="media_year")

    st.divider()

    # Copied from the Deployment Management page: this view is about
    # deployments, and these three answer "which one", "what does it hold" and
    # "give me the list".
    render_deployment_browser(dep)
    st.divider()
    render_exports(dep)
