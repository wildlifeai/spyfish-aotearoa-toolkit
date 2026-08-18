"""Reporting - Annotations: what the annotations actually say.

The Operations side of Annotations asks how much has been annotated and whether
the sources agree with each other. This side asks what was recorded.

First occupant is the per-deployment lookup, moved off Report home: the front
page is a programme-level summary, and "what did we see at this one drop" is a
question about the annotations.
"""

import pandas as pd
import streamlit as st
from ecology_data import source_bucket

from .charts.annotations import render_arrival_and_peak
from .charts.deployments import render_annotation_detail
from .data import arrival_and_peak
from .layout import chips, section


def _reviewed_empty(ann: pd.DataFrame) -> pd.DataFrame:
    """One row per (deployment, source) whose finished review found nothing.

    The loaders map NULL_DEPLOYMENT to NaN, so a source's review was empty
    exactly when it has rows for the drop but none carries a species name. A
    source that never looked has no rows at all and stays out.
    """
    tagged = ann.assign(Source=source_bucket(ann["annotated_by"]))
    per = (
        tagged.groupby(["drop_id", "Source"])["scientific_name"]
        .agg(rows="size", named="count")
        .reset_index()
    )
    return per[(per["rows"] > 0) & (per["named"] == 0)]


def render(ctx: dict) -> None:
    st.caption(
        "What the annotations record. Coverage and source agreement are on the "
        "Operations side."
    )

    chips(["One deployment", "Arrival and MaxN time", "Reviewed, nothing seen"])

    dep = ctx["deployments"]
    if dep.empty:
        st.info("No deployments match the current filters.")
        return

    render_annotation_detail(dep)

    # Behaviour, not data state: when a species turns up and when it peaks is a
    # finding about the fish, so it belongs on the Reporting side. Reads the
    # UNFILTERED annotations because an arrival time can only come from ML,
    # which scores every 10-second interval, and would vanish the moment a
    # reader filtered to expert.
    st.divider()
    render_arrival_and_peak(arrival_and_peak(ctx["annotations_all_sources"]))

    # ── Reviewed, nothing seen ───────────────────────────────────────────────
    st.divider()
    section("Reviewed, nothing seen")
    st.caption(
        "Deployments where a source finished its review and recorded no "
        "animals at all. These are real zeros, the denominators every "
        "abundance figure divides by — not gaps: a source that never looked "
        "does not appear here. All sources are shown regardless of the Source "
        "filter, since an absence is only meaningful next to who else looked."
    )
    empty = _reviewed_empty(ctx["annotations_all_sources"])
    if empty.empty:
        st.info("No deployment in this selection has a reviewed-and-empty result.")
        return

    by_drop = (
        empty.groupby("drop_id")["Source"]
        .apply(lambda s: ", ".join(sorted(s)))
        .reset_index(name="Empty review by")
    )
    st.caption(
        f"**{len(by_drop):,} deployments** carry at least one empty review "
        f"({len(empty):,} source-reviews in total)."
    )
    meta_cols = [
        c
        for c in ("link_to_marine_reserve", "site_id", "survey_year")
        if c in dep.columns
    ]
    table = by_drop.merge(
        dep[["drop_id", *meta_cols]].drop_duplicates("drop_id"),
        on="drop_id",
        how="left",
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(400, 60 + 35 * len(table)),
        column_config={
            "drop_id": st.column_config.TextColumn("DropID", width="large"),
            "Empty review by": st.column_config.TextColumn(
                "Empty review by",
                help="Sources whose finished review of this deployment "
                "recorded nothing. Other sources may have seen animals — "
                "the Source disagreement panel on the Operations side flags "
                "those conflicts.",
            ),
            "link_to_marine_reserve": st.column_config.TextColumn("Marine reserve"),
            "site_id": st.column_config.TextColumn("SiteID", width="small"),
            "survey_year": st.column_config.NumberColumn(
                "Year", format="%d", width="small"
            ),
        },
    )
