"""Reporting - Annotations: what the annotations actually say.

The Operations side of Annotations asks how much has been annotated and whether
the sources agree with each other. This side asks what was recorded.

First occupant is the per-deployment lookup, moved off Report home: the front
page is a programme-level summary, and "what did we see at this one drop" is a
question about the annotations.
"""

import streamlit as st

from .charts.annotations import render_arrival_and_peak
from .charts.deployments import render_annotation_detail
from .data import arrival_and_peak
from .layout import chips


def render(ctx: dict) -> None:
    st.caption(
        "What the annotations record. Coverage and source agreement are on the "
        "Operations side."
    )

    chips(["One deployment", "Arrival and MaxN time"])

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
