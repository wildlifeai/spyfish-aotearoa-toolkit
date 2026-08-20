"""Operations → Metadata: the ranger-entry data quality review.

The review implementation lives in ``error_review.py`` beside this module.
"""

import streamlit as st

from spyfish.config.base import VideoPresence

from .error_review import render_body
from .layout import section


def render(ctx: dict) -> None:
    st.caption(
        "Validation errors raised at ingest, straight from the pipeline "
        "database. These are metadata problems in the SharePoint entry, a "
        "deployment with errors here never reaches the processing stages."
    )
    _render_unreadable_dates(ctx)
    render_body()

    st.divider()
    _render_missing_footage(ctx)


def _render_missing_footage(ctx: dict) -> None:
    """Deployments whose footage never arrived in S3.

    Moved here from the Deployments view. It is the same kind of problem as the
    errors above, something upstream did not deliver what the record says it
    should, and it is chased the same way, at the source rather than anywhere
    in the pipeline.
    """
    section("Missing footage")
    dep = ctx["deployments"]
    missing = dep[dep["video_presence"] == VideoPresence.ABSENT]
    if missing.empty:
        st.success("No deployments are missing footage in this selection.")
        return

    st.caption(
        f"{len(missing):,} deployments expect footage that is not in S3. "
        "Distinct from archived footage, which exists but needs a restore."
    )
    st.dataframe(
        missing[
            ["drop_id", "survey_id", "site_id", "region", "ingest_status"]
        ].sort_values("survey_id"),
        hide_index=True,
        width="stretch",
        height=320,
    )
    st.download_button(
        "Download missing footage list (CSV)",
        missing[["drop_id", "survey_id", "site_id", "region"]]
        .to_csv(index=False)
        .encode(),
        file_name="spyfish_missing_footage.csv",
        mime="text/csv",
    )


def _render_unreadable_dates(ctx: dict) -> None:
    """DropIDs whose date does not parse, which the validator does not catch.

    The DropID pattern in `config.yaml` now constrains the date segment to a
    real calendar date in 20xx, so a DDMMYYYY date (`RTT_23042026_...`, which
    reads as the year 2304) fails validation. Deployments ingested before that
    pattern was tightened still carry `ingest_status = 'ok'` and will not be
    re-checked until the next `--ingest` run, so they are listed here: they drop
    out of every per-year chart in the report without appearing in the error
    table below.
    """
    dep = ctx["all_deployments"]
    undated = dep[dep["survey_year"].isna()]
    if undated.empty:
        return

    st.warning(
        f"**{len(undated)} deployment(s) have a date that cannot be read.** The "
        "DropID validation pattern rejects these now, but these rows were "
        "ingested before it was tightened, so they still read as `ok` and are "
        "not in the table below. They are missing from every per-year chart in "
        "the report until the DropID is corrected upstream and re-ingested."
    )
    with st.expander(f"Show the {len(undated)} DropIDs"):
        st.dataframe(
            undated[["drop_id", "survey_id", "ingest_status", "video_presence"]],
            hide_index=True,
            width="stretch",
        )
