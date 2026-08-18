from .charting import style

"""The metadata error review: validation errors from the pipeline database.

This was the standalone Error Review page (``pages/🔍_Error_Review.py``) until
2026-08-12; the page is gone and this module is the only copy, rendered by the
Operations → Metadata view (``metadata.py``). Moving it here replaced the
load-the-emoji-file-by-path machinery with a normal import.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import CACHE_TTL_SECONDS, sync_db_if_needed

from spyfish.database.manager import DatabaseManager

# Every `ColumnName` that means "the sampling window". The validator writes
# `sampling_window`; the two older names stay because rows already in the
# database carry them, and a name this filter does not match is an error that
# cannot be dismissed from the screen. Add to this list, never replace it.
SAMPLING_COLUMNS = ["sampling_window", "SamplingStart", "SamplingEnd"]


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_error_data():
    """Load validation errors from native SQLite DB, syncing from S3 if needed."""
    try:
        sync_db_if_needed()
        db = DatabaseManager()
        errors = db.get_all_validation_errors()
        return pd.DataFrame(errors)
    except Exception as e:
        st.error(f"Error loading validation errors: {e}")
        return pd.DataFrame()


# --- Display functions ---
def display_error_summary(errors_df: pd.DataFrame):
    if errors_df.empty:
        st.success("✅ No validation errors found!")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Errors", len(errors_df))
    with col2:
        st.metric("Error Types", errors_df["ErrorType"].nunique())
    with col3:
        st.metric("Files with Errors", errors_df["FileName"].nunique())
    with col4:
        st.metric("Surveys with Errors", errors_df["SurveyID"].nunique())
    with col5:
        st.metric("Deployments with Errors", errors_df["DropID"].nunique())


def display_error_type_breakdown(errors_df: pd.DataFrame):
    """Two side-by-side panels: errors by ErrorType, top failing CSV columns."""
    if errors_df.empty:
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Errors by Type")
        st.caption(
            "Which failure mode dominates? "
            "`Missing Required Value` is usually rangers not filling required fields."
        )
        type_counts = (
            errors_df["ErrorType"].fillna("(unspecified)").value_counts().reset_index()
        )
        type_counts.columns = ["ErrorType", "count"]
        fig_et = px.bar(
            type_counts.sort_values("count", ascending=True),
            x="count",
            y="ErrorType",
            orientation="h",
            color="count",
            color_continuous_scale="Reds",
            text="count",
            labels={"count": "Errors", "ErrorType": "Error type"},
            height=320,
        )
        fig_et.update_traces(textposition="outside")
        style(
            fig_et,
            yaxis_title=None,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_et, use_container_width=True)

    with col_b:
        st.subheader("🧭 Top failing columns")
        st.caption(
            "Which CSV columns fail validation most often? "
            "Reveals upstream PowerApps / SharePoint quality issues."
        )
        if "ColumnName" not in errors_df.columns:
            st.info("No column-level error data available.")
            return
        col_counts = (
            errors_df[errors_df["ColumnName"].notna()]["ColumnName"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        col_counts.columns = ["ColumnName", "count"]
        if col_counts.empty:
            st.info("No errors with column-level detail recorded.")
            return
        fig_cc = px.bar(
            col_counts.sort_values("count", ascending=True),
            x="count",
            y="ColumnName",
            orientation="h",
            color="count",
            color_continuous_scale="Oranges",
            text="count",
            labels={"count": "Errors", "ColumnName": "Column"},
            height=320,
        )
        fig_cc.update_traces(textposition="outside")
        style(
            fig_cc,
            yaxis_title=None,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cc, use_container_width=True)


def display_file_breakdown(errors_df: pd.DataFrame):
    if errors_df.empty:
        return
    st.subheader("📁 Errors by File")
    file_counts = errors_df["FileName"].value_counts().reset_index()
    file_counts.columns = ["File Name", "Error Count"]
    st.dataframe(file_counts, width="stretch", hide_index=True, height=300)


def display_error_table(errors_df: pd.DataFrame):
    """Render an error table. Callers filter the frame before passing it."""
    if errors_df.empty:
        st.info("No errors match the current filters")
        return

    st.caption(f"{len(errors_df)} errors")
    st.dataframe(
        errors_df,
        width="stretch",
        hide_index=True,
        column_config={
            "ErrorMessage": st.column_config.TextColumn("Message", width="large"),
        },
    )


def render_body():
    """The whole review, below whatever title/filters the caller owns."""
    raw_errors_df = load_error_data()

    if raw_errors_df.empty:
        st.success("🎉 No validation errors found in the data!")
        return

    # --- Header with Filters and Overview ---
    # Move filters up so they affect all subsequent charts
    st.header("📈 Overview")

    sampling_rows = 0
    if "ColumnName" in raw_errors_df.columns:
        sampling_rows = int(raw_errors_df["ColumnName"].isin(SAMPLING_COLUMNS).sum())

    include_sampling_errors = st.checkbox(
        f"Include sampling errors ({sampling_rows:,})",
        value=False,
        key="error_review_include_sampling",
        help="Errors on the sampling window, the SamplingStart / SamplingEnd "
        "times a ranger enters. Off by default: they are common omissions and "
        "tend to flood the view.",
    )

    errors_df = raw_errors_df.copy()
    if not include_sampling_errors and "ColumnName" in errors_df.columns:
        errors_df = errors_df[~errors_df["ColumnName"].isin(SAMPLING_COLUMNS)]

    display_error_summary(errors_df)
    st.divider()

    # --- Overview Breakdown ---
    display_error_type_breakdown(errors_df)
    st.divider()

    st.header("🔎 Detailed Error Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 By Error Type", "🗺️ By Survey", "📁 By File", "📋 All Errors"]
    )

    with tab1:
        st.subheader("Filter by Error Type")
        # dropna before sorted(): file-level errors can carry null ErrorType /
        # SurveyID, and sorted() raises TypeError on a None among strings.
        selected_error_types = st.multiselect(
            "Select error types",
            options=sorted(errors_df["ErrorType"].dropna().unique()),
        )
        if selected_error_types:
            display_error_table(
                errors_df[errors_df["ErrorType"].isin(selected_error_types)]
            )

    with tab2:
        st.subheader("Filter by Survey")
        selected_surveys = st.multiselect(
            "Select surveys to view",
            options=sorted(errors_df["SurveyID"].dropna().unique()),
        )
        if selected_surveys:
            display_error_table(errors_df[errors_df["SurveyID"].isin(selected_surveys)])

    with tab3:
        display_file_breakdown(errors_df)
        st.subheader("Filter by File")
        selected_files = st.multiselect(
            "Select specific files to view",
            options=sorted(errors_df["FileName"].dropna().unique()),
        )
        if selected_files:
            display_error_table(errors_df[errors_df["FileName"].isin(selected_files)])

    with tab4:
        st.subheader("All Validation Errors")
        display_error_table(errors_df)
