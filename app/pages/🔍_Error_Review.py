from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import render_contact_note, render_sidebar_refresh, sync_db_if_needed

from spyfish.database.manager import DatabaseManager


@st.cache_data(ttl=1)  # Cache for 1 second instead of 5 minutes to feel native
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
        fig_et.update_layout(
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
        fig_cc.update_layout(
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


def display_error_table(errors_df: pd.DataFrame, filters: Optional[dict] = None):
    if errors_df.empty:
        st.info("No errors match the current filters")
        return

    filtered_df = errors_df.copy()
    if filters:
        for col, values in filters.items():
            if values and col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col].isin(values)]

    st.caption(f"{len(filtered_df)} errors")
    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True,
        column_config={
            "ErrorMessage": st.column_config.TextColumn("Message", width="large"),
        },
    )


def main():
    st.set_page_config(page_title="Error Review", page_icon="🔍", layout="wide")

    # --- Sidebar ---
    render_contact_note()
    render_sidebar_refresh()

    st.title("🔍 Data Validation & Error Review")
    st.caption(
        "Review pipeline errors directly pulled and parsed from spyfish_pipeline.db"
    )

    st.divider()
    raw_errors_df = load_error_data()

    if raw_errors_df.empty:
        st.success("🎉 No validation errors found in the data!")
        return

    # --- Header with Filters and Overview ---
    # Move filters up so they affect all subsequent charts
    st.header("📈 Overview")

    include_sampling_errors = st.checkbox(
        "Include sampling errors",
        value=False,
        help="Include errors on the SamplingStart / SamplingEnd columns. "
        "Off by default — these are common ranger-entry omissions and "
        "tend to flood the view.",
    )

    errors_df = raw_errors_df.copy()
    if not include_sampling_errors and "ColumnName" in errors_df.columns:
        errors_df = errors_df[
            ~errors_df["ColumnName"].isin(["SamplingStart", "SamplingEnd"])
        ]

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
        selected_error_types = st.multiselect(
            "Select error types", options=sorted(errors_df["ErrorType"].unique())
        )
        if selected_error_types:
            display_error_table(
                errors_df[errors_df["ErrorType"].isin(selected_error_types)]
            )

    with tab2:
        st.subheader("Filter by Survey")
        selected_surveys = st.multiselect(
            "Select surveys to view", options=sorted(errors_df["SurveyID"].unique())
        )
        if selected_surveys:
            display_error_table(errors_df[errors_df["SurveyID"].isin(selected_surveys)])

    with tab3:
        display_file_breakdown(errors_df)
        st.subheader("Filter by File")
        selected_files = st.multiselect(
            "Select specific files to view",
            options=sorted(errors_df["FileName"].unique()),
        )
        if selected_files:
            display_error_table(errors_df[errors_df["FileName"].isin(selected_files)])

    with tab4:
        st.subheader("All Validation Errors")
        display_error_table(errors_df)


if __name__ == "__main__":
    main()
