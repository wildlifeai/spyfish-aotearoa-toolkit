import pandas as pd
import streamlit as st

from spyfish.database.manager import DatabaseManager
from utils import check_password, sync_db_if_needed, render_sidebar_refresh

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


# --- Load file differences (Mocked for now since state machine handles this) ---
@st.cache_data(ttl=1)
def load_file_differences():
    """Returns empty lists since File Presence is handled via pipeline Status."""
    return [], []


# --- Display functions ---
def display_error_summary(errors_df: pd.DataFrame):
    if errors_df.empty:
        st.success("✅ No validation errors found!")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Errors", len(errors_df))
    with col2: st.metric("Error Types", errors_df["ErrorType"].nunique())
    with col3: st.metric("Files with Errors", errors_df["FileName"].nunique())
    with col4: st.metric("Surveys with Errors", errors_df["SurveyID"].nunique())
    with col5: st.metric("Deployments with Errors", errors_df["DropID"].nunique())

def display_error_type_breakdown(errors_df: pd.DataFrame):
    if errors_df.empty: return
    st.subheader("📊 Errors by Type")
    error_type_counts = errors_df["ErrorType"].value_counts().reset_index()
    error_type_counts.columns = ["Error Type", "Count"]

    col1, col2 = st.columns([2, 1])
    with col1: st.bar_chart(error_type_counts.set_index("Error Type"))
    with col2: st.dataframe(error_type_counts, width='stretch', hide_index=True)

def display_file_breakdown(errors_df: pd.DataFrame):
    if errors_df.empty: return
    st.subheader("📁 Errors by File")
    file_counts = errors_df["FileName"].value_counts().reset_index()
    file_counts.columns = ["File Name", "Error Count"]
    st.dataframe(file_counts, width='stretch', hide_index=True, height=300)

def display_error_table(errors_df: pd.DataFrame, title: str, filters: dict = None):
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
        width='stretch',
        hide_index=True,
        column_config={
            "ErrorMessage": st.column_config.TextColumn("Message", width="large"),
            "status": st.column_config.TextColumn("Deployment Status"),
        }
    )

def main():
    st.set_page_config(page_title="Error Review", page_icon="🔍", layout="wide")
    if not check_password(): st.stop()

    # --- Sidebar ---
    render_sidebar_refresh()

    st.title("🔍 Data Validation & Error Review")
    st.caption("Review pipeline errors directly pulled and parsed from spyfish_pipeline.db")

    st.divider()
    raw_errors_df = load_error_data()

    if raw_errors_df.empty:
        st.success("🎉 No validation errors found in the data!")
        return

    # --- Header with Filters and Overview ---
    # Move filters up so they affect all subsequent charts
    st.header("📈 Overview")



    fcol1, fcol2 = st.columns(2)
    with fcol1:
        show_sampling_errors = st.checkbox("Show Sampling Errors", value=False)
    with fcol2:
        show_complete = st.checkbox("Show Complete", value=False)

    # Apply global filters
    errors_df = raw_errors_df.copy()
    if not show_sampling_errors and "ColumnName" in errors_df.columns:
        errors_df = errors_df[~errors_df["ColumnName"].isin(["SamplingStart", "SamplingEnd"])]
    if not show_complete and "status" in errors_df.columns:
        errors_df = errors_df[errors_df["status"] != "PIPELINE_COMPLETE"]

    # Now display metrics based on filtered data
    display_error_summary(errors_df)
    st.divider()

    # --- Overview Breakdown ---
    display_error_type_breakdown(errors_df)
    st.divider()

    st.header("🔎 Detailed Error Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 By Error Type", "🗺️ By Survey", "📁 By File", "📋 All Errors"])

    with tab1:
        st.subheader("Filter by Error Type")
        selected_error_types = st.multiselect("Select error types", options=sorted(errors_df["ErrorType"].unique()))
        if selected_error_types:
            display_error_table(errors_df[errors_df["ErrorType"].isin(selected_error_types)], "Errors by Type")

    with tab2:
        st.subheader("Filter by Survey")
        selected_surveys = st.multiselect("Select surveys to view", options=sorted(errors_df["SurveyID"].unique()))
        if selected_surveys:
            display_error_table(errors_df[errors_df["SurveyID"].isin(selected_surveys)], "Errors by Survey")

    with tab3:
        display_file_breakdown(errors_df)
        st.subheader("Filter by File")
        selected_files = st.multiselect("Select specific files to view", options=sorted(errors_df["FileName"].unique()))
        if selected_files:
            display_error_table(errors_df[errors_df["FileName"].isin(selected_files)], "Errors by File")

    with tab4:
        st.subheader("All Validation Errors")
        display_error_table(errors_df, "All Errors", None)

if __name__ == "__main__":
    main()
