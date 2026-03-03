import hmac
import pandas as pd
import streamlit as st

from spyfish.database.manager import DatabaseManager


from utils import sync_db_if_needed
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

# --- Password protection ---
def check_password():
    """Returns True if the user entered the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            app_password = st.secrets.get("APP_PASSWORD")
            if app_password is not None and hmac.compare_digest(password, app_password):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        return False
    return True

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
        }
    )

def main():
    st.set_page_config(page_title="Error Review", page_icon="🔍", layout="wide")
    if not check_password(): st.stop()

    st.title("🔍 Data Validation & Error Review")
    st.caption("Review pipeline errors directly pulled and parsed from spyfish_pipeline.db")

    st.divider()
    errors_df = load_error_data()

    st.header("📈 Overview")
    display_error_summary(errors_df)
    st.divider()

    if not errors_df.empty:
        col1, col2 = st.columns(2)
        with col1: display_error_type_breakdown(errors_df)
        with col2: display_file_breakdown(errors_df)

        st.divider()
        st.header("🔎 Detailed Error Exploration")

        tab1, tab2, tab3 = st.tabs(["🎯 By Error Type", "🗺️ By Survey", "📋 All Errors"])

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
            st.subheader("All Validation Errors")
            display_error_table(errors_df, "All Errors", None)
    else:
        st.success("🎉 No validation errors found in the data!")

if __name__ == "__main__":
    main()
