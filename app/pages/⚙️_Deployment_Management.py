import hmac
import pandas as pd
import streamlit as st
import sqlite3
from pathlib import Path
from spyfish.config import config
from spyfish.config import PipelineStatus

# Constants for UI filtering state checks
VIDEO_PRESENT_STATUSES = [
    PipelineStatus.READY_FOR_ML, PipelineStatus.PROCESSING_ML, PipelineStatus.ML_COMPLETE,
    PipelineStatus.READY_FOR_CITSCI, PipelineStatus.PROCESSING_CITSCI, PipelineStatus.CITSCI_COMPLETE,
    PipelineStatus.READY_FOR_EXPERT, PipelineStatus.PROCESSING_EXPERT, PipelineStatus.EXPERT_COMPLETE,
    PipelineStatus.PIPELINE_COMPLETE
]

# --- Load SQLite deployment status data ---
@st.cache_data(ttl=1)  # Refresh instantly
def load_deployment_status():
    """Load deployment status natively from local spyfish_pipeline.db"""
    try:
        conn = sqlite3.connect(config.db_path)
        df = pd.read_sql("SELECT * FROM deployments", conn)

        if df.empty:
            return df

        # Standardize matching case output
        df.rename(columns={
            "drop_id": "DropID",
            "status": "Status",
            "expert_annotations": "ExpertAnnotations",
            "ml_annotations": "MlAnnotations",
            "citsci_annotations": "CitSciAnnotations",
            "is_bad_deployment": "IsBadDeployment"
        }, inplace=True)
        df["IsBadDeployment"] = df["IsBadDeployment"].astype(bool)

        # MOCKING MISSING STATS COLUMNS FOR VISUAL PARITY
        from spyfish.utils import extract_survey_id
        df["SurveyID"] = extract_survey_id(df["DropID"])

        # Map Pipeline Complete (excluding EXCLUDED/bad deployments)
        df["Complete"] = df["Status"] == "PIPELINE_COMPLETE"
        # Read the new native SQLite database columns
        df["ExpertAnnotations"] = df["ExpertAnnotations"].fillna(0).astype(int)
        df["MlAnnotations"] = df["MlAnnotations"].fillna(0).astype(int)
        df["CitSciAnnotations"] = df["CitSciAnnotations"].fillna(0).astype(int)
        df["TotalAnnotations"] = df["ExpertAnnotations"] + df["MlAnnotations"] + df["CitSciAnnotations"]

        # Determine NeedsAction based strictly on db flags and presence
        # Deployments that are complete or bad do not need action
        df["NeedsAction"] = ~(df["Complete"] | df["IsBadDeployment"])


        return df
    except Exception as e:
        st.error(f"Error loading deployment DB: {e}")
        return None

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
def display_deployment_table(df: pd.DataFrame, title: str, description: str):
    st.subheader(title)
    st.caption(description)

    if df.empty:
        st.success("✅ No deployments in this category")
        return

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Total Deployments", len(df))
    with col2: st.metric("Action Req.", (~df["Complete"]).sum())
    with col3:
        st.metric("Unique Surveys", df["SurveyID"].nunique())
    with col4:
        # User request: get this info from status (READY_FOR_ML, PIPELINE_COMPLETE, etc) rather than mocked VideoStatus
        videos_present = df["Status"].isin(VIDEO_PRESENT_STATUSES).sum()
        st.metric("Videos Present", videos_present)

    with col5: st.metric("ML Ann.", (df["MlAnnotations"] > 0).sum())

    with col6: st.metric("CitSci Ann.", (df["CitSciAnnotations"] > 0).sum())
    with col7: st.metric("Expert Ann.", (df["ExpertAnnotations"] > 0).sum())


    show_annotations = st.checkbox("Show Annotation Columns", key=f"show_ann_{title}", value=False)

    display_cols = ["DropID", "SurveyID", "Status", "Complete"]
    if show_annotations:
        display_cols.extend(["MlAnnotations", "CitSciAnnotations","ExpertAnnotations" ])

    st.dataframe(
        df[display_cols],
        width='stretch',
        hide_index=True,
        column_config={
            "Complete": st.column_config.CheckboxColumn("Complete", width="small"),

            "MlAnnotations": st.column_config.NumberColumn("ML", width="small"),
            "CitSciAnnotations": st.column_config.NumberColumn("CitSci", width="small"),
            "ExpertAnnotations": st.column_config.NumberColumn("Expert", width="small"),
        }
    )

def main():
    st.set_page_config(page_title="Deployment Management", page_icon="⚙️", layout="wide")
    if not check_password():
        st.stop()

    st.title("⚙️ Deployment Management")
    st.caption("Dashboard deployment")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh DB", help="Read latest SQLite pipeline status"):
            st.rerun()

    st.divider()
    deployment_df = load_deployment_status()
    if deployment_df is None: return

    st.header("📊 Overview")
    st.caption("Complete deployment status overview")

    # Mini metrics overview
    m1, m2, m3, m4, m5, m6,m7 = st.columns(7)
    with m1: st.metric("Total", len(deployment_df))
    with m2: st.metric("Action Req.", (~deployment_df["Complete"]).sum())
    with m3: st.metric("Surveys", deployment_df["SurveyID"].nunique())

    videos_present = deployment_df["Status"].isin(VIDEO_PRESENT_STATUSES).sum()
    with m4: st.metric("Videos", videos_present)

    with m5: st.metric("ML Ann.", (deployment_df["MlAnnotations"] > 0).sum())

    with m6: st.metric("CitSci Ann.", (deployment_df["CitSciAnnotations"] > 0).sum())
    with m7: st.metric("Expert Ann.", (deployment_df["ExpertAnnotations"] > 0).sum())


    st.divider()

    tab1, tab2 = st.tabs([ "📋 All Deployments", "📊 Survey Overview"])

    with tab2:
        st.subheader("Survey Overview")
        st.caption("Aggregation of video deployments by SurveyID")

        # Survey-level interesting metrics overview
        st.divider()
        s1, s2, s3, s4, s5, s6, s7 = st.columns(7)

        total_surveys = max(1, deployment_df["SurveyID"].nunique())
        with s1: st.metric("Surveys", total_surveys)
        with s2: st.metric("Avg Deps / Survey", round(len(deployment_df) / total_surveys, 1))

        survey_completion = (deployment_df['Complete'].mean() * 100) if not deployment_df.empty else 0
        with s3: st.metric("Completion", f"{round(survey_completion, 1)}%")

        with s4: st.metric("Avg Anns / Survey", round(deployment_df["TotalAnnotations"].sum() / total_surveys))
        with s5: st.metric("Total Annotations", deployment_df["TotalAnnotations"].sum())
        with s6: st.metric("Bad Deployments", deployment_df["IsBadDeployment"].sum())
        with s7: st.metric("Action Required", (~deployment_df["Complete"]).sum())
        st.divider()

        if not deployment_df.empty:
            survey_summary = deployment_df.groupby("SurveyID").agg(
                TotalDeployments=("DropID", "nunique"),
                CompleteDeployments=("Complete", "sum"),
                BadDeployments=("IsBadDeployment", "sum"),
                NeedsAction=("NeedsAction", "sum"),
                VideosPresent=("Status", lambda x: x.isin(VIDEO_PRESENT_STATUSES).sum()),
                MLAnnotated=("MlAnnotations", lambda x: (x > 0).sum()),
                CitSciAnnotated=("CitSciAnnotations", lambda x: (x > 0).sum()),
                ExpertAnnotated=("ExpertAnnotations", lambda x: (x > 0).sum())
            ).reset_index()
            # Calculate percentages cleanly: (Bad + Expert) / Total
            survey_summary["CompletionPct"] = (
                ((survey_summary["BadDeployments"] + survey_summary["ExpertAnnotated"]) / survey_summary["TotalDeployments"]) * 100
            ).round(1).astype(str) + "%"

            st.dataframe(survey_summary, width='stretch', hide_index=True)

    with tab1:

        # Filters
        status_order = [
            "PENDING_ARRIVAL", "READY_FOR_ML", "PROCESSING_ML", "ML_COMPLETE",
            "READY_FOR_CITSCI", "PROCESSING_CITSCI", "CITSCI_COMPLETE",
            "READY_FOR_EXPERT", "PROCESSING_EXPERT", "EXPERT_COMPLETE",
            "PIPELINE_COMPLETE", "EXCLUDED", "ON_HOLD", "ERROR", "MISSING_METADATA"
        ]
        all_statuses = [s for s in status_order if s in deployment_df["Status"].unique()]
        all_statuses += [s for s in deployment_df["Status"].unique() if s not in status_order]

        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=all_statuses,
                default=None
            )
        with col2:
            survey_filter = st.multiselect(
                "Filter by Survey",
                options=sorted(deployment_df["SurveyID"].unique().tolist()),
                default=None
            )
        with col3:
            complete_filter = st.selectbox(
                "Filter by Complete",
                options=["All", "Complete", "Action Required"],
                index=0
            )

        # Apply filters
        filtered_df = deployment_df.copy()
        if status_filter:
            filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]
        if survey_filter:
            filtered_df = filtered_df[filtered_df["SurveyID"].isin(survey_filter)]
        if complete_filter == "Complete":
            filtered_df = filtered_df[filtered_df["Complete"] == True]
        elif complete_filter == "Action Required":
            filtered_df = filtered_df[filtered_df["Complete"] == False]

        display_deployment_table(
            filtered_df,
            f"Filtered Results ({len(filtered_df)} deployments)",
            "Use filters above to narrow down results"
        )

    st.divider()

    # Download section
    st.header("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = deployment_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Status Report",
            data=csv,
            file_name="deployment_status.csv",
            mime="text/csv",
        )
    with col2:
        needs_action_csv = deployment_df[~deployment_df["Complete"]].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Needs Action Only",
            data=needs_action_csv,
            file_name="deployments_needs_action.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
