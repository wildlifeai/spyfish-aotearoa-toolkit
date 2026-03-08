import pandas as pd
import streamlit as st
import sqlite3
from pathlib import Path
from spyfish.config import config, PipelineStatus
from spyfish.utils import extract_survey_id, get_survey_summary
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from utils import sync_db_if_needed, check_password, render_sidebar_refresh

@st.cache_data(ttl=1)  # Refresh UI instantly
def load_deployment_status():
    """Load deployment status natively from local spyfish_pipeline.db, syncing from S3 if needed."""
    try:
        sync_db_if_needed()
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
            "is_bad_deployment": "IsBadDeployment",
            "sampling_start": "SamplingStart"
        }, inplace=True)
        df["IsBadDeployment"] = df["IsBadDeployment"].astype(bool)
        df["SurveyID"] = extract_survey_id(df["DropID"])

        # Map Pipeline Complete (excluding EXCLUDED/bad deployments)
        df["Complete"] = df["Status"] == PipelineStatus.PIPELINE_COMPLETE
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
        # User request: get this info from status (READY_FOR_ML, PIPELINE_COMPLETE, etc) rather than mocked VideoStatus
        videos_present = df["Status"].isin(PipelineStatus.VIDEO_PRESENT_STATUSES).sum()
        st.metric("Videos Present", videos_present)

    with col4:
        st.metric("Unique Surveys", df["SurveyID"].nunique())

    with col5: st.metric("ML Ann.", (df["MlAnnotations"] > 0).sum())

    with col6: st.metric("CitSci Ann.", (df["CitSciAnnotations"] > 0).sum())
    with col7: st.metric("Expert Ann.", (df["ExpertAnnotations"] > 0).sum())

    with st.container(horizontal=True, horizontal_alignment="right"):
        show_annotations = st.checkbox("Show Annotation Columns", key=f"show_ann_{title}", value=False)

    display_cols = ["DropID", "SurveyID", "SamplingStart", "Status","Complete"]
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

def render_overview(deployment_df: pd.DataFrame):
    st.header("📊 Overview")
    st.caption("Complete deployment status overview")

    total = len(deployment_df)
    videos_present = deployment_df["Status"].isin(PipelineStatus.VIDEO_PRESENT_STATUSES).sum()

    # Mini metrics overview
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1: st.metric("Total", total)
    with m2: st.metric("Action Req.", (~deployment_df["Complete"]).sum())
    with m3: st.metric("Videos", videos_present)
    with m4: st.metric("Complete", deployment_df["Complete"].sum())
    with m5: st.metric("ML Ann.", (deployment_df["MlAnnotations"] > 0).sum())
    with m6: st.metric("CitSci Ann.", (deployment_df["CitSciAnnotations"] > 0).sum())
    with m7: st.metric("Expert Ann.", (deployment_df["ExpertAnnotations"] > 0).sum())

def render_survey_tab(deployment_df: pd.DataFrame):
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

    survey_summary = get_survey_summary(deployment_df)
    if not survey_summary.empty:
        st.dataframe(survey_summary, width='stretch', hide_index=True)

def render_pipeline_stage_tab(deployment_df: pd.DataFrame):
    st.subheader("🔄 Pipeline Stage Breakdown")
    st.caption("How many deployments are at each step of the pipeline")

    total = len(deployment_df)
    status_counts = deployment_df["Status"].value_counts().to_dict()
    rows = []
    for status_key, label, description in PipelineStatus.STAGE_ORDER:
        count = status_counts.get(status_key, 0)
        if count == 0:
            continue
        pct = round(count / total * 100, 1) if total > 0 else 0
        rows.append({"Stage": label, "Description": description, "Count": count, "% of Total": f"{pct}%"})

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width='stretch',
            column_config={
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    min_value=0,
                    max_value=total,
                    format="%d",
                ),
            }
        )
    else:
        st.info("No deployed records to display yet.")

    # Filters
    status_order = [s[0] for s in PipelineStatus.STAGE_ORDER]
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

def render_detailed_annotation_tab(deployment_df: pd.DataFrame, ann_db):
    st.header("🔍 Detailed Annotation View")
    st.caption("Select a deployment to view individual annotation records from the annotations database")

    selected_drop_id = st.selectbox(
        "Select DropID to view details",
        options=["None"] + sorted(deployment_df["DropID"].tolist()),
        index=0
    )

    if selected_drop_id != "None":
        # Load detailed annotations
        detailed_anns = ann_db.get_annotations_for_drop(selected_drop_id)

        if detailed_anns:
            ann_df = pd.DataFrame(detailed_anns)
            st.write(f"Showing {len(ann_df)} annotations for **{selected_drop_id}**")

            # Group by source for mini metrics
            s_cols = st.columns(3)
            with s_cols[0]: st.metric("Expert", len(ann_df[ann_df["annotated_by"] == "expert"]))
            with s_cols[1]: st.metric("ML", len(ann_df[ann_df["annotated_by"] == "ml"]))
            with s_cols[2]: st.metric("CitSci", len(ann_df[ann_df["annotated_by"] == "citsci"]))

            st.dataframe(
                ann_df[[
                    "scientific_name", "time_of_max", "max_interval",
                    "annotated_by", "interval_annotation", "confidence_agreement", "external_id"
                ]],
                width='stretch',
                hide_index=True,
                column_config={
                    "scientific_name":      st.column_config.TextColumn("Scientific Name"),
                    "time_of_max":          st.column_config.TextColumn("Time of MaxN"),
                    "max_interval":         st.column_config.NumberColumn("MaxN Count", width="small"),
                    "annotated_by":         st.column_config.TextColumn("Annotated By", width="small"),
                    "interval_annotation":  st.column_config.TextColumn("Interval (s)", width="small"),
                    "confidence_agreement": st.column_config.NumberColumn("Confidence", format="%.2f", width="small"),
                    "external_id":          st.column_config.TextColumn("External ID", width="small"),
                },
            )
        else:
            st.info(f"No detailed annotations found for {selected_drop_id} in the annotations database.")

@st.cache_data(ttl=600)
def get_all_annotations_export(_adb):
    df = _adb.get_all_annotations_export_df()
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.set_page_config(page_title="Deployment Management", page_icon="⚙️", layout="wide")
    if not check_password():
        st.stop()

    render_sidebar_refresh()

    st.title("⚙️ Deployment Management")
    st.caption("Dashboard deployment")

    st.divider()
    deployment_df = load_deployment_status()
    if deployment_df is None: return

    render_overview(deployment_df)
    ann_db = AnnotationDatabaseManager()

    tab1, tab2, tab3 = st.tabs(["📋 Deployments Overview", "📊 Survey Overview", "🐟 Annotations Overview"])

    with tab2:
        render_survey_tab(deployment_df)

    with tab1:
        render_pipeline_stage_tab(deployment_df)

    with tab3:
        render_detailed_annotation_tab(deployment_df, ann_db)

    st.divider()

    # Download section
    st.header("📥 Export Data")
    col1, col2, col3 = st.columns(3)
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

    with col3:
        annotations_csv = get_all_annotations_export(ann_db)
        if annotations_csv:
            st.download_button(
                label="Download All Annotations",
                data=annotations_csv,
                file_name="spyfish_all_annotations.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.button("Download All Annotations", help="No annotations available yet.")

if __name__ == "__main__":
    main()
