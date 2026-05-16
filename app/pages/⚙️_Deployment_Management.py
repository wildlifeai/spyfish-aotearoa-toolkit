import pandas as pd
import streamlit as st
from utils import render_sidebar_refresh, sync_db_if_needed

from spyfish.config.base import (
    CitSciStatus,
    ExpertStatus,
    MlStatus,
    ReportingStatus,
    VideoPresence,
)
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.utils import extract_survey_id, get_survey_summary


@st.cache_data(ttl=1)  # Refresh UI instantly
def load_deployment_status():
    """Load deployments natively from local spyfish_pipeline.db, syncing from S3 if needed.

    Columns keep their raw DB names (snake_case). Derived columns added below
    use snake_case too so the convention is uniform across the whole frame.
    Human-readable labels are applied at render time via st.column_config.
    """
    try:
        sync_db_if_needed()
        db = DatabaseManager()
        with db.get_connection() as conn:
            df = pd.read_sql("SELECT * FROM deployments", conn)

        if df.empty:
            return df

        df["is_bad_deployment"] = df["is_bad_deployment"].astype(bool)
        df["survey_id"] = extract_survey_id(df["drop_id"])

        df["complete"] = (
            (df["expert_status"] == ExpertStatus.COMPLETE)
            | (df["reporting_status"] == ReportingStatus.COMPLETE)
            | (
                (df["ml_status"] == MlStatus.SKIPPED)
                & (df["citsci_status"] == CitSciStatus.SKIPPED)
                & (df["expert_status"] == ExpertStatus.SKIPPED)
            )
        )

        for col in ("expert_annotations", "ml_annotations", "citsci_annotations"):
            df[col] = df[col].fillna(0).astype(int)
        df["total_annotations"] = (
            df["expert_annotations"] + df["ml_annotations"] + df["citsci_annotations"]
        )

        _presence_display = {
            VideoPresence.PRESENT: "Present",
            VideoPresence.ABSENT: "Absent",
            VideoPresence.NO_VIDEO_BAD_DEP: "No video (bad dep.)",
        }
        df["video_status"] = (
            df["video_presence"].map(_presence_display).fillna("Unknown")
        )
        df["needs_action"] = ~(df["complete"] | df["is_bad_deployment"])

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
    with col2:
        st.metric("Action Req.", (~df["complete"]).sum())
    with col3:
        videos_present = (df["video_presence"] == VideoPresence.PRESENT).sum()
        st.metric("Videos Present", videos_present)
    with col4:
        st.metric("Unique Surveys", df["survey_id"].nunique())
    with col5:
        st.metric("ML Ann.", (df["ml_annotations"] > 0).sum())
    with col6:
        st.metric("CitSci Ann.", (df["citsci_annotations"] > 0).sum())
    with col7:
        st.metric("Expert Ann.", (df["expert_annotations"] > 0).sum())

    with st.container(horizontal=True, horizontal_alignment="right"):
        show_annotations = st.checkbox(
            "Show Annotation Columns", key=f"show_ann_{title}", value=False
        )

    display_cols = [
        "drop_id",
        "survey_id",
        "sampling_start",
        "is_bad_deployment",
        "ml_status",
        "citsci_status",
        "expert_status",
        "video_status",
        "complete",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    if show_annotations:
        display_cols.extend(
            ["ml_annotations", "citsci_annotations", "expert_annotations"]
        )

    st.dataframe(
        df[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "drop_id": st.column_config.TextColumn("DropID"),
            "survey_id": st.column_config.TextColumn("SurveyID"),
            "sampling_start": st.column_config.NumberColumn(
                "Sampling Start", width="small"
            ),
            "complete": st.column_config.CheckboxColumn("Complete", width="small"),
            "is_bad_deployment": st.column_config.CheckboxColumn("Bad", width="small"),
            "video_status": st.column_config.TextColumn("Video", width="small"),
            "ml_status": st.column_config.TextColumn("ML", width="small"),
            "citsci_status": st.column_config.TextColumn("CitSci", width="small"),
            "expert_status": st.column_config.TextColumn("Biigle", width="small"),
            "ml_annotations": st.column_config.NumberColumn("ML", width="small"),
            "citsci_annotations": st.column_config.NumberColumn(
                "CitSci", width="small"
            ),
            "expert_annotations": st.column_config.NumberColumn(
                "Expert", width="small"
            ),
        },
    )


def render_overview(deployment_df: pd.DataFrame):
    st.header("📊 Overview")
    st.caption("Complete deployment status overview")

    total = len(deployment_df)
    videos_present = (deployment_df["video_presence"] == VideoPresence.PRESENT).sum()

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.metric("Total", total)
    with m2:
        st.metric("Action Req.", (~deployment_df["complete"]).sum())
    with m3:
        st.metric("Videos", videos_present)
    with m4:
        st.metric("Complete", deployment_df["complete"].sum())
    with m5:
        st.metric("ML Ann.", (deployment_df["ml_annotations"] > 0).sum())
    with m6:
        st.metric("CitSci Ann.", (deployment_df["citsci_annotations"] > 0).sum())
    with m7:
        st.metric("Expert Ann.", (deployment_df["expert_annotations"] > 0).sum())


def render_survey_tab(deployment_df: pd.DataFrame):
    st.subheader("Survey Overview")
    st.caption("Aggregation of video deployments by SurveyID")

    st.divider()
    s1, s2, s3, s4, s5, s6, s7 = st.columns(7)

    total_surveys = max(1, deployment_df["survey_id"].nunique())
    with s1:
        st.metric("Surveys", total_surveys)
    with s2:
        st.metric("Avg Deps / Survey", round(len(deployment_df) / total_surveys, 1))

    survey_completion = (
        (deployment_df["complete"].mean() * 100) if not deployment_df.empty else 0
    )
    with s3:
        st.metric("Completion", f"{round(survey_completion, 1)}%")
    with s4:
        st.metric(
            "Avg Anns / Survey",
            round(deployment_df["total_annotations"].sum() / total_surveys),
        )
    with s5:
        st.metric("Total Annotations", deployment_df["total_annotations"].sum())
    with s6:
        st.metric("Bad Deployments", deployment_df["is_bad_deployment"].sum())
    with s7:
        st.metric("Action Required", (~deployment_df["complete"]).sum())
    st.divider()

    survey_summary = get_survey_summary(deployment_df)
    if not survey_summary.empty:
        st.dataframe(survey_summary, width="stretch", hide_index=True)


def render_pipeline_stage_tab(deployment_df: pd.DataFrame):
    st.subheader("🔄 Pipeline section progress")
    st.caption(
        "100% stacked bars — each pipeline section's distribution of deployments by "
        "status category. Hover for the exact status value within each band."
    )

    sections = [
        ("ingest_status", "Ingest"),
        ("ml_status", "ML"),
        ("citsci_status", "CitSci"),
        ("expert_status", "Expert"),
        ("reporting_status", "Reporting"),
    ]

    total = len(deployment_df)
    rows = []
    for col, label in sections:
        if col not in deployment_df.columns:
            continue
        for status, count in deployment_df[col].value_counts().items():
            pct = round(count / total * 100, 1) if total > 0 else 0
            rows.append(
                {
                    "Section": label,
                    "Status": status,
                    "Count": int(count),
                    "Pct": pct,
                }
            )

    if not rows:
        st.info("No deployed records to display yet.")
    else:
        table_df = (
            pd.DataFrame(rows)
            .rename(columns={"Pct": "%"})[["Section", "Status", "Count", "%"]]
            .sort_values(["Section", "Count"], ascending=[True, False])
        )
        st.dataframe(
            table_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    min_value=0,
                    max_value=total,
                    format="%d",
                ),
                "%": st.column_config.NumberColumn("%", format="%.1f%%"),
            },
        )

    # Filters — pipeline-stage statuses in the top row, context filters below
    st.markdown("**Filter deployments**")
    s1, s2, s3 = st.columns(3)
    with s1:
        ml_filter = st.multiselect(
            "ML status",
            options=sorted(deployment_df["ml_status"].unique().tolist()),
            default=None,
        )
    with s2:
        citsci_filter = st.multiselect(
            "CitSci status",
            options=sorted(deployment_df["citsci_status"].unique().tolist()),
            default=None,
        )
    with s3:
        expert_filter = st.multiselect(
            "Expert status",
            options=sorted(deployment_df["expert_status"].unique().tolist()),
            default=None,
        )

    c1, c2 = st.columns(2)
    with c1:
        survey_filter = st.multiselect(
            "Survey",
            options=sorted(deployment_df["survey_id"].unique().tolist()),
            default=None,
        )
    with c2:
        complete_filter = st.selectbox(
            "Completion",
            options=["All", "Complete", "Action Required"],
            index=0,
        )

    filtered_df = deployment_df.copy()
    if ml_filter:
        filtered_df = filtered_df[filtered_df["ml_status"].isin(ml_filter)]
    if citsci_filter:
        filtered_df = filtered_df[filtered_df["citsci_status"].isin(citsci_filter)]
    if expert_filter:
        filtered_df = filtered_df[filtered_df["expert_status"].isin(expert_filter)]
    if survey_filter:
        filtered_df = filtered_df[filtered_df["survey_id"].isin(survey_filter)]
    if complete_filter == "Complete":
        filtered_df = filtered_df[filtered_df["complete"]]
    elif complete_filter == "Action Required":
        filtered_df = filtered_df[~filtered_df["complete"]]

    display_deployment_table(
        filtered_df,
        f"Filtered Results ({len(filtered_df)} deployments)",
        "Use filters above to narrow down results",
    )


def render_detailed_annotation_tab(deployment_df: pd.DataFrame, ann_db):
    st.header("🔍 Detailed Annotation View")
    st.caption(
        "Select a deployment to view individual annotation records from the annotations database"
    )

    selected_drop_id = st.selectbox(
        "Select DropID to view details",
        options=["None"] + sorted(deployment_df["drop_id"].tolist()),
        index=0,
    )

    if selected_drop_id != "None":
        maxn_df = ann_db.get_maxn_summary(drop_id=selected_drop_id)

        if not maxn_df.empty:
            st.dataframe(
                maxn_df[
                    [
                        "scientific_name",
                        "annotated_by",
                        "maxn",
                        "time_of_max",
                        "confidence_agreement",
                        "external_id",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "scientific_name": st.column_config.TextColumn("Species"),
                    "annotated_by": st.column_config.TextColumn(
                        "Source", width="small"
                    ),
                    "maxn": st.column_config.NumberColumn("Peak MaxN", width="small"),
                    "time_of_max": st.column_config.TextColumn(
                        "Time of MaxN", width="small"
                    ),
                    "confidence_agreement": st.column_config.NumberColumn(
                        "Confidence", format="%.2f", width="small"
                    ),
                    "external_id": st.column_config.TextColumn(
                        "Provenance",
                        help=(
                            "Tracks which source produced each annotation. "
                            "ML rows: model name (e.g. 'species_20260429_081503'). "
                            "Expert rows: BIIGLE annotation_id — the unique ID of "
                            "the bbox in BIIGLE; visible in the BIIGLE UI on each "
                            "annotation's detail panel. "
                            "CitSci rows: empty (Zooniverse classifications aren't "
                            "individually addressable post-aggregation)."
                        ),
                    ),
                },
            )
        else:
            st.info(
                f"No annotations found for {selected_drop_id} in the annotations database."
            )


@st.cache_data(ttl=600)
def get_all_annotations_export(_adb):
    df = _adb.get_all_annotations_export_df()
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.set_page_config(page_title="Deployment Management", page_icon="⚙️", layout="wide")

    render_sidebar_refresh()

    st.title("⚙️ Deployment Management")
    st.caption("Dashboard deployment")

    st.divider()
    deployment_df = load_deployment_status()
    if deployment_df is None:
        return

    render_overview(deployment_df)
    ann_db = AnnotationDatabaseManager()

    tab1, tab2, tab3 = st.tabs(
        ["📋 Deployments Overview", "📊 Survey Overview", "🐟 Annotations Overview"]
    )

    with tab2:
        render_survey_tab(deployment_df)

    with tab1:
        render_pipeline_stage_tab(deployment_df)

    with tab3:
        render_detailed_annotation_tab(deployment_df, ann_db)

    st.divider()

    st.header("📥 Export Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = deployment_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Status Report",
            data=csv,
            file_name="deployment_status.csv",
            mime="text/csv",
        )
    with col2:
        needs_action_csv = (
            deployment_df[~deployment_df["complete"]]
            .to_csv(index=False)
            .encode("utf-8")
        )
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
                type="primary",
            )
        else:
            st.button("Download All Annotations", help="No annotations available yet.")


if __name__ == "__main__":
    main()
