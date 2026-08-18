"""
Model Metrics page for Spyfish Aotearoa pipeline.

Displays metrics for trained ML models, including:
  - mAP@0.5, precision, recall for current and candidate models
  - Per-class breakdown (if available)
  - Confusion matrix images
  - Training curves (loss + mAP over epochs)
  - Manual promote button (copy selected model to production)

Reads metrics CSVs and confusion matrix images from S3 under
    process_files/training/results/
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from utils import CACHE_TTL_SECONDS

from spyfish.config.wrapper import config
from spyfish.storage.s3_handler import S3Handler
from spyfish.utils import validate_model_path

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def list_result_dirs_from_s3(bucket: str, results_prefix: str) -> Optional[list[str]]:
    """List available result directories in S3.

    Returns None when the listing itself failed (no credentials, no network),
    so the caller can tell "S3 unreachable" apart from "no results yet", the
    two need different messages.
    """
    try:
        s3 = S3Handler(bucket=bucket)
        return sorted(s3.list_common_prefixes(results_prefix), reverse=True)
    except Exception as e:
        logging.warning(f"Could not list S3 result dirs: {e}")
        return None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_metrics_csv_from_s3(bucket: str, s3_key: str) -> Optional[pd.DataFrame]:
    """Download and parse a metrics CSV from S3."""
    try:
        s3 = S3Handler(bucket=bucket)
        return s3.read_df_from_s3_csv(s3_key)
    except Exception as e:
        logging.warning(f"Could not load metrics CSV {s3_key}: {e}")
        return None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_image_from_s3(bucket: str, s3_key: str) -> Optional[bytes]:
    """Download an image from S3 and return raw bytes."""
    try:
        s3 = S3Handler(bucket=bucket)
        return s3.read_bytes_from_s3(s3_key)
    except Exception as e:
        logging.warning(f"Could not load image {s3_key}: {e}")
        return None


def load_local_metrics(local_results_dir: Path) -> Optional[pd.DataFrame]:
    """Load metrics from local filesystem (fallback when S3 unavailable)."""
    csv_path = local_results_dir / "metrics_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Render a styled metrics comparison table."""
    display_cols = [
        c
        for c in ["role", "mAP50", "mAP50_95", "precision", "recall", "model_path"]
        if c in metrics_df.columns
    ]
    st.dataframe(
        metrics_df[display_cols].rename(
            columns={
                "role": "Model",
                "mAP50": "mAP@0.5",
                "mAP50_95": "mAP@0.5:0.95",
                "precision": "Precision",
                "recall": "Recall",
                "model_path": "Weights Path",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "mAP@0.5": st.column_config.NumberColumn(format="%.4f"),
            "mAP@0.5:0.95": st.column_config.NumberColumn(format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def render_training_curves(results_df: pd.DataFrame) -> None:
    """Plot training loss and mAP over epochs using Streamlit's native chart."""

    results_df.columns = [c.strip() for c in results_df.columns]
    epoch_col = next((c for c in results_df.columns if "epoch" in c.lower()), None)
    if epoch_col is None:
        st.info("No 'epoch' column found in results.csv.")
        return

    map_cols = [
        c for c in results_df.columns if "map50" in c.lower() and "95" not in c.lower()
    ]
    loss_cols = [c for c in results_df.columns if "loss" in c.lower()]

    if map_cols:
        st.subheader("mAP@0.5 over Epochs")
        chart_df = results_df.set_index(epoch_col)[map_cols]
        st.line_chart(chart_df)

    if loss_cols:
        st.subheader("Loss over Epochs")
        chart_df = results_df.set_index(epoch_col)[loss_cols]
        st.line_chart(chart_df)


def render_promote_button(
    model_path: str,
    model_type: str,
) -> None:
    """Render a promote button that copies the selected model to the local production path."""
    st.divider()
    st.subheader("🚀 Promote Model")

    prod_model_path = config.pipeline_model_path
    st.caption(
        f"Promoting this model will replace the current production model at: `{prod_model_path}`. "
        "The backup to S3 will occur during the next pipeline sync."
    )

    confirm = st.checkbox(
        "I have reviewed the metrics and want to promote this model",
        key="confirm_promote",
    )
    if st.button("✅ Promote to Production", disabled=not confirm, type="primary"):
        with st.spinner(f"Promoting {model_path} ..."):
            try:
                # Ensure destination directory exists
                prod_model_path.parent.mkdir(parents=True, exist_ok=True)

                # Validate source path
                model_path = str(validate_model_path(model_path))

                # Local copy
                shutil.copy2(model_path, prod_model_path)

                st.success(f"✅ Model promoted locally to: `{prod_model_path}`")
                st.info(
                    "The new model will be backed up to S3 during the final sync stage of the next pipeline run."
                )
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Promotion failed: {e}")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Model Metrics", page_icon="📊", layout="wide")

    # Rendered for every page by the entrypoint now.

    st.title("📊 Model Metrics")
    st.caption(
        "Review and compare trained ML model performance before promoting to production."
    )

    bucket = config.s3_bucket
    results_prefix = config.training_results_s3_prefix
    local_results_root = config.training_results_dir

    # --- Sidebar: select result run ---
    st.sidebar.header("Select Run")

    result_dirs = list_result_dirs_from_s3(bucket, results_prefix)
    if result_dirs is None:
        # An outage is not "no results yet", say which one happened, or the
        # message sends someone off to re-run evaluate for nothing.
        st.warning(
            "Could not list results on S3 (check network / AWS credentials). "
            "Showing local results only."
        )
        result_dirs = []

    # Also check local results as a fallback
    local_result_dirs = []
    if local_results_root.exists():
        local_result_dirs = [
            d.name
            for d in sorted(local_results_root.iterdir(), reverse=True)
            if d.is_dir()
        ]

    all_runs = result_dirs + [
        f"[local] {d}" for d in local_result_dirs if d not in result_dirs
    ]

    if not all_runs:
        st.info(
            "No evaluation results found yet. "
            "Run `python -m spyfish.ml.training.evaluate` after training to generate metrics."
        )
        return

    selected_run = st.sidebar.selectbox("Evaluation run", all_runs)
    is_local = selected_run.startswith("[local] ")
    run_name = selected_run.replace("[local] ", "")

    # Infer model type from run name (e.g. '20260304_155200_binary' → 'binary')
    model_type = "binary" if "binary" in run_name else "species"
    st.sidebar.caption(f"Detected model type: **{model_type}**")

    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # --- Load metrics ---
    metrics_df = None
    if is_local:
        metrics_df = load_local_metrics(local_results_root / run_name)
    else:
        metrics_key = f"{results_prefix}/{run_name}/metrics_comparison.csv"
        metrics_df = load_metrics_csv_from_s3(bucket, metrics_key)

    # === Tab layout ===
    tab_metrics, tab_curves, tab_confusion = st.tabs(
        ["📈 Metrics", "📉 Training Curves", "🔲 Confusion Matrix"]
    )

    with tab_metrics:
        st.header("Performance Comparison")
        if metrics_df is not None and not metrics_df.empty:
            render_metrics_table(metrics_df)

            # Delta callout: new vs production
            new_row, prod_row = pd.DataFrame(), pd.DataFrame()
            if "role" in metrics_df.columns:
                new_row = metrics_df[metrics_df["role"] == "new"]
                prod_row = metrics_df[metrics_df["role"] == "production"]

            if not new_row.empty and not prod_row.empty:
                delta = float(new_row["mAP50"].iloc[0]) - float(
                    prod_row["mAP50"].iloc[0]
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New mAP@0.5", f"{float(new_row['mAP50'].iloc[0]):.4f}")
                with col2:
                    st.metric(
                        "Production mAP@0.5", f"{float(prod_row['mAP50'].iloc[0]):.4f}"
                    )
                with col3:
                    # Same config property the evaluate step's promote decision
                    # reads, so the page and the pipeline share one threshold.
                    min_improvement = config.retrain_min_improvement_pct / 100.0
                    st.metric(
                        "Improvement",
                        f"{delta:+.4f}",
                        delta=f"{'✅ above' if delta >= min_improvement else '❌ below'} {min_improvement:.1%} threshold",
                    )
        else:
            st.info("No metrics CSV found for this run.")

        # Promote button (only if we have a new model path)
        if (
            metrics_df is not None
            and "model_path" in metrics_df.columns
            and "role" in metrics_df.columns
        ):
            new_rows = metrics_df[metrics_df["role"] == "new"]
            if not new_rows.empty:
                new_model_path = str(new_rows["model_path"].iloc[0])
                render_promote_button(new_model_path, model_type)

    with tab_curves:
        st.header("Training Curves")
        curves_df = None
        if is_local:
            local_csv = local_results_root / run_name / "results.csv"
            if local_csv.exists():
                curves_df = pd.read_csv(local_csv)
        else:
            curves_key = f"{results_prefix}/{run_name}/results.csv"
            curves_df = load_metrics_csv_from_s3(bucket, curves_key)

        if curves_df is not None and not curves_df.empty:
            render_training_curves(curves_df)
        else:
            st.info(
                "No training curves (results.csv) found for this run. "
                "YOLO saves this automatically when training is run with `save=True`."
            )

    with tab_confusion:
        st.header("Confusion Matrix")
        confusion_img = None
        if is_local:
            for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
                p = local_results_root / run_name / fname
                if p.exists():
                    confusion_img = p.read_bytes()
                    break
        else:
            for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
                key = f"{results_prefix}/{run_name}/{fname}"
                confusion_img = load_image_from_s3(bucket, key)
                if confusion_img:
                    break

        if confusion_img:
            st.image(
                confusion_img,
                caption=f"Confusion matrix, {run_name}",
                use_container_width=True,
            )
        else:
            st.info(
                "No confusion matrix image found. "
                "YOLO saves this automatically during validation when `plots=True`."
            )

    st.divider()
    st.caption(
        f"Results S3 path: `s3://{bucket}/{results_prefix}/`  |  "
        f"Local path: `{local_results_root}`"
    )


if __name__ == "__main__":
    main()
