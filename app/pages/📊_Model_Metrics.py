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
import io
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from spyfish.config import config
from utils import check_password


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def list_result_dirs_from_s3(bucket: str, results_prefix: str) -> list[str]:
    """List available result directories in S3."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{bucket}/{results_prefix.rstrip('/')}/"],
            capture_output=True, text=True, timeout=15,
        )
        dirs = []
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if parts and parts[-1].endswith("/"):
                dirs.append(parts[-1].rstrip("/"))
        return sorted(dirs, reverse=True)  # newest first
    except Exception as e:
        logging.warning(f"Could not list S3 result dirs: {e}")
        return []


@st.cache_data(ttl=300)
def load_metrics_csv_from_s3(bucket: str, s3_key: str) -> Optional[pd.DataFrame]:
    """Download and parse a metrics CSV from S3."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{s3_key}", "-"],
            capture_output=True, timeout=20,
        )
        if result.returncode != 0:
            return None
        return pd.read_csv(io.BytesIO(result.stdout))
    except Exception as e:
        logging.warning(f"Could not load metrics CSV {s3_key}: {e}")
        return None


@st.cache_data(ttl=300)
def load_yolo_results_csv_from_s3(bucket: str, s3_key: str) -> Optional[pd.DataFrame]:
    """Load YOLO's results.csv (training curves) from S3."""
    return load_metrics_csv_from_s3(bucket, s3_key)


@st.cache_data(ttl=300)
def load_image_from_s3(bucket: str, s3_key: str) -> Optional[bytes]:
    """Download an image from S3 and return raw bytes."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{s3_key}", "-"],
            capture_output=True, timeout=20,
        )
        if result.returncode != 0:
            return None
        return result.stdout
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
    display_cols = [c for c in ["role", "mAP50", "mAP50_95", "precision", "recall", "model_path"] if c in metrics_df.columns]
    st.dataframe(
        metrics_df[display_cols].rename(columns={
            "role": "Model",
            "mAP50": "mAP@0.5",
            "mAP50_95": "mAP@0.5:0.95",
            "precision": "Precision",
            "recall": "Recall",
            "model_path": "Weights Path",
        }),
        width="stretch",
        hide_index=True,
        column_config={
            "mAP@0.5":       st.column_config.NumberColumn(format="%.4f"),
            "mAP@0.5:0.95":  st.column_config.NumberColumn(format="%.4f"),
            "Precision":     st.column_config.NumberColumn(format="%.4f"),
            "Recall":        st.column_config.NumberColumn(format="%.4f"),
        },
    )


def render_training_curves(results_df: pd.DataFrame) -> None:
    """Plot training loss and mAP over epochs using Streamlit's native chart."""
    import re

    results_df.columns = [c.strip() for c in results_df.columns]
    epoch_col = next((c for c in results_df.columns if "epoch" in c.lower()), None)
    if epoch_col is None:
        st.info("No 'epoch' column found in results.csv.")
        return

    map_cols = [c for c in results_df.columns if "map50" in c.lower() and "95" not in c.lower()]
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
    bucket: str,
    s3_prefix: str,
    model_type: str,
) -> None:
    """Render a promote button that copies the selected model to the production S3 path."""
    st.divider()
    st.subheader("🚀 Promote Model")
    st.caption(
        f"Copy this model to `{s3_prefix.rstrip('/')}/{model_type}_current.pt` — "
        f"the pipeline will use it on next run."
    )

    confirm = st.checkbox("I have reviewed the metrics and want to promote this model", key="confirm_promote")
    if st.button("✅ Promote to Production", disabled=not confirm, type="primary"):
        target_key = s3_prefix.rstrip("/") + f"/{model_type}_current.pt"
        src_uri = f"s3://{bucket}/{model_path.lstrip('/')}"
        dst_uri = f"s3://{bucket}/{target_key}"
        with st.spinner(f"Promoting {model_path} → {target_key} ..."):
            result = subprocess.run(
                ["aws", "s3", "cp", src_uri, dst_uri, "--only-show-errors"],
                capture_output=True, text=True, timeout=60,
            )
        if result.returncode == 0:
            st.success(f"✅ Model promoted to production: `{dst_uri}`")
            st.cache_data.clear()
        else:
            st.error(f"Promotion failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Model Metrics", page_icon="📊", layout="wide")
    if not check_password():
        st.stop()

    render_sidebar_refresh()

    st.title("📊 Model Metrics")
    st.caption("Review and compare trained ML model performance before promoting to production.")

    training_cfg = config._yaml_config.get("training", {})
    storage_cfg = config._yaml_config.get("storage", {})
    bucket = storage_cfg.get("bucket_name", config.s3_bucket)
    results_prefix = "process_files/training/results"
    output_s3_prefix = training_cfg.get("output_model_s3_prefix", "process_files/models/pipeline_model/")
    local_training_dir = Path(training_cfg.get("local_training_dir", "process_files/training"))

    # --- Sidebar: select result run ---
    st.sidebar.header("Select Run")

    result_dirs = list_result_dirs_from_s3(bucket, results_prefix)

    # Also check local results as a fallback
    local_result_dirs = []
    local_results_root = local_training_dir / "results"
    if local_results_root.exists():
        local_result_dirs = [d.name for d in sorted(local_results_root.iterdir(), reverse=True) if d.is_dir()]

    all_runs = result_dirs + [f"[local] {d}" for d in local_result_dirs if d not in result_dirs]

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
            new_row = metrics_df[metrics_df.get("role", pd.Series()) == "new"]
            prod_row = metrics_df[metrics_df.get("role", pd.Series()) == "production"]
            if not new_row.empty and not prod_row.empty:
                delta = float(new_row["mAP50"].iloc[0]) - float(prod_row["mAP50"].iloc[0])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New mAP@0.5",  f"{float(new_row['mAP50'].iloc[0]):.4f}")
                with col2:
                    st.metric("Production mAP@0.5", f"{float(prod_row['mAP50'].iloc[0]):.4f}")
                with col3:
                    min_improvement = training_cfg.get("retrain_min_improvement_pct", 2.0) / 100.0
                    st.metric(
                        "Improvement",
                        f"{delta:+.4f}",
                        delta=f"{'✅ above' if delta >= min_improvement else '❌ below'} {min_improvement:.1%} threshold",
                    )
        else:
            st.info("No metrics CSV found for this run.")

        # Promote button (only if we have a new model path)
        if metrics_df is not None and "model_path" in metrics_df.columns:
            new_rows = metrics_df[metrics_df.get("role", pd.Series(dtype=str)) == "new"]
            if not new_rows.empty:
                new_model_path = str(new_rows["model_path"].iloc[0])
                render_promote_button(new_model_path, bucket, output_s3_prefix, model_type)

    with tab_curves:
        st.header("Training Curves")
        curves_df = None
        if is_local:
            local_csv = local_results_root / run_name / "results.csv"
            if local_csv.exists():
                curves_df = pd.read_csv(local_csv)
        else:
            curves_key = f"{results_prefix}/{run_name}/results.csv"
            curves_df = load_yolo_results_csv_from_s3(bucket, curves_key)

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
            st.image(confusion_img, caption=f"Confusion matrix — {run_name}", use_container_width=True)
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
