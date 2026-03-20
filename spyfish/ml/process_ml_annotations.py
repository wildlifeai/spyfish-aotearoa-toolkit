from typing import Optional

"""
Post-ML annotation processing: MaxN extraction and optional frame drawing for QA review.
This module is the single entry point called by the pipeline runner after YOLO inference.
"""

import logging
from pathlib import Path

import pandas as pd

from spyfish.config.base import PipelineStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.ml.draw_frames import draw_boxes_on_video_frames
from spyfish.utils import seconds_to_time
from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline


def process_maxn(
    raw_df, output_csv_path, drop_id, interval_seconds, confidence_threshold, model_name
):
    """
    Processes raw ML detections into MaxN intervals.

    For each time interval × class combination, finds the frame with the highest count
    of confident detections. Breaks ties by mean confidence across ALL detections in the frame.

    Args:
        raw_df: DataFrame of raw ML detections (avoids double CSV read).
        output_csv_path: Path to save MaxN CSV results.
        drop_id: Deployment identifier.
        interval_seconds: Width of each MaxN time window.
        confidence_threshold: Minimum confidence for counting fish.
        model_name: Name of the model used for annotation.

    Returns:
        DataFrame with MaxN results.
    """
    logging.debug(
        f"Processing MaxN for {drop_id} (Interval: {interval_seconds}s, MaxN Threshold: {confidence_threshold})"
    )

    if raw_df.empty:
        logging.warning(f"Empty CSV for {drop_id}")
        return pd.DataFrame()

    # Bin each detection into its interval
    raw_df = raw_df.copy()
    raw_df["interval_start"] = (
        raw_df["time_seconds"] // interval_seconds
    ) * interval_seconds

    # Only count detections above the confidence threshold
    df_filtered = raw_df[raw_df["confidence"] >= confidence_threshold]

    if df_filtered.empty:
        logging.warning(
            f"No detections above threshold {confidence_threshold} for {drop_id}"
        )
        return pd.DataFrame()

    # Count detections per frame per class
    frame_counts = (
        df_filtered.groupby(["interval_start", "class", "frame"])
        .size()
        .reset_index(name="count")
    )

    maxn_results = []
    for (interval, cls), group in df_filtered.groupby(["interval_start", "class"]):
        counts_in_interval = frame_counts[
            (frame_counts["interval_start"] == interval)
            & (frame_counts["class"] == cls)
        ]

        max_count = counts_in_interval["count"].max()
        peak_frames = counts_in_interval[counts_in_interval["count"] == max_count][
            "frame"
        ].tolist()

        # Break ties by highest mean confidence across ALL boxes in that frame
        best_confidence = -1
        best_second = 0

        for frame_idx in peak_frames:
            all_boxes_in_frame = group[group["frame"] == frame_idx]
            mean_conf_all = all_boxes_in_frame["confidence"].mean()

            if mean_conf_all > best_confidence:
                best_confidence = mean_conf_all
                best_second = all_boxes_in_frame["time_seconds"].iloc[0]

        maxn_results.append(
            {
                config.drop_id_column: drop_id,
                config.csv_scientific_name_column: cls,
                config.csv_maxn_time_column: seconds_to_time(best_second),
                config.csv_max_interval_column: max_count,
                config.csv_annotated_by_column: model_name,
                config.csv_interval_annotation_column: interval_seconds,
                config.csv_confidence_agreement_column: round(best_confidence, 4),
                config.csv_maxn_time_ms_column: best_second,  # float seconds of the MaxN peak frame (sub-second precision)
            }
        )

    maxn_df = pd.DataFrame(maxn_results)

    if not maxn_df.empty:
        maxn_df = maxn_df.sort_values(
            [
                config.csv_maxn_time_ms_column,
                config.csv_scientific_name_column,
            ]
        )

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    maxn_df.to_csv(output_csv_path, index=False)
    logging.info(f" Saved MaxN {len(maxn_df)} rows for {drop_id} to {output_csv_path}")

    return maxn_df


def _ingest_ml_annotations(
    ann_db: AnnotationDatabaseManager,
    drop_id: str,
    maxn_df: pd.DataFrame,
    model_name: str,
):
    """Extracts ingestion logic into a helper method."""
    annotations_to_add = []
    for _, row in maxn_df.iterrows():
        annotations_to_add.append(
            {
                "drop_id": drop_id,
                "scientific_name": row[config.csv_scientific_name_column],
                "time_of_max": row[config.csv_maxn_time_column],
                "max_interval": row[config.csv_max_interval_column],
                "annotated_by": "ml",
                "interval_annotation": "",
                "confidence_agreement": row[config.csv_confidence_agreement_column],
                "external_id": model_name,
            }
        )
    # TODO check if this is wanted behaviour.
    # Always clear previous ML annotations before writing new ones.
    # If annotations_to_add is empty (zero detections above threshold), we still
    # need to wipe stale rows from any prior run — skipping the clear would leave
    # the old count in the DB while the MaxN CSV on disk shows zero detections.
    ann_db.clear_annotations(drop_id, "ml")
    if annotations_to_add:
        ann_db.add_annotations(annotations_to_add)
        logging.debug(
            f"Ingested {len(annotations_to_add)} ML annotations into detailed database for {drop_id}"
        )


def _run_qa_visualizations(
    raw_df: pd.DataFrame,
    maxn_df: pd.DataFrame,
    drop_id: str,
    video_dir: Path,
    output_root: Path,
    base_conf: float,
    maxn_conf: float,
    interval: float,
    raw_csv_path: Optional[str] = None,
):
    """Draw MaxN timeline plot and lowest-confidence annotated frames for human QA review."""
    # Save MaxN timeline visualisation
    plot_maxn_timeline(
        raw_df=raw_df,
        maxn_df=maxn_df,
        drop_id=drop_id,
        output_dir=output_root / drop_id,
        base_conf=base_conf,
        maxn_conf=maxn_conf,
        interval_seconds=interval,  # type: ignore
    )

    # Draw lowest-confidence frames for QA review
    if raw_df.empty or maxn_df.empty:
        logging.debug(f"Skipping QA frame drawing for {drop_id}: no raw detections.")
        return

    review_df = maxn_df.sort_values(config.csv_confidence_agreement_column).head(10)

    # Map time_of_maxn_ms back to raw CSV frame numbers
    frame_indices = []
    for t_sec in review_df[config.csv_maxn_time_ms_column]:
        closest = raw_df.iloc[(raw_df["time_seconds"] - t_sec).abs().argsort()[:1]]
        if closest.empty:
            continue
        frame_indices.append(int(closest["frame"].iloc[0]))

    # Find video file
    video_path = video_dir / f"{drop_id}.mp4"
    if not video_path.exists():
        logging.warning(f"Video not found at {video_path}, skipping QA frame drawing")
        return

    frames_dir = str(output_root / drop_id / "qa_frames")
    draw_boxes_on_video_frames(
        video_path=video_path,
        raw_csv_path=raw_csv_path,
        output_dir=frames_dir,
        frame_list=frame_indices,
        confidence_threshold=base_conf,
        drop_id=drop_id,
    )


def run_post_ml(
    drop_ids: list,
    annotations_dir: str,
    video_dir: str,
    output_root: str,
    draw_images: bool = True,
):
    """
    For each processed drop:
    1. Extracts MaxN intervals from the raw YOLO CSV
    2. Optionally draws bounding boxes on the lowest-confidence frames for human QA review

    Args:
        drop_ids: List of DropIDs that were successfully processed by YOLO.
        annotations_dir: Directory containing the raw YOLO CSVs.
        video_dir: Directory containing the source video files.
        output_root: Root directory for data quality outputs.
        draw_images: Whether to draw QA review frames (default: True).
    """
    model_name = Path(config.pipeline_model_path).stem

    # Initialize shared resources
    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()

    # Pre-fetch configuration values
    interval = config.interval_seconds
    base_conf = config.confidence_threshold
    maxn_conf = config.maxn_confidence_threshold

    for drop_id in drop_ids:
        logging.debug(f"Post-ML processing: {drop_id}")

        drop_annotations_dir = config.get_drop_annotations_dir(drop_id)
        drop_annotations_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = str(drop_annotations_dir / f"{drop_id}_{model_name}_raw.csv")
        maxn_csv = str(drop_annotations_dir / f"{drop_id}_{model_name}_maxn.csv")

        # Read raw CSV once — shared by process_maxn and draw_frames lookup
        if not Path(raw_csv).exists():
            logging.warning(f"Raw CSV not found for {drop_id} at {raw_csv}. Skipping.")
            continue

        raw_df = pd.read_csv(raw_csv)
        # 1. Extract MaxN (uses higher threshold than base inference)
        maxn_df = process_maxn(
            raw_df,
            maxn_csv,
            drop_id,
            interval_seconds=interval,
            confidence_threshold=maxn_conf,
            model_name=model_name,
        )

        if maxn_df.empty:
            logging.warning(
                f"No MaxN results for {drop_id}. Saving empty CSV for health checks."
            )
            # Create a placeholder empty MaxN CSV with the correct headers
            maxn_df = pd.DataFrame(
                columns=[
                    config.drop_id_column,
                    config.csv_scientific_name_column,
                    config.csv_maxn_time_column,
                    config.csv_max_interval_column,
                    config.csv_annotated_by_column,
                    config.csv_interval_annotation_column,
                    config.csv_confidence_agreement_column,
                    config.csv_maxn_time_ms_column,
                ]
            )
            maxn_df.to_csv(maxn_csv, index=False)

        # 2. Ingest into detailed annotations database
        _ingest_ml_annotations(ann_db, drop_id, maxn_df, model_name)

        if draw_images:
            # 3. Draw QA visualisations (MaxN timeline + lowest-confidence frames).
            # Failures here must not block status advancement — QA viz is diagnostic only.
            try:
                _run_qa_visualizations(
                    raw_df=raw_df,
                    maxn_df=maxn_df,
                    drop_id=drop_id,
                    video_dir=Path(video_dir),
                    output_root=Path(output_root),
                    base_conf=base_conf,
                    maxn_conf=maxn_conf,
                    interval=interval,
                    raw_csv_path=raw_csv,
                )
            except Exception as e:
                logging.error(
                    f"QA visualisation failed for {drop_id} (non-fatal): {e}",
                    exc_info=True,
                )

        logging.info(f"  → Post-ML processing complete for: {drop_id}")

    # 4. Finally sync all updated drops to the main pipeline DB
    if drop_ids:
        db.sync_annotation_counts(drop_ids)
        logging.info(
            f"Synchronized annotation counts for {len(drop_ids)} drops to main pipeline DB"
        )
