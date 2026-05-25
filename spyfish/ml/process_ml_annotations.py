"""
Post-ML annotation processing: MaxN extraction and optional frame drawing for QA review.
This module is the single entry point called by the pipeline runner after YOLO inference.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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
                config.csv_maxn_time_seconds_column: best_second,
            }
        )

    maxn_df = pd.DataFrame(maxn_results)

    if not maxn_df.empty:
        maxn_df = maxn_df.sort_values(
            [
                config.csv_maxn_time_seconds_column,
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
                "time_of_max_seconds": row.get(config.csv_maxn_time_seconds_column),
                "max_interval": row[config.csv_max_interval_column],
                "annotated_by": "ml",
                "interval_annotation": "",
                "confidence_agreement": row[config.csv_confidence_agreement_column],
                "external_id": model_name,
            }
        )
    # Clear only this model's prior rows before re-writing. Scoping by
    # external_id (the model name) means re-running one model leaves any
    # other model's outputs intact in the DB — supports running both
    # binary and species pipelines on the same drop and comparing them
    # side-by-side via the dashboard's Provenance column.
    # The clear must still run when annotations_to_add is empty so a zero-
    # detection re-run wipes stale rows from a prior run of THIS model.
    ann_db.clear_annotations(drop_id, "ml", external_id=model_name)
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
        output_dir=config.get_drop_dir(drop_id),
        base_conf=base_conf,
        maxn_conf=maxn_conf,
        interval_seconds=interval,  # type: ignore
    )

    # Draw lowest-confidence frames for QA review
    if raw_df.empty or maxn_df.empty:
        logging.debug(f"Skipping QA frame drawing for {drop_id}: no raw detections.")
        return

    top_maxn = maxn_df.nlargest(4, config.csv_max_interval_column)
    low_conf = maxn_df.nsmallest(4, config.csv_confidence_agreement_column)
    review_df = pd.concat([top_maxn, low_conf]).drop_duplicates()

    # Map TimeOfMaxAbsSeconds back to raw CSV frame numbers
    frame_indices = []
    for t_sec in review_df[config.csv_maxn_time_seconds_column]:
        closest = raw_df.iloc[(raw_df["time_seconds"] - t_sec).abs().argsort()[:1]]
        if closest.empty:
            continue
        frame_indices.append(int(closest["frame"].iloc[0]))

    # 4 evenly-spaced random frames across the detected range for general coverage
    t_min, t_max = raw_df["time_seconds"].min(), raw_df["time_seconds"].max()
    if t_max > t_min:
        boundaries = np.linspace(t_min, t_max, 5)  # 4 equal bands
        for i in range(4):
            band = raw_df[
                (raw_df["time_seconds"] >= boundaries[i])
                & (raw_df["time_seconds"] < boundaries[i + 1])
            ]
            if not band.empty:
                frame_indices.append(int(band.sample(1)["frame"].iloc[0]))
    # First and last detected frames — quick visual check on detection coverage
    frame_indices.append(int(raw_df.loc[raw_df["time_seconds"].idxmin(), "frame"]))
    frame_indices.append(int(raw_df.loc[raw_df["time_seconds"].idxmax(), "frame"]))

    # First and last detected frames — quick visual check on detection coverage
    frame_indices.append(int(raw_df.loc[raw_df["time_seconds"].idxmin(), "frame"]))
    frame_indices.append(int(raw_df.loc[raw_df["time_seconds"].idxmax(), "frame"]))

    # Find video file
    video_path = video_dir / f"{drop_id}.mp4"
    if not video_path.exists():
        logging.warning(f"Video not found at {video_path}, skipping QA frame drawing")
        return

    frames_dir = str(config.get_drop_dir(drop_id) / "qa_frames")
    draw_boxes_on_video_frames(
        video_path=video_path,
        raw_csv_path=raw_csv_path,
        output_dir=frames_dir,
        frame_list=frame_indices,
        confidence_threshold=base_conf,
        drop_id=drop_id,
    )


def process_one_drop(
    drop_id: str,
    video_dir: Path,
    ann_db: AnnotationDatabaseManager,
    model_name: str,
    interval: float,
    base_conf: float,
    maxn_conf: float,
    draw_images: bool = True,
) -> None:
    """Post-inference processing for a single drop: MaxN + annotation ingest + QA viz.

    QA viz failures are swallowed and logged — they're diagnostic only and must
    not block the caller from marking the drop complete. MaxN extraction and
    annotation ingest failures propagate so the caller can mark the drop as
    errored.
    """
    drop_annotations_dir = config.get_drop_annotations_dir(drop_id)
    drop_annotations_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = str(drop_annotations_dir / f"{drop_id}_{model_name}_raw.csv")
    maxn_csv = str(config.get_maxn_csv_path(drop_id, model_name))

    if not Path(raw_csv).exists():
        raise FileNotFoundError(f"Raw CSV not found for {drop_id} at {raw_csv}")

    raw_df = pd.read_csv(raw_csv)
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
        maxn_df = pd.DataFrame(
            columns=[
                config.drop_id_column,
                config.csv_scientific_name_column,
                config.csv_maxn_time_column,
                config.csv_max_interval_column,
                config.csv_annotated_by_column,
                config.csv_interval_annotation_column,
                config.csv_confidence_agreement_column,
                config.csv_maxn_time_seconds_column,
            ]
        )
        maxn_df.to_csv(maxn_csv, index=False)

    _ingest_ml_annotations(ann_db, drop_id, maxn_df, model_name)

    if draw_images:
        try:
            _run_qa_visualizations(
                raw_df=raw_df,
                maxn_df=maxn_df,
                drop_id=drop_id,
                video_dir=video_dir,
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


def run_post_ml(
    drop_ids: list,
    video_dir: str,
    draw_images: bool = True,
):
    """
    Batch entry point — kept for REPL/notebook use. The pipeline now runs the
    per-drop work inline inside MLRunner.run_inference_loop so that artifacts
    are written before the drop is marked complete.

    For each drop: extracts MaxN, ingests annotations, optionally draws QA frames.
    Then syncs annotation counts to the main pipeline DB once at the end.
    """
    model_name = Path(config.pipeline_model_path).stem
    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()
    interval = config.interval_seconds
    base_conf = config.confidence_threshold
    maxn_conf = config.maxn_confidence_threshold

    for drop_id in drop_ids:
        logging.debug(f"Post-ML processing: {drop_id}")
        try:
            process_one_drop(
                drop_id=drop_id,
                video_dir=Path(video_dir),
                ann_db=ann_db,
                model_name=model_name,
                interval=interval,
                base_conf=base_conf,
                maxn_conf=maxn_conf,
                draw_images=draw_images,
            )
            logging.info(f"  → Post-ML processing complete for: {drop_id}")
        except Exception as e:
            logging.error(
                f"Post-ML processing failed for {drop_id}: {e}", exc_info=True
            )

    if drop_ids:
        db.sync_annotation_counts(drop_ids)
        logging.info(
            f"Synchronized annotation counts for {len(drop_ids)} drops to main pipeline DB"
        )
