"""
Post-ML annotation processing: MaxN extraction and optional frame drawing for QA review.
This module is the single entry point called by the pipeline runner after YOLO inference.
"""
import logging
import pandas as pd
from pathlib import Path

from spyfish.config import config
from spyfish.database.manager import DatabaseManager
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.ml.draw_frames import draw_boxes_on_video_frames
from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline

from spyfish.utils import seconds_to_time


def process_maxn(raw_df, output_csv_path, drop_id,
                 interval_seconds, confidence_threshold, model_name):
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
    logging.debug(f"Processing MaxN for {drop_id} (Interval: {interval_seconds}s, MaxN Threshold: {confidence_threshold})")

    if raw_df.empty:
        logging.warning(f"Empty CSV for {drop_id}")
        return pd.DataFrame()

    # Bin each detection into its interval
    raw_df = raw_df.copy()
    raw_df['interval_start'] = (raw_df['time_seconds'] // interval_seconds) * interval_seconds

    # Only count detections above the confidence threshold
    df_filtered = raw_df[raw_df['confidence'] >= confidence_threshold]

    if df_filtered.empty:
        logging.warning(f"No detections above threshold {confidence_threshold} for {drop_id}")
        return pd.DataFrame()

    # Count detections per frame per class
    frame_counts = df_filtered.groupby(['interval_start', 'class', 'frame']).size().reset_index(name='count')

    maxn_results = []
    for (interval, cls), group in df_filtered.groupby(['interval_start', 'class']):
        counts_in_interval = frame_counts[
            (frame_counts['interval_start'] == interval) &
            (frame_counts['class'] == cls)
        ]

        max_count = counts_in_interval['count'].max()
        peak_frames = counts_in_interval[counts_in_interval['count'] == max_count]['frame'].tolist()

        # Break ties by highest mean confidence across ALL boxes in that frame
        best_confidence = -1
        best_frame = peak_frames[0]
        best_second = 0

        for frame_idx in peak_frames:
            all_boxes_in_frame = group[group['frame'] == frame_idx]
            mean_conf_all = all_boxes_in_frame['confidence'].mean()

            if mean_conf_all > best_confidence:
                best_confidence = mean_conf_all
                best_frame = frame_idx
                best_second = all_boxes_in_frame['time_seconds'].iloc[0]

        maxn_results.append({
            'DropID': drop_id,
            'ScientificName': cls,
            'TimeOfMax': seconds_to_time(best_second),
            'MaxInterval': max_count,
            'AnnotatedBy': model_name,
            'Interval_annotation': interval_seconds,
            'ConfidenceAgreement': round(best_confidence, 4),
            'time_of_maxn_ms': best_second   # float seconds of the MaxN peak frame (sub-second precision)
        })

    maxn_df = pd.DataFrame(maxn_results)

    if not maxn_df.empty:
        maxn_df = maxn_df.sort_values(['time_of_maxn_ms', 'ScientificName'])

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    maxn_df.to_csv(output_csv_path, index=False)
    logging.info(f" Saved MaxN {len(maxn_df)} rows for {drop_id} to {output_csv_path}")

    return maxn_df

def _ingest_ml_annotations(ann_db: AnnotationDatabaseManager, drop_id: str, maxn_df: pd.DataFrame, model_name: str):
    """Extracts ingestion logic into a helper method."""
    annotations_to_add = []
    for _, row in maxn_df.iterrows():
        annotations_to_add.append({
            "drop_id": drop_id,
            "scientific_name": row["ScientificName"],
            "time_of_max": row["TimeOfMax"],
            "max_interval": row["MaxInterval"],
            "annotated_by": "ml",
            "interval_annotation": "",
            "confidence_agreement": row["ConfidenceAgreement"],
            "external_id": model_name
        })

    if annotations_to_add:
        # Clear previous ML syncs for this drop
        ann_db.clear_annotations(drop_id, "ml")
        ann_db.add_annotations(annotations_to_add)
        logging.debug(f"Ingested {len(annotations_to_add)} ML annotations into detailed database for {drop_id}")


def _run_qa_visualizations(raw_df: pd.DataFrame, maxn_df: pd.DataFrame, drop_id: str,
                            video_dir: Path, output_root: Path, base_conf: float,
                            maxn_conf: float, interval: float, sampling_start: int,
                            raw_csv_path: str = None):
    """Draw MaxN timeline plot and lowest-confidence annotated frames for human QA review."""
    # Save MaxN timeline visualisation
    plot_maxn_timeline(
        raw_df=raw_df,
        maxn_df=maxn_df,
        drop_id=drop_id,
        output_dir=output_root / drop_id,
        base_conf=base_conf,
        maxn_conf=maxn_conf,
        interval_seconds=interval,
    )

    # Draw lowest-confidence frames for QA review
    review_df = maxn_df.sort_values('ConfidenceAgreement').head(10)

    # Map time_of_maxn_ms back to raw CSV frame numbers
    frame_indices = []
    for t_sec in review_df['time_of_maxn_ms']:
        closest = raw_df.iloc[(raw_df['time_seconds'] - t_sec).abs().argsort()[:1]]
        frame_indices.append(int(closest['frame'].iloc[0]))

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
        sampling_start=sampling_start,
        drop_id=drop_id
    )

def run_post_ml(drop_ids: list, annotations_dir: str, video_dir: str,
                output_root: str, draw_images: bool = True):
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
    db = DatabaseManager()
    model_name = config.pipeline_model_path
    base_conf = float(config.confidence_threshold)
    maxn_conf = float(config.maxn_confidence_threshold)
    interval = config.interval_seconds
    annotations_dir = Path(annotations_dir)
    video_dir = Path(video_dir)
    output_root = Path(output_root)
    ann_db = AnnotationDatabaseManager()

    for drop_id in drop_ids:
        logging.debug(f"Post-ML processing: {drop_id}")

        drop_annotations_dir = output_root / drop_id / "annotations"
        drop_annotations_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = str(drop_annotations_dir / f"{drop_id}_{model_name}_raw.csv")
        maxn_csv = str(drop_annotations_dir / f"{drop_id}_{model_name}_maxn.csv")

        # Read raw CSV once — shared by process_maxn and draw_frames lookup
        raw_df = pd.read_csv(raw_csv)

        # Get sampling_start from DB. We do this in a short-lived block to prevent holding locks.
        db = DatabaseManager()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT sampling_start FROM deployments WHERE drop_id = ?', (drop_id,))
            row = cursor.fetchone()
            sampling_start = dict(row)['sampling_start'] if row and row['sampling_start'] else 0

        # 1. Extract MaxN (uses higher threshold than base inference)
        maxn_df = process_maxn(
            raw_df, maxn_csv, drop_id,
            interval_seconds=interval,
            confidence_threshold=maxn_conf,
            model_name=model_name
        )

        if maxn_df.empty:
            logging.warning(f"No MaxN results for {drop_id}. Saving empty CSV for health checks.")
            # Create a placeholder empty MaxN CSV with the correct headers
            # TODO how ate these the correct headers
            maxn_df = pd.DataFrame(columns=['DropID', 'ScientificName', 'TimeOfMax', 'MaxInterval', 'AnnotatedBy', 'Interval_annotation', 'ConfidenceAgreement', 'time_of_maxn_ms'])
            maxn_df.to_csv(maxn_csv, index=False)

        # 2. Ingest into detailed annotations database
        _ingest_ml_annotations(ann_db, drop_id, maxn_df, model_name)

        if not draw_images:
            logging.info(f"Post-ML processing complete for {drop_id} (frame & plot drawing skipped)")
            continue

        # 3. Draw QA visualisations (MaxN timeline + lowest-confidence frames)
        _run_qa_visualizations(
            raw_df=raw_df,
            maxn_df=maxn_df,
            drop_id=drop_id,
            video_dir=video_dir,
            output_root=output_root,
            base_conf=base_conf,
            maxn_conf=maxn_conf,
            interval=interval,
            sampling_start=sampling_start,
            raw_csv_path=raw_csv,
        )

        logging.info(f"  → Post-ML processing complete for: {drop_id}")

        # Update status to AWAITING_CITSCI_CLIPS
        db.update_status(drop_id, config.PipelineStatus.AWAITING_CITSCI_CLIPS)

    # 4. Finally sync all updated drops to the main pipeline DB
    if drop_ids:
        db.sync_annotation_counts(drop_ids)
        logging.info(f"Synchronized annotation counts for {len(drop_ids)} drops to main pipeline DB")
