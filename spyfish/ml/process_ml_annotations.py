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
from spyfish.orchestrator.ingest_legacy import sync_annotations_to_main_db
from spyfish.ml.draw_frames import draw_boxes_on_frames
from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline


def seconds_to_time(seconds):
    """Converts seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


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
    logging.info(f"Processing MaxN for {drop_id}")
    logging.info(f"Settings - Interval: {interval_seconds}s, Threshold: {confidence_threshold}")

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
    logging.info(f"Saved MaxN results: {len(maxn_df)} rows to {output_csv_path}")

    return maxn_df


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
    model_name = Path(config.model_path or config.mock_model_path).stem
    base_conf = float(config.confidence_threshold)
    maxn_conf = float(config.maxn_confidence_threshold)
    interval = config.interval_seconds
    annotations_dir = Path(annotations_dir)
    output_root = Path(output_root)

    for drop_id in drop_ids:
        logging.info(f"Post-ML processing: {drop_id}")

        raw_csv = str(annotations_dir / f"{drop_id}_{model_name}_raw.csv")
        maxn_csv = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")

        # Read raw CSV once — shared by process_maxn and draw_frames lookup
        raw_df = pd.read_csv(raw_csv)

        # Get sampling_start from DB
        deployment = db.get_deployment(drop_id)
        sampling_start = deployment['sampling_start'] if deployment and deployment['sampling_start'] else 0

        # 1. Extract MaxN (uses higher threshold than base inference)
        maxn_df = process_maxn(
            raw_df, maxn_csv, drop_id,
            interval_seconds=interval,
            confidence_threshold=maxn_conf,
            model_name=model_name
        )

        if maxn_df.empty:
            logging.warning(f"No MaxN results for {drop_id}, skipping ingestion and QA drawing")
            continue

        # 2. Ingest into detailed annotations database
        ann_db = AnnotationDatabaseManager()
        annotations_to_add = []
        for _, row in maxn_df.iterrows():
            annotations_to_add.append({
                "drop_id": drop_id,
                "scientific_name": row["ScientificName"],
                "timestamp": row["TimeOfMax"],
                "count": row["MaxInterval"],
                "source": "ml",
                "confidence": row["ConfidenceAgreement"],
                "external_id": model_name
            })

        if annotations_to_add:
            with ann_db.get_connection() as conn:
                # TODO do we want this? maybe some comparison?
                # Clear previous ML syncs for this drop
                conn.execute("DELETE FROM annotations WHERE drop_id = ? AND source = 'ml'", (drop_id,))
                ann_db.add_annotations(annotations_to_add)
            logging.info(f"Ingested {len(annotations_to_add)} ML annotations into detailed database for {drop_id}")

        if not draw_images:
            logging.info(f"Post-ML processing complete for {drop_id} (frame & plot drawing skipped)")
            continue

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


        # 2. Draw lowest-confidence frames for QA review
        review_df = maxn_df.sort_values('ConfidenceAgreement').head(10)

        # Map time_of_maxn_ms back to raw CSV frame numbers
        frame_indices = []
        for t_sec in review_df['time_of_maxn_ms']:
            closest = raw_df.iloc[(raw_df['time_seconds'] - t_sec).abs().argsort()[:1]]
            frame_indices.append(int(closest['frame'].iloc[0]))

        # Find video file
        video_path = Path(video_dir) / f"{drop_id}.mp4"
        if not video_path.exists():
            logging.warning(f"Video not found at {video_path}, skipping frame drawing")
            continue

        frames_dir = str(output_root / drop_id / "frames")
        draw_boxes_on_frames(
            video_path=str(video_path),
            raw_csv_path=raw_csv,
            output_dir=frames_dir,
            frame_list=frame_indices,
            confidence_threshold=base_conf,
            sampling_start=sampling_start
        )

        logging.info(f"Post-ML processing complete for {drop_id}")

    # 3. Finally sync all updated drops to the main pipeline DB
    if drop_ids:
        sync_annotations_to_main_db(drop_ids)
        logging.info(f"Synchronized annotation counts for {len(drop_ids)} drops to main pipeline DB")
