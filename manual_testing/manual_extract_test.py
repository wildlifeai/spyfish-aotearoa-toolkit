import logging
import os

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.ml.draw_frames import draw_boxes_on_video_frames


def manual_extract():
    # Test values (update these to match your local data)
    drop_id = "KSF_20240124_BUV_KSF_085_01"
    model_name = "cfd_binary_water_20260301"

    # Construct paths using config helpers
    selections_csv = str(config.get_selections_csv_path(drop_id))
    video_path = str(config.get_video_path(drop_id))
    raw_csv = str(config.get_raw_csv_path(drop_id, model_name))

    qa_dir = config.project_root / "process_files/test_extraction_debug/qa_with_boxes"

    logging.info("-" * 40)
    logging.info("--- Manual Extraction & QA Test ---")
    logging.info(f"Selections: {selections_csv}")
    logging.info(f"Video: {video_path}")
    logging.info(f"Raw ML CSV: {raw_csv}")
    logging.info("-" * 40)

    if not os.path.exists(selections_csv):
        logging.error(f"Selections CSV missing: {selections_csv}")
        return
    if not os.path.exists(raw_csv):
        logging.error(f"Raw CSV missing: {raw_csv}")
        return

    # 1. Extract clean frames → written to canonical frames/ dir for the drop
    logging.info("\n[1/2] Extracting clean frames...")
    extract_frames_from_selections(
        selections_csv_path=selections_csv,
        video_path=video_path,
        raw_csv_path=raw_csv,
    )

    # 2. Extract QA frames (with boxes)
    logging.info("\n[2/2] Drawing QA frames with boxes...")
    df = pd.read_csv(selections_csv)
    raw_df = pd.read_csv(raw_csv)

    # For each peak time, find the nearest frame index in the raw CSV
    frame_indices = []
    for _, row in df.iterrows():
        t_sec = float(row["TimeOfMaxnMs"])
        closest = raw_df.iloc[(raw_df["time_seconds"] - t_sec).abs().argsort()[:1]]
        frame_indices.append(int(closest["frame"].iloc[0]))

    draw_boxes_on_video_frames(
        video_path=video_path,
        raw_csv_path=raw_csv,
        output_dir=qa_dir,
        frame_list=frame_indices,
        confidence_threshold=config.confidence_threshold,
        drop_id=drop_id,
    )

    logging.info("\nDone!")
    logging.info(f"Clean frames: {config.get_frames_dir(drop_id)}")
    logging.info(f"QA frames (with boxes): {qa_dir}")


if __name__ == "__main__":
    manual_extract()
