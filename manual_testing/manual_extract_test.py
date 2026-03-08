import spyfish.logging
import os
import pandas as pd
from pathlib import Path
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.ml.draw_frames import draw_boxes_on_video_frames
from spyfish.config import config

## It should exist alreeady
# # Ensure we can import spyfish if run from the manual_testing folder
# import sys
# project_root = Path(__file__).parent.parent.resolve()
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))


def manual_extract():
    drop_id = "KSF_20240124_BUV_KSF_085_01"
    model_name = "cfd_binary_water_20260301"

    # Construct paths using config helpers
    selections_csv = str(config.get_selections_csv_path(drop_id))
    video_path = str(config.get_video_path(drop_id))
    raw_csv = str(config.get_raw_csv_path(drop_id, model_name))

    # Output folders
    clean_dir = project_root / "process_files/test_extraction_debug/clean"
    qa_dir = project_root / "process_files/test_extraction_debug/qa_with_boxes"

    print(f"--- Manual Extraction & QA Test ---")
    print(f"Selections: {selections_csv}")
    print(f"Video: {video_path}")
    print(f"Raw ML CSV: {raw_csv}")
    print(f"------------------------------------")

    if not os.path.exists(selections_csv):
        print(f"Error: Selections CSV missing.")
        return
    if not os.path.exists(raw_csv):
        print(f"Error: Raw CSV missing.")
        return

    # 1. Extract clean frames (for Biigle)
    print("\n[1/2] Extracting clean frames...")
    extract_frames_from_selections(
        selections_csv_path=selections_csv,
        video_path=video_path,
        raw_csv_path=raw_csv,
        output_dir=clean_dir
    )

    # 2. Extract QA frames (with boxes)
    print("\n[2/2] Drawing QA frames with boxes...")
    df = pd.read_csv(selections_csv)
    raw_df = pd.read_csv(raw_csv)

    # For each peak time, find the nearest frame index in the raw CSV
    frame_indices = []
    for _, row in df.iterrows():
        t_sec = float(row['TimeOfMaxnMs'])
        closest = raw_df.iloc[(raw_df['time_seconds'] - t_sec).abs().argsort()[:1]]
        frame_indices.append(int(closest['frame'].iloc[0]))

    draw_boxes_on_video_frames(
        video_path=video_path,
        raw_csv_path=raw_csv,
        output_dir=qa_dir,
        frame_list=frame_indices,
        confidence_threshold=config.confidence_threshold,
        sampling_start=int(df['SamplingStart'].iloc[0]) if 'SamplingStart' in df.columns else 0,
        drop_id=drop_id
    )

    print(f"\nDone!")
    print(f"Clean frames (for Biigle): {clean_dir}")
    print(f"QA frames (with boxes): {qa_dir}")

if __name__ == "__main__":
    manual_extract()
