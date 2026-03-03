"""
Extract Zooniverse clips from source video files using ffmpeg.

Reads the selections CSV produced by select_clips.py and cuts one mp4 per row.
Clip timestamps are stored relative to SamplingStart in the selections CSV.
sampling_start is added back to get the correct seek position in the full source video.
"""
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

from spyfish.config import config


def extract_clips_from_selections(
    selections_csv_path: str,
    video_path: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Cuts one mp4 clip per row in the selections CSV using ffmpeg.

    DropID and SamplingStart are read from the selections CSV itself.
    A 'ClipPath' column is added to the returned DataFrame so the caller
    has a single self-contained record of each clip and its metadata.

    Args:
        selections_csv_path: CSV produced by select_clips.select_zooniverse_clips().
        video_path: Full path to the source video file.
        output_dir: Directory to write clip mp4 files into.

    Returns:
        selections_df with a 'ClipPath' column added (None where ffmpeg failed).
    """
    if not os.path.exists(selections_csv_path):
        raise FileNotFoundError(f"Selections CSV not found: {selections_csv_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    df = pd.read_csv(selections_csv_path)
    if df.empty:
        logging.warning("Empty selections CSV. Nothing to extract.")
        df["ClipPath"] = pd.Series(dtype=str)
        return df

    os.makedirs(output_dir, exist_ok=True)

    drop_id = df["DropID"].iloc[0]
    sampling_start = int(df["SamplingStart"].iloc[0]) if "SamplingStart" in df.columns else 0


    clip_paths = []
    for idx, row in df.iterrows():
        clip_start_relative = float(row["ClipStartRelative"]) if "ClipStartRelative" in row else 0.0
        clip_end_relative = float(row["ClipEndRelative"]) if "ClipEndRelative" in row else clip_start_relative + 10.0
        clip_duration = clip_end_relative - clip_start_relative
        seek_seconds = sampling_start + clip_start_relative

        out_filename = f"{drop_id}_{int(clip_duration):02d}s_{int(clip_start_relative):05d}s.mp4"
        out_path = Path(output_dir) / out_filename

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_seconds),
            "-i", str(video_path),
            "-t", str(clip_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-an",   # remove audio — standard for Zooniverse clips
            str(out_path),
        ]

        logging.info(f"  [{idx+1}/{len(df)}] {seek_seconds:.1f}s → {out_filename}")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clip_paths.append(str(out_path))
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg failed for clip {idx+1} of {drop_id}: {e}")
            clip_paths.append(None)

    df["ClipPath"] = clip_paths
    successful = df["ClipPath"].notna().sum()
    logging.info(f"Extracted {successful}/{len(df)} clips for {drop_id} → {output_dir}")
    return df


def main():

    logging.info("Running clip extraction in standalone test mode.")
    repo_root = Path(__file__).parent.parent.parent
    drop_id = config.test_drops[0][0]

    annotations_dir = repo_root / config.local_manifest_dir_path
    selections_csv = str(annotations_dir / f"{drop_id}_zooniverse_selections.csv")
    video_path = str(repo_root / config.mock_video_dir / f"{drop_id}.mp4")
    output_dir = str(repo_root / config.local_data_quality_dir / drop_id / "zooniverse_clips")

    extract_clips_from_selections(selections_csv, video_path, output_dir)


if __name__ == "__main__":
    main()
