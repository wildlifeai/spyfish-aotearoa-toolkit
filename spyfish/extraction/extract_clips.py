"""
Extract clips from source video files using ffmpeg.

Reads the selections CSV produced by selection strategies and cuts one mp4 per row.
Clip timestamps are absolute video positions (seconds from start of video file).
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.utils import generate_clip_filename


def extract_clips_from_selections(
    selections_csv_path: str,
    video_path: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Cuts one mp4 clip per row in the selections CSV using ffmpeg.

    DropID is read from the selections CSV itself.
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

    drop_id = df[config.drop_id_column].iloc[0]

    clip_paths: List[Optional[str]] = []
    for idx, row in df.iterrows():
        if config.csv_clip_start_column not in row:
            logging.error(f"Missing {config.csv_clip_start_column} in row: {row}")
            clip_paths.append(None)
            continue

        clip_start = float(row[config.csv_clip_start_column])
        # Use config-defined clip length as fallback
        clip_end = (
            float(row[config.csv_clip_end_column])
            if config.csv_clip_end_column in row
            else clip_start
            + float(
                config.get_section("zooniverse_extraction", {}).get("clip_length")
            )
        )

        clip_duration = clip_end - clip_start
        seek_seconds = clip_start

        out_filename = generate_clip_filename(drop_id, clip_duration, seek_seconds)
        out_path = Path(output_dir) / out_filename

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(seek_seconds),
            "-i",
            str(video_path),
            "-t",
            str(clip_duration),
            "-c:v",
            config.ffmpeg_codec,
            "-preset",
            config.ffmpeg_preset,
            "-crf",
            config.ffmpeg_crf,
            "-an",  # remove audio — standard for processed clips
            str(out_path),
        ]

        logging.info(f"  [{idx+1}/{len(df)}] {seek_seconds:.1f}s → {out_filename}")

        if out_path.exists():
            logging.info(f"    Skipping: File already exists at {out_path}")
            clip_paths.append(str(out_path))
            continue

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            clip_paths.append(str(out_path))
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg failed for clip {idx+1} of {drop_id}: {e}")
            clip_paths.append(None)

    df["ClipPath"] = clip_paths
    successful = df["ClipPath"].notna().sum()
    logging.info(f"Extracted {successful}/{len(df)} clips for {drop_id} → {output_dir}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract clips from selections.")
    parser.add_argument("drop_id", type=str, help="The Drop ID to process.")
    args = parser.parse_args()

    drop_id = args.drop_id
    logging.info(f"Running clip extraction for Drop ID: {drop_id}")
    selections_csv = str(config.get_selections_csv_path(drop_id))
    video_path = str(config.get_video_path(drop_id))
    output_dir = str(config.get_clips_dir(drop_id))

    extract_clips_from_selections(selections_csv, video_path, output_dir)
