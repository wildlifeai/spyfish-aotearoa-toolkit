"""
Extract clips from source video files using ffmpeg.

Reads the selections CSV produced by selection strategies and cuts one mp4 per row.
Clip timestamps are absolute video positions (seconds from start of video file).
"""

import logging
import math
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.utils import generate_clip_filename


def _extract_clip(video_path: str, seek_seconds: float, duration: float, output_path: Path, crf: int):
    """Extract a single clip at the given CRF. Raises subprocess.CalledProcessError on failure."""
    if output_path.exists():
        output_path.unlink()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(seek_seconds),
                "-i", str(video_path),
                "-t", str(duration),
                "-c:v", config.ffmpeg_codec,
                "-preset", config.ffmpeg_preset,
                "-crf", str(crf),
                "-an",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        size_mb = output_path.stat().st_size / (1 << 20)
        logging.info(f"CRF probe: CRF {crf} → {size_mb:.1f} MB")
    except subprocess.CalledProcessError as e:
        logging.error(f"CRF probe failed for CRF {crf}: {e.stderr}")
        size_mb = float("inf")


def _probe_crf(
    video_path: str,
    seek_seconds: float,
    duration: float,
    output_path: Path,
    size_limit_mb: float,
    base_crf: int,
) -> int:
    """Extract the first clip at base_crf, then calculate the exact CRF needed if too large.

    Uses the log₂ relationship between CRF and file size (each +6 CRF ≈ halves size) to
    compute the target CRF in one step rather than trial-and-error. At most two ffmpeg calls.
    The winning clip stays on disk — the main loop's exists() check skips it.
    """
    _extract_clip(video_path, seek_seconds, duration, output_path, base_crf)
    if not output_path.exists():
        logging.error("CRF probe: initial extraction failed. Falling back to base CRF.")
        return base_crf
    size_mb = output_path.stat().st_size / (1 << 20)
    logging.info(f"CRF probe: CRF {base_crf} → {size_mb:.1f} MB")

    if size_mb < size_limit_mb:
        return base_crf

    # Calculate required CRF: each +6 halves the size → delta = 6 * log₂(size / limit)
    # math.ceil ensures we land under the limit; +1 adds a small safety margin
    target_crf = min(math.ceil(base_crf + 6 * math.log2(size_mb / size_limit_mb)) + 1, 51)
    logging.info(f"CRF probe: {size_mb:.1f} MB over limit — recalculating to CRF {target_crf}")

    _extract_clip(video_path, seek_seconds, duration, output_path, target_crf)
    if output_path.exists():
        final_size_mb = output_path.stat().st_size / (1 << 20)
        logging.info(f"CRF probe: CRF {target_crf} → {final_size_mb:.1f} MB")
        if final_size_mb >= size_limit_mb:
            logging.warning(
                f"CRF probe: still {final_size_mb:.1f} MB at CRF {target_crf} "
                "(log₂ estimate was approximate) — upload may be rejected by Zooniverse."
            )
    else:
        logging.error(f"CRF probe: extraction failed at CRF {target_crf} — upload may be rejected.")

    return target_crf


def extract_clips_from_selections(
    selections_csv_path: str,
    video_path: str,
) -> pd.DataFrame:
    """
    Cuts one mp4 clip per row in the selections CSV using ffmpeg.

    DropID is read from the selections CSV itself. Clips are written to the
    canonical clips/ directory for the drop. Already-extracted clips are skipped.

    Args:
        selections_csv_path: CSV produced by select_clips.select_zooniverse_clips().
        video_path: Full path to the source video file.

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

    drop_id = df[config.drop_id_column].iloc[0]

    output_dir = str(config.get_clips_dir(drop_id))
    os.makedirs(output_dir, exist_ok=True)

    # Calibrate CRF using the highest-MaxN clip — most fish = most detail = largest file.
    # This gives a conservative (worst-case) CRF for the whole batch.
    # The loop's exists() check will skip it — extracted exactly once, no waste.
    # On re-runs, if the probe clip already exists and is under the limit, skip entirely.
    col = config.csv_max_interval_column
    if col in df.columns and df[col].notna().any():
        probe_row = df.loc[df[col].idxmax()]
    else:
        probe_row = df.iloc[0]
    first = probe_row
    probe_sampling_start = float(first.get(config.csv_sampling_start_column, 0))
    probe_start = float(first[config.csv_clip_start_column])
    probe_end = (
        float(first[config.csv_clip_end_column])
        if config.csv_clip_end_column in first
        else probe_start + config.clip_length
    )
    probe_duration = probe_end - probe_start
    probe_seek = probe_sampling_start + probe_start
    probe_path = Path(output_dir) / generate_clip_filename(drop_id, probe_duration, probe_seek)

    base_crf = int(config.ffmpeg_crf)
    if probe_path.exists() and probe_path.stat().st_size / (1 << 20) < config.size_limit_mb:
        logging.info("CRF probe: probe clip already exists and is under limit — skipping probe.")
        effective_crf = base_crf
    else:
        effective_crf = _probe_crf(
            video_path=video_path,
            seek_seconds=probe_seek,
            duration=probe_duration,
            output_path=probe_path,
            size_limit_mb=config.size_limit_mb,
            base_crf=base_crf,
        )
    if effective_crf != base_crf:
        logging.info(
            f"Using CRF {effective_crf} (up from {base_crf}) to stay under "
            f"{config.size_limit_mb} MB per clip."
        )

    clip_paths: List[Optional[str]] = []
    for idx, row in df.iterrows():
        if config.csv_clip_start_column not in row:
            logging.error(f"Missing {config.csv_clip_start_column} in row: {row}")
            clip_paths.append(None)
            continue

        sampling_start = float(row.get(config.csv_sampling_start_column, 0))
        clip_start_relative = float(row[config.csv_clip_start_column])
        # Use config-defined clip length as fallback
        clip_end_relative = (
            float(row[config.csv_clip_end_column])
            if config.csv_clip_end_column in row
            else clip_start_relative + config.clip_length
        )

        clip_duration = clip_end_relative - clip_start_relative
        seek_seconds = sampling_start + clip_start_relative

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
            str(effective_crf),
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

    extract_clips_from_selections(selections_csv, video_path)
