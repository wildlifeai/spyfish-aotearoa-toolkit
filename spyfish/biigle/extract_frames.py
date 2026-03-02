"""
Extract frames at MaxN peak times from source videos using ffmpeg.

Reads the selections CSV output from select_clips.py and extracts one clean JPEG
per row at the exact TimeOfMax moment (sampling_start + time_of_maxn_ms).

Also converts the corresponding YOLO bounding boxes from the raw ML CSV into
COCO-format JSON alongside the frames — ready for Biigle upload.

Separation of concerns:
  - select_clips.py  → which intervals are interesting (selections CSV)
  - extract_clips.py → cut 10-second video clips for Zooniverse
  - extract_frames.py (THIS FILE) → grab the single decisive frame for Biigle
"""
import json
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

from spyfish.config import config


# ── ffmpeg frame extraction ──────────────────────────────────────────────────

def extract_frame(video_path: str, seek_seconds: float, out_path: Path, fast: bool = True) -> bool:
    """
    Extract a single JPEG frame from a video at the given seek position using ffmpeg.

    Args:
        video_path: Path to the source video.
        seek_seconds: Absolute seek position in the video (sampling_start + time_of_max).
        out_path: Output JPEG path.
        fast: If True, use fast pre-input seek (-ss before -i). Slightly less accurate
              but 10-100x faster. Set False only if exact frame precision is critical.

    Returns:
        True if successful, False otherwise.
    """
    if fast:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{seek_seconds:.6f}",
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path),
            "-ss", f"{seek_seconds:.6f}",
            "-frames:v", "1", "-q:v", "2",
            str(out_path),
        ]

    try:
        subprocess.run(cmd, check=True)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError("Output file missing or empty after ffmpeg")
        return True
    except (subprocess.CalledProcessError, RuntimeError) as e:
        logging.error(f"ffmpeg failed for {out_path.name}: {e}")
        return False


# ── YOLO → COCO conversion ───────────────────────────────────────────────────

def yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """
    Convert YOLO center-format pixel bbox [cx, cy, w, h] to COCO [x, y, width, height] (pixels).
    """
    return [round(cx - w / 2, 2), round(cy - h / 2, 2), round(w, 2), round(h, 2)]


def build_coco_from_raw_csv(
    raw_csv_path: str,
    frame_records: list[dict],   # [{image_id, file_name, time_of_max, img_w, img_h}, ...]
    img_w: int = 0,
    img_h: int = 0,
) -> dict:
    """
    Build a minimal COCO annotation dict from the raw YOLO ML CSV.

    Annotations are filtered to the single frame nearest to each TimeOfMax.

    Args:
        raw_csv_path: Path to raw ML CSV (columns: frame, time_seconds, class, confidence, x, y, w, h).
        frame_records: List of dicts describing each image (one per selections row).
        img_w, img_h: Image dimensions (0 = unknown, will be filled from actual files later).

    Returns:
        COCO dict ready for json.dump().
    """
    raw_df = pd.read_csv(raw_csv_path) if os.path.exists(raw_csv_path) else pd.DataFrame()

    categories: dict[str, int] = {}
    images = []
    annotations = []
    ann_id = 0

    for rec in frame_records:
        img_id = rec["image_id"]
        time_of_max = rec["time_of_max"]  # seconds relative to sampling_start

        images.append({
            "id": img_id,
            "file_name": rec["file_name"],
            "width": rec.get("img_w", img_w),
            "height": rec.get("img_h", img_h),
            "time_of_max": time_of_max,
            "drop_id": rec.get("drop_id", ""),
            "selection_reason": rec.get("selection_reason", ""),
        })

        if raw_df.empty:
            continue

        # Find the raw CSV frame closest to this time_of_max
        nearest_frame_rows = raw_df.iloc[(raw_df["time_seconds"] - time_of_max).abs().argsort()[:1]]
        if nearest_frame_rows.empty:
            continue

        nearest_time = nearest_frame_rows["time_seconds"].iloc[0]
        frame_rows = raw_df[raw_df["time_seconds"] == nearest_time]

        for _, row in frame_rows.iterrows():
            cls_name = str(row.get("class", "unknown"))
            if cls_name not in categories:
                categories[cls_name] = len(categories)

            # YOLO centers are absolute pixels from Ultralytics xywh
            bbox = yolo_to_coco_bbox(row["x"], row["y"], row["w"], row["h"])

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": categories[cls_name],
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "score": round(float(row.get("confidence", 0.0)), 4),
                # TODO check if it works with this
            })

    return {
        "info": {"description": "Spyfish Aotearoa — ML MaxN peaks", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_id, "name": name} for name, cat_id in sorted(categories.items(), key=lambda x: x[1])],
    }


# ── main function ─────────────────────────────────────────────────────────────

def extract_frames_from_selections(
    selections_csv_path: str,
    video_path: str,
    raw_csv_path: str,
    output_dir: str,
    fast: bool = True,
) -> pd.DataFrame:
    """
    Extract one clean JPEG per row in the selections CSV at the exact MaxN peak frame,
    and produce a COCO JSON with the corresponding YOLO bounding boxes.

    The frame is grabbed at: sampling_start + TimeOfMaxSeconds
    This is the exact frame that was the deciding factor in the MaxN calculation.

    Unlike draw_frames.py (which draws boxes ON the frame using cv2 for QA),
    this extracts a clean frame with annotations stored separately as COCO JSON —
    suitable for upload to Biigle as new annotations.

    Args:
        selections_csv_path: CSV from select_clips.select_zooniverse_clips().
        video_path: Full path to the source video file.
        raw_csv_path: Raw YOLO CSV ({drop_id}_{model}_raw.csv), for COCO annotations.
        output_dir: Directory to write JPEG frames and coco_annotations.json.
        fast: Use fast ffmpeg seek (default True; trades ~1 frame accuracy for speed).

    Returns:
        selections_df with 'FramePath' column added.
    """
    if not os.path.exists(selections_csv_path):
        raise FileNotFoundError(f"Selections CSV not found: {selections_csv_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    df = pd.read_csv(selections_csv_path)
    if df.empty:
        logging.warning("Empty selections CSV. Nothing to extract.")
        df["FramePath"] = pd.Series(dtype=str)
        return df

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_id = df["DropID"].iloc[0]
    sampling_start = int(df["SamplingStart"].iloc[0]) if "SamplingStart" in df.columns else 0

    frame_records = []
    frame_paths = []

    for img_id, (_, row) in enumerate(df.iterrows(), start=1):
        # TimeOfMaxnMs: exact ML peak in seconds (sub-second precision from raw CSV).
        time_of_max_relative = float(row["TimeOfMaxnMs"])

        seek_seconds = sampling_start + time_of_max_relative

        out_filename = f"{drop_id}__frame_{time_of_max_relative:.3f}s.jpg"
        out_path = out_dir / out_filename

        logging.info(f"  [{img_id}/{len(df)}] Frame at {seek_seconds:.3f}s → {out_filename}")
        success = extract_frame(video_path, seek_seconds, out_path, fast=fast)
        frame_paths.append(str(out_path) if success else None)

        img_w, img_h = 0, 0
        if success:
            import cv2
            img = cv2.imread(str(out_path))
            if img is not None:
                img_h, img_w = img.shape[:2]

        frame_records.append({
            "image_id": img_id,
            "file_name": out_filename,
            "time_of_max": time_of_max_relative,
            "drop_id": drop_id,
            "selection_reason": row.get("SelectionReason", ""),
            "img_w": img_w,
            "img_h": img_h,
        })

    df["FramePath"] = frame_paths

    # Build and save COCO JSON
    coco = build_coco_from_raw_csv(raw_csv_path, frame_records)
    coco_path = out_dir / f"{drop_id}_coco_annotations.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    logging.info(f"COCO annotations → {coco_path} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)")

    successful = df["FramePath"].notna().sum()
    logging.info(f"Extracted {successful}/{len(df)} frames for {drop_id} → {output_dir}")
    return df
