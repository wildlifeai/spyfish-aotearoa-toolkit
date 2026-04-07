"""
Extract frames at MaxN peak times from source videos using ffmpeg.

Reads the selections CSV output from selection strategies and extracts one clean JPEG
per row at the exact TimeOfMax moment (absolute video timestamp in seconds).

Also converts the corresponding YOLO bounding boxes from the raw ML CSV into
COCO-format JSON alongside the frames — ready for upload (e.g., to Biigle).

Separation of concerns:
  - select_clips.py  → which intervals are interesting (selections CSV)
  - extract_clips.py → cut video clips
  - extract_frames.py (THIS FILE) → grab the single decisive frame
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import piexif

from spyfish.config.wrapper import config
from spyfish.utils import generate_frame_filename

# ── ffmpeg frame extraction ──────────────────────────────────────────────────


_ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _read_video_rotation(cap: cv2.VideoCapture) -> int:
    """Read rotation degrees from video container metadata (0, 90, 180, or 270)."""
    try:
        degrees = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
        return degrees if degrees in _ROTATION_MAP else 0
    except Exception:
        return 0


def extract_frame(
    video_path: str,
    seek_seconds: float,
    out_path: Path,
    frame_index: Optional[int] = None,
) -> bool:
    """
    Extract a single JPEG frame from a video at the given seek position using OpenCV.
    Using cv2 (instead of ffmpeg) ensures 100% parity with the YOLO inference stream.

    Rotation metadata from the video container is read and applied to the pixel data,
    then EXIF Orientation = 1 is embedded so downstream tools (e.g. Biigle) do not
    attempt a second rotation.

    Args:
        video_path: Path to the source video.
        seek_seconds: Absolute seek position in the video (fallback if frame_index is missing).
        out_path: Output JPEG path.
        frame_index: Exact frame index (0-based) from the raw ML CSV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Could not open video with cv2: {video_path}")
        return False

    rotation = _read_video_rotation(cap)
    if rotation:
        logging.debug(f"Video rotation metadata: {rotation}° — will apply to frames.")

    if frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000.0)

    ret, frame = cap.read()
    if ret:
        if rotation:
            frame = cv2.rotate(frame, _ROTATION_MAP[rotation])
        cv2.imwrite(str(out_path), frame)
        # Embed EXIF Orientation = 1: rotation is already baked into the pixels,
        # so no viewer should attempt to rotate again.
        try:
            exif_bytes = piexif.dump({"0th": {piexif.ImageIFD.Orientation: 1}})
            piexif.insert(exif_bytes, str(out_path))
        except Exception as e:
            logging.debug(f"Could not embed EXIF orientation for {out_path}: {e}")
    else:
        logging.error(
            f"cv2 failed to read frame at {'index ' + str(frame_index) if frame_index is not None else str(seek_seconds) + 's'}"
        )

    cap.release()
    return ret


# ── YOLO → COCO conversion ───────────────────────────────────────────────────


def yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """
    Convert YOLO center-format pixel bbox [cx, cy, w, h] to COCO [x, y, width, height] (pixels).
    """
    return [round(cx - w / 2, 2), round(cy - h / 2, 2), round(w, 2), round(h, 2)]


def build_coco_from_raw_csv(
    raw_csv_path: str,
    frame_records: list[
        dict
    ],  # [{image_id, file_name, time_of_max, img_w, img_h}, ...]
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
    raw_df = (
        pd.read_csv(raw_csv_path) if os.path.exists(raw_csv_path) else pd.DataFrame()
    )

    categories: dict[str, int] = {}
    images = []
    annotations = []
    ann_id = 0

    for rec in frame_records:
        img_id = rec["image_id"]
        time_of_max = rec["time_of_max"]  # seconds relative to sampling_start

        images.append(
            {
                "id": img_id,
                "file_name": rec["file_name"],
                "width": rec.get("img_w", img_w),
                "height": rec.get("img_h", img_h),
                "time_of_max": time_of_max,
                "drop_id": rec.get("drop_id", ""),
                "selection_reason": rec.get("selection_reason", ""),
            }
        )

        if raw_df.empty:
            continue

        # Find the raw CSV frame closest to this time_of_max
        nearest_frame_rows = raw_df.iloc[
            (raw_df["time_seconds"] - time_of_max).abs().argsort()[:1]
        ]
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
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": categories[cls_name],
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                    "score": round(float(row.get("confidence", 0.0)), 4),
                    # TODO: verify Biigle accepts the non-standard `score` field in COCO JSON
                    # (COCO spec doesn't require it, but Biigle may use it for display)
                }
            )

    return {
        "info": {"description": "Spyfish Aotearoa — ML MaxN peaks", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cat_id, "name": name}
            for name, cat_id in sorted(categories.items(), key=lambda x: x[1])
        ],
    }


# ── main function ─────────────────────────────────────────────────────────────


def extract_frames_from_selections(
    selections_csv_path: str,
    video_path: str,
    raw_csv_path: str,
) -> pd.DataFrame:
    """
    Extract one clean JPEG per row in the selections CSV at the exact MaxN peak frame,
    and produce a COCO JSON with the corresponding YOLO bounding boxes.

    The frame is grabbed at the absolute video timestamp in csv_clip_max_time_column (TimeOfMaxnSeconds).
    This is the exact frame that was the deciding factor in the MaxN calculation.

    Unlike draw_frames.py (which draws boxes ON the frame using cv2 for QA),
    this extracts a clean frame with annotations stored separately as COCO JSON —
    suitable for upload.

    Frames are written to the canonical frames/ directory for the drop and shared
    across Zooniverse and Biigle upload steps. Already-extracted frames are skipped.

    Args:
        selections_csv_path: CSV from select_clips.select_zooniverse_clips().
        video_path: Full path to the source video file.
        raw_csv_path: Raw YOLO CSV ({drop_id}_{model}_raw.csv), for COCO annotations.

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

    drop_id = df[config.drop_id_column].iloc[0]

    out_dir = config.get_frames_dir(drop_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = (
        pd.read_csv(raw_csv_path) if os.path.exists(raw_csv_path) else pd.DataFrame()
    )

    # Read video dimensions and rotation once from metadata.
    # extract_frame() applies rotation to pixel data, so swap w/h for 90°/270° videos
    # so that COCO image dimensions match the actual saved frame orientation.
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation = _read_video_rotation(cap)
    cap.release()

    if rotation in (90, 270):
        vid_w, vid_h = vid_h, vid_w

    if vid_w == 0 or vid_h == 0:
        logging.warning(
            f"Could not read video dimensions for {video_path}. COCO image sizes will default to 0."
        )

    frame_records = []
    frame_paths = []

    for img_id, (_, row) in enumerate(df.iterrows(), start=1):
        # TimeOfMaxnMs is an absolute video timestamp — use it directly.
        seek_seconds = float(row[config.csv_clip_max_time_column])
        frame_index = None

        if not raw_df.empty:
            # Find the nearest frame in the raw ML CSV to this peak time
            # Using the exact same matching logic as the COCO builder ensures alignment
            nearest = raw_df.iloc[
                (raw_df["time_seconds"] - seek_seconds).abs().argsort()[:1]
            ]
            if not nearest.empty:
                frame_index = int(nearest["frame"].iloc[0])

        out_filename = generate_frame_filename(drop_id, seek_seconds)
        out_path = out_dir / out_filename

        if out_path.exists():
            logging.debug(
                f"  [{img_id}/{len(df)}] Already extracted, skipping: {out_filename}"
            )
            frame_paths.append(str(out_path))
        else:
            logging.info(
                f"  [{img_id}/{len(df)}] Frame at {seek_seconds:.3f}s (index={frame_index}) → {out_filename}"
            )
            success = extract_frame(
                video_path, seek_seconds, out_path, frame_index=frame_index
            )
            frame_paths.append(str(out_path) if success else None)

        frame_records.append(
            {
                "image_id": img_id,
                "file_name": out_filename,
                "time_of_max": seek_seconds,
                "drop_id": drop_id,
                "selection_reason": row.get("SelectionReason", ""),
                "img_w": vid_w,
                "img_h": vid_h,
            }
        )

    df["FramePath"] = frame_paths

    # Build and save COCO JSON — only include records for frames that were successfully extracted
    successful_records = [
        rec for rec, path in zip(frame_records, frame_paths) if path is not None
    ]
    skipped = len(frame_records) - len(successful_records)
    if skipped:
        logging.warning(
            f"Skipping {skipped} frame(s) from COCO JSON for {drop_id} due to extraction failure"
        )

    # Deduplicate by file_name: multiple selections rows can share the same timestamp
    # (e.g. different species at the same MaxN peak), producing identical frames.
    # Keep the first occurrence so each physical file appears exactly once in the COCO JSON.
    seen: set[str] = set()
    deduped_records = []
    for rec in successful_records:
        if rec["file_name"] not in seen:
            seen.add(rec["file_name"])
            deduped_records.append(rec)
    if len(deduped_records) < len(successful_records):
        logging.debug(
            f"Deduplicated {len(successful_records) - len(deduped_records)} duplicate frame record(s) for {drop_id}"
        )

    coco = build_coco_from_raw_csv(raw_csv_path, deduped_records)

    # Save the COCO annotations to the drop's annotations directory
    annotations_dir = config.get_drop_annotations_dir(drop_id)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    coco_path = annotations_dir / f"{drop_id}_coco_annotations_for_biigle.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    logging.info(
        f"COCO annotations → {coco_path} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)"
    )

    successful = df["FramePath"].notna().sum()
    logging.info(f"Extracted {successful}/{len(df)} frames for {drop_id} → {out_dir}")
    return df
