"""
Biigle volume upload for Spyfish Aotearoa.

Two-step workflow:
  1. upload_frames_to_s3()   — push extracted JPEGs to the S3 bucket Biigle has access to
  2. create_biigle_volume()  — create an image volume in Biigle pointing at that S3 folder

Ported and structured from:
  Spyfish-Aotearoa-toolkit_old/notebooks/biigle_uploader.ipynb
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler

# ── Step 1: S3 upload ────────────────────────────────────────────────────────


def upload_frames_to_s3(
    frames_df: pd.DataFrame,
    s3_frames_prefix: str,
) -> list[str]:
    """
    Upload extracted JPEG frames to S3 so Biigle can access them via the disk mount.

    Args:
        frames_df: DataFrame with a 'FramePath' column (output of extract_frames_from_selections).
                   Rows with None FramePath (extraction failures) are skipped.
        s3_frames_prefix: S3 key prefix for the upload destination, e.g.
                          "process_files/deployment_data/KSF_20240124/KSF_20240124_BUV_KSF_085_01/frames/"

    Returns:
        List of uploaded filenames (basename only, as used in the Biigle volume file list).
    """
    s3 = S3Handler()
    uploaded = []

    frame_paths = list(dict.fromkeys(frames_df["FramePath"].dropna().tolist()))
    if not frame_paths:
        logging.warning("No frame paths to upload (all extractions may have failed).")
        return []

    s3_prefix = s3_frames_prefix.rstrip("/") + "/"
    existing_keys = s3.get_file_paths_set_from_s3(prefix=s3_prefix)

    skipped = 0
    for local_path in frame_paths:
        p = Path(local_path)
        if not p.exists():
            logging.warning(f"Frame not found, skipping: {local_path}")
            continue

        s3_key = s3_prefix + p.name
        if s3_key in existing_keys:
            logging.debug(f"  Already on S3, skipping: {p.name}")
            uploaded.append(p.name)
            skipped += 1
            continue
        s3.upload_file_to_s3(str(p), key=s3_key, content_type="image/jpeg")
        uploaded.append(p.name)
        logging.info(f"  Uploaded {p.name} → s3://{s3_key}")

    newly_uploaded = len(uploaded) - skipped
    logging.info(
        f"Uploaded {newly_uploaded} new frame(s), {skipped} already on S3 "
        f"(total {len(uploaded)}/{len(frame_paths)}) → '{s3_prefix}'"
    )
    return uploaded


# ── Step 2: Biigle volume creation ───────────────────────────────────────────


def create_biigle_volume(
    drop_id: str,
    s3_frames_prefix: str,
    file_names: list[str],
    project_id: Optional[int] = None,
) -> dict:
    """
    Create a Biigle image volume pointing at the S3 folder containing extracted frames.

    The S3 folder must be accessible via the Biigle disk configuration
    (disk-{biigle_disk_id}://{s3_frames_prefix}).

    Args:
        drop_id: Deployment identifier, used as the volume name.
        s3_frames_prefix: S3 key prefix matching what was uploaded (e.g.
                          "process_files/deployment_data/KSF_20240124/KSF_20240124_BUV_KSF_085_01/frames/").
        file_names: List of JPEG filenames within the S3 prefix (basenames only).
        project_id: Biigle project ID. Defaults to config.biigle_project_id.

    Returns:
        Volume info dict from the Biigle API.
    """
    if not file_names:
        raise ValueError("No files to create a Biigle volume with.")

    handler = BiigleHandler()
    s3_url = handler.build_s3_url(s3_frames_prefix)
    volume_name = f"{drop_id} — ML frames"

    logging.info(
        f"Creating Biigle image volume '{volume_name}' "
        f"with {len(file_names)} frames at {s3_url}"
    )
    volume_info = handler.create_volume_from_s3_files(
        volume_name=volume_name,
        s3_url=s3_url,
        files=file_names,
        project_id=project_id,
        media_type="image",
    )
    volume_id = volume_info.get("id", "?")
    logging.info(
        f"Volume created: id={volume_id}  url=https://biigle.de/volumes/{volume_id}"
    )
    return volume_info


# ── Step 3: Annotation Upload ────────────────────────────────────────────────


def upload_coco_annotations_to_biigle(
    volume_id: int,
    coco_data: dict,
    label_id: int = config.default_fish_label_id,
) -> dict:
    """
    Upload COCO annotations to a Biigle volume.
    All bboxes use `label_id` (defaults to `config.default_fish_label_id`).
    TODO: multi-species — map COCO category names to Biigle label IDs via video_labels.csv.

    Args:
        volume_id: Biigle volume ID.
        coco_data: COCO annotations dict (from extract_frames_from_selections).
        label_id: The Biigle label ID to apply to all annotations. Defaults to config value.

    Returns:
        The API response dict from the bulk upload endpoint.
    """
    if not coco_data.get("annotations"):
        logging.info("No annotations found in COCO JSON.")
        return {}

    handler = BiigleHandler()

    # Fetch images from the new volume to map filenames to Biigle image IDs
    volume_images = handler.get_volume_images(volume_id)
    filename_to_biigle_id = {img["filename"]: img["id"] for img in volume_images}

    # Map COCO image IDs to actual filenames
    coco_img_id_to_filename = {
        img["id"]: img["file_name"] for img in coco_data.get("images", [])
    }

    # Map COCO category IDs to species names
    coco_cat_id_to_name = {
        cat["id"]: cat["name"] for cat in coco_data.get("categories", [])
    }

    # Load species → Biigle label ID mapping from config.yaml
    label_mapping = config.label_mapping or {}

    # Build the Biigle bulk payload
    biigle_annotations = []

    for ann in coco_data["annotations"]:
        coco_img_id = ann["image_id"]
        filename = coco_img_id_to_filename.get(coco_img_id)
        if not filename:
            continue

        biigle_img_id = filename_to_biigle_id.get(filename)
        if not biigle_img_id:
            logging.warning(f"Could not find Biigle image ID for file: {filename}")
            continue

        # COCO bbox format: [x, y, w, h] -> Biigle Rectangle format: [x1, y1, x2, y1, x2, y2, x1, y2]
        x, y, w, h = ann["bbox"]

        # Coordinates are verified to be full original frame resolution (e.g. 1920x1080), no scaling needed.
        x1, y1 = float(x), float(y)
        x2, y2 = float(x + w), float(y + h)
        points = [x1, y1, x2, y1, x2, y2, x1, y2]

        # Look up species name and specific Biigle label ID
        cat_id = ann.get("category_id")
        species_name = coco_cat_id_to_name.get(cat_id, "unknown")

        # Routing precedence:
        #   1. Explicit per-species mapping in label_mapping (config.yaml)
        #   2. 'bait' class → default_bait_label_id (tree 3375 → "Bait")
        #   3. Anything else → label_id (fallback, default = default_fish_label_id)
        if species_name in label_mapping:
            assigned_label_id = label_mapping[species_name]
        elif species_name == "bait":
            assigned_label_id = config.default_bait_label_id
        else:
            assigned_label_id = label_id

        biigle_annotations.append(
            {
                "image_id": biigle_img_id,
                "shape_id": 5,  # 5 = Rectangle
                "points": points,
                "label_id": assigned_label_id,
                "confidence": float(ann.get("score", 1.0)),
            }
        )

    if not biigle_annotations:
        logging.warning("No annotations resolved to valid Biigle images.")
        return {}

    result = handler.upload_image_annotations(volume_id, biigle_annotations)
    return result


# ── Combined convenience function ─────────────────────────────────────────────


def upload_frames_to_biigle(
    drop_id: str,
    frames_df: pd.DataFrame,
    project_id: Optional[int] = None,
) -> dict:
    """
    Upload extracted frames to S3 then create a Biigle image volume — full workflow.

    Args:
        drop_id: Deployment identifier.
        frames_df: DataFrame with 'FramePath' column (from extract_frames_from_selections).
        project_id: Biigle project ID. Defaults to config.biigle_project_id.

    Returns:
        Biigle volume info dict (contains 'id', 'name', etc.)
    """
    logging.info(f"Starting Biigle upload for drop {drop_id}")

    s3_prefix = config.get_frames_s3_prefix(drop_id)

    # Verify COCO JSON exists before committing any uploads — a missing file means
    # frame extraction failed and the upload should not proceed at all.
    annotations_dir = config.get_drop_annotations_dir(drop_id)
    coco_json_path = annotations_dir / f"{drop_id}_coco_annotations_for_biigle.json"
    if not coco_json_path.exists():
        raise FileNotFoundError(
            f"COCO annotations JSON not found: {coco_json_path}. "
            "Run frame extraction before Biigle upload."
        )

    with open(coco_json_path) as f:
        coco = json.load(f)
    if not coco.get("annotations"):
        logging.error(
            f"COCO annotations for {drop_id} have 0 annotations — "
            "no ML detections to review. Skipping Biigle upload."
        )
        return None

    # Step 1: Upload frames to S3
    file_names = upload_frames_to_s3(frames_df, s3_prefix)

    if not file_names:
        raise RuntimeError(
            f"No frames uploaded to S3 for {drop_id} — aborting volume creation."
        )

    # Step 2: Create Biigle volume
    volume_info = create_biigle_volume(drop_id, s3_prefix, file_names, project_id)
    volume_id = volume_info.get("id")

    if volume_id:
        logging.info(
            f"✅ Biigle Volume Created! Link: https://biigle.de/projects/{project_id or config.biigle_project_id}/volumes/{volume_id}"
        )

    # Step 3: Save volume_id to database
    if volume_id:
        db = DatabaseManager()
        db.update_biigle_volume_id(drop_id, volume_id)
        logging.info(f"Saved Biigle volume_id {volume_id} to database for {drop_id}")

    # Step 4: Upload COCO bounding box annotations to the new volume
    if volume_id:
        upload_coco_annotations_to_biigle(volume_id, coco)

    return volume_info
