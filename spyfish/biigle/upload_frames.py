"""
Biigle volume upload for Spyfish Aotearoa.

Two-step workflow:
  1. upload_frames_to_s3()   — push extracted JPEGs to the S3 bucket Biigle has access to
  2. create_biigle_volume()  — create an image volume in Biigle pointing at that S3 folder

Ported and structured from:
  Spyfish-Aotearoa-toolkit_old/notebooks/biigle_uploader.ipynb
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.config import config
from spyfish.database.manager import DatabaseManager


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
                          "biigle_frames/KSF_20240124_BUV_KSF_085_01/"

    Returns:
        List of uploaded filenames (basename only, as used in the Biigle volume file list).
    """
    try:
        from spyfish.storage.s3_handler import S3Handler
    except ImportError:
        raise ImportError("spyfish.storage.s3_handler is required for S3 uploads.")

    s3 = S3Handler()
    uploaded = []

    frame_paths = frames_df["FramePath"].dropna().tolist()
    if not frame_paths:
        logging.warning("No frame paths to upload (all extractions may have failed).")
        return []

    for local_path in frame_paths:
        p = Path(local_path)
        if not p.exists():
            logging.warning(f"Frame not found, skipping: {local_path}")
            continue

        s3_key = s3_frames_prefix.rstrip("/") + "/" + p.name
        s3.upload_file_to_s3(str(p), key=s3_key, content_type="image/jpeg")
        uploaded.append(p.name)
        logging.info(f"  Uploaded {p.name} → s3://{s3_key}")

    logging.info(f"Uploaded {len(uploaded)}/{len(frame_paths)} frames to S3 prefix '{s3_frames_prefix}'")
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
                          "biigle_frames/KSF_20240124_BUV_KSF_085_01/").
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
    logging.info(f"Volume created: id={volume_id}  url=https://biigle.de/volumes/{volume_id}")
    return volume_info


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
        s3_frames_prefix: S3 key prefix for the frame folder, e.g.
                          "biigle_frames/KSF_20240124_BUV_KSF_085_01/"
        project_id: Biigle project ID. Defaults to config.biigle_project_id.

    Returns:
        Biigle volume info dict (contains 'id', 'name', etc.)

    TODO — COCO annotation import (not yet implemented):
        After volume creation, Biigle can receive image annotations via:
            POST /api/v1/volumes/{volume_id}/image-annotations/bulk
        This requires:
          1. Fetching the label tree to resolve class name → Biigle label_id
             GET /api/v1/label-trees/{tree_id}/labels
          2. Fetching the new volume's image list to resolve filename → biigle_image_id
             GET /api/v1/volumes/{volume_id}/images
          3. Converting COCO bbox [x, y, w, h] → Biigle Rectangle points
             [x, y, x+w, y, x+w, y+h, x, y+h]  (8-value flat array)
          4. Posting each annotation with shape_id=5 (Rectangle)
        Reference: https://biigle.de/doc/api/index.html#api-ImageAnnotations-StoreVolumeImageAnnotations
        See: {drop_id}_coco_annotations.json (output of extract_frames_from_selections)
    """
    logging.info(f"Starting Biigle upload for drop {drop_id}")

    # Build S3 prefix: {base_prefix}/{survey_id}/{drop_id}/
    # survey_id = first 16 chars of drop_id, e.g. "KSF_20240124_BUV" from "KSF_20240124_BUV_KSF_085_01"
    survey_id = drop_id[:16]
    s3_prefix = f"{config.biigle_s3_images_prefix}/{survey_id}/{drop_id}/"

    # Step 1: Upload frames to S3
    file_names = upload_frames_to_s3(frames_df, s3_prefix)

    if not file_names:
        raise RuntimeError(f"No frames uploaded to S3 for {drop_id} — aborting volume creation.")

    # Step 2: Create Biigle volume
    volume_info = create_biigle_volume(drop_id, s3_prefix, file_names, project_id)
    volume_id = volume_info.get("id")

    # Step 3: Save volume_id to database
    if volume_id:
        db = DatabaseManager()
        db.update_biigle_volume_id(drop_id, volume_id)
        logging.info(f"Saved Biigle volume_id {volume_id} to database for {drop_id}")

    return volume_info
