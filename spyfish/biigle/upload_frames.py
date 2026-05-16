"""
Biigle volume upload for Spyfish Aotearoa.

Two-step workflow:
  1. upload_frames_to_s3()   — push extracted JPEGs to the S3 bucket Biigle has access to
  2. create_biigle_volume()  — create an image volume in Biigle pointing at that S3 folder
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler
from spyfish.utils import load_species_labels

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
                          "process_files/deployment_data/KSF_20240124/KSF_20240124_BUV_KSF_085_01/frames/" # pragma: allowlist secret

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


# ── Find-or-create-and-append: idempotent survey-level volume management ─────


def find_or_create_volume_and_add_frames(
    volume_name: str,
    s3_frames_prefix: str,
    file_names: list[str],
    project_id: Optional[int] = None,
    media_type: str = "image",
) -> Tuple[int, Dict[str, int]]:
    """Land `file_names` into the Biigle volume named `volume_name`, creating
    the volume if it doesn't exist.

    Idempotency lives at two layers we don't manage here:
      - S3: `upload_frames_to_s3` skips objects already in the prefix.
      - Biigle: per-volume filename uniqueness. POSTing existing names is
        expected to be a no-op for those names; we don't pre-diff.

    The expected re-run pattern is governed upstream by
    `db.get_training_biigle_volume_id()` — that already prevents re-uploads
    in the common case, leaving only `--force` re-runs to land here.

    Args:
        volume_name: Exact volume name (used as a lookup key — keep stable
            across runs).
        s3_frames_prefix: S3 prefix the frames live under, used only when
            creating a new volume (Biigle stores it as the volume's `url`).
        file_names: Basenames to ensure are in the volume.
        project_id: Defaults to `config.biigle_project_id`.
        media_type: "image" or "video". Default "image".

    Returns:
        ``(volume_id, filename_to_biigle_id)``. The map covers the files we
        just landed (returned directly by ``add_files_to_volume`` on an
        existing volume; empty on a freshly created one — the caller's
        annotation step handles the create-case fallback). Pass it to
        ``upload_coco_annotations_to_biigle`` to skip a per-drop full
        ``get_volume_images`` sweep, which is the dominant Biigle call
        cost in survey-level batch runs (the volume's image count grows
        per drop and the helper does ~N concurrent GETs).
    """
    if not file_names:
        raise ValueError(
            f"find_or_create_volume_and_add_frames({volume_name!r}): empty file list"
        )

    handler = BiigleHandler()
    project_id = project_id or config.biigle_project_id

    matches = [
        v for v in handler.get_volumes(project_id) if v.get("name") == volume_name
    ]

    if not matches:
        logging.info(
            f"Creating new volume {volume_name!r} with {len(file_names)} file(s)."
        )
        info = handler.create_volume_from_s3_files(
            volume_name=volume_name,
            s3_url=handler.build_s3_url(s3_frames_prefix),
            files=file_names,
            project_id=project_id,
            media_type=media_type,
        )
        # On creation we don't get image IDs back — the caller's annotation
        # step will fall back to one get_volume_images call. That fetch is
        # bounded (just-this-drop's files) since the volume is brand-new,
        # which is the only situation where a fetch is now necessary.
        return int(info["id"]), {}

    # Biigle does not enforce volume-name uniqueness — pick most recent.
    matches.sort(key=lambda v: v.get("created_at", ""), reverse=True)
    volume_id = int(matches[0]["id"])
    if len(matches) > 1:
        logging.warning(
            f"{len(matches)} volumes named {volume_name!r}; using most recent "
            f"(id={volume_id})."
        )
    logging.info(
        f"Adding {len(file_names)} file(s) to existing volume {volume_name!r} "
        f"(id={volume_id})."
    )
    added = handler.add_files_to_volume(volume_id, file_names)
    filename_to_id = {
        item["filename"]: int(item["id"])
        for item in added
        if isinstance(item, dict) and "filename" in item and "id" in item
    }

    # Defensive fallback: if the response didn't include entries for some
    # of our filenames (e.g. they were already in the volume from a prior
    # partial run that didn't reach the annotation step), do a single full
    # image fetch to fill the gaps. This is the only path that triggers
    # the heavyweight call — the rate-limit-clean case avoids it.
    missing = [n for n in file_names if n not in filename_to_id]
    if missing:
        logging.info(
            f"add_files_to_volume returned IDs for {len(filename_to_id)}/"
            f"{len(file_names)} filenames; fetching full image list to "
            f"resolve {len(missing)} missing entr(y/ies) — likely already "
            "in the volume from a prior run."
        )
        all_imgs = handler.get_volume_images(volume_id)
        filename_to_id = {img["filename"]: int(img["id"]) for img in all_imgs}

    return volume_id, filename_to_id


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
                          "process_files/deployment_data/KSF_20240124/KSF_20240124_BUV_KSF_085_01/frames/"). # pragma: allowlist secret
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
    filename_to_biigle_id: Optional[Dict[str, int]] = None,
) -> dict:
    """
    Upload COCO annotations to a Biigle volume.
    All bboxes use `label_id` (defaults to `config.default_fish_label_id`).
    TODO: multi-species — map COCO category names to Biigle label IDs via video_labels.csv.

    Args:
        volume_id: Biigle volume ID.
        coco_data: COCO annotations dict (from extract_frames_from_selections).
        label_id: The Biigle label ID to apply to all annotations. Defaults to config value.
        filename_to_biigle_id: Optional pre-supplied ``{filename: biigle_image_id}``
            map. When supplied AND it covers every filename referenced in
            ``coco_data["images"]``, the per-volume image fetch is skipped —
            saving a 20-thread parallel GET burst that scales with the
            volume's image count and dominates rate-limit consumption in
            survey-level batch runs. Falls back to the full fetch when
            the map is absent or incomplete.

    Returns:
        The API response dict from the bulk upload endpoint.
    """
    if not coco_data.get("annotations"):
        logging.info("No annotations found in COCO JSON.")
        return {}

    handler = BiigleHandler()

    # Map COCO image IDs to actual filenames
    coco_img_id_to_filename = {
        img["id"]: img["file_name"] for img in coco_data.get("images", [])
    }

    # Resolve filename → Biigle image_id. The caller (typically
    # find_or_create_volume_and_add_frames) can pre-supply this map from
    # add_files_to_volume's response, which costs zero extra Biigle calls.
    # Fall back to a full image-list fetch only when the map is missing or
    # doesn't cover every filename we need to annotate.
    needed_names = set(coco_img_id_to_filename.values())
    if filename_to_biigle_id is not None and needed_names.issubset(
        filename_to_biigle_id
    ):
        logging.info(
            f"Volume {volume_id}: using pre-supplied filename→ID map "
            f"({len(filename_to_biigle_id)} entries); skipping get_volume_images."
        )
    else:
        if filename_to_biigle_id is not None:
            missing = needed_names - set(filename_to_biigle_id)
            logging.info(
                f"Volume {volume_id}: pre-supplied map missing "
                f"{len(missing)} filename(s); falling back to full "
                "get_volume_images."
            )
        volume_images = handler.get_volume_images(volume_id)
        filename_to_biigle_id = {
            img["filename"]: int(img["id"]) for img in volume_images
        }

    # Map COCO category IDs to species names
    coco_cat_id_to_name = {
        cat["id"]: cat["name"] for cat in coco_data.get("categories", [])
    }

    # Routing sources, in precedence order:
    #   1. config.label_mapping — explicit per-species override
    #   2. species_labels.csv — full BIIGLE label tree (~175 species)
    #   3. 'bait' class → default_bait_label_id
    #   4. Fallback → label_id (default_fish_label_id), with a warning
    label_mapping = config.label_mapping or {}
    species_tree = load_species_labels().name_to_label_id
    unmatched: Counter = Counter()

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

        cat_id = ann.get("category_id")
        species_name = coco_cat_id_to_name.get(cat_id, "unknown")

        if species_name in label_mapping:
            assigned_label_id = label_mapping[species_name]
        elif species_name in species_tree:
            assigned_label_id = species_tree[species_name]
        elif species_name == "bait":
            assigned_label_id = config.default_bait_label_id
        elif species_name == "fish":
            # `fish` is the model's legitimate "fish present, species unknown"
            # class — Fish: review required is its by-design destination.
            assigned_label_id = label_id
        else:
            unmatched[species_name] += 1
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

    if unmatched:
        details = ", ".join(f"{sp}×{n}" for sp, n in unmatched.most_common())
        logging.warning(
            f"Biigle label routing: {sum(unmatched.values())} annotation(s) across "
            f"{len(unmatched)} species fell back to default_fish_label_id "
            f"({config.default_fish_label_id}). Unmatched: {details}. "
            "Add to config.yaml `biigle.label_mapping` or refresh "
            "process_files/biigle/labels/species_labels.csv from the BIIGLE label tree."
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
    coco_json_path = config.get_coco_annotations_path(drop_id)
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
