"""
biigle_to_yolo.py — Convert local Biigle expert CSV exports → YOLO label .txt files.

This tool is strictly offline; it consumes CSVs previously exported by sync_biigle_annotations
into process_files/deployment_data/{drop_id}/annotations/.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from spyfish.biigle.class_map import load_class_map
from spyfish.config.wrapper import config

# ---------------------------------------------------------------------------
# Core conversion helpers
# ---------------------------------------------------------------------------

# Per-axis AABB shrink fraction. For a rotated rectangle, the closest non-
# corner feature to each AABB edge is one of the 4 edge midpoints (head,
# tail, back, belly of the fish silhouette). Shrinking each AABB axis by
# this fraction of that midpoint's margin recovers background pixels without
# clipping anatomy. 0 disables; 0.5 (half the geometric safety margin)
# leaves the rest as buffer against annotator slop and float noise.
SHRINK_SAFETY = 0.5


def biigle_rect_to_yolo(
    points: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert a Biigle Rectangle annotation to normalised YOLO HBB format.

    Biigle's Rectangle is a quadrilateral with 4 corner points stored as 8
    flat floats. The drawing tool allows rotation, so corners aren't
    guaranteed to be axis-aligned. We take the AABB envelope (min/max over
    all four corners), clamp it to image bounds, then apply a per-axis
    shrink toward the AABB centre based on the closest edge midpoint to
    each AABB edge. Those midpoints are the visible fish features
    (head/tail tips and back/belly midpoints), so shrinking by half their
    margin recovers background pixels without clipping anatomy. For
    axis-aligned rectangles the shrink is zero — midpoints sit on the AABB
    edges and the formula self-disables.

    Returns:
        (cx, cy, w, h) each normalised to [0, 1] by image dimensions.
    """
    if len(points) != 8:
        raise ValueError(
            f"Expected 8 floats (4 corners) for a Biigle Rectangle, got {len(points)}"
        )
    xs = points[0::2]
    ys = points[1::2]
    # Clamp raw corners to image bounds — rotated rectangles can have
    # corners outside the frame, and YOLO rejects labels whose edges fall
    # outside [0, 1].
    x_min = max(0.0, min(float(img_w), min(xs)))
    x_max = max(0.0, min(float(img_w), max(xs)))
    y_min = max(0.0, min(float(img_h), min(ys)))
    y_max = max(0.0, min(float(img_h), max(ys)))

    half_w = (x_max - x_min) / 2.0
    half_h = (y_max - y_min) / 2.0
    if half_w > 0 and half_h > 0:
        midpoints = [
            (
                (points[2 * i] + points[2 * ((i + 1) % 4)]) / 2.0,
                (points[2 * i + 1] + points[2 * ((i + 1) % 4) + 1]) / 2.0,
            )
            for i in range(4)
        ]
        # max(0, ...) handles midpoints outside the clamped AABB — they
        # contribute zero margin, so shrink on that axis self-disables.
        min_x_margin = min(
            min(max(0.0, mx - x_min), max(0.0, x_max - mx)) for mx, _ in midpoints
        )
        min_y_margin = min(
            min(max(0.0, my - y_min), max(0.0, y_max - my)) for _, my in midpoints
        )
        cx_px = (x_min + x_max) / 2.0
        cy_px = (y_min + y_max) / 2.0
        half_w *= 1 - SHRINK_SAFETY * min_x_margin / half_w
        half_h *= 1 - SHRINK_SAFETY * min_y_margin / half_h
        x_min, x_max = cx_px - half_w, cx_px + half_w
        y_min, y_max = cy_px - half_h, cy_px + half_h

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def _read_image_dimensions(img_path: Path) -> Optional[Tuple[int, int]]:
    """Return (width, height) from a JPEG/PNG header, or None if unreadable.

    PIL's `Image.open` only parses the header for these formats — no full decode.
    """
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception as exc:
        logging.debug(f"Could not read dimensions of {img_path}: {exc}")
        return None


def _canvas_from_attributes(group: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """Return the (width, height) the annotations were *drawn on*, from Biigle's
    ``attributes`` JSON column — or None if absent/unparseable.

    Biigle image-annotation reports store per-image metadata as a JSON string,
    e.g. ``{"size":..., "mimetype":"image/jpeg", "width":1920, "height":1080}``.
    This is the **authoritative** normalisation canvas: the resolution the
    expert saw when drawing the box, which is the only correct YOLO denominator.
    It is NOT necessarily the resolution of whatever frame sits on disk — those
    can diverge (e.g. a 1920-wide annotation against a 1440-wide re-extracted
    frame), and trusting the on-disk file then mis-normalises every box.

    Returns the first valid (width, height) found in the group. Video-annotation
    reports omit this column entirely (points live in the video's resolution),
    so callers must fall back when None.
    """
    if "attributes" not in group.columns:
        return None
    for raw in group["attributes"].dropna():
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
            w, h = meta.get("width"), meta.get("height")
            if w and h:
                return int(w), int(h)
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return None


def _rectangle_corners(points_raw) -> Optional[List[float]]:
    """Return a Biigle Rectangle as 8 flat floats, or None if it isn't a single
    rectangle this converter can place.

    The ``points`` field takes two shapes:
      - image annotations: a flat list ``[x1,y1,x2,y2,x3,y3,x4,y4]``.
      - video annotations: a list of per-keyframe coordinate lists
        ``[[x1,y1,...,x4,y4], ...]``. A single keyframe is unwrapped; multiple
        keyframes mean the box moves over time and must be projected to a frame
        upstream (``process_video_annotations``) — we can't place it here.

    Without the unwrap, a nested ``[[...]]`` fails the old ``len == 8`` check, so
    every video box (the bulk of the oriented ones) is silently dropped.
    Rotation itself is fine — the caller takes the AABB envelope, which collapses
    any rotated rectangle to a correct horizontal box, invariant under resize.
    """
    if isinstance(points_raw, str):
        try:
            points_raw = json.loads(points_raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(points_raw, (list, tuple)) or not points_raw:
        return None
    # Unwrap video keyframe nesting: [[...]] -> [...]; skip multi-keyframe boxes.
    if isinstance(points_raw[0], (list, tuple)):
        if len(points_raw) != 1:
            return None
        points_raw = points_raw[0]
    pts = list(points_raw)
    return pts if len(pts) == 8 else None


def _rectangle_tilt_deg(pts: List[float]) -> float:
    """Smallest angle (deg) of the rectangle's first edge off an image axis.
    ~0 == axis-aligned; larger == an oriented (rotated) box, usually an
    annotator slip. Used only for visibility counting, not for the geometry."""
    import math

    a = abs(math.degrees(math.atan2(pts[3] - pts[1], pts[2] - pts[0]))) % 90
    return min(a, 90 - a)


# Workflow / admin labels (Biigle tree 3375) that should NEVER train as fish.
# These are state markers, not detections — annotators apply them to mark
# images as "in progress", "scale bar visible", "couldn't annotate", etc.
# Rows with these labels are dropped from the YOLO export entirely.
_WORKFLOW_LABEL_SKIP = {
    "In progress",
    "Nothing here",
    "Scale bar",
    "Can't annotate (e.g. bad visibility)",
}


def _is_workflow_label(name: str) -> bool:
    """True if a label is an admin/workflow marker that should be skipped from training."""
    if name in _WORKFLOW_LABEL_SKIP:
        return True
    if name.startswith(
        "Done "
    ):  # Done Video, Done Sizes, Done Volume, Done QA Review, ...
        return True
    return False


def convert_annotations_to_yolo(
    df: pd.DataFrame,
    class_map: Dict[str, int],
    labels_dir: Path,
    images_dir: Path,
    default_img_size: Tuple[int, int] = (1920, 1080),
) -> Dict[str, int]:
    """
    Write one YOLO .txt label file per image.

    Args:
        df: Annotation DataFrame with columns: filename, label_name, points (JSON list).
        class_map: label_name → YOLO class_id.
        labels_dir: Output directory for .txt files.
        images_dir: Source directory of the actual frame images. Only consulted
            as a *fallback* denominator — see below.
        default_img_size: Last-resort (width, height) when neither the Biigle
            `attributes` canvas nor an on-disk frame is available.

    Normalisation canvas (the denominator boxes are divided by), in priority:
        1. Biigle `attributes` width/height for the image (authoritative — the
           resolution the expert drew the box on).
        2. The on-disk frame's pixel dimensions (correct only if the frame was
           extracted at the same resolution it was annotated at).
        3. `default_img_size`.
    When (1) and (2) are both known and disagree, the box is normalised to the
    canvas (1) — the on-disk frame is a different resolution than what was
    annotated — and a warning is logged so the divergence is visible at
    conversion time rather than surfacing later as misaligned spot-checks.

    Returns:
        {filename: annotation_count} summary.
    """
    # Drop workflow/admin annotations entirely (don't even route to fish bucket).
    n_pre = len(df)
    skip_mask = df["label_name"].astype(str).map(_is_workflow_label)
    if skip_mask.any():
        skipped_labels = sorted(df.loc[skip_mask, "label_name"].astype(str).unique())
        logging.info(
            f"Skipping {int(skip_mask.sum())} workflow/admin annotation(s) — "
            f"these are status markers, not detections. Labels: {skipped_labels}"
        )
        df = df.loc[~skip_mask].copy()
    if df.empty:
        logging.warning(
            f"All {n_pre} annotations were workflow/admin labels — nothing to write."
        )
        labels_dir.mkdir(parents=True, exist_ok=True)
        return {}

    incoming_labels = set(df["label_name"].dropna().astype(str).unique())
    unknown = sorted(incoming_labels - set(class_map.keys()))
    fish_class_id = class_map.get("fish")  # generic-fish fallback bucket
    if unknown:
        affected = int(df["label_name"].isin(unknown).sum())
        if fish_class_id is None:
            raise ValueError(
                f"class_map is missing {len(unknown)} label(s) and has no "
                f"'fish' fallback bucket — {affected} of {len(df)} rows "
                f"({affected / len(df):.1%}) would be dropped silently.\n"
                f"  Missing labels: {unknown}\n"
                f"  Fix: reseed class_map.json with "
                f"`python -m spyfish.biigle.class_map`, or add a fish bucket."
            )
        logging.warning(
            f"{len(unknown)} label(s) not in class_map — routing {affected} of "
            f"{len(df)} rows ({affected / len(df):.1%}) to the 'fish' bucket "
            f"(class_id {fish_class_id}). Unknown labels: {unknown}"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}
    n_default_fallback = 0
    n_disk_fallback = 0
    n_canvas_mismatch = 0
    n_non_rect = 0
    n_rotated = 0
    mismatch_examples: List[str] = []

    for filename, group in df.groupby("filename"):
        img_path = images_dir / str(filename)
        # The correct YOLO denominator is the canvas the box was DRAWN on, not
        # whatever frame happens to be on disk. Trust Biigle's `attributes`
        # first; fall back to the on-disk frame, then to the default.
        canvas = _canvas_from_attributes(group)
        disk_dims = _read_image_dimensions(img_path) if img_path.exists() else None
        if canvas is not None:
            img_w, img_h = canvas
            if disk_dims is not None and disk_dims != canvas:
                n_canvas_mismatch += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        f"{filename}: canvas {canvas[0]}x{canvas[1]} "
                        f"!= frame {disk_dims[0]}x{disk_dims[1]}"
                    )
        elif disk_dims is not None:
            n_disk_fallback += 1
            img_w, img_h = disk_dims
        else:
            n_default_fallback += 1
            img_w, img_h = default_img_size
        lines = []

        for _, row in group.iterrows():
            class_id = class_map.get(row["label_name"], fish_class_id)

            # Accept flat (image) and nested (video keyframe) rectangle points.
            # Non-rectangles (Point/Circle/LineString/multi-keyframe) -> None.
            points = _rectangle_corners(row.get("points", row.get("shape_points")))
            if points is None:
                n_non_rect += 1
                continue

            # Oriented (rotated) box -- usually an annotator slip. Counted for
            # visibility; the AABB envelope below collapses it to a correct
            # horizontal box regardless (and that result is resize-invariant).
            if _rectangle_tilt_deg(points) > 3:
                n_rotated += 1

            cx, cy, w, h = biigle_rect_to_yolo(points, img_w, img_h)
            if w <= 0 or h <= 0:
                logging.warning(f"Zero-size bbox for {filename}, skipping")
                continue

            lines.append(f"{class_id} {cx} {cy} {w} {h}")

        stem = Path(str(filename)).stem
        txt_path = labels_dir / f"{stem}.txt"
        txt_path.write_text("\n".join(lines))
        summary[str(filename)] = len(lines)

    logging.debug(f"Wrote {len(summary)} label files to {labels_dir}")
    if n_canvas_mismatch:
        logging.warning(
            f"  {n_canvas_mismatch}/{len(summary)} image(s) have a Biigle "
            f"`attributes` canvas that differs from the on-disk frame. Boxes "
            f"were normalised to the CANVAS (the resolution the expert drew on), "
            f"which is correct — but the on-disk frame is a different resolution, "
            f"so confirm it's the same scene (anamorphic / re-extracted at a "
            f"different size). Examples: {mismatch_examples}"
        )
    if n_disk_fallback:
        logging.info(
            f"  {n_disk_fallback}/{len(summary)} image(s) had NO `attributes` "
            f"canvas in the report — normalised by the on-disk frame size. "
            f"Correct only if the frame was extracted at the resolution it was "
            f"annotated at (e.g. TON video-era frames are 1440-wide on disk but "
            f"were annotated on a 1920-wide canvas — re-download with `attributes`)."
        )
    if n_default_fallback:
        logging.warning(
            f"  {n_default_fallback}/{len(summary)} image(s) had NO `attributes` "
            f"canvas AND were not on disk — used default {default_img_size}. "
            f"Boxes may be misaligned if the true resolution differs; re-download "
            f"the report with `attributes`, or place the frames on disk, to fix."
        )
    if n_non_rect:
        logging.debug(
            f"  Skipped {n_non_rect} non-rectangle/unsupported shape(s) "
            f"(Point/Circle/LineString, or multi-keyframe video boxes)."
        )
    if n_rotated:
        logging.info(
            f"  {n_rotated} oriented (rotated >3deg) box(es) collapsed to "
            f"axis-aligned HBB — usually annotator slips. Boxes are correct; "
            f"flag if a volume has many (size/OBB work needs the source frame)."
        )
    return summary


# ---------------------------------------------------------------------------
# Visual spot-check
# ---------------------------------------------------------------------------


def draw_frames_on_images(
    images_dir: Path,
    labels_dir: Path,
    class_map: Dict[str, int],
    output_dir: Path,
    n_samples: int = 5,
) -> None:
    """
    Draw bounding boxes on a random sample of static JPEG images for visual review.
    """
    try:
        import random

        import cv2
    except ImportError:
        logging.warning("opencv-python not installed — skipping spot-check drawing.")
        return

    id_to_name = {v: k for k, v in class_map.items()}

    # rglob so callers can pass either a flat dir of .txt files or a tree
    # like labels_staged_dir/<drop_id>/*.txt (post-flatten layout).
    label_files = list(labels_dir.rglob("*.txt"))
    if not label_files:
        logging.warning("No label files found for spot-check.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = random.sample(label_files, min(n_samples, len(label_files)))

    for label_path in samples:
        img_path = None
        for ext in config.image_extensions:
            # Search in all subdirectories of images_dir (e.g. deployment_data/survey_id/drop_id/frames/)
            for p in images_dir.rglob(label_path.stem + ext):
                img_path = p
                break
            if img_path:
                break

        if img_path is None:
            logging.debug(f"No image found for {label_path.stem}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    id_to_name.get(cls, str(cls)),
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        logging.info(f"Spot-check image saved: {out_path}")

    logging.info(f"Spot-check complete — {len(samples)} images saved to {output_dir}")


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def biigle_to_yolo(
    deployment_data_dir: Path,
    class_map_path: Path,
) -> Dict[str, int]:
    """
    Finds all expert CSVs in deployment_data and converts them to YOLO .txt files.

    Labels are written into each drop's labels/ folder (sibling of frames/).
    prepare_training_data then walks these to assemble the unified training dataset.
    """
    logging.info(f"Searching for expert CSVs in {deployment_data_dir}...")
    csv_paths = []
    all_dfs = []

    # Strictly use the per-drop expert raw CSVs (skip frozen video-era exports)
    raw_glob = f"**/annotations/*{config.biigle_expert_raw_suffix}"
    for csv_path in sorted(deployment_data_dir.glob(raw_glob)):
        if csv_path.name.startswith("legacy_video_"):
            continue
        logging.debug(f"  Found expert CSV: {csv_path}")
        csv_paths.append(csv_path)
        all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        logging.warning("No expert CSV files found. Retraining cannot proceed.")
        return {}

    class_map = load_class_map(class_map_path)
    n_classes = len(set(class_map.values()))
    logging.info(
        f"Loaded class map with {len(class_map)} label aliases "
        f"({n_classes} classes) from {class_map_path}"
    )

    for csv_path, drop_df in zip(csv_paths, all_dfs):
        drop_dir = csv_path.parent.parent
        labels_dir = drop_dir / "labels"
        images_dir = drop_dir / "frames"
        convert_annotations_to_yolo(drop_df, class_map, labels_dir, images_dir)
        logging.debug(f"  Wrote labels for {drop_dir.name} → {labels_dir}")

    return class_map


def download_extra_volume_labels(
    volume_id: int,
    deployment_data_dir: Path,
    class_map_path: Optional[Path] = None,
    report_type: Optional[int] = None,
) -> Dict[str, int]:
    """
    Download annotations from any Biigle volume into a drop-shaped bundle
    under `deployment_data_dir/extra_no_survey_id/volume_<id>/`:

        volume_<id>/
          frames/                                   <- user populates with JPEGs
          annotations/
            volume_<id>_biigle_expert_raw.csv       <- raw Biigle export
            class_map.json                          <- sidecar (audit-only)
          labels/
            <image_stem>.txt                         <- YOLO labels

    This matches the normal deployment-data shape so downstream pipeline steps
    (`biigle_to_yolo`, `flatten_and_remap_labels`) pick it up via their usual
    globs — no parallel code path needed.
    """
    import shutil

    from spyfish.biigle.biigle_parser import BiigleParser

    if report_type is None:
        report_type = config.annotation_report_type_images

    drop_name = f"volume_{volume_id}"
    source_dir = deployment_data_dir / "extra_no_survey_id" / drop_name
    annotations_dir = source_dir / "annotations"
    labels_dir = source_dir / "labels"
    frames_dir = source_dir / "frames"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    parser = BiigleParser()
    logging.info(f"Downloading annotations for volume {volume_id}...")
    df = parser.download_volume_annotations(volume_id, type_id=report_type)

    if df.empty:
        logging.warning(f"No annotations found for volume {volume_id}.")
        return {}

    raw_csv_path = annotations_dir / f"{drop_name}{config.biigle_expert_raw_suffix}"
    df.to_csv(raw_csv_path, index=False)
    logging.info(f"Saved raw CSV ({len(df)} rows) → {raw_csv_path}")

    # Freeze the class_map used at download time (audit trail for humans).
    resolved_class_map_path = class_map_path or config.class_map_path
    sidecar_class_map = annotations_dir / "class_map.json"
    shutil.copy2(resolved_class_map_path, sidecar_class_map)
    logging.info(f"Froze class_map sidecar → {sidecar_class_map}")

    class_map = load_class_map(resolved_class_map_path)
    summary = convert_annotations_to_yolo(df, class_map, labels_dir, frames_dir)
    logging.info(f"Wrote {len(summary)} YOLO label files → {labels_dir}")
    logging.info(
        f"Bundle ready at {source_dir} — populate {frames_dir}/ with JPEGs "
        "before running the training pipeline."
    )
    return class_map


def download_training_volume_labels(
    volume_id: int,
    class_map_path: Optional[Path] = None,
    report_type: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Download expert annotations from a survey-level *Training frames* Biigle
    volume and split them back to each deployment under the canonical
    deployment-data layout. One such volume holds ~10 frames from MANY drops,
    and every frame filename carries its drop_id
    (``{drop_id}__frame_<secs>s.jpg``):

        {survey}/{drop_id}/
          training_frames/                                <- the JPEGs (already present)
          annotations/{drop_id}_biigle_training_raw.csv   <- this drop's slice of the report
          labels/{stem}.txt                               <- YOLO labels; an EMPTY file == background/negative

    Negatives: a Biigle annotation report only contains
    frames that HAVE annotations, so the no-fish frames are
    ``volume file list − annotated``. We fetch the volume's full file list (the
    universe the expert was shown) via ``get_volume_images`` and write an empty
    ``.txt`` for every frame that has no annotation — both the no-fish frames
    inside reviewed drops and every frame of a fully-empty drop. An empty
    ``.txt`` is a YOLO background image; the dataset-level background proportion
    (Ultralytics rec ~0-10%) is a capping decision made later at assembly, not
    here — this step records every reviewed frame's true label honestly.

    No "Done" gate (the report is fetched directly). Training-only: nothing is
    written to the annotations DB and no pipeline status is advanced — unlike
    ``sync_biigle_annotations``.

    The raw CSV uses its own ``_biigle_training_raw.csv`` suffix (not the expert
    ``_biigle_expert_raw.csv``), so a training import can never overwrite the
    expert MaxN-review export or be mistaken for completed expert review.

    Contamination guard: drops that already have an expert MaxN CSV (real
    per-drop species review via ``--biigle-sync``) are SKIPPED. The raw CSVs no
    longer collide, but the per-drop ``labels/`` dir is shared — and MaxN-backed
    drops can be split into val/test, so without this skip their training frames
    would leak into evaluation. Those drops train via the normal MaxN path; this
    mirrors ``discover_extra_drops``, which also skips MaxN-backed drops.

    The integer class ids written into the .txt files are PROVISIONAL — they
    come from the current global class map purely so each box has *some* id.
    Training rebuilds the unified class ordering from the raw CSVs
    (``discover_extra_drops`` + ``flatten_and_remap_labels``) and remaps every
    .txt, so there is deliberately no class-map argument: the latest project map
    is always the right one and the specific ids here don't survive to the model.

    Contrast with ``download_extra_volume_labels``, which collapses a whole
    volume into one flat ``extra_no_survey_id/volume_<id>/`` pseudo-drop. Use
    THIS function when the volume's filenames carry real drop_ids (the
    training-frame convention); use that one for external volumes with no
    survey/drop structure.

    Note: per-frame YOLO box normalisation reads each image's real dimensions
    from ``training_frames/``. When the JPEGs aren't on this machine (e.g. they
    live on the HPC/S3), ``convert_annotations_to_yolo`` falls back to a default
    resolution and logs a warning — empty negatives are unaffected, but
    positive boxes are only geometrically correct where the frames are present.

    Returns ``{drop_id: {"frames": n, "positives": n, "negatives": n, "boxes": n}}``.
    """
    from spyfish.biigle.biigle_handler import BiigleHandler
    from spyfish.biigle.biigle_parser import BiigleParser

    if report_type is None:
        report_type = config.annotation_report_type_images

    handler = BiigleHandler()
    parser = BiigleParser()
    class_map = load_class_map(class_map_path or config.class_map_path)

    # Universe: every frame the expert was shown. The report lists only
    # annotated frames, so negatives = universe − annotated. /volumes/{id}/files
    # returns bare IDs, so get_volume_images is the only way to resolve filenames.
    logging.info(f"Volume {volume_id}: resolving full file list (universe)...")
    images = handler.get_volume_images(volume_id)
    universe_by_drop: Dict[str, List[str]] = {}
    for img in images:
        fname = img.get("filename")
        if not fname or "__frame_" not in fname:
            continue
        universe_by_drop.setdefault(fname.split("__frame_")[0], []).append(fname)

    # Positives: the annotation report, grouped by drop_id parsed from filename.
    df = parser.download_volume_annotations(volume_id, type_id=report_type)
    report_cols = list(df.columns)
    report_by_drop: Dict[str, pd.DataFrame] = {}
    if not df.empty and "filename" in df.columns:
        drop_ids = df["filename"].astype(str).str.split("__frame_").str[0]
        report_by_drop = {d: g.copy() for d, g in df.groupby(drop_ids)}

    all_drops = sorted(set(universe_by_drop) | set(report_by_drop))
    if not all_drops:
        logging.warning(
            f"Volume {volume_id}: no '{'{drop_id}__frame_<secs>s.jpg'}' filenames "
            "found — is this a training-frames volume?"
        )
        return {}

    summary: Dict[str, Dict[str, int]] = {}
    skipped: List[str] = []
    empty_drops: List[str] = []

    for drop_id in all_drops:
        try:
            config.validate_drop_id(drop_id)
        except ValueError:
            skipped.append(drop_id)
            continue

        drop_dir = config.get_drop_dir(drop_id)
        labels_dir = drop_dir / "labels"
        annotations_dir = config.get_drop_annotations_dir(drop_id)
        training_frames_dir = drop_dir / "training_frames"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        drop_df = report_by_drop.get(drop_id)
        is_empty_drop = drop_df is None or drop_df.empty
        if is_empty_drop:
            empty_drops.append(drop_id)

        # Raw CSV slice → the TRAINING suffix (not the expert one), so it never
        # collides with the expert MaxN-review export and isn't mistaken for it.
        # Header-only for fully-empty drops so discover_extra_drops still sees it.
        raw_path = config.get_biigle_training_raw_csv_path(drop_id)
        slice_df = drop_df if drop_df is not None else pd.DataFrame(columns=report_cols)
        slice_df.to_csv(raw_path, index=False)

        # Positives → YOLO boxes. convert_annotations_to_yolo writes one .txt per
        # annotated frame (a frame carrying only workflow labels yields an empty
        # .txt, which is itself a valid negative).
        written_stems: set = set()
        positives = boxes = 0
        if not is_empty_drop:
            file_summary = convert_annotations_to_yolo(
                drop_df, class_map, labels_dir, training_frames_dir
            )
            written_stems = {Path(f).stem for f in file_summary}
            positives = sum(1 for n in file_summary.values() if n > 0)
            boxes = sum(file_summary.values())

        # Negatives: every universe frame without a label gets an empty .txt.
        negatives = 0
        for fname in universe_by_drop.get(drop_id, []):
            stem = Path(fname).stem
            if stem in written_stems:
                continue
            (labels_dir / f"{stem}.txt").write_text("")
            written_stems.add(stem)
            negatives += 1
        # Total negatives = universe negatives just written + any annotated
        # frames whose only labels were workflow markers (convert wrote those as
        # empty .txt too). Derive from the truth — total frames minus positives.
        negatives = len(written_stems) - positives

        summary[drop_id] = {
            "frames": len(written_stems),
            "positives": positives,
            "negatives": negatives,
            "boxes": boxes,
        }

    total_frames = sum(s["frames"] for s in summary.values())
    total_pos = sum(s["positives"] for s in summary.values())
    total_neg = sum(s["negatives"] for s in summary.values())
    total_boxes = sum(s["boxes"] for s in summary.values())
    bg_pct = (100.0 * total_neg / total_frames) if total_frames else 0.0
    logging.info(
        f"Volume {volume_id}: {len(summary)} drop(s) "
        f"({len(empty_drops)} fully-empty) — {total_frames} frames "
        f"({total_pos} positive, {total_neg} background = {bg_pct:.0f}%), "
        f"{total_boxes} boxes."
    )
    if empty_drops:
        logging.info(
            f"  Fully-empty drops (audit these before trusting as negatives): "
            f"{sorted(empty_drops)}"
        )
    if skipped:
        logging.warning(
            f"  Skipped {len(skipped)} filename prefix(es) that aren't valid "
            f"drop_ids: {sorted(skipped)}"
        )
    return summary


def _drop_id_from_volume_name(name: str) -> Optional[str]:
    """Parse a canonical drop_id from a volume name like
    `"{drop_id} — video labels"` or `"{drop_id} — ML frames"`. Returns None when
    the first token doesn't validate as a drop_id (i.e. a multi-drop volume)."""
    head = re.split(r"\s+|—|–|-", name.strip(), maxsplit=1)[0]
    try:
        config.validate_drop_id(head)
    except ValueError:
        return None
    return head


def download_project_volume_labels(
    project_id: int, force: bool = False
) -> Dict[int, Dict[str, Dict[str, int]]]:
    """Per-project download wrapper around `download_training_volume_labels`.

    For each volume in `project_id`:
      - if the volume name starts with a canonical drop_id (per-drop volume)
        and that drop's `_biigle_training_raw.csv` already exists, skip unless
        ``force`` — fast path, no API calls beyond the project's volume list;
      - otherwise hand off to `download_training_volume_labels(volume_id)`
        (which handles multi-drop survey-level volumes via the
        `{drop_id}__frame_<secs>s.jpg` filename convention).

    Returns ``{volume_id: per-drop summary}`` for volumes that were downloaded.
    Skipped volumes are not in the dict.
    """
    from spyfish.biigle.biigle_handler import BiigleHandler

    handler = BiigleHandler()
    volumes = handler.get_volumes(project_id)
    logging.info(f"Project {project_id}: {len(volumes)} volume(s)")

    out: Dict[int, Dict[str, Dict[str, int]]] = {}
    skipped = downloaded = 0
    for v in volumes:
        vol_id = v["id"]
        name = v.get("name", "")
        drop_id = _drop_id_from_volume_name(name)
        if drop_id and not force:
            raw_path = config.get_biigle_training_raw_csv_path(drop_id)
            if raw_path.exists():
                logging.info(f"  skip {vol_id} ({drop_id}) — {raw_path.name} exists")
                skipped += 1
                continue
        logging.info(f"  download {vol_id} ({name})")
        out[vol_id] = download_training_volume_labels(vol_id)
        downloaded += 1

    logging.info(
        f"Project {project_id}: downloaded {downloaded}, skipped {skipped} "
        f"(use --force to re-download)"
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert local Biigle expert CSVs to YOLO labels."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Existing: convert local CSVs
    convert_cmd = subparsers.add_parser(
        "convert", help="Convert local expert CSVs to YOLO labels"
    )
    convert_cmd.add_argument(
        "--data-dir", required=True, type=Path, help="Root deployment_data directory"
    )
    convert_cmd.add_argument(
        "--class-map",
        required=True,
        type=Path,
        help="Path to class_map.json (seed via `python -m spyfish.biigle.class_map`)",
    )

    # New: download from arbitrary volume
    download_cmd = subparsers.add_parser(
        "download-volume", help="Download labels from any Biigle volume"
    )
    download_cmd.add_argument(
        "--volume-id", required=True, type=int, help="Biigle volume ID"
    )
    download_cmd.add_argument(
        "--deployment-data-dir",
        type=Path,
        default=None,
        help=(
            "Root deployment_data directory (default: config.deployment_data_dir). "
            "A drop-shaped bundle is created at "
            "<deployment-data-dir>/extra_no_survey_id/volume_<id>/ containing "
            "frames/, annotations/ (raw CSV + class_map sidecar), and labels/."
        ),
    )
    download_cmd.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Path to class_map.json (defaults to config.class_map_path)",
    )
    download_cmd.add_argument(
        "--report-type",
        type=int,
        default=None,
        help="Biigle report type ID (default: image annotations)",
    )

    # New: download a survey-level *Training frames* volume, split per drop_id.
    # Intentionally only --volume-id: the class map is always the latest project
    # map (training rewrites the ids anyway) and the report is always image-type.
    train_cmd = subparsers.add_parser(
        "download-training-volume",
        help=(
            "Download labels from a survey-level Training-frames volume and "
            "split them per drop_id into survey/dropid/ (boxes + empty-.txt "
            "negatives). Training-only: no DB writes, no Done gate."
        ),
    )
    train_cmd.add_argument(
        "--volume-id", required=True, type=int, help="Biigle training-frames volume ID"
    )

    # New: per-project loop. Skips per-drop volumes whose training raw CSV is
    # already on disk; --force re-downloads everything.
    proj_cmd = subparsers.add_parser(
        "download-project",
        help=(
            "Download labels for every volume in a BIIGLE project. Per-drop "
            "volumes whose `_biigle_training_raw.csv` already exists are "
            "skipped (use --force to re-download)."
        ),
    )
    proj_cmd.add_argument(
        "--project-id", required=True, type=int, help="BIIGLE project ID"
    )
    proj_cmd.add_argument(
        "--force", action="store_true", help="Re-download even when local CSV exists"
    )

    args = parser.parse_args()

    if args.command == "convert":
        biigle_to_yolo(args.data_dir, args.class_map)
    elif args.command == "download-volume":
        deployment_data_dir = args.deployment_data_dir or config.deployment_data_dir
        download_extra_volume_labels(
            args.volume_id, deployment_data_dir, args.class_map, args.report_type
        )
    elif args.command == "download-training-volume":
        download_training_volume_labels(args.volume_id)
    elif args.command == "download-project":
        download_project_volume_labels(args.project_id, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
