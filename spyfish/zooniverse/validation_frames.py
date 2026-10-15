"""
Zooniverse validation subjects: frames for recall, crops for precision.

Volunteers check the ML output directly, on the same moments a BIIGLE expert
would review:

  * **frames**: whole frames with every ML box drawn, no labels. Volunteers
    draw the fish the model missed.
  * **crops**: one crop per ML box in those frames, labelled with the predicted
    common name. Volunteers judge right species / right box.

Every box in a picked frame is cropped, so each picked frame ends up fully
checked from both sides.

Frames are picked by `select_frames.select_frames`, the BIIGLE selector, with
its quotas from `extraction.frame_strategy` and only the effort multiplier
taken from `zooniverse.validation`. The selection is written once per drop and
shared: whichever of --zooniverse-frames / --zooniverse-crops runs first writes
it, the other reads it.

Everything is built from the existing `--ml` output: the raw CSV and the local
video it ran on. Nothing is re-inferred, so volunteers judge exactly the boxes
the pipeline reported.

The inference grid
------------------
`run_inference` samples one frame every `stride` frames from wherever cv2's
`POS_MSEC` seek to `sampling_start` lands, and writes rows only for frames that
had detections. The stride is not saved anywhere. `inference_grid` rebuilds it
the same way, so a blind frame can be snapped onto a frame the model actually
looked at: "no box" then means "the model looked and saw nothing", not "the
model never looked here".
"""

import bisect
import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from spyfish.config.base import ZooCropsStatus, ZooFramesStatus
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_frames import (
    read_frame_from_cap,
    read_video_rotation,
    write_frame_jpeg,
)
from spyfish.extraction.render_annotations import (
    center_to_corners,
    crop_window,
    render_annotated_frame,
    render_detection_crop,
    rotate_bbox,
    warn_if_rotated,
)
from spyfish.extraction.select_frames import select_frames
from spyfish.ml.run_inference import inference_stride, seek_to_sampling_start
from spyfish.utils import generate_frame_filename

# The uploader is referenced through its module, not by importing names: the
# upload functions are patched on `spyfish.zooniverse.upload` in tests and in
# --dry-run reasoning, and a bound name here would not see those patches.
from spyfish.zooniverse import upload as zoo_upload
from spyfish.zooniverse.subject_keys import CROP_PATH_COLUMN, FRAME_PATH_COLUMN

# Red with black hairlines (draw_box adds them): the crop pilot's look, and the
# most visible stroke against blue/green water. Presentation, not a threshold.
BOX_BGR = (0, 0, 255)
JPEG_QUALITY = 90


@dataclass
class ValidationContext:
    """Everything both stages need for one drop, resolved once."""

    drop_id: str
    model_name: str
    video_path: Path
    sampling_start: float
    detections: pd.DataFrame  # raw CSV rows ≥ confidence_threshold
    frames: pd.DataFrame  # one row per picked frame, snapped to the grid


# ── inputs ───────────────────────────────────────────────────────────────────


def load_detections(raw_csv: Path) -> pd.DataFrame:
    """Raw CSV rows at or above the pipeline confidence threshold.

    Filtered here rather than trusted as-is: a raw CSV written under an older,
    lower threshold would otherwise show volunteers boxes the pipeline no
    longer reports.
    """
    df = pd.read_csv(raw_csv)
    df = df[df["confidence"] >= config.confidence_threshold].copy()
    # Row order within a frame is the model's output order, kept stable so a
    # box's index in its frame (the DetectionKey) is reproducible.
    df["box_index"] = df.groupby("frame").cumcount()
    return df.reset_index(drop=True)


def _open_local(video_path: Path, drop_id: str) -> cv2.VideoCapture:
    """The local video, never an S3 stream (see `prepare_validation`)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"{drop_id}: cannot open {video_path}")
    return cap


def inference_grid(
    cap: cv2.VideoCapture,
    sampling_start: float,
    sampling_end: float,
    detections: pd.DataFrame,
) -> tuple[float, list[int]]:
    """``(fps, frame indices)`` that `--ml` sampled, rebuilt with its own helpers.

    Same seek (`seek_to_sampling_start`) and same stride (`inference_stride`)
    as `run_inference`, stepping while ``frame / fps <= sampling_end``. If the
    raw CSV disagrees (detection frames off the rebuilt grid), the grid is
    re-anchored on the detections, since those are frames inference provably
    saw, and the disagreement is logged.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"video reported an invalid FPS of {fps}")
    stride = inference_stride(fps)
    f0 = seek_to_sampling_start(cap, sampling_start)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not detections.empty:
        offsets = set(int(f) % stride for f in detections["frame"].unique())
        if len(offsets) == 1 and offsets != {f0 % stride}:
            anchor = offsets.pop()
            logging.warning(
                f"Rebuilt grid (start frame {f0}, stride {stride}) is off the raw "
                f"CSV's frames; re-anchoring on the detections (offset {anchor})."
            )
            f0 += (anchor - f0) % stride
        elif len(offsets) > 1:
            logging.warning(
                f"Raw CSV frames sit on {len(offsets)} different offsets of stride "
                f"{stride}; was it written by more than one run? Using the seek."
            )

    last = min(total - 1, int(sampling_end * fps))
    grid = [f for f in range(f0, last + 1, stride) if f / fps <= sampling_end]
    return fps, grid


def _snap(grid: list[int], target: int) -> int:
    i = bisect.bisect_left(grid, target)
    candidates = [grid[j] for j in (i - 1, i) if 0 <= j < len(grid)]
    return min(candidates, key=lambda f: abs(f - target))


# ── selection ────────────────────────────────────────────────────────────────


def get_or_create_selection(
    drop_id: str, raw_csv: Path, model_name: str
) -> pd.DataFrame:
    """The drop's shared validation selection, written on first use.

    Reused only when it was made by the same model at the same effort; anything
    else is stale and is rebuilt, because a selection from another model points
    at moments its boxes no longer describe.
    """
    path = config.get_validation_selection_path(drop_id)
    effort = config.zooniverse_validation_effort_multiplier
    if path.exists():
        existing = pd.read_csv(path)
        # Columns checked for presence first: `.all()` on a missing column's
        # empty stand-in is True, which would reuse any old selection.
        if (
            not existing.empty
            and {"ModelName", "EffortMultiplier"} <= set(existing.columns)
            and existing["ModelName"].eq(model_name).all()
            and existing["EffortMultiplier"].eq(effort).all()
        ):
            logging.info(f"{drop_id}: reusing validation selection {path.name}")
            return existing
        logging.info(f"{drop_id}: validation selection is stale; rebuilding.")

    selection = select_frames(
        str(raw_csv), str(path), drop_id, effort_multiplier=effort
    )

    selection["ModelName"] = model_name
    selection["EffortMultiplier"] = effort
    selection.to_csv(path, index=False)
    return selection


def snap_selection_to_grid(
    selection: pd.DataFrame, grid: list[int], fps: float
) -> pd.DataFrame:
    """One row per distinct grid frame, carrying the first reason that picked it.

    Detection-driven picks already sit on the grid (their times came from raw
    CSV rows), so snapping leaves them where they are. Blind picks are
    free-floating seconds and move to the nearest sampled frame. Two picks that
    land on the same frame are one subject.
    """
    time_col = config.csv_clip_max_time_column
    species_col = config.csv_scientific_name_column
    reason_col = config.selection_reason_column
    snapped = selection.sort_values(time_col, kind="stable")
    snapped = snapped.assign(
        frame=[_snap(grid, int(round(float(t) * fps))) for t in snapped[time_col]]
    ).drop_duplicates("frame")
    snapped["time_seconds"] = snapped["frame"] / fps
    if species_col not in snapped:
        snapped[species_col] = None
    return snapped[["frame", "time_seconds", reason_col, species_col]].reset_index(
        drop=True
    )


# Cached per drop for the life of the process: --zooniverse-frames and
# --zooniverse-crops run together on the same drops, and both need exactly
# this (raw CSV, video probe, grid, selection). The second stage reuses it.
@functools.lru_cache(maxsize=8)
def prepare_validation(drop_id: str) -> Optional[ValidationContext]:
    """Resolve inputs and the snapped selection, or None if the drop isn't ready.

    Not ready = no local video, no raw CSV for the pipeline model, or no
    sampling window. The stage
    leaves the status untouched in that case. Local video only, never streamed
    from S3: `--ml` already put it in `media/`, and a 10 GB stream per stage run
    is not a fallback worth having.
    """
    config.validate_drop_id(drop_id)
    model_name = Path(config.pipeline_model_path).stem
    video_path = config.get_video_path(drop_id)
    raw_csv = config.get_raw_csv_path(drop_id, model_name)
    if not video_path.exists():
        logging.warning(
            f"{drop_id}: video not on local disk ({video_path}); not ready."
        )
        return None
    if not raw_csv.exists():
        logging.warning(
            f"{drop_id}: no raw CSV for {model_name} ({raw_csv}); not ready."
        )
        return None

    deployment = DatabaseManager().get_deployment(drop_id)
    if not deployment or None in (
        deployment.get("sampling_start"),
        deployment.get("sampling_end"),
    ):
        # Not ready rather than an error, same as BIIGLE upload: without a
        # window there are no blind frames, so wait for the metadata.
        logging.warning(f"{drop_id}: no sampling window yet; not ready.")
        return None
    start, end = float(deployment["sampling_start"]), float(deployment["sampling_end"])

    detections = load_detections(raw_csv)
    cap = _open_local(video_path, drop_id)
    try:
        fps, grid = inference_grid(cap, start, end, detections)
    finally:
        cap.release()
    selection = get_or_create_selection(drop_id, raw_csv, model_name)
    frames = snap_selection_to_grid(selection, grid, fps)
    logging.info(
        f"{drop_id}: {len(frames)} validation frame(s) from {len(selection)} "
        f"selection(s), {len(detections)} detection(s) at conf ≥ "
        f"{config.confidence_threshold}."
    )
    return ValidationContext(
        drop_id=drop_id,
        model_name=model_name,
        video_path=video_path,
        sampling_start=start,
        detections=detections,
        frames=frames,
    )


# ── rendering ────────────────────────────────────────────────────────────────


def _with_corners(boxes: pd.DataFrame, rotation: int, src_w: int, src_h: int):
    """Add corner columns once: ``rx1..ry2`` raw (video pixels, as in the raw
    CSV) and ``x1..y2`` in the extracted frame's space (rotated like its pixels)."""
    raw = [center_to_corners(*r) for r in boxes[["x", "y", "w", "h"]].to_numpy()]
    shown = [rotate_bbox(*c, rotation, src_w, src_h) for c in raw] if rotation else raw
    return boxes.join(
        [
            pd.DataFrame(raw, columns=["rx1", "ry1", "rx2", "ry2"], index=boxes.index),
            pd.DataFrame(shown, columns=["x1", "y1", "x2", "y2"], index=boxes.index),
        ]
    )


def _read_frames(ctx: ValidationContext):
    """Yield ``(frame_row, image, clean_path, boxes)`` per picked frame.

    The clean frame lives in the drop's shared `frames/` folder under the
    standard name, like every other extracted frame, so any later step (BIIGLE,
    training) finds it there. It is read by frame index, the same index
    inference recorded, so the pixels are the ones the boxes were computed on. A frame already on disk (BIIGLE review
    took the same moment) is reused, not extracted twice. `boxes` carries
    corner columns (see `_with_corners`) so consumers never convert again.
    """
    frames_dir = config.get_frames_dir(ctx.drop_id)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = _open_local(ctx.video_path, ctx.drop_id)
    rotation = read_video_rotation(cap)
    warn_if_rotated(rotation, ctx.drop_id)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    picked = ctx.detections[ctx.detections["frame"].isin(ctx.frames["frame"])]
    boxes_all = _with_corners(picked, rotation, src_w, src_h)
    by_frame = dict(tuple(boxes_all.groupby("frame")))
    empty = boxes_all.iloc[0:0]
    try:
        for _, fr in ctx.frames.iterrows():
            clean = frames_dir / generate_frame_filename(
                ctx.drop_id, fr["time_seconds"]
            )
            if clean.exists():
                image = cv2.imread(str(clean))
            else:
                # Read once and keep the pixels: saved as the clean frame and
                # drawn on directly, never re-read from the JPEG just written.
                image = read_frame_from_cap(
                    cap, fr["time_seconds"], int(fr["frame"]), rotation
                )
                if image is None:
                    logging.warning(
                        f"{ctx.drop_id}: could not read frame {fr['frame']}"
                    )
                    continue
                write_frame_jpeg(image, clean)
            yield fr, image, clean, by_frame.get(int(fr["frame"]), empty)
    finally:
        cap.release()


def _base_row(ctx: ValidationContext, fr, clean: Path) -> dict:
    """Columns `upload._build_base_subject_meta` reads, same names as the clip CSVs."""
    return {
        config.drop_id_column: ctx.drop_id,
        config.csv_sampling_start_column: ctx.sampling_start,
        config.csv_clip_max_time_column: round(float(fr["time_seconds"]), 3),
        config.selection_reason_column: fr[config.selection_reason_column],
        "FrameIndex": int(fr["frame"]),
        "CleanFramePath": str(clean),
        "ModelName": ctx.model_name,
    }


def _box_json(boxes: pd.DataFrame) -> str:
    """Source-pixel (unrotated, as in the raw CSV) corner boxes for the manifest."""
    return json.dumps(
        [
            {
                "box_index": int(d.box_index),
                "class": d["class"],
                "confidence": round(float(d.confidence), 4),
                "bbox": [round(float(v), 1) for v in (d.rx1, d.ry1, d.rx2, d.ry2)],
            }
            for _, d in boxes.iterrows()
        ]
    )


def _write(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise RuntimeError(f"failed to write {path}")


def render_validation_frames(ctx: ValidationContext) -> pd.DataFrame:
    """Write one JPEG per picked frame with every ML box drawn; return the manifest."""
    out_dir = config.get_validation_upload_dir(ctx.drop_id)
    max_w = config.zooniverse_validation_frame_max_width_px
    rows = []
    for fr, image, clean, boxes in _read_frames(ctx):
        dets = boxes[["x1", "y1", "x2", "y2"]].to_dict("records")
        rendered = render_annotated_frame(
            image, dets, box_colour=BOX_BGR, max_width_px=max_w
        )
        name = generate_frame_filename(ctx.drop_id, fr["time_seconds"]).replace(
            ".jpg", "__vframe.jpg"
        )
        _write(out_dir / name, rendered)
        rows.append(
            {
                **_base_row(ctx, fr, clean),
                FRAME_PATH_COLUMN: str(out_dir / name),
                config.csv_scientific_name_column: fr[
                    config.csv_scientific_name_column
                ],
                config.csv_max_interval_column: len(boxes),
                config.csv_confidence_agreement_column: (
                    round(float(boxes["confidence"].mean()), 4) if len(boxes) else None
                ),
                # Volunteer-drawn boxes come back in the subject's pixels; this
                # maps them to video pixels.
                "RenderScale": round(rendered.shape[1] / image.shape[1], 6),
                "Boxes": _box_json(boxes),
            }
        )
    manifest = pd.DataFrame(rows)
    _save_manifest(ctx.drop_id, "frames", manifest)
    return manifest


def render_validation_crops(ctx: ValidationContext) -> pd.DataFrame:
    """Write one crop per ML box in the picked frames; return the manifest.

    Boxes too small to judge once enlarged, or too large for a crop window
    (they would render as two lines across the scene), get a manifest row with
    no image and a reason. They are still judged on the frames.
    """
    out_dir = config.get_validation_upload_dir(ctx.drop_id)
    params = config.zooniverse_validation_crops
    registry = species_registry()
    rows = []
    for fr, image, clean, boxes in _read_frames(ctx):
        frame_h, frame_w = image.shape[:2]
        counts = boxes["class"].value_counts()
        for _, d in boxes.iterrows():
            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
            raw_corners = (d.rx1, d.ry1, d.rx2, d.ry2)
            idx = int(d["box_index"])
            label = registry.label_for(d["class"])
            row = {
                **_base_row(ctx, fr, clean),
                CROP_PATH_COLUMN: None,
                config.csv_scientific_name_column: d["class"],
                config.csv_max_interval_column: int(counts[d["class"]]),
                config.csv_confidence_agreement_column: round(
                    float(d["confidence"]), 4
                ),
                "DetectionKey": f"{ctx.drop_id}|{int(fr['frame'])}|{idx}",
                "BBox": ",".join(str(int(round(v))) for v in raw_corners),
                "LabelShown": label,
                "NotCroppedReason": None,
            }
            left, top, right, bottom = crop_window(
                x1,
                y1,
                x2,
                y2,
                frame_w,
                frame_h,
                params["crop_scale"],
                params["max_frame_fraction"],
            )
            if max(x2 - x1, y2 - y1) < params["min_bbox_px"]:
                row["NotCroppedReason"] = f"box under {params['min_bbox_px']} px"
            elif (x2 - x1) > (right - left) or (y2 - y1) > (bottom - top):
                row["NotCroppedReason"] = "box larger than crop window"
            else:
                crop = render_detection_crop(
                    image,
                    x1,
                    y1,
                    x2,
                    y2,
                    label,
                    scale=params["crop_scale"],
                    max_frame_fraction=params["max_frame_fraction"],
                    output_px=params["crop_image_px"],
                    box_colour=BOX_BGR,
                )
                # The frame prefix links a crop to its frame by string match;
                # the box suffix lets anyone redraw it from the name alone.
                bx = "_".join(str(int(round(v))) for v in raw_corners)
                name = generate_frame_filename(ctx.drop_id, fr["time_seconds"]).replace(
                    ".jpg", f"__crop_{idx:02d}__box_{bx}.jpg"
                )
                _write(out_dir / name, crop)
                row[CROP_PATH_COLUMN] = str(out_dir / name)
            rows.append(row)
    manifest = pd.DataFrame(rows)
    _save_manifest(ctx.drop_id, "crops", manifest)
    return manifest


def _save_manifest(drop_id: str, kind: str, manifest: pd.DataFrame) -> None:
    path = config.get_validation_manifest_path(drop_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    logging.info(f"{drop_id}: {kind} manifest → {path} ({len(manifest)} row(s))")


# ── stage entry points (wired into run_pipeline.py STAGES) ───────────────────
#
# DropStage contract: (drop_id) -> next status, or None to leave it unchanged
# (not ready, or a dry run). The pipeline file only registers these.


def _already_uploaded(kind, kind_name: str, path_column: str, drop_id: str) -> bool:
    """True if this drop's set is already complete on Zooniverse; checked before
    any rendering. A set with a different count is NOT complete: the stage
    renders and the uploader resumes, adding only the missing subjects."""
    on_zooniverse = zoo_upload.existing_subject_count(kind, drop_id)
    if on_zooniverse is None:
        return False
    manifest = config.get_validation_manifest_path(drop_id, kind_name)
    expected = (
        int(pd.read_csv(manifest)[path_column].notna().sum())
        if manifest.exists()
        else None
    )
    if expected == on_zooniverse:
        logging.info(
            f"{drop_id}: {kind.set_name_prefix}{drop_id} already on Zooniverse "
            f"with all {expected} {kind_name}; nothing to render or upload."
        )
        return True
    logging.warning(
        f"{drop_id}: {kind.set_name_prefix}{drop_id} exists on Zooniverse with "
        f"{on_zooniverse} subjects but the local manifest lists {expected}; "
        "rendering and resuming the upload (missing subjects only)."
    )
    return False


def run_validation_frames_drop(drop_id: str, dry_run: bool = False) -> Optional[str]:
    """--zooniverse-frames: picked frames with every ML box drawn, for recall."""
    if not dry_run and _already_uploaded(
        zoo_upload.VALIDATION_FRAME_KIND, "frames", FRAME_PATH_COLUMN, drop_id
    ):
        return ZooFramesStatus.UPLOADED
    ctx = prepare_validation(drop_id)
    if ctx is None:
        return None
    manifest = render_validation_frames(ctx)
    if manifest.empty:
        # Every picked frame failed to decode (a truncated or partial video,
        # or indices past its end). Nothing to upload and nothing wrong with
        # the drop's status, so leave it pending for the next run.
        logging.warning(
            f"{drop_id}: no validation frame could be read from the video; "
            "leaving the drop pending."
        )
        return None
    if dry_run:
        logging.info(
            f"{drop_id}: --dry-run, {len(manifest)} frame(s) in "
            f"{config.get_validation_upload_dir(drop_id)}; nothing uploaded."
        )
        return None
    zoo_upload.upload_validation_frames_to_zooniverse(manifest)
    return ZooFramesStatus.UPLOADED


def run_validation_crops_drop(drop_id: str, dry_run: bool = False) -> Optional[str]:
    """--zooniverse-crops: one crop per ML box in the validation frames, for precision."""
    if not dry_run and _already_uploaded(
        zoo_upload.CROP_KIND, "crops", CROP_PATH_COLUMN, drop_id
    ):
        return ZooCropsStatus.UPLOADED
    ctx = prepare_validation(drop_id)
    if ctx is None:
        return None
    manifest = render_validation_crops(ctx)
    n_crops = int(manifest[CROP_PATH_COLUMN].notna().sum()) if len(manifest) else 0
    if dry_run:
        logging.info(
            f"{drop_id}: --dry-run, {n_crops} crop(s) in "
            f"{config.get_validation_upload_dir(drop_id)}; nothing uploaded."
        )
        return None
    if n_crops == 0:
        logging.info(f"{drop_id}: no croppable boxes in the validation frames.")
        return ZooCropsStatus.SKIPPED
    zoo_upload.upload_validation_crops_to_zooniverse(manifest)
    return ZooCropsStatus.UPLOADED
