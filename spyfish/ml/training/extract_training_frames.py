"""
Standalone CLI: pull N frames per drop directly from S3 (cv2 over presigned URL),
run the configured detector model, and upload to a survey-level Biigle volume
for expert annotation.

Purpose: bootstrap a labeled training set when there's no good species model yet.
Frames are sampled across the deployment's [sampling_start, sampling_end] window,
back-loaded toward the end (where bait-attracted fish density is highest).

Configuration lives under `training_extraction:` in config.yaml; see
`config.training_extraction_n_frames` and `config.training_extraction_annotation_type`.

Usage (from project root):
    python -m spyfish.ml.training.extract_training_frames --drop-id    <DROP>
    python -m spyfish.ml.training.extract_training_frames --survey-id  <SURVEY>
    python -m spyfish.ml.training.extract_training_frames --survey-id  <SURVEY> --force
    python -m spyfish.ml.training.extract_training_frames --drop-id    <DROP>    --no-upload
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd

from spyfish.biigle.upload_frames import (
    find_or_create_volume_and_add_frames,
    upload_coco_annotations_to_biigle,
    upload_frames_to_s3,
)
from spyfish.config.base import VideoPresence
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_frames import (
    _extract_one_frame_from_cap,
    _read_video_rotation,
    build_coco_from_raw_csv,
)
from spyfish.storage.s3_handler import S3Handler
from spyfish.utils import generate_frame_filename

SURVEY_VOLUME_NAME_TEMPLATE = "{survey_id} — Training frames"

# COCO `selection_reason` value for every record produced by this module —
# distinguishable from MaxN-peak frames in the existing extraction flow.
_SELECTION_REASON = "training_frame_random"


# ── Timestamp generation ─────────────────────────────────────────────────────


def _quadratic_timestamps(
    start: float, end: float, n: int, power: float = 2.0
) -> List[float]:
    """Generate N back-loaded timestamps in `[start, end]`.

    Density increases toward `end`: t_i = start + (end - start) × (1 − ((N − i) / N)^power).
    With `power=2.0` and N=10 in [60, 1800], roughly half the points fall in the
    final third of the window — chosen to match BUV bait-attraction dynamics
    (more fish in the second half of a deployment).

    The last timestamp is exactly `end`; all timestamps are strictly increasing.
    Use `power < 2` for milder back-loading, `power > 2` for heavier.

    Doubling N produces a superset (every old t_i appears as the new t_{2i}),
    so re-running with a larger N adds new frames without colliding with the
    original ones at the S3/Biigle filename layer. Verified in unit tests.

    Raises:
        ValueError: n < 1, end <= start, or power <= 0.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")
    if power <= 0:
        raise ValueError(f"power must be > 0, got {power}")
    span = end - start
    return [start + span * (1.0 - ((n - i) / n) ** power) for i in range(1, n + 1)]


# ── Extraction ───────────────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Output of `extract_frames_for_drop`."""

    drop_id: str
    survey_id: str
    frame_paths: List[Path]
    timestamps: List[float]
    fps: float
    img_w: int
    img_h: int
    output_dir: Path


def _training_frames_dir(drop_id: str) -> Path:
    """`process_files/deployment_data/{survey}/{drop}/training_frames/`"""
    return config.get_drop_dir(drop_id) / "training_frames"


def extract_frames_for_drop(
    drop_id: str,
    *,
    n_frames: Optional[int] = None,
    force: bool = False,
) -> ExtractionResult:
    """Extract N back-loaded frames from a drop's S3 video using cv2 over a
    presigned URL — single cv2 open, N seeks, no full-video download.

    Pre-checks:
      - Drop exists in DB.
      - Has sampling_start AND sampling_end.
      - video_presence == 'present' (excludes ABSENT, ARCHIVED, NO_VIDEO_BAD_DEP).
      - training_frames/ dir is empty or missing, unless `force=True`.

    Args:
        drop_id: Deployment identifier.
        n_frames: Override config.training_extraction_n_frames.
        force: Allow re-extraction over existing files.

    Returns:
        ExtractionResult with frame paths (in chronological order), the
        timestamps used, the video's reported FPS + dimensions, and the
        output directory.
    """
    db = DatabaseManager()
    s3 = S3Handler()

    deployment = db.get_deployment(drop_id)
    if deployment is None:
        raise ValueError(f"{drop_id}: not found in deployments DB")

    sampling_start = deployment.get("sampling_start")
    sampling_end = deployment.get("sampling_end")
    if sampling_start is None or sampling_end is None:
        raise ValueError(
            f"{drop_id}: missing sampling window "
            f"(start={sampling_start}, end={sampling_end})"
        )

    presence = deployment.get("video_presence")
    if presence != VideoPresence.PRESENT:
        raise ValueError(
            f"{drop_id}: video_presence={presence!r}; "
            f"only {VideoPresence.PRESENT!r} videos are extractable."
        )

    survey_id = config.get_survey_id_from_drop(drop_id)
    out_dir = _training_frames_dir(drop_id)
    if out_dir.exists() and any(out_dir.glob("*.jpg")) and not force:
        raise FileExistsError(
            f"{drop_id}: training frames already exist at {out_dir}; "
            f"pass --force to regenerate."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(n_frames) if n_frames is not None else config.training_extraction_n_frames
    timestamps = _quadratic_timestamps(float(sampling_start), float(sampling_end), n=n)

    s3_key = config.get_video_s3_key(drop_id)
    t_start = time.monotonic()
    url = s3.generate_presigned_url(s3_key, expiration=3600)
    if url is None:
        raise FileNotFoundError(
            f"{drop_id}: could not generate presigned URL for s3://.../{s3_key} "
            "(404 or insufficient permissions)."
        )
    logging.info(
        f"{drop_id}: presigned URL ready in {time.monotonic() - t_start:.2f}s "
        f"({n} frames in [{sampling_start}, {sampling_end}]s)"
    )

    t_open = time.monotonic()
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(
            f"{drop_id}: cv2 failed to open the presigned URL "
            f"(s3://.../{s3_key}). Possible network/auth issue, or the MP4 "
            "container is unsupported by the local ffmpeg backend."
        )
    logging.info(
        f"{drop_id}: cv2 opened the URL in {time.monotonic() - t_open:.2f}s "
        "(this is the moov-atom fetch — should be a few seconds, not minutes)"
    )

    try:
        rotation = _read_video_rotation(cap)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise ValueError(
                f"{drop_id}: video reports invalid FPS={fps}; can't compute frame indices."
            )
        # Capture container dimensions once. Saved into COCO so Biigle/other
        # consumers can validate annotation overlay geometry. After rotation
        # bake-in, swap w↔h for portrait videos so dims match the saved JPGs.
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rotation in (90, 270):
            vid_w, vid_h = vid_h, vid_w

        paths: List[Path] = []
        for i, t in enumerate(timestamps, start=1):
            out_path = out_dir / generate_frame_filename(drop_id, t)
            t_seek = time.monotonic()
            ok = _extract_one_frame_from_cap(cap, t, out_path, rotation=rotation)
            dt = time.monotonic() - t_seek
            logging.info(f"  [{i:2d}/{n}] t={t:8.3f}s  dt={dt:.2f}s  → {out_path.name}")
            if not ok:
                raise RuntimeError(
                    f"{drop_id}: cv2 failed to extract frame at t={t:.3f}s"
                )
            paths.append(out_path)
    finally:
        cap.release()

    logging.info(f"{drop_id}: extracted {len(paths)} frame(s) → {out_dir}")
    return ExtractionResult(
        drop_id=drop_id,
        survey_id=survey_id,
        frame_paths=paths,
        timestamps=timestamps,
        fps=fps,
        img_w=vid_w,
        img_h=vid_h,
        output_dir=out_dir,
    )


# ── Inference ────────────────────────────────────────────────────────────────


def run_inference_to_csv(
    extraction: ExtractionResult,
    *,
    model_path: Optional[Path] = None,
    annotation_type: Optional[str] = None,
    confidence: Optional[float] = None,
    imgsz: Optional[int] = None,
) -> Path:
    """Run the configured detector model on extracted frames and write a raw CSV
    in the same format produced by `spyfish.ml.run_inference` — meaning the
    existing `build_coco_from_raw_csv` consumes it without modification.

    Output schema: ``frame, time_seconds, class, confidence, x, y, w, h``

    The output path is `{training_frames}/{drop_id}_{kind}_raw.csv`, mirroring
    the `_raw.csv` convention for inference outputs in this project.
    """
    # Lazy import: ultralytics pulls in torch which is heavy. Most callers of
    # this module only need `_quadratic_timestamps` (e.g. unit tests).
    from ultralytics import YOLO

    kind = annotation_type or config.training_extraction_annotation_type
    if model_path is None:
        model_path = config.get_pipeline_model(kind)
    conf = (
        float(confidence)
        if confidence is not None
        else float(config.confidence_threshold)
    )
    img_size = int(imgsz) if imgsz is not None else int(config.imgsz)

    out_csv = extraction.output_dir / f"{extraction.drop_id}_{kind}_raw.csv"
    logging.info(
        f"{extraction.drop_id}: running '{kind}' inference "
        f"on {len(extraction.frame_paths)} frame(s) using {model_path.name}"
    )

    model = YOLO(str(model_path))

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]
        )

        n_detections = 0
        for path, t in zip(extraction.frame_paths, extraction.timestamps):
            # Synthesized frame index — there's no contiguous decode here
            # (we ran cv2 .set/.read per timestamp), so this is the nominal
            # frame number for the seek time. Downstream COCO matching uses
            # `time_seconds`, not `frame`, so this is informational.
            frame_idx = int(round(t * extraction.fps))
            results = model.predict(
                source=str(path),
                conf=conf,
                imgsz=img_size,
                verbose=False,
                project=None,
                save=False,
            )
            r = results[0]
            for box in r.boxes:
                x, y, w, h = box.xywh[0].tolist()
                cls_id = int(box.cls[0])
                writer.writerow(
                    [
                        frame_idx,
                        float(t),
                        model.names[cls_id],
                        float(box.conf[0]),
                        x,
                        y,
                        w,
                        h,
                    ]
                )
                n_detections += 1

    logging.info(
        f"{extraction.drop_id}: {n_detections} detection(s) across "
        f"{len(extraction.frame_paths)} frame(s) → {out_csv.name}"
    )
    return out_csv


# ── COCO + Biigle upload ─────────────────────────────────────────────────────


def upload_drop_to_survey_volume(
    extraction: ExtractionResult,
    raw_csv_path: Path,
) -> int:
    """Upload one drop's training frames + ML annotations to the survey volume.

    Builds a COCO JSON from the raw inference CSV (reusing the existing
    `build_coco_from_raw_csv`), saves it next to the frames, ensures the
    survey-level Biigle volume exists, uploads the frames to S3, attaches them
    to the volume, and pushes the COCO annotations.

    Returns the Biigle volume ID for storage in the per-drop DB row.
    """
    # 1. Build COCO from the raw CSV. Image dimensions come from the cv2 cap
    #    that was open during extraction (insurance against any future
    #    Biigle display-side validation that compares against image dims).
    frame_records = [
        {
            "image_id": i,
            "file_name": path.name,
            "time_of_max": float(t),
            "drop_id": extraction.drop_id,
            "selection_reason": _SELECTION_REASON,
            "img_w": extraction.img_w,
            "img_h": extraction.img_h,
        }
        for i, (path, t) in enumerate(
            zip(extraction.frame_paths, extraction.timestamps), start=1
        )
    ]

    coco = build_coco_from_raw_csv(str(raw_csv_path), frame_records)
    coco_path = extraction.output_dir / (
        f"{extraction.drop_id}_coco_annotations_for_biigle.json"
    )
    with open(coco_path, "w") as fh:
        json.dump(coco, fh, indent=2)
    logging.info(
        f"{extraction.drop_id}: COCO → {coco_path.name} "
        f"({len(coco['images'])} images, {len(coco['annotations'])} annotations)"
    )

    # 2. Upload JPGs to the survey-level S3 prefix (idempotent: existing keys skipped).
    s3_prefix = config.get_training_frames_s3_prefix(extraction.survey_id)
    frames_df = pd.DataFrame({"FramePath": [str(p) for p in extraction.frame_paths]})
    file_names = upload_frames_to_s3(frames_df, s3_prefix)
    if not file_names:
        raise RuntimeError(
            f"{extraction.drop_id}: no frames uploaded to S3 — aborting Biigle step."
        )

    # 3. Find or create the survey volume; attach our files.
    volume_name = SURVEY_VOLUME_NAME_TEMPLATE.format(survey_id=extraction.survey_id)
    volume_id = find_or_create_volume_and_add_frames(
        volume_name=volume_name,
        s3_frames_prefix=s3_prefix,
        file_names=file_names,
        media_type="image",
    )

    # 4. Push the ML annotations into the volume.
    upload_coco_annotations_to_biigle(volume_id, coco)
    logging.info(
        f"{extraction.drop_id}: training frames in Biigle volume {volume_id} "
        f"({volume_name})"
    )
    return volume_id


# ── Orchestration ────────────────────────────────────────────────────────────


@dataclass
class DropResult:
    drop_id: str
    ok: bool
    volume_id: Optional[int] = None
    n_frames: Optional[int] = None
    stage: str = ""
    error: str = ""


def process_drop(
    drop_id: str, *, force: bool = False, no_upload: bool = False
) -> DropResult:
    """Full lifecycle for one drop: extract → infer → (upload → DB update).

    Returns a `DropResult` whether successful or not — survey runs use this
    so a single failed drop doesn't abort the batch.

    Skips the work entirely if the drop already has a `training_biigle_volume_id`
    set in the DB and `force` is False.

    If `no_upload` is set, stops after inference. Useful for dry-runs that
    confirm extraction + binary detector behave correctly without touching
    Biigle. Re-running with `no_upload=False` will still do the upload step.

    Note: after a `--no-upload` run, the next run on the same drop will hit
    `FileExistsError` from `extract_frames_for_drop` because training_frames/
    is now populated. To proceed with the upload step, pass `force=True` —
    the existing JPGs get overwritten and the upload step runs.
    """
    db = DatabaseManager()

    existing = db.get_training_biigle_volume_id(drop_id)
    if existing is not None and not force:
        logging.info(
            f"{drop_id}: already uploaded to volume {existing}; skipping "
            "(pass --force to redo)."
        )
        return DropResult(drop_id=drop_id, ok=True, volume_id=existing, stage="skipped")

    try:
        extraction = extract_frames_for_drop(drop_id, force=force)
    except Exception as e:
        logging.error(f"{drop_id}: extraction failed — {e}")
        return DropResult(drop_id=drop_id, ok=False, stage="extract", error=str(e))

    try:
        raw_csv = run_inference_to_csv(extraction)
    except Exception as e:
        logging.error(f"{drop_id}: inference failed — {e}")
        return DropResult(
            drop_id=drop_id,
            ok=False,
            n_frames=len(extraction.frame_paths),
            stage="inference",
            error=str(e),
        )

    if no_upload:
        logging.info(
            f"{drop_id}: --no-upload set; "
            f"frames + raw CSV at {extraction.output_dir}; skipping Biigle/DB."
        )
        return DropResult(
            drop_id=drop_id,
            ok=True,
            n_frames=len(extraction.frame_paths),
            stage="extract+infer (no-upload)",
        )

    try:
        volume_id = upload_drop_to_survey_volume(extraction, raw_csv)
    except Exception as e:
        logging.error(f"{drop_id}: Biigle upload failed — {e}")
        return DropResult(
            drop_id=drop_id,
            ok=False,
            n_frames=len(extraction.frame_paths),
            stage="biigle",
            error=str(e),
        )

    db.update_training_biigle_volume_id(drop_id, volume_id)
    logging.info(f"{drop_id}: DB updated — training_biigle_volume_id = {volume_id}")
    return DropResult(
        drop_id=drop_id,
        ok=True,
        volume_id=volume_id,
        n_frames=len(extraction.frame_paths),
        stage="done",
    )


def process_survey(
    survey_id: str, *, force: bool = False, no_upload: bool = False
) -> List[DropResult]:
    """Run `process_drop` over every eligible drop in a survey.

    Eligible = has a present video and a defined sampling window (ignores
    is_bad_deployment, per spec). Continues on per-drop failure; writes a
    `_failures.csv` next to the survey's training_frames area summarising
    what broke.
    """
    db = DatabaseManager()
    drops = db.get_drops_for_survey_with_video_window(survey_id)
    if not drops:
        logging.warning(
            f"{survey_id}: no eligible drops "
            "(must have video_presence='present' AND sampling_start AND sampling_end)."
        )
        return []

    logging.info(f"{survey_id}: processing {len(drops)} drop(s)")
    results: List[DropResult] = []
    for i, drop_id in enumerate(drops, start=1):
        logging.info(f"━━━ [{i}/{len(drops)}] {drop_id} ━━━")
        results.append(process_drop(drop_id, force=force, no_upload=no_upload))

    n_ok = sum(1 for r in results if r.ok)
    failures = [r for r in results if not r.ok]
    logging.info(
        f"{survey_id}: {n_ok}/{len(results)} succeeded " f"({len(failures)} failed)"
    )

    if failures:
        survey_dir = config.deployment_data_dir / survey_id
        survey_dir.mkdir(parents=True, exist_ok=True)
        # Timestamped filename so re-runs preserve the audit trail rather
        # than clobbering the previous failure list.
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        failures_csv = survey_dir / f"training_frames_failures_{run_stamp}.csv"
        with open(failures_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["drop_id", "stage", "error"])
            for r in failures:
                w.writerow([r.drop_id, r.stage, r.error])
        logging.warning(f"{survey_id}: failures logged to {failures_csv}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spyfish.ml.training.extract_training_frames",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--drop-id", help="Process a single drop.")
    target.add_argument(
        "--survey-id",
        help="Process every eligible drop in a survey "
        "(video present + sampling window defined).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract over existing training_frames/ and re-upload "
        "(S3+Biigle dedup absorb anything already there).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Stop after extract + inference. JPGs and raw CSV land locally, "
        "nothing is uploaded to S3 or Biigle and the DB is not updated. "
        "Safe for testing without polluting Biigle.",
    )
    args = parser.parse_args(argv)
    # Logging is already configured by spyfish/log_config.py (imported via
    # spyfish/__init__.py) — no basicConfig call needed here.

    if args.drop_id:
        result = process_drop(args.drop_id, force=args.force, no_upload=args.no_upload)
        return 0 if result.ok else 1

    results = process_survey(args.survey_id, force=args.force, no_upload=args.no_upload)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
