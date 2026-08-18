"""
Standalone CLI: pull N frames per drop directly from S3 (cv2 over presigned URL),
run the configured detector model, and upload to a survey-level Biigle volume
for expert annotation.

Purpose: bootstrap a labeled training set when there's no good species model yet.
Frames are sampled across the deployment's [sampling_start, sampling_end] window,
back-loaded toward the end (where bait-attracted fish density is highest).

Configuration lives under `training_extraction:` in config.yaml; see
`config.training_extraction_n_frames` and `config.training_extraction_annotation_type`.

Frames are selected from the detections `--ml` already produced: ``maxn_per_species``
peak frames per species, topped up to ``n_frames`` with back-loaded timestamps.
Requires the drop to be past `--ml`. Pass ``--test-frames`` to skip that and
sample timestamps blind, running the model on just those frames, no video
download, useful for a quick look or fast training data.

Usage (from project root):
    python -m spyfish.ml.training.extract_training_frames --drop-id    <DROP>
    python -m spyfish.ml.training.extract_training_frames --survey-id  <SURVEY>
    python -m spyfish.ml.training.extract_training_frames --survey-id  <SURVEY> --test-frames
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

import pandas as pd

from spyfish.biigle.upload_frames import (
    find_or_create_volume_and_add_frames,
    upload_coco_annotations_to_biigle,
    upload_frames_to_s3,
)
from spyfish.config.base import MlStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_frames import (
    build_coco_from_raw_csv,
    extract_frames_from_selections,
)
from spyfish.extraction.select_frames import blind_selections, select_frames

SURVEY_VOLUME_NAME_TEMPLATE = "{survey_id}. Training frames"


# ── ML-driven frame selection ────────────────────────────────────────────────


def require_ml_raw_csv(drop_id: str) -> Path:
    """The drop's full-video detection CSV, or a clear error pointing at `--ml`.

    Video download, inference, MaxN and status all belong to the `--ml` stage;
    this module only consumes what that stage produced. An earlier version
    re-ran inference here, which duplicated `MLRunner` while skipping its
    `process_maxn` and status advance, so a drop could end up holding
    detections but no MaxN and no `ml_complete`.
    """
    model_path = config.get_pipeline_model(config.training_extraction_annotation_type)
    raw_csv = config.get_raw_csv_path(drop_id, model_path.stem)
    if raw_csv.exists():
        return raw_csv

    status = (DatabaseManager().get_deployment(drop_id) or {}).get(MlStatus.COLUMN)
    raise FileNotFoundError(
        f"{drop_id}: no detections at {raw_csv.name} (ml_status={status!r}). "
        "Run `python run_pipeline.py --ml` for this drop first, or pass "
        "--test-frames to sample frames without ML."
    )


# ── Extraction ───────────────────────────────────────────────────────────────


def _per_frame_csv_name(drop_id: str, annotation_type: Optional[str] = None) -> str:
    """Filename for the detections run over a drop's extracted frames.

    Carries the weights filename rather than just "binary"/"species" so that
    promoting a new model invalidates the old CSV by name, matching how
    `config.get_raw_csv_path` names the full-video one. Without the version a
    re-run would silently reuse the previous model's boxes.
    """
    kind = annotation_type or config.training_extraction_annotation_type
    return f"{drop_id}_{config.get_pipeline_model(kind).stem}_raw.csv"


# ── Inference ────────────────────────────────────────────────────────────────


def load_inference_model(annotation_type: Optional[str] = None):
    """Load the configured pipeline model into memory.

    Centralised here so survey runs can load once at the top of the loop and
    pass the loaded instance through `process_drop → run_inference_to_csv`,
    avoiding ~1-3s of per-drop reload (weights + GPU init) overhead.

    Single-drop CLI runs let `run_inference_to_csv` lazy-load instead.
    """
    from ultralytics import YOLO

    kind = annotation_type or config.training_extraction_annotation_type
    model_path = config.get_pipeline_model(kind)
    logging.info(f"loading {kind} pipeline model: {model_path.name}")
    return YOLO(str(model_path))


def _frame_records(drop_id: str, paths: list, times: list) -> list:
    """COCO image records for already-extracted frames.

    ``file_name`` must match the name the frame is registered under in Biigle,
    because `upload_coco_annotations_to_biigle` joins the two on that string to
    resolve image IDs. For survey volumes that is the path relative to the
    survey dir (``{drop}/frames/…``), which is exactly what
    `upload_frames_to_s3(..., relative_to=...)` returns.
    """
    import cv2 as _cv2

    survey_root = config.deployment_data_dir / config.get_survey_id_from_drop(drop_id)
    sample = _cv2.imread(str(paths[0]))
    h, w = sample.shape[:2] if sample is not None else (0, 0)

    def _name(p) -> str:
        try:
            return Path(p).resolve().relative_to(survey_root.resolve()).as_posix()
        except ValueError:
            return Path(p).name

    return [
        {
            "image_id": i,
            "file_name": _name(p),
            "time_of_max": float(t),
            "drop_id": drop_id,
            "img_w": int(w),
            "img_h": int(h),
        }
        for i, (p, t) in enumerate(zip(paths, times), start=1)
    ]


def run_inference_on_paths(drop_id: str, paths: list, times: list, model=None) -> Path:
    """Run the configured model over already-extracted frames.

    Only the --test-frames path needs this: with no full-video detections, the
    frames themselves are the only thing to run the model over.
    """
    from spyfish.ml.run_inference import predict_on_frame_paths

    kind = config.training_extraction_annotation_type
    if model is None:
        model = load_inference_model(annotation_type=kind)
    out_csv = Path(paths[0]).parent / _per_frame_csv_name(drop_id, kind)
    logging.info(f"{drop_id}: running '{kind}' inference over {len(paths)} frame(s)")
    return predict_on_frame_paths(
        frame_paths=[Path(p) for p in paths],
        timestamps=times,
        output_csv=out_csv,
        model=model,
        fps=None,
    )


# ── Selections ───────────────────────────────────────────────────────────────


def write_blind_selections(
    drop_id: str, output_path: Path, n_frames: Optional[int] = None
) -> pd.DataFrame:
    """Selections CSV built without consulting the model, the --test-frames path.

    Produces the same CSV `select_frames` does, so everything downstream is
    shared; only where the timestamps came from differs.
    """
    deployment = DatabaseManager().get_deployment(drop_id)
    if deployment is None:
        raise ValueError(f"{drop_id}: not found in deployments DB")
    start = deployment.get("sampling_start")
    end = deployment.get("sampling_end")
    if start is None or end is None:
        raise ValueError(
            f"{drop_id}: missing sampling window (start={start}, end={end})"
        )

    n = n_frames or config.training_extraction_n_frames
    df = blind_selections(
        drop_id=drop_id,
        sampling_start=float(start),
        sampling_end=float(end),
        taken_times=pd.Series(dtype=float),
        spacing=config.frame_strategy["temporal_spacing_seconds"],
        n=n,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"{drop_id}: {len(df)} blind selection(s) → {output_path.name}")
    return df


def upsert_selections(
    prior: Optional[pd.DataFrame], fresh: pd.DataFrame
) -> pd.DataFrame:
    """Merge `fresh` into the prior pass's selections, keyed on timestamp.

    Biigle volumes APPEND rather than replace, so a second pass over a drop adds
    frames to the volume. A selections CSV that got clobbered would then no
    longer describe what the volume holds. Upserting keeps it a true record of
    every frame ever sent for this drop, with `SelectionReason` distinguishing
    which pass each came from.

    `prior` must be captured BEFORE the selection step runs: both selection
    functions write `fresh` to the selections CSV themselves, so reading the
    file here would always see `fresh` and the merge would keep nothing.
    """
    if prior is None or prior.empty:
        return fresh
    key = config.csv_clip_max_time_column
    kept = prior[~prior[key].round(3).isin(fresh[key].round(3))]
    merged = pd.concat([kept, fresh], ignore_index=True).sort_values(key)
    if len(kept):
        logging.info(
            f"Selections upsert: kept {len(kept)} selection(s) from an earlier "
            f"pass, added {len(fresh)}."
        )
    return merged.reset_index(drop=True)


# ── Biigle upload ────────────────────────────────────────────────────────────


def upload_to_survey_volume(drop_id: str, frame_paths: list, coco: dict) -> int:
    """Upload a drop's frames + COCO to its SURVEY-level Biigle volume.

    The only part of this module that is genuinely training-specific: a
    survey-pooled volume and prefix rather than the per-drop ones the expert
    path uses. Everything before it is shared.
    """
    survey_id = config.get_survey_id_from_drop(drop_id)
    s3_prefix = config.get_training_frames_s3_prefix(survey_id)
    frames_df = pd.DataFrame({"FramePath": [str(p) for p in frame_paths]})
    # Names are relative to the survey dir, so each carries its
    # {drop}/frames/ segment and S3 mirrors the local layout.
    file_names = upload_frames_to_s3(
        frames_df, s3_prefix, relative_to=config.deployment_data_dir / survey_id
    )
    if not file_names:
        raise RuntimeError(f"{drop_id}: no frames uploaded to S3, aborting Biigle.")

    volume_name = SURVEY_VOLUME_NAME_TEMPLATE.format(survey_id=survey_id)
    volume_id, filename_to_id = find_or_create_volume_and_add_frames(
        volume_name=volume_name,
        s3_frames_prefix=s3_prefix,
        file_names=file_names,
        media_type="image",
    )
    upload_coco_annotations_to_biigle(
        volume_id, coco, filename_to_biigle_id=filename_to_id
    )
    logging.info(f"{drop_id}: frames in Biigle volume {volume_id} ({volume_name})")
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
    drop_id: str,
    *,
    force: bool = False,
    no_upload: bool = False,
    model=None,
    test_frames: bool = False,
) -> DropResult:
    """One drop: select -> extract -> COCO -> upload to the survey volume.

    Both modes share every stage but the first. The default reads the detections
    `--ml` produced and runs the shared `select_frames`; ``test_frames`` picks
    timestamps blind and runs the model over just those frames afterwards, which
    needs no video download and no `--ml` prerequisite.
    """
    db = DatabaseManager()
    existing = db.get_training_biigle_volume_id(drop_id)
    if existing is not None and not force:
        logging.info(
            f"{drop_id}: already uploaded to volume {existing}; skipping "
            "(pass --force to redo)."
        )
        return DropResult(drop_id=drop_id, ok=True, volume_id=existing, stage="skipped")

    selections_path = config.get_biigle_selections_csv_path(drop_id)
    # Capture the prior pass now, the selection calls below overwrite the CSV.
    prior = pd.read_csv(selections_path) if selections_path.exists() else None
    raw_csv: Optional[Path] = None
    try:
        if test_frames:
            fresh = write_blind_selections(drop_id, selections_path)
        else:
            raw_csv = require_ml_raw_csv(drop_id)
            fresh = select_frames(str(raw_csv), str(selections_path), drop_id)
        merged = upsert_selections(prior, fresh)
        merged.to_csv(selections_path, index=False)
    except Exception as e:
        logging.error(f"{drop_id}: selection failed, {e}")
        return DropResult(drop_id=drop_id, ok=False, stage="select", error=str(e))

    try:
        frames_df = extract_frames_from_selections(
            str(selections_path),
            str(config.get_video_path(drop_id)),
            str(raw_csv) if raw_csv else "",
            write_coco=not test_frames,
        )
        # Filter failed extractions row-wise so FramePath and its timestamp
        # stay paired, dropping paths alone would shift every later frame's
        # time by one slot when an extraction fails mid-list.
        ok_rows = frames_df[frames_df["FramePath"].fillna("").astype(str) != ""]
        paths = ok_rows["FramePath"].tolist()
        if not paths:
            raise RuntimeError("no frames extracted")
    except Exception as e:
        logging.error(f"{drop_id}: extraction failed, {e}")
        return DropResult(drop_id=drop_id, ok=False, stage="extract", error=str(e))

    coco_path = config.get_coco_annotations_path(drop_id)
    if test_frames:
        # No full-video detections exist, so the model runs over the extracted
        # frames and the COCO is built from that.
        try:
            times = [float(t) for t in ok_rows[config.csv_clip_max_time_column]]
            frame_csv = run_inference_on_paths(drop_id, paths, times, model=model)
            coco = build_coco_from_raw_csv(
                str(frame_csv), _frame_records(drop_id, paths, times)
            )
            coco_path.parent.mkdir(parents=True, exist_ok=True)
            coco_path.write_text(json.dumps(coco, indent=2))
        except Exception as e:
            logging.error(f"{drop_id}: inference on frames failed, {e}")
            return DropResult(
                drop_id=drop_id, ok=False, stage="inference", error=str(e)
            )
    else:
        coco = json.loads(coco_path.read_text())

    if no_upload:
        logging.info(f"{drop_id}: --no-upload set; artefacts left on disk.")
        return DropResult(
            drop_id=drop_id, ok=True, n_frames=len(paths), stage="no-upload"
        )

    try:
        volume_id = upload_to_survey_volume(drop_id, paths, coco)
    except Exception as e:
        logging.error(f"{drop_id}: Biigle upload failed, {e}")
        return DropResult(
            drop_id=drop_id, ok=False, n_frames=len(paths), stage="biigle", error=str(e)
        )

    db.update_training_biigle_volume_id(drop_id, volume_id)
    return DropResult(
        drop_id=drop_id,
        ok=True,
        volume_id=volume_id,
        n_frames=len(paths),
        stage="done",
    )


def process_survey(
    survey_id: str,
    *,
    force: bool = False,
    no_upload: bool = False,
    test_frames: bool = False,
) -> List[DropResult]:
    """Run `process_drop` over every eligible drop in a survey.

    Eligible = has a present video and a defined sampling window (ignores
    is_bad_deployment, per spec). Continues on per-drop failure; writes a
    `_failures.csv` in the survey directory summarising
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

    # Load the pipeline model once for the whole survey, re-loading per drop
    # costs 1-3s of disk + GPU init each, which compounds badly across
    # hundreds of drops. process_drop accepts model=None when called from the
    # single-drop CLI path; here we pre-load and pass it through.
    model = load_inference_model()

    logging.info(f"{survey_id}: processing {len(drops)} drop(s)")
    results: List[DropResult] = []
    for i, drop_id in enumerate(drops, start=1):
        logging.info(f"━━━ [{i}/{len(drops)}] {drop_id} ━━━")
        results.append(
            process_drop(
                drop_id,
                force=force,
                no_upload=no_upload,
                model=model,
                test_frames=test_frames,
            )
        )

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
        help="Re-extract over existing frames/ and re-upload "
        "(S3+Biigle dedup absorb anything already there).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Stop after extract + inference. JPGs and raw CSV land locally, "
        "nothing is uploaded to S3 or Biigle and the DB is not updated. "
        "Safe for testing without polluting Biigle.",
    )
    parser.add_argument(
        "--test-frames",
        action="store_true",
        help="Skip ML entirely: sample timestamps blind and run the model on "
        "just those frames. No video download and no `--ml` prerequisite, so "
        "it is the fast way to eyeball a deployment or produce training data "
        "before a model exists. Without this flag, frames are chosen from the "
        "detections `--ml` produced, and the drop must have been through it.",
    )
    args = parser.parse_args(argv)
    # Logging is already configured by spyfish/log_config.py (imported via
    # spyfish/__init__.py), no basicConfig call needed here.

    if args.drop_id:
        result = process_drop(
            args.drop_id,
            force=args.force,
            no_upload=args.no_upload,
            test_frames=args.test_frames,
        )
        return 0 if result.ok else 1

    results = process_survey(
        args.survey_id,
        force=args.force,
        no_upload=args.no_upload,
        test_frames=args.test_frames,
    )
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
