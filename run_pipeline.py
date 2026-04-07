"""
Spyfish Aotearoa — Single-command pipeline runner.

Usage:
    python run_pipeline.py                   # Run all steps (default)
    python run_pipeline.py --ingest          # Only run Step 1 (metadata ingestion)
    python run_pipeline.py --ml              # Only run Steps 2+3 (ML inference + post-processing)
    python run_pipeline.py --zooniverse-clips# Only run Step 4 (Zooniverse clip extraction)
    python run_pipeline.py --zooniverse-images# Only run Step 5 (Zooniverse image extraction)
    python run_pipeline.py --zooniverse-sync # Only run Step 5b (Zooniverse volunteer sync-back)
    python run_pipeline.py --biigle-upload   # Only run Step 6 (Biigle frame extraction + upload)
    python run_pipeline.py --biigle-sync     # Only run Step 7 (Biigle annotation sync)
    python run_pipeline.py --retrain         # Only run Step 8 (model retraining)
    python run_pipeline.py --test-run        # Run in test mode with mock data

Steps can be combined: python run_pipeline.py --ingest --biigle-sync
If no step flags are given, ALL steps run.

Adding a new pipeline stage
----------------------------
1. Write the step function below (GlobalStage: () -> None, DropStage: (drop_id) -> str).
2. Add one entry to STAGES.
Argparse, eligibility, status transitions, and logging are automatic.
"""

import argparse
import functools
import logging
import sys
from dataclasses import replace
from pathlib import Path

from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.biigle.upload_frames import upload_frames_to_biigle
from spyfish.config.base import PipelineStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.extraction.select_frames import select_frames
from spyfish.log_config import log_header
from spyfish.ml.process_ml_annotations import run_post_ml
from spyfish.orchestrator.ingest import check_pending_arrivals, run_ingestion
from spyfish.orchestrator.ingest_legacy import ingest_legacy_expert_annotations
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.orchestrator.retrain_runner import run_retraining
from spyfish.orchestrator.stage import DropStage, GlobalStage, StageRunner
from spyfish.storage.db_sync import sync_pipeline_results
from spyfish.zooniverse.select_zooniverse_clips import process_zooniverse_clips
from spyfish.zooniverse.upload import (
    upload_clips_to_zooniverse,
    upload_frames_to_zooniverse,
)

# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------


def _get_common_paths(drop_id: str) -> dict:
    """Returns standardized paths for a drop."""
    model_name = Path(config.pipeline_model_path).stem
    return {
        "model_name": model_name,
        "maxn_csv": str(config.get_maxn_csv_path(drop_id, model_name)),
        "selections_csv": str(config.get_selections_csv_path(drop_id)),
        "raw_csv": str(config.get_raw_csv_path(drop_id, model_name)),
        "video_path": str(config.get_video_path(drop_id)),
    }


# ---------------------------------------------------------------------------
# Global stage functions — run once, manage their own iteration internally
# ---------------------------------------------------------------------------


def _run_step1_ingest() -> None:
    run_ingestion()
    ingest_legacy_expert_annotations()


def _run_arrival_check() -> None:
    check_pending_arrivals()


def _run_set_targets(push_s3: bool = True) -> None:
    from spyfish.test_setup import process_csv_targets

    csv_path = config.pipeline_targets_csv
    if not csv_path:
        logging.error(
            "--set-targets requires paths.pipeline_targets_csv to be set in config.yaml."
        )
        sys.exit(1)
    process_csv_targets(csv_path, push_s3=push_s3)


def _run_steps2_and_3_ml() -> None:
    runner = MLRunner()
    targets = runner.get_inference_targets()

    if not targets:
        logging.info("No drops available for ML processing.")
        return

    all_drop_ids = [t["drop_id"] for t in targets]
    results = runner.run_inference_loop(targets)
    runner.finalize_batch_results(results, all_drop_ids=all_drop_ids)

    if results:
        run_post_ml(
            drop_ids=results,
            video_dir=str(config.media_dir),
        )


def _step_zooniverse_sync_drop(drop_id: str) -> str | None:
    """Zooniverse volunteer annotation sync-back.

    Checks whether volunteer classification is complete for the subject set
    associated with this drop, then downloads and stores results.

    TODO: Implement Zooniverse API check via panoptes_client:
      - Query subject set classification counts for the drop's subject set.
      - Define "done" threshold (e.g. minimum N classifications per subject,
        or Caesar reduction pipeline completion, or manual sign-off flag).
      - On completion: download classification export, parse volunteer
        annotations, store them for downstream use.
      - Until done: return None so the runner leaves the drop at
        AWAITING_CITSCI_FRAMES and tries again on the next pipeline run.

    Returns None (not ready) until the Zooniverse API check is implemented.
    """
    logging.info(
        f"zooniverse-sync: Zooniverse API check not yet implemented for {drop_id}. "
        "Leaving at AWAITING_CITSCI_FRAMES until volunteer annotations are confirmed complete."
    )
    return None


def _run_step7_biigle_sync() -> None:
    sync_biigle_annotations()


def _run_step8_retrain() -> None:
    run_retraining(auto_promote=True)


# ---------------------------------------------------------------------------
# Per-drop step functions — (drop_id: str) -> target PipelineStatus str
# The runner handles the loop, db.advance_status(), and error propagation.
# ---------------------------------------------------------------------------


def _step4_process_drop(drop_id: str) -> str:
    """Step 4: Zooniverse clip selection + extraction."""
    paths = _get_common_paths(drop_id)

    try:
        selections_df = process_zooniverse_clips(
            paths["maxn_csv"], paths["selections_csv"], drop_id
        )
    except FileNotFoundError as e:
        logging.error(f"MaxN CSV missing for {drop_id}, cannot select clips: {e}")
        return None

    if selections_df.empty:
        logging.error(
            f"No clips selected for {drop_id} — sampling window may be too short for clip length."
        )
        return None

    clips_df = extract_clips_from_selections(
        selections_csv_path=paths["selections_csv"],
        video_path=paths["video_path"],
    )
    logging.info(f"Uploading {len(clips_df)} clips for {drop_id} to Zooniverse.")
    upload_clips_to_zooniverse(clips_df)
    return PipelineStatus.CITSCI_CLIPS_COMPLETE


def _step5_process_drop(drop_id: str) -> str:
    """Step 5: Zooniverse frame extraction + upload."""
    paths = _get_common_paths(drop_id)

    if not Path(paths["selections_csv"]).exists():
        logging.error(
            f"Missing selections CSV for {drop_id} — step 4 should have written it."
        )
        return None

    frames_df = extract_frames_from_selections(
        selections_csv_path=paths["selections_csv"],
        video_path=paths["video_path"],
        raw_csv_path=paths["raw_csv"],
    )
    logging.info(f"Uploading {len(frames_df)} frames for {drop_id} to Zooniverse.")
    upload_frames_to_zooniverse(frames_df)
    return PipelineStatus.AWAITING_CITSCI_FRAMES


def _step6_process_drop(drop_id: str) -> str:
    """Step 6: Biigle frame extraction + volume upload."""
    paths = _get_common_paths(drop_id)
    # Biigle uses its own selections CSV (multiplier applied) separate from the
    # Zooniverse one so step 4 and step 6 don't overwrite each other's output.
    biigle_selections_path = config.get_biigle_selections_csv_path(drop_id)

    try:
        select_frames(paths["raw_csv"], str(biigle_selections_path), drop_id)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Biigle frame selection failed for {drop_id}: {e}")
        return None

    frames_df = extract_frames_from_selections(
        selections_csv_path=str(biigle_selections_path),
        video_path=paths["video_path"],
        raw_csv_path=paths["raw_csv"],
    )
    volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
    if volume_info is None:
        return None
    logging.info(f"Biigle volume created for {drop_id}: id={volume_info.get('id')}")
    return PipelineStatus.AWAITING_EXPERT_REVIEW


# ---------------------------------------------------------------------------
# Dynamic input statuses — Biigle-direct path also picks up ML_COMPLETE
# when Zooniverse steps are not running.
# ---------------------------------------------------------------------------


def _biigle_input_statuses(args: argparse.Namespace, run_all: bool) -> list[str]:
    """Returns the statuses Step 6 should query based on which stages are active.

    Zooniverse path:   only CITSCI_COMPLETE (drops have passed through Zooniverse)
    Biigle-direct:     CITSCI_COMPLETE + ML_COMPLETE (skip Zooniverse entirely)
    """
    statuses = [PipelineStatus.CITSCI_COMPLETE]
    skip_zooniverse = not (
        run_all
        or getattr(args, "zooniverse_clips", False)
        or getattr(args, "zooniverse_images", False)
        or getattr(args, "zooniverse_sync", False)
    )
    if skip_zooniverse:
        statuses.append(PipelineStatus.ML_COMPLETE)
    return statuses


# ---------------------------------------------------------------------------
# Stage registry — add a new pipeline stage by adding ONE entry here
# ---------------------------------------------------------------------------

STAGES: list = [
    GlobalStage("ingest", "Step 1: metadata ingestion", _run_step1_ingest),
    GlobalStage(
        "check-arrivals",
        "Check S3 for video arrivals",
        _run_arrival_check,
        run_in_all=False,
    ),
    GlobalStage(
        "set-targets",
        "Bulk set pipeline stages from CSV",
        _run_set_targets,
        run_in_all=False,
    ),
    GlobalStage(
        "ml", "Steps 2+3: ML inference + post-processing", _run_steps2_and_3_ml
    ),
    DropStage(
        "zooniverse-clips",
        "Step 4: Zooniverse clip extraction",
        _step4_process_drop,
        [PipelineStatus.ML_COMPLETE, PipelineStatus.AWAITING_CITSCI_CLIPS],
        queue_status=PipelineStatus.AWAITING_CITSCI_CLIPS,
    ),
    DropStage(
        "zooniverse-images",
        "Step 5: Zooniverse image extraction",
        _step5_process_drop,
        [PipelineStatus.CITSCI_CLIPS_COMPLETE],
    ),
    DropStage(
        "zooniverse-sync",
        "Step 5b: Zooniverse volunteer sync-back",
        _step_zooniverse_sync_drop,
        [PipelineStatus.AWAITING_CITSCI_FRAMES],
    ),
    DropStage(
        "biigle-upload",
        "Step 6: Biigle frame extraction + upload",
        _step6_process_drop,
        _biigle_input_statuses,
    ),
    GlobalStage(
        "biigle-sync", "Step 7: Biigle annotation sync", _run_step7_biigle_sync
    ),
    GlobalStage(
        "retrain",
        "Step 8: Retraining pipeline (run --biigle-sync first)",
        _run_step8_retrain,
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    db = DatabaseManager()
    runner = StageRunner(STAGES, db)

    parser = runner.build_parser()
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip all S3 uploads (DB, models, results)",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Connectivity check: print config summary and exit without running the pipeline",
    )
    args = parser.parse_args()

    logging.info("═" * 60)
    logging.info(" SPYFISH AOTEAROA PIPELINE ".center(60, "═"))
    logging.info(f" NO-UPLOAD: {args.no_upload} ".center(60, "═"))
    logging.info("═" * 60)

    if args.ping:
        log_header("PING: CONFIG CHECK")
        logging.info(f"S3 bucket: {config.s3_bucket}")
        logging.info(f"Base dir:  {config.base_dir}")
        logging.info(f"Test run:  {config.is_test_run}")
        return

    # Bind no_upload to set-targets (only stage whose behaviour depends on it)
    patched_stages = [
        (
            replace(
                s, fn=functools.partial(_run_set_targets, push_s3=not args.no_upload)
            )
            if s.flag == "set-targets"
            else s
        )
        for s in STAGES
    ]
    runner = StageRunner(patched_stages, db)
    runner.run(args)

    # Push final state (DBs + ML CSVs) to S3
    if args.no_upload:
        logging.info("No-upload set: skipping final S3 sync.")
        log_header("PIPELINE COMPLETE (LOCAL ONLY)", character="═")
    elif config.is_test_run:
        logging.debug("Test run: skipping final S3 sync of annotations directory.")
        log_header("PIPELINE COMPLETE (TEST RUN)", character="═")
    else:
        logging.info("Syncing final results to S3...")
        if sync_pipeline_results():
            log_header("PIPELINE SUCCESS (SYNCED TO S3)", character="═")
        else:
            logging.critical(
                "CRITICAL: S3 Sync failed. Pipeline state might be inconsistent on S3."
            )
            log_header("PIPELINE FAILED (SYNC ERROR)", character="═")
            sys.exit(1)


if __name__ == "__main__":
    main()
