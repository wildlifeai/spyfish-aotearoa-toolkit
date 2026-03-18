"""
Spyfish Aotearoa — Single-command pipeline runner.

Usage:
    python run_pipeline.py                   # Run all steps (default)
    python run_pipeline.py --ingest          # Only run Step 1 (metadata ingestion)
    python run_pipeline.py --ml              # Only run Steps 2+3 (ML inference + post-processing)
    python run_pipeline.py --zooniverse-clips# Only run Step 4 (Zooniverse clip extraction)
    python run_pipeline.py --zooniverse-images# Only run Step 5 (Zooniverse image extraction)
    python run_pipeline.py --biigle-upload   # Only run Step 6 (Biigle frame extraction + upload)
    python run_pipeline.py --biigle-sync     # Only run Step 7 (Biigle annotation sync)
    python run_pipeline.py --biigle-test     # Seed DB with test Biigle drops + run Step 7
    python run_pipeline.py --test-run        # Run in test mode with mock data

Steps can be combined: python run_pipeline.py --ingest --biigle-sync
If no step flags are given, ALL steps run.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.biigle.upload_frames import upload_frames_to_biigle
from spyfish.config.base import PipelineStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.log_config import log_header
from spyfish.ml.process_ml_annotations import run_post_ml
from spyfish.orchestrator.ingest import check_pending_arrivals, run_ingestion
from spyfish.orchestrator.ingest_legacy import ingest_legacy_expert_annotations
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.orchestrator.retrain_runner import run_retraining
from spyfish.storage.db_sync import sync_pipeline_results
from spyfish.zooniverse.select_zooniverse_clips import process_zooniverse_clips
from spyfish.zooniverse.upload import (
    check_clip_sizes,
    upload_clips_to_zooniverse,
    upload_frames_to_zooniverse,
)


def _run_step0_test_run():
    log_header("STEP 0: TEST RUN, THIS IS IT")
    logging.info(f"bucket name: {config.s3_bucket}")


def _run_step1_ingest():
    log_header("STEP 1: METADATA INGESTION: Ingesting metadata into pipeline database")
    run_ingestion()
    ingest_legacy_expert_annotations()


def _run_arrival_check():
    log_header("ORCHESTRATION: CHECKING FOR NEW VIDEO ARRIVALS")
    check_pending_arrivals()


def _run_step2_ml_inference() -> list:
    log_header("STEP 2: ML INFERENCE")
    runner = MLRunner()
    targets = runner.get_inference_targets()

    if not targets:
        logging.info("No drops available for ML processing.")
        return []

    results = runner.run_inference_loop(targets)
    runner.finalize_batch_results(results)
    return results


def _run_step3_post_ml(drop_ids: list):
    log_header("STEP 3: POST-ML Processing (MaxN + QA frames)")
    run_post_ml(
        drop_ids=drop_ids,
        annotations_dir=str(config.data_quality_dir),
        video_dir=str(config.media_dir),
        output_root=str(config.data_quality_dir),
    )


def _get_common_paths(drop_id: str):
    """Helper to get standardized paths for a drop."""
    model_name = Path(config.pipeline_model_path).stem
    return {
        "model_name": model_name,
        "maxn_csv": str(config.get_maxn_csv_path(drop_id, model_name)),
        "selections_csv": str(config.get_selections_csv_path(drop_id)),
        "raw_csv": str(config.get_raw_csv_path(drop_id, model_name)),
        "video_path": str(config.get_video_path(drop_id)),
        "zooniverse_clips": str(config.get_clips_dir(drop_id, target="zooniverse")),
        "zooniverse_frames": str(config.get_frames_dir(drop_id, target="zooniverse")),
        "biigle_frames": str(config.get_frames_dir(drop_id, target="biigle")),
    }


def _run_step4_zooniverse_clips(db: DatabaseManager):
    log_header("STEP 4: Zooniverse clip selection + extraction")

    records = db.get_deployments_by_status(PipelineStatus.AWAITING_CITSCI_CLIPS)
    drop_ids = [r["drop_id"] for r in records]

    if not drop_ids:
        logging.info(
            f"No deployments found with status {PipelineStatus.AWAITING_CITSCI_CLIPS}. Skipping Step 4."
        )
        return

    for drop_id in drop_ids:
        logging.info(f"Processing clips for {drop_id}...")
        paths = _get_common_paths(drop_id)

        selections_df = process_zooniverse_clips(
            paths["maxn_csv"], paths["selections_csv"], drop_id, config
        )
        if selections_df is None or selections_df.empty:
            logging.info(
                f"No high-confidence clips found for {drop_id}. Advancing to CITSCI_CLIPS_COMPLETE."
            )
            db.update_status(drop_id, PipelineStatus.CITSCI_CLIPS_COMPLETE)
            continue

        clips_df = extract_clips_from_selections(
            selections_csv_path=paths["selections_csv"],
            video_path=paths["video_path"],
            output_dir=paths["zooniverse_clips"],
        )
        clips_df = check_clip_sizes(clips_df)

        logging.info(f"Uploading {len(clips_df)} clips for {drop_id} to Zooniverse.")
        upload_clips_to_zooniverse(clips_df)
        db.update_status(drop_id, PipelineStatus.CITSCI_CLIPS_COMPLETE)


def _run_step5_zooniverse_images(db: DatabaseManager):
    log_header("STEP 5: Zooniverse frame extraction + upload")

    records = db.get_deployments_by_status(PipelineStatus.CITSCI_CLIPS_COMPLETE)
    drop_ids = [r["drop_id"] for r in records]

    if not drop_ids:
        logging.info("No deployments ready for Zooniverse images. Skipping Step 5.")
        return

    for drop_id in drop_ids:
        logging.info(f"Processing frames for {drop_id}...")
        paths = _get_common_paths(drop_id)

        if not Path(paths["selections_csv"]).exists():
            logging.warning(
                f"Missing {paths['selections_csv']} for {drop_id}. Advancing to CITSCI_COMPLETE."
            )
            db.update_status(drop_id, PipelineStatus.CITSCI_COMPLETE)
            continue

        frames_df = extract_frames_from_selections(
            selections_csv_path=paths["selections_csv"],
            video_path=paths["video_path"],
            raw_csv_path=paths["raw_csv"],
            output_dir=paths["zooniverse_frames"],
        )
        logging.info(f"Uploading {len(frames_df)} frames for {drop_id} to Zooniverse.")
        upload_frames_to_zooniverse(frames_df)
        db.update_status(drop_id, PipelineStatus.CITSCI_COMPLETE)


def _run_step6_biigle_upload(db: DatabaseManager, include_ml_complete: bool = False):
    log_header("─── STEP 6: Biigle frame extraction + volume upload ───")

    statuses = [PipelineStatus.CITSCI_COMPLETE]
    if include_ml_complete:
        statuses.append(PipelineStatus.ML_COMPLETE)
        logging.info(
            f"Including {PipelineStatus.ML_COMPLETE} deployments as requested."
        )

    records = db.get_deployments_by_statuses(statuses)
    drop_ids = [r["drop_id"] for r in records]

    if not drop_ids:
        logging.info("No deployments ready for Biigle upload. Skipping Step 6.")
        return

    for drop_id in drop_ids:
        logging.info(f"Uploading volume for {drop_id} to Biigle...")
        paths = _get_common_paths(drop_id)

        if not Path(paths["selections_csv"]).exists():
            logging.error(
                f"Missing {paths['selections_csv']} for {drop_id}. Advancing to AWAITING_EXPERT_REVIEW (skipped upload)."
            )
            db.update_status(drop_id, PipelineStatus.AWAITING_EXPERT_REVIEW)
            continue

        # Extract clean JPEGs + COCO JSON
        frames_df = extract_frames_from_selections(
            selections_csv_path=paths["selections_csv"],
            video_path=paths["video_path"],
            raw_csv_path=paths["raw_csv"],
            output_dir=paths["biigle_frames"],
        )

        # Upload to S3 + create Biigle volume
        volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
        logging.info(f"Biigle volume created for {drop_id}: id={volume_info.get('id')}")

        db.update_status(drop_id, PipelineStatus.AWAITING_EXPERT_REVIEW)


def _run_step7_biigle_sync():
    log_header("STEP 7:  Sync Biigle annotations ")
    sync_biigle_annotations()


def execute_step(step_func, *args, **kwargs):
    """Wrapper to run a step and handle exceptions cleanly."""
    try:
        return step_func(*args, **kwargs)
    except Exception as e:
        step_name = step_func.__name__.replace("_run_", "").replace("_", " ").upper()
        logging.error(f"{step_name} FAILED: {e}")
        logging.error(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run the Spyfish pipeline. Runs all steps by default."
    )
    parser.add_argument("--step0", action="store_true", help="Run Step 0: test run")
    parser.add_argument(
        "--ingest", action="store_true", help="Run Step 1: metadata ingestion"
    )
    parser.add_argument(
        "--check-arrivals",
        action="store_true",
        help="Check S3 for video arrivals (advances PENDING_ARRIVAL)",
    )
    parser.add_argument(
        "--set-targets",
        action="store_true",
        help="Bulk set Pipeline Stages from the CSV at paths.pipeline_targets_csv in config.yaml.",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Run Steps 2+3: ML inference + post-processing",
    )
    parser.add_argument(
        "--zooniverse-clips",
        action="store_true",
        help="Run Step 4: Zooniverse clip extraction",
    )
    parser.add_argument(
        "--zooniverse-images",
        action="store_true",
        help="Run Step 5: Zooniverse image extraction",
    )
    parser.add_argument(
        "--biigle-upload",
        action="store_true",
        help="Run Step 6: Biigle frame extraction + upload",
    )
    parser.add_argument(
        "--biigle-sync", action="store_true", help="Run Step 7: Biigle annotation sync"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Step 8: Run the full retraining pipeline",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip all S3 uploads (DB, models, results)",
    )
    parser.add_argument(
        "--upload-videos",
        action="store_true",
        help="Upload videos to S3 during final sync",
    )
    parser.add_argument(
        "--test-run", action="store_true", help="Run in test mode with mock data"
    )
    args = parser.parse_args()

    # If no step flags are given, run everything
    run_all = not any(
        [
            args.step0,
            args.ingest,
            args.check_arrivals,
            args.set_targets,
            args.ml,
            args.zooniverse_clips,
            args.zooniverse_images,
            args.biigle_upload,
            args.biigle_sync,
            args.retrain,
        ]
    )

    active_steps = ", ".join(
        s
        for s, v in [
            ("step0", args.step0),
            ("set-targets", args.set_targets),
            ("ingest", run_all or args.ingest),
            ("ml", run_all or args.ml),
            ("zooniverse-clips", run_all or args.zooniverse_clips),
            ("zooniverse-images", run_all or args.zooniverse_images),
            ("biigle-upload", run_all or args.biigle_upload),
            ("biigle-sync", run_all or args.biigle_sync),
            ("retrain", run_all or args.retrain),
        ]
        if v
    )
    logging.info(f"active steps: {active_steps}")

    logging.info("═" * 60)
    logging.info(" SPYFISH AOTEAROA PIPELINE ".center(60, "═"))
    logging.info(f" STEPS: {active_steps} ".center(60, "═"))
    logging.info(f" NO-UPLOAD: {args.no_upload} ".center(60, "═"))
    logging.info("═" * 60)

    results = []

    if args.step0:
        execute_step(_run_step0_test_run)
        logging.info("Step 0 completed.")
        return

    if args.set_targets:
        log_header("ORCHESTRATION: SETTING PIPELINE TARGETS FROM CSV")
        from spyfish.test_setup import process_csv_targets

        csv_path = config.pipeline_targets_csv
        if not csv_path:
            logging.error(
                "--set-targets requires paths.pipeline_targets_csv to be set in config.yaml."
            )
            sys.exit(1)
        execute_step(process_csv_targets, csv_path, push_s3=not args.no_upload)

    if run_all or args.ingest:
        execute_step(_run_step1_ingest)

    if args.check_arrivals:
        execute_step(_run_arrival_check)

    db = DatabaseManager()

    if run_all or args.ml:
        results = execute_step(_run_step2_ml_inference)
        if results:
            execute_step(_run_step3_post_ml, results)

    if run_all or args.zooniverse_clips:
        execute_step(_run_step4_zooniverse_clips, db)
    else:
        logging.info("─── STEP 4: SKIPPED (--zooniverse-clips not set) ───")

    if run_all or args.zooniverse_images:
        execute_step(_run_step5_zooniverse_images, db)
    else:
        logging.info("─── STEP 5: SKIPPED (--zooniverse-images not set) ───")

    if run_all or args.biigle_upload:
        # If we are running biigle_upload but NOT running Zooniverse steps,
        # tell Step 6 to pull directly from ML_COMPLETE.
        skip_zooniverse = not (
            run_all or args.zooniverse_clips or args.zooniverse_images
        )
        logging.info(f"Skipping Zooniverse steps: {skip_zooniverse}")
        execute_step(_run_step6_biigle_upload, db, include_ml_complete=skip_zooniverse)

    if run_all or args.biigle_sync:
        execute_step(_run_step7_biigle_sync)

    if run_all or args.retrain:
        log_header("STEP 8: RETRAINING PIPELINE")
        execute_step(run_retraining, auto_promote=True)

    # Push final state (DBs + ML CSVs) to S3
    if args.no_upload:
        logging.info("No-upload set: skipping final S3 sync.")
        log_header("PIPELINE COMPLETE (LOCAL ONLY)", character="═")
    elif config.is_test_run:
        logging.debug("Test run: skipping final S3 sync of annotations directory.")
        log_header("PIPELINE COMPLETE (TEST RUN)", character="═")
    else:
        logging.info("Syncing final results to S3...")
        # Note: sync_pipeline_results uses `aws s3 sync` which handles file comparison.
        # It won't override files that haven't changed locally.
        # TODO add a system to review if it's completely updated
        if sync_pipeline_results(upload_videos=args.upload_videos):
            log_header("PIPELINE SUCCESS (SYNCED TO S3)", character="═")
        else:
            logging.critical(
                "CRITICAL: S3 Sync failed. Pipeline state might be inconsistent on S3."
            )
            log_header("PIPELINE FAILED (SYNC ERROR)", character="═")
            sys.exit(1)

    logging.info(f"Processed {len(results)} drops successfully.")


if __name__ == "__main__":
    main()
