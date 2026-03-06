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
    python run_pipeline.py --staged-test     # Seed DB with staggered drops across all pipeline stages
    python run_pipeline.py --test-run        # Run in test mode with mock data

Steps can be combined: python run_pipeline.py --ingest --biigle-sync
If no step flags are given, ALL steps run.
"""

import argparse
import logging
import traceback
from pathlib import Path

from spyfish.config import config, PipelineStatus
from spyfish.log_config import log_header
from spyfish.storage.db_sync import sync_pipeline_results
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import select_maxn_clips_for_review
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.biigle.upload_frames import upload_frames_to_biigle
from spyfish.ml.select_zooniverse_clips import process_zooniverse_clips
from spyfish.zooniverse.upload import upload_clips_to_zooniverse, check_clip_sizes
from spyfish.ml.process_ml_annotations import run_post_ml
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.orchestrator.ingest import run_ingestion
from spyfish.orchestrator.ingest_legacy import ingest_legacy_expert_annotations
from spyfish.zooniverse.upload import upload_frames_to_zooniverse
from spyfish.test_setup import inject_test_drops


def _run_step1_ingest():
    log_header("STEP 1: METADATA INGESTION: Ingesting metadata into pipeline database")
    run_ingestion()
    ingest_legacy_expert_annotations()


def _run_step2_ml_inference(is_test_run: bool) -> list:
    log_header("STEP 2: ML INFERENCE")
    runner = MLRunner()
    if is_test_run:
        runner.is_test_run = True
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
        annotations_dir=config.local_data_quality_dir,
        video_dir=config.local_video_dir,
        output_root=config.local_data_quality_dir,
    )

def _run_step4_zooniverse_clips(db: DatabaseManager):
    log_header("STEP 4: Zooniverse clip selection + extraction")

    records = db.get_deployments_by_status(PipelineStatus.AWAITING_CITSCI_CLIPS)
    drop_ids = [r['drop_id'] for r in records]

    if not drop_ids:
        logging.info(f"No deployments found with status {PipelineStatus.AWAITING_CITSCI_CLIPS}. Skipping Step 4.")
        return

    model_name = config.pipeline_model_path

    for drop_id in drop_ids:
        logging.info(f"Processing clips for {drop_id}...")
        maxn_csv       = str(config.get_maxn_csv_path(drop_id, model_name))
        selections_csv = str(config.get_selections_csv_path(drop_id))
        video_path     = str(config.get_video_path(drop_id))
        clips_dir      = str(config.get_clips_dir(drop_id))

        selections_df = process_zooniverse_clips(maxn_csv, selections_csv, drop_id, config)
        if selections_df is None or selections_df.empty:
            logging.info(f"No high-confidence clips found for {drop_id}.")
            continue

        clips_df = extract_clips_from_selections(selections_csv_path=selections_csv, video_path=video_path, output_dir=clips_dir)
        clips_df = check_clip_sizes(clips_df)

        logging.info(f"Uploading {len(clips_df)} clips for {drop_id} to Zooniverse.")
        upload_clips_to_zooniverse(clips_df)
        db.update_status(drop_id, PipelineStatus.CITSCI_CLIPS_COMPLETE)

def _run_step5_zooniverse_images(db: DatabaseManager):
    log_header("STEP 5: Zooniverse frame extraction + upload")

    records = db.get_deployments_by_status(PipelineStatus.CITSCI_CLIPS_COMPLETE)
    drop_ids = [r['drop_id'] for r in records]

    if not drop_ids:
        logging.info("No deployments ready for Zooniverse images. Skipping Step 5.")
        return

    model_name = config.pipeline_model_path

    for drop_id in drop_ids:
        logging.info(f"Processing frames for {drop_id}...")
        selections_csv = str(config.get_selections_csv_path(drop_id))
        raw_csv        = str(config.get_raw_csv_path(drop_id, model_name))
        video_path     = str(config.get_video_path(drop_id))
        frames_dir     = str(config.get_frames_dir(drop_id, target="zooniverse"))

        if not Path(selections_csv).exists():
            logging.warning(f"Missing {selections_csv} for {drop_id}. Skipping.")
            continue

        frames_df = extract_frames_from_selections(selections_csv_path=selections_csv, video_path=video_path, raw_csv_path=raw_csv, output_dir=frames_dir)
        logging.info(f"Uploading {len(frames_df)} frames for {drop_id} to Zooniverse.")
        upload_frames_to_zooniverse(frames_df)
        db.update_status(drop_id, PipelineStatus.CITSCI_COMPLETE)

def _run_step6_biigle_upload(db: DatabaseManager, include_ml_complete: bool = False):
    log_header("─── STEP 6: Biigle frame extraction + volume upload ───")

    statuses = [PipelineStatus.CITSCI_COMPLETE]
    if include_ml_complete:
        statuses.append(PipelineStatus.AWAITING_CITSCI_CLIPS)
        logging.info(f"Including {PipelineStatus.AWAITING_CITSCI_CLIPS} deployments as requested.")

    records = db.get_deployments_by_statuses(statuses)
    drop_ids = [r['drop_id'] for r in records]

    if not drop_ids:
        logging.info("No deployments ready for Biigle upload. Skipping Step 6.")
        return

    # TODO what is happening here, why two model paths
    model_name = config.pipeline_model_path
    records_by_id = db.get_deployments_by_ids(drop_ids)

    for drop_id in drop_ids:
        logging.info(f"Uploading volume for {drop_id} to Biigle...")
        record = records_by_id.get(drop_id, {})
        selections_csv = str(config.get_selections_csv_path(drop_id))
        raw_csv        = str(config.get_raw_csv_path(drop_id, model_name))
        video_path     = str(config.get_video_path(drop_id))
        frames_dir     = str(config.get_frames_dir(drop_id, target="biigle"))

        if not Path(selections_csv).exists():
            logging.error(f"Missing {selections_csv} for {drop_id}.")
            continue

        # Extract clean JPEGs + COCO JSON
        frames_df = extract_frames_from_selections(selections_csv_path=selections_csv, video_path=video_path, raw_csv_path=raw_csv, output_dir=frames_dir)

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
        step_name = step_func.__name__.replace('_run_', '').replace('_', ' ').upper()
        logging.error(f"{step_name} FAILED: {e}")
        logging.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description="Run the Spyfish pipeline. Runs all steps by default.")
    parser.add_argument("--ingest",        action="store_true", help="Run Step 1: metadata ingestion")
    parser.add_argument("--ml",            action="store_true", help="Run Steps 2+3: ML inference + post-processing")
    parser.add_argument("--zooniverse-clips", action="store_true", help="Run Step 4: Zooniverse clip extraction")
    parser.add_argument("--zooniverse-images",action="store_true", help="Run Step 5: Zooniverse image extraction")
    parser.add_argument("--biigle-upload", action="store_true", help="Run Step 6: Biigle frame extraction + upload")
    parser.add_argument("--biigle-sync",   action="store_true", help="Run Step 7: Biigle annotation sync")
    parser.add_argument("--staged-test",   action="store_true", help="Seed DB with staggered drops across all pipeline stages")
    parser.add_argument("--test-run",      action="store_true", help="Run in test mode with mock data")
    args = parser.parse_args()

    db = DatabaseManager()

    # If no step flags are given, run everything
    run_all = not any([args.ingest, args.ml, args.zooniverse_clips, args.zooniverse_images, args.biigle_upload, args.biigle_sync, args.staged_test])

    active_steps = "ALL" if run_all else ", ".join(
        s for s, v in [("ingest", args.ingest), ("ml", args.ml),
                       ("zooniverse-clips", args.zooniverse_clips), ("zooniverse-images", args.zooniverse_images),
                       ("biigle-upload", args.biigle_upload), ("biigle-sync", args.biigle_sync),
                       ("staged-test", args.staged_test)] if v
    )

    logging.info("═" * 60)
    logging.info(" SPYFISH AOTEAROA PIPELINE ".center(60, "═"))
    logging.info(f" STEPS: {active_steps} ".center(60, "═"))
    logging.info("═" * 60)

    results = []

    if run_all or args.ingest:
        execute_step(_run_step1_ingest)
    else:
        logging.info("─── STEP 1: SKIPPED (--skip-ingest) ───")

    if config.is_test_run:
        inject_test_drops(db=db, use_pipeline_status=args.staged_test)
        logging.info("Test drops seeded.")

    if run_all or args.ml:
        results = execute_step(_run_step2_ml_inference, args.test_run)
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
        skip_zooniverse = not (run_all or args.zooniverse_clips or args.zooniverse_images)
        execute_step(_run_step6_biigle_upload, db, include_ml_complete=skip_zooniverse)

    if run_all or args.biigle_sync:
        execute_step(_run_step7_biigle_sync)

    # Push final state (DBs + ML CSVs) to S3
    logging.info("Syncing final results to S3...")
    sync_pipeline_results()
    if not config.is_test_run:
        pass
    else:
        logging.debug("Test run: skipping final S3 sync of annotations directory.")

    log_header("PIPELINE COMPLETE", character="═")
    logging.info(f"Processed {len(results)} drops successfully.")

if __name__ == "__main__":
    main()
