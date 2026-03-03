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
from spyfish.storage.db_sync import upload_db, upload_annotations_db
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
from spyfish.test_setup import inject_staged_test_drops
from spyfish.test_setup import inject_biigle_test_drops


def _run_step1_ingest():
    logging.info("─── STEP 1: Ingesting metadata into pipeline database ───")
    run_ingestion()
    ingest_legacy_expert_annotations()

def _run_biigle_test_seed(db: DatabaseManager):
    logging.info("─── BIIGLE TEST: Seeding DB with test Biigle drops ───")

    inject_biigle_test_drops(db)
    logging.info("Biigle test drops seeded. Will now run Step 7 (Biigle sync).")

def _run_staged_test_seed(db: DatabaseManager):
    logging.info("─── STAGED TEST: Seeding DB with staggered dummy drops ───")

    inject_staged_test_drops(db)
    logging.info("Staged test drops seeded.")

def _run_step2_ml_inference(is_test_run: bool) -> list:
    logging.info("─── STEP 2: Running ML inference loop ───")
    runner = MLRunner()
    if is_test_run:
        runner.is_test_run = True
    targets = runner.generate_manifest()

    if not targets:
        logging.info("No targets to process.")
        return []

    results = runner.run_inference_loop(targets)
    runner.finalize_batch_results(results)
    return results

def _run_step3_post_ml(drop_ids: list):
    logging.info("─── STEP 3: Post-ML processing (MaxN + QA frames) ───")
    run_post_ml(
        drop_ids=drop_ids,
        annotations_dir=config.local_manifest_dir_path,
        video_dir=config.mock_video_dir,
        output_root=config.local_data_quality_dir,
    )

def _run_step4_zooniverse_clips(db: DatabaseManager):
    logging.info("─── STEP 4: Zooniverse clip selection + extraction ───")
    records = db.get_deployments_by_status(PipelineStatus.ML_COMPLETE)
    drop_ids = [r['drop_id'] for r in records]

    if not drop_ids:
        logging.info("No ML results to upload to Zooniverse.")
        return

    annotations_dir = Path(config.local_manifest_dir_path)
    video_dir = Path(config.mock_video_dir) if config.is_test_run else Path(config.nesi_video_dir)
    dq_dir = Path(config.local_data_quality_dir)
    model_name = Path(config.model_path or config.mock_model_path).stem

    for drop_id in drop_ids:
        maxn_csv       = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")
        selections_csv = str(annotations_dir / f"{drop_id}_frames_selection.csv")
        video_path     = str(video_dir / f"{drop_id}.mp4")
        clips_dir      = str(dq_dir / drop_id / "zooniverse_clips")

        selections_df = process_zooniverse_clips(maxn_csv, selections_csv, drop_id, config)
        if selections_df is None or selections_df.empty:
            continue

        clips_df = extract_clips_from_selections(selections_csv_path=selections_csv, video_path=video_path, output_dir=clips_dir)
        clips_df = check_clip_sizes(clips_df)

        logging.warning(f"Clips for {drop_id}: {clips_df}")

        # Uncomment to actually upload:
        # upload_clips_to_zooniverse(clips_df)
        db.update_status(drop_id, PipelineStatus.READY_FOR_CITSCI)

def _run_step5_zooniverse_images(db: DatabaseManager):
    logging.info("─── STEP 5: Zooniverse frame extraction + upload ───")
    records = db.get_deployments_by_status(PipelineStatus.READY_FOR_CITSCI)
    drop_ids = [r['drop_id'] for r in records]

    if not drop_ids:
        logging.info("No ML results to upload to Zooniverse.")
        return

    annotations_dir = Path(config.local_manifest_dir_path)
    video_dir = Path(config.mock_video_dir) if config.is_test_run else Path(config.nesi_video_dir)
    dq_dir = Path(config.local_data_quality_dir)
    model_name = Path(config.model_path or config.mock_model_path).stem

    for drop_id in drop_ids:
        selections_csv = str(annotations_dir / f"{drop_id}_frames_selection.csv")
        raw_csv        = str(annotations_dir / f"{drop_id}_{model_name}_raw.csv")
        video_path     = str(video_dir / f"{drop_id}.mp4")
        frames_dir     = str(dq_dir / drop_id / "zooniverse_frames")

        # Assume selections_csv is prepared by user script or earlier
        if not Path(selections_csv).exists():
            logging.error(f"Missing {selections_csv}.")
            continue

        frames_df = extract_frames_from_selections(selections_csv_path=selections_csv, video_path=video_path, raw_csv_path=raw_csv, output_dir=frames_dir)

        logging.warning(f"Frames for {drop_id}: {frames_df}")

        # upload_frames_to_zooniverse(frames_df)
        db.update_status(drop_id, PipelineStatus.CITSCI_COMPLETE)

def _run_step6_biigle_upload(db: DatabaseManager):
    logging.info("─── STEP 6: Biigle frame extraction + volume upload ───")
    records = db.get_deployments_by_status(PipelineStatus.CITSCI_COMPLETE)
    drop_ids = [r['drop_id'] for r in records]

    if not drop_ids:
        logging.info("No test/citsci results to upload to Biigle.")
        return

    annotations_dir = Path(config.local_manifest_dir_path)
    video_dir = Path(config.mock_video_dir) if config.is_test_run else Path(config.nesi_video_dir)
    dq_dir = Path(config.local_data_quality_dir)
    model_name = Path(config.model_path or config.mock_model_path).stem

    for drop_id in drop_ids:
        record = db.get_deployment(drop_id)
        sampling_start = int(record["sampling_start"]) if record and record.get("sampling_start") else 0

        selections_csv = str(annotations_dir / f"{drop_id}_frames_selection.csv")
        raw_csv        = str(annotations_dir / f"{drop_id}_{model_name}_raw.csv")
        video_path     = str(video_dir / f"{drop_id}.mp4")
        frames_dir     = str(dq_dir / drop_id / "biigle_frames")

        if not Path(selections_csv).exists():
            logging.error(f"Missing {selections_csv} for Biigle upload. Was Step 4 run?")
            continue

        # 6b — Extract clean JPEGs + COCO JSON
        frames_df = extract_frames_from_selections(selections_csv_path=selections_csv, video_path=video_path, raw_csv_path=raw_csv, output_dir=frames_dir)

        # 6c — Upload to S3 + create Biigle volume
        volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
        logging.info(f"Step 6 complete for {drop_id}: Biigle volume id={volume_info.get('id')}")

        db.update_status(drop_id, PipelineStatus.READY_FOR_EXPERT)

def _run_step7_biigle_sync():
    logging.info("─── STEP 7: Syncing Biigle annotations ───")
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
    parser.add_argument("--biigle-upload", action="store_true", help="Run Step 6: Biigle frame extraction + volume upload")
    parser.add_argument("--biigle-sync",   action="store_true", help="Run Step 7: Biigle annotation sync")
    parser.add_argument("--biigle-test",   action="store_true", help="Seed DB with BIIGLE_TEST_DROPS (READY_FOR_EXPERT + known volume ID), then run Step 7")
    parser.add_argument("--staged-test",   action="store_true", help="Seed DB with staggered drops across all pipeline stages")
    parser.add_argument("--test-run",      action="store_true", help="Run in test mode with mock data")
    args = parser.parse_args()

    db = DatabaseManager()

    # If no step flags are given, run everything
    run_all = not any([args.ingest, args.ml, args.zooniverse_clips, args.zooniverse_images, args.biigle_upload, args.biigle_sync, args.biigle_test, args.staged_test])

    active_steps = "ALL" if run_all else ", ".join(
        s for s, v in [("ingest", args.ingest), ("ml", args.ml),
                       ("zooniverse-clips", args.zooniverse_clips), ("zooniverse-images", args.zooniverse_images),
                       ("biigle-upload", args.biigle_upload), ("biigle-sync", args.biigle_sync),
                       ("biigle-test", args.biigle_test), ("staged-test", args.staged_test)] if v
    )

    logging.info("=" * 60)
    logging.info("SPYFISH PIPELINE")
    logging.info(f"STEPS: {active_steps}")
    logging.info("=" * 60)

    results = []

    if run_all or args.ingest:
        execute_step(_run_step1_ingest)
    else:
        logging.info("─── STEP 1: SKIPPED (--skip-ingest) ───")

    if args.biigle_test:
        execute_step(_run_biigle_test_seed, db)

    if args.staged_test:
        execute_step(_run_staged_test_seed, db)

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
        execute_step(_run_step6_biigle_upload, db)
    else:
        logging.info("─── STEP 6: SKIPPED (--biigle-upload not set) ───")

    if run_all or args.biigle_sync or args.biigle_test:
        execute_step(_run_step7_biigle_sync)

    # Push final DB state to S3 for Streamlit apps to read
    upload_db()
    upload_annotations_db()

    logging.info("=" * 60)
    logging.info(f"PIPELINE COMPLETE — {len(results)} drops processed")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
