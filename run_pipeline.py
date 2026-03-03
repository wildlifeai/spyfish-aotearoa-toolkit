"""
Spyfish Aotearoa — Single-command pipeline runner.

Usage:
    python run_pipeline.py                   # Run all steps (default)
    python run_pipeline.py --ingest          # Only run Step 1 (metadata ingestion)
    python run_pipeline.py --ml              # Only run Steps 2+3 (ML inference + post-processing)
    python run_pipeline.py --biigle-upload   # Only run Step 5 (Biigle frame extraction + upload)
    python run_pipeline.py --biigle-sync     # Only run Step 6 (Biigle annotation sync)
    python run_pipeline.py --biigle-test     # Seed DB with test Biigle drops + run Step 6
    python run_pipeline.py --test-run        # Run in test mode with mock data

Steps can be combined: python run_pipeline.py --ingest --biigle-sync
If no step flags are given, ALL steps run.
"""

import argparse
import logging
import traceback
from pathlib import Path

from spyfish.config import config, PipelineStatus
from spyfish.storage.db_sync import upload_db
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import select_maxn_clips_for_review
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.biigle.upload_frames import upload_frames_to_biigle
from spyfish.ml.process_ml_annotations import run_post_ml
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.orchestrator.ingest import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Run the Spyfish pipeline. Runs all steps by default.")
    parser.add_argument("--ingest",        action="store_true", help="Run Step 1: metadata ingestion")
    parser.add_argument("--ml",            action="store_true", help="Run Steps 2+3: ML inference + post-processing")
    parser.add_argument("--biigle-upload", action="store_true", help="Run Step 5: Biigle frame extraction + volume upload")
    parser.add_argument("--biigle-sync",   action="store_true", help="Run Step 6: Biigle annotation sync")
    parser.add_argument("--biigle-test",   action="store_true", help="Seed DB with BIIGLE_TEST_DROPS (READY_FOR_EXPERT + known volume ID), then run Step 6")
    parser.add_argument("--test-run",      action="store_true", help="Run in test mode with mock data")
    args = parser.parse_args()

    # If no step flags are given, run everything
    run_all = not any([args.ingest, args.ml, args.biigle_upload, args.biigle_sync, args.biigle_test])

    active_steps = "ALL" if run_all else ", ".join(
        s for s, v in [("ingest", args.ingest), ("ml", args.ml),
                       ("biigle-upload", args.biigle_upload), ("biigle-sync", args.biigle_sync),
                       ("biigle-test", args.biigle_test)] if v
    )

    logging.info("=" * 60)
    logging.info("SPYFISH PIPELINE")
    logging.info(f"STEPS: {active_steps}")
    logging.info("=" * 60)

    results = []

    # ── Step 1: Ingest metadata ──────────────────────────────────
    if run_all or args.ingest:
        logging.info("─── STEP 1: Ingesting metadata into pipeline database ───")
        try:
            run_ingestion()
        except Exception as e:
            logging.error(f"Step 1 FAILED: {e}")
            logging.error(traceback.format_exc())
            raise
    else:
        logging.info("─── STEP 1: SKIPPED (--skip-ingest) ───")

    # ── Biigle test seed ─────────────────────────────────────────
    if args.biigle_test:
        logging.info("─── BIIGLE TEST: Seeding DB with test Biigle drops ───")
        try:
            from spyfish.test_setup import inject_biigle_test_drops
            db = DatabaseManager()
            inject_biigle_test_drops(db)
            logging.info("Biigle test drops seeded. Will now run Step 6 (Biigle sync).")
            # Force biigle_sync to also run
            args.biigle_sync = True
        except Exception as e:
            logging.error(f"Biigle test seed FAILED: {e}")
            logging.error(traceback.format_exc())
            raise

    # ── Step 2: Run ML inference ─────────────────────────────────
    if run_all or args.ml:
        results = []
        logging.info("─── STEP 2: Running ML inference loop ───")
        try:
            runner = MLRunner()
            if args.test_run:
                runner.is_test_run = True
            targets = runner.generate_manifest()
            if not targets:
                logging.info("No targets to process.")
            else:
                results = runner.run_inference_loop(targets)
                runner.finalize_batch_results(results)
        except Exception as e:
            logging.error(f"Step 2 FAILED: {e}")
            logging.error(traceback.format_exc())
            raise

        # ── Step 3: Post-ML processing (MaxN + Draw Frames) ─────
        if results:
            logging.info("─── STEP 3: Post-ML processing (MaxN + QA frames) ───")
            try:
                run_post_ml(
                    drop_ids=results,
                    annotations_dir=config.local_manifest_dir_path,
                    video_dir=config.mock_video_dir,
                    output_root=config.local_data_quality_dir,
                )
            except Exception as e:
                logging.error(f"Step 3 FAILED: {e}")
                logging.error(traceback.format_exc())
                raise

    # ── Step 4: Zooniverse (TODO) ────────────────────────────────
    # TODO: add --zooniverse flag and implement
    logging.info("─── SKIPPING STEP 4: Zooniverse clip selection + extraction ───")
    if False:  # TODO: skipping this step
        if results:
            try:
                db = DatabaseManager()
                annotations_dir = Path(config.local_manifest_dir_path)
                video_dir = Path(config.mock_video_dir)
                dq_dir = Path(config.local_data_quality_dir)
                model_name = Path(config.model_path or config.mock_model_path).stem

                for drop_id in results:
                    record = db.get_deployment(drop_id)
                    sampling_start = int(record["sampling_start"]) if record and record.get("sampling_start") else 0
                    maxn_csv = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")
                    selections_csv = str(annotations_dir / f"{drop_id}_zooniverse_selections.csv")
                    video_path = str(video_dir / f"{drop_id}.mp4")
                    clips_dir = str(dq_dir / drop_id / "zooniverse_clips")

                    select_maxn_clips_for_review(
                        maxn_csv_path=maxn_csv,
                        output_selections_path=selections_csv,
                        drop_id=drop_id,
                        sampling_start=sampling_start,
                    )
                    # Returns selections_df with ClipPath column added
                    clips_df = extract_clips_from_selections(
                        selections_csv_path=selections_csv,
                        video_path=video_path, # TODO, probably combined path
                        output_dir=clips_dir,
                    )
                    logging.info(f"Step 4 complete for {drop_id}: {clips_df['ClipPath'].notna().sum()} clips ready for upload.")
                    # To upload: from spyfish.zooniverse.upload import upload_clips_to_zooniverse
                    #            upload_clips_to_zooniverse(clips_df)
            except Exception as e:
                logging.error(f"Step 4 FAILED: {e}")
                logging.error(traceback.format_exc())
                raise

    # ── Step 5: Biigle frame extraction + volume upload ──────────
    if run_all or args.biigle_upload:
        logging.info("─── STEP 5: Biigle frame extraction + volume upload ───")
        if not results:
            logging.info("No ML results to upload to Biigle.")
        else:
            db = DatabaseManager()
            annotations_dir = Path(config.local_manifest_dir_path)
            video_dir = Path(config.mock_video_dir)
            dq_dir = Path(config.local_data_quality_dir)
            model_name = Path(config.model_path or config.mock_model_path).stem

            for drop_id in results:
                try:
                    record = db.get_deployment(drop_id)
                    sampling_start = int(record["sampling_start"]) if record and record.get("sampling_start") else 0

                    maxn_csv       = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")
                    selections_csv = str(annotations_dir / f"{drop_id}_biigle_selections.csv")
                    raw_csv        = str(annotations_dir / f"{drop_id}_{model_name}_raw.csv")
                    video_path     = str(video_dir / f"{drop_id}.mp4")
                    frames_dir     = str(dq_dir / drop_id / "biigle_frames")

                    # 5a — Select the MaxN peak frames
                    select_maxn_clips_for_review(maxn_csv_path=maxn_csv, output_selections_path=selections_csv, drop_id=drop_id, sampling_start=sampling_start)

                    # 5b — Extract clean JPEGs + COCO JSON
                    frames_df = extract_frames_from_selections(selections_csv_path=selections_csv, video_path=video_path, raw_csv_path=raw_csv, output_dir=frames_dir)

                    # 5c — Upload to S3 + create Biigle volume
                    volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
                    logging.info(f"Step 5 complete for {drop_id}: Biigle volume id={volume_info.get('id')}")

                    db.update_status(drop_id, PipelineStatus.READY_FOR_CITSCI)

                except Exception as e:
                    logging.error(f"Step 5 FAILED for {drop_id}: {e}")
                    logging.error(traceback.format_exc())
                    raise

    else:
        logging.info("─── STEP 5: SKIPPED (--skip-biigle-upload) ───")

    # ── Step 6: Biigle annotation sync ──────────────────────────
    if run_all or args.biigle_sync:
        logging.info("─── STEP 6: Syncing Biigle annotations ───")
        try:
            sync_biigle_annotations()
        except Exception as e:
            logging.error(f"Step 6 FAILED: {e}")
            logging.error(traceback.format_exc())
            raise

    # Push final DB state to S3 for Streamlit apps to read
    upload_db()

    logging.info("=" * 60)
    logging.info(f"PIPELINE COMPLETE — {len(results)} drops processed")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
