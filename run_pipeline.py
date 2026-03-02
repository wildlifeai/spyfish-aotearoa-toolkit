"""
Spyfish Aotearoa — Single-command pipeline runner.

Usage:
    python run_pipeline.py                  # Run full pipeline
    python run_pipeline.py --skip-ingest    # Resume from ML step (skip metadata ingestion)
"""

import argparse
import logging
import traceback

from spyfish.config import config


def main():
    parser = argparse.ArgumentParser(description="Run the full Spyfish pipeline end-to-end.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Step 1 (metadata ingestion)")
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("SPYFISH PIPELINE — FULL RUN")
    logging.info("=" * 60)

    # ── Step 1: Ingest metadata ──────────────────────────────────
    if not args.skip_ingest:
        logging.info("")
        logging.info("─── STEP 1: Ingesting metadata into pipeline database ───")
        try:
            from spyfish.orchestrator.ingest import run_ingestion
            run_ingestion()
        except Exception as e:
            logging.error(f"Step 1 FAILED: {e}")
            logging.error(traceback.format_exc())
            logging.warning("Continuing to Step 2...")
    else:
        logging.info("─── STEP 1: SKIPPED (--skip-ingest) ───")

    # ── Step 2: Run ML inference ─────────────────────────────────
    logging.info("")
    logging.info("─── STEP 2: Running ML inference via Snakemake ───")
    results = []
    try:
        from spyfish.orchestrator.ml_runner import MLRunner
        runner = MLRunner()
        runner.sync_down()
        targets = runner.generate_manifest()
        if not targets:
            logging.info("No targets to process. Pipeline complete.")
            return
        results = runner.run_snakemake(targets)
        runner.sync_up(results)
    except Exception as e:
        logging.error(f"Step 2 FAILED: {e}")
        logging.error(traceback.format_exc())

    # ── Step 3: Post-ML processing (MaxN + Draw Frames) ─────────
    if results:
        logging.info("")
        logging.info("─── STEP 3: Post-ML processing (MaxN + QA frames) ───")
        try:
            from spyfish.ml.process_ml_annotations import run_post_ml
            run_post_ml(
                drop_ids=results,
                annotations_dir=config.local_manifest_dir_path,
                video_dir=config.mock_video_dir,
                output_root=config.local_data_quality_dir,
            )
        except Exception as e:
            logging.error(f"Step 3 FAILED: {e}")
            logging.error(traceback.format_exc())

    logging.info("")
    logging.info("─── SKIPPING STEP 4: Zooniverse clip selection + extraction ───")
    if False:
        # ── Step 4: Zooniverse clip selection + extraction ───────────
        if results:
            logging.info("")
            logging.info("─── STEP 4: Zooniverse clip selection + extraction ───")
            try:
                from pathlib import Path
                from spyfish.database.manager import DatabaseManager
                from spyfish.zooniverse.select_clips import select_zooniverse_clips
                from spyfish.zooniverse.extract_clips import extract_clips_from_selections

                db = DatabaseManager()
                annotations_dir = Path(config.local_manifest_dir_path)
                video_dir = Path(config.mock_video_dir)
                dq_dir = Path(config.local_data_quality_dir)
                model_name = Path(config.model_path or config.mock_model_path).stem

                for drop_id in results:
                    # sampling_start stored in DB so ffmpeg seeks to the right position in the full video
                    record = db.get_deployment(drop_id)
                    sampling_start = int(record["sampling_start"]) if record and record.get("sampling_start") else 0

                    maxn_csv = str(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")
                    selections_csv = str(annotations_dir / f"{drop_id}_zooniverse_selections.csv")
                    video_path = str(video_dir / f"{drop_id}.mp4")
                    clips_dir = str(dq_dir / drop_id / "zooniverse_clips")

                    select_zooniverse_clips(
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

    logging.info("")
    logging.info("=" * 60)
    logging.info(f"PIPELINE COMPLETE — Processed {len(results)} targets")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
