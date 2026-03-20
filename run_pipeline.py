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

import pandas as pd

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
from spyfish.orchestrator.stage import DropStage, GlobalStage, StageRunner
from spyfish.storage.db_sync import sync_pipeline_results
from spyfish.zooniverse.select_zooniverse_clips import process_zooniverse_clips
from spyfish.zooniverse.upload import (
    check_clip_sizes,
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
        "zooniverse_clips": str(config.get_clips_dir(drop_id, target="zooniverse")),
        "zooniverse_frames": str(config.get_frames_dir(drop_id, target="zooniverse")),
        "biigle_frames": str(config.get_frames_dir(drop_id, target="biigle")),
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

    all_drop_ids = [t[config.drop_id_column] for t in targets]
    results = runner.run_inference_loop(targets)
    runner.finalize_batch_results(results, all_drop_ids=all_drop_ids)

    if results:
        run_post_ml(
            drop_ids=results,
            annotations_dir=str(config.data_quality_dir),
            video_dir=str(config.media_dir),
            output_root=str(config.data_quality_dir),
        )


def _step_zooniverse_sync_drop(drop_id: str) -> str:
    """Zooniverse volunteer annotation sync-back.

    Checks whether volunteer classification is complete for the subject set
    associated with this drop, then downloads and stores results.

    TODO: Implement Zooniverse API check via panoptes_client:
      - Query subject set classification counts for the drop's subject set.
      - Define "done" threshold (e.g. minimum N classifications per subject,
        or Caesar reduction pipeline completion, or manual sign-off flag).
      - On completion: download classification export, parse volunteer
        annotations, store them for downstream use.
      - Until done: leave status as AWAITING_CITSCI_FRAMES and return early
        (do not advance).

    For now this is a no-op placeholder that immediately advances to
    CITSCI_COMPLETE so the rest of the pipeline (Biigle upload) can proceed.
    """
    logging.warning(
        f"zooniverse-sync is a placeholder — no Zooniverse API check performed for {drop_id}. "
        "Advancing to CITSCI_COMPLETE without verifying volunteer annotation completion."
    )
    return PipelineStatus.CITSCI_COMPLETE


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

    selections_df = process_zooniverse_clips(
        paths["maxn_csv"], paths["selections_csv"], drop_id, config
    )
    if selections_df is None or selections_df.empty:
        logging.info(
            f"No high-confidence clips for {drop_id}. Advancing to CITSCI_CLIPS_COMPLETE."
        )
        return PipelineStatus.CITSCI_CLIPS_COMPLETE

    clips_df = extract_clips_from_selections(
        selections_csv_path=paths["selections_csv"],
        video_path=paths["video_path"],
        output_dir=paths["zooniverse_clips"],
    )
    clips_df = check_clip_sizes(clips_df)
    logging.info(f"Uploading {len(clips_df)} clips for {drop_id} to Zooniverse.")
    upload_clips_to_zooniverse(clips_df)
    return PipelineStatus.CITSCI_CLIPS_COMPLETE


def _step5_process_drop(drop_id: str) -> str:
    """Step 5: Zooniverse frame extraction + upload."""
    paths = _get_common_paths(drop_id)

    if not Path(paths["selections_csv"]).exists():
        logging.warning(
            f"Missing {paths['selections_csv']} for {drop_id}. Advancing to CITSCI_COMPLETE."
        )
        return PipelineStatus.CITSCI_COMPLETE

    frames_df = extract_frames_from_selections(
        selections_csv_path=paths["selections_csv"],
        video_path=paths["video_path"],
        raw_csv_path=paths["raw_csv"],
        output_dir=paths["zooniverse_frames"],
    )
    logging.info(f"Uploading {len(frames_df)} frames for {drop_id} to Zooniverse.")
    upload_frames_to_zooniverse(frames_df)
    return PipelineStatus.AWAITING_CITSCI_FRAMES


def _step6_process_drop(drop_id: str) -> str:
    """Step 6: Biigle frame extraction + volume upload."""
    paths = _get_common_paths(drop_id)
    selections_path = Path(paths["selections_csv"])

    if not selections_path.exists():
        # Biigle-direct path: generate frame selections from MaxN CSV
        maxn_path = Path(paths["maxn_csv"])
        if not maxn_path.exists():
            logging.error(
                f"Missing MaxN CSV at {maxn_path} for {drop_id}. Cannot generate frame selections."
            )
            return PipelineStatus.AWAITING_EXPERT_REVIEW

        maxn_df = pd.read_csv(maxn_path)
        if maxn_df.empty:
            logging.warning(
                f"Empty MaxN CSV for {drop_id} — no detections, no frames to upload to Biigle. "
                "Advancing to AWAITING_EXPERT_REVIEW."
            )
            return PipelineStatus.AWAITING_EXPERT_REVIEW

        maxn_df = maxn_df.rename(
            columns={config.csv_maxn_time_ms_column: config.csv_clip_max_time_column}
        )
        maxn_df["SelectionReason"] = "MaxN Peak"
        selections_path.parent.mkdir(parents=True, exist_ok=True)
        maxn_df.to_csv(selections_path, index=False)
        logging.info(
            f"Generated {len(maxn_df)} frame selections from MaxN CSV for {drop_id} (biigle-direct path)."
        )

    frames_df = extract_frames_from_selections(
        selections_csv_path=paths["selections_csv"],
        video_path=paths["video_path"],
        raw_csv_path=paths["raw_csv"],
        output_dir=paths["biigle_frames"],
    )
    volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
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
    parser.add_argument("--step0", action="store_true", help="Run Step 0: test run")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip all S3 uploads (DB, models, results)",
    )
    parser.add_argument(
        "--test-run", action="store_true", help="Run in test mode with mock data"
    )
    args = parser.parse_args()

    logging.info("═" * 60)
    logging.info(" SPYFISH AOTEAROA PIPELINE ".center(60, "═"))
    logging.info(f" NO-UPLOAD: {args.no_upload} ".center(60, "═"))
    logging.info("═" * 60)

    if args.step0:
        log_header("STEP 0: TEST RUN")
        logging.info(f"bucket name: {config.s3_bucket}")
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
