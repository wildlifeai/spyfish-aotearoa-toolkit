"""
Spyfish Aotearoa — Single-command pipeline runner.

Happy path (no flags runs this sequence end-to-end):

    ingest → ml → zooniverse-clips → zooniverse-images → zooniverse-sync
           → biigle-upload → biigle-sync → retrain

Usage:
    python run_pipeline.py                    # Run the happy-path sequence (default)
    python run_pipeline.py --ingest           # Metadata ingestion
    python run_pipeline.py --ml               # ML inference + post-processing
    python run_pipeline.py --zooniverse-clips # Zooniverse clip extraction + upload
    python run_pipeline.py --zooniverse-images# Zooniverse frame extraction + upload
    python run_pipeline.py --zooniverse-sync  # Zooniverse volunteer sync-back
    python run_pipeline.py --biigle-upload    # Biigle frame extraction + upload
    python run_pipeline.py --biigle-sync      # Biigle annotation sync
    python run_pipeline.py --retrain          # Model retraining
    python run_pipeline.py --check-arrivals   # Poll S3 for newly arrived videos (off happy path)
    python run_pipeline.py --set-targets      # Bulk-set pipeline stages from CSV  (off happy path)
    python run_pipeline.py --legacy           # Historical backfill                (off happy path)

Flags can be combined: python run_pipeline.py --ingest --biigle-sync
If no flags are given, the full happy-path sequence runs.

Adding a new stage: write the function, add one entry to STAGES. Argparse,
eligibility, status transitions, and logging are wired up automatically.
"""

import argparse
import functools
import logging
import sys
from dataclasses import replace
from pathlib import Path

from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.biigle.upload_frames import upload_frames_to_biigle
from spyfish.config.base import BiigleStatus, CitSciStatus, MlStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.extraction.select_frames import select_frames
from spyfish.log_config import log_header
from spyfish.orchestrator.ingest import check_pending_arrivals, run_ingestion
from spyfish.orchestrator.legacy_extract import ingest_legacy_expert_annotations
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


def _run_ingest() -> None:
    run_ingestion()


def _run_legacy() -> None:
    """Historical backfill: expert annotation CSV + legacy Zooniverse CSV ingestion."""
    from spyfish.zooniverse.legacy_extract import run_legacy_zooniverse_backfill

    ingest_legacy_expert_annotations()
    run_legacy_zooniverse_backfill()


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


def _run_ml() -> None:
    runner = MLRunner()
    targets = runner.get_inference_targets()

    if not targets:
        logging.info("No drops available for ML processing.")
        return

    all_drop_ids = [t["drop_id"] for t in targets]
    # MaxN + QA frames are written per-drop inside run_inference_loop, before
    # each drop is marked ml_complete. finalize_batch_results is only the safety
    # net for any drops still stuck in ml_running after the loop exits.
    results = runner.run_inference_loop(targets)
    runner.finalize_batch_results(results, all_drop_ids=all_drop_ids)


def _run_zooniverse_sync_drop(drop_id: str) -> str | None:
    from spyfish.zooniverse.parse_classifications import sync_zooniverse_drop

    return sync_zooniverse_drop(drop_id)


def _run_biigle_sync() -> None:
    sync_biigle_annotations()


def _run_retrain(
    data_prep: bool = True, binary: bool = True, species: bool = True
) -> None:
    run_retraining(
        data_prep=data_prep, binary=binary, species=species, auto_promote=True
    )


# ---------------------------------------------------------------------------
# Per-drop stage functions — (drop_id: str) -> target section status str | None
# ---------------------------------------------------------------------------


def _run_zooniverse_clips_drop(drop_id: str) -> str | None:
    """Zooniverse clip selection + extraction + upload."""
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
    return CitSciStatus.CLIPS_UPLOADED


def _run_zooniverse_images_drop(drop_id: str) -> str | None:
    """Zooniverse frame extraction + upload."""
    paths = _get_common_paths(drop_id)

    if not Path(paths["selections_csv"]).exists():
        logging.error(
            f"Missing selections CSV for {drop_id} — zooniverse-clips should have written it."
        )
        return None

    frames_df = extract_frames_from_selections(
        selections_csv_path=paths["selections_csv"],
        video_path=paths["video_path"],
        raw_csv_path=paths["raw_csv"],
    )
    logging.info(f"Uploading {len(frames_df)} frames for {drop_id} to Zooniverse.")
    upload_frames_to_zooniverse(frames_df)
    return CitSciStatus.FRAMES_UPLOADED


def _run_biigle_upload_drop(drop_id: str) -> str | None:
    """Biigle frame extraction + volume upload."""
    paths = _get_common_paths(drop_id)
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
    return BiigleStatus.UPLOADED


# ---------------------------------------------------------------------------
# Dynamic prerequisites for biigle-upload
# ---------------------------------------------------------------------------


def _biigle_prerequisites(args: argparse.Namespace, run_all: bool) -> dict[str, str]:
    """Returns the prerequisite section condition for biigle-upload.

    Zooniverse path (default):   wait for citsci_status=complete
    Biigle-direct (skip zooniverse): wait for ml_status=complete
    """
    skip_zooniverse = not (
        run_all
        or getattr(args, "zooniverse_clips", False)
        or getattr(args, "zooniverse_images", False)
        or getattr(args, "zooniverse_sync", False)
    )
    if skip_zooniverse:
        return {"ml_status": MlStatus.COMPLETE}
    return {"citsci_status": CitSciStatus.COMPLETE}


# ---------------------------------------------------------------------------
# Stage registry — add a new pipeline stage by adding ONE entry here
# ---------------------------------------------------------------------------

STAGES: list = [
    GlobalStage("ingest", "Metadata ingestion", _run_ingest),
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
        "legacy",
        "Historical backfill: expert annotations + legacy Zooniverse CSVs",
        _run_legacy,
        run_in_all=False,
    ),
    GlobalStage("ml", "ML inference + post-processing", _run_ml),
    DropStage(
        "zooniverse-clips",
        "Zooniverse clip extraction",
        _run_zooniverse_clips_drop,
        section="citsci_status",
        input_statuses=[CitSciStatus.PENDING],
        prerequisites={"ml_status": MlStatus.COMPLETE},
    ),
    DropStage(
        "zooniverse-images",
        "Zooniverse image extraction",
        _run_zooniverse_images_drop,
        section="citsci_status",
        input_statuses=[CitSciStatus.CLIPS_UPLOADED],
    ),
    DropStage(
        "zooniverse-sync",
        "Zooniverse volunteer sync-back",
        _run_zooniverse_sync_drop,
        section="citsci_status",
        input_statuses=[CitSciStatus.FRAMES_UPLOADED],
    ),
    DropStage(
        "biigle-upload",
        "Biigle frame extraction + upload",
        _run_biigle_upload_drop,
        section="biigle_status",
        input_statuses=[BiigleStatus.PENDING],
        prerequisites=_biigle_prerequisites,
    ),
    GlobalStage("biigle-sync", "Biigle annotation sync", _run_biigle_sync),
    GlobalStage(
        "retrain",
        "Model retraining (run --biigle-sync first)",
        _run_retrain,
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
        "--data-prep",
        action="store_true",
        help="On --retrain, include the data prep step. If no step flags "
        "(--data-prep, --binary, --species) are passed, all three run.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="On --retrain, include the binary training step.",
    )
    parser.add_argument(
        "--species",
        action="store_true",
        help="On --retrain, include the species training step.",
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
        return

    def _patch_stage(s):
        if s.flag == "set-targets":
            return replace(
                s, fn=functools.partial(_run_set_targets, push_s3=not args.no_upload)
            )
        if s.flag == "retrain":
            # Compose-style: no flags = all steps; any flag = only those steps.
            no_step_specified = not (args.data_prep or args.binary or args.species)
            do_data_prep = args.data_prep or no_step_specified
            do_binary = args.binary or no_step_specified
            do_species = args.species or no_step_specified
            return replace(
                s,
                fn=functools.partial(
                    _run_retrain,
                    data_prep=do_data_prep,
                    binary=do_binary,
                    species=do_species,
                ),
            )
        return s

    patched_stages = [_patch_stage(s) for s in STAGES]
    runner = StageRunner(patched_stages, db)
    runner.run(args)

    # Always run the S3 sync — we want the partial results of whatever did
    # succeed to be persisted, even if some stage failed.
    if args.no_upload:
        logging.info("No-upload set: skipping final S3 sync.")
        log_header("PIPELINE COMPLETE (LOCAL ONLY)", character="═")
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

    # Exit non-zero if any global stage raised — this way CI / cron can tell
    # the run wasn't fully clean even though the pipeline kept going.
    if runner.failed_stages:
        logging.critical(
            f"Pipeline exiting with non-zero status due to failed stages: "
            f"{', '.join(runner.failed_stages)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
