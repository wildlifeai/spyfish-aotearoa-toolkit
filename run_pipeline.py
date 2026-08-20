"""
Spyfish Aotearoa — Single-command pipeline runner.

Data pipeline (no flags runs this sequence end-to-end):

    ml → zooniverse-clips → zooniverse-sync → biigle-upload → biigle-sync → retrain

    --ml                 ML inference + post-processing
    --zooniverse-clips   Zooniverse clip extraction + upload
    --zooniverse-sync    Zooniverse volunteer sync-back (per-subject-set API fetch)
    --biigle-upload      Biigle frame extraction + upload
    --biigle-sync        Biigle annotation sync
    --retrain            Model retraining

Admin / maintenance (always explicit — never run by default):

    --ingest             Refresh metadata from SharePoint CSVs on S3
    --check-arrivals     Cheap S3 poll for newly arrived videos
    --set-targets        Bulk-set pipeline stages from CSV
    --legacy-experts     Historical backfill: legacy expert annotations CSV
    --legacy-zooniverse  Historical backfill: legacy Zooniverse classifications
    --db-refresh         Reconcile DB status with on-disk + API state

Typical cron pattern:
    python run_pipeline.py --ingest   # load new metadata first
    python run_pipeline.py            # then process the funnel

Flags can be combined: python run_pipeline.py --ingest --biigle-sync

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
from spyfish.config.base import CitSciStatus, ExpertStatus, MlStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.extraction.select_frames import (
    select_frames,
    select_frames_from_zooniverse,
)
from spyfish.log_config import log_header
from spyfish.ml.run_inference import rerun_inference_on_extracted_frames
from spyfish.orchestrator.db_refresh import run_db_refresh
from spyfish.orchestrator.ingest import check_pending_arrivals, run_ingestion
from spyfish.orchestrator.legacy_extract import ingest_legacy_expert_annotations
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.orchestrator.retrain_runner import run_retraining
from spyfish.orchestrator.stage import DropStage, GlobalStage, StageRunner
from spyfish.storage.db_sync import sync_pipeline_results
from spyfish.zooniverse.legacy_extract import run_legacy_zooniverse_backfill
from spyfish.zooniverse.select_zooniverse_clips import process_zooniverse_clips
from spyfish.zooniverse.upload import upload_clips_to_zooniverse

# ---------------------------------------------------------------------------
# Global stage functions — run once, manage their own iteration internally
# ---------------------------------------------------------------------------


def _run_ingest() -> None:
    run_ingestion()


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


def _run_ml(survey_id: str | None = None, force: bool = False) -> None:
    runner = MLRunner()
    targets = runner.get_inference_targets(survey_id=survey_id, force=force)

    if not targets:
        logging.info("No drops available for ML processing.")
        return

    all_drop_ids = [t["drop_id"] for t in targets]
    # MaxN + QA frames are written per-drop inside run_inference_loop, before
    # each drop is marked ml_complete. finalize_batch_results is only the safety
    # net for any drops still stuck in ml_running after the loop exits.
    results = runner.run_inference_loop(targets)
    runner.finalize_batch_results(results, all_drop_ids=all_drop_ids)


def _run_zooniverse_sync(force: bool = False) -> None:
    """Wire to ``spyfish.zooniverse.sync.sync_zooniverse_drops``."""
    from spyfish.zooniverse.sync import sync_zooniverse_drops

    sync_zooniverse_drops(force=force)


def _run_biigle_sync() -> None:
    sync_biigle_annotations()


def _run_retrain(
    data_prep: bool = True,
    binary: bool = True,
    species: bool = True,
    dry_run: bool = False,
) -> None:
    run_retraining(
        data_prep=data_prep,
        binary=binary,
        species=species,
        auto_promote=True,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Per-drop stage functions — (drop_id: str) -> target section status str | None
# ---------------------------------------------------------------------------


def _run_zooniverse_clips_drop(drop_id: str) -> str | None:
    """Zooniverse clip selection + extraction + upload."""
    model_name = Path(config.pipeline_model_path).stem
    maxn_csv = str(config.get_maxn_csv_path(drop_id, model_name))
    selections_csv = str(config.get_selections_csv_path(drop_id))
    video_path = str(config.get_video_path(drop_id))

    try:
        selections_df = process_zooniverse_clips(maxn_csv, selections_csv, drop_id)
    except FileNotFoundError as e:
        logging.error(f"MaxN CSV missing for {drop_id}, cannot select clips: {e}")
        return None

    if selections_df.empty:
        logging.error(
            f"No clips selected for {drop_id} — sampling window may be too short for clip length."
        )
        return None

    clips_df = extract_clips_from_selections(
        selections_csv_path=selections_csv,
        video_path=video_path,
    )
    logging.info(f"Uploading {len(clips_df)} clips for {drop_id} to Zooniverse.")
    upload_clips_to_zooniverse(clips_df)
    return CitSciStatus.CLIPS_UPLOADED


def _run_biigle_upload_drop(drop_id: str) -> str | None:
    """Biigle frame extraction + volume upload.

    Routing is "use the best available data" — independent of which CLI
    flags were passed. The CLI mode controls *eligibility* (see
    ``_biigle_prerequisites``); this function controls *data source*
    given the drop's current state:
      - ``citsci_status == citsci_complete`` → Zooniverse volunteer MaxN CSV
        (frames at volunteer-identified peaks).
      - otherwise → ML raw CSV (frames at model-detected peaks).

    So a drop that reached citsci_complete via the full pipeline AND a
    drop that the user is force-pushing through with ``--biigle-upload``
    alone both take the Zooniverse path when the volunteer data exists.

    On the Zooniverse path, frames are extracted at volunteer-identified
    timestamps where the ML raw CSV has no detections, so we re-run YOLO
    (species + binary ensemble, IoU-merged) on each extracted JPEG and
    rebuild the COCO JSON before upload — see
    ``rerun_inference_on_extracted_frames``. Without that step the
    BIIGLE upload would carry blank frames and experts would annotate
    from scratch instead of correcting model boxes.
    """
    model_name = Path(config.pipeline_model_path).stem
    raw_csv = str(config.get_raw_csv_path(drop_id, model_name))
    video_path = str(config.get_video_path(drop_id))
    biigle_selections_path = config.get_biigle_selections_csv_path(drop_id)

    # Resolve the ML raw CSV for citsci-path peak augmentation. Prefer the
    # canonical species-model path; fall back to any non-zooniverse-rerun
    # raw CSV in the drop's annotations dir. Lets the augmentation pick up
    # data from a model whose stem doesn't match config.pipeline_model_path
    # (e.g. a binary or sweep model whose MaxN landed on disk via a one-off).
    ml_raw_csv: str | None = None
    if Path(raw_csv).exists():
        ml_raw_csv = raw_csv
    else:
        non_rerun_raws = sorted(
            p
            for p in config.get_drop_annotations_dir(drop_id).glob(
                f"{drop_id}_*_raw.csv"
            )
            if "zooniverse_frames" not in p.name
        )
        if non_rerun_raws:
            ml_raw_csv = str(non_rerun_raws[0])
            logging.info(
                f"{drop_id}: using non-canonical ML raw CSV "
                f"{Path(ml_raw_csv).name} for ML peak augmentation."
            )

    db = DatabaseManager()
    deployment = db.get_deployment(drop_id)
    use_zooniverse = (
        deployment is not None
        and deployment.get("citsci_status") == CitSciStatus.COMPLETE
    )

    # ValueError → "input is empty by design" (volunteers said NOTHINGHERE,
    #              or ML found nothing). Nothing for the expert to review →
    #              advance to expert_skipped so the drop exits the queue.
    # FileNotFoundError → artifact missing despite the upstream status saying
    #              it should exist. That's a state inconsistency, not a
    #              "nothing to review" case → re-raise and let StageRunner
    #              mark expert_error so it's visible.
    try:
        if use_zooniverse:
            logging.info(
                f"{drop_id}: citsci_complete — selecting frames from Zooniverse volunteer consensus."
            )
            select_frames_from_zooniverse(
                str(config.get_zooniverse_maxn_csv_path(drop_id)),
                str(biigle_selections_path),
                drop_id,
                ml_raw_csv_path=ml_raw_csv,
            )
        else:
            select_frames(raw_csv, str(biigle_selections_path), drop_id)
    except ValueError as e:
        logging.info(
            f"{drop_id}: no detections to review ({e}) — marking expert_skipped."
        )
        return ExpertStatus.SKIPPED

    # Zooniverse-selected timestamps don't align with ML raw CSV detections,
    # so on that path we skip the COCO write here and let the rerun-inference
    # step write the COCO from a fresh ensemble pass.
    frames_df = extract_frames_from_selections(
        selections_csv_path=str(biigle_selections_path),
        video_path=video_path,
        raw_csv_path="" if use_zooniverse else raw_csv,
        write_coco=not use_zooniverse,
    )

    if use_zooniverse:
        rerun_inference_on_extracted_frames(drop_id, frames_df)

    volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
    if volume_info is None:
        # upload_frames_to_biigle returns None when the COCO has zero
        # annotations — no boxes for experts to correct. Skip cleanly
        # rather than leave the drop stuck at expert_pending.
        logging.info(
            f"{drop_id}: BIIGLE upload skipped (no annotations in COCO) — "
            "marking expert_skipped."
        )
        return ExpertStatus.SKIPPED
    logging.info(f"Biigle volume created for {drop_id}: id={volume_info.get('id')}")
    return ExpertStatus.UPLOADED


# ---------------------------------------------------------------------------
# Dynamic prerequisites for biigle-upload
# ---------------------------------------------------------------------------


def _biigle_prerequisites(
    args: argparse.Namespace, run_all: bool
) -> dict[str, str | list[str]]:
    """Returns the prerequisite *eligibility* condition for biigle-upload.

    Controls **when a drop becomes eligible**, not which data source the
    upload uses. Per-drop data-source selection lives in
    ``_run_biigle_upload_drop`` and always prefers Zooniverse data when
    it's present.

    Full-pipeline mode (``--zooniverse-*`` also passed, or run_all):
        ``citsci_status IN (complete, skipped)`` — wait for Zooniverse to
        finish (or be explicitly skipped) before pushing to BIIGLE.

    Biigle-direct mode (``--biigle-upload`` alone):
        ``ml_status = complete`` — don't wait for Zooniverse, advance any
        drop whose ML is done. If that drop also happens to be
        citsci_complete, ``_run_biigle_upload_drop`` will still take the
        Zooniverse path — eligibility is loosened, the data source isn't.
    """
    skip_zooniverse = not (
        run_all
        or getattr(args, "zooniverse_clips", False)
        or getattr(args, "zooniverse_sync", False)
    )
    if skip_zooniverse:
        return {"ml_status": MlStatus.COMPLETE}
    return {"citsci_status": [CitSciStatus.COMPLETE, CitSciStatus.SKIPPED]}


# ---------------------------------------------------------------------------
# Stage registry — add a new pipeline stage by adding ONE entry here
# ---------------------------------------------------------------------------

STAGES: list = [
    # ── Admin / maintenance — off the happy path, call explicitly ────────────
    GlobalStage("ingest", "Metadata ingestion", _run_ingest, run_in_all=False),
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
        "legacy-experts",
        "Historical backfill: legacy expert annotations CSV from S3",
        ingest_legacy_expert_annotations,
        run_in_all=False,
    ),
    GlobalStage(
        "legacy-zooniverse",
        "Historical backfill: legacy Zooniverse classification CSVs",
        run_legacy_zooniverse_backfill,
        run_in_all=False,
    ),
    GlobalStage(
        "db-refresh",
        "Reconcile DB status with on-disk artifacts and live Zooniverse/Biigle API state",
        run_db_refresh,
        run_in_all=False,
    ),
    # ── Data pipeline — runs by default ─────────────────────────────────────
    GlobalStage("ml", "ML inference + post-processing", _run_ml),
    DropStage(
        "zooniverse-clips",
        "Zooniverse clip extraction",
        _run_zooniverse_clips_drop,
        section="citsci_status",
        input_statuses=[CitSciStatus.PENDING],
        prerequisites={"ml_status": MlStatus.COMPLETE},
    ),
    GlobalStage(
        "zooniverse-sync",
        "Zooniverse volunteer sync-back (per-subject-set)",
        _run_zooniverse_sync,
    ),
    DropStage(
        "biigle-upload",
        "Biigle frame extraction + upload",
        _run_biigle_upload_drop,
        section="expert_status",
        input_statuses=[ExpertStatus.PENDING],
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
        "--dry-run",
        action="store_true",
        help="On --retrain --data-prep, run the fast part (flatten + maps + "
        "split summary) and stop before the slow assembly. Produces the maps "
        "scripts/wip/suggest_val_drops.py needs to plan the val split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="On --zooniverse-sync: re-fetch from API even if raw CSV already "
        "exists on disk. On --ml --survey: also reset the survey's "
        "ml_complete/ml_error drops to ml_ready and re-run inference on them "
        "(same-model outputs are overwritten; a new model writes alongside).",
    )
    parser.add_argument(
        "--survey",
        metavar="SURVEY_ID",
        help="On --ml: run inference on every ml_ready drop in this survey "
        "(bypasses limit_processing). Drops still awaiting video confirmation "
        "are reported; run --check-arrivals first to advance them.",
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
        if s.flag == "ml" and args.survey:
            return replace(
                s,
                fn=functools.partial(_run_ml, survey_id=args.survey, force=args.force),
            )
        if s.flag == "zooniverse-sync":
            return replace(
                s, fn=functools.partial(_run_zooniverse_sync, force=args.force)
            )
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
                    dry_run=args.dry_run,
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
