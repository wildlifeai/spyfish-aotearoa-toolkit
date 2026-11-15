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

ML split for SLURM (explicit; a plain --ml does all three in one process):

    --download           CPU: fetch videos to media_dir (the survey's, or the next batch's)
    --ml-infer           GPU: inference only, ml_ready → ml_inferred
    --ml-post            CPU: MaxN + ingest + QA frames, ml_inferred → ml_complete
                         scripts/slurm/submit_ml.sh SURVEY chains them.

Admin / maintenance (always explicit — never run by default):

    --ingest             Refresh metadata from SharePoint CSVs on S3
    --check-arrivals     Cheap S3 poll for newly arrived videos
    --set-targets        Bulk-set pipeline stages from CSV
    --legacy-experts     Historical backfill: legacy expert annotations CSV
    --legacy-zooniverse  Historical backfill: legacy Zooniverse classifications
    --db-refresh         Reconcile DB status with on-disk + API state

Zooniverse validation (opt-in for now, public uploads; drops chosen by priority):

    --zooniverse-frames  Picked frames, every ML box drawn → vframes_{drop} (recall)
    --zooniverse-crops   One crop per ML box in those frames → crops_{drop} (precision)
                         Add --dry-run to render and stop, nothing uploaded.
    --zooniverse-validation-sync
                         Retired vframes_/crops_ sets → per-drop verdict CSVs,
                         *_uploaded → *_complete. --force re-fetches cached answers.

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
from datetime import datetime, timezone
from pathlib import Path

from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.biigle.upload_frames import (
    upload_frames_to_biigle,
    upload_frames_to_expert_survey_volume,
)
from spyfish.config.base import (
    CitSciStatus,
    ExpertStatus,
    MlStatus,
    ZooCropsStatus,
    ZooFramesStatus,
)
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import require_species_synonyms
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.extract_clips import extract_clips_from_selections
from spyfish.extraction.extract_frames import extract_frames_from_selections
from spyfish.extraction.select_frames import (
    select_frames,
    select_frames_from_zooniverse,
)
from spyfish.extraction.slate import extract_slate_frame
from spyfish.log_config import log_header
from spyfish.ml.run_inference import rerun_inference_on_extracted_frames
from spyfish.orchestrator.db_refresh import run_db_refresh
from spyfish.orchestrator.download import download_videos, select_download_drops
from spyfish.orchestrator.ingest import check_pending_arrivals, run_ingestion
from spyfish.orchestrator.legacy_extract import ingest_legacy_expert_annotations
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.orchestrator.retrain_runner import run_retraining
from spyfish.orchestrator.stage import DropStage, GlobalStage, StageRunner
from spyfish.storage.db_sync import sync_pipeline_results
from spyfish.zooniverse.legacy_extract import run_legacy_zooniverse_backfill
from spyfish.zooniverse.select_zooniverse_clips import process_zooniverse_clips
from spyfish.zooniverse.upload import (
    record_clip_uploads,
    upload_clips_to_zooniverse,
)
from spyfish.zooniverse.validation_frames import (
    locally_ready,
    run_validation_crops_drop,
    run_validation_frames_drop,
)

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


def _run_ml(
    survey_id: str | None = None, force: bool = False, post_process: bool = True
) -> None:
    """--ml, or with ``post_process=False`` its GPU half (--ml-infer), where
    drops stop at ml_inferred and nothing is ingested."""
    if post_process:
        # Only the ingest needs the synonyms map; the GPU half never ingests.
        require_species_synonyms()
    runner = MLRunner()
    targets = runner.get_inference_targets(survey_id=survey_id, force=force)

    if not targets:
        logging.info("No drops available for ML processing.")
        return

    # MaxN + QA frames are written per-drop inside run_inference_loop, before
    # each drop is marked ml_complete. finalize_batch_results is only the safety
    # net for any drops still stuck in ml_running after the loop exits, and it
    # scopes itself to runner.claimed_drop_ids, targets lost to a concurrent job
    # are not ours to mark ml_error.
    results = runner.run_inference_loop(targets, post_process=post_process)
    runner.finalize_batch_results(results)


# --ml split into three jobs so only the GPU-bound part runs on a GPU node.
# scripts/slurm/submit_ml.sh chains them with SLURM dependencies:
#   --download    (CPU)  S3 → media_dir, nothing claimed (generic: any stage's videos)
#   --ml-infer    (GPU)  ml_ready → ml_running → ml_inferred (raw CSV on disk)
#   --ml-post     (CPU)  ml_inferred → ml_complete (MaxN, ingest, QA frames)
# A plain --ml still does all three in one process.


def _run_download(survey_id: str | None = None) -> None:
    """Fetch videos to media_dir so later jobs find them on disk (CPU node).

    With --survey, every drop of that survey; without, the next --ml batch.
    See ``spyfish.orchestrator.download.select_download_drops``. Claims nothing
    and changes no status. --force is not needed here: the GPU job does the
    ml_ready reset, and in survey mode every video is fetched anyway.
    """
    db = DatabaseManager()
    drop_ids = select_download_drops(db, survey_id)
    if not drop_ids:
        logging.info("Nothing to download.")
        return
    result = download_videos(drop_ids, db=db)
    on_disk = result["downloaded"] + result["present"]
    logging.info(f"{len(on_disk)} video(s) on local disk.")
    _extract_slates(db, on_disk)


def _extract_slates(db: DatabaseManager, drop_ids: list[str]) -> None:
    """Slate JPEGs on the CPU download job, not the GPU inference job.

    The scan decodes up to ~120 opening frames per drop; in the three-job
    chain that is GPU time spent idle. `extract_slate_frame` skips a drop
    whose slate is already saved, so the same call kept in the inference loop
    (for plain --ml, and as the "slate before ML" guarantee) returns at once
    after this has run. One drop's failure never stops the others.
    """
    for drop_id in drop_ids:
        rec = db.get_deployment(drop_id) or {}
        key = str(rec.get("video_path") or "").strip()
        if not key:
            continue
        try:
            extract_slate_frame(
                drop_id,
                config.media_dir / Path(key).name,
                max_seconds=rec.get("sampling_start"),
            )
        except Exception as e:  # noqa: BLE001, a slate miss never fails the job
            logging.warning(f"{drop_id}: slate extraction failed, continuing: {e}")


def _run_ml_post(survey_id: str | None = None) -> None:
    """CPU half of --ml: MaxN + ingest + QA frames for every ml_inferred drop."""
    require_species_synonyms()
    runner = MLRunner()
    done = runner.run_post_processing(survey_id=survey_id)
    logging.info(f"{len(done)} drop(s) reached {MlStatus.COMPLETE}.")


def _run_zooniverse_sync(force: bool = False) -> None:
    """Wire to ``spyfish.zooniverse.sync.sync_zooniverse_drops``."""
    from spyfish.zooniverse.sync import sync_zooniverse_drops

    require_species_synonyms()
    sync_zooniverse_drops(force=force)


def _run_validation_sync(force: bool = False) -> None:
    """Wire to ``spyfish.zooniverse.validation_sync.sync_validation_drops``."""
    from spyfish.zooniverse.validation_sync import sync_validation_drops

    sync_validation_drops(force=force)


def _run_biigle_sync() -> None:
    require_species_synonyms()
    sync_biigle_annotations()


def _run_retrain(
    data_prep: bool = True,
    species: bool = True,
    dry_run: bool = False,
) -> None:
    run_retraining(
        data_prep=data_prep,
        species=species,
        auto_promote=True,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Per-drop stage functions — (drop_id: str) -> target section status str | None
# ---------------------------------------------------------------------------


def _run_zooniverse_clips_drop(drop_id: str) -> str | None:
    """Zooniverse clip selection + extraction + upload."""
    # MaxN of the model that ran on the drop: the pipeline model's when it
    # exists, else the newest model's (see config.resolve_ml_raw_csv).
    resolved = config.resolve_ml_raw_csv(drop_id)
    model_name = resolved[0] if resolved else config.pipeline_model_name
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
    record_clip_uploads(
        drop_id, clips_df, datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    return CitSciStatus.CLIPS_UPLOADED


def _run_biigle_upload_drop(drop_id: str, survey_volume: bool = False) -> str | None:
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
    timestamps where the ML raw CSV has no detections, so we re-run the
    pipeline model on each extracted JPEG and
    rebuild the COCO JSON before upload — see
    ``rerun_inference_on_extracted_frames``. Without that step the
    BIIGLE upload would carry blank frames and experts would annotate
    from scratch instead of correcting model boxes.
    """
    model_name = config.pipeline_model_name
    raw_csv = str(config.get_raw_csv_path(drop_id, model_name))
    video_path = str(config.get_video_path(drop_id))
    biigle_selections_path = config.get_biigle_selections_csv_path(drop_id)

    # Resolve the ML raw CSV for both paths. Prefer the pipeline model's;
    # fall back to the newest other model that ran on the drop (newest model
    # wins, by training stamp). A promotion or rename must not strand every
    # ml_complete drop at expert_pending: their detections are on disk under
    # the model that produced them, and that model's boxes are what the
    # expert corrects.
    ml_raw_csv: str | None = None
    if Path(raw_csv).exists():
        ml_raw_csv = raw_csv
    else:
        other_models = config.list_ml_raw_csvs(drop_id)
        if other_models:
            newest = max(other_models, key=config.model_name_stamp)
            ml_raw_csv = str(other_models[newest])
            logging.info(
                f"{drop_id}: no raw CSV for the pipeline model {model_name}; "
                f"using {Path(ml_raw_csv).name} (model {newest})."
            )

    db = DatabaseManager()
    deployment = db.get_deployment(drop_id)
    use_zooniverse = (
        deployment is not None
        and deployment.get("citsci_status") == CitSciStatus.COMPLETE
    )

    # Every drop that reaches this stage gets frames: an expert looking at the
    # blind ones is the only check on a drop where nobody (model or
    # volunteers) saw anything. So there is no skip here any more:
    #   - the model saw nothing → select_frames returns blind frames only;
    #   - volunteers saw nothing (empty Zooniverse MaxN) → ML selection, see TODO;
    #   - no sampling window, or no raw CSV for the ML path → not ready yet:
    #     return None so the drop stays expert_pending and is retried.
    if deployment is None or None in (
        deployment.get("sampling_start"),
        deployment.get("sampling_end"),
    ):
        logging.warning(f"{drop_id}: no sampling window yet; leaving expert_pending.")
        return None
    if use_zooniverse:
        logging.info(
            f"{drop_id}: citsci_complete — selecting frames from Zooniverse volunteer consensus."
        )
        try:
            select_frames_from_zooniverse(
                str(config.get_zooniverse_maxn_csv_path(drop_id)),
                str(biigle_selections_path),
                drop_id,
                ml_raw_csv_path=ml_raw_csv,
            )
        except ValueError as e:
            # TODO: decide what an all-"nothing here" citsci drop should send the
            # expert, once there is citsci data to look at. ML peaks + blind
            # frames for now; blind only, or a citsci-specific check (clips the
            # volunteers flagged as unclear), may turn out to be the better
            # second opinion. See claude_docs/todo.md.
            logging.info(f"{drop_id}: {e} Falling back to the ML selection.")
            use_zooniverse = False
    if not use_zooniverse:
        if ml_raw_csv is None:
            logging.warning(f"{drop_id}: no ML raw CSV yet; leaving expert_pending.")
            return None
        select_frames(ml_raw_csv, str(biigle_selections_path), drop_id)

    # Zooniverse-selected timestamps don't align with ML raw CSV detections,
    # so on that path we skip the COCO write here and let the rerun-inference
    # step write the COCO from a fresh inference pass.
    frames_df = extract_frames_from_selections(
        selections_csv_path=str(biigle_selections_path),
        video_path=video_path,
        raw_csv_path="" if use_zooniverse else ml_raw_csv,
        write_coco=not use_zooniverse,
    )

    if use_zooniverse:
        rerun_inference_on_extracted_frames(drop_id, frames_df)

    # Pooled mode: every drop of a survey shares one volume named for the
    # survey, so the expert reviews survey by survey instead of volume by
    # volume. Bound from --survey-volume in _patch_stage.
    if survey_volume:
        volume_info = upload_frames_to_expert_survey_volume(drop_id, frames_df)
    else:
        volume_info = upload_frames_to_biigle(drop_id=drop_id, frames_df=frames_df)
    logging.info(f"Biigle volume created for {drop_id}: id={volume_info.get('id')}")
    return ExpertStatus.UPLOADED


# ---------------------------------------------------------------------------
# Dynamic prerequisites for biigle-upload
# ---------------------------------------------------------------------------


def _has_sampling_window(rec: dict) -> bool:
    return None not in (rec.get("sampling_start"), rec.get("sampling_end"))


# Disk/DB-only readiness checks for DropStage.ready: drops that would return
# None are filtered out before the top-`limit_processing` cut, so they cannot
# hold every batch slot run after run. Each mirrors its step's own None rules.


def _clips_ready(rec: dict) -> bool:
    resolved = config.resolve_ml_raw_csv(rec["drop_id"])
    model = resolved[0] if resolved else config.pipeline_model_name
    return config.get_maxn_csv_path(rec["drop_id"], model).exists()


def _biigle_upload_ready(rec: dict) -> bool:
    if not _has_sampling_window(rec):
        return False
    if config.resolve_ml_raw_csv(rec["drop_id"]) is not None:
        return True
    return (
        rec.get("citsci_status") == CitSciStatus.COMPLETE
        and config.get_zooniverse_maxn_csv_path(rec["drop_id"]).exists()
    )


def _validation_ready(rec: dict) -> bool:
    return _has_sampling_window(rec) and locally_ready(rec["drop_id"])


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
    # The same stage in three parts, for CPU / GPU / CPU SLURM jobs
    # (scripts/slurm/submit_ml.sh). Off the default path: --ml covers it.
    GlobalStage(
        "download",
        "Fetch videos to media_dir: the survey's (--survey) or the next --ml "
        "batch's (CPU node, before any stage that reads video)",
        _run_download,
        run_in_all=False,
    ),
    GlobalStage(
        "ml-infer",
        "ML inference only: ml_ready → ml_inferred (GPU node)",
        functools.partial(_run_ml, post_process=False),
        run_in_all=False,
    ),
    GlobalStage(
        "ml-post",
        "ML post-processing only: ml_inferred → ml_complete (CPU node)",
        _run_ml_post,
        run_in_all=False,
    ),
    DropStage(
        "zooniverse-clips",
        "Zooniverse clip extraction",
        _run_zooniverse_clips_drop,
        section="citsci_status",
        input_statuses=[CitSciStatus.PENDING],
        prerequisites={"ml_status": MlStatus.COMPLETE},
        ready=_clips_ready,
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
        ready=_biigle_upload_ready,
    ),
    GlobalStage("biigle-sync", "Biigle annotation sync", _run_biigle_sync),
    # ── Zooniverse validation — opt-in until the pipeline runs routinely ────
    # Public uploads, so never part of a plain run (see claude_docs/todo.md).
    # ERROR drops are re-selected: the upload is idempotent per subject set.
    DropStage(
        "zooniverse-frames",
        "Zooniverse validation frames (every ML box drawn, missed-fish check)",
        run_validation_frames_drop,
        section=ZooFramesStatus.COLUMN,
        input_statuses=[ZooFramesStatus.PENDING, ZooFramesStatus.ERROR],
        prerequisites={MlStatus.COLUMN: MlStatus.COMPLETE},
        run_in_all=False,
        ready=_validation_ready,
    ),
    DropStage(
        "zooniverse-crops",
        "Zooniverse validation crops (one per ML box, species check)",
        run_validation_crops_drop,
        section=ZooCropsStatus.COLUMN,
        input_statuses=[ZooCropsStatus.PENDING, ZooCropsStatus.ERROR],
        prerequisites={MlStatus.COLUMN: MlStatus.COMPLETE},
        run_in_all=False,
        ready=_validation_ready,
    ),
    GlobalStage(
        "zooniverse-validation-sync",
        "Zooniverse validation sync-back (retired sets → verdict CSVs, *_complete)",
        _run_validation_sync,
        run_in_all=False,
    ),
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
        "--sync-only",
        action="store_true",
        help="Run no stage: do only the end-of-run S3 sync (both DBs + the "
        "deployment data dir) and exit. The CPU job that follows a "
        "--no-upload GPU job (scripts/slurm/submit_retrain.sh). Refused with "
        "--no-upload, --dry-run or any stage flag.",
    )
    parser.add_argument(
        "--data-prep",
        action="store_true",
        help="On --retrain, include the data prep step. If no step flags "
        "(--data-prep, --species) are passed, both run.",
    )
    parser.add_argument(
        "--species",
        action="store_true",
        help="On --retrain, include the species training step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do the safe part and stop before the slow or permanent one. "
        "On --retrain --data-prep: the fast part (flatten + maps + split "
        "summary), stopping before the slow assembly. "
        "On --zooniverse-frames / --zooniverse-crops: select and render, "
        "write the manifest, then stop. Nothing uploaded (the final S3 sync "
        "is skipped too), status unchanged.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="On --zooniverse-sync / --zooniverse-validation-sync: re-fetch "
        "from API even if the raw CSV / classifications cache already "
        "exists on disk. On --ml / --ml-infer with --survey: "
        "also reset the survey's ml_complete/ml_error/ml_inferred/ml_running "
        "drops to ml_ready and re-run inference "
        "on them (same-model outputs are overwritten; a new model writes "
        "alongside). ml_running drops are usually stranded by a killed job - do "
        "not use --force while another ML job is running on the same survey.",
    )
    parser.add_argument(
        "--survey",
        metavar="SURVEY_ID",
        help="Restrict processing to one survey (DropID prefix). On --ml / "
        "--download / --ml-infer / --ml-post: "
        "run on every eligible drop in this survey (bypasses "
        "limit_processing; drops still awaiting video confirmation are "
        "reported, run --check-arrivals first to advance them). On drop "
        "stages (--zooniverse-clips, --zooniverse-frames, --zooniverse-crops, "
        "--biigle-upload): only this survey's "
        "eligible drops are processed.",
    )
    parser.add_argument(
        "--survey-volume",
        action="store_true",
        help="On --biigle-upload: pool each survey's frames into ONE shared "
        "Biigle volume named '{survey_id} ML' instead of one volume per "
        "deployment. Combine with --survey to upload exactly one survey's "
        "worth.",
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
        require_species_synonyms()
        logging.info(f"Synonyms:  {config.species_synonyms_path}")
        return

    if args.sync_only:
        # No stage flag means "run all" to StageRunner, so this returns before
        # the runner is ever started.
        stage_flags = [
            f"--{s.flag}" for s in STAGES if getattr(args, s.flag.replace("-", "_"))
        ]
        if args.no_upload or args.dry_run:
            parser.error(
                "--sync-only does nothing but upload to S3: it cannot be "
                "combined with --no-upload or --dry-run"
            )
        if stage_flags:
            parser.error(
                f"--sync-only runs no stage: drop {' '.join(stage_flags)}, or "
                "drop --sync-only (every run without --no-upload ends with "
                "this sync)"
            )
        log_header("SYNC ONLY: RESULTS TO S3")
        if not sync_pipeline_results():
            logging.critical(
                "CRITICAL: S3 Sync failed. Pipeline state might be inconsistent on S3."
            )
            log_header("SYNC FAILED", character="═")
            sys.exit(1)
        log_header("SYNC COMPLETE (SYNCED TO S3)", character="═")
        return

    def _patch_stage(s):
        if s.flag in ("ml", "ml-infer"):
            # --force only acts with --survey (get_inference_targets).
            return replace(
                s,
                fn=functools.partial(s.fn, survey_id=args.survey, force=args.force),
            )
        if s.flag in ("download", "ml-post"):
            return replace(s, fn=functools.partial(s.fn, survey_id=args.survey))
        if s.flag == "biigle-upload" and args.survey_volume:
            return replace(
                s,
                fn=functools.partial(_run_biigle_upload_drop, survey_volume=True),
            )
        if args.dry_run and s.flag in ("zooniverse-frames", "zooniverse-crops"):
            return replace(s, fn=functools.partial(s.fn, dry_run=True))
        if s.flag == "zooniverse-sync":
            return replace(
                s, fn=functools.partial(_run_zooniverse_sync, force=args.force)
            )
        if s.flag == "zooniverse-validation-sync":
            return replace(
                s, fn=functools.partial(_run_validation_sync, force=args.force)
            )
        if s.flag == "set-targets":
            return replace(
                s, fn=functools.partial(_run_set_targets, push_s3=not args.no_upload)
            )
        if s.flag == "retrain":
            # Compose-style: no flags = all steps; any flag = only those steps.
            no_step_specified = not (args.data_prep or args.species)
            do_data_prep = args.data_prep or no_step_specified
            do_species = args.species or no_step_specified
            return replace(
                s,
                fn=functools.partial(
                    _run_retrain,
                    data_prep=do_data_prep,
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
    if args.no_upload or args.dry_run:
        # --dry-run promises nothing permanent, so it never uploads either.
        logging.info("No-upload or dry-run set: skipping final S3 sync.")
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
