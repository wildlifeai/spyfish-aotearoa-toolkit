"""
DB refresh: reconcile pipeline status columns and spyfish_annotations.db
with on-disk artifacts and live API state.

Run when the DB is out of sync with reality — e.g. after a fresh clone,
a DB nuke, or work done outside the pipeline. Safe to re-run: only sets
status forward on drops that have artifacts, never resets pipeline progress.

Three passes, cheapest first:

  1. ml          — scan local maxn CSV → ml_complete + re-ingest ML annotations
  2. zooniverse  — local MaxN CSV first; Panoptes API batch call if absent
  3. biigle      — local raw/maxn CSV first; Biigle API batch call if absent

Each pass is independent. Wire flags in run_pipeline.py to scope the run.
"""

import logging
from pathlib import Path

import pandas as pd

from spyfish.config.base import CitSciStatus, ExpertStatus, MlStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ok_drops_not_at(
    db: DatabaseManager, section: str, exclude: list[str]
) -> list[dict]:
    """All ingest_status=ok drops whose `section` value is not in `exclude`."""
    db.validate_column(section)
    placeholders = ", ".join(["?"] * len(exclude))
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT * FROM deployments "
            f"WHERE ingest_status = 'ok' AND {section} NOT IN ({placeholders}) "
            f"ORDER BY priority DESC",
            exclude,
        )
        return [dict(row) for row in cursor.fetchall()]


def _ingest_ml_from_csv(
    ann_db: AnnotationDatabaseManager,
    drop_id: str,
    maxn_path: Path,
    model_name: str,
) -> int:
    """Read a maxn CSV from disk and insert rows into spyfish_annotations.db."""
    maxn_df = pd.read_csv(maxn_path)
    sec_col = config.csv_maxn_time_seconds_column
    rows = [
        {
            "drop_id": drop_id,
            "scientific_name": row[config.csv_scientific_name_column],
            "time_of_max": row[config.csv_maxn_time_column],
            "time_of_max_seconds": row[sec_col] if sec_col in maxn_df.columns else None,
            "max_interval": row[config.csv_max_interval_column],
            "annotated_by": "ml",
            "interval_annotation": "",
            "confidence_agreement": row[config.csv_confidence_agreement_column],
            "external_id": model_name,
        }
        for _, row in maxn_df.iterrows()
    ]
    ann_db.clear_annotations(drop_id, "ml")
    if rows:
        ann_db.add_annotations(rows)
    return len(rows)


# ── Pass 1 — ML ───────────────────────────────────────────────────────────────


def refresh_ml(db: DatabaseManager, model_name: str) -> int:
    """
    Mark ml_complete for any ok drop that has a maxn CSV on disk.
    Re-ingests ML annotations from the existing CSV into spyfish_annotations.db.
    Returns count of drops updated.
    """
    drops = _ok_drops_not_at(db, "ml_status", [MlStatus.COMPLETE, MlStatus.SKIPPED])
    ann_db = AnnotationDatabaseManager()
    updated = 0

    for drop in drops:
        drop_id = drop["drop_id"]
        maxn_path = config.get_maxn_csv_path(drop_id, model_name)
        if not maxn_path.exists():
            continue
        n = _ingest_ml_from_csv(ann_db, drop_id, maxn_path, model_name)
        db.update_section_status(drop_id, "ml_status", MlStatus.COMPLETE)
        logging.info(
            f"  ml: {drop_id} → {MlStatus.COMPLETE} ({n} rows re-ingested from disk)"
        )
        updated += 1

    return updated


# ── Pass 2 — Zooniverse ───────────────────────────────────────────────────────


def refresh_zooniverse(db: DatabaseManager) -> int:
    """
    For ok drops not yet citsci_complete:
      - Zooniverse MaxN CSV on disk → ingest + citsci_complete (no API call).
      - No local CSV → one batch Panoptes API call; mark citsci_clips_uploaded
        for fully-retired drops so --zooniverse-sync fetches classifications.

    Returns count of drops updated.
    """
    from spyfish.zooniverse.parse_classifications import (
        connect_to_zooniverse,
        ingest_zooniverse_annotations,
        subject_completion_from_api,
    )

    drops = _ok_drops_not_at(
        db, "citsci_status", [CitSciStatus.COMPLETE, CitSciStatus.SKIPPED]
    )
    updated = 0
    needs_api: list[str] = []

    for drop in drops:
        drop_id = drop["drop_id"]
        maxn_path = config.get_zooniverse_maxn_csv_path(drop_id)
        if maxn_path.exists():
            ingest_zooniverse_annotations(drop_id)
            db.update_section_status(drop_id, "citsci_status", CitSciStatus.COMPLETE)
            logging.info(
                f"  zoo: {drop_id} → {CitSciStatus.COMPLETE} (MaxN CSV on disk)"
            )
            updated += 1
        else:
            needs_api.append(drop_id)

    if not needs_api:
        return updated

    logging.info(
        f"  zoo: {len(needs_api)} drop(s) have no local MaxN CSV — checking Panoptes API..."
    )
    connect_to_zooniverse()
    completion = subject_completion_from_api()

    if completion is None or completion.empty:
        logging.warning("  zoo: Panoptes returned no subject sets — skipping API pass.")
        return updated

    for drop_id in needs_api:
        clips_rows = completion[
            (completion["drop_id"] == drop_id)
            & (completion["subject_set_type"] == "clips")
        ]
        if clips_rows.empty:
            continue
        row = clips_rows.iloc[0]
        if row["fully_complete"]:
            db.update_section_status(
                drop_id, "citsci_status", CitSciStatus.CLIPS_UPLOADED
            )
            logging.info(
                f"  zoo: {drop_id} → {CitSciStatus.CLIPS_UPLOADED} "
                "(clips retired — run --zooniverse-sync to fetch classifications)"
            )
            updated += 1
        else:
            logging.info(
                f"  zoo: {drop_id}: {int(row['retired'])}/{int(row['total'])} subjects "
                f"retired ({row['pct_retired']:.0f}%) — not yet complete"
            )

    return updated


# ── Pass 3 — Biigle ───────────────────────────────────────────────────────────


def refresh_biigle(db: DatabaseManager) -> int:
    """
    For ok drops at expert_pending:
      - Biigle expert raw/maxn CSV on disk → mark expert_uploaded so
        --biigle-sync re-ingests annotations and advances to expert_complete.
      - No local CSV → one batch Biigle API call; find volumes by name
        ("{drop_id} — ML frames"); mark expert_uploaded for drops with a match.

    Returns count of drops updated.
    """
    from spyfish.biigle.biigle_handler import BiigleHandler

    drops = _ok_drops_not_at(
        db,
        "expert_status",
        [ExpertStatus.UPLOADED, ExpertStatus.COMPLETE, ExpertStatus.SKIPPED],
    )
    updated = 0
    needs_api: list[str] = []

    for drop in drops:
        drop_id = drop["drop_id"]
        has_local = (
            config.get_biigle_expert_raw_csv_path(drop_id).exists()
            or config.get_biigle_expert_maxn_csv_path(drop_id).exists()
        )
        if has_local:
            db.update_section_status(drop_id, "expert_status", ExpertStatus.UPLOADED)
            logging.info(
                f"  biigle: {drop_id} → {ExpertStatus.UPLOADED} "
                "(CSV on disk — run --biigle-sync to ingest annotations)"
            )
            updated += 1
        else:
            needs_api.append(drop_id)

    if not needs_api:
        return updated

    logging.info(
        f"  biigle: {len(needs_api)} drop(s) have no local CSV — checking Biigle API..."
    )
    handler = BiigleHandler()
    # Volumes may live in either the in-progress project or the done project,
    # so reconcile against both.
    all_volumes = handler.get_volumes(
        config.biigle_upload_project_id
    ) + handler.get_volumes(config.biigle_done_project_id)

    # Build lookup: drop_id → first matching volume. Names follow "{drop_id} — ML frames".
    needs_api_set = set(needs_api)
    volume_by_drop: dict[str, dict] = {}
    for v in all_volumes:
        name = v.get("name", "")
        for drop_id in needs_api_set:
            if drop_id in name and drop_id not in volume_by_drop:
                volume_by_drop[drop_id] = v

    for drop_id in needs_api:
        v = volume_by_drop.get(drop_id)
        if v:
            db.update_section_status(drop_id, "expert_status", ExpertStatus.UPLOADED)
            logging.info(
                f"  biigle: {drop_id} → {ExpertStatus.UPLOADED} "
                f"(volume '{v['name']}' found — run --biigle-sync to ingest annotations)"
            )
            updated += 1

    return updated


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_db_refresh(
    ml: bool = True,
    zooniverse: bool = True,
    biigle: bool = True,
) -> None:
    """
    Reconcile pipeline DB status with artifact reality.
    Runs all three passes by default; pass ml/zooniverse/biigle=False to scope.
    Finishes with sync_annotation_counts() to refresh denormalized count columns.
    """
    from spyfish.log_config import log_header

    db = DatabaseManager()
    model_name = Path(config.pipeline_model_path).stem
    total = 0

    if ml:
        log_header("DB Refresh — ML")
        n = refresh_ml(db, model_name)
        logging.info(f"ML pass: {n} drop(s) updated.")
        total += n

    if zooniverse:
        log_header("DB Refresh — Zooniverse")
        n = refresh_zooniverse(db)
        logging.info(f"Zooniverse pass: {n} drop(s) updated.")
        total += n

    if biigle:
        log_header("DB Refresh — Biigle")
        n = refresh_biigle(db)
        logging.info(f"Biigle pass: {n} drop(s) updated.")
        total += n

    logging.info(f"DB refresh complete: {total} total status update(s).")
    db.sync_annotation_counts()
