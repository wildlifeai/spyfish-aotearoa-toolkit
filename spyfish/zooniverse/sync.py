"""
Per-subject-set Zooniverse sync.

For each drop at ``citsci_clips_uploaded``, check whether its Zooniverse
clips subject set is fully retired. If yes, fetch all classifications for
that set, parse, aggregate, write the raw CSV + MaxN CSV, ingest the
annotations into ``spyfish_annotations.db``, and advance the drop to
``citsci_complete``.

Idempotent: if the per-drop raw CSV already exists, the Panoptes fetch is
skipped and classification rows are re-read from disk. Pass ``force=True``
to bypass the cache and re-fetch from the API.

Entry point is ``sync_zooniverse_drops`` — wired into ``run_pipeline.py``
behind the ``--zooniverse-sync`` flag.
"""

import logging

import pandas as pd

from spyfish.config.base import CitSciStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.zooniverse.parse_classifications import (
    aggregate_by_subject_species,
    connect_to_zooniverse,
    fetch_classifications_for_set,
    ingest_zooniverse_annotations,
    parse_classifications,
    subject_completion_from_api,
    write_empty_zooniverse_maxn_csv,
    write_zooniverse_maxn_csv,
)


def _sync_one_drop(drop_id: str, completion: pd.DataFrame, force: bool) -> bool:
    """Process one drop. Returns True if the drop advanced to citsci_complete.

    The empty-classifications case (volunteers retired the subjects as
    all-NOTHINGHERE) still produces an empty MaxN CSV and ingests, so the
    drop progresses rather than getting stuck at clips_uploaded.
    """
    clips_rows = completion[
        (completion["drop_id"] == drop_id) & (completion["subject_set_type"] == "clips")
    ]
    if clips_rows.empty:
        logging.info(
            f"  {drop_id}: no clips subject set found in Zooniverse "
            "(may not have been uploaded yet) — skipping."
        )
        return False

    row = clips_rows.iloc[0]
    if not row["fully_complete"]:
        logging.info(
            f"  {drop_id}: {int(row['retired'])}/{int(row['total'])} subjects retired "
            f"({row['pct_retired']:.0f}%) — not ready yet."
        )
        return False

    subject_set_id = row["subject_set_id"]
    raw_csv = config.get_zooniverse_raw_csv_path(drop_id)

    if not force and raw_csv.exists():
        logging.info(
            f"  {drop_id}: raw CSV found — re-aggregating from disk "
            "(pass --force to re-fetch from API)."
        )
        parsed_df = pd.read_csv(raw_csv)
    else:
        if force:
            logging.info(f"  {drop_id}: --force — re-fetching from API.")
        raw = fetch_classifications_for_set(subject_set_id)
        if not raw:
            logging.info(
                f"  {drop_id}: no classifications returned — "
                "writing empty MaxN CSV (all-NOTHINGHERE)."
            )
            write_empty_zooniverse_maxn_csv(drop_id)
            ingest_zooniverse_annotations(drop_id)
            return True
        parsed_df = parse_classifications(raw)
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        parsed_df.to_csv(raw_csv, index=False)
        logging.info(f"  {drop_id}: raw CSV → {raw_csv} ({len(parsed_df)} rows)")

    aggregated_df = aggregate_by_subject_species(parsed_df)

    if aggregated_df.empty:
        write_empty_zooniverse_maxn_csv(drop_id)
    else:
        write_zooniverse_maxn_csv(aggregated_df)

    ingest_zooniverse_annotations(drop_id)
    logging.info(f"  {drop_id}: → {CitSciStatus.COMPLETE}")
    return True


def sync_zooniverse_drops(force: bool = False) -> None:
    """Run the per-subject-set Zooniverse sync for every eligible drop.

    Eligibility: ``citsci_status = citsci_clips_uploaded`` AND
    ``ingest_status = ok`` (enforced by ``get_deployments_eligible``).
    Per-drop completion check is one batch Panoptes call up front;
    the loop then dispatches to a per-drop fetch only for fully-retired
    sets without a cached raw CSV.
    """
    db = DatabaseManager()
    eligible = db.get_deployments_eligible(
        "citsci_status", [CitSciStatus.CLIPS_UPLOADED]
    )
    drop_ids = [record["drop_id"] for record in eligible]

    if not drop_ids:
        logging.info("No drops eligible for zooniverse-sync.")
        return

    logging.info(f"{len(drop_ids)} drop(s) eligible for zooniverse-sync.")

    connect_to_zooniverse()
    completion = subject_completion_from_api()

    if completion is None or completion.empty:
        logging.info("Completion data unavailable from Panoptes — nothing to do.")
        return

    for drop_id in drop_ids:
        _sync_one_drop(drop_id, completion, force)
