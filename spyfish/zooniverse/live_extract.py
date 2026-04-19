"""
Live Zooniverse sync. Entry point: ``run_live_zooniverse_sync()``, runnable via
``python -m spyfish.zooniverse.live_extract``.

Fetches retired classifications from the Panoptes API, aggregates by species,
and writes a per-drop MaxN CSV. Only drop_ids whose subject set is fully
complete are exported. Historical CSV backfill lives in ``legacy_extract.py``.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.utils import seconds_to_time
from spyfish.zooniverse.parse_classifications import (
    aggregate_by_subject_species,
    connect_to_zooniverse,
    fetch_classifications,
    parse_classifications,
    sample_nothing_here_clips,
    subject_completion_from_api,
)

_LAST_RUN_KEY = "zooniverse_last_run_at"


def _resolve_since(since_arg: str | None, db: DatabaseManager) -> str | None:
    """Resolve --since to an ISO 8601 string, or None for full backfill."""
    if since_arg == "all":
        logging.info("--since all: forcing full backfill.")
        return None

    if since_arg is None or since_arg == "auto":
        ts = db.get_metadata(_LAST_RUN_KEY)
        if ts:
            logging.info(f"--since auto: using {_LAST_RUN_KEY}={ts} from DB")
            return ts
        logging.info("No last run recorded in DB — full backfill.")
        return None

    return since_arg


# Mirrors of helpers in legacy_extract.py — duplicated so live and legacy
# are each self-contained. Update both copies together if the MaxN format
# changes.


def _filter_to_complete_drops(
    df: pd.DataFrame, completion_df: pd.DataFrame
) -> pd.DataFrame:
    """Keep only rows whose drop_id is fully retired; log skipped partials."""
    complete = set(
        completion_df.loc[completion_df["fully_complete"], "drop_id"].dropna()
    )
    partial = completion_df[~completion_df["fully_complete"]].sort_values(
        "pct_retired", ascending=False
    )
    for _, row in partial.iterrows():
        logging.info(
            f"  Skipping {row['drop_id']} — {row['retired']}/{row['total']} subjects retired "
            f"({row['pct_retired']:.0f}%)"
        )

    mask = df["drop_id"].isin(complete)
    n_before = df["drop_id"].nunique()
    n_after = df.loc[mask, "drop_id"].nunique()
    logging.info(
        f"Completion gate: {n_after}/{n_before} drop_ids fully complete and ready for export."
    )
    return df[mask].copy()


def _export_maxn_csvs(aggregated_df: pd.DataFrame) -> None:
    """Write one MaxN CSV per drop_id to ``config.get_zooniverse_maxn_csv_path(drop_id)``."""
    matched = aggregated_df[aggregated_df["drop_id"].notna()].copy()
    if matched.empty:
        logging.warning("No matched subjects to export as MaxN CSVs.")
        return

    for drop_id, grp in matched.groupby("drop_id"):
        rows = []
        for _, row in grp.iterrows():
            rows.append(
                {
                    config.drop_id_column: drop_id,
                    config.csv_scientific_name_column: row["species"],
                    config.csv_maxn_time_column: seconds_to_time(row["mode_seconds"]),
                    config.csv_max_interval_column: row.get("mode_count", 0),
                    config.csv_annotated_by_column: "citsci",
                    config.csv_interval_annotation_column: config.clip_length,
                    config.csv_confidence_agreement_column: round(
                        row["agreement_pct"] / 100, 4
                    ),
                    config.csv_maxn_time_seconds_column: row["mode_seconds"],
                }
            )

        out_path = config.get_zooniverse_maxn_csv_path(drop_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        (
            pd.DataFrame(rows)
            .sort_values(config.csv_maxn_time_column)
            .to_csv(out_path, index=False)
        )
        logging.info(f"MaxN CSV → {out_path} ({len(rows)} rows)")


def run_live_zooniverse_sync(
    since: str | None = None, limit: int | None = None
) -> None:
    """
    End-to-end live Zooniverse ingestion: API fetch → parse → aggregate →
    completion gate → export MaxN CSVs.

    Args:
        since: ISO 8601 date, 'auto' (DB last-run), 'all' (full backfill),
               or None (defaults to 'auto').
        limit: process only the last N classifications (for testing).
    """
    db = DatabaseManager()
    run_started_at = datetime.now(timezone.utc).isoformat()

    resolved_since = _resolve_since(since, db)
    connect_to_zooniverse()
    raw = fetch_classifications(since=resolved_since)
    logging.info("Fetching subject set completion from API...")
    completion_df = subject_completion_from_api()

    if not raw:
        logging.info("No classifications found. Nothing to do.")
        db.set_metadata(_LAST_RUN_KEY, run_started_at)
        return

    if limit:
        raw = raw[-limit:]
        logging.info(f"--limit {limit}: using last {len(raw)} classifications.")

    parsed_df = parse_classifications(raw)
    aggregated_df = aggregate_by_subject_species(parsed_df)

    if aggregated_df.empty:
        logging.info("No rows passed the min_votes filter. Done.")
        db.set_metadata(_LAST_RUN_KEY, run_started_at)
        return

    if completion_df is not None and not completion_df.empty:
        aggregated_df = _filter_to_complete_drops(aggregated_df, completion_df)
    else:
        logging.info("Completion gate skipped.")

    if aggregated_df.empty:
        logging.info("No fully-complete drop_ids to export. Done.")
        db.set_metadata(_LAST_RUN_KEY, run_started_at)
        return

    nothing_here_df = sample_nothing_here_clips(aggregated_df)
    if not nothing_here_df.empty:
        logging.info(
            f"NOTHINGHERE sampling: {len(nothing_here_df)} subjects selected "
            f"across {nothing_here_df['drop_id'].nunique()} drops."
        )

    export_df = aggregated_df[~aggregated_df["suspicious_minority_find"]].copy()
    _export_maxn_csvs(export_df)

    db.set_metadata(_LAST_RUN_KEY, run_started_at)
    logging.info("Done. Run frame extraction independently on the resulting MaxN CSVs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse live Zooniverse classifications into per-drop MaxN CSVs."
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="DATE_OR_AUTO",
        help=(
            "ISO 8601 date to fetch classifications since (e.g. 2024-06-01), "
            "'auto' to read the last-run timestamp from the pipeline DB, "
            "or 'all' for a full backfill. Defaults to 'auto'."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the last N classifications (for testing).",
    )
    args = parser.parse_args()
    run_live_zooniverse_sync(since=args.since, limit=args.limit)


if __name__ == "__main__":
    main()
