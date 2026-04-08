"""
Parse Zooniverse volunteer classifications → MaxN CSVs.

Fetches retired classifications via the Panoptes API, aggregates by species,
and writes a per-drop MaxN CSV in the same format as ML MaxN CSVs so that
downstream frame extraction and Biigle ingestion work identically.

Only drop_ids whose subject set is **fully complete** (every uploaded subject
retired) are exported. Partially-retired drops are logged but skipped —
they will be picked up on the next run once volunteers finish.

Frame extraction is a separate step — run it independently after this script.

Usage:
    # Full backfill (first run, or --since all)
    python parse_zooniverse_classifications.py

    # Only fetch classifications newer than a given date
    python parse_zooniverse_classifications.py --since 2024-06-01

    # Auto-detect: use the most recent created_at seen in previous runs
    python parse_zooniverse_classifications.py --since auto

    # CSV backfill with completion check (pass subjects export alongside)
    python parse_zooniverse_classifications.py --from-csv --subjects-csv path/to/subjects.csv

Phases:
    0  Fetch from Panoptes API (or load from CSV backfill)
    1  Parse & resolve drop_ids
    2  Aggregate by (subject_id, species), apply min_votes
    2b Check subject-set completion — skip drops that are not fully retired
    3  NOTHINGHERE sampling
    4  Export Zooniverse MaxN CSV per drop_id (fully-complete drops only)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.utils import seconds_to_time
from spyfish.zooniverse.parse_classifications import (
    aggregate_by_subject_species,
    connect_to_zooniverse,
    fetch_classifications,
    get_all_db_drop_ids,
    load_classifications_from_csv,
    parse_classifications,
    sample_nothing_here_clips,
    subject_completion_from_api,
    subject_completion_from_csv,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ZOONIVERSE_DIR = config.data_quality_dir.parent / "zooniverse"
_LEGACY_CSV_DIR = _ZOONIVERSE_DIR / "legacy_classifications"
_LEGACY_SUBJECTS_DIR = _ZOONIVERSE_DIR / "legacy_subjects"
_LAST_RUN_FILE = _ZOONIVERSE_DIR / "last_run.json"
_REVIEW_CSV = _ZOONIVERSE_DIR / "zooniverse_review.csv"


# ── --since handling ──────────────────────────────────────────────────────────


def _resolve_since(since_arg: str | None) -> str | None:
    """
    Resolve --since argument to an ISO 8601 string, or None for full backfill.

    "auto"  → reads last_run.json; falls back to None (full backfill) if missing.
    "all"   → None (force full backfill).
    None    → also treated as "auto" (default behaviour).
    """
    if since_arg == "all":
        logging.info("--since all: forcing full backfill.")
        return None

    if since_arg is None or since_arg == "auto":
        if _LAST_RUN_FILE.exists():
            try:
                data = json.loads(_LAST_RUN_FILE.read_text())
                ts = data.get("last_run_at")
                if ts:
                    logging.info(
                        f"--since auto: using last_run_at={ts} from {_LAST_RUN_FILE}"
                    )
                    return ts
            except Exception as e:
                logging.warning(
                    f"Could not read {_LAST_RUN_FILE}: {e}. Falling back to full backfill."
                )
        logging.info("No last_run.json found — full backfill.")
        return None

    # Explicit ISO date/datetime string
    return since_arg


def _write_last_run(run_at: str) -> None:
    _ZOONIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    _LAST_RUN_FILE.write_text(json.dumps({"last_run_at": run_at}, indent=2))
    logging.info(f"Wrote last_run_at={run_at} to {_LAST_RUN_FILE}")


# ── Phase 2b — Completion gate ────────────────────────────────────────────────


def _filter_to_complete_drops(
    df: pd.DataFrame, completion_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Keep only rows whose drop_id is fully complete (all subjects retired).

    Logs partial drops so the operator knows they will be picked up later.
    Returns a filtered copy; raises no errors on empty result.
    """
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


# ── Phase 4 — MaxN CSV export ─────────────────────────────────────────────────


def export_maxn_csvs(aggregated_df: pd.DataFrame) -> None:
    """
    Write one MaxN CSV per drop_id in the same column format as ML MaxN CSVs:
        DropID, ScientificName, TimeOfMax, MaxInterval, AnnotatedBy,
        IntervalAnnotation, ConfidenceAgreement, TimeOfMaxAbsSeconds

    Only rows that passed the min_votes filter are included.
    Output path: {drop_annotations_dir}/{drop_id}_zooniverse_maxn.csv
    """
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

        out_dir = config.get_drop_annotations_dir(drop_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{drop_id}_zooniverse_maxn.csv"
        (
            pd.DataFrame(rows)
            .sort_values(config.csv_maxn_time_column)
            .to_csv(out_path, index=False)
        )
        logging.info(f"MaxN CSV → {out_path} ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Zooniverse classifications into per-drop MaxN CSVs."
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="DATE_OR_AUTO",
        help=(
            "ISO 8601 date to fetch classifications since (e.g. 2024-06-01), "
            "'auto' to use last_run.json, or 'all' for a full backfill. "
            "Defaults to 'auto'. Ignored when --from-csv is set."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the last N classifications (for testing).",
    )
    parser.add_argument(
        "--from-csv",
        nargs="*",
        metavar="CSV_PATH",
        help=(
            "Load from downloaded Zooniverse export CSV(s) instead of the API. "
            "Pass one or more paths, or no paths to use all CSVs in "
            f"{_LEGACY_CSV_DIR}."
        ),
    )
    parser.add_argument(
        "--subjects-csv",
        nargs="*",
        metavar="SUBJECTS_CSV_PATH",
        help=(
            "Zooniverse subjects export CSV(s) used for the completion gate when "
            "--from-csv is set. Pass one or more paths, or no paths to use all "
            f"CSVs in {_LEGACY_SUBJECTS_DIR}. "
            "If omitted and no files are found in the default dir, the completion "
            "gate is skipped for CSV backfills."
        ),
    )
    args = parser.parse_args()

    run_started_at = datetime.now(timezone.utc).isoformat()

    # ── Phase 0: Fetch ────────────────────────────────────────────────────────
    completion_df = None  # resolved in phase 0, applied in phase 2b

    if args.from_csv is not None:
        # --from-csv with no paths → default to legacy_classifications dir
        if args.from_csv:
            csv_paths = [Path(p) for p in args.from_csv]
        else:
            csv_paths = sorted(_LEGACY_CSV_DIR.glob("*.csv"))
            if not csv_paths:
                logging.error(
                    f"No CSVs found in {_LEGACY_CSV_DIR}. Pass paths explicitly."
                )
                sys.exit(1)
        logging.info(
            f"Loading from {len(csv_paths)} CSV(s): {[p.name for p in csv_paths]}"
        )
        raw = load_classifications_from_csv([str(p) for p in csv_paths])

        # Resolve subjects CSVs for completion gate
        if args.subjects_csv is not None:
            subjects_paths = (
                [Path(p) for p in args.subjects_csv]
                if args.subjects_csv
                else sorted(_LEGACY_SUBJECTS_DIR.glob("*.csv"))
            )
            if subjects_paths:
                logging.info(
                    f"Loading subjects from {len(subjects_paths)} file(s) for completion gate."
                )
                completion_df = subject_completion_from_csv(
                    [str(p) for p in subjects_paths]
                )
            else:
                logging.warning(
                    f"--subjects-csv given but no files found in {_LEGACY_SUBJECTS_DIR}. "
                    "Completion gate skipped — all matched drop_ids will be exported."
                )
        else:
            # Check default dir without --subjects-csv flag
            default_subjects = sorted(_LEGACY_SUBJECTS_DIR.glob("*.csv"))
            if default_subjects:
                logging.info(
                    f"Found {len(default_subjects)} subjects CSV(s) in {_LEGACY_SUBJECTS_DIR} — "
                    "using for completion gate."
                )
                completion_df = subject_completion_from_csv(
                    [str(p) for p in default_subjects]
                )
            else:
                logging.warning(
                    f"No subjects CSVs found in {_LEGACY_SUBJECTS_DIR}. "
                    "Completion gate skipped for CSV backfill — all matched drop_ids will be exported. "
                    f"To enable the gate, place a Zooniverse subjects export in {_LEGACY_SUBJECTS_DIR}."
                )
    else:
        since = _resolve_since(args.since)
        connect_to_zooniverse()
        raw = fetch_classifications(since=since)
        logging.info("Fetching subject set completion from API...")
        completion_df = subject_completion_from_api()

    if not raw:
        logging.info("No classifications found. Nothing to do.")
        _write_last_run(run_started_at)
        return

    if args.limit:
        raw = raw[-args.limit :]
        logging.info(f"--limit {args.limit}: using last {len(raw)} classifications.")

    # ── Phase 1: Parse ────────────────────────────────────────────────────────
    db_drop_ids = get_all_db_drop_ids()
    parsed_df = parse_classifications(raw, db_drop_ids)

    # ── Phase 2: Aggregate ────────────────────────────────────────────────────
    aggregated_df = aggregate_by_subject_species(parsed_df)

    # Write audit CSV (all rows, including suspicious minority finds)
    _ZOONIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(_REVIEW_CSV, index=False)
    logging.info(f"Audit log → {_REVIEW_CSV} ({len(aggregated_df)} rows)")

    if aggregated_df.empty:
        logging.info("No rows passed the min_votes filter. Done.")
        _write_last_run(run_started_at)
        return

    # ── Phase 2b: Completion gate ─────────────────────────────────────────────
    if completion_df is not None and not completion_df.empty:
        aggregated_df = _filter_to_complete_drops(aggregated_df, completion_df)
    else:
        logging.info("Completion gate skipped.")

    if aggregated_df.empty:
        logging.info("No fully-complete drop_ids to export. Done.")
        _write_last_run(run_started_at)
        return

    # ── Phase 3: NOTHINGHERE sampling ─────────────────────────────────────────
    nothing_here_df = sample_nothing_here_clips(aggregated_df)
    if not nothing_here_df.empty:
        nh_path = _ZOONIVERSE_DIR / "zooniverse_nothing_here_sample.csv"
        nothing_here_df.to_csv(nh_path, index=False)
        logging.info(f"NOTHINGHERE sample audit → {nh_path}")

    # ── Phase 4: MaxN CSV export ──────────────────────────────────────────────
    # Filter out suspicious minority finds before exporting MaxN CSVs
    export_df = aggregated_df[~aggregated_df["suspicious_minority_find"]].copy()
    export_maxn_csvs(export_df)

    # ── Write last_run timestamp ──────────────────────────────────────────────
    _write_last_run(run_started_at)
    logging.info("Done. Run frame extraction independently on the resulting MaxN CSVs.")


if __name__ == "__main__":
    main()
