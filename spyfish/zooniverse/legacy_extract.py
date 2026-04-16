"""
Historical Zooniverse backfill. Entry point: ``run_legacy_zooniverse_backfill()``,
triggered from ``run_pipeline.py --legacy``. Core must not import from this module.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.utils import seconds_to_time
from spyfish.zooniverse.parse_classifications import (
    aggregate_by_subject_species,
    get_all_db_drop_ids,
    ingest_zooniverse_annotations,
    parse_classifications,
    sample_nothing_here_clips,
)
from spyfish.zooniverse.subject_keys import SubjectKeys

# Map every legacy / variant key we've ever seen to its current equivalent.
# Maintained here so the strict core parser only knows about current keys.
# When you discover a new variant in production data, add it here, not in core.
_LEGACY_KEY_MAP: dict[str, str] = {
    # video filename
    "video_filename": SubjectKeys.VIDEO_FILENAME,
    "#video_filename": SubjectKeys.VIDEO_FILENAME,
    # upl seconds
    "upl_seconds": SubjectKeys.UPL_SECONDS,
    "#upl_seconds": SubjectKeys.UPL_SECONDS,
    "#UplAbsSeconds": SubjectKeys.UPL_SECONDS,
    # subject type
    "SubjectType": SubjectKeys.SUBJECT_TYPE,
    "subject_type": SubjectKeys.SUBJECT_TYPE,
    "#subject_type": SubjectKeys.SUBJECT_TYPE,
    "Subject_type": SubjectKeys.SUBJECT_TYPE,
    # time of max
    "TimeOfMaxAbsSeconds": SubjectKeys.TIME_OF_MAX,
    # site name
    "#SiteID": SubjectKeys.SITE_NAME,
    "site_name": SubjectKeys.SITE_NAME,
    # link to reserve
    "!LinkToMarineReserve": SubjectKeys.LINK_TO_RESERVE,
    "link_to_reserve": SubjectKeys.LINK_TO_RESERVE,
    # event date
    "event_date": SubjectKeys.EVENT_DATE,
}


def _normalize_legacy_metadata(meta: dict) -> dict:
    """Add canonical keys for any legacy-named values present in `meta`.

    Returns a copy with canonical aliases added. Existing canonical keys are
    never overwritten — they take precedence over legacy variants.
    """
    out = dict(meta)
    for legacy_key, canonical_key in _LEGACY_KEY_MAP.items():
        if canonical_key not in out and legacy_key in meta:
            out[canonical_key] = meta[legacy_key]
    return out


# {RESERVE}_{SITE}_{DD}_{MM}_{YYYY}  e.g. AHE_062_25_04_2022
_LEGACY_DMY_PATTERN = re.compile(r"^([A-Z]{2,4})_(\d{3})_(\d{2})_(\d{2})_(\d{4})$")
# {RESERVE}_{SITE}_{YYYY}_{MM}_{DD}  e.g. KPT_012_2021_06_03
_LEGACY_YMD_PATTERN = re.compile(r"^([A-Z]{2,4})_(\d{3})_(\d{4})_(\d{2})_(\d{2})$")
_REPLICATE_SUFFIX_RE = re.compile(r"^(.+_)(\d+)$")
_NEW_FORMAT_PATTERN = re.compile(r"^[A-Z]+_\d{8}_BUV_.*")


def build_legacy_prefix_index(db_drop_ids: list[str]) -> dict[str, list[str]]:
    """
    Group canonical drop_ids by their pre-replicate prefix for O(1) lookups.

    The prefix key includes the trailing underscore before the replicate
    suffix so site numbers cannot prefix-collide (``062`` vs ``0621``).
    Drop_ids without a trailing ``_<digits>`` replicate are skipped — they
    cannot be hit by the query keys this resolver builds.

    Each bucket is sorted so ``bucket[-1]`` yields the highest replicate
    under lexicographic ordering.
    """
    index: dict[str, list[str]] = {}
    for drop_id in db_drop_ids:
        m = _REPLICATE_SUFFIX_RE.match(drop_id)
        if not m:
            continue
        prefix = m.group(1)
        index.setdefault(prefix, []).append(drop_id)
    for bucket in index.values():
        bucket.sort()
    return index


def resolve_legacy_drop_id(
    stem: str, legacy_index: dict[str, list[str]]
) -> Optional[str]:
    """
    Resolve a legacy filename stem to the highest-replicate canonical drop_id.

    Returns ``None`` if the stem does not match either legacy pattern
    (DMY or YMD) or if no matching canonical drop_id exists in the DB.
    """
    m = _LEGACY_DMY_PATTERN.match(stem)
    if m:
        reserve, site, dd, mm, yyyy = m.groups()
        date_str = f"{yyyy}{mm}{dd}"
    else:
        m = _LEGACY_YMD_PATTERN.match(stem)
        if m:
            reserve, site, yyyy, mm, dd = m.groups()
            date_str = f"{yyyy}{mm}{dd}"
        else:
            return None

    prefix = f"{reserve}_{date_str}_BUV_{reserve}_{site}_"
    matches = legacy_index.get(prefix)
    if not matches:
        return None
    return matches[-1]


def load_classifications_from_csv(csv_paths: list[str]) -> list[dict]:
    """Load Zooniverse classification CSV exports into the same dict shape as fetch_classifications()."""
    records = []
    for path in csv_paths:
        logging.info(f"Loading classifications from {path}")
        df = pd.read_csv(path, dtype=str)
        logging.info(f"  {len(df)} rows")

        for _, row in df.iterrows():
            raw_sd = row.get("subject_data", "")
            try:
                subject_data_map = json.loads(raw_sd) if pd.notna(raw_sd) else {}
            except (json.JSONDecodeError, TypeError):
                continue

            raw_ids = row.get("subject_ids", "")
            subject_ids = (
                [s.strip() for s in str(raw_ids).split(";")]
                if pd.notna(raw_ids)
                else []
            )
            if not subject_ids:
                continue

            subject_id = subject_ids[0]
            subject_entry = subject_data_map.get(subject_id, {})
            subject_metadata = _normalize_legacy_metadata(
                subject_entry.get("metadata", subject_entry)
            )

            raw_anns = row.get("annotations", "")
            try:
                annotations = json.loads(raw_anns) if pd.notna(raw_anns) else []
            except (json.JSONDecodeError, TypeError):
                annotations = []

            records.append(
                {
                    "classification_id": row.get("classification_id"),
                    "created_at": row.get("created_at"),
                    "user_name": row.get("user_name"),
                    "user_id": row.get("user_id"),
                    "annotations": annotations,
                    "subject_id": subject_id,
                    "subject_set_id": None,  # not in CSV export
                    "workflow_id": row.get("workflow_id"),
                    "subject_metadata": subject_metadata,
                    "subject_locations": [],  # not in CSV export
                }
            )

    logging.info(f"Loaded {len(records)} classifications from {len(csv_paths)} CSV(s).")
    return records


def subject_completion_from_csv(subjects_csv_paths: list[str]) -> pd.DataFrame:
    """
    Per drop_id retirement completion from Zooniverse subjects CSV exports.

    A drop_id is ``fully_complete`` when every one of its uploaded subjects
    is retired. Resolves both canonical new-format filenames and legacy
    date-pattern filenames via the legacy prefix index.

    Returns:
        DataFrame with columns: drop_id, total, retired, pct_retired, fully_complete.
    """
    legacy_index = build_legacy_prefix_index(get_all_db_drop_ids())

    def _resolve_from_meta(meta_str: str) -> Optional[str]:
        try:
            m = json.loads(meta_str)
        except Exception:
            return None
        vf = (
            m.get("#VideoFilename")
            or m.get("video_filename")
            or m.get("#video_filename")
            or ""
        )
        if not vf:
            return None
        stem = Path(vf).stem
        # New format: stem is the drop_id directly. No DB validation here —
        # we're just grouping for completion counts, not verifying existence.
        if _NEW_FORMAT_PATTERN.match(stem):
            return stem
        return resolve_legacy_drop_id(stem, legacy_index)

    frames = []
    for path in subjects_csv_paths:
        df = pd.read_csv(path, dtype=str)
        df["drop_id"] = df["metadata"].apply(_resolve_from_meta)
        df["is_retired"] = df["retired_at"].notna()
        frames.append(df[["subject_id", "drop_id", "is_retired", "subject_set_id"]])

    combined = pd.concat(frames, ignore_index=True)

    summary = (
        combined.groupby("drop_id")
        .agg(total=("subject_id", "count"), retired=("is_retired", "sum"))
        .reset_index()
    )
    summary["pct_retired"] = (summary["retired"] / summary["total"] * 100).round(1)
    summary["fully_complete"] = summary["retired"] == summary["total"]

    return summary.sort_values("pct_retired", ascending=False).reset_index(drop=True)


# ── Parse post-processor (Option C: run core strict, then patch legacy rows) ─


def parse_legacy_classifications(
    raw_classifications: list[dict],
    db_drop_ids: list[str],
) -> pd.DataFrame:
    """
    Parse historical Zooniverse classifications with legacy filename support.

    Runs core's strict ``parse_classifications`` then patches rows where
    ``drop_id`` is None by resolving the filename stem through the legacy
    prefix index. Unresolvable legacy stems stay as ``drop_id=None``.
    """
    df = parse_classifications(raw_classifications)

    if df.empty:
        return df

    legacy_index = build_legacy_prefix_index(db_drop_ids)

    unmatched_mask = (
        df["drop_id"].isna()
        & df["video_filename"].notna()
        & (df["video_filename"].astype(str) != "")
    )
    n_unmatched = int(unmatched_mask.sum())
    if n_unmatched == 0:
        return df

    n_resolved = 0
    for idx in df.index[unmatched_mask]:
        vf = df.at[idx, "video_filename"]
        legacy_drop_id = resolve_legacy_drop_id(Path(str(vf)).stem, legacy_index)
        if legacy_drop_id is None:
            continue
        df.at[idx, "drop_id"] = legacy_drop_id
        n_resolved += 1

    logging.info(
        f"Legacy post-process: resolved {n_resolved}/{n_unmatched} legacy row(s)"
    )
    return df


# Mirrors of helpers in parse_zooniverse_classifications.py — duplicated so
# legacy is self-contained. Update both copies together if the MaxN format
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


# ── Orchestrator — the --legacy entry point for Zooniverse backfill ──────────


def run_legacy_zooniverse_backfill() -> None:
    """
    End-to-end historical Zooniverse ingestion: load legacy CSVs → parse →
    aggregate → completion gate → export MaxN CSVs → ingest into annotations DB.

    Idempotent: ``ingest_zooniverse_annotations`` clears previous citsci rows
    per drop before writing, so re-running ``--legacy`` is safe.

    All CSVs live flat in ``config.legacy_zooniverse_dir``. Classification
    exports are identified by ``*classification*`` in the filename; subjects
    exports by ``*subject*``.
    """
    legacy_dir = config.legacy_zooniverse_dir
    if not legacy_dir.exists():
        logging.info(
            f"Legacy backfill: {legacy_dir} does not exist. Nothing to ingest."
        )
        return

    classification_csvs = sorted(legacy_dir.glob("*classification*.csv"))
    subjects_csvs = sorted(legacy_dir.glob("*subject*.csv"))

    if not classification_csvs:
        logging.info(f"Legacy backfill: no *classification*.csv in {legacy_dir}.")
        return

    logging.info(
        f"Legacy backfill: loading {len(classification_csvs)} classification CSV(s)"
    )
    raw = load_classifications_from_csv([str(p) for p in classification_csvs])
    if not raw:
        logging.info("Legacy backfill: no classifications loaded. Done.")
        return

    db_drop_ids = get_all_db_drop_ids()
    parsed_df = parse_legacy_classifications(raw, db_drop_ids)
    aggregated_df = aggregate_by_subject_species(parsed_df)

    legacy_dir.mkdir(parents=True, exist_ok=True)
    audit_path = legacy_dir / "audit.csv"
    aggregated_df.to_csv(audit_path, index=False)
    logging.info(f"Legacy audit log → {audit_path} ({len(aggregated_df)} rows)")

    if aggregated_df.empty:
        logging.info("Legacy backfill: no rows passed min_votes filter. Done.")
        return

    if subjects_csvs:
        logging.info(
            f"Legacy completion gate: loading {len(subjects_csvs)} subjects CSV(s)"
        )
        completion_df = subject_completion_from_csv([str(p) for p in subjects_csvs])
        aggregated_df = _filter_to_complete_drops(aggregated_df, completion_df)
    else:
        logging.warning(f"No *subject*.csv in {legacy_dir} — completion gate skipped.")

    if aggregated_df.empty:
        logging.info("Legacy backfill: no fully-complete drop_ids to export. Done.")
        return

    nothing_here_df = sample_nothing_here_clips(aggregated_df)
    if not nothing_here_df.empty:
        logging.info(
            f"NOTHINGHERE sampling: {len(nothing_here_df)} subjects selected "
            f"across {nothing_here_df['drop_id'].nunique()} drops."
        )

    # Suspicious minority finds stay in the audit CSV but do not get exported.
    export_df = aggregated_df[~aggregated_df["suspicious_minority_find"]].copy()
    _export_maxn_csvs(export_df)

    drop_ids_to_ingest = sorted(export_df["drop_id"].dropna().unique())
    logging.info(f"Legacy backfill: ingesting {len(drop_ids_to_ingest)} drop(s)")
    total_rows = 0
    for drop_id in drop_ids_to_ingest:
        total_rows += ingest_zooniverse_annotations(drop_id)

    logging.info(
        f"Legacy backfill complete: {total_rows} citsci rows across {len(drop_ids_to_ingest)} drop(s)."
    )
