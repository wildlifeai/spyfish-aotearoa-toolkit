"""
Reusable Zooniverse classification parsing logic.

Covers Phases 0-3 of the Zooniverse Classifications → Frame Extraction plan:
  Phase 0 — Fetch from Panoptes API
  Phase 1 — Parse API response (annotation type detection, drop_id resolution)
  Phase 2 — Aggregate by (subject_id, species), apply min_votes filter
  Phase 3 — NOTHINGHERE sampling

The top-level runner (parse_zooniverse_classifications.py) handles Phases 4-6
(MaxN CSV export, deduplication, frame extraction).
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from panoptes_client import Classification, Panoptes

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager

# ── Constants ────────────────────────────────────────────────────────────────

# Zooniverse question key variants (old and new workflow naming)
_TIMESTAMP_KEYS = [
    "WHATISTHEEARLIESTPOINTTHATYOUSEETHEMOSTINDIVIDUALSOFTHISSPECIES",
    "WHENDOYOUSEETHEMOSTINDIVIDUALSOFTHISSPECIES",
]

_NEW_FORMAT_PATTERN = re.compile(r"^[A-Z]+_\d{8}_BUV_.*")

# Zooniverse bucket answers for HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP.
# Stored as concatenated range strings (e.g. "2030" = 20-30 animals).
# We use the midpoint; experts review anyway.
_COUNT_BUCKETS: dict[str, int] = {
    "2030": 25,
    "3040": 35,
}

# Legacy filename: {RESERVE}_{SITE}_{DD}_{MM}_{YYYY}  e.g. AHE_062_25_04_2022
_LEGACY_DMY_PATTERN = re.compile(r"^([A-Z]{2,4})_(\d{3})_(\d{2})_(\d{2})_(\d{4})$")
# Legacy filename: {RESERVE}_{SITE}_{YYYY}_{MM}_{DD}  e.g. KPT_012_2021_06_03
_LEGACY_YMD_PATTERN = re.compile(r"^([A-Z]{2,4})_(\d{3})_(\d{4})_(\d{2})_(\d{2})$")


# ── Phase 0 — Fetch from API ─────────────────────────────────────────────────


def connect_to_zooniverse() -> None:
    """Authenticate with Panoptes using credentials from environment."""
    Panoptes.connect(username=config.user, password=config.password)


def fetch_classifications(since: Optional[str] = None) -> list[dict]:
    """
    Fetch retired classifications from the Zooniverse API.

    Args:
        since: ISO 8601 datetime string to fetch only newer classifications,
               or None to fetch from the beginning of the project.

    Returns:
        List of raw classification dicts with all fields needed for parsing.
    """
    source_ids = config.zooniverse_source_project_ids
    logging.info(
        f"Fetching classifications from {len(source_ids)} project(s): {source_ids}"
        + (f" since {since}" if since else " (full backfill)")
    )

    # Subject metadata is embedded in each classification response under
    # c.raw['subject_data'][subject_id] — no Subject.find() call needed.
    # We do NOT filter by Zooniverse retirement status here: Subject.find() returns
    # 404 for deleted subjects (which still have valid classification records), and
    # retirement is equivalent to "enough votes" — which min_votes in aggregation
    # already enforces.
    classifications = []

    for project_id in source_ids:
        kwargs: dict = {"project_id": project_id, "scope": "project"}
        if since:
            kwargs["created_at"] = since
        logging.info(f"  Fetching project {project_id}...")

        for c in Classification.where(**kwargs):
            # Use c.raw throughout — the LinkResolver (.links.workflow, .links.subjects
            # etc.) calls object_class(id) internally and errors when a class isn't
            # registered (e.g. Workflow returns NoneType).
            raw = c.raw
            links = raw.get("links", {})
            subject_ids = links.get("subjects", [])
            if not subject_ids:
                continue

            subject_id = str(subject_ids[0])

            subject_data = raw.get("subject_data", {}).get(subject_id, {})
            metadata = subject_data.get("metadata", {})
            locations = subject_data.get("locations", [])
            subject_set_ids = subject_data.get("links", {}).get("subject_sets", [None])

            classifications.append(
                {
                    "classification_id": raw.get("id"),
                    "created_at": raw.get("created_at"),
                    "user_name": raw.get("user_name"),
                    "user_id": raw.get("user_id"),
                    "annotations": raw.get("annotations", []),
                    "subject_id": subject_id,
                    "subject_set_id": (
                        str(subject_set_ids[0]) if subject_set_ids else None
                    ),
                    "workflow_id": str(links.get("workflow")),
                    "subject_metadata": metadata,
                    "subject_locations": locations,
                }
            )

        logging.info(f"  Project {project_id}: {len(classifications)} total so far.")

    logging.info(
        f"Fetched {len(classifications)} classifications across {len(source_ids)} project(s)."
    )
    return classifications


# ── Phase 1 — Parse ──────────────────────────────────────────────────────────


def _resolve_legacy_drop_id(stem: str, db_drop_ids: list[str]) -> Optional[str]:
    """
    Try to resolve a legacy-format filename stem to a drop_id using date patterns.

    Handles two formats:
      {RESERVE}_{SITE}_{DD}_{MM}_{YYYY}  e.g. AHE_062_25_04_2022
      {RESERVE}_{SITE}_{YYYY}_{MM}_{DD}  e.g. KPT_012_2021_06_03

    Constructs the drop_id prefix {RESERVE}_{YYYYMMDD}_BUV_{RESERVE}_{SITE}
    and finds the highest replicate in the DB with that prefix.
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
    matches = sorted([d for d in db_drop_ids if d.startswith(prefix)])
    if not matches:
        return None

    # Use the highest replicate (last alphabetically = highest _NN suffix)
    return matches[-1]


def _resolve_drop_id(
    video_filename: str, db_drop_ids: list[str]
) -> tuple[Optional[str], bool, str]:
    """
    Resolve a video filename to a drop_id.

    Returns:
        (drop_id, video_exists, match_path) where match_path is one of:
          "matched" | "unmatched_downloaded" | "unmatched_no_media"
    """
    if not video_filename:
        return None, False, "unmatched_no_media"

    stem = Path(video_filename).stem

    # New-format drop IDs: structurally predictable, validate and check video on disk
    if _NEW_FORMAT_PATTERN.match(stem):
        try:
            config.validate_drop_id(stem)
            video_path = config.get_video_path(stem)
            return stem, video_path.exists(), "matched"
        except Exception:
            pass

    # Legacy date-pattern filenames: reconstruct drop_id from parsed date + site
    legacy_drop_id = _resolve_legacy_drop_id(stem, db_drop_ids)
    if legacy_drop_id:
        video_path = config.media_dir / video_filename
        return legacy_drop_id, video_path.exists(), "matched"

    return None, False, "unmatched_downloaded"


def _parse_annotation(ann: dict) -> list[dict]:
    """
    Parse a single Panoptes annotation dict into a list of normalised rows.

    Handles two annotation types:
      - Classification (choice key present)
      - Drawing/measurement (x1/y1 keys, legacy)
    """
    rows = []
    for value_item in ann.get("value", []):
        if not isinstance(value_item, dict):
            continue

        annotation_type = "classification"
        species = None
        count = 0
        annotation_seconds = None
        bbox = {"x1": None, "y1": None, "x2": None, "y2": None}
        is_nothing_here = False

        if "choice" in value_item:
            # Classification task
            choice = value_item["choice"]
            is_nothing_here = choice in ("NOTHINGHERE", "NOTHING HERE", "NOTHING_HERE")
            species = None if is_nothing_here else choice

            answers = value_item.get("answers", {})

            # Timestamp: try both key variants
            for ts_key in _TIMESTAMP_KEYS:
                raw_ts = answers.get(ts_key)
                if raw_ts is not None:
                    # Format is e.g. "3S" → 3 seconds
                    try:
                        annotation_seconds = float(str(raw_ts).rstrip("Ss"))
                    except ValueError:
                        pass
                    break

            # Count
            count_raw = str(
                answers.get("HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP", "0")
            ).strip()
            if count_raw in _COUNT_BUCKETS:
                count = _COUNT_BUCKETS[count_raw]
            else:
                try:
                    count = int(count_raw.rstrip("+"))
                except (ValueError, AttributeError):
                    count = 0

        elif "x1" in value_item or "y1" in value_item:
            annotation_type = "drawing"
            species = value_item.get("tool_label")
            bbox = {
                "x1": value_item.get("x1"),
                "y1": value_item.get("y1"),
                "x2": value_item.get("x2"),
                "y2": value_item.get("y2"),
            }
        else:
            continue

        rows.append(
            {
                "annotation_type": annotation_type,
                "species": species,
                "count": count,
                "annotation_seconds": annotation_seconds,
                "is_nothing_here": is_nothing_here,
                **bbox,
            }
        )

    return rows


def parse_classifications(
    raw_classifications: list[dict],
    db_drop_ids: list[str],
) -> pd.DataFrame:
    """
    Parse raw Panoptes classification dicts into a flat DataFrame.

    Args:
        raw_classifications: Output of fetch_classifications().
        db_drop_ids: All known drop_ids from the DB, for legacy filename matching.

    Returns:
        DataFrame with one row per (classification, species annotation).
    """
    records = []

    for c in raw_classifications:
        meta = c["subject_metadata"]

        video_filename = (
            meta.get("video_filename")
            or meta.get("#video_filename")
            or meta.get("#VideoFilename")  # CSV export uses capital V/F
            or ""
        )
        upl_seconds = None
        # Read new key first, fall back to old "upl_seconds" for subjects uploaded before rename
        raw_upl = (
            meta.get("UplAbsSeconds")
            or meta.get("#UplAbsSeconds")
            or meta.get("upl_seconds")
            or meta.get("#upl_seconds")
        )
        if raw_upl is not None:
            try:
                upl_seconds = float(raw_upl)
            except (ValueError, TypeError):
                pass

        subject_type = (
            meta.get("subject_type")
            or meta.get("#subject_type")
            or meta.get("Subject_type", "clip")
        )
        time_of_max_seconds = None
        raw_tom = meta.get("#TimeOfMaxAbsSeconds") or meta.get("TimeOfMaxAbsSeconds")
        if raw_tom is not None:
            try:
                time_of_max_seconds = float(raw_tom)
            except (ValueError, TypeError):
                pass

        site_name = (
            meta.get("#siteName") or meta.get("#SiteID") or meta.get("site_name", "")
        )
        link_to_reserve = meta.get("!LinkToMarineReserve") or meta.get(
            "link_to_reserve", ""
        )
        event_date = meta.get("#EventDate") or meta.get("event_date", "")

        drop_id, video_exists, match_path = _resolve_drop_id(
            video_filename, db_drop_ids
        )

        ann_rows = []
        for ann in c["annotations"]:
            ann_rows.extend(_parse_annotation(ann))

        if not ann_rows:
            # Preserve classifications with no parseable annotations (e.g. pure NOTHINGHERE)
            ann_rows = [
                {
                    "annotation_type": "classification",
                    "species": None,
                    "count": 0,
                    "annotation_seconds": None,
                    "is_nothing_here": True,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                }
            ]

        for ann in ann_rows:
            annotation_seconds = ann["annotation_seconds"]

            # Absolute seconds in original video
            if subject_type == "frame" and time_of_max_seconds is not None:
                absolute_seconds = time_of_max_seconds
            elif upl_seconds is not None and annotation_seconds is not None:
                absolute_seconds = upl_seconds + annotation_seconds
            else:
                absolute_seconds = annotation_seconds

            records.append(
                {
                    "classification_id": c["classification_id"],
                    "created_at": c["created_at"],
                    "user_name": c["user_name"],
                    "user_id": c["user_id"],
                    "subject_id": c["subject_id"],
                    "subject_set_id": c["subject_set_id"],
                    "workflow_id": c["workflow_id"],
                    "video_filename": video_filename,
                    "drop_id": drop_id,
                    "video_exists": video_exists,
                    "match_path": match_path,
                    "subject_type": subject_type,
                    "upl_seconds": upl_seconds,
                    "species": ann["species"],
                    "count": ann["count"],
                    "annotation_seconds": ann["annotation_seconds"],
                    "absolute_seconds": absolute_seconds,
                    "annotation_type": ann["annotation_type"],
                    "bbox_x1": ann["x1"],
                    "bbox_y1": ann["y1"],
                    "bbox_x2": ann["x2"],
                    "bbox_y2": ann["y2"],
                    "site_name": site_name,
                    "link_to_reserve": link_to_reserve,
                    "event_date": event_date,
                    "is_nothing_here": ann["is_nothing_here"],
                    "is_retired": True,  # hard-filtered to retired only in fetch step
                    "subject_locations": c["subject_locations"],
                }
            )

    df = pd.DataFrame(records)
    logging.info(
        f"Parsed {len(df)} annotation rows from {len(raw_classifications)} classifications."
    )
    return df


# ── Phase 2 — Aggregate ──────────────────────────────────────────────────────


def aggregate_by_subject_species(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (subject_id, drop_id, video_filename, species) and compute vote counts.

    Applies the min_votes filter: rows below the threshold are dropped.
    Flags suspicious_minority_find where nothing_here_votes dominate but some
    species got at least 1 vote.

    Returns:
        Aggregated DataFrame sorted by (video_filename, vote_count desc).
    """
    if df.empty:
        return pd.DataFrame()

    # Total classifiers per subject (regardless of what they said)
    total_classifiers = (
        df.groupby("subject_id")["classification_id"]
        .nunique()
        .rename("total_classifiers")
    )

    # Nothing-here votes per subject
    nothing_here = (
        df[df["is_nothing_here"]]
        .groupby("subject_id")["classification_id"]
        .nunique()
        .rename("nothing_here_votes")
    )

    # Species rows only
    species_df = df[~df["is_nothing_here"] & df["species"].notna()].copy()

    agg = (
        species_df.groupby(
            ["subject_id", "drop_id", "video_filename", "species"],
            dropna=False,
        )
        .agg(
            vote_count=("classification_id", "nunique"),
            mean_seconds=("absolute_seconds", "mean"),
            mode_seconds=(
                "absolute_seconds",
                lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None,
            ),
            mode_count=(
                # TODO: review whether mode or max is more appropriate here.
                # MaxN convention = max individuals seen, so max() could be argued,
                # but experts review anyway so mode (consensus) is safer for now.
                "count",
                lambda x: int(x.dropna().mode().iloc[0]) if not x.dropna().empty else 0,
            ),
            upl_seconds=("upl_seconds", "first"),
            subject_set_id=("subject_set_id", "first"),
            workflow_id=("workflow_id", "first"),
            subject_locations=("subject_locations", "first"),
            video_exists=("video_exists", "first"),
        )
        .reset_index()
    )

    agg = agg.join(total_classifiers, on="subject_id")
    agg = agg.join(nothing_here, on="subject_id")
    agg["nothing_here_votes"] = agg["nothing_here_votes"].fillna(0).astype(int)
    agg["agreement_pct"] = (agg["vote_count"] / agg["total_classifiers"] * 100).round(1)

    # Flag suspicious minority finds: nothing_here dominates but someone found something
    agg["suspicious_minority_find"] = (
        agg["nothing_here_votes"] > agg["total_classifiers"] / 2
    ) & (agg["vote_count"] >= 1)

    # Apply min_votes filter (suspicious finds are kept in audit CSV but not extracted)
    min_votes = config.zooniverse_min_votes
    passed = agg[agg["vote_count"] >= min_votes].copy()

    logging.info(
        f"Aggregated: {len(passed)} (subject, species) rows pass min_votes={min_votes} "
        f"(from {len(agg)} total, {agg['suspicious_minority_find'].sum()} suspicious minority finds)"
    )

    return passed.sort_values(["video_filename", "vote_count"], ascending=[True, False])


# ── Phase 3 — NOTHINGHERE sampling ───────────────────────────────────────────


def sample_nothing_here_clips(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each drop_id where ≥10% of retired subjects are dominated by NOTHINGHERE votes,
    sample 10% of those subjects (min 1) and generate one frame row per sampled subject.

    Returns:
        DataFrame of sampled NOTHINGHERE rows with upl_seconds as the timestamp.
    """
    if df.empty:
        return pd.DataFrame()

    # All retired subjects per drop_id with nothing_here_votes > species votes
    nothing_dominated = df[
        df["nothing_here_votes"] > df["total_classifiers"] / 2
    ].drop_duplicates("subject_id")

    total_per_drop = (
        df.drop_duplicates("subject_id").groupby("drop_id")["subject_id"].count()
    )

    rows = []
    for drop_id, grp in nothing_dominated.groupby("drop_id"):
        total = total_per_drop.get(drop_id, 0)
        if total == 0:
            continue
        pct_nothing = len(grp) / total
        if pct_nothing < 0.10:
            continue

        sample_n = max(1, int(len(grp) * 0.10))
        sampled = grp.sample(n=min(sample_n, len(grp)), random_state=42)

        for _, row in sampled.iterrows():
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "drop_id": drop_id,
                    "video_filename": row["video_filename"],
                    "upl_seconds": row["upl_seconds"],
                    "species": "NOTHINGHERE",
                    "sample_reason": "NOTHINGHERE sampling",
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        logging.info(
            f"NOTHINGHERE sampling: {len(result)} subjects selected across {result['drop_id'].nunique()} drops."
        )
    return result


# ── CSV loader ───────────────────────────────────────────────────────────────


def load_classifications_from_csv(csv_paths: list[str]) -> list[dict]:
    """
    Load Zooniverse export CSVs and normalise rows into the same dict format
    as fetch_classifications(), so parse_classifications() works unchanged.

    CSV subject_data fields are flat (no nested 'metadata' key) and use
    different capitalisation to the API (e.g. '#VideoFilename' vs 'video_filename').
    These are passed through as-is; parse_classifications() handles all variants.

    Args:
        csv_paths: Paths to one or more Zooniverse classification export CSVs.

    Returns:
        List of classification dicts in the same format as fetch_classifications().
    """
    import json

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
            # CSV subject_data: fields are at top level (not nested under 'metadata')
            subject_entry = subject_data_map.get(subject_id, {})

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
                    "subject_metadata": subject_entry,
                    "subject_locations": [],  # not in CSV export
                }
            )

    logging.info(f"Loaded {len(records)} classifications from {len(csv_paths)} CSV(s).")
    return records


# ── Subject completion ────────────────────────────────────────────────────────


def subject_completion_from_csv(subjects_csv_paths: list[str]) -> pd.DataFrame:
    """
    Load Zooniverse subjects export CSV(s) and return retirement completion per drop_id.

    A drop_id is "fully complete" when every one of its uploaded subjects is retired.
    Partially retired drop_ids are flagged so they are not treated as done.

    Returns:
        DataFrame with columns: drop_id, total, retired, pct_retired, fully_complete
    """
    import json as _json

    frames = []
    for path in subjects_csv_paths:
        df = pd.read_csv(path, dtype=str)

        def _get_video_filename(meta_str):
            try:
                m = _json.loads(meta_str)
                return (
                    m.get("#VideoFilename")
                    or m.get("video_filename")
                    or m.get("#video_filename")
                    or ""
                )
            except Exception:
                return ""

        df["video_filename"] = df["metadata"].apply(_get_video_filename)
        df["drop_id"] = (
            df["video_filename"].str.replace(r"\.mp4$", "", regex=True).str.strip()
        )
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


def subject_completion_from_api() -> pd.DataFrame:
    """
    Fetch live subject retirement counts from the Panoptes API per subject set,
    resolve each subject set to a drop_id via its subject metadata, and return
    the same completion summary as subject_completion_from_csv().

    Requires an active Panoptes connection (call connect_to_zooniverse() first).
    Slower than the CSV approach — use for live status checks, not bulk backfill.
    """
    from panoptes_client import Subject, SubjectSet

    rows = []
    for project_id in config.zooniverse_source_project_ids:
        logging.info(f"  Fetching subject sets for project {project_id}...")
        for ss in SubjectSet.where(project_id=project_id):
            ss_id = ss.id
            # Sample the first subject to resolve drop_id from its metadata
            drop_id = None
            for subj in Subject.where(subject_set_id=ss_id):
                raw = subj.raw
                meta = raw.get("metadata", {})
                vf = (
                    meta.get("#VideoFilename")
                    or meta.get("video_filename")
                    or meta.get("#video_filename")
                    or ""
                )
                drop_id = vf.replace(".mp4", "").strip() or None
                break  # only need one subject to identify the drop

            counts = ss.raw.get("set_member_subjects_count", 0)
            retired = ss.raw.get("retired_set_member_subjects_count", 0)
            rows.append(
                {
                    "project_id": project_id,
                    "subject_set_id": ss_id,
                    "drop_id": drop_id,
                    "total": int(counts or 0),
                    "retired": int(retired or 0),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pct_retired"] = (df["retired"] / df["total"].replace(0, pd.NA) * 100).round(1)
    df["fully_complete"] = df["retired"] == df["total"]
    return df.sort_values("pct_retired", ascending=False).reset_index(drop=True)


# ── DB helpers ────────────────────────────────────────────────────────────────


def get_all_db_drop_ids() -> list[str]:
    """Fetch all known drop_ids from the pipeline DB for legacy filename matching."""
    db = DatabaseManager()
    deployments = db.get_all_deployments_map()
    return list(deployments.keys())


# ── Phase 5 — DB ingestion ────────────────────────────────────────────────────


def ingest_zooniverse_annotations(drop_id: str) -> int:
    """
    Read the per-drop Zooniverse MaxN CSV (written by parse_zooniverse_classifications.py)
    and store annotations in spyfish_annotations.db with annotated_by='citsci'.

    Clears any previous citsci annotations for this drop before writing,
    so re-running is safe.

    Returns:
        Number of annotation rows ingested (0 if CSV not found or empty).
    """
    from spyfish.database.annotation_manager import AnnotationDatabaseManager
    from spyfish.database.manager import DatabaseManager as PipelineDB

    maxn_csv = (
        config.get_drop_annotations_dir(drop_id) / f"{drop_id}_zooniverse_maxn.csv"
    )
    if not maxn_csv.exists():
        logging.info(
            f"ingest_zooniverse: No MaxN CSV found for {drop_id} at {maxn_csv}"
        )
        return 0

    df = pd.read_csv(maxn_csv)
    if df.empty:
        logging.info(f"ingest_zooniverse: MaxN CSV is empty for {drop_id}")
        return 0

    annotations = []
    for _, row in df.iterrows():
        annotations.append(
            {
                "drop_id": drop_id,
                "scientific_name": row[config.csv_scientific_name_column],
                "time_of_max": row[config.csv_maxn_time_column],
                "max_interval": row[config.csv_max_interval_column],
                "annotated_by": "citsci",
                "interval_annotation": row.get(
                    config.csv_interval_annotation_column, ""
                ),
                "confidence_agreement": row.get(config.csv_confidence_agreement_column),
                "external_id": None,
            }
        )

    ann_db = AnnotationDatabaseManager()
    ann_db.clear_annotations(drop_id, "citsci")
    ann_db.add_annotations(annotations)
    logging.info(
        f"ingest_zooniverse: Stored {len(annotations)} citsci annotations for {drop_id}"
    )

    # Sync annotation counts back to the pipeline deployments table
    pipeline_db = PipelineDB()
    pipeline_db.sync_annotation_counts([drop_id])

    return len(annotations)
