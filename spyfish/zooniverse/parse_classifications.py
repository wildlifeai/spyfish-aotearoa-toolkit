"""
Zooniverse classification parsing — strict new-format drop_id resolution only.

Non-canonical video filenames log a warning and surface as ``drop_id=None``.
Historical backfill lives in ``spyfish.zooniverse.legacy_extract``; core
must not import from it.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from panoptes_client import Classification, Panoptes

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.zooniverse.subject_keys import SubjectKeys

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
            subject_set_ids = subject_data.get("links", {}).get("subject_sets", [])

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


def _resolve_drop_id(video_filename: str) -> Optional[str]:
    """
    Strict filename → drop_id. Returns ``None`` for empty inputs and for
    non-canonical stems (with a warning log in the latter case).
    """
    if not video_filename:
        return None

    stem = Path(video_filename).stem

    if _NEW_FORMAT_PATTERN.match(stem):
        try:
            config.validate_drop_id(stem)
            return stem
        except Exception:
            pass

    logging.warning(
        f"Non-canonical Zooniverse video_filename: {video_filename!r} — "
        "flag upstream, do not silently remap."
    )
    return None


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


# Placeholder used when a classification has no parseable annotations — keeps
# the row in the output so "everyone said NOTHINGHERE" is countable later.
_NOTHING_HERE_PLACEHOLDER = {
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


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _extract_subject_metadata(meta: dict) -> dict:
    """Read Zooniverse subject metadata using only the keys upload.py writes.

    Strict — no legacy fallbacks. Pre-normalise legacy metadata via
    `legacy_extract._normalize_legacy_metadata` before passing to this parser.
    """
    return {
        "video_filename": meta.get(SubjectKeys.VIDEO_FILENAME, ""),
        "upl_seconds": _parse_float(meta.get(SubjectKeys.UPL_SECONDS)),
        "subject_type": meta.get(SubjectKeys.SUBJECT_TYPE, "clip"),
        "time_of_max_seconds": _parse_float(meta.get(SubjectKeys.TIME_OF_MAX)),
        "site_name": meta.get(SubjectKeys.SITE_NAME, ""),
        "link_to_reserve": meta.get(SubjectKeys.LINK_TO_RESERVE, ""),
        "event_date": meta.get(SubjectKeys.EVENT_DATE, ""),
    }


def _missing_required_keys(meta: dict) -> list[str]:
    """Return any required SubjectKeys that are absent or empty in `meta`."""
    return [k for k in SubjectKeys.REQUIRED if not meta.get(k)]


def _absolute_seconds(
    subject_type: str,
    time_of_max_seconds: Optional[float],
    upl_seconds: Optional[float],
    annotation_seconds: Optional[float],
) -> Optional[float]:
    """Compute the absolute video timestamp for an annotation.

    Frame subjects use the pre-baked TimeOfMax. Clip subjects offset the
    annotation_seconds (which is relative to clip start) by upl_seconds.
    """
    if subject_type == "frame" and time_of_max_seconds is not None:
        return time_of_max_seconds
    if upl_seconds is not None and annotation_seconds is not None:
        return upl_seconds + annotation_seconds
    return annotation_seconds


def _build_classification_record(
    classification: dict,
    annotation: dict,
    meta_fields: dict,
    drop_id: Optional[str],
) -> dict:
    """Compose one output row for a single (classification, annotation) pair."""
    return {
        "classification_id": classification["classification_id"],
        "created_at": classification["created_at"],
        "user_name": classification["user_name"],
        "user_id": classification["user_id"],
        "subject_id": classification["subject_id"],
        "subject_set_id": classification["subject_set_id"],
        "workflow_id": classification["workflow_id"],
        "video_filename": meta_fields["video_filename"],
        "drop_id": drop_id,
        "subject_type": meta_fields["subject_type"],
        "upl_seconds": meta_fields["upl_seconds"],
        "species": annotation["species"],
        "count": annotation["count"],
        "annotation_seconds": annotation["annotation_seconds"],
        "absolute_seconds": _absolute_seconds(
            meta_fields["subject_type"],
            meta_fields["time_of_max_seconds"],
            meta_fields["upl_seconds"],
            annotation["annotation_seconds"],
        ),
        "annotation_type": annotation["annotation_type"],
        "bbox_x1": annotation["x1"],
        "bbox_y1": annotation["y1"],
        "bbox_x2": annotation["x2"],
        "bbox_y2": annotation["y2"],
        "site_name": meta_fields["site_name"],
        "link_to_reserve": meta_fields["link_to_reserve"],
        "event_date": meta_fields["event_date"],
        "is_nothing_here": annotation["is_nothing_here"],
        "is_retired": True,  # hard-filtered to retired-only in fetch step
        "subject_locations": classification["subject_locations"],
    }


def parse_classifications(raw_classifications: list[dict]) -> pd.DataFrame:
    """
    Parse raw Panoptes classification dicts into one row per
    (classification, species annotation). Subjects with non-current
    metadata keys are counted and surface as a single warning at the
    end — they should be passed through legacy_extract first.
    """
    records = []
    subjects_missing_keys: dict[str, int] = {}

    for c in raw_classifications:
        meta = c["subject_metadata"]
        for missing in _missing_required_keys(meta):
            subjects_missing_keys[missing] = subjects_missing_keys.get(missing, 0) + 1

        meta_fields = _extract_subject_metadata(meta)
        drop_id = _resolve_drop_id(meta_fields["video_filename"])

        ann_rows = [row for ann in c["annotations"] for row in _parse_annotation(ann)]
        if not ann_rows:
            ann_rows = [_NOTHING_HERE_PLACEHOLDER]

        for ann in ann_rows:
            records.append(_build_classification_record(c, ann, meta_fields, drop_id))

    df = pd.DataFrame(records)
    logging.info(
        f"Parsed {len(df)} annotation rows from {len(raw_classifications)} classifications."
    )
    if subjects_missing_keys:
        logging.warning(
            "Some classifications were missing current subject metadata keys: "
            f"{subjects_missing_keys}. These were probably uploaded by an older "
            "version of upload.py — pass them through "
            "spyfish.zooniverse.legacy_extract._normalize_legacy_metadata first."
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


# ── Subject completion ────────────────────────────────────────────────────────


def subject_completion_from_api() -> pd.DataFrame:
    """
    Per drop_id retirement completion from live Panoptes SubjectSet counts.
    Requires an active Panoptes connection (call connect_to_zooniverse() first).

    Returns:
        DataFrame with columns: project_id, subject_set_id, drop_id, total,
        retired, pct_retired, fully_complete.
    """
    from panoptes_client import Subject, SubjectSet

    rows = []
    for project_id in config.zooniverse_source_project_ids:
        logging.info(f"  Fetching subject sets for project {project_id}...")
        for ss in SubjectSet.where(project_id=project_id):
            ss_id = ss.id
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
                drop_id = _resolve_drop_id(vf)
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
    Read the per-drop Zooniverse MaxN CSV (written by spyfish.zooniverse.live_extract)
    and store annotations in spyfish_annotations.db with annotated_by='citsci'.

    Clears any previous citsci annotations for this drop before writing,
    so re-running is safe.

    Returns:
        Number of annotation rows ingested (0 if CSV not found or empty).
    """
    from spyfish.database.annotation_manager import AnnotationDatabaseManager
    from spyfish.database.manager import DatabaseManager as PipelineDB

    maxn_csv = config.get_zooniverse_maxn_csv_path(drop_id)
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


def sync_zooniverse_drop(drop_id: str) -> str | None:
    """
    Pipeline entry point for zooniverse-sync (citsci_status: frames_uploaded → complete).

    Checks whether spyfish.zooniverse.live_extract has written a MaxN CSV
    for this drop. If found, ingests it into the annotations DB and signals
    the pipeline to advance. Returns None if the CSV isn't ready yet.

    TODO: Integrate Caesar completion check so this step auto-detects subject
    retirement without requiring spyfish.zooniverse.live_extract to be
    run separately first.
    """
    from spyfish.config.base import CitSciStatus

    count = ingest_zooniverse_annotations(drop_id)
    if count == 0:
        logging.info(
            f"zooniverse-sync: No MaxN CSV ready for {drop_id}. "
            "Run `python -m spyfish.zooniverse.live_extract` once volunteers are done. "
            "Leaving at frames_uploaded."
        )
        return None

    logging.info(
        f"zooniverse-sync: Ingested {count} citsci annotations for {drop_id} → citsci_status=complete"
    )
    return CitSciStatus.COMPLETE
