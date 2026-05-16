"""
Zooniverse classification parsing — strict new-format drop_id resolution only.

Non-canonical video filenames log a warning and surface as ``drop_id=None``.
Historical backfill lives in ``spyfish.zooniverse.legacy_extract``; core
must not import from it.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from panoptes_client import Panoptes

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.utils import load_species_labels, normalise_zoo_choice, seconds_to_time
from spyfish.zooniverse.subject_keys import SubjectKeys

# ── Constants ────────────────────────────────────────────────────────────────

# Zooniverse question key variants (old and new workflow naming)
_TIMESTAMP_KEYS = [
    "WHATISTHEEARLIESTPOINTTHATYOUSEETHEMOSTINDIVIDUALSOFTHISSPECIES",
    "WHENDOYOUSEETHEMOSTINDIVIDUALSOFTHISSPECIES",
]

# Zooniverse bucket answers for HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP.
# Stored as concatenated range strings (e.g. "2030" = 20-30 animals).
# We use the midpoint; experts review anyway.
_COUNT_BUCKETS: dict[str, int] = {
    "2030": 25,
    "3040": 35,
}

# Volunteer quality-control thresholds for aggregate_by_subject_species.
# Self-contained — delete both constants and the filter block in the
# aggregator to remove this behaviour entirely without touching config.yaml.
_USER_EXCLUSION_NH_PCT_THRESHOLD = 0.90  # exclude users with >= 90% NH rate
_USER_EXCLUSION_MIN_CLASSIFICATIONS = 100  # only after this many classifications


# ── Phase 0 — Fetch from API ─────────────────────────────────────────────────


def connect_to_zooniverse() -> None:
    """Authenticate with Panoptes using credentials from environment."""
    Panoptes.connect(username=config.user, password=config.password)


def fetch_classifications_for_set(subject_set_id: str) -> list[dict]:
    """
    Fetch all retired classifications for a single subject set.

    Used by the per-subject-set sync path: call ``subject_completion_from_api``
    to confirm ``fully_complete=True``, then call this to retrieve the data.
    Returns the same dict shape as ``fetch_classifications`` so both feed into
    ``parse_classifications`` without modification.

    A non-retired subject in a "fully complete" set is unexpected but possible
    if retirement changed between the completion check and this fetch; those
    classifications are skipped and logged.
    """
    from panoptes_client import Classification

    classifications = []
    n_skipped = 0

    # `scope="project"` is required for two reasons:
    #   1. Without it, Panoptes returns only the authenticated user's own
    #      classifications (default "private" scope), not the volunteer pool —
    #      ~21 of ~500 for a typical retired drop on the team account.
    #   2. Without it, the response strips `subject_data` (no metadata, no
    #      retired field), so the per-subject retirement filter below rejects
    #      every classification regardless of actual status.
    for c in Classification.where(subject_set_id=subject_set_id, scope="project"):
        raw = c.raw
        links = raw.get("links", {})
        subject_ids = links.get("subjects", [])
        if not subject_ids:
            continue

        subject_id = str(subject_ids[0])
        subject_data = raw.get("subject_data", {}).get(subject_id, {})

        if not subject_data.get("retired"):
            n_skipped += 1
            continue

        metadata = subject_data.get("metadata", {})
        locations = subject_data.get("locations", [])
        subject_set_ids = subject_data.get("links", {}).get("subject_sets", [])

        classifications.append(
            {
                "classification_id": raw.get("id"),
                "created_at": raw.get("created_at"),
                "user_name": raw.get("user_name"),
                "user_id": raw.get("user_id"),
                # Hashed-IP token Panoptes assigns to anonymous classifications
                # (and includes for logged-in too). Carried through so the
                # dedupe in aggregate_by_subject_species can fall back to it
                # when user_id is None — otherwise all anonymous votes on a
                # subject collapse to a single (None, subject, species) row.
                "user_ip": raw.get("user_ip"),
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

    if n_skipped:
        logging.warning(
            f"Subject set {subject_set_id}: skipped {n_skipped} classification(s) "
            "from non-retired subjects — set marked complete but retirement may be "
            "in progress. Rerun to pick up when fully settled."
        )

    logging.info(
        f"Subject set {subject_set_id}: fetched {len(classifications)} classification(s)."
    )
    return classifications


# ── Phase 1 — Parse ──────────────────────────────────────────────────────────


def _resolve_drop_id(video_filename: str) -> Optional[str]:
    """
    Strict filename → drop_id. Returns ``None`` for empty inputs and for
    any stem that doesn't pass ``config.validate_drop_id``. Silent by design —
    a per-call warning here would flood logs on legacy-heavy backfills where
    most rows fail by construction. Callers aggregate the unresolved set and
    surface a single summary (see ``parse_classifications``). Legacy stems
    are resolved separately by ``legacy_extract.parse_legacy_classifications``.
    """
    if not video_filename:
        return None
    try:
        return config.validate_drop_id(Path(video_filename).stem)
    except ValueError:
        return None


def _parse_annotation(ann: dict) -> list[dict]:
    """
    Parse a single Panoptes annotation dict into a list of normalised rows.

    Handles three annotation types:
      - Classification (choice key present)
      - Line/measurement drawing (x1/y1 keys)
      - Bounding box drawing (x/y/width/height keys, e.g. workflow 17057)
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

        elif "x" in value_item and "width" in value_item:
            # Bounding box drawing tool (x/y = top-left corner, width/height =
            # dimensions). Each box is one drawn individual → count=1.
            # tool_label carries the species name (mixed-case, unlike the
            # all-caps choice strings from clip workflows — normalise downstream
            # if aggregating across both annotation types).
            # TODO: per-classification box count (n boxes of same species =
            # MaxN for that classifier) requires grouping by classification_id
            # before aggregation — not yet done here.
            annotation_type = "drawing"
            species = value_item.get("tool_label")
            x = value_item.get("x", 0)
            y = value_item.get("y", 0)
            bbox = {
                "x1": x,
                "y1": y,
                "x2": x + value_item.get("width", 0),
                "y2": y + value_item.get("height", 0),
            }
            count = 1

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
    "is_blank_submission": False,
    "x1": None,
    "y1": None,
    "x2": None,
    "y2": None,
}

# Placeholder for a volunteer who submitted without selecting any option
# (value: [] in every task). Distinct from an explicit NOTHINGHERE click.
# Excluded from aggregation — a non-response carries no information.
_BLANK_SUBMISSION_PLACEHOLDER = {
    **_NOTHING_HERE_PLACEHOLDER,
    "is_blank_submission": True,
}


def _is_blank_submission(annotations: list[dict]) -> bool:
    """True when every annotation task was submitted with an empty value list."""
    return bool(annotations) and all(not ann.get("value") for ann in annotations)


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
    if subject_type == "frame":
        return time_of_max_seconds if time_of_max_seconds is not None else upl_seconds
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
        "user_ip": classification.get("user_ip"),
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
        "is_blank_submission": annotation.get("is_blank_submission", False),
        "is_retired": True,  # fetch_classifications skips non-retired subjects
        "subject_locations": classification["subject_locations"],
    }


def parse_classifications(raw_classifications: list[dict]) -> pd.DataFrame:
    """
    Parse raw Panoptes classification dicts into one row per
    (classification, species annotation). Two summary warnings surface at
    the end: subjects with non-current metadata keys, and unresolved
    non-canonical video filenames. Both are usually expected on legacy
    data — pass through ``legacy_extract`` first to resolve them.
    """
    records = []
    subjects_missing_keys: dict[str, int] = {}
    unresolved_filenames: dict[str, int] = {}

    for c in raw_classifications:
        meta = c["subject_metadata"]
        for missing in _missing_required_keys(meta):
            subjects_missing_keys[missing] = subjects_missing_keys.get(missing, 0) + 1

        meta_fields = _extract_subject_metadata(meta)
        video_filename = meta_fields["video_filename"]
        drop_id = _resolve_drop_id(video_filename)
        if drop_id is None and video_filename:
            stem = Path(video_filename).stem
            unresolved_filenames[stem] = unresolved_filenames.get(stem, 0) + 1

        ann_rows = [row for ann in c["annotations"] for row in _parse_annotation(ann)]
        if not ann_rows:
            placeholder = (
                _BLANK_SUBMISSION_PLACEHOLDER
                if _is_blank_submission(c["annotations"])
                else _NOTHING_HERE_PLACEHOLDER
            )
            ann_rows = [placeholder]

        for ann in ann_rows:
            records.append(_build_classification_record(c, ann, meta_fields, drop_id))

    df = pd.DataFrame(records)
    logging.info(
        f"Parsed {len(df)} annotation rows from {len(raw_classifications)} classifications."
    )
    if subjects_missing_keys:
        logging.warning(
            "Some subjects were missing current metadata keys: "
            f"{subjects_missing_keys}. "
            "subject_type defaults to 'clip'; upl_seconds defaults to None. "
            "Likely frame subjects or uploads predating the #SubjectType convention."
        )
    if unresolved_filenames:
        n_distinct = len(unresolved_filenames)
        n_rows = sum(unresolved_filenames.values())
        top = sorted(unresolved_filenames.items(), key=lambda kv: -kv[1])[:5]
        logging.warning(
            f"Strict parse: {n_distinct} non-canonical filename stem(s) left drop_id=None "
            f"({n_rows} classification row(s)). Top 5: {top}. "
            "These remain unresolved if called directly; use parse_legacy_classifications "
            "to resolve legacy DMY/YMD stems via the prefix index."
        )
    return df


# ── Phase 2 — Aggregate ──────────────────────────────────────────────────────


def aggregate_by_subject_species(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (subject_id, drop_id, video_filename, species) and emit one row
    of consensus statistics per (subject, species).

    Count axis carries three statistics for three consumers:
      mode_count → conservative integer for training labels
      mean_count → continuous for ranking frames (mean_count × agreement_pct)
      max_count  → true MaxN (peak abundance) for ecology dashboards

    Two disagreement flags surface rows that warrant expert review:
      suspicious_minority_find → species disagreement (most said NOTHINGHERE)
      count_disagreement       → count disagreement (max_count ≥ mode + 2)

    Applies the agreement_pct filter (`vote_count / total_classifiers`)
    as the primary gate — invariant to workflow retirement count, so
    expert and broad-public workflows both pass on the same principle.
    Suspicious-minority and count-disagreement flags are advisory —
    rows still pass the filter, but the flags route them into the
    appropriate BIIGLE pool downstream.

    Pre-aggregation exclusions (applied in order before counting):
      1. Blank submissions (is_blank_submission=True) — value:[] payload,
         no answer selected. Carries no information; excluded from all counts.
      2. High-NH users — NH rate >= zooniverse_user_exclusion_nh_pct_threshold
         AND >= zooniverse_user_exclusion_min_classifications. Catches
         click-through users without requiring a manual exclusion list.
      3. Dedupe by (user_id, subject_id, species) — collapses CSV export
         row inflation (observed at 10–30× on some workflows).

    Returns:
        Aggregated DataFrame sorted by (video_filename, vote_count desc).
    """
    if df.empty:
        return pd.DataFrame()

    # Drop blank submissions before any counting — a non-response carries no
    # information and should not inflate total_classifiers or nothing_here_votes.
    if "is_blank_submission" in df.columns:
        n_blank = int(df["is_blank_submission"].sum())
        if n_blank:
            logging.info(
                f"Excluding {n_blank} blank submission(s) from aggregation "
                f"({df.loc[df['is_blank_submission'], 'user_name'].nunique()} user(s))."
            )
            df = df[~df["is_blank_submission"]].copy()
        if df.empty:
            return pd.DataFrame()

    # Predicate-based user exclusion: drop users whose NH rate exceeds the
    # configured threshold (default 90%) AND who have enough classifications
    # to distinguish genuine click-through from an unlucky run of empty clips.
    # Runs after blank-submission removal so blank rows don't inflate NH%.
    if "user_name" in df.columns:
        named = df[df["user_name"].notna()]
        user_stats = named.groupby("user_name").agg(
            n_classifications=("classification_id", "nunique"),
            nh_pct=("is_nothing_here", "mean"),
        )
        nh_threshold = _USER_EXCLUSION_NH_PCT_THRESHOLD
        min_class = _USER_EXCLUSION_MIN_CLASSIFICATIONS
        flagged = user_stats[
            (user_stats["nh_pct"] >= nh_threshold)
            & (user_stats["n_classifications"] >= min_class)
        ]
        if not flagged.empty:
            logging.info(
                f"Excluding {len(flagged)} high-NH user(s) from aggregation: "
                + ", ".join(
                    f"{u} ({row['nh_pct'] * 100:.0f}% NH, "
                    f"{int(row['n_classifications'])} classifications)"
                    for u, row in flagged.iterrows()
                )
            )
            df = df[~df["user_name"].isin(flagged.index)].copy()
        if df.empty:
            return pd.DataFrame()

    # Build a per-volunteer dedupe token that survives anonymous classifications.
    # Panoptes sets user_id=None for not-logged-in volunteers, so a naive
    # (user_id, subject, species) dedupe collapses every anonymous vote to a
    # single (None, subject, species) row — silent data loss for workflows
    # with sizeable anonymous traffic (e.g. 14054). Fall back through
    # user_name then user_ip (hashed IP token Panoptes provides on
    # anonymous classifications). Final fallback is the classification_id
    # itself, which preserves the row rather than collapsing it when we
    # genuinely cannot identify the volunteer.
    def _col_or_nan(col: str) -> pd.Series:
        return df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)

    df = df.copy()
    df["_volunteer_key"] = (
        _col_or_nan("user_id")
        .astype("object")
        .fillna(_col_or_nan("user_name"))
        .fillna(_col_or_nan("user_ip"))
        .fillna(df["classification_id"].astype("object"))
    )

    # Enforce one-row-per-(volunteer, subject, species) before any counting.
    # The CSV export from Zooniverse occasionally re-emits the same logical
    # classification under multiple distinct classification_ids (workflow
    # 23923 was observed at ~30× inflation in the legacy corpus), which
    # would inflate vote_count and total_classifiers. Deduping on the
    # logical key — what the volunteer actually claimed — collapses those
    # to one row per real claim. Multi-species clicks (one volunteer voting
    # both BLUECOD and SNAPPER on one subject) are preserved because species
    # is part of the key. Multi-click on the same species at different
    # timestamps is the only signal lost — rare in practice (<1% of rows
    # per the multi-click distribution) and an acceptable trade.
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["_volunteer_key", "subject_id", "species"], keep="first"
    )
    if n_before != len(df):
        n_removed = n_before - len(df)
        logging.info(
            f"aggregate_by_subject_species: deduped {n_before:,} -> {len(df):,} "
            f"rows ({n_removed:,} removed, "
            f"{n_removed / n_before * 100:.1f}%) via (volunteer_key, subject_id, species)"
        )

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

    # Species rows only. Split OTHER out — "OTHER" is the catch-all bucket
    # for species not in the named list, so two volunteers saying OTHER on the
    # same clip might be flagging different animals (a ray and a lobster).
    # Treating them as consensus would silently merge distinct findings, so
    # each OTHER vote is preserved as its own row.
    species_df = df[~df["is_nothing_here"] & df["species"].notna()].copy()
    other_mask = species_df["species"] == "OTHER"
    other_df = species_df[other_mask].copy()
    non_other_df = species_df[~other_mask].copy()

    AGG_FIELDS = dict(
        vote_count=("classification_id", "nunique"),
        mean_seconds=("absolute_seconds", "mean"),
        mode_seconds=(
            "absolute_seconds",
            lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None,
        ),
        # mode = consensus integer; canonical for training labels. Companion
        # mean_count and max_count below cover ranking and ecology.
        mode_count=(
            "count",
            lambda x: int(x.dropna().mode().iloc[0]) if not x.dropna().empty else 0,
        ),
        mean_count=("count", "mean"),
        max_count=("count", "max"),
        upl_seconds=("upl_seconds", "first"),
        subject_set_id=("subject_set_id", "first"),
        workflow_id=("workflow_id", "first"),
        subject_locations=("subject_locations", "first"),
    )

    agg_non_other = (
        non_other_df.groupby(
            ["subject_id", "drop_id", "video_filename", "species"],
            dropna=False,
        )
        .agg(**AGG_FIELDS)
        .reset_index()
    )

    # OTHER: group by volunteer too so each vote is its own row.
    # Uses _volunteer_key (not user_id) so anonymous voters don't collapse —
    # see the dedupe-key construction above. Drop the key from the output
    # schema so downstream consumers see the same columns as agg_non_other.
    agg_other = (
        other_df.groupby(
            ["subject_id", "drop_id", "video_filename", "species", "_volunteer_key"],
            dropna=False,
        )
        .agg(**AGG_FIELDS)
        .reset_index()
        .drop(columns=["_volunteer_key"])
    )

    agg = pd.concat([agg_non_other, agg_other], ignore_index=True)

    agg = agg.join(total_classifiers, on="subject_id")
    agg = agg.join(nothing_here, on="subject_id")
    agg["nothing_here_votes"] = agg["nothing_here_votes"].fillna(0).astype(int)
    agg["agreement_pct"] = (agg["vote_count"] / agg["total_classifiers"] * 100).round(1)

    # Flag suspicious minority finds: nothing_here dominates but someone found something.
    # Does not apply to OTHER — every OTHER row is a single volunteer vote by
    # construction; expecting agreement on it isn't meaningful.
    agg["suspicious_minority_find"] = (
        (agg["species"] != "OTHER")
        & (agg["nothing_here_votes"] > agg["total_classifiers"] / 2)
        & (agg["vote_count"] >= 1)
    )

    # Flag count disagreement: everyone agrees the species is here but disagree
    # on count. max_count >= mode + 2 catches the meaningful spread (e.g.
    # {3,4,3,6,3} → mode=3, max=6, gap=3 → flagged) while ignoring trivial
    # spreads at low counts ({1,1,1,1,2} → gap=1, not flagged). Requires
    # vote_count ≥ 3 so "disagreement" has at least three voices.
    agg["count_disagreement"] = (agg["max_count"] >= agg["mode_count"] + 2) & (
        agg["vote_count"] >= 3
    )

    # Apply agreement_pct filter to named-species rows only. OTHER rows
    # always pass — each is a single volunteer's potentially-unique find and
    # an agreement threshold doesn't apply.
    min_agreement_pct = config.zooniverse_min_agreement_pct
    gate_mask = (agg["species"] == "OTHER") | (
        agg["agreement_pct"] >= min_agreement_pct
    )
    passed = agg[gate_mask].copy()

    # Consensus-fish rule. For subjects where named-species claims existed
    # but none cleared the agreement gate, if a majority of voters still saw
    # SOMETHING (i.e., not NOTHINGHERE), emit one consensus row labelled
    # species="fish". Encodes "we agree there's a fish here, we don't agree
    # which" — a high-value signal for expert review that would otherwise be
    # lost as a scatter of weak per-species rows below the agreement floor.
    consensus_pct = config.zooniverse_consensus_something_here_pct
    subjects_with_named_passed = set(
        passed.loc[passed["species"] != "OTHER", "subject_id"]
    )
    subjects_with_named_attempt = (
        set(non_other_df["subject_id"]) - subjects_with_named_passed
    )

    consensus_rows = pd.DataFrame()
    if subjects_with_named_attempt:
        # Compute something_here_pct per candidate subject
        something_here_votes = total_classifiers - nothing_here.reindex(
            total_classifiers.index, fill_value=0
        )
        something_here_pct = (something_here_votes / total_classifiers * 100).round(1)
        # Dict lookup is unambiguous; Series[key] can return a Series when the
        # index has duplicates, which broke `pct > threshold` truth-checks.
        pct_by_subject = something_here_pct.to_dict()

        consensus_subject_ids = [
            sid
            for sid in subjects_with_named_attempt
            if pct_by_subject.get(sid, 0) > consensus_pct
        ]

        if consensus_subject_ids:
            # Aggregate across ALL named-species claims at each consensus subject
            base = non_other_df[non_other_df["subject_id"].isin(consensus_subject_ids)]
            consensus_rows = (
                base.groupby(["subject_id", "drop_id", "video_filename"], dropna=False)
                .agg(
                    mean_seconds=("absolute_seconds", "mean"),
                    mode_seconds=(
                        "absolute_seconds",
                        lambda x: (
                            x.dropna().mode().iloc[0] if not x.dropna().empty else None
                        ),
                    ),
                    mode_count=(
                        "count",
                        lambda x: (
                            int(x.dropna().mode().iloc[0])
                            if not x.dropna().empty
                            else 0
                        ),
                    ),
                    mean_count=("count", "mean"),
                    max_count=("count", "max"),
                    upl_seconds=("upl_seconds", "first"),
                    subject_set_id=("subject_set_id", "first"),
                    workflow_id=("workflow_id", "first"),
                    subject_locations=("subject_locations", "first"),
                )
                .reset_index()
            )
            consensus_rows["species"] = "fish"
            # vote_count = voters who saw something (any species, not NOTHINGHERE).
            # total_classifiers and nothing_here_votes come from subject-level aggs.
            consensus_rows = consensus_rows.join(total_classifiers, on="subject_id")
            consensus_rows = consensus_rows.join(nothing_here, on="subject_id")
            consensus_rows["nothing_here_votes"] = (
                consensus_rows["nothing_here_votes"].fillna(0).astype(int)
            )
            consensus_rows["vote_count"] = (
                consensus_rows["total_classifiers"]
                - consensus_rows["nothing_here_votes"]
            )
            consensus_rows["agreement_pct"] = (
                consensus_rows["vote_count"] / consensus_rows["total_classifiers"] * 100
            ).round(1)
            consensus_rows["suspicious_minority_find"] = False
            consensus_rows["count_disagreement"] = False

            # Ensure column order matches `passed` so concat is clean
            consensus_rows = consensus_rows[passed.columns]
            passed = pd.concat([passed, consensus_rows], ignore_index=True)

    n_other = int((passed["species"] == "OTHER").sum())
    n_consensus = len(consensus_rows)
    logging.info(
        f"Aggregated: {len(passed)} rows total "
        f"(agreement_pct >= {min_agreement_pct}% for named species, "
        f"all OTHER kept, consensus_pct > {consensus_pct}% for fish-consensus) "
        f"from {len(agg)} pre-gate; "
        f"{int(agg['suspicious_minority_find'].sum())} suspicious-minority, "
        f"{int(agg['count_disagreement'].sum())} count-disagreement, "
        f"{n_other} individual OTHER votes, "
        f"{n_consensus} consensus-fish rows."
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


def _parse_subject_set_display_name(
    display_name: str,
) -> tuple[str | None, str | None]:
    """
    Parse a subject set display_name into (subject_set_type, drop_id).

    upload.py writes display_name as ``clips_{drop_id}`` or ``frames_{drop_id}``.
    Returns (None, None) for any set that doesn't follow this convention
    (manually created sets, legacy sets uploaded before the convention).
    """
    for prefix in ("clips_", "frames_"):
        if display_name.startswith(prefix):
            candidate = display_name[len(prefix) :]
            try:
                return prefix.rstrip("_"), config.validate_drop_id(candidate)
            except ValueError:
                return None, None
    return None, None


def subject_completion_from_api() -> pd.DataFrame:
    """
    Per drop_id retirement completion from live Panoptes SubjectSet counts.
    Requires an active Panoptes connection (call connect_to_zooniverse() first).

    Reads drop_id and subject_set_type from the SubjectSet display_name
    (``clips_{drop_id}`` / ``frames_{drop_id}``) — O(num_sets), no per-subject
    iteration. Sets that predate the display_name convention get drop_id=None.

    Returns:
        DataFrame with columns: project_id, subject_set_id, subject_set_type,
        drop_id, total, retired, pct_retired, fully_complete.
    """
    from panoptes_client import SubjectSet

    rows = []
    n_unrecognised = 0
    for project_id in config.zooniverse_source_project_ids:
        logging.info(f"  Fetching subject sets for project {project_id}...")
        for ss in SubjectSet.where(project_id=project_id):
            subject_set_type, drop_id = _parse_subject_set_display_name(
                ss.display_name or ""
            )
            if drop_id is None:
                n_unrecognised += 1

            total = int(ss.raw.get("set_member_subjects_count") or 0)
            # Retirement progress lives in `completeness` — a dict keyed by
            # workflow_id, value is the fraction (0.0–1.0) of subjects retired
            # in that workflow. A subject set linked to multiple workflows is
            # retired once any workflow has classified everything, so we take
            # the max. The Panoptes API does NOT return a count field for
            # retired subjects directly.
            completeness = ss.raw.get("completeness") or {}
            pct_retired = max(completeness.values()) * 100 if completeness else 0.0
            retired = int(round(total * pct_retired / 100))
            rows.append(
                {
                    "project_id": project_id,
                    "subject_set_id": ss.id,
                    "subject_set_type": subject_set_type,
                    "drop_id": drop_id,
                    "total": total,
                    "retired": retired,
                    "pct_retired": round(pct_retired, 1),
                }
            )

    if n_unrecognised:
        logging.info(
            f"  {n_unrecognised} subject set(s) have non-standard display names "
            "(manually created or pre-convention) — drop_id set to None for those."
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # fully_complete must require an actual non-empty set; otherwise
    # `retired == total` is True for `0 == 0` and an empty/scaffold set
    # falsely appears as "done".
    df["fully_complete"] = (df["total"] > 0) & (df["pct_retired"] >= 100.0)
    return df.sort_values("pct_retired", ascending=False).reset_index(drop=True)


# ── Zooniverse choice → scientific name mapping ──────────────────────────────


def _zoo_choice_to_scientific(choice: str) -> Optional[str]:
    """Resolve a Zooniverse choice key to a scientific name.

    Returns the mapped scientific name when known. Returns ``"fish"`` as the
    generic fallback for choices that don't map (e.g. ``OTHER``, or a legacy
    common-name variant we haven't catalogued) — the volunteer still saw a
    fish, just couldn't identify it. Matches the binary-fish-class floor the
    ML model uses for rare species. Returns ``None`` only for empty/blank input.
    """
    if not choice:
        return None
    return load_species_labels().zoo_choice_to_scientific.get(
        normalise_zoo_choice(choice), "fish"
    )


# ── Phase 4 — MaxN CSV export ────────────────────────────────────────────────


def zooniverse_maxn_columns() -> list[str]:
    """Ordered column list for a Zooniverse MaxN CSV.

    ``subject_id`` is included so each row traces back to the Zooniverse
    clip it came from — feeds the annotations-DB ``external_id`` field on
    citsci rows, which the dashboard surfaces as Provenance.
    """
    return [
        config.drop_id_column,
        config.csv_subject_id_column,
        config.csv_scientific_name_column,
        config.csv_maxn_time_column,
        config.csv_max_interval_column,
        config.csv_annotated_by_column,
        config.csv_interval_annotation_column,
        config.csv_confidence_agreement_column,
        config.csv_maxn_time_seconds_column,
    ]


def write_empty_zooniverse_maxn_csv(drop_id: str) -> None:
    """Write a header-only MaxN CSV for a fully-retired all-NOTHINGHERE drop.

    Without this file, sync_zooniverse_drop finds no CSV and the drop stays
    stuck at citsci_frames_uploaded even though volunteers saw nothing.
    """
    out_path = config.get_zooniverse_maxn_csv_path(drop_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=zooniverse_maxn_columns()).to_csv(out_path, index=False)
    logging.info(f"Empty MaxN CSV (all-NOTHINGHERE) → {out_path}")


def write_zooniverse_maxn_csv(aggregated_df: pd.DataFrame) -> None:
    """Write one MaxN CSV per drop_id from the aggregated volunteer consensus.

    Rows with suspicious_minority_find=True are excluded from the export —
    they remain in the DB audit trail but should not drive BIIGLE frame selection.
    Expects the same DataFrame shape produced by aggregate_by_subject_species.
    """
    export_df = aggregated_df[
        aggregated_df["drop_id"].notna() & ~aggregated_df["suspicious_minority_find"]
    ].copy()

    if export_df.empty:
        logging.warning(
            "write_zooniverse_maxn_csv: no rows to export (all filtered or drop_id=None)."
        )
        return

    n_skipped_null = 0
    n_mapped_to_fish = 0
    for drop_id, grp in export_df.groupby("drop_id"):
        rows = []
        for _, row in grp.iterrows():
            scientific = _zoo_choice_to_scientific(row["species"])
            if scientific is None:
                n_skipped_null += 1
                continue
            if scientific == "fish":
                n_mapped_to_fish += 1
            rows.append(
                {
                    config.drop_id_column: drop_id,
                    config.csv_subject_id_column: row.get("subject_id"),
                    config.csv_scientific_name_column: scientific,
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
        if not rows:
            logging.info(f"  {drop_id}: no usable rows — skipping MaxN CSV.")
            continue
        out_path = config.get_zooniverse_maxn_csv_path(drop_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values(config.csv_maxn_time_column).to_csv(
            out_path, index=False
        )
        logging.info(f"MaxN CSV → {out_path} ({len(rows)} rows)")

    if n_mapped_to_fish:
        logging.info(
            f"write_zooniverse_maxn_csv: {n_mapped_to_fish} row(s) mapped to "
            "generic 'fish' fallback (choice was OTHER or an unmapped variant)."
        )
    if n_skipped_null:
        logging.info(
            f"write_zooniverse_maxn_csv: {n_skipped_null} row(s) skipped — "
            "blank/null species choice."
        )


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

    Clears any previous citsci annotations for this drop before writing, so
    re-running is safe. `sync_annotation_counts([drop_id])` advances
    citsci_status → citsci_complete automatically when count > 0. The
    all-NOTHINGHERE case (empty CSV, fully retired with zero positive
    findings) needs an explicit advance since the data-presence rule wouldn't
    trigger — that's still a valid completion, just with no observations.

    Returns:
        Number of annotation rows ingested (0 if CSV not found or empty).
    """
    from spyfish.config.base import CitSciStatus
    from spyfish.database.annotation_manager import AnnotationDatabaseManager
    from spyfish.database.manager import DatabaseManager as PipelineDB

    maxn_csv = config.get_zooniverse_maxn_csv_path(drop_id)
    if not maxn_csv.exists():
        logging.info(
            f"ingest_zooniverse: No MaxN CSV found for {drop_id} at {maxn_csv}"
        )
        return 0

    ann_db = AnnotationDatabaseManager()
    ann_db.clear_annotations(drop_id, "citsci")

    df = pd.read_csv(maxn_csv)
    pipeline_db = PipelineDB()

    if df.empty:
        logging.info(
            f"ingest_zooniverse: MaxN CSV is empty for {drop_id} (all-NOTHINGHERE)"
        )
        pipeline_db.sync_annotation_counts([drop_id])
        # Confirmed-empty review still completes the citsci stage —
        # data-presence rule in sync_annotation_counts won't trigger
        # because count = 0, so advance explicitly.
        pipeline_db.bulk_update_section_status(
            [drop_id],
            CitSciStatus.COLUMN,
            CitSciStatus.COMPLETE,
            skip_if_in=[CitSciStatus.COMPLETE],
        )
        return 0

    annotations = []
    for _, row in df.iterrows():
        annotations.append(
            {
                "drop_id": drop_id,
                "scientific_name": row[config.csv_scientific_name_column],
                "time_of_max": row[config.csv_maxn_time_column],
                "time_of_max_seconds": row.get(config.csv_maxn_time_seconds_column),
                "max_interval": row[config.csv_max_interval_column],
                "annotated_by": "citsci",
                "interval_annotation": row.get(
                    config.csv_interval_annotation_column, ""
                ),
                "confidence_agreement": row.get(config.csv_confidence_agreement_column),
                # subject_id is the Zooniverse clip identifier — stored as
                # the citsci external_id so any row can be traced back to its
                # source clip on zooniverse.org/subjects/<id>. Older MaxN
                # CSVs predating this column read as NaN; coerce to None.
                "external_id": (
                    str(row[config.csv_subject_id_column])
                    if config.csv_subject_id_column in row
                    and pd.notna(row[config.csv_subject_id_column])
                    else None
                ),
            }
        )

    ann_db.add_annotations(annotations)
    logging.info(
        f"ingest_zooniverse: Stored {len(annotations)} citsci annotations for {drop_id}"
    )

    pipeline_db.sync_annotation_counts([drop_id])

    return len(annotations)
