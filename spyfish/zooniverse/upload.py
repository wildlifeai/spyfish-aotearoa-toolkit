"""
Upload extracted Zooniverse clips or frames to a Zooniverse project as a new subject set.

The two public entry points, upload_clips_to_zooniverse and
upload_frames_to_zooniverse, share the same credential check, subject-set
existence check, and per-row upload loop. The per-type differences (path
column, mimetype, subject set name prefix, upload-time computation) are
captured in a small SubjectKind dataclass and consumed by the internal
`_upload_subjects_to_zooniverse` helper.

Idempotency contract: if a subject set with the target display name already
exists in the project, count its subjects. Match the expected upload count
→ skip cleanly (drop is already on Zooniverse). Mismatch → raise (state is
inconsistent, needs human investigation).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from panoptes_client import Panoptes, Project, Subject, SubjectSet

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.zooniverse.subject_keys import (
    CROP_PATH_COLUMN,
    FRAME_PATH_COLUMN,
    SubjectKeys,
)

# The manifest writer owns these column names; the upload reads the same ones.

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_site_reserve_meta(site_id: str) -> dict:
    """Look up !LinkToMarineReserve and ProtectionStatus for a SiteID from the pipeline DB."""
    site = DatabaseManager().get_site(site_id)
    if not site:
        logging.warning(
            f"No site found for SiteID '{site_id}' in DB, reserve metadata omitted. Run ingest first."
        )
        return {}
    return {
        SubjectKeys.LINK_TO_RESERVE: site.get(config.link_to_marine_reserve_column, ""),
        "ProtectionStatus": site.get(config.protection_status_column, ""),
    }


def _build_base_subject_meta(
    row,
    drop_id: str,
    video_filename: str,
    site_id: str,
    site_reserve_meta: dict,
    subject_type: str,
    blank_ok: tuple = (),
) -> dict:
    """Common Zooniverse subject metadata shared between clips and frames uploads.

    `subject_type` ("clip", "frame" or "crop") is written to #SubjectType on
    each subject so parse_classifications and any future retirement-gate check
    can tell them apart without relying on subject set display names.

    `blank_ok` names keys that are legitimately empty for this kind (a blind
    validation frame has no target species), so only unexpected blanks warn.
    """
    meta = {
        "DropID": drop_id,
        SubjectKeys.VIDEO_FILENAME: video_filename,
        SubjectKeys.SITE_NAME: site_id,
        SubjectKeys.SUBJECT_TYPE: subject_type,
        "#SelectionReason": row.get(config.selection_reason_column, ""),
        "#TargetSpecies": row.get(config.csv_scientific_name_column, ""),
        "#MaxInterval": row.get(config.csv_max_interval_column, ""),
        "#ConfidenceAgreement": row.get(config.csv_confidence_agreement_column, ""),
        "#SamplingStart": row[config.csv_sampling_start_column],
        **site_reserve_meta,
    }
    empty_keys = [
        k for k, v in meta.items() if (v == "" or pd.isna(v)) and k not in blank_ok
    ]
    if empty_keys:
        logging.warning(
            f"Subject metadata has empty fields: {empty_keys}. Row columns: {list(row.index)}"
        )
    return meta


# ── subject kind descriptor + shared upload loop ─────────────────────────────

# Subjects saved per SubjectSet.add call. Small enough that a crash loses little
# linking work, large enough not to make one API call per subject.
_LINK_CHUNK = 25


@dataclass(frozen=True)
class SubjectKind:
    """All the per-subject-type differences between clips and frames uploads.

    Centralised here so the shared upload loop can iterate generically, and
    so adding a future subject type (e.g. an annotated-frame variant) is a
    single new SubjectKind instance, not another 100-line copy-paste.
    """

    noun_singular: str  # "clip" / "frame", for logs and #SubjectType
    noun_plural: str  # "clips" / "frames", for logs
    path_column: str  # DataFrame column holding the local file path
    set_name_prefix: str  # "clips_" / "frames_"
    mimetype: str  # Passed to Subject.add_location
    upl_seconds_fn: Callable  # (row) -> int | float. UplAbsSeconds value
    # (row) -> dict of kind-specific hidden keys, merged over the base meta.
    extra_meta_fn: Callable | None = None
    blank_ok: tuple = ()  # base-meta keys allowed to be empty for this kind
    # Which workflow the set belongs on. Sets are attached by hand in the project
    # builder, like clips, and an unattached set is shown to nobody, so the
    # upload ends by saying where this one goes.
    workflow_hint: str = ""
    # Metadata keys that identify one subject within its set. A re-run reads
    # them off the subjects already on Zooniverse and uploads only the rows
    # whose identity is missing, so a crash mid-upload resumes instead of
    # refusing, and a second run can never add a duplicate. Empty means the
    # kind cannot be matched and the count rule applies.
    identity_keys: tuple = ()


def _compute_clip_upl_seconds(row) -> int:
    return int(float(row[config.csv_clip_start_absolute_column]))


def _compute_frame_upl_seconds(row) -> float:
    """Frames: the ClipMaxTime column already holds the absolute seek time."""
    return float(row[config.csv_clip_max_time_column])


CLIP_KIND = SubjectKind(
    noun_singular="clip",
    noun_plural="clips",
    path_column="ClipPath",
    set_name_prefix="clips_",
    mimetype="video/mp4",
    upl_seconds_fn=_compute_clip_upl_seconds,
    identity_keys=(SubjectKeys.VIDEO_FILENAME, SubjectKeys.UPL_SECONDS),
)

FRAME_KIND = SubjectKind(
    noun_singular="frame",
    noun_plural="frames",
    path_column="FramePath",
    set_name_prefix="frames_",
    mimetype="image/jpeg",
    upl_seconds_fn=_compute_frame_upl_seconds,
    identity_keys=(SubjectKeys.VIDEO_FILENAME, SubjectKeys.UPL_SECONDS),
)


# ── validation kinds (--zooniverse-frames / --zooniverse-crops) ──────────────

#
# Manifests come from spyfish/zooniverse/validation_frames.py and use the same
# column names as the clip CSVs, so the shared base meta reads them unchanged.
# Extra keys are all `#`-hidden: they let a returning classification be joined
# back to the model, frame and (for crops) the exact box it judged.


def _validation_frame_meta(row) -> dict:
    return {
        "#ModelName": row["ModelName"],
        "#FrameIndex": int(row["FrameIndex"]),
        "#RenderScale": float(row["RenderScale"]),
    }


def _crop_meta(row) -> dict:
    return {
        "#ModelName": row["ModelName"],
        "#FrameIndex": int(row["FrameIndex"]),
        # drop|frame|box index: joins an answer to exactly one ML box.
        "#DetectionKey": row["DetectionKey"],
        "#BBox": row["BBox"],  # x1,y1,x2,y2 in video pixels
        "#LabelShown": row["LabelShown"],
    }


# `vframes_`, not `frames_`: old frames_* sets already exist on Zooniverse, and
# the existing-set count check would treat a name clash as inconsistent state.
VALIDATION_FRAME_KIND = SubjectKind(
    noun_singular="frame",
    noun_plural="validation frames",
    path_column=FRAME_PATH_COLUMN,
    set_name_prefix="vframes_",
    mimetype="image/jpeg",
    upl_seconds_fn=_compute_frame_upl_seconds,
    extra_meta_fn=_validation_frame_meta,
    # Blind frames have no target species, and a frame with no boxes has no
    # mean confidence. Both are the point of a recall check, not missing data.
    blank_ok=("#TargetSpecies", "#ConfidenceAgreement"),
    workflow_hint="the missed-fish workflow (boxes right? fish without a box? draw them)",
    identity_keys=("#ModelName", "#FrameIndex"),
)

CROP_KIND = SubjectKind(
    noun_singular="crop",
    noun_plural="crops",
    path_column=CROP_PATH_COLUMN,
    set_name_prefix="crops_",
    mimetype="image/jpeg",
    upl_seconds_fn=_compute_frame_upl_seconds,
    extra_meta_fn=_crop_meta,
    workflow_hint="the species-check workflow (right species, in the box? Yes/No)",
    identity_keys=("#DetectionKey",),
)


def _identity(meta: dict, keys: tuple) -> "tuple | None":
    """A subject's identity within its set, from its metadata; None if unreadable.

    Numbers are normalised to three decimals so `650`, `650.0` and `"650"`
    (Zooniverse hands metadata back as strings) compare equal; that string
    drift is how re-runs once added the same clip a second time.
    """
    if not keys:
        return None
    out = []
    for key in keys:
        value = meta.get(key)
        if value is None or value == "":
            return None
        try:
            out.append(f"{float(value):.3f}")
        except (TypeError, ValueError):
            out.append(str(value).strip())
    return tuple(out)


def _existing_identities(subject_set, kind: SubjectKind) -> tuple[set, int, int]:
    """(identities present, subjects without a readable identity, total subjects)."""
    present, unreadable, total = set(), 0, 0
    for subject in subject_set.subjects:
        total += 1
        ident = _identity(getattr(subject, "metadata", None) or {}, kind.identity_keys)
        if ident is None:
            unreadable += 1
        else:
            present.add(ident)
    return present, unreadable, total


def _upload_subjects_to_zooniverse(
    df: pd.DataFrame,
    kind: SubjectKind,
    subject_set_name: str | None,
    test_upload: bool,
) -> None:
    """Shared upload loop for clips and frames.

    Responsible for: credential check, filtering uploadable rows, connecting
    to Panoptes, checking for an existing subject set, creating the set if
    absent, and building per-row subject metadata. The only per-type logic
    is pulled from `kind`.

    An existing set is RESUMED, not refused: each planned subject's identity
    (`kind.identity_keys`, read from the metadata already on Zooniverse) is
    compared with what the set holds, and only the missing ones are uploaded.
    A crash mid-upload therefore finishes on the next run, and a second run
    never adds a duplicate. Only when the existing subjects carry no identity
    metadata (sets from before the keys existed) does the old count rule
    apply: same count skips, a different count raises for a human to look.
    """
    if not all([config.user, config.password, config.zooniverse_project_id]):
        raise EnvironmentError(
            "Missing Zooniverse credentials. Set ZOONIVERSE_USER, ZOONIVERSE_PASSWORD, "
            "and ZOONIVERSE_PROJECT_ID in your .env file."
        )

    uploadable = df[df[kind.path_column].notna()].copy()
    if uploadable.empty:
        logging.error(
            f"No {kind.noun_plural} available to upload "
            f"({kind.path_column} is empty for all rows)."
        )
        return

    # Missing files are dropped BEFORE counting. Counting first made a set with
    # one missing file come up one short on every later run, which the
    # existing-set check below then treats as inconsistent state.
    exists = uploadable[kind.path_column].map(lambda p: Path(p).exists())
    for p in uploadable.loc[~exists, kind.path_column]:
        logging.warning(
            f"{kind.noun_singular.capitalize()} file missing, skipping: {Path(p).name}"
        )
    uploadable = uploadable[exists]
    if uploadable.empty:
        logging.error(f"None of the {kind.noun_plural} files exist on disk.")
        return

    if test_upload:
        uploadable = uploadable.iloc[:1]
        logging.info("test_upload=True, limiting to 1 subject.")

    drop_id = uploadable["DropID"].iloc[0]
    n = len(uploadable)
    site_id = config.get_site_id_from_drop(drop_id)
    video_filename = config.get_video_s3_key(drop_id)
    site_reserve_meta = _get_site_reserve_meta(site_id)

    logging.info(f"Connecting to Zooniverse as {config.user}...")
    Panoptes.connect(username=config.user, password=config.password)
    zoo_project = Project.find(config.zooniverse_project_id)

    set_name = subject_set_name or f"{kind.set_name_prefix}{drop_id}"

    # Every subject's metadata first: it is also the subject's identity within
    # the set, which is how a re-run tells "already there" from "missing".
    planned = []
    for _, row in uploadable.iterrows():
        meta = {
            **_build_base_subject_meta(
                row,
                drop_id,
                video_filename,
                site_id,
                site_reserve_meta,
                subject_type=kind.noun_singular,
                blank_ok=kind.blank_ok,
            ),
            SubjectKeys.UPL_SECONDS: kind.upl_seconds_fn(row),
            **(kind.extra_meta_fn(row) if kind.extra_meta_fn else {}),
        }
        planned.append((row, meta))

    existing_sets = list(
        SubjectSet.where(project_id=zoo_project.id, display_name=set_name)
    )
    if existing_sets:
        subject_set = existing_sets[0]
        present, unreadable, existing_count = _existing_identities(subject_set, kind)
        if unreadable:
            # Pre-identity subjects: the count is all there is to go on.
            if existing_count == n:
                logging.info(
                    f"Subject set '{set_name}' already exists with {existing_count} "
                    f"{kind.noun_plural} (matches expected). Skipping upload, drop "
                    f"already on Zooniverse."
                )
                return
            raise RuntimeError(
                f"Subject set '{set_name}' already exists with {existing_count} "
                f"subjects but {n} {kind.noun_plural} were expected, and "
                f"{unreadable} of its subjects carry no identity metadata "
                f"({kind.identity_keys}), so the missing ones cannot be told "
                "apart. State is inconsistent, investigate in Zooniverse Lab "
                "before re-running."
            )
        planned = [
            (row, meta)
            for row, meta in planned
            if _identity(meta, kind.identity_keys) not in present
        ]
        if not planned:
            logging.info(
                f"Subject set '{set_name}' already holds all {n} "
                f"{kind.noun_plural}. Nothing to upload."
            )
            return
        logging.warning(
            f"Subject set '{set_name}' exists with {existing_count} of {n} "
            f"{kind.noun_plural}; resuming, uploading the {len(planned)} missing "
            f"one(s). Subjects are matched on {kind.identity_keys}, so nothing "
            "already there is re-uploaded."
        )
    else:
        subject_set = SubjectSet()
        subject_set.links.project = zoo_project
        subject_set.display_name = set_name
        subject_set.save()
        logging.info(f"Subject set created: '{set_name}'")

    # Linked to the set in chunks as they are saved, not in one call at the end:
    # a crash mid-run used to leave every saved subject orphaned outside the set.
    linked, pending = [], []

    def _link() -> None:
        batch = list(pending)  # a copy: `pending` is reused for the next chunk
        subject_set.add(batch)
        linked.extend(batch)
        pending.clear()

    for row, meta in planned:
        file_path = Path(row[kind.path_column])
        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location(str(file_path), manual_mimetype=kind.mimetype)
        subject.metadata.update(meta)
        subject.save()
        pending.append(subject)
        logging.info(f"  Saved: {file_path.name} ({row.get('SelectionReason', '')})")
        if len(pending) >= _LINK_CHUNK:
            _link()
    if pending:
        _link()
    logging.info(
        f"Uploaded {len(linked)}/{n} {kind.noun_plural} to subject set '{set_name}'."
    )
    if kind.workflow_hint:
        logging.warning(
            f"Attach subject set '{set_name}' to {kind.workflow_hint} in the "
            "Zooniverse project builder. An unattached set is shown to nobody."
        )


# ── public entry points ──────────────────────────────────────────────────────


def upload_clips_to_zooniverse(
    selections_df: pd.DataFrame,
    subject_set_name: str | None = None,
    test_upload: bool = False,
) -> None:
    """
    Uploads extracted clips to Zooniverse as a new subject set.

    Reads clip paths and metadata from selections_df (produced by
    extract_clips_from_selections). DropID and ClipPath are read from the
    DataFrame directly.

    Metadata attached to each subject:
        '#' prefix, hidden from volunteers, used for traceability and QA.
        '!' prefix, visible to volunteers in Talk (discussion) only.

        #DropID, #VideoFilename, #siteName, #SubjectType,
        #SelectionReason, #TargetSpecies, #MaxInterval, #ConfidenceAgreement,
        #SamplingStart, UplAbsSeconds,
        !LinkToMarineReserve, ProtectionStatus (when BUV Sites CSV is reachable on S3)
    """
    _upload_subjects_to_zooniverse(
        selections_df, CLIP_KIND, subject_set_name, test_upload
    )


def upload_frames_to_zooniverse(
    frames_df: pd.DataFrame,
    subject_set_name: str | None = None,
    test_upload: bool = False,
) -> None:
    """
    Uploads extracted frames to Zooniverse as a new subject set.

    Reads frame paths and metadata from frames_df (produced by
    extract_frames_from_selections). DropID and FramePath are read from the
    DataFrame directly.
    """
    _upload_subjects_to_zooniverse(frames_df, FRAME_KIND, subject_set_name, test_upload)


def existing_subject_count(kind: SubjectKind, drop_id: str) -> int | None:
    """Subjects already in this drop's set of `kind` on Zooniverse, or None if no set.

    One small metadata query, so a stage can skip all rendering for a drop
    that is already up (a retry after an error, or a re-run).
    """
    Panoptes.connect(username=config.user, password=config.password)
    sets = list(
        SubjectSet.where(
            project_id=config.zooniverse_project_id,
            display_name=f"{kind.set_name_prefix}{drop_id}",
        )
    )
    return sum(1 for _ in sets[0].subjects) if sets else None


def upload_validation_frames_to_zooniverse(manifest_df: pd.DataFrame) -> None:
    """Upload a drop's validation frames (every ML box drawn) as ``vframes_{drop}``.

    `manifest_df` is from `validation_frames.render_validation_frames`. The set
    is NOT attached to a workflow, attach it by hand, as for clips.
    """
    _upload_subjects_to_zooniverse(manifest_df, VALIDATION_FRAME_KIND, None, False)


def upload_validation_crops_to_zooniverse(manifest_df: pd.DataFrame) -> None:
    """Upload a drop's validation crops (one per ML box) as ``crops_{drop}``.

    `manifest_df` is from `validation_frames.render_validation_crops`. Rows
    with no CropPath (boxes not cropped) are skipped by the shared loop.
    """
    _upload_subjects_to_zooniverse(manifest_df, CROP_KIND, None, False)
