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
import threading
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


def _json_safe(meta: dict) -> dict:
    """Blank out NaN / None so Panoptes accepts the metadata.

    A manifest column that is empty on SOME rows (a blind frame's target
    species, a boxless frame's mean confidence) comes out of the DataFrame as
    NaN, and Panoptes posts metadata with ``allow_nan=False``: the whole
    subject save fails with "Out of range float values are not JSON
    compliant" (HOR trial, 2026-10-16). Blank is what those values mean.
    """
    return {
        key: (
            ""
            if value is None or (not isinstance(value, str) and pd.isna(value))
            else value
        )
        for key, value in meta.items()
    }


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


def _existing_identities(subjects, kind: SubjectKind) -> tuple[set, int, int]:
    """(identities present, subjects without a readable identity, total subjects).

    `subjects` is the set's subject listing, already fetched (`ExistingSet`):
    paging a set is one request per page, so it is done once per drop.
    """
    present, unreadable, total = set(), 0, 0
    for subject in subjects:
        total += 1
        ident = _identity(getattr(subject, "metadata", None) or {}, kind.identity_keys)
        if ident is None:
            unreadable += 1
        else:
            present.add(ident)
    return present, unreadable, total


# ── one login and one project lookup per process ─────────────────────────────

# Thread-local because panoptes_client keeps its logged-in client per thread.
# The client refreshes its own bearer token, so one login lasts a whole run.
_session = threading.local()


def _connect() -> None:
    """Log in to Panoptes once per process (and per set of credentials).

    A stage run uploads many drops; logging in again for each one only repeats
    the same sign-in requests.
    """
    credentials = (config.user, config.password)
    if getattr(_session, "credentials", None) == credentials:
        return
    logging.info(f"Connecting to Zooniverse as {config.user}...")
    Panoptes.connect(username=config.user, password=config.password)
    _session.credentials = credentials
    _session.projects = {}


def _upload_project():
    """The upload target project, looked up once per login."""
    _connect()
    project_id = config.zooniverse_project_id
    if project_id not in _session.projects:
        _session.projects[project_id] = Project.find(project_id)
    return _session.projects[project_id]


def reset_session() -> None:
    """Forget the cached login and project (tests, or after a credential change)."""
    _session.__dict__.clear()


@dataclass(frozen=True)
class ExistingSet:
    """A subject set already on Zooniverse and its subjects, listed once.

    The stage's "already uploaded?" check and the uploader's resume both need
    the set's subjects. Whoever lists them first hands this to the other.
    """

    subject_set: object
    subjects: list


def fetch_existing_set(kind: SubjectKind, drop_id: str) -> ExistingSet | None:
    """This drop's set of `kind` on Zooniverse with its subjects, or None if no set.

    One query for the set and one page walk over its subjects, so a stage can
    skip all rendering for a drop that is already up (a retry after an error,
    or a re-run). Pass the result to the upload as `existing` so the set is
    not looked up and paged a second time.
    """
    _connect()
    return _find_set(config.zooniverse_project_id, f"{kind.set_name_prefix}{drop_id}")


def _find_set(project_id, set_name: str) -> ExistingSet | None:
    sets = list(SubjectSet.where(project_id=project_id, display_name=set_name))
    if not sets:
        return None
    return ExistingSet(subject_set=sets[0], subjects=list(sets[0].subjects))


def _upload_subjects_to_zooniverse(
    df: pd.DataFrame,
    kind: SubjectKind,
    subject_set_name: str | None,
    test_upload: bool,
    existing: ExistingSet | None = None,
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

    `existing` is the set as `fetch_existing_set` already found it in this
    run; given, the set is neither looked up nor paged again. Without it (or
    when the earlier check found no set) the lookup happens here, so two
    machines racing on one drop still cannot create the set twice.
    """
    if not all([config.user, config.password, config.zooniverse_project_id]):
        raise EnvironmentError(
            "Missing Zooniverse credentials. Set ZOONIVERSE_USER, ZOONIVERSE_PASSWORD, "
            "and ZOONIVERSE_PROJECT_ID in your .env file."
        )

    uploadable = df[df[kind.path_column].notna()].copy()
    # Nothing to upload raises rather than returns: every caller advances the
    # drop to *_UPLOADED when this returns, so a silent return recorded a set
    # on Zooniverse that does not exist and the drop was never retried.
    if uploadable.empty:
        raise RuntimeError(
            f"No {kind.noun_plural} available to upload "
            f"({kind.path_column} is empty for all rows)."
        )

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
        raise FileNotFoundError(
            f"None of the {kind.noun_plural} files exist on disk "
            f"({len(exists)} listed in {kind.path_column})."
        )

    if test_upload:
        uploadable = uploadable.iloc[:1]
        logging.info("test_upload=True, limiting to 1 subject.")

    drop_id = uploadable["DropID"].iloc[0]
    n = len(uploadable)
    site_id = config.get_site_id_from_drop(drop_id)
    video_filename = config.get_video_s3_key(drop_id)
    site_reserve_meta = _get_site_reserve_meta(site_id)

    zoo_project = _upload_project()

    set_name = subject_set_name or f"{kind.set_name_prefix}{drop_id}"

    # Every subject's metadata first: it is also the subject's identity within
    # the set, which is how a re-run tells "already there" from "missing".
    planned = []
    for _, row in uploadable.iterrows():
        meta = _json_safe(
            {
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
        )
        planned.append((row, meta))

    if existing is None:
        existing = _find_set(zoo_project.id, set_name)
    if existing is not None:
        subject_set = existing.subject_set
        present, unreadable, existing_count = _existing_identities(
            existing.subjects, kind
        )
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
    # Within a chunk the saves run on panoptes_client's own thread pool
    # (`Subject.async_saves`): the same requests, several in flight at once.
    linked = []
    for start in range(0, len(planned), _LINK_CHUNK):
        chunk = planned[start : start + _LINK_CHUNK]
        batch = []
        with Subject.async_saves():
            for row, meta in chunk:
                subject = Subject()
                subject.links.project = zoo_project
                subject.add_location(
                    str(row[kind.path_column]), manual_mimetype=kind.mimetype
                )
                subject.metadata.update(meta)
                subject.save()  # queued; the `with` waits for the whole chunk
                batch.append(subject)
        for subject, (row, _) in zip(batch, chunk):
            file_name = Path(row[kind.path_column]).name
            # A failed save re-raises here. Nothing of this chunk is linked
            # then, exactly as when a one-at-a-time save raised mid-chunk.
            if not subject.async_save_result:
                raise RuntimeError(
                    f"{file_name}: subject save did not complete; "
                    f"'{set_name}' is partial, re-run to resume."
                )
            logging.info(f"  Saved: {file_name} ({row.get('SelectionReason', '')})")
        subject_set.add(batch)
        linked.extend(batch)
    logging.info(
        f"Uploaded {len(linked)}/{n} {kind.noun_plural} to subject set '{set_name}'."
    )
    if kind.workflow_hint:
        logging.warning(
            f"Attach subject set '{set_name}' to {kind.workflow_hint} in the "
            "Zooniverse project builder. An unattached set is shown to nobody."
        )


# ── public entry points ──────────────────────────────────────────────────────


def record_clip_uploads(drop_id: str, clips_df: pd.DataFrame, uploaded_at: str) -> Path:
    """Append this run's clips to the drop's upload record, keyed on clip start.

    Call only after the upload returned: it raises on failure, so a clip in
    the record is a clip on Zooniverse. A clip already recorded keeps its
    first `UploadedAt`, since the upload skipped it as already present. Clips
    uploaded before this record existed get the date of the first run that
    logs them, not their real upload date.
    """
    path = config.get_clip_uploads_csv_path(drop_id)
    start_col = config.csv_clip_start_absolute_column
    fresh = clips_df[clips_df[CLIP_KIND.path_column].notna()].drop(
        columns=[CLIP_KIND.path_column]
    )
    fresh = fresh.assign(UploadedAt=uploaded_at)
    if path.exists():
        fresh = pd.concat([pd.read_csv(path), fresh], ignore_index=True)
    record = (
        fresh.assign(_key=fresh[start_col].astype(float).round(3))
        .drop_duplicates("_key", keep="first")
        .drop(columns="_key")
        .sort_values(start_col)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record.to_csv(path, index=False)
    logging.info(f"Clip upload record: {len(record)} clip(s) in {path.name}")
    return path


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

    Counts the subjects actually listed. For many drops at once, read the
    counts off one subject-set listing instead (`--db-refresh` does); to go on
    and upload, use `fetch_existing_set` and pass its result along.
    """
    found = fetch_existing_set(kind, drop_id)
    return len(found.subjects) if found else None


def upload_validation_frames_to_zooniverse(
    manifest_df: pd.DataFrame, existing: ExistingSet | None = None
) -> None:
    """Upload a drop's validation frames (every ML box drawn) as ``vframes_{drop}``.

    `manifest_df` is from `validation_frames.render_validation_frames`. The set
    is NOT attached to a workflow, attach it by hand, as for clips. `existing`
    is the stage's own `fetch_existing_set` result, if it found the set.
    """
    _upload_subjects_to_zooniverse(
        manifest_df, VALIDATION_FRAME_KIND, None, False, existing
    )


def upload_validation_crops_to_zooniverse(
    manifest_df: pd.DataFrame, existing: ExistingSet | None = None
) -> None:
    """Upload a drop's validation crops (one per ML box) as ``crops_{drop}``.

    `manifest_df` is from `validation_frames.render_validation_crops`. Rows
    with no CropPath (boxes not cropped) are skipped by the shared loop.
    `existing` is the stage's own `fetch_existing_set` result, if it found the set.
    """
    _upload_subjects_to_zooniverse(manifest_df, CROP_KIND, None, False, existing)
