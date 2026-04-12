"""
Upload extracted Zooniverse clips or frames to a Zooniverse project as a new subject set.

The two public entry points — upload_clips_to_zooniverse and
upload_frames_to_zooniverse — share the same credential check, subject-set
setup, already-uploaded filtering, and per-row upload loop. The per-type
differences (path column, mimetype, subject set name prefix, upload-time
computation) are captured in a small SubjectKind dataclass and consumed by
the internal `_upload_subjects_to_zooniverse` helper.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from panoptes_client import Panoptes, Project, Subject, SubjectSet

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.zooniverse.subject_keys import SubjectKeys

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_site_reserve_meta(site_id: str) -> dict:
    """Look up !LinkToMarineReserve and ProtectionStatus for a SiteID from the pipeline DB."""
    site = DatabaseManager().get_site(site_id)
    if not site:
        logging.warning(
            f"No site found for SiteID '{site_id}' in DB — reserve metadata omitted. Run ingest first."
        )
        return {}
    return {
        SubjectKeys.LINK_TO_RESERVE: site.get(config.link_to_marine_reserve_column, ""),
        "ProtectionStatus": site.get(config.protection_status_column, ""),
    }


def _get_uploaded_filenames(subject_set) -> set:
    """Return the set of filenames already uploaded to a subject set.

    Used to skip re-uploading on interrupted runs. Panoptes paginates — we
    exhaust the iterator to get all subjects.
    """
    uploaded = set()
    for subject in Subject.where(subject_set_id=subject_set.id):
        for loc in subject.raw.get("locations", []):
            # loc is {"image/jpeg": "https://..."} or {"video/mp4": "..."}
            for url in loc.values():
                uploaded.add(
                    url.split("/")[-1].split("?")[0]
                )  # strip path + query string
    if uploaded:
        logging.info(
            f"Found {len(uploaded)} already-uploaded subjects in set — will skip duplicates."
        )
    return uploaded


def _build_base_subject_meta(
    row,
    drop_id: str,
    video_filename: str,
    site_id: str,
    site_reserve_meta: dict,
    subject_type: str,
) -> dict:
    """Common Zooniverse subject metadata shared between clips and frames uploads.

    `subject_type` must be "clip" or "frame" — written to #SubjectType on each
    subject so parse_classifications and any future retirement-gate check can
    tell them apart without relying on subject set display names.
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
    empty_keys = [k for k, v in meta.items() if v == "" or pd.isna(v)]
    if empty_keys:
        logging.warning(
            f"Subject metadata has empty fields: {empty_keys}. Row columns: {list(row.index)}"
        )
    return meta


# ── subject kind descriptor + shared upload loop ─────────────────────────────


@dataclass(frozen=True)
class SubjectKind:
    """All the per-subject-type differences between clips and frames uploads.

    Centralised here so the shared upload loop can iterate generically, and
    so adding a future subject type (e.g. an annotated-frame variant) is a
    single new SubjectKind instance, not another 100-line copy-paste.
    """

    noun_singular: str  # "clip" / "frame" — for logs and #SubjectType
    noun_plural: str  # "clips" / "frames" — for logs
    path_column: str  # DataFrame column holding the local file path
    set_name_prefix: str  # "clips_" / "frames_"
    mimetype: str  # Passed to Subject.add_location
    upl_seconds_fn: Callable  # (row) -> int | float — UplAbsSeconds value


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
)

FRAME_KIND = SubjectKind(
    noun_singular="frame",
    noun_plural="frames",
    path_column="FramePath",
    set_name_prefix="frames_",
    mimetype="image/jpeg",
    upl_seconds_fn=_compute_frame_upl_seconds,
)


def _upload_subjects_to_zooniverse(
    df: pd.DataFrame,
    kind: SubjectKind,
    subject_set_name: str | None,
    test_upload: bool,
) -> None:
    """Shared upload loop for clips and frames.

    Responsible for: credential check, filtering uploadable rows, connecting
    to Panoptes, creating or reusing the subject set, skipping already-
    uploaded filenames, and building per-row subject metadata. The only
    per-type logic is pulled from `kind`.
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

    if test_upload:
        uploadable = uploadable.iloc[:1]
        logging.info("test_upload=True — limiting to 1 subject.")

    drop_id = uploadable["DropID"].iloc[0]
    n = len(uploadable)
    survey_id = config.get_survey_id_from_drop(drop_id)
    site_id = config.get_site_id_from_drop(drop_id)
    video_filename = f"media/{survey_id}/{drop_id}/{drop_id}.mp4"
    site_reserve_meta = _get_site_reserve_meta(site_id)

    logging.info(f"Connecting to Zooniverse as {config.user}...")
    Panoptes.connect(username=config.user, password=config.password)
    zoo_project = Project.find(config.zooniverse_project_id)

    set_name = subject_set_name or f"{kind.set_name_prefix}{drop_id}"

    # Reuse subject set if it exists, otherwise create it.
    existing_sets = list(
        SubjectSet.where(project_id=zoo_project.id, display_name=set_name)
    )
    if existing_sets:
        logging.info(f"Using existing subject set: '{set_name}'")
        subject_set = existing_sets[0]
    else:
        subject_set = SubjectSet()
        subject_set.links.project = zoo_project
        subject_set.display_name = set_name
        subject_set.save()
        logging.info(f"Subject set created: '{set_name}'")

    already_uploaded = _get_uploaded_filenames(subject_set)

    new_subjects = []
    for _, row in uploadable.iterrows():
        file_path = Path(row[kind.path_column])
        if not file_path.exists():
            logging.warning(
                f"{kind.noun_singular.capitalize()} file missing, skipping: {file_path.name}"
            )
            continue
        if file_path.name in already_uploaded:
            logging.info(f"  Already uploaded, skipping: {file_path.name}")
            continue

        meta = {
            **_build_base_subject_meta(
                row,
                drop_id,
                video_filename,
                site_id,
                site_reserve_meta,
                subject_type=kind.noun_singular,
            ),
            SubjectKeys.UPL_SECONDS: kind.upl_seconds_fn(row),
        }

        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location(str(file_path), manual_mimetype=kind.mimetype)
        subject.metadata.update(meta)
        subject.save()
        new_subjects.append(subject)
        logging.info(f"  Saved: {file_path.name} ({row.get('SelectionReason', '')})")

    subject_set.add(new_subjects)
    logging.info(
        f"Uploaded {len(new_subjects)}/{n} {kind.noun_plural} to subject set '{set_name}'."
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
        '#' prefix — hidden from volunteers, used for traceability and QA.
        '!' prefix — visible to volunteers in Talk (discussion) only.

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
