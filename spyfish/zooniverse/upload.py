"""
Upload extracted Zooniverse clips to a Zooniverse project as a new subject set.
"""

import logging
from pathlib import Path

import pandas as pd
from panoptes_client import Panoptes, Project, Subject, SubjectSet

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager

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
        "LinkToMarineReserve": site.get(config.link_to_marine_reserve_column, ""),
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
    row, drop_id: str, video_filename: str, site_id: str, site_reserve_meta: dict
) -> dict:
    """Common Zooniverse subject metadata shared between clips and frames uploads."""
    meta = {
        "DropID": drop_id,
        "#VideoFilename": video_filename,
        "#siteName": site_id,
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


def upload_clips_to_zooniverse(
    selections_df: pd.DataFrame,
    subject_set_name: str | None = None,
    test_upload: bool = False,
) -> None:
    """
    Uploads extracted clips to Zooniverse as a new subject set.

    Reads clip paths and metadata from selections_df (produced by extract_clips_from_selections).
    DropID and ClipPath are read from the DataFrame directly.

    Metadata attached to each subject:
        '#' prefix — hidden from volunteers, used for traceability and QA.
        '!' prefix — visible to volunteers in Talk (discussion) only, not during classification.

        #DropID, #VideoFilename, #siteName,
        #SelectionReason, #TargetSpecies, #MaxInterval, #ConfidenceAgreement,
        #StartTime, #EndTime, #SamplingStart,
        !LinkToMarineReserve, ProtectionStatus (when BUV Sites CSV is reachable on S3)

    Args:
        selections_df: DataFrame with one row per clip, including 'ClipPath' and 'DropID' columns.
                       As returned by extract_clips_from_selections().
        subject_set_name: Optional override for the Zooniverse subject set display name.
    """

    if not all(
        [
            config.user,
            config.password,
            config.zooniverse_project_id,
        ]
    ):
        raise EnvironmentError(
            "Missing Zooniverse credentials. Set ZOONIVERSE_USER, ZOONIVERSE_PASSWORD, "
            "and ZOONIVERSE_PROJECT_ID in your .env file."
        )

    # Filter to clips that were actually extracted
    uploadable = selections_df[selections_df["ClipPath"].notna()].copy()
    if uploadable.empty:
        logging.error("No clips available to upload (ClipPath is empty for all rows).")
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

    set_name = subject_set_name or f"clips_{drop_id}"

    # Check if subject set already exists to avoid "Display name has already been taken"
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

    logging.info(f"DataFrame columns: {list(uploadable.columns)}")
    logging.info(
        f"Config column names — drop_id: '{config.drop_id_column}', selection_reason: '{config.selection_reason_column}', species: '{config.csv_scientific_name_column}', max_interval: '{config.csv_max_interval_column}', sampling_start: '{config.csv_sampling_start_column}'"
    )
    if not uploadable.empty:
        logging.info(f"First row sample: {uploadable.iloc[0].to_dict()}")

    new_subjects = []
    for _, row in uploadable.iterrows():
        clip_path = Path(row["ClipPath"])
        if not clip_path.exists():
            logging.warning(f"Clip file missing, skipping: {clip_path.name}")
            continue
        if clip_path.name in already_uploaded:
            logging.info(f"  Already uploaded, skipping: {clip_path.name}")
            continue

        sampling_start = float(row.get(config.csv_sampling_start_column, 0))
        upl_abs_seconds = sampling_start + float(row[config.csv_clip_start_column])

        meta = {
            **_build_base_subject_meta(
                row, drop_id, video_filename, site_id, site_reserve_meta
            ),
            "UplAbsSeconds": int(upl_abs_seconds),
        }

        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location(str(clip_path), manual_mimetype="video/mp4")
        subject.metadata.update(meta)
        subject.save()
        new_subjects.append(subject)
        logging.info(f"  Saved: {clip_path.name} ({row.get('SelectionReason', '')})")

    subject_set.add(new_subjects)
    logging.info(f"Uploaded {len(new_subjects)}/{n} clips to subject set '{set_name}'.")


def upload_frames_to_zooniverse(
    frames_df: pd.DataFrame,
    subject_set_name: str | None = None,
    test_upload: bool = False,
) -> None:
    """
    Uploads extracted frames to Zooniverse as a new subject set.

    Reads frame paths and metadata from frames_df (produced by extract_frames_from_selections).
    DropID and FramePath are read from the DataFrame directly.

    Args:
        frames_df: DataFrame with one row per frame, including 'FramePath' and 'DropID' columns.
        subject_set_name: Optional override for the Zooniverse subject set display name.
    """

    if not all(
        [
            config.user,
            config.password,
            config.zooniverse_project_id,
        ]
    ):
        raise EnvironmentError(
            "Missing Zooniverse credentials. Set ZOONIVERSE_USER, ZOONIVERSE_PASSWORD, "
            "and ZOONIVERSE_PROJECT_ID in your .env file."
        )

    # Filter to frames that were actually extracted
    uploadable = frames_df[frames_df["FramePath"].notna()].copy()
    if uploadable.empty:
        logging.error(
            "No frames available to upload (FramePath is empty for all rows)."
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

    set_name = subject_set_name or f"frames_{drop_id}"

    # Check if subject set already exists
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
        frame_path = Path(row["FramePath"])
        if not frame_path.exists():
            logging.warning(f"Frame file missing, skipping: {frame_path.name}")
            continue
        if frame_path.name in already_uploaded:
            logging.info(f"  Already uploaded, skipping: {frame_path.name}")
            continue
        upl_abs_seconds = float(row[config.csv_clip_max_time_column])

        meta = {
            **_build_base_subject_meta(
                row, drop_id, video_filename, site_id, site_reserve_meta
            ),
            "UplAbsSeconds": upl_abs_seconds,
        }

        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location(str(frame_path), manual_mimetype="image/jpeg")
        subject.metadata.update(meta)
        subject.save()
        new_subjects.append(subject)
        logging.info(f"  Saved: {frame_path.name} ({row.get('SelectionReason', '')})")
    subject_set.add(new_subjects)
    logging.info(
        f"Uploaded {len(new_subjects)}/{n} frames to subject set '{set_name}'."
    )
