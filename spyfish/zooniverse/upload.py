"""
Upload extracted Zooniverse clips to a Zooniverse project as a new subject set.
"""

import logging
from pathlib import Path

import pandas as pd
from panoptes_client import Panoptes, Project, Subject, SubjectSet

from spyfish.config.wrapper import config
from spyfish.utils import seconds_to_time

# ── helpers ──────────────────────────────────────────────────────────────────


def check_clip_sizes(clips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks clip file sizes from a selections_df with a ClipPath column.
    Logs a warning for clips over 12 MB (Zooniverse practical limit).

    Returns:
        clips_df with a SizeMB column added.
    """
    clips_df = clips_df.copy()
    clips_df["SizeMB"] = clips_df["ClipPath"].apply(
        lambda p: Path(p).stat().st_size / (1 << 20) if p and Path(p).exists() else None
    )
    over_limit = clips_df[clips_df["SizeMB"] >= 12]
    if not over_limit.empty:
        logging.warning(
            f"{len(over_limit)} clips exceed 12 MB and may be rejected by Zooniverse. "
            f"Consider re-encoding with a higher CRF.{clips_df["SizeMB"]}"
        )
    else:
        logging.info(
            f"All {clips_df['SizeMB'].notna().sum()} clips are under 12 MB. Ready to upload."
        )
    return clips_df


def upload_clips_to_zooniverse(
    selections_df: pd.DataFrame,
    subject_set_name: str | None = None,
) -> None:
    """
    Uploads extracted clips to Zooniverse as a new subject set.

    Reads clip paths and metadata from selections_df (produced by extract_clips_from_selections).
    DropID and ClipPath are read from the DataFrame directly.

    Metadata attached to each subject (hidden from volunteers, prefix '#'):
        #DropID, #SelectionReason, #TargetSpecies, #MaxCount, #Confidence,
        #StartTime, #EndTime, #SamplingStart

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

    drop_id = uploadable["DropID"].iloc[0]
    n = len(uploadable)

    logging.info(f"Connecting to Zooniverse as {config.user}...")
    Panoptes.connect(username=config.user, password=config.password)
    zoo_project = Project.find(config.zooniverse_project_id)

    set_name = subject_set_name or f"spyfish_{drop_id}_{n}clips"

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

    new_subjects = []
    for _, row in uploadable.iterrows():
        clip_path = Path(row["ClipPath"])
        if not clip_path.exists():
            logging.warning(f"Clip file missing, skipping: {clip_path.name}")
            continue

        # Derive human-readable times for volunteers from numeric columns
        start_sec = float(row[config.csv_clip_start_column])
        end_sec = float(row[config.csv_clip_end_column])

        # Metadata hidden from volunteers (prefix '#')
        # These help trace classifications back to source deployments and inform QA
        meta = {
            "#DropID": row.get(config.drop_id_column, drop_id),
            "#SelectionReason": row.get("SelectionReason", ""),
            "#TargetSpecies": row.get("TargetSpecies", ""),
            "#MaxInterval": row.get("MaxInterval", ""),
            "#ConfidenceAgreement": row.get(config.csv_confidence_agreement_column, ""),
            "#StartTime": seconds_to_time(start_sec),
            "#EndTime": seconds_to_time(end_sec),
            "#SamplingStart": row[config.csv_sampling_start_column],
        }

        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location({"video/mp4": str(clip_path)})
        subject.metadata.update(meta)
        subject.save()
        new_subjects.append(subject)
        logging.info(f"  Saved: {clip_path.name} ({row.get('SelectionReason', '')})")

    subject_set.add(new_subjects)
    logging.info(f"Uploaded {len(new_subjects)}/{n} clips to subject set '{set_name}'.")


def upload_frames_to_zooniverse(
    frames_df: pd.DataFrame,
    subject_set_name: str | None = None,
) -> None:
    """
    Uploads extracted frames to Zooniverse as a new subject set.

    Reads frame paths and metadata from frames_df (produced by extract_frames_from_selections).
    DropID and FramePath are read from the DataFrame directly.
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

    drop_id = uploadable["DropID"].iloc[0]
    n = len(uploadable)

    logging.info(f"Connecting to Zooniverse as {config.user}...")
    Panoptes.connect(username=config.user, password=config.password)
    zoo_project = Project.find(config.zooniverse_project_id)

    set_name = subject_set_name or f"spyfish_{drop_id}_{n}images"

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

    new_subjects = []
    for _, row in uploadable.iterrows():
        frame_path = Path(row["FramePath"])
        if not frame_path.exists():
            logging.warning(f"Frame file missing, skipping: {frame_path.name}")
            continue
        # Metadata strictly mirrors standardized selections outputs. Legacy selections are handled separately.
        time_of_max = float(row[config.csv_clip_max_time_column])

        meta = {
            "#DropID": row.get(config.drop_id_column, drop_id),
            "#SelectionReason": row.get("SelectionReason", ""),
            "#TargetSpecies": row.get("TargetSpecies", ""),
            "#MaxInterval": row.get("MaxInterval", ""),
            "#ConfidenceAgreement": row.get(config.csv_confidence_agreement_column, ""),
            "#TimeOfMaxnMs": seconds_to_time(time_of_max),
            "#SamplingStart": row[config.csv_sampling_start_column],
        }

        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location({"image/jpeg": str(frame_path)})
        subject.metadata.update(meta)
        subject.save()
        new_subjects.append(subject)
        logging.info(f"  Saved: {frame_path.name} ({row.get('SelectionReason', '')})")
    logging.info(
        f"!!!!! !!!! !!!This is what the rows have {uploadable.columns} check if metadata is good {meta}"
    )
    subject_set.add(new_subjects)
    logging.info(
        f"Uploaded {len(new_subjects)}/{n} frames to subject set '{set_name}'."
    )
