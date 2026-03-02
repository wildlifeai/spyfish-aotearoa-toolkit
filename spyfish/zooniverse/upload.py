"""
Upload extracted Zooniverse clips to a Zooniverse project as a new subject set.
"""
import logging
from pathlib import Path

import pandas as pd

from spyfish.config import config
from panoptes_client import Panoptes, Project, Subject, SubjectSet


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
            "Consider re-encoding with a higher CRF."
        )
    else:
        logging.info(f"All {clips_df['SizeMB'].notna().sum()} clips are under 12 MB. Ready to upload.")
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

    if not all([config.zooniverse_user, config.zooniverse_password, config.zooniverse_project_id]):
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

    logging.info(f"Connecting to Zooniverse as {config.zooniverse_user}...")
    Panoptes.connect(username=config.zooniverse_user, password=config.zooniverse_password)
    zoo_project = Project.find(config.zooniverse_project_id)

    set_name = subject_set_name or f"spyfish_{drop_id}_{n}clips"
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

        # Metadata hidden from volunteers (prefix '#')
        # These help trace classifications back to source deployments and inform QA
        meta = {
            "#DropID": row.get("DropID", drop_id),
            "#SelectionReason": row.get("SelectionReason", ""),
            "#TargetSpecies": row.get("TargetSpecies", ""),
            "#MaxCount": row.get("MaxCount", ""),
            "#Confidence": row.get("Confidence", ""),
            "#StartTime": row.get("StartTime", ""),
            "#EndTime": row.get("EndTime", ""),
            "#SamplingStart": row.get("SamplingStart", 0),
        }

        subject = Subject()
        subject.links.project = zoo_project
        subject.add_location(str(clip_path))
        subject.metadata.update(meta)
        subject.save()
        new_subjects.append(subject)
        logging.info(f"  Saved: {clip_path.name} ({row.get('SelectionReason', '')})")

    subject_set.add(new_subjects)
    logging.info(f"Uploaded {len(new_subjects)}/{n} clips to subject set '{set_name}'.")
