"""Attach "Pick" selection-reason image labels to a Biigle volume already uploaded.

The labels attach to images by ID and nothing about them depends on upload
time, so any existing volume can be brought up to date without re-uploading a
single frame. This is what makes the feature retroactive: every drop's
selections CSV is kept on disk as a permanent record of what was sent (see
`upsert_selections`), so the reason for each frame is still recoverable long
after the volume was created.

Frame filenames are reconstructed from the selections CSV timestamps with
`generate_frame_filename`, the same helper the extractor used to name them, so
the join back to the volume's registered images is exact rather than fuzzy.
Works for per-drop and survey-pooled volumes alike: the drop each image belongs
to is parsed from its filename, so a pooled volume spanning ten deployments
reads ten selections CSVs and attaches in one batch.

Re-running is safe. Biigle rejects a label the image already carries, which is
counted and reported rather than treated as an error, so a second pass only
fills in what is missing.

Usage (from project root):
    python scripts/backfill_biigle_pick_labels.py --volume-id 1234
    python scripts/backfill_biigle_pick_labels.py --volume-id 1234 --apply
    python scripts/backfill_biigle_pick_labels.py --drop-id KSF_20240124_BUV_KSF_085_01 --apply
"""

import argparse
import logging
from collections import Counter

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.biigle.biigle_to_yolo import drop_id_from_frame_filename
from spyfish.biigle.selection_reason import canonical_reason
from spyfish.biigle.upload_frames import attach_selection_reason_labels
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.utils import generate_frame_filename


def _frames_df_for_drop(drop_id: str) -> pd.DataFrame:
    """Selections CSV → the (FramePath, SelectionReason) frame the attacher wants.

    FramePath is synthesised rather than read: the extractor adds that column to
    its in-memory DataFrame but never writes it back to the CSV, so the on-disk
    record has timestamps only. Regenerating the name with the same helper the
    extractor used keeps the two in lockstep.
    """
    config.validate_drop_id(drop_id)
    path = config.get_biigle_selections_csv_path(drop_id)
    if not path.exists():
        logging.warning(f"{drop_id}: no selections CSV at {path}, skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    time_col = config.csv_clip_max_time_column
    reason_col = config.selection_reason_column
    for col in (time_col, reason_col):
        if col not in df.columns:
            logging.warning(f"{drop_id}: selections CSV has no {col!r} column.")
            return pd.DataFrame()

    return pd.DataFrame(
        {
            "FramePath": [
                generate_frame_filename(drop_id, float(t)) for t in df[time_col]
            ],
            reason_col: df[reason_col],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--volume-id", type=int, help="Biigle volume to backfill.")
    target.add_argument(
        "--drop-id",
        help="Backfill the volume this deployment was uploaded to (read from the DB).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually attach the labels. Without this only the plan is printed.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    volume_id = args.volume_id
    if volume_id is None:
        config.validate_drop_id(args.drop_id)
        deployment = DatabaseManager().get_deployment(args.drop_id)
        raw = deployment and deployment["biigle_volume_id"]
        if not raw:
            raise SystemExit(f"{args.drop_id} has no biigle_volume_id in the database.")
        volume_id = int(raw)
        logging.info(f"{args.drop_id} → volume {volume_id}")

    if not config.selection_reason_label_ids:
        raise SystemExit(
            "No biigle.selection_reason_labels configured. Run "
            "scripts/create_biigle_pick_labels.py --apply first, then set the IDs "
            "in config.yaml."
        )

    handler = BiigleHandler()
    images = handler.get_volume_images(volume_id)
    if not images:
        raise SystemExit(f"Volume {volume_id} has no images.")

    name_map = {img["filename"]: int(img["id"]) for img in images}
    drop_ids = sorted({drop_id_from_frame_filename(n) for n in name_map})
    logging.info(
        f"Volume {volume_id}: {len(images)} image(s) across {len(drop_ids)} "
        f"deployment(s): {', '.join(drop_ids)}"
    )

    frames = [_frames_df_for_drop(d) for d in drop_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No selections CSVs found for any deployment in this volume.")
    frames_df = pd.concat(frames, ignore_index=True)

    buckets = Counter(
        canonical_reason(r) or "UNRECOGNISED"
        for r in frames_df[config.selection_reason_column]
    )
    logging.info(f"\n{len(frames_df)} selection row(s) across all deployments:")
    for bucket, count in buckets.most_common():
        configured = bucket in config.selection_reason_label_ids
        note = "" if configured else "   (no label configured, will be skipped)"
        logging.info(f"  {bucket:<14} {count:>4}{note}")

    if not args.apply:
        logging.info("\nDry run, nothing was attached. Re-run with --apply.")
        return

    attached = attach_selection_reason_labels(
        volume_id, frames_df, filename_to_biigle_id=name_map
    )
    logging.info(f"\nAttached {attached} image label(s) to volume {volume_id}.")


if __name__ == "__main__":
    main()
