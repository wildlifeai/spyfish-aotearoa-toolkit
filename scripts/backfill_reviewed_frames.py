"""Record the reviewed-frame list for volumes synced before it was tracked.

A Biigle annotation report contains only frames that HAVE annotations, so on its
own it cannot distinguish "the expert looked and saw nothing" from "never
uploaded". Both look like a missing row, and the second must never become a
training negative. `sync_biigle_annotations` now writes a per-drop reviewed-frames CSV
(the volume's own file list) at sync time, which lets `biigle_to_yolo` emit an
empty .txt — a YOLO background image — for every reviewed-but-empty frame.

That only helps drops synced from 2026-08-24 onward. This script backfills the
rest: 160 deployments at expert_complete and 29 at expert_uploaded already have
their volumes in Biigle, and their negatives are currently discarded at parse
time. Before this, the corpus held ONE background frame in 5054 while the frame
selector was deliberately choosing "Blind (False Negative Check)" frames for
review — the experts were producing negatives and nothing kept them.

Read-only against Biigle: fetches each volume's file list and writes one small
CSV per drop. No status transitions, no re-download of annotations, no upload.
Re-running is safe; existing reviewed-frames CSVs are skipped unless --force.

Volumes are fetched ONCE and shared across every drop in them, so a survey-pooled
volume spanning twenty deployments costs one API call, not twenty.

Training volumes (--training): the survey training-frame volumes in the
training project get the same treatment, writing
`{drop}_biigle_training_reviewed_frames.csv`. Their negatives used to be
written straight into labels/ at download time, and biigle_to_yolo wipes
labels/ on every rebuild, so they never reached training. Only drops whose
training raw CSV is already on disk get a list: a list without the
annotations would make every frame of the drop look empty.

Usage (from project root):
    python scripts/backfill_reviewed_frames.py                 # dry run, shows plan
    python scripts/backfill_reviewed_frames.py --training --apply
    python scripts/backfill_reviewed_frames.py --apply
    python scripts/backfill_reviewed_frames.py --apply --force # rewrite existing
    python scripts/backfill_reviewed_frames.py --apply --drop-id KSF_20240124_BUV_KSF_085_01
"""

import argparse
import logging
from collections import Counter, defaultdict

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.biigle.biigle_to_yolo import drop_id_from_frame_filename

# Same helper the live sync path uses, so a backfilled CSV is byte-identical to
# one written at sync time. Importing it rather than reimplementing is the point:
# two ways of deciding "which frames belong to this drop" would drift.
from spyfish.biigle.sync_annotations import _write_reviewed_frames_csv
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Backfill reviewed-frame list CSVs from Biigle volumes."
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the files (default: dry run)"
    )
    parser.add_argument(
        "--force", action="store_true", help="rewrite reviewed-frames CSVs that exist"
    )
    parser.add_argument("--drop-id", help="restrict to one deployment")
    parser.add_argument("--volume-id", type=int, help="restrict to one Biigle volume")
    parser.add_argument(
        "--training",
        action="store_true",
        help="backfill the training project's survey volumes instead of expert volumes",
    )
    args = parser.parse_args()

    if args.training:
        backfill_training(args)
        return

    db = DatabaseManager()
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT drop_id, biigle_volume_id, expert_status FROM deployments "
            "WHERE biigle_volume_id IS NOT NULL AND biigle_volume_id != '' "
            "ORDER BY drop_id"
        ).fetchall()

    targets = []
    skipped_existing = 0
    for r in rows:
        drop_id, vol = r["drop_id"], str(r["biigle_volume_id"])
        if args.drop_id and drop_id != args.drop_id:
            continue
        if args.volume_id and vol != str(args.volume_id):
            continue
        try:
            path = config.get_biigle_expert_reviewed_frames_csv_path(drop_id)
        except ValueError as e:
            # Malformed drop_id that reached the deployments table. Report and
            # carry on: a backfill is a reconciliation pass, not a validator.
            logging.warning(f"  skipping {drop_id}: {e}")
            continue
        if path.exists() and not args.force:
            skipped_existing += 1
            continue
        targets.append((drop_id, int(vol), r["expert_status"]))

    by_status = Counter(t[2] for t in targets)
    by_volume = defaultdict(list)
    for drop_id, vol, _ in targets:
        by_volume[vol].append(drop_id)

    logging.info(
        f"{len(rows)} deployment(s) have a Biigle volume; "
        f"{skipped_existing} already have a reviewed-frames CSV; "
        f"{len(targets)} to backfill across {len(by_volume)} volume(s)."
    )
    for status, n in sorted(by_status.items()):
        logging.info(f"  {status}: {n}")

    if not targets:
        logging.info("Nothing to do.")
        return
    if not args.apply:
        logging.info("DRY RUN — pass --apply to write. First few:")
        for vol, drops in list(by_volume.items())[:5]:
            logging.info(f"  volume {vol}: {len(drops)} drop(s), e.g. {drops[:3]}")
        return

    handler = BiigleHandler()
    cache: dict = {}  # volume_id -> filenames, so each volume is fetched once
    written = empty = failed = 0
    for vol, drops in sorted(by_volume.items()):
        for drop_id in drops:
            try:
                n = _write_reviewed_frames_csv(handler, vol, drop_id, cache)
            except Exception as e:
                # One unreachable volume must not abort a 189-drop backfill.
                logging.warning(f"  {drop_id} (volume {vol}): {type(e).__name__}: {e}")
                failed += 1
                continue
            if n:
                written += 1
            else:
                # No filenames matched this drop in that volume. Real cause is
                # usually a pooled volume that never received this deployment;
                # writing nothing is correct — an empty list would later be
                # read as "reviewed, saw nothing" for frames never uploaded.
                logging.info(f"  {drop_id}: no files in volume {vol}, skipped")
                empty += 1

    logging.info(
        f"Backfill complete: {written} reviewed-frames CSV(s) written, "
        f"{empty} drop(s) had no files in their volume, {failed} failed."
    )
    logging.info(
        "Negatives appear at the next `--retrain --data-prep`: biigle_to_yolo "
        "writes an empty .txt for every reviewed frame the report never mentioned."
    )


def backfill_training(args) -> None:
    """Write `_biigle_training_reviewed_frames.csv` for every drop in a training volume."""
    handler = BiigleHandler()
    project_id = config.biigle_training_project_id
    volumes = handler.get_volumes(project_id)
    if args.volume_id:
        volumes = [v for v in volumes if v["id"] == args.volume_id]
    logging.info(f"Training project {project_id}: {len(volumes)} volume(s)")

    written = existing = no_csv = 0
    for v in volumes:
        by_drop = defaultdict(list)
        for img in handler.get_volume_images(v["id"]):
            fname = img.get("filename") or ""
            if "__frame_" in fname:
                by_drop[drop_id_from_frame_filename(fname)].append(fname)
        if not by_drop:
            logging.info(
                f"  volume {v['id']} ({v.get('name')}): no drop-id filenames, skipped"
            )
            continue
        logging.info(f"  volume {v['id']} ({v.get('name')}): {len(by_drop)} drop(s)")
        for drop_id, names in sorted(by_drop.items()):
            if args.drop_id and drop_id != args.drop_id:
                continue
            try:
                path = config.get_biigle_training_reviewed_frames_csv_path(drop_id)
            except ValueError as e:
                logging.warning(f"    skipping {drop_id}: {e}")
                continue
            if not config.get_biigle_training_raw_csv_path(drop_id).exists():
                logging.info(f"    {drop_id}: no training raw CSV on disk, skipped")
                no_csv += 1
                continue
            if path.exists() and not args.force:
                existing += 1
                continue
            if args.apply:
                pd.DataFrame({"filename": sorted(names)}).to_csv(path, index=False)
            written += 1

    verb = "written" if args.apply else "to write (DRY RUN, pass --apply)"
    logging.info(
        f"Training backfill: {written} reviewed-frames CSV(s) {verb}, "
        f"{existing} already present, {no_csv} drop(s) without a training raw CSV."
    )


if __name__ == "__main__":
    main()
