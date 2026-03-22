"""
CLI utility to manually create or update a deployment in the pipeline database.

Usage examples:

  # Update status of an existing record
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 READY_FOR_ML

  # Update status + a field
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 READY_FOR_ML --sampling-start 0

  # Create a new record (upsert)
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 READY_FOR_ML \\
      --sampling-start 0 --sampling-end 29 --create

  # Inspect the current state of a record
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 --show
"""

import argparse
import logging
import sys

from spyfish.config.base import PipelineStatus
from spyfish.database.manager import DatabaseManager


def _valid_statuses():
    return [s[0] for s in PipelineStatus.STAGE_ORDER]


def cmd_show(db: DatabaseManager, drop_id: str):
    record = db.get_deployment(drop_id)
    if not record:
        logging.error(f"'{drop_id}' not found in database.")
        sys.exit(1)
    col_width = max(len(k) for k in record)
    for k, v in record.items():
        print(f"  {k:<{col_width}}  {v}")


def cmd_upsert(db: DatabaseManager, args):
    valid = _valid_statuses()
    if args.status not in valid:
        logging.error(f"Invalid status '{args.status}'. Valid options:\n  {valid}")
        sys.exit(1)

    existing = db.get_deployment(args.drop_id)
    if not existing and not args.create:
        logging.error(
            f"'{args.drop_id}' not found in database. "
            "Pass --create to insert a new record."
        )
        sys.exit(1)

    if existing:
        # Update: only touch fields explicitly provided
        fields = {"status": args.status}
        if args.sampling_start is not None:
            fields["sampling_start"] = args.sampling_start
        if args.sampling_end is not None:
            fields["sampling_end"] = args.sampling_end
        if args.video_path is not None:
            fields["video_path"] = args.video_path
        db.update_deployment_fields(args.drop_id, **fields)
        logging.info(f"Updated '{args.drop_id}': {fields}")
    else:
        # Create via upsert
        video_path = args.video_path or ""
        db.add_or_update_deployment(
            drop_id=args.drop_id,
            status=args.status,
            video_path=video_path,
            sampling_start=args.sampling_start,
            sampling_end=args.sampling_end,
        )
        logging.info(f"Created '{args.drop_id}' with status '{args.status}'.")

    if args.show:
        cmd_show(db, args.drop_id)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Create or update a deployment record in the pipeline database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("drop_id", help="DropID to create/update")
    parser.add_argument(
        "status",
        nargs="?",
        help=f"Pipeline status. Valid options: {_valid_statuses()}",
    )
    parser.add_argument("--sampling-start", type=int, default=None)
    parser.add_argument("--sampling-end", type=int, default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Insert a new record if the drop_id doesn't exist yet.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the full record after applying changes.",
    )

    args = parser.parse_args()
    db = DatabaseManager()

    if args.status is None:
        # No status given — show only
        cmd_show(db, args.drop_id)
    else:
        cmd_upsert(db, args)


if __name__ == "__main__":
    main()
