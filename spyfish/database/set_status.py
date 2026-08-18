"""
CLI utility to inspect or manually override section statuses for a deployment.

Usage examples:

  # Inspect current state
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03

  # Override a section status (bypasses transition checks, admin use only)
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 --ml-status ml_ready
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 --citsci-status citsci_skipped

  # Combine multiple overrides
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 \\
      --ml-status ml_ready --sampling-start 0 --priority 10

  # Create a new record (upsert)
  python -m spyfish.database.set_status KSF_20240124_BUV_KSF_085_03 \\
      --ml-status ml_ready --sampling-start 0 --sampling-end 1800 --create
"""

import argparse
import logging
import sys

from spyfish.config.base import SECTION_VALUES as _SECTION_VALUES
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager


def cmd_show(db: DatabaseManager, drop_id: str):
    record = db.get_deployment(drop_id)
    if not record:
        logging.error(f"'{drop_id}' not found in database.")
        sys.exit(1)
    col_width = max(len(k) for k in record)
    for k, v in record.items():
        print(f"  {k:<{col_width}}  {v}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Inspect or manually override deployment statuses in the pipeline database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("drop_id", help="DropID to inspect or update")

    # Section status overrides
    parser.add_argument("--ingest-status", choices=_SECTION_VALUES["ingest_status"])
    parser.add_argument("--ml-status", choices=_SECTION_VALUES["ml_status"])
    parser.add_argument("--citsci-status", choices=_SECTION_VALUES["citsci_status"])
    parser.add_argument("--expert-status", choices=_SECTION_VALUES["expert_status"])
    parser.add_argument(
        "--reporting-status", choices=_SECTION_VALUES["reporting_status"]
    )

    # Metadata overrides
    parser.add_argument("--sampling-start", type=int, default=None)
    parser.add_argument("--sampling-end", type=int, default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument(
        "--video-presence",
        default=None,
        choices=["present", "absent", "no_video_bad_dep"],
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Processing priority (higher = processed first). Default 0.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Insert a new record if the drop_id doesn't exist yet.",
    )

    args = parser.parse_args()
    db = DatabaseManager()

    section_updates = {
        col: val
        for col, val in [
            ("ingest_status", args.ingest_status),
            ("ml_status", args.ml_status),
            ("citsci_status", args.citsci_status),
            ("expert_status", args.expert_status),
            ("reporting_status", args.reporting_status),
        ]
        if val is not None
    }
    field_updates = {
        col: val
        for col, val in [
            ("sampling_start", args.sampling_start),
            ("sampling_end", args.sampling_end),
            ("video_path", args.video_path),
            ("video_presence", args.video_presence),
            ("priority", args.priority),
        ]
        if val is not None
    }

    has_updates = section_updates or field_updates

    if not has_updates:
        cmd_show(db, args.drop_id)
        return

    existing = db.get_deployment(args.drop_id)
    if not existing and not args.create:
        logging.error(
            f"'{args.drop_id}' not found in database. Pass --create to insert a new record."
        )
        sys.exit(1)

    # Validate before any DB writes, fail the whole command atomically.
    start = (
        args.sampling_start
        if args.sampling_start is not None
        else (existing or {}).get("sampling_start")
    )
    end = (
        args.sampling_end
        if args.sampling_end is not None
        else (existing or {}).get("sampling_end")
    )
    if start is not None and end is not None:
        errors = config.validate_sampling_window(args.drop_id, float(start), float(end))
        if errors:
            for e in errors:
                logging.error(e)
            logging.error("No changes applied, fix the values above and rerun.")
            sys.exit(1)

    if not existing and args.create:
        db.add_or_update_deployment(drop_id=args.drop_id)
        logging.info(f"Created new record for '{args.drop_id}'.")

    for section, value in section_updates.items():
        db.update_section_status(args.drop_id, section, value)
        logging.info(f"Set {section}={value!r} for '{args.drop_id}'.")

    if field_updates:
        db.update_deployment_fields(args.drop_id, **field_updates)
        logging.info(
            f"Updated fields {list(field_updates.keys())} for '{args.drop_id}'."
        )

    cmd_show(db, args.drop_id)


if __name__ == "__main__":
    main()
