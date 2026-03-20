"""
Simple CLI utility to manually update the status of a deployment drop.
Usage: python -m spyfish.database.set_status <drop_id> <new_status>
"""

import argparse
import logging
import sys

from spyfish.config.base import PipelineStatus
from spyfish.database.manager import DatabaseManager


def main():
    # Get valid statuses from PipelineStatus class attributes
    valid_statuses = [
        attr
        for attr in dir(PipelineStatus)
        if attr.isupper()
        and not attr.startswith("_")
        and isinstance(getattr(PipelineStatus, attr), str)
    ]

    parser = argparse.ArgumentParser(
        description="Manually update a deployment status in the Spyfish pipeline database."
    )
    parser.add_argument(
        "drop_id", help="The DropID to update (e.g. KSF_20240124_BUV_KSF_085_01)"
    )
    parser.add_argument(
        "status", help=f"The new status. Valid options: {valid_statuses}"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Validate status
    if args.status not in valid_statuses:
        logging.error(
            f"Invalid status '{args.status}'. Must be one of: {valid_statuses}"
        )
        sys.exit(1)

    new_status = getattr(PipelineStatus, args.status)
    db = DatabaseManager()

    # Check if drop exists
    record = db.get_deployment(args.drop_id)
    if not record:
        logging.error(f"Drop ID '{args.drop_id}' not found in database.")
        sys.exit(1)

    old_status = record["status"]
    logging.info(f"Updating {args.drop_id}: {old_status} -> {new_status}")

    db.update_status(args.drop_id, new_status)
    logging.info("Successfully updated status.")


if __name__ == "__main__":
    main()
