"""Download deployment videos from S3 into the media dir, ahead of a GPU job.

Interactive front end for ``spyfish.orchestrator.download`` (the ``--download``
pipeline stage). Use the stage from SLURM (``sbatch scripts/slurm/cpu.sl
--download --survey X``); use this when you want named drops or the targets CSV:

    # the drops listed in paths.deployment_targets_csv (what --set-targets reads)
    python scripts/download_videos.py --targets
    # named drops
    python scripts/download_videos.py HOR_20251204_BUV_HOR_003_01 HOR_20251204_BUV_HOR_012_01
    # every drop of a survey, whatever its pipeline state
    python scripts/download_videos.py --survey HOR_20251204_BUV
    # show the plan, transfer nothing
    python scripts/download_videos.py --targets --dry-run

Destination is the runner's: ``config.media_dir / <basename of video_path>``
(flat, not the nested S3 layout). Existing files are left alone. Every drop
is attempted whatever its ingest_status; S3 itself (not the DB's
video_presence) says whether the video exists and is readable. Drops with no
video are listed but do not fail the run. Exit code 1 if any drop is still in
DEEP_ARCHIVE or a transfer failed.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.manager import DatabaseManager  # noqa: E402
from spyfish.orchestrator.download import (  # noqa: E402
    download_videos,
    select_download_drops,
)


def _drop_ids(args, db: DatabaseManager) -> list[str]:
    if args.survey:
        return select_download_drops(db, args.survey)
    if args.targets:
        csv_path = config.pipeline_targets_csv
        if not csv_path or not Path(csv_path).exists():
            sys.exit(f"targets CSV not found: {csv_path}")
        return [
            str(d).strip()
            for d in pd.read_csv(csv_path)[config.drop_id_column]
            if str(d).strip() and str(d) != "nan"
        ]
    return args.drop_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drop_ids", nargs="*", help="DropIDs to fetch")
    parser.add_argument(
        "--targets",
        action="store_true",
        help="Fetch the drops listed in paths.deployment_targets_csv",
    )
    parser.add_argument(
        "--survey", metavar="SURVEY_ID", help="Every drop of this survey"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan, transfer nothing"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    db = DatabaseManager()
    drop_ids = _drop_ids(args, db)
    if not drop_ids:
        sys.exit("nothing to download: no drop ids given / found")
    result = download_videos(drop_ids, db=db, dry_run=args.dry_run)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
