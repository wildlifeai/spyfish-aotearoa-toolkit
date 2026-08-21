"""Fetch Zooniverse data exports straight into the legacy-classifications dir.

Replaces the manual loop of requesting exports in the project builder,
waiting for the email, downloading to a laptop and copying them around.
Run it on NeSI and the files land directly where ``--legacy-zooniverse``
reads them; add ``--to-s3`` and they continue to the bucket over the
datacenter link, keeping the download-if-missing fallback current.

Usage:
    python scripts/fetch_zooniverse_exports.py             # latest existing exports
    python scripts/fetch_zooniverse_exports.py --generate  # request fresh ones and wait
    python scripts/fetch_zooniverse_exports.py --to-s3     # also sync the dir to S3

Notes:
    * ``--generate`` asks Zooniverse to build new exports and waits for them.
      Generation for a project this size can take from minutes to hours, and
      Zooniverse allows one generation per export type per 24 h.
    * Without ``--generate`` you get whatever was last generated, which may be
      months old; check the log line with each file's first classification date
      if freshness matters.
    * Credentials come from .env (ZOONIVERSE_USER / ZOONIVERSE_PASSWORD), the
      same ones the pipeline uses.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panoptes_client import Project  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.zooniverse.parse_classifications import (  # noqa: E402
    connect_to_zooniverse,
)

EXPORT_TYPES = ("classifications", "subjects")


def fetch_project_exports(project_id: int, out_dir: Path, generate: bool) -> None:
    project = Project.find(project_id)
    # Filename prefix from the project slug tail ("owner/spyfish-aotearoa-ey"
    # → "spyfish-aotearoa-ey"), matching the existing export file naming, so
    # the backfill's *classification*/*subject* globs pick them up unchanged.
    prefix = str(project.slug).rsplit("/", 1)[-1]
    for export_type in EXPORT_TYPES:
        target = out_dir / f"{prefix}-{export_type}.csv"
        logging.info(
            f"Project {project_id} ({prefix}): fetching {export_type} export"
            f"{' (generating fresh, this can take a while)' if generate else ''}..."
        )
        response = project.get_export(export_type, generate=generate, wait=generate)
        size = 0
        with open(target, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                size += len(chunk)
        logging.info(f"  → {target} ({size / 1e6:.1f} MB)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Request fresh exports and wait for Zooniverse to build them "
        "(minutes to hours; limited to one per type per 24 h).",
    )
    parser.add_argument(
        "--to-s3",
        action="store_true",
        help="After downloading, sync the legacy dir's CSVs to S3 so the "
        "backfill's download-if-missing fallback serves current files.",
    )
    args = parser.parse_args()

    out_dir = config.legacy_zooniverse_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    connect_to_zooniverse()
    for project_id in config.zooniverse_source_project_ids:
        fetch_project_exports(project_id, out_dir, generate=args.generate)

    if args.to_s3:
        from spyfish.storage.s3_handler import S3Handler

        S3Handler().sync_local_to_s3(
            str(out_dir),
            config.legacy_zooniverse_s3_prefix,
            filters=["--exclude", "*", "--include", "*.csv"],
        )

    logging.info(
        "Done. Run `python run_pipeline.py --legacy-zooniverse --db-refresh "
        "--no-upload` to ingest."
    )


if __name__ == "__main__":
    main()
