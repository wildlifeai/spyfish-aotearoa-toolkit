"""Check whether topping a drop's Zooniverse clip set up to the full-video grid is safe.

Read-only: nothing is uploaded and the DB is not written. Run it before a
top-up (``sample_all_clips`` for one drop whose set already holds its ML
clips) to see what the upload would do:

    python scripts/check_clip_topup.py SLI_20260114_BUV_SLI_085_01
    python scripts/check_clip_topup.py SLI_20260114_BUV_SLI_084_01 SLI_20260114_BUV_SLI_113_01

Why it can be safe: the ML clip selector buckets every clip onto the same
``clip_length`` grid, counted from ``sampling_start``, that full-video mode
builds (``ClipSelector._bucket`` and ``_select_all_clips``). The upload skips
a clip whose identity (video filename + start second) is already in the set,
so the clips already up are skipped and only the missing grid windows go up.

When it is NOT safe, and what this reports:
  * live clips that are off the grid: ``sampling_start`` changed since the
    upload, or the window is longer than ``clip_cap`` windows and the cap's
    even thinning skips some grid slots. Those clips stay, the grid is
    uploaded around them, and the set ends above the cap.
  * a set that does not exist yet: nothing to top up, a normal clips run does it.

Cost: per drop, one set lookup plus one page walk over its subjects. Panoptes
has no bulk form for the subjects of many sets, so this is one request chain
per drop, not per clip.

Exit code 1 if any drop is not safe to top up.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.manager import DatabaseManager  # noqa: E402
from spyfish.zooniverse.select_zooniverse_clips import (  # noqa: E402
    _select_all_clips,
)
from spyfish.zooniverse.subject_keys import SubjectKeys  # noqa: E402
from spyfish.zooniverse.upload import CLIP_KIND, fetch_existing_set  # noqa: E402


def _live_starts(subjects: list) -> tuple[set, set]:
    """(start seconds, video filenames) of the clips already in the set."""
    starts, filenames = set(), set()
    for subject in subjects:
        meta = getattr(subject, "metadata", None) or {}
        value = meta.get(SubjectKeys.UPL_SECONDS)
        if value not in (None, ""):
            starts.add(int(float(value)))
        filenames.add(meta.get(SubjectKeys.VIDEO_FILENAME))
    return starts, filenames


def check_drop(drop_id: str, record: dict | None) -> bool:
    """Log what a top-up of this drop would do; True if it is safe."""
    if record is None:
        logging.error(f"{drop_id}: not in the deployments table")
        return False
    start, end = record.get("sampling_start"), record.get("sampling_end")
    if start is None or end is None:
        logging.error(f"{drop_id}: no sampling window in the DB")
        return False

    grid = _select_all_clips(drop_id, start, end, config.clip_length, config.clip_cap)
    planned = {CLIP_KIND.upl_seconds_fn(row) for _, row in grid.iterrows()}

    existing = fetch_existing_set(CLIP_KIND, drop_id)
    if existing is None:
        logging.info(
            f"{drop_id}: no {CLIP_KIND.set_name_prefix}{drop_id} set in project "
            f"{config.zooniverse_project_id}, nothing to top up"
        )
        return False

    live, filenames = _live_starts(existing.subjects)
    unreadable = len(existing.subjects) - len(live)
    off_grid = live - planned
    new = planned - live
    final = len(live) + len(new) + unreadable

    logging.info(
        f"{drop_id}: window {start}-{end} s ({end - start} s), grid {len(planned)} "
        f"clips | set {existing.subject_set.id}: {len(existing.subjects)} live, "
        f"{len(live & planned)} on the grid, {len(off_grid)} off it | "
        f"would upload {len(new)}, set ends at {final}"
    )
    problems = []
    if off_grid:
        problems.append(
            f"{len(off_grid)} live clip(s) off the grid, e.g. "
            f"{sorted(off_grid)[:5]} (sampling_start changed, or the cap thinned "
            "the grid)"
        )
    if unreadable:
        problems.append(
            f"{unreadable} subject(s) without a start second, the upload cannot "
            "match them"
        )
    if len(filenames) > 1:
        problems.append(f"set holds clips from several videos: {sorted(filenames)}")
    for problem in problems:
        logging.warning(f"{drop_id}: NOT SAFE, {problem}")
    if not problems:
        logging.info(f"{drop_id}: safe, every live clip is on the grid")
    return not problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drop_ids", nargs="+", help="DropIDs whose clip sets to check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    drop_ids = [config.validate_drop_id(d) for d in args.drop_ids]
    records = DatabaseManager().get_deployments_by_ids(drop_ids)
    results = [check_drop(d, records.get(d)) for d in drop_ids]
    logging.info(f"{sum(results)} of {len(results)} drop(s) safe to top up")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
