"""Re-pull the expert annotations of FINISHED review volumes after a "To review" pass.

``--biigle-sync`` only syncs drops still at ``expert_uploaded``; a drop already
``expert_complete`` is never pulled again, so edits made since (relabelled
"To review" boxes, deletions) never reach ``{drop}_biigle_expert_raw.csv``,
which is what training reads for review frames. This script re-downloads the
volume report (one BIIGLE report call per volume) for the given drops and
rewrites that CSV, after keeping the old one as
``{drop}_biigle_expert_raw.before_resync_<stamp>.csv`` (never overwritten).

Scope: the raw CSV only, i.e. training labels. The annotations database and
MaxN are not touched; fish added in review can raise a drop's MaxN, and that
belongs to a proper ``--biigle-sync`` re-run path, not to this script.

Training volumes are not handled here: re-pull those with
``python -m spyfish.biigle.biigle_to_yolo download-training-volume --volume-id <id>``,
and volume 30505 with ``download-volume --volume-id 30505``.

    TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/resync_review_volumes.py --from-manifests
    ... --from-manifests --write
    ... --all            # every finished review volume (done project); add --write to pull
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.manager import DatabaseManager  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR  # noqa: E402


def drops_from_manifests() -> dict[str, set[int]]:
    """{drop: volumes its "To review" boxes went to}, from every upload manifest."""
    out: dict[str, set[int]] = {}
    for p in sorted(OUT_DIR.glob(artifacts.REVIEW_UPLOAD_GLOB)):
        m = pd.read_csv(p)
        if "volume_id" not in m.columns or "drop" not in m.columns:
            continue
        for drop, vid in (
            m.dropna(subset=["volume_id"]).groupby(["drop", "volume_id"]).size().index
        ):
            out.setdefault(drop, set()).add(int(vid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--drops", nargs="+")
    pick.add_argument(
        "--from-manifests",
        action="store_true",
        help="every drop whose To review boxes went to its own review volume",
    )
    pick.add_argument(
        "--all",
        action="store_true",
        help="every drop whose review volume is in the done project (one list call to find them)",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=5.0,
        help="seconds between volume exports, to stay well under BIIGLE's rate limit",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="download and rewrite (default: list what would be re-pulled)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    deployments = DatabaseManager().get_all_deployments_map()
    if args.all:
        from spyfish.biigle.biigle_handler import BiigleHandler

        vols_done = BiigleHandler().get_volumes(config.biigle_done_project_id)
        done = {int(v["id"]) for v in vols_done}
        sent = {
            d: set()
            for d, r in deployments.items()
            if r.get("biigle_volume_id") and int(r["biigle_volume_id"]) in done
        }
        print(
            f"{len(sent)} drops have their review volume in the done project ({len(done)} volumes there)"
        )
        # volumes no drop points to: if the name is a DropID, it is that drop's
        # review volume and the database just lacks the id; anything else is listed
        linked = {int(deployments[d]["biigle_volume_id"]) for d in sent}
        extra_todo, unknown = [], []
        for v in vols_done:
            vid, name = int(v["id"]), str(v.get("name", ""))
            if vid in linked:
                continue
            # every volume in done is an image volume; names like
            # "TON_..._02 — ML frames": the drop is the first token
            lead = name.split()[0] if name.split() else ""
            if lead in deployments:
                extra_todo.append((lead, vid))
            else:
                unknown.append((vid, name))
        if extra_todo:
            print(f"by name, not in the database: {extra_todo}")
        if unknown:
            print("in the done project but matched to no drop (not re-pulled):")
            for vid, name in unknown:
                print(f"  {vid}  {name}")
    else:
        sent = (
            drops_from_manifests()
            if args.from_manifests
            else {d: set() for d in args.drops}
        )
        extra_todo = []
    todo = list(extra_todo)
    for drop, vols in sorted(sent.items()):
        review_vid = (deployments.get(drop) or {}).get("biigle_volume_id")
        if not review_vid:
            continue  # no review volume: training-volume or curated-volume boxes, re-pulled elsewhere
        if vols and int(review_vid) not in vols:
            continue  # this drop's boxes went to a training volume, not its review volume
        todo.append((drop, int(review_vid)))
    if not todo:
        raise SystemExit("no review-volume drops to re-pull")
    share = pd.Series([v for _, v in todo]).value_counts()
    print(pd.DataFrame(todo, columns=["drop", "review_volume"]).to_string(index=False))
    if not args.write:
        print(f"\n{len(todo)} drops in {len(share)} volumes; add --write to re-pull")
        return

    from spyfish.biigle.biigle_parser import BiigleParser
    from spyfish.biigle.sync_annotations import rows_for_drop

    parser, cache, stamp = BiigleParser(), {}, time.strftime("%Y%m%d_%H%M%S")
    for drop, vid in todo:
        if vid not in cache:
            if cache:
                time.sleep(args.pause)
            cache[vid] = parser.download_volume_annotations(
                volume_id=vid, type_id=config.annotation_report_type_images
            )
        report = cache[vid]
        # a survey review volume holds several drops: keep this drop's rows only
        if not report.empty and "filename" in report.columns:
            if not report["filename"].astype(str).str.contains(drop, regex=False).all():
                report = rows_for_drop(report, drop)
        raw = config.get_biigle_expert_raw_csv_path(drop)
        before = len(pd.read_csv(raw)) if raw.exists() else 0
        if raw.exists():
            backup = raw.with_name(raw.stem + f".before_resync_{stamp}.csv")
            if backup.exists():
                raise SystemExit(f"{backup} exists; not overwriting a backup")
            shutil.copy2(raw, backup)
        raw.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(raw, index=False)
        print(
            f"{drop}: volume {vid}, rows {before} -> {len(report)}; old CSV kept as *.before_resync_{stamp}.csv"
        )


if __name__ == "__main__":
    main()
