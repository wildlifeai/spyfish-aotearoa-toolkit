"""Which BIIGLE project each volume is in, when it last changed, and what is still open.

For the volumes of the given drops (or of every drop with a review or training
volume in the pipeline database):

* project: in progress / done / training, from three list calls (one per
  project), no per-volume requests;
* last_change: the newest ``created_at`` in the volume's LOCAL export, i.e.
  the last box drawn or label set when it was last pulled. Deletions leave no
  timestamp, and edits made since the last pull are not seen: re-pull first
  for an up-to-date answer;
* open checks: boxes still labelled "To review" or "ML: check".

    python scripts/two_stage/biigle_volume_status.py --drops SLI_20260114_BUV_SLI_117_02 TON_20221205_BUV_TON_001_01
    python scripts/two_stage/biigle_volume_status.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.manager import DatabaseManager  # noqa: E402

CHECK_LABELS = {"To review", "ML: check"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--drops", nargs="+")
    pick.add_argument("--all", action="store_true")
    ap.add_argument(
        "--no-api",
        action="store_true",
        help="skip the three BIIGLE list calls (no project column)",
    )
    args = ap.parse_args()

    deployments = DatabaseManager().get_all_deployments_map()
    drops = sorted(deployments) if args.all else args.drops
    rows = []
    for d in drops:
        r = deployments.get(d) or {}
        for kind, col, suffix in (
            ("review", "biigle_volume_id", config.biigle_expert_raw_suffix),
            (
                "training",
                "training_biigle_volume_id",
                config.biigle_training_raw_suffix,
            ),
        ):
            if not r.get(col):
                continue
            csv = config.get_drop_annotations_dir(d) / f"{d}{suffix}"
            last, n, open_checks = "", 0, 0
            if csv.exists():
                df = pd.read_csv(
                    csv, usecols=lambda c: c in {"created_at", "label_name"}
                )
                n = len(df)
                if "created_at" in df.columns and n:
                    last = str(df["created_at"].max())
                if "label_name" in df.columns:
                    open_checks = int(df["label_name"].isin(CHECK_LABELS).sum())
            rows.append(
                {
                    "drop": d,
                    "kind": kind,
                    "volume_id": int(r[col]),
                    "local_rows": n,
                    "last_change": last,
                    "open_checks": open_checks,
                    "local_csv": csv.exists(),
                }
            )
    if not rows:
        raise SystemExit("none of these drops has a BIIGLE volume in the database")
    out = pd.DataFrame(rows)

    if not args.no_api:
        from spyfish.biigle.biigle_handler import BiigleHandler

        handler = BiigleHandler()
        where = {}
        for name, pid in (
            ("in progress", config.biigle_upload_project_id),
            ("done", config.biigle_done_project_id),
            ("training", config.biigle_training_project_id),
        ):
            for v in handler.get_volumes(pid):
                where.setdefault(int(v["id"]), []).append(f"{name} ({pid})")
        out["project"] = out["volume_id"].map(
            lambda v: ", ".join(where.get(v, ["not in these projects"]))
        )

    pd.set_option("display.width", 220)
    print(out.sort_values(["drop", "kind"]).to_string(index=False))


if __name__ == "__main__":
    main()
