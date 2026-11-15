"""Pick example frames with confident model-only boxes, for a look on the Mac.

Reads ``model_only_<predictions>.csv`` (audit_dataset.py), keeps boxes at or
above --min-conf, drops the excluded drops, ranks drops by how many such boxes
they have and takes up to --per-drop frames from each (the frames with the
most of them) until --frames are chosen. Writes a list of paths relative to the
repo root, the CSV itself first, so ONE rsync from the Mac pulls everything::

    # on NeSI
    TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/sample_review_frames.py --exclude volume_30505
    # on the Mac
    rsync -av --files-from=mahuika:/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/process_files/wip_two_stage/sample_frames.txt \\
        mahuika:/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/ process_files/nesi_mirror/pulled/
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR, frames_on_disk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", default="model_only_detector_predictions.csv")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--exclude", nargs="*", default=[], help="drops to leave out")
    ap.add_argument(
        "--drops", nargs="*", default=[], help="only these drops, in this order"
    )
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--per-drop", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(OUT_DIR / args.csv)
    df = df[(df["conf"] >= args.min_conf) & ~df["drop"].isin(args.exclude)]
    if args.drops:
        df = df[df["drop"].isin(args.drops)]
        order = [d for d in args.drops if d in set(df["drop"])]
    else:
        order = df.groupby("drop").size().sort_values(ascending=False).index
    picked = []
    for drop in order:
        g = df[df["drop"] == drop]
        top = (
            g.groupby("filename")["conf"]
            .agg(["size", "max"])
            .sort_values(["size", "max"], ascending=False)
        )
        picked += list(top.index[: args.per_drop])
        if len(picked) >= args.frames:
            break
    picked = picked[: args.frames]

    frames = frames_on_disk()
    root = config.project_root.resolve()
    rel = []
    for f in picked:
        target = Path(os.path.realpath(frames[f]))
        rel.append(str(target.relative_to(root)))
    csv_rel = str((OUT_DIR / args.csv).resolve().relative_to(root))
    out = OUT_DIR / "sample_frames.txt"
    out.write_text("\n".join([csv_rel] + rel) + "\n")
    print(
        f"{len(rel)} frames from {df[df['filename'].isin(picked)]['drop'].nunique()} drops "
        f"(boxes >= {args.min_conf}: {len(df)} in {df['drop'].nunique()} drops) -> {out}"
    )


if __name__ == "__main__":
    main()
