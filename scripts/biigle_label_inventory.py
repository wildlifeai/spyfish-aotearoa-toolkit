"""Inventory of BIIGLE training labels: which labels sit in which drops.

Reads only files already on disk (no BIIGLE calls): every drop's
`_biigle_expert_raw.csv` and `_biigle_training_raw.csv`, plus empty YOLO
`.txt` files in the drop's `labels/` dir (reviewed frames with no animal).
Labels are raw BIIGLE names, before training's species canonicalisation;
workflow labels ("Done ...", "Nothing here", ...) are dropped.

Writes to process_files/training_lists/:
  biigle_label_by_drop.csv   one row per (label, drop): boxes, frames, sources
  biigle_label_summary.csv   one row per label: drops, frames, boxes, drop list

Input for scripts/propose_test_split.py. Run where training runs (NeSI), since the
local and NeSI deployment_data trees are different snapshots.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from spyfish.biigle.biigle_to_yolo import _is_workflow_label  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402

root = config.deployment_data_dir
parts, drop_dirs = [], {}
for src in ("expert", "training"):
    for f in root.rglob(f"*_biigle_{src}_raw.csv"):
        df = pd.read_csv(
            f, usecols=lambda c: c in {"label_name", "filename"}, low_memory=False
        )
        df["drop_id"], df["source"] = f.parent.parent.name, src
        parts.append(df)
        drop_dirs[f.parent.parent.name] = f.parent.parent
ann = pd.concat(parts, ignore_index=True)
ann = ann[~ann["label_name"].astype(str).map(_is_workflow_label)]

# Background frames: an empty YOLO .txt in the drop's labels/ dir is a
# reviewed frame with no animal (written by biigle_to_yolo for both sources).
empty = []
for drop, d in drop_dirs.items():
    for txt in (d / "labels").glob("*.txt"):
        if not txt.read_text().strip():
            empty.append(
                {
                    "drop_id": drop,
                    "filename": txt.stem,
                    "label_name": "(empty frame)",
                    "source": "labels",
                }
            )
ann = pd.concat([ann, pd.DataFrame(empty)], ignore_index=True)

long = (
    ann.groupby(["label_name", "drop_id"])
    .agg(
        n_boxes=("filename", "size"),
        n_frames=("filename", "nunique"),
        sources=("source", lambda s: ";".join(sorted(set(s)))),
    )
    .reset_index()
)
long.loc[long["label_name"] == "(empty frame)", "n_boxes"] = 0
summary = (
    long.groupby("label_name")
    .agg(
        n_drops=("drop_id", "nunique"),
        n_frames=("n_frames", "sum"),
        n_boxes=("n_boxes", "sum"),
        drop_ids=("drop_id", lambda s: ";".join(sorted(s))),
    )
    .reset_index()
    .sort_values("n_drops", ascending=False)
)
out = config.training_excluded_drops_file.parent
out.mkdir(parents=True, exist_ok=True)
long.to_csv(out / "biigle_label_by_drop.csv", index=False)
summary.to_csv(out / "biigle_label_summary.csv", index=False)
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(summary.drop(columns="drop_ids").to_string(index=False))
print(
    f"\n{ann['drop_id'].nunique()} drops. Wrote {out}/biigle_label_by_drop.csv and biigle_label_summary.csv"
)
