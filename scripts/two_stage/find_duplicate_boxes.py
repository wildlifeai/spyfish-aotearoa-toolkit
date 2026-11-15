"""Find boxes that sit on (almost) the same spot on one frame, in the BIIGLE exports.

Reads every BIIGLE raw CSV under ``deployment_data`` (``*_biigle_expert_raw.csv``
from review volumes, ``*_biigle_training_raw.csv`` from survey training
volumes, the curated volumes' CSVs), so it checks what BIIGLE holds, not the
training labels made from it. For each frame it compares every pair of
rectangles and reports pairs with IoU >= --min-iou:

* ``identical``: the same corner points, i.e. one box uploaded or drawn twice
  (a double ML upload into a training volume looks like this);
* ``same_label`` / different labels: two names for one fish teaches the model
  one animal as two classes.

Each row carries both annotation ids, so the duplicate can be found and
deleted in BIIGLE. Runs anywhere the CSVs are (NeSI has all of them)::

    python scripts/two_stage/find_duplicate_boxes.py                 # IoU >= 0.9
    python scripts/two_stage/find_duplicate_boxes.py --min-iou 1.0   # exact copies only

Writes ``duplicate_boxes.csv`` next to the other two-stage outputs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR, find_biigle_raw_csvs  # noqa: E402


def boxes_from(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    need = {"filename", "label_name", "points", "shape_name"}
    if df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    df = df[df["shape_name"] == "Rectangle"].copy()
    pts = df["points"].map(json.loads)
    xs, ys = pts.map(lambda p: p[0::2]), pts.map(lambda p: p[1::2])
    df["x0"], df["x1"] = xs.map(min), xs.map(max)
    df["y0"], df["y1"] = ys.map(min), ys.map(max)
    df["source_csv"] = csv.name
    keep = [
        "filename",
        "label_name",
        "annotation_id",
        "image_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "source_csv",
    ]
    if "volume_id" in df.columns:
        keep.append("volume_id")
    return df[[c for c in keep if c in df.columns]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--min-iou", type=float, default=0.9)
    args = ap.parse_args()

    csvs = find_biigle_raw_csvs(
        (config.biigle_expert_raw_suffix, config.biigle_training_raw_suffix),
        any_depth=True,
    )
    parts = [boxes_from(p) for p in csvs]
    boxes = pd.concat([p for p in parts if len(p)], ignore_index=True)
    # the same frame can sit in two exports (review + training of one drop);
    # compare within one export only, cross-export overlap is not a duplicate in BIIGLE
    rows = []
    for (csv, fname), g in boxes.groupby(["source_csv", "filename"]):
        if len(g) < 2:
            continue
        x0, y0, x1, y1 = (g[c].to_numpy(float) for c in ("x0", "y0", "x1", "y1"))
        area = (x1 - x0) * (y1 - y0)
        iw = np.clip(np.minimum.outer(x1, x1) - np.maximum.outer(x0, x0), 0, None)
        ih = np.clip(np.minimum.outer(y1, y1) - np.maximum.outer(y0, y0), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
        i, j = np.triu_indices(len(g), 1)
        for a, b in zip(i, j):
            if iou[a, b] + 1e-9 < args.min_iou:
                continue
            ra, rb = g.iloc[a], g.iloc[b]
            rows.append(
                {
                    "source_csv": csv,
                    "filename": fname,
                    "iou": round(float(iou[a, b]), 4),
                    "identical": bool(
                        np.allclose(
                            [ra.x0, ra.y0, ra.x1, ra.y1],
                            [rb.x0, rb.y0, rb.x1, rb.y1],
                            atol=0.5,
                        )
                    ),
                    "label_a": ra.label_name,
                    "label_b": rb.label_name,
                    "same_label": ra.label_name == rb.label_name,
                    # BIIGLE exports one row per (annotation, label): the same id twice is ONE
                    # box carrying two labels, not two boxes; delete the extra label, not a box
                    "same_annotation": pd.notna(ra.get("annotation_id"))
                    and ra.get("annotation_id") == rb.get("annotation_id"),
                    "annotation_id_a": ra.get("annotation_id"),
                    "annotation_id_b": rb.get("annotation_id"),
                    "image_id": ra.get("image_id"),
                    "volume_id": ra.get("volume_id", ""),
                    "biigle_image": (
                        f"https://biigle.de/images/{int(ra.image_id)}"
                        if pd.notna(ra.get("image_id"))
                        else ""
                    ),
                }
            )
    out = pd.DataFrame(rows)
    print(
        f"{len(boxes)} rectangles in {len(csvs)} BIIGLE CSVs; {len(out)} pairs with IoU >= {args.min_iou}"
    )
    if len(out):
        print(
            f"one box with two labels (same annotation id): {int(out['same_annotation'].sum())}; "
            f"two separate boxes: {int((~out['same_annotation']).sum())}; "
            f"of all pairs, different labels: {int((~out['same_label']).sum())}\n"
        )
        summary = out.groupby("source_csv").agg(
            pairs=("iou", "size"),
            identical=("identical", "sum"),
            different_labels=("same_label", lambda s: int((~s).sum())),
        )
        print(summary.sort_values("pairs", ascending=False).to_string())
    path = OUT_DIR / "duplicate_boxes.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
