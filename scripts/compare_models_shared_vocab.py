"""Compare species checkpoints on images NONE of them trained on.

The three models were trained with incompatible class-id schemes (17 / 20 / 50
classes, two alphabetical, the newest in frozen roster order). Pointing
`model.val` at another model's data.yaml is what produced `KeyError: 16` in the
pipeline: ids mean different species in each scheme, and Ultralytics silently
drops names it cannot resolve.

So each model is scored on identical IMAGES with GT rewritten into its own id
space, restricted to the vocabulary they all share:

  * GT species outside the shared set are remapped to 'fish' (what each model's
    own class floor would have done with them anyway).
  * `classes=` restricts each model's PREDICTIONS to the shared ids, so a model
    is not charged false positives for knowing species the others never saw.

CONTAMINATION FILTER (2026-08-25). The live `training/species/` workspace is
rebuilt by every data prep, so its val split is NOT the one any given model was
validated against — parts of it were in that model's TRAINING set. Scoring a
model on images it trained on flatters it, and the flattery is uneven, so the
ranking tracks training exposure rather than skill. Measured on the 20260822 /
20260823 pair: 171 of 640 val images (26.7%) had been in the older model's train
split, and the older model then scored HIGHER on the newer model's val than on
its own — the signature of exactly that.

This script therefore excludes any image appearing in ANY model's frozen
train.txt snapshot, leaving only images none of them has seen. A model with no
snapshot (20260821 kept none, the feature postdates it) cannot be filtered and
is reported as UNFILTERABLE rather than quietly compared.
"""

import argparse
import csv
import json
import os
import shutil

from ultralytics import YOLO

BASE = "/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/process_files/training"
RUNS = f"{BASE}/runs"
SRC_IMAGES = f"{BASE}/species/images/val"
SRC_LABELS = f"{BASE}/species/labels/val"
OUT = os.path.expanduser("~/model_compare")
FALLBACK = "fish"

# Default set. Override with positional run names:
#   python scripts/compare_models_shared_vocab.py 20260823_031610 20260825_053442
#
# Comparing exactly TWO is usually the better question ("did this run beat the
# last one?"), and it keeps far more images: the clean set excludes every image
# in ANY compared model's train split, so each extra model shrinks it. 20260821
# is deliberately absent from the default - it kept no dataset snapshot, so its
# training images cannot be excluded and including it silently reintroduces the
# contamination this filter exists to remove.
DEFAULT_MODELS = [
    "20260822_192536",
    "20260823_031610",
    "20260825_053442",
]

parser = argparse.ArgumentParser(
    description="Compare species checkpoints on images none of them trained on."
)
parser.add_argument(
    "runs",
    nargs="*",
    default=None,
    help="run names, e.g. 20260823_031610 20260825_053442 (default: three latest)",
)
parser.add_argument(
    "--allow-unfilterable",
    action="store_true",
    help="proceed even if a model has no snapshot (its score may be inflated)",
)
args = parser.parse_args()

run_names = args.runs or DEFAULT_MODELS
MODELS = {r: f"{RUNS}/{r}_species/weights/best.pt" for r in run_names}
missing = [r for r, p in MODELS.items() if not os.path.exists(p)]
if missing:
    raise SystemExit(f"No best.pt for: {missing}")
if len(MODELS) < 2:
    raise SystemExit("Need at least two models to compare.")


def _drop_of(fname):
    return fname.split("__frame")[0]


# ── Build the contamination-free evaluation set ──────────────────────────────
val_images = sorted(
    f for f in os.listdir(SRC_IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png"))
)
seen_by, unfilterable = {}, []
for tag in MODELS:
    train_txt = f"{RUNS}/{tag}_species/dataset/train.txt"
    if not os.path.exists(train_txt):
        unfilterable.append(tag)
        continue
    seen_by[tag] = {
        line.strip()
        for line in open(train_txt)
        if line.strip().lower().endswith(".jpg")
    }

contaminated = set()
for s in seen_by.values():
    contaminated |= s
clean = [f for f in val_images if f not in contaminated]

print("=" * 78)
print("EVALUATION SET")
print("=" * 78)
print(
    f"  current val split           : {len(val_images):>5} images from "
    f"{len({_drop_of(f) for f in val_images}):>4} deployments"
)
for tag in MODELS:
    if tag in seen_by:
        overlap = [f for f in val_images if f in seen_by[tag]]
        print(
            f"  seen in {tag} train : {len(overlap):>5} images from "
            f"{len({_drop_of(f) for f in overlap}):>4} deployments  (excluded)"
        )
    else:
        print(f"  {tag}: NO SNAPSHOT — cannot filter, results UNFILTERABLE")
print(
    f"  -> clean evaluation set     : {len(clean):>5} images from "
    f"{len({_drop_of(f) for f in clean}):>4} deployments"
)
if unfilterable:
    print(
        f"  WARNING: {unfilterable} lack snapshots; their scores may include "
        f"images they trained on."
    )
if unfilterable and not args.allow_unfilterable:
    raise SystemExit(
        f"{unfilterable} have no dataset snapshot, so images they trained on "
        f"cannot be excluded and their scores would be inflated. Drop them, or "
        f"pass --allow-unfilterable if you accept that."
    )
if not clean:
    raise SystemExit("No uncontaminated images left — nothing to compare.")

names_by_model = {tag: YOLO(p).names for tag, p in MODELS.items()}
shared = sorted(set.intersection(*[set(n.values()) for n in names_by_model.values()]))
print(f"\nshared vocabulary: {len(shared)} classes\n{shared}\n")
# The live labels are written in the newest prep's unified ordering, so
# decode them with the newest model's names.
cur_names = names_by_model[sorted(MODELS)[-1]]

os.makedirs(OUT, exist_ok=True)
results = {
    "_eval_images": len(clean),
    "_eval_deployments": len({_drop_of(f) for f in clean}),
    "_unfilterable": unfilterable,
    "_shared_vocabulary": shared,
}

for tag, weights in MODELS.items():
    names = names_by_model[tag]
    name_to_id = {v: k for k, v in names.items()}
    root = f"{OUT}/{tag}"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(f"{root}/images/val")
    os.makedirs(f"{root}/labels/val")

    # Hardlinks, not symlinks: Ultralytics resolves the image path BEFORE
    # swapping '/images/' for '/labels/', so a symlinked dir resolves back to
    # the original tree and the run silently scores against the ORIGINAL labels
    # (seen 2026-08-23: two models scored mAP 0.002 that way).
    remapped = 0
    for fn in clean:
        try:
            os.link(f"{SRC_IMAGES}/{fn}", f"{root}/images/val/{fn}")
        except OSError:
            shutil.copy2(f"{SRC_IMAGES}/{fn}", f"{root}/images/val/{fn}")
        stem = fn.rsplit(".", 1)[0]
        src_lbl = f"{SRC_LABELS}/{stem}.txt"
        lines = []
        if os.path.exists(src_lbl):
            for line in open(src_lbl):
                parts = line.split()
                if len(parts) != 5:
                    continue
                species = cur_names[int(parts[0])]
                if species not in name_to_id:
                    species = FALLBACK
                    remapped += 1
                lines.append(" ".join([str(name_to_id[species])] + parts[1:]))
        open(f"{root}/labels/val/{stem}.txt", "w").write("\n".join(lines) + "\n")

    yaml_path = f"{root}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\nval: images/val\ntrain: images/val\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i in range(len(names)):
            f.write(f"  {i}: {names[i]}\n")

    keep_ids = sorted(name_to_id[s] for s in shared)
    print(
        f"\n=== {tag}: nc={len(names)}, {remapped} GT boxes remapped to fish, "
        f"scoring {len(keep_ids)} shared classes on {len(clean)} images",
        flush=True,
    )
    m = YOLO(weights).val(
        data=yaml_path,
        split="val",
        imgsz=640,
        plots=True,
        classes=keep_ids,
        project=OUT,
        name=f"val_{tag}",
        exist_ok=True,
        verbose=True,
    )
    results[tag] = {
        "nc": len(names),
        "map50": round(float(m.box.map50), 4),
        "map50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class": [
            {
                "name": names[int(ci)],
                "mAP50": round(float(m.box.ap50[i]), 4),
                "mAP50_95": round(float(m.box.ap[i]), 4),
                "P": round(float(m.box.p[i]), 4),
                "R": round(float(m.box.r[i]), 4),
            }
            for i, ci in enumerate(m.box.ap_class_index)
        ],
    }

json.dump(results, open(f"{OUT}/comparison.json", "w"), indent=2)
with open(f"{OUT}/comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "nc", "mAP50", "mAP50_95", "precision", "recall"])
    for tag in MODELS:
        r = results[tag]
        w.writerow(
            [tag, r["nc"], r["map50"], r["map50_95"], r["precision"], r["recall"]]
        )

print("\n" + "=" * 78)
print(
    f"SCORED ON {results['_eval_images']} images from "
    f"{results['_eval_deployments']} deployments, seen by no model"
)
print("=" * 78)
print(f"{'model':<20}{'nc':>4}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}")
for tag in MODELS:
    r = results[tag]
    print(
        f"{tag:<20}{r['nc']:>4}{r['map50']:>9.4f}{r['map50_95']:>10.4f}"
        f"{r['precision']:>8.4f}{r['recall']:>8.4f}"
    )
print(f"\nWROTE {OUT}/comparison.json")
