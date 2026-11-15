"""Build a detector dataset (fish, or fish + a few kept classes) from the species dataset.

Plan §5.2: every species label becomes `fish`, except the classes in --classes,
which keep their own. The detector then learns "animal here" from every expert
box there is, and the species stay with whoever names them.

    --classes fish Jasus edwardsii bait            # the three-class run (default)
    --classes fish --out detector_fish1            # one class: every box is a fish to the finder
    --classes fish "Parapercis colias" --out detector_bluecod   # one named class + catch-all

Bait: the only visible bait fish in the corpus is in volume 30505 ("Old images",
where it lay on top of the box) and it is labelled there; in current surveys the
bait sits inside the box and is never seen. So a bait box folds into `fish` like
any other fish. A --drop option that trained those boxes as background existed
until 2026-11-10 and was removed: it taught the finder that a visible fish is not one.

Reads  training.local_training_dir/species/  (images/{split} symlinks, labels/{split}, data.yaml)
Writes training.local_training_dir/detector/ (same splits, same frames, remapped labels, data.yaml)

The frozen test split is carried over untouched, so the detector is scored on
the same drops as everything else. Images are symlinked to the real frames
(the species dataset's own link targets), never copied.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.training.yolo_labels import (  # noqa: E402
    DATA_YAML_FILENAME,
    SPLITS,
    read_class_names,
    remap_label_file,
)

DETECTOR_CLASSES = ["fish", "Jasus edwardsii", "bait"]  # frozen order for this detector


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DETECTOR_CLASSES,
        help="kept classes, first must be fish",
    )
    parser.add_argument(
        "--out", default="detector", help="subfolder of local_training_dir to write"
    )
    args = parser.parse_args()
    keep = list(args.classes)
    assert keep[0] == "fish", "the catch-all must be class 0"

    src = config.training_dataset_dir
    dst = config.local_training_dir / args.out
    names = read_class_names(config.training_data_yaml, required=True)
    remap = {i: (keep.index(n) if n in keep else 0) for i, n in enumerate(names)}
    print(
        f"{len(names)} species classes -> {len(keep)} detector classes: "
        + ", ".join(f"{n}->{keep[remap[i]]}" for i, n in enumerate(names) if n in keep)
        + "; every other class -> fish"
    )

    counts = {split: {c: 0 for c in keep} for split in SPLITS}
    for split in SPLITS:
        src_img, src_lbl = src / "images" / split, src / "labels" / split
        if not src_img.is_dir():
            continue
        dst_img, dst_lbl = dst / "images" / split, dst / "labels" / split
        for d in (dst_img, dst_lbl):
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
            d.mkdir(parents=True, exist_ok=True)
        n_img = 0
        for img in sorted(src_img.iterdir()):
            if img.suffix.lower() not in set(config.image_extensions):
                continue
            target = Path(os.readlink(img)) if img.is_symlink() else img
            if not target.is_absolute():
                target = (img.parent / target).resolve()
            os.symlink(target, dst_img / img.name)
            n_img += 1
            lbl = src_lbl / f"{img.stem}.txt"
            out_lbl = dst_lbl / f"{img.stem}.txt"
            if not lbl.exists():
                out_lbl.write_text("")
                continue
            # remap covers every class in data.yaml, so an unknown id is a
            # corrupt label file and raises
            seen = remap_label_file(
                lbl, remap, dst=out_lbl, missing="raise", trailing_newline=True
            )
            for old_id, n in seen.items():
                counts[split][keep[remap[old_id]]] += n
        print(f"{split}: {n_img} images, boxes {counts[split]}")

    data = {
        "path": str(dst.resolve()),
        "train": str((dst / "images" / "train").resolve()),
        "val": str((dst / "images" / "val").resolve()),
        "names": {i: n for i, n in enumerate(keep)},
        "nc": len(keep),
    }
    if (dst / "images" / "test").is_dir():
        data["test"] = str((dst / "images" / "test").resolve())
    (dst / DATA_YAML_FILENAME).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {dst / DATA_YAML_FILENAME}")


if __name__ == "__main__":
    main()
