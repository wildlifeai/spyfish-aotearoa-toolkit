"""Score any detector's boxes against the expert boxes: recall per species, by split.

    TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/detector_recall.py detector_predictions.csv
    TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/detector_recall.py yolo_predictions.csv   # the species model, for comparison

Recall = share of expert boxes of that species matched by ANY detector box at
IoU >= 0.5, whatever the detector called it; that is the question a three-class
detector is built to answer. Precision = share of detector boxes that match some
expert box (the rest are hard-negative candidates, or fish the expert skipped).
Boxes the detector labels `bait` are left out of precision: experts rarely box bait.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from spyfish.config.species import species_registry  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import (  # noqa: E402
    OUT_DIR,
    load_boxes,
    read_predictions,
)
from spyfish.ml.classifier.metrics import recall_table, score_predictions  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else artifacts.YOLO_PREDICTIONS
    pred = read_predictions(OUT_DIR / name)
    experts = load_boxes()
    if "split" not in experts.columns:
        experts["split"] = "all"
    common = species_registry().common_name_map()
    experts, summary = score_predictions(pred, experts)
    print(
        f"{name}: {len(pred)} boxes; expert boxes matched {summary['recall']:.3f} overall; "
        f"detector boxes that match an expert box {summary['precision']:.3f}\n"
    )
    for split, g in experts.groupby("split"):
        print(f"## {split}: {len(g)} expert boxes, recall {g['found'].mean():.3f}\n")
        print(recall_table(g, common).round(3).to_markdown(), "\n")


if __name__ == "__main__":
    main()
