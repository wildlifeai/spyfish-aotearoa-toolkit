"""Run the current species YOLO model on every expert-annotated frame on disk.

Gives the "fair comparison" side of plan §4.4: the current model's boxes and
labels on the same frames the classifier is evaluated on. Writes
``yolo_predictions.csv`` (filename, class, conf, x_min, y_min, x_max, y_max).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR, load_boxes, pick_device  # noqa: E402
from spyfish.ml.model_meta import resolve_imgsz  # noqa: E402

MODEL = (
    config.pipeline_model_path
)  # the same file --ml runs: newest training stamp in pipeline_model/


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=None, help="weights to run; default: the pipeline model"
    )
    parser.add_argument(
        "--out",
        default=artifacts.YOLO_PREDICTIONS,
        help="file name under process_files/wip_two_stage/",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="default: the size the weights were trained at",
    )
    parser.add_argument(
        "--conf", type=float, default=None, help="default: config.confidence_threshold"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="frames per predict() call. Ultralytics caps NMS at 2 s + 0.05 s per frame in the "
        "batch and DROPS the remaining frames' boxes when it runs out ('NMS time limit exceeded'); "
        "on a Mac at imgsz 1280 that fires at 32, so use 4 there. Fine at 32 on an NVIDIA GPU.",
    )
    args = parser.parse_args()
    weights = Path(args.model) if args.model else MODEL
    imgsz = args.imgsz or resolve_imgsz(weights)
    conf = args.conf if args.conf is not None else config.confidence_threshold

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_boxes().drop_duplicates("filename")[["filename", "frame_path"]]
    model = YOLO(str(weights))
    print(
        f"{weights.name}: classes {list(model.names.values())}, imgsz {imgsz}, conf {conf}"
    )
    rows = []
    paths = [str(p) for p in frames["frame_path"]]
    for start in range(0, len(paths), args.batch):
        batch = paths[start : start + args.batch]
        results = model.predict(
            source=batch,
            conf=conf,
            imgsz=imgsz,
            iou=config.ml_nms_iou,
            agnostic_nms=config.ml_nms_agnostic,
            device=pick_device(),
            half=True,
            verbose=False,
        )
        for res, path in zip(results, batch):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                rows.append(
                    (
                        Path(path).name,
                        model.names[int(b.cls[0])],
                        float(b.conf[0]),
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )
        print(f"{min(start + args.batch, len(paths))}/{len(paths)} frames", flush=True)
    out = pd.DataFrame(
        rows,
        columns=["filename", "yolo_class", "conf", "x_min", "y_min", "x_max", "y_max"],
    )
    out.to_csv(OUT_DIR / args.out, index=False)
    print(
        f"wrote {len(out)} boxes on {out['filename'].nunique()} frames -> {OUT_DIR / args.out}"
    )


if __name__ == "__main__":
    main()
