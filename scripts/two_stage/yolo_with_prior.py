"""Put the species × reserve prior into the current YOLO model, at inference.

YOLO's head scores every class on every box with an independent sigmoid;
Ultralytics keeps only the argmax after NMS. Here the raw (4 + 22) × 8400
tensor is taken before that step. Detection is left exactly as it is (the
confidence filter and NMS use the original top score), and only the class
decision changes: ``argmax(scores * prior[reserve])``. So a box never appears
or disappears because of the prior; it can only be renamed.

Separate from the pipeline on purpose. Writes to the two-stage output folder:
    yolo_prior_predictions.csv   per detected box: original class, reweighted class, both scores
    yolo_prior_report.md         per-class tables with and without the prior, on expert-matched boxes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier.classes import OTHER, classes_for  # noqa: E402
from spyfish.ml.classifier.crops import (  # noqa: E402
    OUT_DIR,
    load_boxes,
    pick_device,
    read_crops_meta,
)
from spyfish.ml.classifier.metrics import (  # noqa: E402
    load_site_prior,
    match_yolo,
    per_class_table,
)
from spyfish.ml.model_meta import resolve_imgsz  # noqa: E402

MODEL = config.pipeline_model_path

DEVICE = pick_device()
BATCH = 16
NOT_SPECIES = {"fish", "bait"}  # no prior: the detector's own buckets


def prior_vectors(names: dict[int, str], prior: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup = prior.set_index(["reserve_code", "scientific_name"])["presence_prob"]
    floor = float(prior["presence_prob"].min())  # the floor the prior was built with
    out = {}
    for res in prior["reserve_code"].unique():
        out[res] = np.array(
            [
                1.0 if names[i] in NOT_SPECIES else lookup.get((res, names[i]), floor)
                for i in range(len(names))
            ]
        )
    return out


def detect(raw: torch.Tensor, prior_vec: np.ndarray | None, conf: float, iou: float):
    """raw: (4 + nc, N). Returns (boxes xyxy, orig_cls, orig_score, new_cls, new_score) on kept anchors."""
    boxes, scores = raw[:4].T, raw[4:].T  # (N, 4) xywh, (N, nc)
    top, _ = scores.max(dim=1)
    keep = top >= conf
    boxes, scores, top = boxes[keep], scores[keep], top[keep]
    xyxy = torch.cat(
        [boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=1
    )
    kept = torchvision.ops.nms(xyxy.float().cpu(), top.float().cpu(), iou)
    xyxy, scores, top = (
        xyxy[kept.to(xyxy.device)],
        scores[kept.to(scores.device)],
        top[kept.to(top.device)],
    )
    orig_cls = scores.argmax(dim=1)
    if prior_vec is None:
        new = scores
    else:
        new = scores * torch.as_tensor(
            prior_vec, dtype=scores.dtype, device=scores.device
        )
    new_cls = new.argmax(dim=1)
    return (
        xyxy.cpu().numpy(),
        orig_cls.cpu().numpy(),
        top.cpu().numpy(),
        new_cls.cpu().numpy(),
        new.max(dim=1).values.cpu().numpy(),
    )


def run_inference() -> pd.DataFrame:
    boxes_df = load_boxes()
    frames = boxes_df.drop_duplicates("filename")[
        ["filename", "frame_path", "reserve"]
    ].reset_index(drop=True)
    prior = load_site_prior()
    model = YOLO(str(MODEL))
    names = model.names
    net = model.model.to(DEVICE).eval()
    vecs = prior_vectors(names, prior)
    imgsz = resolve_imgsz(MODEL, model=model)
    letterbox = LetterBox((imgsz, imgsz), auto=False)
    conf, iou = config.confidence_threshold, config.ml_nms_iou

    rows = []
    for start in range(0, len(frames), BATCH):
        chunk = frames.iloc[start : start + BATCH]
        imgs = [cv2.imread(str(p)) for p in chunk["frame_path"]]
        lbs = [letterbox(image=im) for im in imgs]
        x = (
            torch.from_numpy(np.stack([lb[..., ::-1].copy() for lb in lbs]))
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .to(DEVICE)
        )
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16):
            out = net(x)
        raw = (out[0] if isinstance(out, (list, tuple)) else out).float()
        for i, r in enumerate(chunk.itertuples()):
            vec = vecs.get(r.reserve)
            xyxy, oc, os_, nc, ns = detect(raw[i], vec, conf, iou)
            if len(xyxy) == 0:
                continue
            # undo letterbox: scale and pad
            h0, w0 = imgs[i].shape[:2]
            gain = min(imgsz / h0, imgsz / w0)
            pad_w, pad_h = (imgsz - w0 * gain) / 2, (imgsz - h0 * gain) / 2
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / gain
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / gain
            for b, a, s1, c, s2 in zip(xyxy, oc, os_, nc, ns):
                rows.append(
                    (
                        r.filename,
                        r.reserve,
                        names[int(a)],
                        float(s1),
                        names[int(c)],
                        float(s2),
                        *[float(v) for v in b],
                    )
                )
        print(f"{min(start + BATCH, len(frames))}/{len(frames)} frames", flush=True)
    pred = pd.DataFrame(
        rows,
        columns=[
            "filename",
            "reserve",
            "yolo_class",
            "conf",
            "prior_class",
            "prior_score",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
    )
    pred.to_csv(OUT_DIR / "yolo_prior_predictions.csv", index=False)
    return pred


def main() -> None:
    cached = OUT_DIR / "yolo_prior_predictions.csv"
    if cached.exists() and "--rerun" not in sys.argv:
        print(f"using cached {cached} (pass --rerun to run inference again)")
        pred = pd.read_csv(cached)
    else:
        pred = run_inference()

    # ── score both labelings on expert-matched boxes, same classes as evaluate.py ──
    meta = read_crops_meta()
    target, classes, _stats = classes_for(meta)
    y = meta["species"].where(meta["species"].isin(target), OTHER).to_numpy()
    common = species_registry().common_name_map()

    def mapped(col):
        frame = (
            pred
            if col == "yolo_class"
            else pred.drop(columns=["yolo_class"]).rename(columns={col: "yolo_class"})
        )
        m = match_yolo(meta, frame)
        lab = m["yolo_label"].where(m["yolo_label"].isin(target), OTHER)
        return (
            lab.where(m["yolo_label"] != "fish", OTHER),
            m["yolo_label"].notna().to_numpy(),
        )

    orig, has = mapped("yolo_class")
    with_prior, _ = mapped("prior_class")
    changed = pred["yolo_class"] != pred["prior_class"]
    t_orig = per_class_table(y[has], orig[has].to_numpy(), classes)
    t_prior = per_class_table(y[has], with_prior[has].to_numpy(), classes)
    flips = pd.DataFrame(
        {
            "expert": y[has],
            "before": orig[has].to_numpy(),
            "after": with_prior[has].to_numpy(),
        }
    )
    flips = flips[flips["before"] != flips["after"]]
    fixed = (flips["after"] == flips["expert"]).sum()
    broken = (flips["before"] == flips["expert"]).sum()

    def fmt(t):
        return (
            t.assign(common=[common.get(c, "") for c in t.index]).round(3).to_markdown()
        )

    lines = [
        "# Current YOLO with the site prior on its class scores",
        "",
        f"{len(pred)} detected boxes on {pred['filename'].nunique()} frames; the prior renamed {int(changed.sum())} of them "
        f"({changed.mean():.1%}). Detection itself is unchanged: the prior only re-picks the class from YOLO's 22 scores. "
        f"On expert-matched boxes it fixed {fixed} labels and broke {broken}. UoA volumes have no reserve, so no prior there.",
        "",
        f"## YOLO as is: accuracy {(orig[has] == y[has]).mean():.3f}, macro F1 {t_orig['f1'].mean():.3f}",
        "",
        fmt(t_orig),
        "",
        f"## YOLO + site prior: accuracy {(with_prior[has] == y[has]).mean():.3f}, macro F1 {t_prior['f1'].mean():.3f}",
        "",
        fmt(t_prior),
        "",
        "## Renamed boxes by reserve and species (all detections, not only expert-matched)",
        "",
        pred[changed]
        .groupby(["reserve", "yolo_class", "prior_class"])
        .size()
        .rename("boxes")
        .reset_index()
        .sort_values("boxes", ascending=False)
        .head(20)
        .to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "yolo_prior_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
