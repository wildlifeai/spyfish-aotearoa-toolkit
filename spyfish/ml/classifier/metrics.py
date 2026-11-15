"""Box matching and per-class scoring shared by the two-stage scripts.

* ``match_yolo``: pair detector boxes with expert boxes on a frame, greedy by
  IoU, so a classifier and the detector can be scored on the same crops.
  ``iou`` / ``best_iou`` are the ones in ``extraction.render_annotations``,
  re-exported here for the scripts that import them from this module.
* ``score_predictions`` / ``recall_table``: detector recall per species (did
  it find the fish at all, whatever it called it).
* ``load_site_prior`` / ``apply_prior``: read the prior table, multiply class probabilities by the site prior
  (``scripts/build_site_prior.py``) at the crop's reserve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from spyfish.config.wrapper import config
from spyfish.extraction.render_annotations import best_iou, iou  # noqa: F401
from spyfish.ml.classifier.classes import OTHER
from spyfish.ml.classifier.crops import NOT_FISH

IOU_MATCH = config.classifier_training_iou_match
PRIOR_ALPHA = config.classifier_training_prior_alpha
PRIOR_CSV = config.site_prior_csv_path


def match_yolo(
    meta: pd.DataFrame, yolo: pd.DataFrame, iou_match: float = IOU_MATCH
) -> pd.DataFrame:
    """Greedy IoU match of YOLO boxes to expert crops, per frame.

    Returns ``yolo_label`` / ``yolo_conf`` aligned with ``meta``; unmatched
    crops get NaN."""
    out = pd.Series(index=meta.index, dtype=object)
    conf = pd.Series(index=meta.index, dtype=float)
    for fname, g in meta.groupby("filename"):
        preds = yolo[yolo["filename"] == fname]
        if preds.empty:
            continue
        pboxes = preds[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        pairs = []
        for i, r in g.iterrows():
            eb = (r.x_min, r.y_min, r.x_max, r.y_max)
            for j, pb in enumerate(pboxes):
                v = iou(eb, pb)
                if v >= iou_match:
                    pairs.append((v, i, j))
        used_e, used_p = set(), set()
        for v, i, j in sorted(pairs, reverse=True):
            if i in used_e or j in used_p:
                continue
            used_e.add(i)
            used_p.add(j)
            out[i] = preds.iloc[j]["yolo_class"]
            conf[i] = preds.iloc[j]["conf"]
    return pd.DataFrame({"yolo_label": out, "yolo_conf": conf})


def per_class_table(y_true, y_pred, classes, labels=None) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, indexed by class name.

    ``y_true`` / ``y_pred`` hold class names by default. A caller whose labels
    are integer class ids (the fine-tune) passes ``labels=range(len(classes))``;
    the table is still indexed by name.
    """
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes if labels is None else list(labels),
        zero_division=0,
    )
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "n": s}, index=classes)


def load_site_prior(path=None) -> pd.DataFrame:
    """The site prior table ``scripts/build_site_prior.py`` writes
    (``reserve_code, scientific_name, presence_prob, ...``), from
    ``PRIOR_CSV`` unless another path is given."""
    return pd.read_csv(path if path is not None else PRIOR_CSV)


def apply_prior(
    proba: np.ndarray,
    classes: list[str],
    reserves: pd.Series,
    prior: pd.DataFrame,
    alpha: float,
) -> np.ndarray:
    """Multiply class probabilities by ``presence_prob`` at the crop's reserve
    and renormalise. ``OTHER`` and ``NOT_FISH`` keep probability 1 (no prior).
    Reserves absent from the prior (UOA) are left untouched."""
    lookup = prior.set_index(["reserve_code", "scientific_name"])["presence_prob"]
    floor = float(prior["presence_prob"].min())  # the floor the prior was built with
    known = set(prior["reserve_code"])
    out = proba.copy()
    for res in reserves.unique():
        if res not in known:
            continue
        rows = (reserves == res).to_numpy()
        w = np.array(
            [
                1.0 if c in (OTHER, NOT_FISH) else lookup.get((res, c), floor)
                for c in classes
            ]
        )
        out[rows] = out[rows] * (w**alpha)
    return out / out.sum(axis=1, keepdims=True)


def score_predictions(
    pred: pd.DataFrame,
    experts: pd.DataFrame,
    conf: float | None = None,
    iou_match: float = IOU_MATCH,
) -> tuple[pd.DataFrame, dict]:
    """Greedy IoU match of detector boxes to expert boxes, per frame.

    Returns the expert table with a boolean ``found`` column and a summary
    ``{"n_pred", "matched_pred", "precision", "recall"}``. ``conf`` drops
    detector boxes below that confidence first, so one predictions file scored
    at the inference floor can be re-scored at any higher threshold. Only
    frames the experts annotated count towards precision: a box on a frame
    nobody looked at is neither right nor wrong here. Boxes the detector calls
    ``bait`` are left out: experts rarely box bait.
    """
    if conf is not None:
        pred = pred[pred["conf"] >= conf]
    pred = pred[pred["yolo_class"] != "bait"]
    experts = experts.copy()
    found = pd.Series(False, index=experts.index)
    matched_pred = n_pred = 0
    by_frame = {f: g for f, g in pred.groupby("filename")}
    for fname, g in experts.groupby("filename"):
        p = by_frame.get(fname)
        if p is None:
            continue
        n_pred += len(p)
        pb = p[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        used = set()
        for i, r in g.iterrows():
            eb = (r.x_min, r.y_min, r.x_max, r.y_max)
            best, best_j = 0.0, None
            for j, b in enumerate(pb):
                if j in used:
                    continue
                v = iou(eb, b)
                if v > best:
                    best, best_j = v, j
            if best >= iou_match:
                found[i] = True
                used.add(best_j)
        matched_pred += len(used)
    experts["found"] = found
    return experts, {
        "n_pred": n_pred,
        "matched_pred": matched_pred,
        "precision": matched_pred / max(n_pred, 1),
        "recall": float(found.mean()) if len(found) else 0.0,
    }


def recall_table(experts: pd.DataFrame, common: dict) -> pd.DataFrame:
    """Per-species expert boxes, how many were found, and the share."""
    t = (
        experts.groupby("species")["found"]
        .agg(expert_boxes="size", found="sum", recall="mean")
        .sort_values("expert_boxes", ascending=False)
    )
    t["common"] = [common.get(s, "") for s in t.index]
    return t
