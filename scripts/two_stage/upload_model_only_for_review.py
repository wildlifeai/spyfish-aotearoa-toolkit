"""Put model-only boxes back into BIIGLE as "To review", for an expert pass.

A model-only box is a detector box with IoU < 0.3 to every expert box on an
expert-annotated frame (`audit_dataset.py` writes them to
``model_only_<predictions>.csv``). On dense frames most are fish nobody boxed
(about 95% on volume 30505, 2026-11-05). Uploading them labelled "To review"
(workflow tree, ``biigle.workflow_label_tree_id``) lets the expert decide each:

* relabel to a species: a fish that was missing, it trains from the next pull;
* delete it: junk, and the manifest written here turns it into a reviewed
  `not a fish` negative.

"To review" is a workflow state, not a sighting: training and the annotations
database both drop it (``biigle.workflow_tree_keep_labels`` leaves it out), so a
box nobody has reviewed yet changes nothing.

Choose what goes up with one of:

    --volume-id 30505                      one BIIGLE training volume (drop volume_30505)
    --top 25 --splits train                the N drops with the most boxes
    --drops TON_20211026_BUV_TON_053_01 …  named drops
    --survey BNP_20250118_BUV              every drop of one survey

A pipeline drop's frames can sit in its per-drop review volume (``{drop_id}``),
the survey review volume (``{survey_id} ML``) or the survey training volume
(``{survey_id}``), in the in-progress, done or training project, plus the volume ids the
pipeline database records for the drop. The script lists both
projects' volumes (two calls), fetches each candidate volume's filenames (one
call each) and places every frame; a frame it cannot place stops the run before
anything is sent. Then one call per 100 boxes per volume.

Dry run by default: prints what would be sent and writes the manifest. Only
``--upload`` writes to BIIGLE. Run on NeSI after ``audit_dataset.py``.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR, frames_on_disk  # noqa: E402

# Training skips this label (biigle_to_yolo._WORKFLOW_LABEL_SKIP); create it in
# tree 3375 before the first upload. "To review" was used for the first rounds
# (2026-11-05/06) but trains as `fish`, so those boxes must all be relabelled or
# deleted before a retrain.
REVIEW_LABEL = "ML: check"
RECTANGLE = 5  # BIIGLE shape id
COLUMNS = [
    "drop",
    "split",
    "filename",
    "model_class",
    "conf",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


def choose(boxes: pd.DataFrame, args) -> pd.DataFrame:
    if args.volume_id:
        return boxes[boxes["drop"] == f"volume_{args.volume_id}"]
    if args.splits:
        boxes = boxes[boxes["split"].isin(args.splits)]
    boxes = boxes[~boxes["drop"].isin(args.exclude)]
    if args.drops:
        return boxes[boxes["drop"].isin(args.drops)]
    if args.survey:
        return boxes[boxes["drop"].str.startswith(f"{args.survey}_")]
    top = boxes.groupby("drop").size().sort_values(ascending=False).head(args.top).index
    return boxes[boxes["drop"].isin(top)]


def likely_fish(boxes: pd.DataFrame) -> pd.DataFrame:
    """Drop the two kinds of model-only box that review showed are not new fish.

    * inside: >= 60% of the box lies within one expert box, i.e. a fin, tail or
      second box on a fish already boxed (TUH_034: 11 of 12);
    * fixed: the same spot (centre within 3% of the frame) on >= 3 frames of a
      drop, i.e. an object that does not move (BNP_162's edge object).
    Boxes at the frame edge are kept: experts box edge fish too (26% of their
    boxes), and an edge rule threw away real faint cod (TON_003).
    """
    from spyfish.ml.classifier.crops import load_boxes

    experts = load_boxes()
    by_frame = {f: g for f, g in experts.groupby("filename")}
    sizes: dict[str, tuple[int, int]] = {}
    frames = frames_on_disk()

    def inside(r) -> bool:
        g = by_frame.get(r.filename)
        if g is None:
            return False
        iw = (g["x_max"].clip(upper=r.x_max) - g["x_min"].clip(lower=r.x_min)).clip(
            lower=0
        )
        ih = (g["y_max"].clip(upper=r.y_max) - g["y_min"].clip(lower=r.y_min)).clip(
            lower=0
        )
        area = max((r.x_max - r.x_min) * (r.y_max - r.y_min), 1.0)
        return bool(((iw * ih) / area).max() >= 0.6)

    def size(f):
        if f not in sizes:
            with Image.open(frames[f]) as im:
                sizes[f] = im.size
        return sizes[f]

    b = boxes.copy()
    b["inside"] = [inside(r) for r in b.itertuples()]
    wh = [size(f) for f in b["filename"]]
    b["cx"] = (b["x_min"] + b["x_max"]) / 2 / [w for w, _ in wh]
    b["cy"] = (b["y_min"] + b["y_max"]) / 2 / [h for _, h in wh]
    fixed = pd.Series(False, index=b.index)
    for _, g in b.groupby("drop"):
        for r in g.itertuples():
            near = ((g["cx"] - r.cx).abs() < 0.03) & ((g["cy"] - r.cy).abs() < 0.03)
            fixed[r.Index] = g.loc[near, "filename"].nunique() >= 3
    b["fixed"] = fixed
    keep = ~(b["inside"] | b["fixed"])
    print(
        f"--likely-fish: kept {int(keep.sum())} of {len(b)} (inside an expert box {int(b['inside'].sum())}, fixed spot {int(b['fixed'].sum())})"
    )
    return boxes[keep]


def clamp(boxes: pd.DataFrame) -> pd.DataFrame:
    """Clip to the frame: BIIGLE rejects a rectangle outside the image, and one
    bad box fails its whole batch of 100."""
    frames = frames_on_disk()
    missing = sorted(set(boxes["filename"]) - set(frames))
    if missing:
        raise SystemExit(f"{len(missing)} frames not on disk, e.g. {missing[:3]}")
    rows = []
    for fname, g in boxes.groupby("filename"):
        with Image.open(frames[fname]) as im:
            W, H = im.size
        for r in g.itertuples():
            x1, y1 = max(0.0, r.x_min), max(0.0, r.y_min)
            x2, y2 = min(float(W), r.x_max), min(float(H), r.y_max)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                rows.append(
                    (r.drop, r.split, fname, r.yolo_class, r.conf, x1, y1, x2, y2)
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def locate(handler, sent: pd.DataFrame, volume_id) -> dict:
    """filename -> (volume_id, image_id); SystemExit naming what cannot be placed."""
    if volume_id:
        candidates = [volume_id]
    else:
        names = set()
        for drop in sent["drop"].unique():
            survey = config.get_survey_id_from_drop(drop)
            names |= {
                config.expert_drop_volume_name_template.format(drop_id=drop),
                config.expert_survey_volume_name_template.format(survey_id=survey),
                config.training_survey_volume_name_template.format(survey_id=survey),
            }
        candidates = []
        # in-progress, done (finished reviews move there) and training projects
        for project in {
            config.biigle_project_id,
            config.biigle_done_project_id,
            config.biigle_training_project_id,
        }:
            candidates += [
                int(v["id"])
                for v in handler.get_volumes(project)
                if v.get("name") in names
            ]
        # the database records each drop's own volumes, whatever they were named
        from spyfish.database.manager import DatabaseManager

        deployments = DatabaseManager().get_all_deployments_map()
        for drop in sent["drop"].unique():
            row = deployments.get(drop) or {}
            for col in ("biigle_volume_id", "training_biigle_volume_id"):
                if row.get(col):
                    candidates.append(int(row[col]))
        candidates = sorted(set(candidates))
        logging.info(
            f"{len(candidates)} candidate volumes for {sent['drop'].nunique()} drops: {candidates}"
        )
    wanted, where = set(sent["filename"]), {}
    for vid in candidates:
        for im in handler.get_volume_images(vid):
            if im["filename"] in wanted:
                where.setdefault(im["filename"], (vid, int(im["id"])))
    absent = sorted(wanted - set(where))
    if absent:
        raise SystemExit(
            f"{len(absent)} frames are in none of the candidate volumes, e.g. {absent[:3]}; nothing sent"
        )
    return where


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--volume-id", type=int, help="one BIIGLE training volume")
    pick.add_argument("--top", type=int, help="the N drops with the most boxes")
    pick.add_argument("--drops", nargs="+", help="these drops")
    pick.add_argument(
        "--survey", help="every drop of this survey, e.g. BNP_20250118_BUV"
    )
    ap.add_argument(
        "--splits",
        nargs="*",
        default=[],
        help="only drops in these splits (train, val, test)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=["volume_30505"],
        help="drops to leave out (already reviewed)",
    )
    ap.add_argument(
        "--csv",
        default="model_only_detector_predictions.csv",
        help="audit output under wip_two_stage",
    )
    ap.add_argument(
        "--min-conf", type=float, default=0.4, help="skip fainter boxes, mostly seabed"
    )
    ap.add_argument(
        "--likely-fish",
        action="store_true",
        help="drop boxes inside an expert box or on a fixed spot (see likely_fish)",
    )
    ap.add_argument(
        "--label",
        default=REVIEW_LABEL,
        help="workflow-tree label for the boxes; must not be in biigle.workflow_tree_keep_labels",
    )
    ap.add_argument(
        "--manifest", default=None, help="manifest name under wip_two_stage"
    )
    ap.add_argument(
        "--upload", action="store_true", help="write to BIIGLE (default: dry run)"
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    boxes = pd.read_csv(OUT_DIR / args.csv)
    boxes = choose(boxes[boxes["conf"] >= args.min_conf], args)
    if boxes.empty:
        raise SystemExit(f"nothing selected at conf >= {args.min_conf} from {args.csv}")
    if args.likely_fish:
        boxes = likely_fish(boxes)
    sent = clamp(boxes)
    if args.volume_id:
        tag = f"volume_{args.volume_id}"
    else:
        tag = (f"top{args.top}" if args.top else args.survey or "drops") + "".join(
            f"_{s}" for s in args.splits
        )
    # one manifest per upload, never overwritten: it is the only record of which
    # boxes were sent, so a later upload must not replace an earlier one
    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = OUT_DIR / (
        args.manifest or f"{artifacts.REVIEW_UPLOAD_PREFIX}{tag}_{stamp}.csv"
    )
    if args.upload and manifest.exists():
        raise SystemExit(f"{manifest} exists; pick another --manifest name")
    per_drop = sent.groupby(["drop", "split"]).agg(
        boxes=("conf", "size"), frames=("filename", "nunique")
    )
    print(per_drop.sort_values("boxes", ascending=False).to_string())
    print(
        f"\n{len(sent)} boxes on {sent['filename'].nunique()} frames in {sent['drop'].nunique()} drops at conf >= {args.min_conf}"
    )

    if not args.upload:
        sent.to_csv(manifest, index=False)
        print(f"dry run: nothing sent. Manifest of what would go: {manifest}")
        return

    from spyfish.biigle.biigle_handler import BiigleHandler

    handler = BiigleHandler()
    tree = config.biigle_workflow_label_tree_id
    label_ids = [
        int(lbl["id"])
        for lbl in handler.get_label_tree_labels(tree)
        if lbl["name"] == args.label
    ]
    if len(label_ids) != 1:
        raise SystemExit(
            f"expected one '{args.label}' label in tree {tree}, found {label_ids}"
        )
    where = locate(handler, sent, args.volume_id)
    sent["volume_id"] = [where[f][0] for f in sent["filename"]]
    sent["image_id"] = [where[f][1] for f in sent["filename"]]
    sent["label_id"] = label_ids[0]
    sent["annotation_id"] = None
    sent.to_csv(
        manifest, index=False
    )  # before sending: a failed batch still leaves the plan on disk
    for vid, g in sent.groupby("volume_id"):
        payload = [
            {
                "image_id": int(r.image_id),
                "shape": RECTANGLE,  # BIIGLE renamed shape_id -> shape (bulk endpoint validation, 2026)
                "shape_id": RECTANGLE,  # older servers
                "points": [
                    r.x_min,
                    r.y_min,
                    r.x_max,
                    r.y_min,
                    r.x_max,
                    r.y_max,
                    r.x_min,
                    r.y_max,
                ],
                "label_id": int(r.label_id),
                "confidence": float(r.conf),
            }
            for r in g.itertuples()
        ]
        details = handler.upload_image_annotations(int(vid), payload).get("details", [])
        if len(details) != len(payload):
            raise SystemExit(
                f"volume {vid}: BIIGLE returned {len(details)} annotations for {len(payload)}; manifest {manifest}"
            )
        sent.loc[g.index, "annotation_id"] = [d.get("id") for d in details]
        sent.to_csv(manifest, index=False)
        print(f"volume {vid}: {len(payload)} '{args.label}' boxes uploaded")
    print(
        f"done: {len(sent)} boxes in {sent['volume_id'].nunique()} volumes; manifest {manifest}"
    )


if __name__ == "__main__":
    main()
