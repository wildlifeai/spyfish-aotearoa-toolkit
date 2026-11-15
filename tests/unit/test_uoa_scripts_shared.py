"""The UoA one-off scripts after they were pointed at the shared helpers
(2026-11-11): each test pins what the hand-written copy computed.

The two-stage UoA modules read ``TWO_STAGE_SOURCE`` when they are imported and
refuse anything but ``uoa``, and ``spyfish.ml.classifier.classes`` binds the
source at its own import. Importing them here would leave the rest of the test
session in UoA mode, so those checks run once in a child interpreter
(``_DRIVER``) and the tests read its JSON. No GPU, ffmpeg or video anywhere: the
image encoder and the frame drawer are stubs.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from spyfish.config.wrapper import config

REPO = Path(__file__).resolve().parents[2]
DEPLOYMENT = "ESK_B_20220119_PM_M"

_DRIVER = r'''
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

tmp = Path(sys.argv[1])
out = {}

from spyfish.config.wrapper import config
from spyfish.ml.classifier import artifacts, classes as cl, crops
from uoa.scripts.two_stage import shared

out["split_csv_is_the_one_load_boxes_reads"] = (
    shared.DETECTOR_SPLIT_CSV == crops.UOA_DETECTOR_SPLIT
)
out["split_csv_name"] = shared.DETECTOR_SPLIT_CSV.name
out["finetuned_path_tail"] = list(shared.FINETUNED_PATH.parts[-2:])

# ── fit_classifier: bundle and sweep against the hand-written recipe ─────────
from uoa.scripts.two_stage import fit_classifier as fc

rng = np.random.default_rng(0)
species = [crops.UOA_CLASSES[0], crops.UOA_CLASSES[1], "Parika scaber"]
rows = []
for d in range(8):
    for i in range(12):
        rows.append({
            "crop_id": f"{len(rows):06d}",
            "species": species[(d + i) % 3],
            "drop": f"ESK_B_2022011{8 + d % 2}_PM_{'ABCDEFGH'[d]}",
        })
meta = pd.DataFrame(rows)
centre = {s: rng.normal(size=16) for s in species}
X = np.stack([centre[s] + rng.normal(scale=1.5, size=16) for s in meta["species"]])
np.save(tmp / artifacts.EMBEDDINGS, X)

fc.OUT_DIR = tmp
fc.PMIN_SWEEP_CSV = tmp / "pmin_sweep.csv"
fc.PROBE_PATH = tmp / "classifier_probe.joblib"
fc.read_crops_meta = lambda: meta
fc.main()


def old_fit_probe(X, y):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(
        scaler.transform(X), y
    )
    return scaler, clf


target, classes, _ = cl.classes_for(meta)
y = meta["species"].where(meta["species"].isin(target), cl.OTHER).to_numpy()
groups = meta["drop"].to_numpy()
oof = np.zeros((len(meta), len(classes)))
for tr, te in cl.grouped_folds(groups):
    scaler, clf = old_fit_probe(X[tr], y[tr])
    p = clf.predict_proba(scaler.transform(X[te]))
    for ci, c in enumerate(clf.classes_):
        oof[te, classes.index(c)] = p[:, ci]
top = np.array(classes)[oof.argmax(axis=1)]
keep = cl.scored_rows(groups)
common = {}
from spyfish.config.species import species_registry
old_table = fc.sweep(
    y[keep], top[keep], oof.max(axis=1)[keep], classes,
    {sp.scientific_name: sp.common_name
     for sp in species_registry().by_scientific.values()},
)
old_table.to_csv(tmp / "old_sweep.csv", index=False)
out["sweep_identical"] = (
    (tmp / "old_sweep.csv").read_text() == (tmp / "pmin_sweep.csv").read_text()
)

bundle = joblib.load(tmp / "classifier_probe.joblib")
scaler, clf = old_fit_probe(X, y)
out["bundle_keys"] = sorted(bundle)
out["bundle_types"] = [type(bundle["scaler"]).__name__, type(bundle["clf"]).__name__]
out["bundle_classes"] = list(bundle["classes"]) == classes
out["bundle_scaler_diff"] = float(np.abs(bundle["scaler"].mean_ - scaler.mean_).max())
out["bundle_coef_diff"] = float(np.abs(bundle["clf"].coef_ - clf.coef_).max())
out["bundle_proba_diff"] = float(np.abs(
    bundle["clf"].predict_proba(bundle["scaler"].transform(X))
    - clf.predict_proba(scaler.transform(X))
).max())

# ── freeze_rules: the record says what run_day crops and counts with ─────────
from uoa.scripts.two_stage import freeze_rules as fr

out["frozen_literals"] = [fr.UOA_OTHER, fr.INTERVAL_SEC, fr.BAIT_SOAK_SEC]
out["crop_margin_is_the_crop_default"] = (
    fr.CROP_MARGIN == crops.crop_with_margin.__defaults__[0]
    == config.classifier_crop_margin
)
out["run_names"] = [
    fr.run_name_of(Path("runs/uoa_fish_20261110/weights/best.pt")),
    fr.run_name_of(Path("models/cfd-yolov12x-1.00.pt")),
]

# ── raw CSV readers: a rejected box is not a detection ───────────────────────
from uoa.scripts.two_stage import hand_count_frames as hc
from uoa.scripts.two_stage import maxn_compare as mc

raw = tmp / "raw" / "20220119"
raw.mkdir(parents=True)
pd.DataFrame({
    "frame": [0, 0, 20, 20, 20],
    "time_seconds": [0.0, 0.0, 1.0, 1.0, 1.0],
    "class": ["fish", "Scorpis lineolata", "fish", config.classifier_reject_label, "fish"],
    "confidence": [0.9, 0.8, 0.9, 0.9, 0.2],
    "x": 1.0, "y": 1.0, "w": 1.0, "h": 1.0,
}).to_csv(raw / "ESK_B_20220119_PM_M_v01_runA_raw.csv", index=False)
counts = hc.ml_box_counts(raw, 0.4)
out["box_counts"] = {f"{k[0]}|{k[1]}": [[float(t), int(n)] for t, n in v]
                     for k, v in counts.items()}
mc.RAW_DIR = tmp / "raw"
raw_by_key, durations, run_classes = mc.load_run("20220119", "runA")
out["load_run_rows"] = {f"{k[0]}|{k[1]}": len(v) for k, v in raw_by_key.items()}
out["load_run_classes"] = sorted(run_classes)

vol = tmp / "vol" / "ESK_B_20220119_PM_M"
(vol / "annotations").mkdir(parents=True)
pd.DataFrame({"filename": ["ESK_B_20220119_PM_M_v01__t000012.50.jpg"]}).to_csv(
    vol / "annotations" / f"ESK_B_20220119_PM_M{crops.RAW_SUFFIX}", index=False
)
out["existing_frames"] = {f"{k[0]}|{k[1]}": v
                          for k, v in hc.existing_frames([vol], []).items()}

# ── run_day.Classifier: the fine-tuned tail goes back in checkpoint order ────
from uoa.scripts.two_stage import run_day
from spyfish.ml.classifier.species_classifier import tail_parameters


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList(
            torch.nn.Linear(4, 4) for _ in range(3)
        )
        self.ln_post = torch.nn.LayerNorm(4)
        self.proj = torch.nn.Parameter(torch.zeros(4, 3))


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()

    def encode_image(self, x):
        return x.mean(dim=(2, 3))


def old_tail(visual, unfreeze):
    """The list run_day built by hand before tail_parameters."""
    trainable = []
    for blk in visual.transformer.resblocks[-unfreeze:]:
        trainable += list(blk.parameters())
    trainable += list(visual.ln_post.parameters())
    if getattr(visual, "proj", None) is not None:
        trainable.append(visual.proj)
    return trainable


torch.manual_seed(0)
probe_model = Encoder()
new, old = tail_parameters(probe_model.visual, 2), old_tail(probe_model.visual, 2)
out["tail_same_objects"] = len(new) == len(old) and all(a is b for a, b in zip(new, old))

saved = [torch.full_like(p, float(i + 1)) for i, p in enumerate(old)]
head = torch.nn.Linear(3, 2)
with torch.no_grad():
    head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    head.bias.zero_()
torch.save(
    {"classes": ["Scorpis lineolata", "fish"], "unfreeze": 2,
     "encoder_tail": saved, "head": head.state_dict()},
    tmp / "final.pt",
)
fake = Encoder()
first_block_before = fake.visual.transformer.resblocks[0].weight.clone()
colour = {"red": torch.tensor([1.0, 0.0, 0.0]), "green": torch.tensor([0.0, 1.0, 0.0])}


def preprocess(image):
    r, g, b = image.getpixel((0, 0))
    return (torch.tensor([r, g, b]) / 255.0).view(3, 1, 1).expand(3, 2, 2).clone()


run_day.DEVICE = "cpu"
run_day.load_bioclip = lambda device: (fake, preprocess, None)
clf = run_day.Classifier("finetuned", tmp / "final.pt")
loaded = old_tail(fake.visual, 2)
out["tail_loaded_in_order"] = all(
    bool((p == float(i + 1)).all()) for i, p in enumerate(loaded)
)
out["frozen_block_untouched"] = bool(
    (fake.visual.transformer.resblocks[0].weight == first_block_before).all()
)
tops, probs = clf([Image.new("RGB", (4, 4), (255, 0, 0)), Image.new("RGB", (4, 4), (0, 255, 0))])
out["finetuned_tops"] = tops
out["finetuned_probs_ok"] = all(0.5 < p < 1.0 for p in probs)

joblib.dump({"scaler": bundle["scaler"], "clf": bundle["clf"], "classes": classes},
            tmp / "probe.joblib")
probe_clf = run_day.Classifier("probe", tmp / "probe.joblib")
out["probe_loads"] = [type(probe_clf.scaler).__name__, type(probe_clf.clf).__name__,
                      probe_clf.classes == classes]
out["min_frame_coverage"] = run_day.MIN_FRAME_COVERAGE

print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def uoa(tmp_path_factory):
    """What the UoA two-stage modules compute, from one child interpreter."""
    tmp = tmp_path_factory.mktemp("uoa_two_stage")
    env = {**os.environ, "TWO_STAGE_SOURCE": "uoa"}
    env.pop("TWO_STAGE_UOA_RUN", None)
    run = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(tmp)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_shared_paths_use_the_artifact_names(uoa):
    assert uoa["split_csv_is_the_one_load_boxes_reads"]
    assert uoa["split_csv_name"] == "detector_split.csv"
    assert uoa["finetuned_path_tail"] == ["finetune", "final.pt"]


def test_fit_classifier_writes_the_bundle_run_day_loads(uoa):
    """The shared probe is a pipeline; the bundle still holds its two fitted
    steps under the old names, fitted to the same numbers."""
    assert uoa["bundle_keys"] == [
        "classes",
        "clf",
        "deployments",
        "fitted_at",
        "n_crops",
        "scaler",
    ]
    assert uoa["bundle_types"] == ["StandardScaler", "LogisticRegression"]
    assert uoa["bundle_classes"]
    assert uoa["bundle_scaler_diff"] == 0.0
    assert uoa["bundle_coef_diff"] < 1e-9
    assert uoa["bundle_proba_diff"] < 1e-9
    assert uoa["probe_loads"] == ["StandardScaler", "LogisticRegression", True]


def test_fit_classifier_sweep_is_the_hand_written_out_of_fold_sweep(uoa):
    assert uoa["sweep_identical"]


def test_frozen_rules_record_the_values_the_run_uses(uoa):
    assert uoa["frozen_literals"] == ["fish", 360, 360]
    assert uoa["crop_margin_is_the_crop_default"]
    assert uoa["run_names"] == ["uoa_fish_20261110", "cfd-yolov12x-1.00"]
    assert uoa["min_frame_coverage"] == config.ml_min_frame_coverage


def test_two_stage_raw_csv_readers_leave_rejected_boxes_out(uoa):
    key = f"{DEPLOYMENT}|1"
    # second 1.0: one box over the floor, one rejected, one under the floor
    assert uoa["box_counts"] == {key: [[0.0, 2], [1.0, 1]]}
    assert uoa["load_run_rows"] == {key: 4}
    assert uoa["load_run_classes"] == ["Scorpis lineolata", "fish"]
    assert uoa["existing_frames"] == {key: [12.5]}


def test_run_day_loads_the_finetuned_tail_in_checkpoint_order(uoa):
    assert uoa["tail_same_objects"]
    assert uoa["tail_loaded_in_order"]
    assert uoa["frozen_block_untouched"]
    assert uoa["finetuned_tops"] == ["Scorpis lineolata", "fish"]
    assert uoa["finetuned_probs_ok"]


# ── scripts that do not need UoA mode ────────────────────────────────────────


def _points(x1, y1, x2, y2) -> str:
    return json.dumps([x1, y1, x2, y1, x2, y2, x1, y2])


def _old_rehost_iou(a: str, b: str) -> float:
    """The copy that lived in uoa/scripts/rehost_biigle_4510.py."""

    def box(points):
        xy = json.loads(points)
        xs, ys = xy[0::2], xy[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = box(a), box(b)
    w, h = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    inter = max(w, 0) * max(h, 0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def test_rehost_duplicate_box_overlap_equals_the_copy_it_replaced():
    from spyfish.extraction.render_annotations import best_iou
    from spyfish.ml.classifier.crops import rect_from_points

    rng = random.Random(4510)
    cases = [(_points(1, 1, 1, 1), _points(1, 1, 1, 1))]  # zero area: union 0
    for _ in range(2000):
        x, y = rng.uniform(0, 1900), rng.uniform(0, 1000)
        w, h = rng.uniform(1, 400), rng.uniform(1, 400)
        dx, dy = rng.uniform(-300, 300), rng.uniform(-300, 300)
        cases.append(
            (
                _points(round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)),
                _points(
                    round(x + dx, 2),
                    round(y + dy, 2),
                    round(x + dx + w * rng.uniform(0.5, 1.5), 2),
                    round(y + dy + h * rng.uniform(0.5, 1.5), 2),
                ),
            )
        )
    for a, b in cases:
        new = best_iou(rect_from_points(a), [rect_from_points(b)])
        assert new == _old_rehost_iou(a, b)
    a = cases[1][0]
    assert best_iou(rect_from_points(a), []) == 0.0  # no box on the original
    assert best_iou(rect_from_points(a), [rect_from_points(a)]) == 1.0


def _old_iou_cxcywh(a, b):
    """The copy that lived in run_ml_videos_and_upload_to_biigle.py."""
    ax1, ay1, ax2, ay2 = (
        a[0] - a[2] / 2,
        a[1] - a[3] / 2,
        a[0] + a[2] / 2,
        a[1] + a[3] / 2,
    )
    bx1, by1, bx2, by2 = (
        b[0] - b[2] / 2,
        b[1] - b[3] / 2,
        b[0] + b[2] / 2,
        b[1] + b[3] / 2,
    )
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def test_coco_nms_overlap_equals_the_copy_it_replaced():
    """Same number to rounding (the old copy took the areas as w*h, the shared
    one as corner differences), and the same boxes survive the NMS."""
    from uoa.scripts import run_ml_videos_and_upload_to_biigle as script

    rng = random.Random(45)
    rows = []
    for _ in range(3000):
        a = (
            rng.uniform(0, 1920),
            rng.uniform(0, 1080),
            rng.uniform(1, 400),
            rng.uniform(1, 400),
        )
        b = (
            a[0] + rng.uniform(-200, 200),
            a[1] + rng.uniform(-200, 200),
            rng.uniform(1, 400),
            rng.uniform(1, 400),
        )
        assert script._iou_cxcywh(a, b) == pytest.approx(
            _old_iou_cxcywh(a, b), abs=1e-12
        )
        rows.append(a)
    assert script._iou_cxcywh((5, 5, 2, 2), (5, 5, 2, 2)) == 1.0
    assert script._iou_cxcywh((5, 5, 2, 2), (50, 50, 2, 2)) == 0.0

    df = pd.DataFrame(
        {
            "drop_id": f"{DEPLOYMENT}_v01",
            "time_seconds": [1.0, 1.0, 1.0, 2.0],
            "x": [100.0, 104.0, 400.0, 100.0],
            "y": [100.0, 102.0, 400.0, 100.0],
            "w": 50.0,
            "h": 50.0,
            "confidence": [0.6, 0.9, 0.5, 0.4],
        }
    )
    kept = script.nms_coco_dataframe(df, iou_threshold=0.45)
    # the 0.6 box stacks on the 0.9 one; the same box a second later is kept
    assert sorted(kept.index) == [1, 2, 3]


def test_deployment_maxn_frame_counts_what_it_draws(tmp_path, monkeypatch):
    """The peak is counted on the rows the drawer will draw: a frame whose
    extra boxes are all rejected by the classifier is not the MaxN frame."""
    from uoa.scripts import render_deployment_maxn_frames as script

    raw_dir, videos, out_dir = tmp_path / "raw", tmp_path / "videos", tmp_path / "out"
    raw_dir.mkdir()
    videos.mkdir()
    (videos / f"{DEPLOYMENT}_v02.mp4").touch()
    reject = config.classifier_reject_label
    pd.DataFrame(
        {
            # frame 30: two fish. frame 60: one fish and three rejected boxes.
            "frame": [30, 30, 60, 60, 60, 60],
            "time_seconds": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "class": ["fish", "fish", "fish", reject, reject, reject],
            "confidence": 0.9,
            "x": 10.0,
            "y": 10.0,
            "w": 5.0,
            "h": 5.0,
        }
    ).to_csv(raw_dir / f"{DEPLOYMENT}_v02_runA_raw.csv", index=False)

    drawn = {}

    def fake_draw(**kwargs):
        drawn.update(kwargs)
        return [str(out_dir / "frame.jpg")]

    monkeypatch.setattr(script, "draw_boxes_on_video_frames", fake_draw)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_deployment_maxn_frames",
            "--raw-dir",
            str(raw_dir),
            "--videos-dir",
            str(videos),
            "--out-dir",
            str(out_dir),
        ],
    )
    script.main()

    manifest = pd.read_csv(out_dir / "deployment_maxn_manifest_runA.csv")
    row = manifest.iloc[0]
    assert row["deployment_id"] == DEPLOYMENT
    assert (row["maxn_count"], row["peak_frame"], row["peak_time_s"]) == (2, 30, 1.0)
    assert drawn["frame_list"] == [30]
    assert drawn["drop_id"] == f"{DEPLOYMENT}_MAXN"


def test_video_readers_come_from_video_io():
    """The readers moved to spyfish.extraction.video_io; the scripts take them
    from there, not through run_inference's compatibility import."""
    from spyfish.extraction import video_io
    from uoa.scripts import run_ml_videos_and_upload_to_biigle as script

    assert script.ffmpeg_frame_at is video_io.ffmpeg_frame_at
    assert script.get_video_fps is video_io.get_video_fps
    for name in (
        "run_ml_videos_and_upload_to_biigle.py",
        "two_stage/run_day.py",
        "two_stage/hand_count_frames.py",
    ):
        source = (REPO / "uoa" / "scripts" / name).read_text()
        assert "from spyfish.extraction.video_io import" in source
        assert "run_inference import ffmpeg" not in source


def test_build_volume_dataset_writes_the_same_dataset(tmp_path, monkeypatch, capsys):
    """Split names and data.yaml now come from yolo_labels; the dataset on disk
    is what the literals produced."""
    from uoa.scripts import build_volume_dataset as script

    vol = tmp_path / "volume_1"
    for sub in ("frames", "labels", "annotations"):
        (vol / sub).mkdir(parents=True)
    (vol / "annotations" / "class_map.json").write_text(
        json.dumps(
            {
                "0": {"scientific_name": "Scorpis lineolata"},
                "1": {"scientific_name": "Girella tricuspidata"},
            }
        )
    )
    for dep, cls in ((DEPLOYMENT, 0), ("MOT_S_20220118_MD_C", 1)):
        stem = f"{dep}_v01__t000010.00"
        (vol / "frames" / f"{stem}.jpg").touch()
        (vol / "labels" / f"{stem}.txt").write_text(f"{cls} 0.5 0.5 0.1 0.1\n")
    out = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_volume_dataset",
            "--volumes",
            str(vol),
            "--out",
            str(out),
            "--val-groups",
            DEPLOYMENT,
            "--single-class",
            "fish",
        ],
    )
    script.main()
    capsys.readouterr()

    assert (out / "data.yaml").read_text() == "\n".join(
        [
            f"path: {out.resolve()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names: ['fish']",
            "",
        ]
    )
    assert [p.name for p in (out / "images" / "val").iterdir()] == [
        f"{DEPLOYMENT}_v01__t000010.00.jpg"
    ]
    assert (
        out / "labels" / "train" / "MOT_S_20220118_MD_C_v01__t000010.00.txt"
    ).read_text() == "0 0.5 0.5 0.1 0.1"
    assert not (out / "images" / "test").exists()
