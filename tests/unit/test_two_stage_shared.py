"""The helpers the two-stage scripts share (duplication cleanup, 2026-11-11):
one IoU, one probe recipe, one crop embedder, one font loader, one way to find
BIIGLE raw CSVs, and the box-loader tail. Each replaced several hand copies, so
the tests pin the behaviour the copies had.
"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spyfish.config.wrapper import config
from spyfish.extraction.render_annotations import best_iou, center_to_corners, iou
from spyfish.ml.classifier import artifacts
from spyfish.ml.classifier import classes as cl
from spyfish.ml.classifier import crops, metrics, site_prior
from spyfish.ml.classifier.sheets import load_font

REPO = Path(__file__).resolve().parents[2]
DROP = "KSF_20240124_BUV_KSF_085_01"

# ── IoU ──────────────────────────────────────────────────────────────────────


def _old_corner_iou(a, b):
    """The copy that lived in classifier/metrics.py."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _old_centre_iou(a, b):
    """The copy that lived in scripts/confidence_threshold_experiment.py."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def test_iou_equals_the_copies_it_replaced():
    rng = random.Random(7)
    for _ in range(2000):
        x, y = rng.uniform(0, 50), rng.uniform(0, 50)
        a = (x, y, x + rng.uniform(0, 40), y + rng.uniform(0, 40))
        x, y = rng.uniform(0, 50), rng.uniform(0, 50)
        b = (x, y, x + rng.uniform(0, 40), y + rng.uniform(0, 40))
        assert iou(a, b) == pytest.approx(_old_corner_iou(a, b), abs=1e-12)
    for _ in range(2000):
        a = tuple(rng.uniform(1, 60) for _ in range(4))
        b = tuple(rng.uniform(1, 60) for _ in range(4))
        new = iou(center_to_corners(*a), center_to_corners(*b))
        assert new == pytest.approx(_old_centre_iou(a, b), abs=1e-9)
    # touching, contained, identical, zero-area, and numpy rows
    edge = [
        ((0, 0, 10, 10), (10, 0, 20, 10)),
        ((0, 0, 10, 10), (2, 2, 4, 4)),
        ((0, 0, 10, 10), (0, 0, 10, 10)),
        ((5, 5, 5, 5), (5, 5, 5, 5)),
        ((0, 0, 10, 10), (5, 5, 5, 9)),
    ]
    for a, b in edge:
        assert iou(a, b) == pytest.approx(_old_corner_iou(a, b), abs=1e-12)
        assert iou(np.array(a, float), np.array(b, float)) == pytest.approx(
            _old_corner_iou(a, b), abs=1e-12
        )
    assert metrics.iou is iou  # the classifier's name for it is the same function


def test_best_iou_is_the_max_and_zero_for_nothing():
    box = (0, 0, 10, 10)
    others = np.array([[20, 20, 30, 30], [0, 0, 10, 5], [0, 0, 10, 10]], float)
    assert best_iou(box, others) == 1.0
    assert best_iou(box, others[:2]) == pytest.approx(0.5)
    assert best_iou(box, []) == 0.0
    assert best_iou(box, np.empty((0, 4))) == 0.0


# ── metrics ──────────────────────────────────────────────────────────────────


def test_per_class_table_takes_names_or_integer_ids():
    classes = ["Parapercis colias", "Chrysophrys auratus", "other fish"]
    y_true = np.array([0, 0, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])
    by_id = metrics.per_class_table(y_true, y_pred, classes, labels=range(3))
    by_name = metrics.per_class_table(
        np.array(classes)[y_true], np.array(classes)[y_pred], classes
    )
    pd.testing.assert_frame_equal(by_id, by_name)
    assert list(by_id.index) == classes
    assert by_id.loc["Parapercis colias", "recall"] == 0.5
    assert by_id.loc["other fish", "n"] == 3


def test_load_site_prior_reads_the_default_or_a_given_path(tmp_path, monkeypatch):
    table = pd.DataFrame(
        {
            "reserve_code": ["KSF"],
            "scientific_name": ["Parapercis colias"],
            "presence_prob": [0.8],
        }
    )
    default, other = tmp_path / "prior.csv", tmp_path / "other.csv"
    table.to_csv(default, index=False)
    table.assign(presence_prob=0.1).to_csv(other, index=False)
    monkeypatch.setattr(metrics, "PRIOR_CSV", default)
    assert metrics.load_site_prior()["presence_prob"].tolist() == [0.8]
    assert metrics.load_site_prior(other)["presence_prob"].tolist() == [0.1]


# ── the probe ────────────────────────────────────────────────────────────────


def _blobs(n_per_class=40, classes=("a", "b", "c")):
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.normal(loc=3 * i, scale=1.0, size=(n_per_class, 4)) for i in range(3)]
    )
    y = np.repeat(np.array(classes), n_per_class)
    return X, y


def test_probe_recipe_comes_from_config(monkeypatch):
    recipe = config.classifier_probe
    assert recipe.max_iter > 0 and recipe.c > 0
    monkeypatch.setattr(
        type(config),
        "_ct",
        lambda self, key: {"max_iter": 10} if key == "probe" else {},
    )
    with pytest.raises(KeyError):  # a missing key fails loudly, no default
        config.classifier_probe


def test_fit_probe_matches_the_hand_written_scaler_plus_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = _blobs()
    recipe = config.classifier_probe
    scaler = StandardScaler().fit(X)
    old = LogisticRegression(
        max_iter=recipe.max_iter, C=recipe.c, class_weight="balanced"
    ).fit(scaler.transform(X), y)
    probe = cl.fit_probe(X, y)
    assert list(probe.classes_) == list(old.classes_)
    np.testing.assert_allclose(
        probe.predict_proba(X), old.predict_proba(scaler.transform(X)), atol=1e-12
    )
    assert (probe.predict(X) == old.predict(scaler.transform(X))).all()


def test_probe_probabilities_follow_class_order_and_zero_fill_unseen():
    X, y = _blobs()
    probe = cl.fit_probe(X, y)
    classes = ["c", "never seen", "a", "b"]
    full = cl.probe_probabilities(probe, X[:5], classes)
    raw = probe.predict_proba(X[:5])  # columns in probe.classes_ order: a, b, c
    assert full.shape == (5, 4)
    np.testing.assert_array_equal(full[:, 0], raw[:, 2])
    np.testing.assert_array_equal(full[:, 2], raw[:, 0])
    np.testing.assert_array_equal(full[:, 3], raw[:, 1])
    assert (full[:, 1] == 0).all()


# ── reserve of a drop ────────────────────────────────────────────────────────


def test_reserve_helpers_keep_their_own_fallbacks():
    assert crops.reserve_of(DROP) == "KSF"
    assert crops.reserve_of("CRP_20030423_BUV_Liitle-waterfall_01") == "CRP"
    assert crops.reserve_of("volume_30505") == "UOA"
    assert crops.reserve_of("a1b2c3d4-uuid-frame") == crops.NO_RESERVE
    assert site_prior.reserve_code(DROP) == "KSF"
    assert site_prior.reserve_code("CRP_20030423_BUV_Liitle-waterfall_01") == "CRP"
    with pytest.raises(ValueError, match="no reserve prefix"):
        site_prior.reserve_code("volume_30505")


# ── crops.py: lazy names, CSV discovery, loader tail ─────────────────────────


def test_environment_names_resolve_on_first_use(monkeypatch):
    for name in crops._LAZY:
        monkeypatch.delitem(crops.__dict__, name, raising=False)
    monkeypatch.setenv("TWO_STAGE_SOURCE", "not a source")
    with pytest.raises(ValueError, match="TWO_STAGE_SOURCE"):
        crops.SOURCE
    assert "SOURCE" not in crops.__dict__  # a failed resolve caches nothing

    monkeypatch.setenv("TWO_STAGE_SOURCE", "uoa")
    monkeypatch.setenv("TWO_STAGE_UOA_RUN", "no_stills")
    assert crops.SOURCE == "uoa"
    assert crops.OUT_DIR == config.two_stage_dir / "no_stills"
    assert crops.UOA_DETECTOR_SPLIT == crops.OUT_DIR / artifacts.DETECTOR_SPLIT
    assert crops.MAX_PER_DROP_SPECIES is None
    # read once per process: a later change of the environment is not seen
    monkeypatch.setenv("TWO_STAGE_SOURCE", "yolo_dataset")
    assert crops.SOURCE == "uoa"
    with pytest.raises(AttributeError):
        crops.NOT_A_NAME
    # leave nothing resolved under the fake environment for other tests
    for name in crops._LAZY:
        monkeypatch.delitem(crops.__dict__, name, raising=False)


def test_find_biigle_raw_csvs(tmp_path, monkeypatch):
    expert, training = (
        config.biigle_expert_raw_suffix,
        config.biigle_training_raw_suffix,
    )
    ann = tmp_path / "KSF_20240124_BUV" / DROP / "annotations"
    deep = tmp_path / "curated" / "a" / "volume_30505" / "annotations"
    for d in (ann, deep):
        d.mkdir(parents=True)
    for p in (
        ann / f"{DROP}{expert}",
        ann / f"{DROP}{training}",
        ann / f"legacy_video_{DROP}{expert}",
        ann / f"{DROP}_zooniverse_maxn.csv",
        deep / f"volume_30505{expert}",
    ):
        p.write_text("filename\n")
    monkeypatch.setattr(crops, "DEPLOYMENT_DATA", tmp_path)

    assert crops.find_biigle_raw_csvs() == [ann / f"{DROP}{expert}"]
    assert crops.find_expert_csvs() == crops.find_biigle_raw_csvs()
    both = crops.find_biigle_raw_csvs((expert, training), any_depth=True)
    assert both == sorted(
        [
            ann / f"{DROP}{expert}",
            ann / f"{DROP}{training}",
            deep / f"volume_30505{expert}",
        ]
    )


def test_finish_boxes_from_points_and_from_corners():
    m = crops.MIN_BOX_PX
    big = f"[10,20,{10 + 3 * m},20,{10 + 3 * m},{20 + 2 * m},10,{20 + 2 * m}]"
    thin = f"[0,0,{m - 1},0,{m - 1},{5 * m},0,{5 * m}]"
    from_points = crops._finish_boxes(
        pd.DataFrame(
            {
                "filename": ["a.jpg", "b.jpg"],
                "label_name": ["Blue cod", "Blue cod"],
                "points": [big, thin],
            },
            index=[7, 9],
        )
    )
    assert list(from_points.index) == [0]
    assert "points" not in from_points and "label_name" not in from_points
    row = from_points.iloc[0]
    assert (row.x_min, row.y_min, row.x_max, row.y_max) == (
        10,
        20,
        10 + 3 * m,
        20 + 2 * m,
    )
    assert (row.w, row.h) == (3 * m, 2 * m)

    from_corners = crops._finish_boxes(
        pd.DataFrame(
            {
                "x_min": [0.0, 0.0],
                "y_min": [0.0, 0.0],
                "x_max": [float(m), float(m - 1)],  # exactly the minimum is kept
                "y_max": [float(m), float(5 * m)],
            }
        ),
        log_as="test",
    )
    assert from_corners[["w", "h"]].values.tolist() == [[m, m]]


def test_load_yolo_dataset_boxes_reads_labels_through_the_shared_parsers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        type(config), "training_dataset_dir", property(lambda self: tmp_path)
    )
    (tmp_path / "data.yaml").write_text(
        "names:\n  1: Chrysophrys auratus\n  0: Parapercis colias\n"
    )
    for split in ("train", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(
        tmp_path / "images/train" / f"{DROP}__frame_1.jpg"
    )
    Image.new("RGB", (200, 100)).save(tmp_path / "images/test" / f"{DROP}__frame_2.jpg")
    Image.new("RGB", (200, 100)).save(tmp_path / "images/test" / f"{DROP}__frame_3.jpg")
    (tmp_path / "labels/train" / f"{DROP}__frame_1.txt").write_text(
        "1 0.5 0.5 0.5 0.8\n\n0 0.25 0.5 0.3 0.6\n"
    )
    (tmp_path / "labels/test" / f"{DROP}__frame_2.txt").write_text("")  # background

    boxes = crops.load_yolo_dataset_boxes()
    assert boxes["species"].tolist() == ["Chrysophrys auratus", "Parapercis colias"]
    assert boxes["split"].tolist() == ["train", "train"]
    assert boxes["drop"].tolist() == [DROP, DROP]
    assert boxes["reserve"].tolist() == ["KSF", "KSF"]
    first = boxes.iloc[0]
    assert (first.x_min, first.y_min, first.x_max, first.y_max) == pytest.approx(
        (50, 10, 150, 90)
    )

    # a row that is not `class cx cy w h` is an error, never skipped
    (tmp_path / "labels/test" / f"{DROP}__frame_3.txt").write_text("0 0.5 0.5 0.5\n")
    with pytest.raises(ValueError):
        crops.load_yolo_dataset_boxes()


# ── embedding crops ──────────────────────────────────────────────────────────


class _MeanColourEncoder:
    """Stands in for the image encoder: mean R, G, B of each crop plus a one."""

    def __init__(self):
        self.batch_sizes = []

    def encode_image(self, x):
        import torch

        self.batch_sizes.append(len(x))
        return torch.cat([x.mean(dim=(2, 3)), torch.ones(len(x), 1)], dim=1)


def _preprocess(crop):
    import torch

    return torch.from_numpy(
        np.asarray(crop.resize((8, 8)), dtype=np.float32) / 255
    ).permute(2, 0, 1)


def test_embed_batch_normalises_rows():
    model = _MeanColourEncoder()
    tensors = [_preprocess(Image.new("RGB", (20, 20), c)) for c in ("red", "blue")]
    e = crops.embed_batch(model, tensors, "cpu")
    assert e.shape == (2, 4) and e.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-3)
    assert e[0, 0] > e[0, 2] and e[1, 2] > e[1, 0]


def test_embed_crops_keeps_row_order_batches_and_calls_back(tmp_path):
    red, blue = tmp_path / "red.jpg", tmp_path / "blue.jpg"
    Image.new("RGB", (100, 100), "red").save(red)
    Image.new("RGB", (100, 100), "blue").save(blue)
    # frames interleaved, so rows of one frame are not contiguous
    boxes = pd.DataFrame(
        {
            "filename": ["red.jpg", "blue.jpg", "red.jpg", "blue.jpg", "red.jpg"],
            "x_min": [10.0] * 5,
            "y_min": [10.0] * 5,
            "x_max": [60.0] * 5,
            "y_max": [50.0] * 5,
        }
    )
    boxes["frame_path"] = boxes["filename"].map({"red.jpg": red, "blue.jpg": blue})
    model, saved, ticks = _MeanColourEncoder(), [], []
    e = crops.embed_crops(
        boxes,
        "frame_path",
        model,
        _preprocess,
        "cpu",
        save_thumbnail=lambda crop, r: saved.append((r.filename, crop.size)),
        batch_size=2,
        progress=lambda done, total: ticks.append((done, total)),
    )
    assert e.shape == (5, 4)
    is_red = e[:, 0] > e[:, 2]
    assert is_red.tolist() == [True, False, True, False, True]
    assert model.batch_sizes == [2, 2, 1]
    assert ticks == [(2, 5), (4, 5)]
    assert len(saved) == 5 and {s for _, s in saved} == {(60, 48)}  # box + 10% margin

    by_mapping = crops.embed_crops(
        boxes.drop(columns="frame_path"),
        {"red.jpg": red, "blue.jpg": blue},
        _MeanColourEncoder(),
        _preprocess,
        "cpu",
    )
    np.testing.assert_array_equal(by_mapping, e)
    empty = crops.embed_crops(boxes.iloc[:0], "frame_path", model, _preprocess, "cpu")
    assert empty.shape == (0, 0)


# ── sheets ───────────────────────────────────────────────────────────────────


def test_load_font_never_raises_for_a_missing_font_file():
    from PIL import ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (200, 50)))
    for font in (
        load_font(18, candidates=("/no/such/font.ttc",)),
        load_font(18, candidates=()),
        load_font(18),
    ):
        draw.text((2, 2), "Blue cod 0.91", font=font)  # usable, whichever it is


# ── scripts ──────────────────────────────────────────────────────────────────


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts/two_stage" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_detector_dataset_remaps_labels_and_counts(tmp_path, monkeypatch, capsys):
    src = tmp_path / "species"
    monkeypatch.setattr(
        type(config), "local_training_dir", property(lambda self: tmp_path)
    )
    monkeypatch.setattr(
        type(config), "training_dataset_dir", property(lambda self: src)
    )
    (src / "images/train").mkdir(parents=True)
    (src / "labels/train").mkdir(parents=True)
    (src / "data.yaml").write_text(
        "names:\n- Parapercis colias\n- Jasus edwardsii\n- bait\n- Chrysophrys auratus\n"
    )
    for stem in ("a", "b"):
        Image.new("RGB", (8, 8)).save(src / "images/train" / f"{stem}.jpg")
    (src / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n\n3 0.7 0.7 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
    )  # b.jpg has no label file: a background frame

    module = _load_script("make_detector_dataset")
    monkeypatch.setattr("sys.argv", ["make_detector_dataset.py"])
    module.main()

    dst = tmp_path / "detector"
    assert (dst / "labels/train/a.txt").read_text() == (
        "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n0 0.7 0.7 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
    )
    assert (dst / "labels/train/b.txt").read_text() == ""
    assert (dst / "images/train/a.jpg").is_symlink()
    assert (
        "boxes {'fish': 2, 'Jasus edwardsii': 1, 'bait': 1}" in capsys.readouterr().out
    )
    from spyfish.ml.training.yolo_labels import read_class_names

    assert read_class_names(dst / "data.yaml") == ["fish", "Jasus edwardsii", "bait"]

    # an id data.yaml does not name is a corrupt label file
    (src / "labels/train/a.txt").write_text("9 0.5 0.5 0.1 0.1\n")
    with pytest.raises(KeyError):
        module.main()


def test_audit_load_dataset_reads_names_splits_and_boxes(tmp_path):
    (tmp_path / "data.yaml").write_text(
        "names:\n  0: Parapercis colias\n  1: Chrysophrys auratus\n"
    )
    for split in ("train", "val", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "images/train" / f"{DROP}__f1.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "images/test" / "plain.jpg")
    (tmp_path / "labels/train" / f"{DROP}__f1.txt").write_text(
        "1 0.5 0.5 0.2 0.4\n\n0 0.1 0.1 0.1 0.1\n"
    )
    frames, boxes = _load_script("audit_dataset").load_dataset(tmp_path)
    assert frames[["split", "drop", "n_boxes"]].values.tolist() == [
        ["train", DROP, 2],
        ["test", "?", 0],
    ]
    assert boxes["species"].tolist() == ["Chrysophrys auratus", "Parapercis colias"]
    assert boxes.iloc[0][["cx", "cy", "w", "h"]].tolist() == [0.5, 0.5, 0.2, 0.4]


def test_rows_for_drop_is_public():
    """resync_review_volumes.py imports it; it was a private name before."""
    from spyfish.biigle.sync_annotations import rows_for_drop

    other = "KSF_20240124_BUV_KSF_085_02"
    report = pd.DataFrame(
        {
            "filename": [
                f"{DROP}/frames/{DROP}__f1.jpg",
                f"{other}/frames/{other}__f1.jpg",
            ]
        }
    )
    assert len(rows_for_drop(report, DROP)) == 1


def test_tail_parameters_order_is_blocks_then_ln_post_then_proj():
    """A checkpoint's `encoder_tail` is this list, so the order is a contract
    between the fine-tune that saves it and the loader that copies it back."""
    import torch

    from spyfish.ml.classifier.species_classifier import (
        _infer_unfreeze,
        tail_parameters,
    )

    class Visual(torch.nn.Module):
        def __init__(self, with_proj=True):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList(
                torch.nn.Linear(2, 2) for _ in range(4)
            )
            self.ln_post = torch.nn.LayerNorm(2)
            self.proj = torch.nn.Parameter(torch.zeros(2, 2)) if with_proj else None

    visual = Visual()
    tail = tail_parameters(visual, 2)
    blocks = visual.transformer.resblocks
    expected = [
        blocks[2].weight,
        blocks[2].bias,
        blocks[3].weight,
        blocks[3].bias,
        visual.ln_post.weight,
        visual.ln_post.bias,
        visual.proj,
    ]
    assert len(tail) == len(expected)
    assert all(a is b for a, b in zip(tail, expected))
    assert _infer_unfreeze(visual, tail) == 2
    assert len(tail_parameters(Visual(with_proj=False), 1)) == 4
