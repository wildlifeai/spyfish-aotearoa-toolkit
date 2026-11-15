"""Second-stage species classifier: cropping, thresholds and name mapping.

The model is a fake ``predict_probs``; open_clip is never imported.
"""

import numpy as np
import pytest
from PIL import Image

from spyfish.ml.classifier.species_classifier import (
    Naming,
    SpeciesClassifier,
    crop_box,
)

CLASSES = ["Pagrus auratus", "Parapercis colias", "other fish"]
CANONICAL = {"Pagrus auratus": "Chrysophrys auratus"}


def _probs_for(*rows):
    """A fake model that answers with the given probability rows, in order."""
    queue = [np.asarray(r, dtype=float) for r in rows]

    def predict(crops):
        out = np.stack([queue.pop(0) for _ in crops])
        return out

    predict.queue = queue
    return predict


def _classifier(predict, **over):
    kwargs = dict(
        crop_margin=0.10,
        min_box_px=16,
        batch_size=2,
        p_min=0.5,
        fallback_label="fish",
        label_map={"other fish": "fish"},
        canonical=lambda n: CANONICAL.get(n, n),
    )
    kwargs.update(over)
    return SpeciesClassifier(predict, CLASSES, **kwargs)


def test_labels_resolve_through_label_map_then_registry():
    clf = _classifier(_probs_for())
    assert clf.labels == ["Chrysophrys auratus", "Parapercis colias", "fish"]


def test_name_crops_top_class_fallback_and_batching():
    predict = _probs_for([0.9, 0.05, 0.05], [0.3, 0.4, 0.3], [0.1, 0.1, 0.8])
    clf = _classifier(predict)
    img = Image.new("RGB", (10, 10))
    named = clf.name_crops([img, img, img])
    assert named[0] == Naming("Chrysophrys auratus", "Pagrus auratus", 0.9, True)
    assert named[1] == Naming("Parapercis colias", "Parapercis colias", 0.4, False)
    assert named[2] == Naming("fish", "other fish", 0.8, True)  # mapped, not fallback
    # below p_min the final label is the fallback, or YOLO's when it is sure and allowed
    assert clf.final_label(named[1], "Pagrus auratus", 0.9) == "fish"
    yolo_back = _classifier(
        _probs_for(), fallback_to_yolo=True, fallback_yolo_min_confidence=0.6
    )
    assert yolo_back.final_label(named[1], "Pagrus auratus", 0.9) == "Pagrus auratus"
    assert yolo_back.final_label(named[1], "Pagrus auratus", 0.5) == "fish"
    assert yolo_back.final_label(named[0], "fish", 0.9) == "Chrysophrys auratus"
    assert predict.queue == []  # three crops consumed over batches of 2 and 1


def test_tiny_boxes_keep_their_place_and_are_not_classified():
    predict = _probs_for([0.1, 0.8, 0.1])
    clf = _classifier(predict)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [(0, 50, 50, 10, 40), (0, 50, 50, 40, 40)]  # first is 10 px wide
    out = clf.name_boxes([frame], boxes)
    assert out[0] is None
    assert out[1] == Naming("Parapercis colias", "Parapercis colias", 0.8)


def test_name_boxes_with_nothing_classifiable_never_calls_the_model():
    def boom(crops):
        raise AssertionError("model called")

    clf = _classifier(boom)
    assert clf.name_boxes([np.zeros((20, 20, 3), np.uint8)], [(0, 5, 5, 4, 4)]) == [
        None
    ]


def test_crop_box_adds_margin_clips_to_frame_and_converts_bgr_to_rgb():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure blue in BGR
    crop = crop_box(frame, x=20, y=50, w=40, h=40, margin=0.25)
    # box spans x 0..40, margin 10 each side → clipped at 0, right edge 50
    assert crop.size == (50, 60)  # (width, height): y 20..80 with margin
    assert crop.getpixel((0, 0)) == (0, 0, 255)  # RGB blue, so BGR was converted


def test_p_min_zero_never_falls_back():
    clf = _classifier(_probs_for([0.34, 0.33, 0.33]), p_min=0.0)
    (named,) = clf.name_crops([Image.new("RGB", (8, 8))])
    assert (
        named.confident and clf.final_label(named, "fish", 0.3) == "Chrysophrys auratus"
    )


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_config_rejects_p_min_outside_unit_interval(bad, monkeypatch):
    from spyfish.config.wrapper import config

    monkeypatch.setattr(
        type(config), "_classifier", property(lambda self: {"p_min": bad})
    )
    with pytest.raises(ValueError, match="p_min"):
        config.classifier_p_min


def test_box_hanging_off_the_frame_is_declined_not_crashed():
    """A clipped-to-nothing crop would reach the resize as a 0-px image and
    divide by zero; it must come back as None like any other tiny box."""
    predict = _probs_for([0.1, 0.8, 0.1])
    clf = _classifier(predict)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [(0, 150, 50, 40, 40), (0, 112, 50, 40, 40), (0, 50, 50, 40, 40)]
    out = clf.name_boxes([frame], boxes)
    assert out[0] is None  # entirely outside
    assert out[1] is None  # 12 px left after clipping (88..100), under min_box_px
    assert out[2] == Naming("Parapercis colias", "Parapercis colias", 0.8)
    assert crop_box(frame, 150, 50, 40, 40, 0.1) is None


def test_unfreeze_is_inferred_from_tail_length():
    """Checkpoints from the wip fine-tune have no `unfreeze` key; the block count
    falls out of the tail's length, and a length no block count explains raises."""
    import torch

    from spyfish.ml.classifier.species_classifier import _infer_unfreeze

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2)  # 2 params
            self.b = torch.nn.LayerNorm(2)  # 2 params

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList(Block() for _ in range(6))
            self.ln_post = torch.nn.LayerNorm(2)
            self.proj = torch.nn.Parameter(torch.zeros(2, 2))

    visual = Visual()
    assert _infer_unfreeze(visual, [0] * (3 * 4 + 3)) == 3
    with pytest.raises(ValueError):
        _infer_unfreeze(visual, [0] * (3 * 4 + 4))
    with pytest.raises(ValueError):
        _infer_unfreeze(visual, [0] * (7 * 4 + 3))  # more blocks than the backbone has


def test_not_a_fish_is_kept_in_raw_csv_but_dropped_downstream(tmp_path):
    """A confident `not a fish` is written as is (no registry lookup) and every
    downstream reader leaves the row out; the file itself keeps it."""
    import pandas as pd

    from spyfish.config.wrapper import config
    from spyfish.ml.raw_detections import read_raw_detections

    reject = config.classifier_reject_label
    looked_up = []
    clf = SpeciesClassifier(
        lambda crops: np.array([[0.1, 0.9]] * len(crops)),
        ["Parapercis colias", reject],
        crop_margin=0.1,
        min_box_px=1,
        batch_size=8,
        p_min=0.5,
        fallback_label="fish",
        label_map={},
        canonical=lambda n: looked_up.append(n) or n,
        reject_label=reject,
    )
    assert looked_up == ["Parapercis colias"]
    naming = clf.name_crops([Image.new("RGB", (8, 8))])[0]
    assert clf.final_label(naming, "Parapercis colias", 0.9) == reject

    raw = tmp_path / "x_raw.csv"
    pd.DataFrame(
        {
            "frame": [1, 1],
            "class": ["Parapercis colias", reject],
            "confidence": [0.8, 0.7],
        }
    ).to_csv(raw, index=False)
    assert list(read_raw_detections(raw)["class"]) == ["Parapercis colias"]
    assert len(pd.read_csv(raw)) == 2


# ── Threaded crop preprocessing ──────────────────────────────────────────────


def _bioclip_like_preprocess():
    """The shape of open_clip's eval transform (bicubic resize, centre crop,
    tensor, normalise), built from torchvision so open_clip stays unimported."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48, 0.46, 0.41), (0.27, 0.26, 0.28)),
        ]
    )


def test_pooled_preprocessing_is_the_serial_tensors_in_the_same_order():
    """A thread pool only changes when each crop is resized, never the result:
    same pixels, same order, on crops of every odd size."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from spyfish.ml.classifier.species_classifier import preprocess_crops

    rng = np.random.default_rng(7)
    crops = [
        Image.fromarray(
            rng.integers(
                0, 256, (int(rng.integers(17, 400)), int(rng.integers(17, 700)), 3)
            ).astype(np.uint8)
        )
        for _ in range(67)
    ]
    preprocess = _bioclip_like_preprocess()
    serial = preprocess_crops(preprocess, crops)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pooled = preprocess_crops(preprocess, crops, pool)
        again = preprocess_crops(preprocess, crops, pool)
    assert len(pooled) == len(serial) == len(crops)
    assert torch.equal(torch.stack(pooled), torch.stack(serial))
    assert torch.equal(torch.stack(again), torch.stack(serial))


def test_pooled_preprocessing_keeps_order_when_crops_finish_out_of_order():
    import time
    from concurrent.futures import ThreadPoolExecutor

    from spyfish.ml.classifier.species_classifier import preprocess_crops

    def slow_for_early_items(i):
        time.sleep(0.02 * (5 - i))  # the first crop finishes last
        return i

    with ThreadPoolExecutor(max_workers=5) as pool:
        assert preprocess_crops(slow_for_early_items, range(5), pool) == [0, 1, 2, 3, 4]
    assert preprocess_crops(slow_for_early_items, [4], None) == [4]


def test_a_preprocess_error_in_a_worker_thread_reaches_the_caller():
    from concurrent.futures import ThreadPoolExecutor

    from spyfish.ml.classifier.species_classifier import preprocess_crops

    def boom(crop):
        raise ValueError("bad crop")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(ValueError, match="bad crop"):
            preprocess_crops(boom, [1, 2, 3], pool)


def test_preprocess_workers_is_a_required_non_negative_config_key(monkeypatch):
    from spyfish.config.wrapper import config

    assert config.classifier_preprocess_workers >= 0
    clf = dict(config._classifier)
    monkeypatch.setattr(
        type(config),
        "_classifier",
        property(lambda s: {**clf, "preprocess_workers": -1}),
    )
    with pytest.raises(ValueError, match="preprocess_workers"):
        config.classifier_preprocess_workers
    clf.pop("preprocess_workers")
    monkeypatch.setattr(type(config), "_classifier", property(lambda s: clf))
    with pytest.raises(KeyError):
        config.classifier_preprocess_workers
