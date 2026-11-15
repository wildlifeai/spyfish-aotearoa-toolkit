"""run_yolo_inference: the decode thread feeds predict() the same frames, in
the same order and with the same absolute indexes, as the old sequential loop,
and neither a decoder failure nor a predict failure can hang the run."""

import csv
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spyfish.ml import run_inference as ri

FPS = 30.0
N_FRAMES = 300  # 10 s of video


class FakeCapture:
    """Sequential cv2.VideoCapture stand-in: read() returns the frame index as
    a 1x1 array so the test can check which frame reached the model."""

    def __init__(self, n_frames=N_FRAMES, fail_at=None):
        self.pos = 0
        self.n = n_frames
        self.fail_at = fail_at

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        return {
            cv2.CAP_PROP_FRAME_COUNT: self.n,
            cv2.CAP_PROP_POS_FRAMES: self.pos,
            cv2.CAP_PROP_FPS: FPS,
        }[prop]

    def set(self, prop, value):
        self.pos = int(round(value / 1000.0 * FPS))

    def grab(self):
        if self.pos >= self.n:
            return False
        self.pos += 1
        return True

    def read(self):
        if self.fail_at is not None and self.pos >= self.fail_at:
            raise OSError("decoder blew up")
        if self.pos >= self.n:
            return False, None
        frame = np.full((1, 1), self.pos, dtype=np.int32)
        self.pos += 1
        return True, frame

    def release(self):
        pass


def _box(cls=0, conf=0.9):
    t = lambda *v: SimpleNamespace(tolist=lambda: list(v))  # noqa: E731
    return SimpleNamespace(
        xywh=[t(1.0, 2.0, 3.0, 4.0)],
        cls=[cls],
        conf=[conf],
    )


def _fake_model(predict_fail_on_call=None):
    model = MagicMock()
    model.names = {0: "fish"}
    calls = {"n": 0, "batches": []}

    def predict(source, **kwargs):
        calls["n"] += 1
        calls["batches"].append([int(f[0, 0]) for f in source])
        if predict_fail_on_call == calls["n"]:
            raise RuntimeError("CUDA out of memory")
        return [SimpleNamespace(boxes=[_box()]) for _ in source]

    model.predict.side_effect = predict
    model._calls = calls
    return model


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    """The video model is cached per process; no test may see another's fake."""
    ri._load_video_model.cache_clear()
    yield
    ri._load_video_model.cache_clear()


def _weights(tmp_path, name="model.pt"):
    """A weights file on disk: the model cache keys on its path, mtime and size."""
    path = tmp_path / name
    if not path.exists():
        path.write_bytes(b"weights")
    return str(path)


@pytest.fixture
def small_batches(monkeypatch):
    monkeypatch.setattr(type(ri.config), "ml_batch_size", property(lambda s: 4))
    monkeypatch.setattr(type(ri.config), "ml_half", property(lambda s: True))
    monkeypatch.setattr(type(ri.config), "ml_nms_iou", property(lambda s: 0.45))
    monkeypatch.setattr(type(ri.config), "ml_nms_agnostic", property(lambda s: True))
    monkeypatch.setattr(ri, "get_cached_classifier", lambda: None)


def _run(tmp_path, cap, model, sampling_start=1.0, sampling_end=5.0, stride=10):
    out = tmp_path / "raw.csv"
    with (
        patch.object(ri.cv2, "VideoCapture", return_value=cap),
        patch.object(ri, "YOLO", return_value=model),
    ):
        ri.run_yolo_inference(
            "video.mp4",
            _weights(tmp_path),
            0.25,
            640,
            str(out),
            FPS,
            stride,
            "KSF_20240124_BUV_KSF_085_01",
            sampling_start,
            sampling_end,
        )
    return out


def test_threaded_decode_feeds_the_same_frames_as_the_sequential_loop(
    tmp_path, small_batches
):
    cap = FakeCapture()
    model = _fake_model()
    out = _run(tmp_path, cap, model, sampling_start=1.0, sampling_end=5.0, stride=10)

    # Seek lands on frame 30; one sample every 10 frames until 5 s (frame 150).
    expected = list(range(30, 151, 10))
    fed = [f for batch in model._calls["batches"] for f in batch]
    assert fed == expected
    # batches of ml_batch_size (4), last one partial
    assert [len(b) for b in model._calls["batches"]] == [4, 4, 4, 1]
    # absolute frame index and its time land in the CSV
    rows = list(csv.DictReader(out.open()))
    assert [int(r["frame"]) for r in rows] == expected
    assert float(rows[0]["time_seconds"]) == pytest.approx(1.0)
    assert out.exists() and not (tmp_path / "raw.csv.partial").exists()
    # FP16 came from config
    assert model.predict.call_args.kwargs["half"] is True


def test_decoder_failure_surfaces_and_leaves_no_csv(tmp_path, small_batches):
    cap = FakeCapture(fail_at=100)
    model = _fake_model()
    with pytest.raises(RuntimeError, match="decode failed"):
        _run(tmp_path, cap, model)
    assert not (tmp_path / "raw.csv").exists()


def test_predict_failure_stops_the_decoder_and_propagates(tmp_path, small_batches):
    """The decoder blocks on a full queue; a predict() error must unblock it
    (stop event) so the call returns instead of hanging the whole batch."""
    import threading

    before = threading.active_count()
    cap = FakeCapture()
    model = _fake_model(predict_fail_on_call=2)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _run(tmp_path, cap, model, sampling_end=9.0, stride=1)
    assert threading.active_count() == before  # decoder thread joined
    assert not (tmp_path / "raw.csv").exists()


class FakeClassifier:
    """Names every box it is given blue cod; refuses nothing."""

    def __init__(self):
        self.calls = []

    def name_boxes(self, frames, boxes):
        from spyfish.ml.classifier.species_classifier import Naming

        self.calls.append((len(frames), list(boxes)))
        return [Naming("Parapercis colias", "Parapercis colias", 0.97) for _ in boxes]

    def final_label(self, naming, yolo_class, yolo_conf):
        return naming.label


def test_without_classifier_the_raw_csv_keeps_its_eight_columns(
    tmp_path, small_batches
):
    out = _run(tmp_path, FakeCapture(), _fake_model(), sampling_end=2.0, stride=10)
    header = out.read_text().splitlines()[0].split(",")
    assert header == ri.RAW_COLUMNS


def test_classifier_names_boxes_and_keeps_yolo_opinion(
    tmp_path, small_batches, monkeypatch
):
    clf = FakeClassifier()
    monkeypatch.setattr(ri, "get_cached_classifier", lambda: clf)
    out = _run(tmp_path, FakeCapture(), _fake_model(), sampling_end=2.0, stride=10)
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0].keys()) == ri.RAW_COLUMNS + ri.CLASSIFIER_COLUMNS
    assert {r["class"] for r in rows} == {"Parapercis colias"}
    assert {r["yolo_class"] for r in rows} == {"fish"}
    assert float(rows[0]["confidence"]) == 0.9  # detection score, not the classifier's
    assert float(rows[0]["yolo_confidence"]) == 0.9
    assert float(rows[0]["clf_confidence"]) == 0.97
    # one classifier call per YOLO batch, each box tagged with its frame's index in the batch
    assert [n for n, _ in clf.calls] == [4]
    assert [b[0] for b in clf.calls[0][1]] == [0, 1, 2, 3]


def test_box_the_classifier_declines_keeps_yolo_name_and_blank_columns(
    tmp_path, small_batches, monkeypatch
):
    class Declines(FakeClassifier):
        def name_boxes(self, frames, boxes):
            return [None for _ in boxes]

    monkeypatch.setattr(ri, "get_cached_classifier", lambda: Declines())
    out = _run(tmp_path, FakeCapture(), _fake_model(), sampling_end=2.0, stride=10)
    row = next(csv.DictReader(out.open()))
    assert row["class"] == "fish" and row["yolo_class"] == "fish"
    assert row["clf_class"] == "" and row["clf_confidence"] == ""


class StallingCapture(FakeCapture):
    """Header says N_FRAMES, but read()/grab() report end of video after
    `readable` frames: the UoA Vault reads on NeSI (2026-11-11)."""

    def __init__(self, readable):
        super().__init__()
        self.readable = readable

    def grab(self):
        return super().grab() if self.pos < self.readable else False

    def read(self):
        return super().read() if self.pos < self.readable else (False, None)


def test_stalled_read_is_not_promoted_as_a_finished_csv(tmp_path, small_batches):
    cap = StallingCapture(readable=60)
    model = _fake_model()
    with pytest.raises(RuntimeError, match="before the capture stopped"):
        _run(tmp_path, cap, model, sampling_start=1.0, sampling_end=None)
    assert not (tmp_path / "raw.csv").exists()
    assert (tmp_path / "raw.csv.partial").exists()


def test_second_stage_off_never_loads_the_classifier(
    tmp_path, small_batches, monkeypatch
):
    """A binary run (UoA) passes second_stage=False: `class` stays YOLO's, even
    with ml_inference.classifier.enabled in config."""

    def _boom():
        raise AssertionError("classifier loaded despite second_stage=False")

    monkeypatch.setattr(ri, "get_cached_classifier", _boom)
    out = tmp_path / "raw.csv"
    with (
        patch.object(ri.cv2, "VideoCapture", return_value=FakeCapture()),
        patch.object(ri, "YOLO", return_value=_fake_model()),
    ):
        ri.run_yolo_inference(
            "video.mp4",
            _weights(tmp_path),
            0.25,
            640,
            str(out),
            FPS,
            10,
            "KSF_20240124_BUV_KSF_085_01",
            0.0,
            None,
            second_stage=False,
        )
    rows = list(csv.DictReader(out.open()))
    assert rows and {r["class"] for r in rows} == {"fish"}
    assert "clf_class" not in rows[0]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_ffmpeg_reader_yields_the_frames_cv2_numbers(tmp_path):
    """Frame k of the ffmpeg pipe is cv2's frame k: each frame of a synthetic
    video is painted with its own index, and the sampled grid reads back the
    indexes it claims, from a non-zero start."""
    import cv2

    path = tmp_path / "grid.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(120):
        w.write(np.full((48, 64, 3), 2 * i, dtype=np.uint8))
    w.release()
    got = list(ri.ffmpeg_sampled_frames(path, start_frame=7, stride=10, n_frames=11))
    assert [i for i, _ in got] == list(range(7, 117, 10))
    for i, frame in got:
        assert frame.shape == (48, 64, 3)
        assert abs(float(frame.mean()) - 2 * i) < 3, (i, frame.mean())


def test_predict_kwargs_pass_half_only_when_given(small_batches):
    """Video inference sends FP16; prediction on frame files sends no `half`
    at all, so Ultralytics keeps its own default there."""
    shared = {
        "conf": 0.25,
        "imgsz": 640,
        "iou": 0.45,
        "agnostic_nms": True,
        "verbose": False,
        "project": None,
        "name": "ul_predict",
        "exist_ok": True,
        "save": False,
    }
    assert ri._predict_kwargs(0.25, 640) == shared
    assert ri._predict_kwargs(0.25, 640, half=True) == {**shared, "half": True}
    assert ri._predict_kwargs(0.25, 640, half=False) == {**shared, "half": False}


def test_video_readers_are_still_importable_from_run_inference():
    """Scripts outside the package import these from here; they live in
    spyfish.extraction.video_io."""
    from spyfish.extraction import video_io

    assert ri.ffmpeg_sampled_frames is video_io.ffmpeg_sampled_frames
    assert ri.ffmpeg_frame_at is video_io.ffmpeg_frame_at
    assert ri.get_video_fps is video_io.get_video_fps


# ── One YOLO load per weights file per process ───────────────────────────────


def _run_drop(tmp_path, weights, out_name, conf=0.25, imgsz=640):
    out = tmp_path / out_name
    with patch.object(ri.cv2, "VideoCapture", return_value=FakeCapture()):
        ri.run_yolo_inference(
            "video.mp4",
            weights,
            conf,
            imgsz,
            str(out),
            FPS,
            10,
            "KSF_20240124_BUV_KSF_085_01",
            1.0,
            2.0,
        )
    return out


def test_two_drops_on_one_model_construct_yolo_once(tmp_path, small_batches):
    """The second drop reuses the first drop's model, and still gets its own
    conf / imgsz: they are passed to predict() on every call."""
    built = []

    def fake_yolo(path):
        built.append(path)
        return _fake_model()

    weights = _weights(tmp_path)
    with patch.object(ri, "YOLO", side_effect=fake_yolo):
        first = _run_drop(tmp_path, weights, "a.csv", conf=0.25, imgsz=640)
        model = ri.get_cached_video_model(weights, True)
        kwargs_first = dict(model.predict.call_args.kwargs)
        second = _run_drop(tmp_path, weights, "b.csv", conf=0.4, imgsz=1280)
        kwargs_second = dict(model.predict.call_args.kwargs)

    assert len(built) == 1
    assert first.read_text() == second.read_text()  # same frames, same boxes
    assert (kwargs_first["conf"], kwargs_first["imgsz"]) == (0.25, 640)
    assert (kwargs_second["conf"], kwargs_second["imgsz"]) == (0.4, 1280)
    assert kwargs_first["half"] is True and kwargs_second["half"] is True


def test_another_weights_file_gets_its_own_model(tmp_path, small_batches):
    """`--model` runs and the UoA detector must never be served the pipeline
    model: the cache is per weights file."""
    built = []

    def fake_yolo(path):
        built.append(Path(path).name)
        return _fake_model()

    a, b = _weights(tmp_path, "a.pt"), _weights(tmp_path, "b.pt")
    with patch.object(ri, "YOLO", side_effect=fake_yolo):
        _run_drop(tmp_path, a, "a.csv")
        _run_drop(tmp_path, b, "b.csv")
        _run_drop(tmp_path, a, "a2.csv")
        assert ri.get_cached_video_model(a, True) is not ri.get_cached_video_model(
            b, True
        )
    assert built == ["a.pt", "b.pt"]


def test_model_cache_reloads_replaced_weights_and_splits_on_half(tmp_path):
    """Weights replaced on disk under the same name are a new model, and FP16
    and FP32 never share an instance (Ultralytics fixes the precision when the
    predictor is first set up)."""
    import os

    built = []

    def fake_yolo(path):
        built.append(path)
        return _fake_model()

    weights = _weights(tmp_path)
    with patch.object(ri, "YOLO", side_effect=fake_yolo):
        half = ri.get_cached_video_model(weights, True)
        assert ri.get_cached_video_model(weights, True) is half
        assert ri.get_cached_video_model(weights, False) is not half
        assert len(built) == 2
        Path(weights).write_bytes(b"retrained weights")
        os.utime(weights, ns=(1, 1))
        assert ri.get_cached_video_model(weights, True) is not half
    assert len(built) == 3


def test_missing_weights_still_raise_file_not_found(tmp_path, small_batches):
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        _run_drop(tmp_path, str(tmp_path / "absent.pt"), "a.csv")
