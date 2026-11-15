"""Unit tests for spyfish.extraction.extract_frames helpers."""

from __future__ import annotations

import csv

from spyfish.extraction.extract_frames import build_coco_from_raw_csv

# build_coco_from_raw_csv requires every frame record to name its source video
# (single-video raw CSVs carry no drop_id column, so the record is the only
# thing scoping the nearest-time lookup).
DROP_ID = "KSF_20240124_BUV_KSF_085_01"


def _write_raw_csv(path, rows):
    """Write a minimal raw inference CSV with the columns build_coco expects."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"])
        for r in rows:
            w.writerow(r)


class TestBuildCocoTimeTolerance:
    """The tolerance guard in build_coco_from_raw_csv prevents the
    training-frames flow from inheriting annotations from a distant
    detection just because no closer row exists in a sparse raw CSV.
    """

    def test_distant_detection_does_not_snap_to_empty_frame(self, tmp_path):
        # Raw CSV has a single detection at t=120s. A training frame at
        # t=1700s must NOT pick up that detection — they're 1580s apart.
        raw_csv = tmp_path / "raw.csv"
        _write_raw_csv(
            raw_csv,
            [[3600, 120.0, "fish", 0.9, 10, 10, 50, 50]],
        )

        frame_records = [
            {
                "image_id": 1,
                "file_name": "near.jpg",
                "time_of_max": 120.05,  # within default 1.0s tolerance
                "img_w": 1920,
                "img_h": 1080,
                "drop_id": DROP_ID,
            },
            {
                "image_id": 2,
                "file_name": "far.jpg",
                "time_of_max": 1700.0,  # way outside tolerance
                "img_w": 1920,
                "img_h": 1080,
                "drop_id": DROP_ID,
            },
        ]

        coco = build_coco_from_raw_csv(str(raw_csv), frame_records)

        # Both frames present as images.
        assert {img["id"] for img in coco["images"]} == {1, 2}
        # But only the near frame got the annotation.
        annotated_image_ids = {ann["image_id"] for ann in coco["annotations"]}
        assert annotated_image_ids == {1}, (
            f"Expected only image_id=1 to get annotations, got "
            f"{annotated_image_ids} — the distant frame is wrongly inheriting "
            "a detection from a faraway timestamp."
        )

    def test_custom_tolerance_widens_match_window(self, tmp_path):
        # With a generous 10s tolerance, a frame 5s away from the detection
        # SHOULD pick it up.
        raw_csv = tmp_path / "raw.csv"
        _write_raw_csv(
            raw_csv,
            [[3600, 120.0, "fish", 0.9, 10, 10, 50, 50]],
        )

        frame_records = [
            {
                "image_id": 1,
                "file_name": "f.jpg",
                "time_of_max": 125.0,  # 5s away — outside default, inside 10s
                "img_w": 1920,
                "img_h": 1080,
                "drop_id": DROP_ID,
            },
        ]

        coco_default = build_coco_from_raw_csv(str(raw_csv), frame_records)
        assert len(coco_default["annotations"]) == 0

        coco_wide = build_coco_from_raw_csv(
            str(raw_csv), frame_records, max_time_delta_seconds=10.0
        )
        assert len(coco_wide["annotations"]) == 1
        assert coco_wide["annotations"][0]["image_id"] == 1

    def test_dense_csv_within_default_tolerance_still_matches(self, tmp_path):
        # MaxN flow has a row per video frame — nearest row is sub-second
        # away from time_of_max. The default tolerance must NOT break this.
        raw_csv = tmp_path / "raw.csv"
        # 30fps-ish rows around t=600s, plus a detection at the target time.
        rows = []
        for i in range(580 * 30, 620 * 30):
            t = i / 30.0
            if abs(t - 600.0) < 0.001:
                rows.append([i, t, "snapper", 0.8, 100, 100, 80, 80])
        _write_raw_csv(raw_csv, rows)

        frame_records = [
            {
                "image_id": 1,
                "file_name": "maxn.jpg",
                "time_of_max": 600.0,
                "img_w": 1920,
                "img_h": 1080,
                "drop_id": DROP_ID,
            },
        ]

        coco = build_coco_from_raw_csv(str(raw_csv), frame_records)
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["image_id"] == 1


class _FakeCapture:
    """cv2.VideoCapture stand-in whose frames are 1x2 arrays holding their own
    index, and which records how each frame was reached."""

    def __init__(self, n_frames=1000, unreadable=(), opened=True, rotation=0):
        self.pos = 0
        self.n = n_frames
        self.unreadable = set(unreadable)
        self.opened = opened
        self.rotation = rotation
        self.seeks = []
        self.grabs = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.rotation

    def set(self, prop, value):
        self.seeks.append(int(value))
        self.pos = int(value)

    def grab(self):
        self.grabs += 1
        self.pos += 1
        return self.pos <= self.n

    def read(self):
        import numpy as np

        idx = self.pos
        self.pos += 1
        if idx >= self.n or idx in self.unreadable:
            return False, None
        return True, np.full((1, 2), idx, dtype=np.int32)

    def release(self):
        self.released = True


def _raw_rows(frames):
    import pandas as pd

    return pd.DataFrame(
        {
            "frame": frames,
            "time_seconds": [f / 30.0 for f in frames],
            "x": range(len(frames)),
        }
    )


class TestIterRawFrames:
    def _iterate(self, monkeypatch, cap, df, **kwargs):
        from spyfish.extraction import extract_frames as ef

        monkeypatch.setattr(ef.cv2, "VideoCapture", lambda path: cap)
        return [
            (idx, None if frame is None else int(frame[0, 0]), list(rows["x"]))
            for idx, frame, rows in ef.iter_raw_frames("video.mp4", df, **kwargs)
        ]

    def test_seeks_to_every_frame_in_order_with_its_rows(self, monkeypatch):
        cap = _FakeCapture()
        got = self._iterate(monkeypatch, cap, _raw_rows([40, 10, 40, 500]))
        assert got == [(10, 10, [1]), (40, 40, [0, 2]), (500, 500, [3])]
        assert cap.seeks == [10, 40, 500] and cap.grabs == 0
        assert cap.released

    def test_read_ahead_decodes_short_gaps_and_seeks_long_ones(self, monkeypatch):
        cap = _FakeCapture()
        got = self._iterate(
            monkeypatch, cap, _raw_rows([0, 10, 20, 500, 510]), max_read_ahead=120
        )
        assert [(i, f) for i, f, _ in got] == [(i, i) for i in (0, 10, 20, 500, 510)]
        # 0, 10, 20 and 510 are read forward; only the 480-frame jump seeks.
        assert cap.seeks == [500]
        assert cap.grabs == 9 + 9 + 9

    def test_failed_read_yields_none_and_later_frames_still_come(self, monkeypatch):
        cap = _FakeCapture(unreadable={300})
        got = self._iterate(
            monkeypatch, cap, _raw_rows([290, 300, 310]), max_read_ahead=120
        )
        assert [(i, f) for i, f, _ in got] == [(290, 290), (300, None), (310, 310)]
        # position unknown after the failure, so 310 is reached by a seek
        assert cap.seeks == [290, 310]

    def test_failed_read_near_the_start_still_seeks_for_the_next(self, monkeypatch):
        # Frame 20 is within read-ahead of the start of the file, but after a
        # failed read the capture is not at the start any more.
        cap = _FakeCapture(unreadable={10})
        got = self._iterate(monkeypatch, cap, _raw_rows([10, 20]), max_read_ahead=120)
        assert [(i, f) for i, f, _ in got] == [(10, None), (20, 20)]
        assert cap.seeks == [20]

    def test_rotation_from_the_container_is_applied_on_both_paths(self, monkeypatch):
        for kwargs in ({}, {"max_read_ahead": 120}):
            cap = _FakeCapture(rotation=90)
            from spyfish.extraction import extract_frames as ef

            monkeypatch.setattr(ef.cv2, "VideoCapture", lambda path, cap=cap: cap)
            shapes = [
                frame.shape
                for _, frame, _ in ef.iter_raw_frames(
                    "video.mp4", _raw_rows([0, 5]), **kwargs
                )
            ]
            assert shapes == [(2, 1), (2, 1)]  # 1x2 turned on its side

    def test_video_that_will_not_open_raises_its_own_error(self, monkeypatch):
        import pytest

        from spyfish.extraction.extract_frames import VideoOpenError

        with pytest.raises(VideoOpenError, match="video.mp4"):
            self._iterate(monkeypatch, _FakeCapture(opened=False), _raw_rows([0]))
