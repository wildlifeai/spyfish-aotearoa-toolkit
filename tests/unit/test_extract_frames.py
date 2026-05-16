"""Unit tests for spyfish.extraction.extract_frames helpers."""

from __future__ import annotations

import csv

from spyfish.extraction.extract_frames import build_coco_from_raw_csv


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
            },
            {
                "image_id": 2,
                "file_name": "far.jpg",
                "time_of_max": 1700.0,  # way outside tolerance
                "img_w": 1920,
                "img_h": 1080,
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
            },
        ]

        coco = build_coco_from_raw_csv(str(raw_csv), frame_records)
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["image_id"] == 1
