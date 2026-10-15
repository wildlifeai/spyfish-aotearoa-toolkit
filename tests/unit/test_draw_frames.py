"""The shared QA-frame banner and count summary (spyfish/ml/draw_frames.py)."""

import numpy as np
import pandas as pd

from spyfish.ml.draw_frames import count_summary, draw_banner, draw_boxes


def test_draw_banner_paints_a_black_box_with_white_text_top_left():
    frame = np.full((200, 400, 3), 128, dtype=np.uint8)
    draw_banner(frame, "model_x conf>=0.15 @ 00:01:02  |  Pagrus auratus 2")
    assert (frame[:20, :200] == 0).any(), "banner background must be painted black"
    assert (frame[:40, :400] == 255).any(), "banner text must be painted white"
    # The rest of the frame is untouched.
    assert (frame[100:, :] == 128).all()


def test_count_summary_counts_only_boxes_above_threshold():
    rows = pd.DataFrame(
        {
            "class": ["Pagrus auratus", "Pagrus auratus", "fish", "fish"],
            "confidence": [0.9, 0.5, 0.3, 0.1],
        }
    )
    assert count_summary(rows, 0.15) == "Pagrus auratus 2, fish 1"
    assert count_summary(rows, 0.95) == "nothing"


def test_count_summary_and_boxes_use_the_labeller_for_names_only():
    """What a person reads is the label; the class name still keys the colour."""
    rows = pd.DataFrame(
        {
            "class": ["Pagrus auratus", "Pagrus auratus", "fish"],
            "confidence": [0.9, 0.5, 0.3],
            "x": [100, 300, 60],
            "y": [100, 100, 40],
            "w": [40, 40, 20],
            "h": [40, 40, 20],
        }
    )
    common = {"Pagrus auratus": "Snapper"}
    labeller = lambda c: common.get(c, c)  # noqa: E731
    assert count_summary(rows, 0.15, labeller) == "Snapper 2, fish 1"

    by_class = np.full((200, 400, 3), 128, dtype=np.uint8)
    by_label = by_class.copy()
    draw_boxes(by_class, rows, 0.15)
    draw_boxes(by_label, rows, 0.15, labeller)
    # Same box pixels either way (colour keyed on the class), only the tab text differs.
    assert (by_class[80:82, 80:120] == by_label[80:82, 80:120]).all()
    assert (by_class != by_label).any()


def test_video_frames_are_rotated_like_the_clean_frames(tmp_path, monkeypatch):
    """A 90° container rotation is baked into the QA JPEG, as the clean and
    legacy frames do, so the saved image is portrait for a portrait video."""
    import cv2

    from spyfish.ml.draw_frames import draw_boxes_on_video_frames

    drop_id = "KSF_20240124_BUV_KSF_085_01"
    video = tmp_path / f"{drop_id}.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 3, (64, 48))
    for i in range(9):
        w.write(np.full((48, 64, 3), (i * 10) % 255, dtype=np.uint8))
    w.release()

    raw = pd.DataFrame(
        [
            {
                "frame": 6,
                "time_seconds": 2.0,
                "class": "fish",
                "confidence": 0.9,
                "x": 32,
                "y": 24,
                "w": 20,
                "h": 16,
            }
        ]
    )
    raw_csv = tmp_path / "raw.csv"
    raw.to_csv(raw_csv, index=False)

    # cv2.VideoWriter writes no rotation tag, so stand in for the container.
    monkeypatch.setattr("spyfish.ml.draw_frames.read_video_rotation", lambda cap: 90)
    monkeypatch.setattr("spyfish.ml.draw_frames.common_name_labeller", lambda: str)

    out = draw_boxes_on_video_frames(
        video, raw_csv, tmp_path / "qa", [6], 0.25, drop_id=drop_id, model_name="m"
    )
    assert [p.split("/")[-1] for p in out] == [f"{drop_id}__frame_0002.000s.jpg"]
    img = cv2.imread(out[0])
    assert img.shape[:2] == (64, 48)  # landscape 64x48 source, now portrait
