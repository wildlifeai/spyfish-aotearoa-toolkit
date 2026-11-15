"""Every BIIGLE annotation upload leaves a dated record that is never overwritten.

The working COCO keeps a fixed name and can be rewritten by a later
re-extraction; the S3 sync never deletes but does replace same-name files, so
without a new name per upload the record of what the expert reviewed could be
lost everywhere (2026-11-07).
"""

import json

from spyfish.biigle import upload_frames
from spyfish.config.wrapper import config

DROP = "KSF_20240124_BUV_KSF_085_01"


def test_record_holds_coco_and_ids_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "get_drop_annotations_dir", lambda self, d: tmp_path
    )
    monkeypatch.setattr(upload_frames.time, "strftime", lambda fmt: "20261107_120000")
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 2, 3, 4]}],
    }
    first = upload_frames.save_upload_record(DROP, 123, coco, {"details": [{"id": 9}]})
    second = upload_frames.save_upload_record(
        DROP, 123, {"images": [], "annotations": []}, {}
    )
    assert first != second, "same second, same volume: a second name, not an overwrite"
    rec = json.loads(first.read_text())
    assert (
        rec["coco"] == coco
        and rec["biigle_annotations"] == [{"id": 9}]
        and rec["volume_id"] == 123
    )
    assert first.name == f"{DROP}_biigle_upload_123_20261107_120000.json"


def test_training_records_are_named_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "get_drop_annotations_dir", lambda self, d: tmp_path
    )
    p = upload_frames.save_upload_record(
        DROP, 7, {"images": [], "annotations": []}, {}, target="training"
    )
    assert p.name.startswith(f"{DROP}_training_biigle_upload_7_")
