"""Every clip sent to Zooniverse stays in an append-only record with its date.

The selections CSV is rewritten by each clips run (the 2026-11-02 top-up
replaced SLI_085's ML selection with the full grid), so it cannot say what
volunteers were shown.
"""

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.zooniverse.upload import CLIP_KIND, record_clip_uploads

DROP = "KSF_20240124_BUV_KSF_085_01"


def _clips(starts, reason, paths=True):
    start = config.csv_clip_start_absolute_column
    return pd.DataFrame(
        {
            config.drop_id_column: DROP,
            start: [float(s) for s in starts],
            "SelectionReason": reason,
            CLIP_KIND.path_column: [
                f"/clips/{s}.mp4" if paths else None for s in starts
            ],
        }
    )


def test_record_appends_and_keeps_first_upload_date(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "get_drop_annotations_dir", lambda self, d: tmp_path
    )
    record_clip_uploads(
        DROP, _clips([65, 75], "ML Strategy"), "2026-10-14T01:00:00+00:00"
    )
    path = record_clip_uploads(
        DROP, _clips([65, 75, 85], "Full Video Sample"), "2026-11-02T22:00:00+00:00"
    )

    rec = pd.read_csv(path)
    start = config.csv_clip_start_absolute_column
    assert rec[start].tolist() == [65.0, 75.0, 85.0]
    first = rec.set_index(start)
    assert first.loc[65.0, "UploadedAt"] == "2026-10-14T01:00:00+00:00"
    assert (
        first.loc[65.0, "SelectionReason"] == "ML Strategy"
    ), "the top-up keeps the original reason"
    assert first.loc[85.0, "UploadedAt"] == "2026-11-02T22:00:00+00:00"
    assert (
        CLIP_KIND.path_column not in rec.columns
    ), "local clip paths are not a record of what went up"
    assert path.name == f"{DROP}_zooniverse_clip_uploads.csv"


def test_clips_that_failed_extraction_are_not_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "get_drop_annotations_dir", lambda self, d: tmp_path
    )
    path = record_clip_uploads(
        DROP, _clips([65], "ML Strategy", paths=False), "2026-11-02T22:00:00+00:00"
    )
    assert pd.read_csv(path).empty
