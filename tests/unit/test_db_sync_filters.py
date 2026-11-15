"""Which files `sync_annotations` sends to S3.

`aws s3 sync` takes the rules as a flat list where a later rule wins, so a
file's fate is decided by the last pattern it matches. These tests replay the
list the same way instead of asserting on its wording.
"""

from fnmatch import fnmatchcase
from unittest.mock import patch

from spyfish.storage import db_sync

DROP = "KSF_20240124_BUV_KSF_085_01"
FOLDER = f"KSF_20240124_BUV/{DROP}"


def _synced(key: str) -> bool:
    with patch.object(db_sync, "S3Handler") as handler:
        db_sync.sync_annotations()
    filters = handler.return_value.sync_local_to_s3.call_args.kwargs["filters"]
    keep = True
    for rule, pattern in zip(filters[::2], filters[1::2]):
        if fnmatchcase(key, pattern):
            keep = rule == "--include"
    return keep


def test_slate_frame_is_synced():
    assert _synced(f"{FOLDER}/{DROP}_slate.jpg")


def test_slate_runner_ups_and_video_are_not():
    assert not _synced(f"{FOLDER}/{DROP}_slate_candidates/1_0003.20s.jpg")
    assert not _synced(f"{FOLDER}/{DROP}.mp4")


def test_slate_under_an_external_volume_stays_local():
    assert not _synced(f"extra_no_survey_id/vol/{DROP}_slate.jpg")


def test_review_frames_and_their_manifest_are_synced():
    review = f"{FOLDER}/review"
    assert _synced(
        f"{review}/{DROP}__frame_0100.000s__maxn_expert_3_Parapercis_colias.jpg"
    )
    assert _synced(f"{review}/maxn_frames.csv")
    assert not _synced(f"{review}/notes.jpg")
    assert not _synced(f"{review}/scores.csv")
    assert not _synced("extra_no_survey_id/vol/review/maxn_frames.csv")


def test_existing_rules_still_hold():
    assert _synced(f"{FOLDER}/annotations/{DROP}_zooniverse_maxn.csv")
    assert _synced(f"{FOLDER}/qa_frames/frame_000100.jpg")
