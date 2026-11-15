"""Unit tests for the Survey review data layer (`app/doc_report/review_data.py`).

The view draws whatever these functions hand it, so a wrong answer here is a
wrong picture under a deployment or a wrong mean in the inside / outside table,
with nothing raised.

`app/` is not a package, so it goes on the path the same way the Streamlit
pages do.
"""

import io
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report import review_data  # noqa: E402

from spyfish.config.base import IngestStatus, VideoPresence  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402

DROP = "KSF_20240124_BUV_KSF_085_01"
OTHER_DROP = "KSF_20240124_BUV_KSF_085_02"
SYNONYMS = {"Pagrus auratus": "Chrysophrys auratus"}


def _canonical(name: str) -> str:
    return SYNONYMS.get(name, name)


def _review(root: Path, drop_id: str = DROP) -> Path:
    """The drop's review folder under a deployment data root."""
    return root / "KSF_20240124_BUV" / drop_id / config.review_dir_name


def _slate(root: Path, drop_id: str = DROP) -> Path:
    return root / "KSF_20240124_BUV" / drop_id / f"{drop_id}{config.slate_frame_suffix}"


def _jpeg(path: Path, mtime: float, size=(1920, 1080)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "navy").save(path, "JPEG")
    os.utime(path, (mtime, mtime))
    return path


# ── collect_frames ───────────────────────────────────────────────────────────


def test_renamed_species_keeps_only_the_newest_file(tmp_path):
    review = _review(tmp_path)
    _jpeg(review / f"{DROP}__frame_100.0s__maxn_expert_3_Pagrus_auratus.jpg", 1000)
    new = _jpeg(
        review / f"{DROP}__frame_120.0s__maxn_expert_4_Chrysophrys_auratus.jpg",
        2000,
    )

    frames = review_data.collect_frames(tmp_path, _canonical)[DROP]

    assert [f.path for f in frames] == [new]
    assert frames[0].scientific == "Chrysophrys auratus"
    assert frames[0].maxn == 4
    assert frames[0].source == "expert"


def test_manifest_file_wins_over_a_newer_leftover(tmp_path):
    review = _review(tmp_path)
    listed = _jpeg(
        review / f"{DROP}__frame_100.0s__maxn_expert_3_Parapercis_colias.jpg", 1000
    )
    _jpeg(review / f"{DROP}__frame_130.0s__maxn_expert_3_Parapercis_colias.jpg", 2000)
    pd.DataFrame([{"file": listed.name, "seconds": 100.4, "agreement": "off"}]).to_csv(
        review / config.review_manifest_name, index=False
    )

    frames = review_data.collect_frames(tmp_path, _canonical)[DROP]

    assert [f.path for f in frames] == [listed]
    assert frames[0].current
    assert frames[0].agreement == "off"
    assert frames[0].seconds == 100.4


def test_slate_comes_first_then_sources_best_first(tmp_path):
    review = _review(tmp_path)
    _jpeg(review / f"{DROP}__frame_10.0s__maxn_ml_2_Parapercis_colias.jpg", 1000)
    _jpeg(review / f"{DROP}__frame_20.0s__maxn_legacy_5_Parapercis_colias.jpg", 1000)
    _jpeg(_slate(tmp_path), 1000)
    _jpeg(review / "not_a_frame.jpg", 1000)
    # a frame outside a review folder is not picked up: no flat-folder fallback
    _jpeg(tmp_path / f"{DROP}__frame_30.0s__maxn_expert_9_Parapercis_colias.jpg", 1000)

    frames = review_data.collect_frames(tmp_path, _canonical)[DROP]

    assert [(f.kind, f.source) for f in frames] == [
        ("slate", ""),
        ("maxn", "expert_legacy"),
        ("maxn", "ml"),
    ]


def test_random_frame_caption_says_nothing_was_recorded(tmp_path):
    _jpeg(_review(tmp_path) / f"{DROP}__frame_1700.0s__maxn_random_0_none.jpg", 1000)

    (frame,) = review_data.collect_frames(tmp_path, _canonical)[DROP]

    assert frame.kind == review_data.RANDOM
    assert frame.caption(str).startswith("No MaxN recorded: random frame at 00:28:20")


# ── status in words ──────────────────────────────────────────────────────────


def test_short_reason():
    assert review_data.short_reason("sampling window is 42s shorter") == (
        "window 42 s short"
    )
    assert (
        review_data.short_reason(
            "sampling_end=1700s is shorter than expected (1800s after start)"
        )
        == "window 100 s short"
    )
    assert review_data.short_reason("missing sampling window") == "no sampling window"
    # this message also says "missing sampling window": the specific one wins
    assert (
        review_data.short_reason("sampling_start=0, missing sampling window")
        == "no sampling start"
    )
    assert review_data.short_reason("something else") == "something else"


def test_deployment_status():
    ok = {IngestStatus.COLUMN: IngestStatus.OK, "video_presence": VideoPresence.PRESENT}
    assert review_data.deployment_status(ok, []) == ("ok", "")

    no_video = {**ok, "video_presence": VideoPresence.ABSENT}
    assert review_data.deployment_status(no_video, [])[0] == "no video"

    bad = {IngestStatus.COLUMN: IngestStatus.EXCLUDED, "is_bad_deployment": 1}
    assert review_data.deployment_status(bad, []) == ("bad deployment", "not counted")

    error = {IngestStatus.COLUMN: IngestStatus.VALIDATION_ERROR}
    assert review_data.deployment_status(error, []) == ("error", "no reason recorded")
    assert review_data.deployment_status(error, ["sampling window is 42s shorter"]) == (
        "error",
        "window 42 s short",
    )


def test_slate_state():
    on_disk, verified, no_board = {DROP}, {DROP}, {OTHER_DROP}
    assert review_data.slate_state(DROP, on_disk, verified, no_board) == "confirmed"
    assert review_data.slate_state(DROP, on_disk, set(), no_board) == "not confirmed"
    assert review_data.slate_state(OTHER_DROP, on_disk, verified, no_board) == (
        "no board shown"
    )
    assert review_data.slate_state(OTHER_DROP, on_disk, verified, set()) == "missing"


# ── best_source_maxn ─────────────────────────────────────────────────────────


def _annotations(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["drop_id", "scientific_name", "max_interval", "source"]
    )


def test_best_source_only_and_absence_is_a_row_of_zeros():
    ann = _annotations(
        [
            (DROP, "Parapercis colias", 5, "expert"),
            (DROP, "Parapercis colias", 9, "ml"),
            (DROP, "Notolabrus celidotus", 4, "ml"),
            (OTHER_DROP, None, 0, "expert"),
        ]
    )

    chosen, per_drop = review_data.best_source_maxn(
        ann, [DROP, OTHER_DROP], ["expert", "ml"]
    )

    assert chosen.to_dict() == {DROP: "expert", OTHER_DROP: "expert"}
    # the ML rows of an expert-reviewed drop are not mixed in
    assert list(per_drop.columns) == ["Parapercis colias"]
    assert per_drop.loc[DROP, "Parapercis colias"] == 5
    assert per_drop.loc[OTHER_DROP, "Parapercis colias"] == 0


def test_best_source_respects_the_allowed_sources():
    ann = _annotations(
        [(DROP, "Parapercis colias", 5, "expert"), (DROP, "Parapercis colias", 9, "ml")]
    )

    chosen, per_drop = review_data.best_source_maxn(ann, [DROP], ["ml"])
    assert chosen[DROP] == "ml"
    assert per_drop.loc[DROP, "Parapercis colias"] == 9

    chosen, per_drop = review_data.best_source_maxn(ann, [DROP], ["citsci"])
    assert chosen.empty and per_drop.empty


# ── images ────────────────────────────────────────────────────────


def test_jpeg_bytes_shrinks_and_frames_outside_the_picture(tmp_path):
    src = _jpeg(tmp_path / "frame.jpg", 1000)

    thumb = Image.open(
        io.BytesIO(review_data.jpeg_bytes(src, 640, 72, border="#3CAA3C", border_px=6))
    )

    assert thumb.size == (640 + 12, 360 + 12)


# ── which frames a deployment shows ──────────────────────────────────────────


def test_pick_frames_falls_back_to_the_model_then_the_random_frame(tmp_path):
    review = _review(tmp_path)
    _jpeg(review / f"{DROP}__frame_10.0s__maxn_ml_2_Parapercis_colias.jpg", 1000)
    _jpeg(review / f"{DROP}__frame_1700.0s__maxn_random_0_none.jpg", 1000)
    _jpeg(_slate(tmp_path), 1000)
    mine = review_data.collect_frames(tmp_path, _canonical)[DROP]

    slate, picks = review_data.pick_frames(mine, ["expert"])
    assert [f.kind for f in slate] == ["slate"]
    assert [f.source for f in picks] == ["ml"]
    assert review_data.caveats(picks) == {"ml"}

    _, picks = review_data.pick_frames(
        [f for f in mine if f.source != "ml"], ["expert"]
    )
    assert [f.kind for f in picks] == [review_data.RANDOM]
    assert review_data.caveats(picks) == {review_data.RANDOM}


# ── frames the export could not cut, and timing flags ───────────────────────


def test_skipped_peaks_are_listed_and_timing_flags_reach_the_caption(tmp_path):
    review = _review(tmp_path)
    shown = _jpeg(
        review / f"{DROP}__frame_1900.0s__maxn_legacy_2_Parapercis_colias.jpg", 1000
    )
    pd.DataFrame(
        [
            {
                "drop_id": DROP,
                "source": "legacy",
                "scientific_name": "Parapercis colias",
                "maxn": 2,
                "seconds": 1900.0,
                "file": shown.name,
                "timing": "after sampling end (1860s): timing may be wrong",
            },
            {
                "drop_id": DROP,
                "source": "legacy",
                "scientific_name": "Pagrus auratus",
                "maxn": 5,
                "seconds": 2400.0,
                "file": "",
                "skipped": "no frame at 2400s, past the end of the video (1900s): timing may be wrong",
            },
            {
                "drop_id": DROP,
                "source": "citsci",
                "scientific_name": "Parapercis colias",
                "maxn": 1,
                "seconds": "",
                "file": "",
                "skipped": "no second recorded",
            },
        ]
    ).to_csv(review / config.review_manifest_name, index=False)

    mine = review_data.collect_frames(tmp_path, _canonical)[DROP]
    _, picks = review_data.pick_frames(mine, ["expert_legacy"])

    assert [f.path for f in picks] == [shown]  # a skipped peak is never drawn
    assert "⚠ after sampling end" in picks[0].caption(str)
    notes = review_data.skipped_notes(mine, ["expert_legacy"], str)
    assert notes == [
        "Chrysophrys auratus MaxN 5 at 00:40:00: not shown, no frame at 2400s, "
        "past the end of the video (1900s): timing may be wrong"
    ]  # canonical name; the citsci row is not a shown source
    assert len(review_data.skipped_notes(mine, ["expert_legacy", "citsci"], str)) == 2
