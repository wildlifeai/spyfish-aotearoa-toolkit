"""Slate detection on synthetic frames, and the per-drop save/skip behaviour."""

import dataclasses

import cv2
import numpy as np
import pytest

from spyfish.config.wrapper import config
from spyfish.extraction import slate
from spyfish.extraction.slate import SlateParams, extract_slate_frame, score_frame

DROP = "KSF_20240124_BUV_KSF_085_01"
SEA = (200, 120, 40)  # BGR: saturated blue, like water or a deck-side sea view


def _sea(w=640, h=360):
    return np.full((h, w, 3), SEA, dtype=np.uint8)


DECK = (
    95,
    95,
    95,
)  # BGR: grey non-slip deck (value ~93 measured), what fills a deck shot around the board


def _deck(w=640, h=360):
    return np.full((h, w, 3), DECK, dtype=np.uint8)


WOOD = (90, 150, 215)  # BGR: varnished clipboard, like the AHE slate


def _board(
    frame,
    x=150,
    y=60,
    w=340,
    h=250,
    strokes=30,
    colour=(245, 245, 245),
    ink=(20, 20, 20),
    mark=6,
):
    """A board with handwriting-sized marks, held roughly square-on."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, -1)
    rng = np.random.default_rng(0)
    for _ in range(strokes):
        sx = int(rng.integers(x + 15, x + w - 25))
        sy = int(rng.integers(y + 15, y + h - 25))
        cv2.rectangle(frame, (sx, sy), (sx + mark, sy + mark), ink, -1)
    return frame


@pytest.fixture
def params():
    return SlateParams.from_config()


def test_params_come_from_config(params):
    assert params.stroke_area_frac == tuple(config.slate_detection["stroke_area_frac"])
    assert params.max_seconds == config.slate_detection["max_seconds"]


def test_written_board_scores(params):
    fs = score_frame(_board(_sea()), params)
    assert fs.score > 0
    assert fs.strokes >= params.min_strokes
    assert fs.aspect < params.max_aspect


def test_full_written_board_beats_mostly_blank_close_up(params):
    """Held close, the board fills the frame but the writing is cut off: the
    wider shot with all the handwriting is the one worth reading."""
    close_up = score_frame(
        _board(_sea(), x=60, y=0, w=520, h=360, strokes=params.min_strokes), params
    )
    full = score_frame(_board(_sea(), strokes=30), params)
    assert close_up.score > 0
    assert close_up.paper_frac > full.paper_frac
    assert full.score > close_up.score


def test_wooden_board_scores_as_wood_for_a_wood_survey(params, monkeypatch):
    """AHE's clipboard is orange wood, invisible to the white mask."""
    monkeypatch.setitem(config.slate_detection, "wood_surveys", ["KSF_20240124_BUV"])
    fs = score_frame(_board(_sea(), colour=WOOD), SlateParams.from_config(DROP))
    assert fs.score > 0
    assert fs.board == "wood"


def test_wood_mask_is_off_for_whiteboard_surveys(params, monkeypatch):
    """Skin shares wood's colour; where the board is white, wood never runs."""
    monkeypatch.setitem(config.slate_detection, "wood_surveys", [])
    assert SlateParams.from_config(DROP).boards == ("white",)
    assert (
        score_frame(_board(_sea(), colour=WOOD), SlateParams.from_config(DROP)).score
        == 0
    )


@pytest.fixture
def wood_params(params):
    return dataclasses.replace(params, boards=("white", "wood"))


def test_convex_but_not_rectangular_blob_does_not_score(wood_params):
    params = wood_params
    """Hands and lit deck corners pass solidity (convex enough) but do not fill
    their bounding box the way a board does; rect_fill is the gate for them."""
    frame = _deck()
    tri = np.array([[150, 310], [490, 310], [320, 60]], dtype=np.int32)
    cv2.fillPoly(frame, [tri], WOOD)
    rng = np.random.default_rng(1)
    for _ in range(30):  # marks inside the triangle, so the stroke gate passes
        sx, sy = int(rng.integers(260, 380)), int(rng.integers(200, 290))
        cv2.rectangle(frame, (sx, sy), (sx + 6, sy + 6), (20, 20, 20), -1)
    fs = score_frame(frame, params)
    assert fs.solidity >= params.min_solidity
    assert fs.rect_fill < params.min_rect_fill
    assert fs.score == 0


def test_plain_wood_without_writing_does_not_score(wood_params):
    """A wooden deck or thwart in the same colour has no strokes."""
    assert score_frame(_board(_sea(), strokes=0, colour=WOOD), wood_params).score == 0


def test_blurred_board_is_less_sharp(params):
    crisp = _board(_sea())
    blurred = cv2.GaussianBlur(crisp, (0, 0), 3)
    assert score_frame(crisp, params).sharpness > score_frame(blurred, params).sharpness


def test_rank_candidates_prefers_sharp_within_band():
    """Near-equal scores: sharpest first; a clearly lower score stays behind."""
    mk = lambda s, sharp: slate.FrameScore(
        0.0, 0.3, 0.9, 1.3, 0.8, 0.05, 20, s, "white", sharp
    )
    top, near, far = mk(10.0, 50.0), mk(9.0, 400.0), mk(5.0, 999.0)
    ranked = slate.rank_candidates([top, near, far], tie_band=0.2)
    assert [f.score for f in ranked] == [9.0, 10.0, 5.0]


def test_near_miss_names_the_blocking_gate(params):
    """A board with one mark too few fails only the stroke gate, and says so."""
    fs = score_frame(_board(_sea(), strokes=2), params)
    assert fs.score == 0
    assert fs.fails == "strokes"


def test_green_marker_counts_as_ink(params):
    """SLI 2022 wrote in green, OKA in blue: saturated, not dark, still handwriting."""
    fs = score_frame(_board(_sea(), ink=(60, 170, 60)), params)
    assert fs.strokes >= params.min_strokes
    assert fs.score > 0


def test_big_letters_on_a_close_up_board_still_count(params):
    """Stroke size scales with the board: a letter is big when the board fills the frame."""
    close = _board(_sea(), x=40, y=20, w=560, h=320, strokes=20, mark=22)
    fs = score_frame(close, params)
    assert fs.strokes >= 15


def test_whole_board_beats_the_same_board_cut_off(params):
    whole = score_frame(_board(_deck(), x=150, y=60), params)
    cut = score_frame(
        _board(_deck(), x=400, y=200), params
    )  # runs off the right and bottom edges
    assert cut.score > 0
    assert whole.score > cut.score


def test_written_board_wins_over_a_bigger_blank_white_hull(params):
    """A white boat hull beside the board is the largest pale blob in the frame.
    It has no writing, so it must not hide the smaller written board behind it
    (SLI_20220228_BUV_SLI_066_01 at 62 s, TON 2024: every drop)."""
    frame = _sea()
    cv2.rectangle(
        frame, (0, 200), (640, 360), (240, 240, 240), -1
    )  # hull, blank, cut off at 3 edges
    _board(frame, x=360, y=20, w=240, h=160, strokes=25)
    fs = score_frame(frame, params)
    assert fs.score > 0
    assert fs.strokes >= params.min_strokes
    assert fs.paper_frac < 0.25  # the board's area, not the hull's


def test_pale_flood_off_three_edges_does_not_score(params):
    """Deck, sky and sun glare fill the frame from edge to edge; a held board
    never runs off more than two (KSF_085_06 at 5 s: deck + chilly-bin lid,
    strokes from rope texture, cut on 3 sides, outscored the real board)."""
    frame = _deck()
    cv2.rectangle(
        frame, (0, 0), (640, 300), (235, 235, 235), -1
    )  # runs off top, left, right
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y = int(rng.integers(10, 620)), int(rng.integers(10, 280))
        cv2.rectangle(frame, (x, y), (x + 6, y + 6), (30, 30, 30), -1)
    fs = score_frame(frame, params)
    assert fs.score == 0
    assert "cut_off" in fs.fails


def test_blank_board_does_not_score(params):
    """A white panel with nothing written on it is a wall or a deck, not a slate."""
    fs = score_frame(_board(_sea(), strokes=0), params)
    assert fs.score == 0


def test_underwater_frame_does_not_score(params):
    fs = score_frame(_sea(), params)
    assert fs.score == 0
    assert fs.fails == "water"


SHADED = (
    169,
    140,
    125,
)  # BGR: white paper in open-sky shade, blue-tinted (hue 107, sat 66, val 169 measured on KOK)


def test_shaded_board_is_still_paper(params):
    """Shade under open sky tints white paper blue; the whole board must stay in
    the mask (at saturation 50 only its lit half did, KOK_054 came back 0.03)."""
    fs = score_frame(_board(_deck(), colour=SHADED), params)
    assert fs.fails == ""
    assert fs.paper_frac > 0.3


SEABED = tuple(
    [216, 178, 102]
)  # BGR of HSV (100, 135, 216): the median seabed pixel, KSF_085_02 (2026-11-06)


def test_pale_seabed_is_skipped_as_water(params):
    """A pale seabed passes the paper mask like a board; its saturated blue says water."""
    frame = np.full((360, 640, 3), SEABED, dtype=np.uint8)
    rng = np.random.default_rng(2)
    for _ in range(60):  # rocks and shells: dark specks that would count as strokes
        x, y = int(rng.integers(0, 630)), int(rng.integers(0, 350))
        cv2.rectangle(frame, (x, y), (x + 5, y + 5), (90, 70, 60), -1)
    fs = score_frame(frame, params)
    assert fs.score == 0
    assert fs.fails == "water"


def test_long_thin_white_strip_does_not_score(params):
    """Deck edges and gunwales are bright and marked, but 4:1 or longer."""
    fs = score_frame(_board(_sea(), x=20, y=100, w=600, h=100), params)
    assert fs.aspect > params.max_aspect
    assert fs.score == 0


def test_extract_saves_jpeg_to_drop_path(tmp_path, monkeypatch):
    out = tmp_path / "drop" / f"{DROP}_slate.jpg"
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)
    best = slate.FrameScore(2.0, 0.25, 0.9, 1.3, 0.8, 0.08, 25, 0.2)
    seen = {}

    def fake_find(
        video,
        p=None,
        max_seconds=None,
        k=1,
        diag=None,
        exclude=(),
        distinct_seconds=0.0,
    ):
        seen["max_seconds"] = max_seconds
        return [(best, _board(_sea()))]

    monkeypatch.setattr(slate, "find_slate_candidates", fake_find)

    assert extract_slate_frame(DROP, tmp_path / "v.mp4", max_seconds=42.0) == out
    assert seen["max_seconds"] == 42.0
    assert out.exists()
    assert cv2.imread(str(out)).shape == (360, 640, 3)


def test_extract_returns_none_when_no_slate(tmp_path, monkeypatch):
    out = tmp_path / f"{DROP}_slate.jpg"
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)
    monkeypatch.setattr(
        slate,
        "find_slate_candidates",
        lambda v, p=None, max_seconds=None, k=1, diag=None, exclude=(), distinct_seconds=0.0: [],
    )

    assert extract_slate_frame(DROP, tmp_path / "v.mp4") is None
    assert not out.exists()


def test_extract_skips_when_jpeg_already_exists(tmp_path, monkeypatch):
    out = tmp_path / f"{DROP}_slate.jpg"
    out.write_bytes(b"old")
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)

    def boom(_, max_seconds=None):
        raise AssertionError("video must not be opened when the JPEG exists")

    monkeypatch.setattr(slate, "find_slate_candidates", boom)
    assert extract_slate_frame(DROP, tmp_path / "v.mp4") == out
    assert out.read_bytes() == b"old"


def test_extract_force_rescans_over_existing_jpeg(tmp_path, monkeypatch):
    out = tmp_path / f"{DROP}_slate.jpg"
    out.write_bytes(b"old")
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)
    best = slate.FrameScore(2.0, 0.25, 0.9, 1.3, 0.8, 0.08, 25, 0.2)
    monkeypatch.setattr(
        slate,
        "find_slate_candidates",
        lambda v, p=None, max_seconds=None, k=1, diag=None, exclude=(), distinct_seconds=0.0: [
            (best, _board(_sea()))
        ],
    )

    assert extract_slate_frame(DROP, tmp_path / "v.mp4", force=True) == out
    assert out.read_bytes() != b"old"
    assert cv2.imread(str(out)).shape == (360, 640, 3)


def test_extract_rejects_bad_drop_id(tmp_path):
    with pytest.raises(ValueError):
        extract_slate_frame("../oops", tmp_path / "v.mp4")


def test_near_misses_skip_water_and_keep_a_gap():
    mk = lambda t, fails, paper, strokes: slate.FrameScore(
        t, paper, 0.8, 1.3, 0.7, 0.1, strokes, 0.0, "white", fails=fails
    )  # noqa: E731
    seen = [
        mk(0.0, "water", 0.5, 50),  # underwater: never offered
        mk(1.0, "strokes", 0.3, 3),  # one gate failed
        mk(2.0, "strokes", 0.3, 4),  # same moment as 1 s
        mk(5.0, "edge_frac,strokes", 0.4, 2),
        mk(9.0, "", 0.0, 0),  # no blob at all
        mk(12.0, "aspect", 0.2, 30),  # one gate failed, more writing
    ]
    got = [f.seconds for f in slate.rank_near_misses(seen, k=3)]
    assert got == [12.0, 2.0, 5.0]


def test_spread_frames_skip_water_and_taken_moments():
    mk = lambda t, fails: slate.FrameScore(
        t, 0.2, 0.8, 1.3, 0.7, 0.1, 3, 0.0, "white", fails=fails
    )  # noqa: E731
    seen = [mk(float(s), "strokes") for s in range(0, 20)] + [
        mk(float(s), "water") for s in range(20, 60)
    ]
    got = [f.seconds for f in slate.spread_frames(seen, 2, taken=[5.0])]
    assert all(s < 20 for s in got)  # above water only
    assert all(abs(s - 5.0) >= 2 for s in got)  # not the moment already offered
    assert len(got) == 2
    assert slate.spread_frames([mk(1.0, "water")], 2, taken=[]) == []


def test_random_spread_frames_vary_but_stay_in_their_stretch():
    import random

    mk = lambda t: slate.FrameScore(
        t, 0.2, 0.8, 1.3, 0.7, 0.1, 3, 0.0, "white", fails="strokes"
    )  # noqa: E731
    seen = [mk(float(s)) for s in range(0, 40)]
    draws = {
        tuple(
            f.seconds
            for f in slate.spread_frames(seen, 2, taken=[], rng=random.Random(seed))
        )
        for seed in range(20)
    }
    assert len(draws) > 5  # different rounds, different frames
    assert all(a < 20 <= b for a, b in draws)  # one from each half of the stretch


def test_rejected_seconds_are_never_offered_again(monkeypatch):
    """A rescan after a rejection must not come back with the same frames."""
    frames = [(float(s), _board(_sea())) for s in range(0, 6)]
    monkeypatch.setattr(
        slate, "scan_frames", lambda video, p, max_seconds: iter(frames)
    )
    monkeypatch.setattr(
        slate, "read_full_frames", lambda video, picked: [(f, None) for f in picked]
    )
    p = SlateParams.from_config()
    first = [fs.seconds for fs, _ in slate.find_slate_candidates("v.mp4", p, k=3)]
    again = [
        fs.seconds
        for fs, _ in slate.find_slate_candidates("v.mp4", p, k=3, exclude=tuple(first))
    ]
    assert first and again
    assert not set(first) & set(again)


def test_shaded_board_filling_the_frame_is_not_taken_for_sea(params):
    """Blue-tinted shaded paper (sat ~66) is not water; saturated seabed blue is."""
    fs = score_frame(
        _board(
            np.full((360, 640, 3), SHADED, dtype=np.uint8), x=10, y=10, w=620, h=340
        ),
        params,
    )
    assert "water" not in fs.fails


def test_runner_ups_are_distinct_moments():
    mk = lambda t, s: slate.FrameScore(
        t, 0.3, 0.9, 1.3, 0.8, 0.05, 20, s, "white", 100.0
    )  # noqa: E731
    ranked = [mk(4.0, 9.0), mk(5.0, 8.9), mk(6.0, 8.8), mk(64.0, 5.0), mk(30.0, 4.0)]
    assert [f.seconds for f in slate.distinct_moments(ranked, 5.0)] == [4.0, 64.0, 30.0]


def test_review_scan_does_not_stop_after_early_hits(monkeypatch):
    """A board at 64 s must still be scored when false hits came at 4-6 s."""
    frames = [
        (float(s), _board(_sea()) if s in (4, 5, 6, 64) else _sea())
        for s in range(0, 70)
    ]
    monkeypatch.setattr(
        slate, "scan_frames", lambda video, p, max_seconds: iter(frames)
    )
    monkeypatch.setattr(
        slate, "read_full_frames", lambda video, picked: [(f, None) for f in picked]
    )
    p = SlateParams.from_config()
    quick = [fs.seconds for fs, _ in slate.find_slate_candidates("v.mp4", p, k=3)]
    review = [
        fs.seconds
        for fs, _ in slate.find_slate_candidates("v.mp4", p, k=3, distinct_seconds=5.0)
    ]
    assert 64.0 not in quick
    assert 64.0 in review


def test_saved_slate_is_no_wider_than_save_width(tmp_path, monkeypatch):
    monkeypatch.setitem(config.slate_detection, "save_width", 320)
    out = tmp_path / f"{DROP}_slate.jpg"

    slate._write_slate(out, _board(_sea()))  # a 640 x 360 frame

    assert cv2.imread(str(out)).shape == (180, 320, 3)
