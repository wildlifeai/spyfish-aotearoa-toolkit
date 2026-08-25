"""Selection-reason strings must survive their parameters on the way to Biigle.

`SelectionReason` is written for humans reading a selections CSV, so every
bucket except "Video Start" embeds parameters: species names, a size band
index, a confidence threshold and count. Biigle needs a fixed label ID. The
risk this file guards is silent drift, a retuned threshold or a new species
changing the tail of the string and quietly turning matched frames into
unlabelled ones, so the reasons are asserted in the exact f-string forms
select_frames.py produces.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spyfish.biigle import selection_reason as sr
from spyfish.biigle.biigle_to_yolo import drop_id_from_frame_filename
from spyfish.biigle.selection_reason import canonical_reason, reason_label_id
from spyfish.biigle.upload_frames import attach_selection_reason_labels

DROP = "KSF_20240124_BUV_KSF_085_01"

# The literal shapes emitted by select_frames.py, parameters and all.
REAL_REASONS = [
    ("MaxN (Parapercis colias)", "maxn_peak"),
    ("MaxN (Parapercis colias, Pagrus auratus)", "maxn_peak"),
    ("Uncertain (Pagrus auratus)", "uncertain_id"),
    ("ML peak (conf>=0.25, count=4)", "ml_peak"),
    ("ML peak (conf>=0.4, count=11)", "ml_peak"),
    ("Fish size band 1", "fish_variety"),
    ("Fish size band 3", "fish_variety"),
    ("Blind (False Negative Check)", "spot_check"),
    ("Video Start", "video_start"),
]


# ── canonical_reason ─────────────────────────────────────────────────────


@pytest.mark.parametrize("reason,expected", REAL_REASONS)
def test_real_reason_strings_map_to_their_bucket(reason, expected):
    assert canonical_reason(reason) == expected


def test_every_bucket_is_reachable():
    """No rule is dead code, and no two reasons collapse onto one key."""
    keys = {canonical_reason(r) for r, _ in REAL_REASONS}
    assert keys == {key for key, _ in sr._REASON_RULES}


@pytest.mark.parametrize(
    "reason",
    [
        "MaxN (Anguilla australis)",  # species never seen before
        "ML peak (conf>=0.9, count=137)",  # retuned threshold, huge count
        "Fish size band 12",  # more bands configured
    ],
)
def test_parameters_do_not_affect_matching(reason):
    """The whole point: only the head is matched, the tail is free to change."""
    assert canonical_reason(reason) is not None


@pytest.mark.parametrize("reason", [None, "", "   ", float("nan") and None])
def test_missing_reason_is_none_not_a_crash(reason):
    assert canonical_reason(reason) is None


def test_unknown_bucket_returns_none_rather_than_guessing():
    """A new bucket must go unlabelled, not inherit a neighbour's label."""
    assert canonical_reason("Substrate sample") is None


# ── legacy spellings ─────────────────────────────────────────────────────
#
# Not hypothetical. `upsert_selections` merges each prior pass's selections CSV
# into the fresh one, so reasons written by older code survive on disk and get
# re-uploaded on a second pass. There are rows reading "Confusing (fish)" in
# the repo right now, from before that bucket was renamed to "Uncertain".
# select_clips.py still emits the older vocabulary for the citsci clip path.


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("Confusing (fish)", "uncertain_id"),  # on disk today
        ("Confusing (High count, low conf)", "uncertain_id"),
        ("Absolute MaxN", "maxn_peak"),
        ("Empty (False Negative Check)", "spot_check"),
        ("Global Empty", "spot_check"),
        ("Global Video Start", "video_start"),
    ],
)
def test_legacy_spellings_map_to_the_same_bucket(reason, expected):
    assert canonical_reason(reason) == expected


def test_matching_is_case_insensitive():
    assert canonical_reason("video start") == "video_start"


# ── reason_label_id ──────────────────────────────────────────────────────


def test_label_id_resolves_through_config():
    with patch.object(
        type(sr.config),
        "selection_reason_label_ids",
        new_callable=lambda: property(lambda self: {"maxn_peak": 555001}),
    ):
        assert reason_label_id("MaxN (Parapercis colias)") == 555001


def test_bucket_without_a_configured_label_is_skipped():
    """Rolling out one bucket at a time must not raise on the others."""
    with patch.object(
        type(sr.config),
        "selection_reason_label_ids",
        new_callable=lambda: property(lambda self: {"maxn_peak": 555001}),
    ):
        assert reason_label_id("Video Start") is None


# ── attach_selection_reason_labels ───────────────────────────────────────


def _frames_df(rows):
    return pd.DataFrame(rows)


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_attaches_one_label_per_frame(mock_handler_cls):
    handler = MagicMock()
    handler.attach_image_labels.return_value = 2
    mock_handler_cls.return_value = handler

    frames = _frames_df(
        [
            {
                "FramePath": f"/data/{DROP}/frames/{DROP}__frame_0012.500s.jpg",
                "SelectionReason": "MaxN (Parapercis colias)",
            },
            {
                "FramePath": f"/data/{DROP}/frames/{DROP}__frame_0099.000s.jpg",
                "SelectionReason": "Blind (False Negative Check)",
            },
        ]
    )
    name_map = {
        f"{DROP}__frame_0012.500s.jpg": 111,
        f"{DROP}__frame_0099.000s.jpg": 222,
    }

    with (
        patch.object(
            type(sr.config),
            "selection_reason_label_ids",
            new_callable=lambda: property(
                lambda self: {"maxn_peak": 900, "spot_check": 901}
            ),
        ),
        patch.object(
            type(sr.config),
            "selection_reason_include_species",
            new_callable=lambda: property(lambda self: False),
        ),
    ):
        attached = attach_selection_reason_labels(7, frames, name_map)

    assert attached == 2
    pairs = handler.attach_image_labels.call_args[0][0]
    assert sorted(pairs) == [(111, 900), (222, 901)]


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_no_configured_labels_is_a_noop_with_no_api_calls(mock_handler_cls):
    """Before the labels exist in Biigle the feature must stay entirely off."""
    frames = _frames_df(
        [{"FramePath": "/f/a.jpg", "SelectionReason": "MaxN (Parapercis colias)"}]
    )
    with patch.object(
        type(sr.config),
        "selection_reason_label_ids",
        new_callable=lambda: property(lambda self: {}),
    ):
        assert attach_selection_reason_labels(7, frames, {}) == 0
    mock_handler_cls.assert_not_called()


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_extraction_failures_are_skipped(mock_handler_cls):
    """Rows with no FramePath were never uploaded, so nothing to label."""
    handler = MagicMock()
    handler.attach_image_labels.return_value = 1
    mock_handler_cls.return_value = handler

    frames = _frames_df(
        [
            {"FramePath": None, "SelectionReason": "MaxN (Parapercis colias)"},
            {"FramePath": "/f/b.jpg", "SelectionReason": "Video Start"},
        ]
    )
    with (
        patch.object(
            type(sr.config),
            "selection_reason_label_ids",
            new_callable=lambda: property(
                lambda self: {"maxn_peak": 900, "video_start": 902}
            ),
        ),
        patch.object(
            type(sr.config),
            "selection_reason_include_species",
            new_callable=lambda: property(lambda self: False),
        ),
    ):
        attach_selection_reason_labels(7, frames, {"b.jpg": 222})

    assert handler.attach_image_labels.call_args[0][0] == [(222, 902)]


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_labelling_failure_never_fails_the_upload(mock_handler_cls):
    """Frames and boxes have already landed; a label problem must not raise."""
    handler = MagicMock()
    handler.attach_image_labels.side_effect = RuntimeError("Biigle 500")
    mock_handler_cls.return_value = handler

    frames = _frames_df([{"FramePath": "/f/a.jpg", "SelectionReason": "Video Start"}])
    with patch.object(
        type(sr.config),
        "selection_reason_label_ids",
        new_callable=lambda: property(lambda self: {"video_start": 902}),
    ):
        assert attach_selection_reason_labels(7, frames, {"a.jpg": 111}) == 0


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_survey_pooled_names_join_on_basename(mock_handler_cls):
    """Pooled volumes register {drop}/frames/{base}; frames_df has local paths."""
    handler = MagicMock()
    handler.attach_image_labels.return_value = 1
    mock_handler_cls.return_value = handler

    frames = _frames_df(
        [
            {
                "FramePath": f"/local/{DROP}/frames/{DROP}__frame_0005.000s.jpg",
                "SelectionReason": "Fish size band 2",
            }
        ]
    )
    name_map = {f"{DROP}/frames/{DROP}__frame_0005.000s.jpg": 333}

    with patch.object(
        type(sr.config),
        "selection_reason_label_ids",
        new_callable=lambda: property(lambda self: {"fish_variety": 903}),
    ):
        attach_selection_reason_labels(7, frames, name_map)

    assert handler.attach_image_labels.call_args[0][0] == [(333, 903)]


# ── backfill filename reconstruction ─────────────────────────────────────


def test_backfill_regenerates_the_exact_uploaded_filename():
    """The backfill's whole join rests on this.

    Selections CSVs store timestamps, not frame paths (the extractor adds
    FramePath to its DataFrame but never writes it back), so the backfill
    rebuilds the name from the timestamp. If it drifts from what the extractor
    wrote, every image silently fails to match and nothing gets labelled.
    """
    from spyfish.utils import generate_frame_filename

    for seconds in (5.0, 12.5, 99.0, 1234.567):
        rebuilt = generate_frame_filename(DROP, seconds)
        assert rebuilt.startswith(f"{DROP}__frame_")
        assert rebuilt.endswith("s.jpg")
        # Zero-padded so BIIGLE's lexicographic file listing matches time order.
        assert drop_id_from_frame_filename(rebuilt) == DROP


def test_backfill_filenames_sort_in_time_order():
    """Padding is load-bearing for the expert's reading order in the volume."""
    from spyfish.utils import generate_frame_filename

    times = [5.0, 12.5, 99.0, 1234.567]
    names = [generate_frame_filename(DROP, t) for t in times]
    assert names == sorted(names)


# ── species carried by a reason ──────────────────────────────────────────


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("MaxN (Parapercis colias)", ["Parapercis colias"]),
        # collapse_peaks merges peaks landing on one frame, so a reason can
        # name several species and each earns its own image label.
        (
            "MaxN (Parapercis colias, Pagrus auratus)",
            ["Parapercis colias", "Pagrus auratus"],
        ),
        ("Uncertain (Pagrus auratus)", ["Pagrus auratus"]),
        ("MaxN (fish)", ["fish"]),  # the catch-all is a legitimate class
        ("Confusing (fish)", ["fish"]),  # legacy spelling still yields species
    ],
)
def test_species_parsed_from_reason(reason, expected):
    assert sr.species_in_reason(reason) == expected


@pytest.mark.parametrize(
    "reason",
    [
        "ML peak (conf>=0.25, count=4)",  # parenthetical is NOT species
        "Blind (False Negative Check)",  # nor here
        "Fish size band 2",
        "Video Start",
        None,
    ],
)
def test_non_species_buckets_yield_no_species(reason):
    """Parsing these parentheticals as species would invent label lookups."""
    assert sr.species_in_reason(reason) == []


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_species_label_attached_alongside_reason(mock_handler_cls):
    handler = MagicMock()
    handler.attach_image_labels.return_value = 2
    mock_handler_cls.return_value = handler

    frames = _frames_df(
        [
            {
                "FramePath": f"{DROP}__frame_0012.500s.jpg",
                "SelectionReason": "MaxN (Parapercis colias)",
            }
        ]
    )
    with (
        patch.object(
            type(sr.config),
            "selection_reason_label_ids",
            new_callable=lambda: property(lambda self: {"maxn_peak": 900}),
        ),
        patch.object(
            type(sr.config),
            "selection_reason_include_species",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "spyfish.biigle.upload_frames.resolve_species_label_id", return_value=477318
        ),
    ):
        attach_selection_reason_labels(7, frames, {f"{DROP}__frame_0012.500s.jpg": 111})

    pairs = handler.attach_image_labels.call_args[0][0]
    assert sorted(pairs) == [(111, 900), (111, 477318)] or sorted(pairs) == sorted(
        [(111, 900), (111, 477318)]
    )


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_species_labels_can_be_turned_off(mock_handler_cls):
    handler = MagicMock()
    handler.attach_image_labels.return_value = 1
    mock_handler_cls.return_value = handler

    frames = _frames_df(
        [
            {
                "FramePath": f"{DROP}__frame_0012.500s.jpg",
                "SelectionReason": "MaxN (Parapercis colias)",
            }
        ]
    )
    with (
        patch.object(
            type(sr.config),
            "selection_reason_label_ids",
            new_callable=lambda: property(lambda self: {"maxn_peak": 900}),
        ),
        patch.object(
            type(sr.config),
            "selection_reason_include_species",
            new_callable=lambda: property(lambda self: False),
        ),
    ):
        attach_selection_reason_labels(7, frames, {f"{DROP}__frame_0012.500s.jpg": 111})

    assert handler.attach_image_labels.call_args[0][0] == [(111, 900)]


@patch("spyfish.biigle.upload_frames.BiigleHandler")
def test_two_species_peaking_in_one_frame_collapse_to_one_image(mock_handler_cls):
    """Two selection rows, same timestamp, one physical frame, no duplicates."""
    handler = MagicMock()
    handler.attach_image_labels.return_value = 3
    mock_handler_cls.return_value = handler

    name = f"{DROP}__frame_0012.500s.jpg"
    frames = _frames_df(
        [
            {"FramePath": name, "SelectionReason": "MaxN (Parapercis colias)"},
            {"FramePath": name, "SelectionReason": "MaxN (Pagrus auratus)"},
        ]
    )
    species = {"Parapercis colias": 477318, "Pagrus auratus": 477319}

    with (
        patch.object(
            type(sr.config),
            "selection_reason_label_ids",
            new_callable=lambda: property(lambda self: {"maxn_peak": 900}),
        ),
        patch.object(
            type(sr.config),
            "selection_reason_include_species",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "spyfish.biigle.upload_frames.resolve_species_label_id",
            side_effect=lambda n, c=None: species[n],
        ),
    ):
        attach_selection_reason_labels(7, frames, {name: 111})

    pairs = handler.attach_image_labels.call_args[0][0]
    # One image, three distinct labels: the shared reason plus both species.
    assert {p[0] for p in pairs} == {111}
    assert sorted(p[1] for p in pairs) == [900, 477318, 477319]
