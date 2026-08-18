"""Tests for substrate percent-cover parsing in BiigleParser.process_substrate.

Substrate (CMECS) annotations are measured as area cover, not counted as a
species. The key behaviours locked in here:

- A magic-wand Polygon and a traced (unclosed) LineString are measured the same
  way — both close into a ring via the shoelace formula.
- Substrate is identified by label-TREE membership (a row's `label_id` is in the
  substrate tree's labels), NOT by shape — so a fish-SIZE LineString (species /
  "Scale bar" label, a different tree) is excluded and never pollutes the output.
- `pct_of_annotated` (primary) sums to 100% per image; `pct_of_image` uses the
  per-row `attributes` width×height as the denominator.
"""

from __future__ import annotations

import json

import pandas as pd

from spyfish.biigle.biigle_parser import BiigleParser
from spyfish.config.wrapper import config

DROP_ID = "KSF_20240124_BUV_KSF_085_01"
FRAME = f"{DROP_ID}__frame_10.0s.jpg"
# 1000x1000 canvas → area 1,000,000 px. Each substrate square below is
# 500x500 = 250,000 px = 25% of the image, 50% of the annotated area.
_ATTRS = json.dumps({"mimetype": "image/jpeg", "width": 1000, "height": 1000})
# label_ids belonging to the (mocked) substrate label tree.
_SUBSTRATE_IDS = {9001, 9002}


def _row(label_id, label_name, shape_name, points):
    return {
        config.drop_id_column: DROP_ID,
        "filename": FRAME,
        "label_id": label_id,
        "label_name": label_name,
        "shape_name": shape_name,
        # Biigle reports store `points` as a JSON string after CSV round-trip.
        "points": json.dumps(points),
        "attributes": _ATTRS,
    }


def _substrate_df():
    return pd.DataFrame(
        [
            # Magic-wand Polygon (substrate tree): closed square (0,0)-(500,500).
            _row(9001, "Sand", "Polygon", [0, 0, 500, 0, 500, 500, 0, 500]),
            # Traced LineString (substrate tree), UNCLOSED square (500,0)-(1000,500).
            # Same area as the polygon once the shoelace auto-closes the ring.
            _row(9002, "Reef", "LineString", [500, 0, 1000, 0, 1000, 500, 500, 500]),
            # Fish-SIZE LineString — species tree label, NOT substrate. Excluded.
            _row(477416, "Snapper - Pagrus auratus", "LineString", [10, 10, 200, 10]),
            # Scale bar size line — also excluded.
            _row(531298, "Scale bar", "LineString", [0, 0, 100, 0]),
        ]
    )


def test_shoelace_area_of_unit_square():
    parser = BiigleParser.__new__(BiigleParser)  # no API handler needed
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert parser.shoelace_area(pts) == 100.0


def test_shoelace_area_degenerate_returns_zero():
    parser = BiigleParser.__new__(BiigleParser)
    assert parser.shoelace_area([(0.0, 0.0), (10.0, 0.0)]) == 0.0


def test_parse_geometry_points_unwraps_video_keyframe():
    parser = BiigleParser.__new__(BiigleParser)
    # Nested single-keyframe shape (video report) unwraps to the same pairs.
    assert parser.parse_geometry_points([[0, 0, 10, 0, 10, 10]]) == [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
    ]
    # Multi-keyframe (moving) shapes can't be reduced to one ring.
    assert parser.parse_geometry_points([[0, 0], [1, 1]]) == []


def test_process_substrate_excludes_fish_and_computes_percentages():
    parser = BiigleParser.__new__(BiigleParser)
    result = parser.process_substrate(_substrate_df(), _SUBSTRATE_IDS)

    # Only the two substrate rows survive — fish size lines are gone.
    assert set(result["substrate"]) == {"Sand", "Reef"}

    # Normalised cover sums to 100% across the image.
    assert result["pct_of_annotated"].sum() == 100.0
    assert (result["pct_of_annotated"] == 50.0).all()

    # Each 250,000px patch is 25% of the 1,000,000px image.
    assert (result["pct_of_image"] == 25.0).all()
    assert (result["area_px"] == 250000.0).all()
    assert (result["image_area_px"] == 1000000.0).all()


def test_process_substrate_derives_drop_id_from_filename():
    """The raw Biigle report has no DropID column — it must be derived from the
    image filename, or the groupby would KeyError in the real sync path."""
    parser = BiigleParser.__new__(BiigleParser)
    df = _substrate_df().drop(columns=[config.drop_id_column])
    result = parser.process_substrate(df, _SUBSTRATE_IDS)
    assert (result[config.drop_id_column] == DROP_ID).all()
    assert set(result["substrate"]) == {"Sand", "Reef"}


def test_process_substrate_empty_when_no_substrate():
    parser = BiigleParser.__new__(BiigleParser)
    fish_only = pd.DataFrame(
        [
            _row(
                477416,
                "Snapper - Pagrus auratus",
                "Rectangle",
                [0, 0, 10, 0, 10, 10, 0, 10],
            )
        ]
    )
    assert parser.process_substrate(fish_only, _SUBSTRATE_IDS).empty
