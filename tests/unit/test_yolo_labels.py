"""The shared YOLO label / data.yaml helpers. Each remap mode is the exact
behaviour of one caller that used to carry its own copy of the loop."""

from collections import Counter

import pytest
import yaml

from spyfish.ml.training import rollup, yolo_labels
from spyfish.ml.training.yolo_labels import (
    SPLITS,
    iter_class_ids,
    read_class_names,
    read_class_names_by_id,
    read_label_rows,
    remap_label_file,
    write_class_names,
)

# Two boxes of class 3, one of class 0, a blank line, a row with a non-integer
# id, and a truncated row (class 7, three fields).
MESSY = "3 .5 .5 .1 .1\n\n0 .2 .2 .1 .1\nx .1 .1 .1 .1\n7 .1 .1\n3 .6 .6 .1 .1\n"


@pytest.fixture
def label(tmp_path):
    p = tmp_path / "frame.txt"
    p.write_text(MESSY)
    return p


def test_splits_has_one_definition():
    assert SPLITS == ("train", "val", "test")
    assert rollup.SPLITS is yolo_labels.SPLITS


def test_iter_class_ids_skips_blank_and_non_integer_rows(label):
    assert list(iter_class_ids(label)) == [3, 0, 7, 3]
    # boxes_only also leaves out the truncated row
    assert list(iter_class_ids(label, boxes_only=True)) == [3, 0, 3]


def test_read_label_rows_keeps_fields_as_strings(label):
    assert read_label_rows(label)[0] == ["3", ".5", ".5", ".1", ".1"]
    assert [r[0] for r in read_label_rows(label)] == ["3", "0", "x", "3"]
    assert [r[0] for r in read_label_rows(label, boxes_only=False)] == [
        "3",
        "0",
        "x",
        "7",
        "3",
    ]


def test_remap_drop_writes_to_dst_and_drops_unmapped_and_none(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("3 .5 .5 .1 .1\n0 .2 .2 .1 .1\n4 .3 .3 .1 .1\n7 .1 .1\n")
    dst = tmp_path / "dst.txt"
    seen = remap_label_file(src, {3: 1, 4: None}, dst=dst, missing="drop")
    # 0 is not in the map, 4 maps to None, the 3-field row is not a box
    assert dst.read_text() == "1 .5 .5 .1 .1"
    assert src.read_text().startswith("3 ")
    assert seen == Counter({3: 1, 0: 1, 4: 1})


def test_remap_keep_rewrites_in_place_and_leaves_other_ids(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("3 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")
    remap_label_file(p, {3: 9}, missing="keep")
    assert p.read_text() == "9 .5 .5 .1 .1\n0 .2 .2 .1 .1"


def test_remap_raise_on_an_id_outside_the_map(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("3 .5 .5 .1 .1\n")
    with pytest.raises(KeyError):
        remap_label_file(p, {0: 0}, missing="raise")
    with pytest.raises(ValueError):
        remap_label_file(p, {0: 0}, missing="ignore")


def test_remap_non_integer_id_raises_in_every_mode(label):
    for mode in ("drop", "keep", "raise"):
        with pytest.raises(ValueError):
            remap_label_file(label, {3: 1, 0: 0}, missing=mode)


def test_remap_trailing_newline_only_when_there_are_rows(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("3 .5 .5 .1 .1\n")
    remap_label_file(p, {3: 1}, trailing_newline=True)
    assert p.read_text() == "1 .5 .5 .1 .1\n"
    remap_label_file(p, {}, trailing_newline=True)
    assert p.read_text() == ""


def test_remap_in_place_writes_through_a_symlink(tmp_path):
    """Assembled datasets symlink labels to the staged tree: the link must
    survive and the staged file must change."""
    real = tmp_path / "staged.txt"
    real.write_text("3 .5 .5 .1 .1\n")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    remap_label_file(link, {3: 1}, missing="keep")
    assert link.is_symlink()
    assert real.read_text() == "1 .5 .5 .1 .1"


def test_read_class_names_from_list_and_from_mapping(tmp_path):
    as_list = tmp_path / "list.yaml"
    as_list.write_text(yaml.safe_dump({"names": ["fish", "Parapercis colias"]}))
    as_map = tmp_path / "map.yaml"
    as_map.write_text(yaml.safe_dump({"names": {1: "Parapercis colias", 0: "fish"}}))
    for p in (as_list, as_map):
        assert read_class_names(p) == ["fish", "Parapercis colias"]
        assert read_class_names_by_id(p) == {0: "fish", 1: "Parapercis colias"}


def test_read_class_names_without_names_key(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"nc": 0}))
    assert read_class_names(p) == []
    with pytest.raises(KeyError):
        read_class_names(p, required=True)


def test_write_class_names_keeps_other_keys_and_their_order(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(
        yaml.safe_dump(
            {"nc": 1, "names": ["fish"], "train": "/t", "val": "/v"}, sort_keys=False
        )
    )
    write_class_names(p, ["fish", "Parapercis colias"])
    data = yaml.safe_load(p.read_text())
    assert list(data) == ["nc", "names", "train", "val"]
    assert data["nc"] == 2 and data["names"] == ["fish", "Parapercis colias"]
