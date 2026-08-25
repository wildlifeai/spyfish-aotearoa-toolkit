"""The production comparison must refuse a checkpoint whose class list does not
match the dataset.

Run 8598080 (2026-08-23) died with `KeyError: 16` comparing a 20-class
production model against a 50-class data.yaml. Ultralytics drops ids it has no
name for and then indexes past the end of the shortened dict. Even without the
crash the numbers are meaningless: the same id names a different species on
each side.
"""

from unittest.mock import MagicMock, patch

import yaml

from spyfish.ml.training.evaluate import _class_list_mismatch, compare_with_production


def _yaml(tmp_path, names):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"nc": len(names), "names": names}))
    return str(p)


def _model(names):
    m = MagicMock()
    m.names = dict(enumerate(names))
    return m


ROSTER = ["fish", "Pagrus auratus", "Parapercis colias"]


@patch("ultralytics.YOLO")
def test_identical_lists_are_no_mismatch(mock_yolo, tmp_path):
    mock_yolo.return_value = _model(ROSTER)
    assert _class_list_mismatch("m.pt", _yaml(tmp_path, ROSTER)) is None


@patch("ultralytics.YOLO")
def test_extra_dataset_classes_are_caught(mock_yolo, tmp_path):
    """The exact 8598080 shape: dataset has ids the model cannot name."""
    mock_yolo.return_value = _model(ROSTER)
    msg = _class_list_mismatch("m.pt", _yaml(tmp_path, ROSTER + ["Squalus acanthias"]))
    assert msg and "unknown to the model" in msg


@patch("ultralytics.YOLO")
def test_same_length_different_order_is_caught(mock_yolo, tmp_path):
    """Alphabetical vs frozen-roster order: same count, every id means something else."""
    mock_yolo.return_value = _model(["Parapercis colias", "Pagrus auratus", "fish"])
    msg = _class_list_mismatch("m.pt", _yaml(tmp_path, ROSTER))
    assert msg and "different species" in msg


@patch("spyfish.ml.training.evaluate.evaluate_model")
@patch("ultralytics.YOLO")
def test_compare_skips_and_refuses_promotion_on_mismatch(
    mock_yolo, mock_eval, tmp_path
):
    """A mismatch must not promote, and must never reach model.val()."""
    mock_yolo.return_value = _model(ROSTER)
    prod = tmp_path / "prod.pt"
    prod.write_text("x")

    metrics, promote = compare_with_production(
        {"mAP50": 0.9}, str(prod), _yaml(tmp_path, ROSTER + ["Squalus acanthias"])
    )

    assert metrics == {} and promote is False
    mock_eval.assert_not_called()
