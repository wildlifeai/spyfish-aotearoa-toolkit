"""Both Biigle volume filename conventions must resolve to the same drop_id.

Biigle resolves an image as ``volume.url + "/" + filename``, so a survey-pooled
volume can either point straight at a shared ``training_frames/`` dir and hold
bare basenames (the legacy layout), or point at the survey dir and carry the
per-drop path in the filename (the current one, which makes S3 mirror the local
layout `prepare_training_data` walks).

A volume's url is fixed at creation, so both layouts exist in production
simultaneously and every filename parser has to read either.
"""

from spyfish.biigle.biigle_to_yolo import drop_id_from_frame_filename
from spyfish.config.wrapper import config

DROP = "SLI_20260114_BUV_SLI_044_01"
FRAME = f"{DROP}__frame_412.000s.jpg"


def test_flat_filename_resolves():
    """Legacy layout: volume url is {survey}/training_frames, filename is a basename."""
    assert drop_id_from_frame_filename(FRAME) == DROP


def test_nested_filename_resolves():
    """Current layout: volume url is {survey}, filename carries {drop}/training_frames/."""
    assert drop_id_from_frame_filename(f"{DROP}/training_frames/{FRAME}") == DROP


def test_both_conventions_agree():
    """The whole point: one parser, one answer, regardless of layout."""
    assert drop_id_from_frame_filename(
        f"{DROP}/training_frames/{FRAME}"
    ) == drop_id_from_frame_filename(FRAME)


def test_nested_is_not_parsed_as_part_of_the_drop_id():
    """Regression: a bare split("__frame_")[0] returned the whole leading path.

    That silently produced drop_ids like
    "SLI_..._044_01/training_frames/SLI_..._044_01", which match no deployment
    and no path helper — the frames just vanish from training with no error.
    """
    assert "/" not in drop_id_from_frame_filename(f"{DROP}/training_frames/{FRAME}")


def test_review_and_training_coco_paths_differ():
    """The two workflows select different frames, so their COCOs must not collide.

    Sharing one filename meant whichever extractor ran last destroyed the
    other's record of what the model predicted.
    """
    review = config.get_coco_annotations_path(DROP)
    training = config.get_coco_annotations_path(DROP, target="training")
    assert review != training
    assert review.parent == training.parent
    assert "training" in training.name
