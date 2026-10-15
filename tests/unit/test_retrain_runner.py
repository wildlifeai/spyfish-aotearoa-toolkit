"""
Unit tests for retrain_runner helpers.

Focused on the promotion filename derivation and the archive step —
narrow, deterministic, no YOLO or training pipeline involved.
"""

import re

from spyfish.orchestrator.retrain_runner import _derive_promoted_filename


def test_promoted_filename_uses_training_run_directory():
    """Happy path: `.../runs/{type}_{timestamp}/weights/best.pt` yields
    `{type}_{timestamp}.pt`, the model name standard."""
    model_path = "/tmp/training/runs/binary_20260411_142301/weights/best.pt"
    assert (
        _derive_promoted_filename(model_path, "binary") == "binary_20260411_142301.pt"
    )


def test_promoted_filename_converts_a_pre_standard_run_directory():
    """Run dirs from before the standard were `{timestamp}_{type}`; the
    training stamp is kept and the type moved to the front."""
    model_path = "/tmp/training/runs/20260411_142301_binary/weights/best.pt"
    assert (
        _derive_promoted_filename(model_path, "binary") == "binary_20260411_142301.pt"
    )


def test_promoted_filename_species_variant():
    model_path = "/tmp/training/runs/species_20260413_090000/weights/best.pt"
    assert (
        _derive_promoted_filename(model_path, "species") == "species_20260413_090000.pt"
    )


def test_promoted_filename_fallback_when_path_unexpected():
    """If the path doesn't contain the model_type in the grandparent dir name,
    we fall back to `{model_type}_{now}.pt`: still a name the standard accepts,
    stamped with the promotion time."""
    # Weird path — no training/runs structure
    model_path = "/some/random/place/model.pt"
    result = _derive_promoted_filename(model_path, "binary")
    assert re.fullmatch(r"binary_20\d{6}_\d{6}\.pt", result)


def test_promoted_filename_fallback_when_path_is_short():
    """Short path (no grandparent) should not crash — falls back cleanly."""
    model_path = "model.pt"
    result = _derive_promoted_filename(model_path, "binary")
    assert re.fullmatch(r"binary_20\d{6}_\d{6}\.pt", result)
