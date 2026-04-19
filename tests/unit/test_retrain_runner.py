"""
Unit tests for retrain_runner helpers.

Focused on the promotion filename derivation and the archive step —
narrow, deterministic, no YOLO or training pipeline involved.
"""

from spyfish.orchestrator.retrain_runner import _derive_promoted_filename


def test_promoted_filename_uses_training_run_directory():
    """Happy path: path like `.../runs/{timestamp}_{type}/weights/best.pt`
    yields `{timestamp}_{type}.pt` — so every promotion is uniquely named."""
    model_path = "/tmp/training/runs/20260411_142301_binary/weights/best.pt"
    assert (
        _derive_promoted_filename(model_path, "binary") == "20260411_142301_binary.pt"
    )


def test_promoted_filename_species_variant():
    model_path = "/tmp/training/runs/20260413_090000_species/weights/best.pt"
    assert (
        _derive_promoted_filename(model_path, "species") == "20260413_090000_species.pt"
    )


def test_promoted_filename_fallback_when_path_unexpected():
    """If the path doesn't contain the model_type in the grandparent dir name,
    we fall back to the fixed `promoted_{model_type}.pt` rather than producing
    a misleading filename."""
    # Weird path — no training/runs structure
    model_path = "/some/random/place/model.pt"
    result = _derive_promoted_filename(model_path, "binary")
    assert result == "promoted_binary.pt"


def test_promoted_filename_fallback_when_path_is_short():
    """Short path (no grandparent) should not crash — falls back cleanly."""
    model_path = "model.pt"
    result = _derive_promoted_filename(model_path, "binary")
    assert result == "promoted_binary.pt"
