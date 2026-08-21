"""Unit tests for spyfish.ml.training.extract_training_frames.

Selection, spacing and timestamp generation now live in
`spyfish/extraction/select_frames.py` and are tested in
`test_frame_buckets.py` — this module keeps only what is still
training-specific.
"""

from __future__ import annotations

DROP_ID = "KSF_20240124_BUV_KSF_085_01"


def _patch_pipeline_model(monkeypatch, path):
    """Point config.pipeline_model_path (a property) at a fake weights file."""
    import spyfish.ml.training.extract_training_frames as mod

    monkeypatch.setattr(
        type(mod.config), "pipeline_model_path", property(lambda self: path)
    )
    return mod


class TestPerFrameCsvName:
    def test_name_carries_the_model_version(self, monkeypatch):
        from pathlib import Path

        mod = _patch_pipeline_model(
            monkeypatch, Path("/models/species_20260429_081503.pt")
        )
        name = mod._per_frame_csv_name(DROP_ID)
        assert name == f"{DROP_ID}_species_20260429_081503_raw.csv"

    def test_promoting_a_new_model_changes_the_name(self, monkeypatch):
        from pathlib import Path

        mod = _patch_pipeline_model(monkeypatch, Path("/models/species_v1.pt"))
        old = mod._per_frame_csv_name(DROP_ID)
        _patch_pipeline_model(monkeypatch, Path("/models/species_v2.pt"))
        assert mod._per_frame_csv_name(DROP_ID) != old
