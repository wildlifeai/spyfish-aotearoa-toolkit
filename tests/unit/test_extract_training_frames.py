"""Unit tests for spyfish.ml.training.extract_training_frames.

Selection, spacing and timestamp generation now live in
`spyfish/extraction/select_frames.py` and are tested in
`test_frame_buckets.py` — this module keeps only what is still
training-specific.
"""

from __future__ import annotations

DROP_ID = "KSF_20240124_BUV_KSF_085_01"


class TestPerFrameCsvName:
    def test_name_carries_the_model_version_not_just_the_kind(self, monkeypatch):
        from pathlib import Path

        import spyfish.ml.training.extract_training_frames as mod

        monkeypatch.setattr(
            mod.config,
            "get_pipeline_model",
            lambda kind: Path(f"/models/{kind}_20260429_081503.pt"),
        )
        name = mod._per_frame_csv_name(DROP_ID, "species")
        assert name == f"{DROP_ID}_species_20260429_081503_raw.csv"

    def test_promoting_a_new_model_changes_the_name(self, monkeypatch):
        from pathlib import Path

        import spyfish.ml.training.extract_training_frames as mod

        monkeypatch.setattr(
            mod.config, "get_pipeline_model", lambda kind: Path("/models/species_v1.pt")
        )
        old = mod._per_frame_csv_name(DROP_ID, "species")
        monkeypatch.setattr(
            mod.config, "get_pipeline_model", lambda kind: Path("/models/species_v2.pt")
        )
        assert mod._per_frame_csv_name(DROP_ID, "species") != old
