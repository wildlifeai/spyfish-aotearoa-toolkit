"""The model name standard: {type}_{YYYYMMDD}[_{HHMMSS}], type first, stamp last."""

import pytest

from spyfish.config.wrapper import config


@pytest.mark.parametrize(
    "name, stamp",
    [
        ("species_20260822_192536", "20260822192536"),
        ("species_20260603", "20260603000000"),
        ("cfd_binary_water_20260301", "20260301000000"),
        ("binary_cfd_sweep_20260427_002337", "20260427002337"),
    ],
)
def test_canonical_names_yield_their_stamp(name, stamp):
    assert config.model_name_stamp(name) == stamp
    assert config.validate_model_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "20260429_081503_species",  # stamp first
        "promoted_species",  # no stamp
        "best",  # a bare YOLO export
        "species_20260822_192536_v2",  # something after the stamp
        "Species_20260822_192536",  # capitals
        "",
    ],
)
def test_non_standard_names_are_refused(name):
    assert config.model_name_stamp(name) is None
    with pytest.raises(ValueError, match="does not follow the standard"):
        config.validate_model_name(name)


def test_pipeline_model_path_picks_the_newest_stamp_not_the_newest_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        type(config), "pipeline_model_dir", property(lambda self: tmp_path)
    )
    old = tmp_path / "species_20260301_000000.pt"
    new = tmp_path / "species_20260822_192536.pt"
    new.write_bytes(b"x")
    old.write_bytes(b"x")  # written last, so the newest mtime is the OLD model
    assert config.pipeline_model_path == new


def test_pipeline_model_path_refuses_a_directory_with_only_random_names(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        type(config), "pipeline_model_dir", property(lambda self: tmp_path)
    )
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "20260429_081503_species.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="best.pt"):
        config.pipeline_model_path
