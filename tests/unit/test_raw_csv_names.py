"""config.parse_ml_raw_csv_name / list_ml_raw_csvs: which files are ML raw CSVs."""

from spyfish.config.wrapper import config

DROP = "KSF_20240124_BUV_KSF_085_01"


def test_parse_reads_back_what_get_raw_csv_path_writes():
    name = config.get_raw_csv_path(DROP, "species_20260822_192536").name
    assert config.parse_ml_raw_csv_name(name) == (DROP, "species_20260822_192536")
    # an S3 key or a full path works the same way
    assert config.parse_ml_raw_csv_name(f"deployment_data/x/{name}") == (
        DROP,
        "species_20260822_192536",
    )


def test_parse_rejects_the_other_raw_csvs():
    """They share the suffix but are not model output."""
    for other in (
        f"{DROP}_zooniverse_raw.csv",
        f"{DROP}_zooniverse_frames_raw.csv",
        f"{DROP}_biigle_expert_raw.csv",
        f"{DROP}_biigle_training_raw.csv",
        f"{DROP}_species_20260822_192536_conf050_raw.csv",
        f"{DROP}_species_20260822_192536_maxn.csv",
        "volume_30505_biigle_expert_raw.csv",
    ):
        assert config.parse_ml_raw_csv_name(other) is None, other


def test_list_ml_raw_csvs_keeps_models_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "get_drop_annotations_dir", lambda self, drop_id: tmp_path
    )
    for stem in (
        "species_20260601",
        "species_20260801",
        "zooniverse",
        "biigle_expert",
        "zooniverse_frames",
    ):
        (tmp_path / f"{DROP}_{stem}_raw.csv").write_text("frame\n")

    assert config.list_ml_raw_csvs(DROP) == {
        "species_20260601": tmp_path / f"{DROP}_species_20260601_raw.csv",
        "species_20260801": tmp_path / f"{DROP}_species_20260801_raw.csv",
    }
