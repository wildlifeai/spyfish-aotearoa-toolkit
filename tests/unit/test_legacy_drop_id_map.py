"""Legacy expert rows filed under a DropID SharePoint does not have get renamed on ingest."""

import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.orchestrator.legacy_extract import (
    apply_legacy_drop_id_map,
    load_legacy_drop_id_map,
)

OLD, NEW = "SLI_20220228_BUV_SLI_038_02", "SLI_20220228_BUV_SLI_038_01"
OTHER = "SLI_20220228_BUV_SLI_003_01"


def test_mapped_ids_are_renamed_others_untouched():
    df = pd.DataFrame({config.drop_id_column: [OLD, OLD, OTHER]})
    out = apply_legacy_drop_id_map(df, {OLD: NEW})
    assert out[config.drop_id_column].tolist() == [NEW, NEW, OTHER]
    assert df[config.drop_id_column].tolist() == [OLD, OLD, OTHER]  # input not mutated


def test_map_file_is_validated(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text(f"{OLD}: {NEW}\n")
    assert load_legacy_drop_id_map(p) == {OLD: NEW}
    p.write_text(f"{OLD}: ../oops\n")
    with pytest.raises(ValueError):
        load_legacy_drop_id_map(p)


def test_missing_map_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_legacy_drop_id_map(tmp_path / "absent.yaml")


def test_empty_map_file_is_fine(tmp_path):
    p = tmp_path / "map.yaml"
    p.write_text("# nothing yet\n")
    assert load_legacy_drop_id_map(p) == {}
