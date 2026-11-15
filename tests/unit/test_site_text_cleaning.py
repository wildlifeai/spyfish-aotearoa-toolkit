"""A blank sites-CSV cell must store as "", not the text "nan"."""

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager


def test_blank_site_name_is_stored_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "db_path", property(lambda self: tmp_path / "p.db")
    )
    db = DatabaseManager()
    df = pd.DataFrame(
        {
            config.site_id_column: ["SLI_001", "SLI_002"],
            config.site_name_column: [float("nan"), "Mokohinau"],
            config.link_to_marine_reserve_column: [float("nan"), "Some Reserve"],
            config.protection_status_column: ["No protection", "No protection"],
        }
    )
    db.upsert_sites(df)
    assert db.get_site("SLI_001")[config.site_name_column] == ""
    assert db.get_site("SLI_002")[config.site_name_column] == "Mokohinau"
