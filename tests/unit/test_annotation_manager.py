"""
Tests for AnnotationDatabaseManager — annotation CRUD and aggregation.

Covers:
  - clear_synced_annotations: deletes only biigle_volume_* rows (legacy + manual survive)
  - get_annotations_for_drop: with and without annotated_by filter
  - get_maxn_summary: MAX(max_interval) per drop × species × source
"""

import pytest

from spyfish.database.annotation_manager import AnnotationDatabaseManager

DROP = "KSF_20240124_BUV_KSF_085_01"


@pytest.fixture
def ann_db(tmp_path):
    db = AnnotationDatabaseManager(db_path=str(tmp_path / "test_annotations.db"))
    return db


def _seed(ann_db, rows):
    ann_db.add_annotations(rows)


def _ann(
    drop_id=DROP,
    species="Pagrus auratus",
    max_interval=1,
    annotated_by="ml",
    external_id="yolov8n",
    time_of_max="00:00:05",
):
    return {
        "drop_id": drop_id,
        "scientific_name": species,
        "time_of_max": time_of_max,
        "max_interval": max_interval,
        "annotated_by": annotated_by,
        "interval_annotation": "",
        "confidence_agreement": 0.9,
        "external_id": external_id,
    }


# ── clear_synced_annotations ────────────────────────────────────────────────


def test_clear_synced_only_deletes_with_external_id(ann_db):
    """Rows with external_id=None (manual entry) should survive the clear."""
    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id="biigle_42"),
            _ann(annotated_by="expert", external_id=None, species="Manual Entry"),
        ],
    )
    ann_db.clear_synced_annotations(DROP, "expert")
    remaining = ann_db.get_annotations_for_drop(DROP, "expert")
    assert len(remaining) == 1
    assert remaining[0]["scientific_name"] == "Manual Entry"


def test_clear_synced_does_not_touch_other_sources(ann_db):
    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id="biigle_1"),
            _ann(annotated_by="ml", external_id="yolov8n"),
        ],
    )
    ann_db.clear_synced_annotations(DROP, "expert")
    ml = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(ml) == 1


# ── get_annotations_for_drop ────────────────────────────────────────────────


def test_get_annotations_for_drop_all(ann_db):
    _seed(
        ann_db,
        [
            _ann(annotated_by="ml"),
            _ann(annotated_by="expert", external_id="biigle_1"),
            _ann(annotated_by="citsci", external_id=None),
        ],
    )
    all_anns = ann_db.get_annotations_for_drop(DROP)
    assert len(all_anns) == 3


def test_get_annotations_for_drop_filtered(ann_db):
    _seed(
        ann_db,
        [
            _ann(annotated_by="ml"),
            _ann(annotated_by="expert", external_id="biigle_1"),
        ],
    )
    expert_only = ann_db.get_annotations_for_drop(DROP, annotated_by="expert")
    assert len(expert_only) == 1
    assert expert_only[0]["annotated_by"] == "expert"


def test_get_annotations_for_drop_empty(ann_db):
    assert ann_db.get_annotations_for_drop("NONEXISTENT") == []


# ── get_maxn_summary ────────────────────────────────────────────────────────


def test_get_maxn_summary_returns_peak_per_species(ann_db):
    """MAX(max_interval) across time intervals, not SUM."""
    _seed(
        ann_db,
        [
            _ann(max_interval=3, time_of_max="00:00:05"),
            _ann(max_interval=5, time_of_max="00:00:15"),
            _ann(max_interval=2, time_of_max="00:00:25"),
        ],
    )
    df = ann_db.get_maxn_summary(drop_id=DROP, annotated_by="ml")
    assert len(df) == 1
    assert df.iloc[0]["maxn"] == 5  # peak, not sum


def test_get_maxn_summary_separates_species(ann_db):
    _seed(
        ann_db,
        [
            _ann(species="Pagrus auratus", max_interval=3),
            _ann(species="Notolabrus fucicola", max_interval=7, time_of_max="00:00:15"),
        ],
    )
    df = ann_db.get_maxn_summary(drop_id=DROP)
    assert len(df) == 2
    species_maxn = dict(zip(df["scientific_name"], df["maxn"]))
    assert species_maxn["Pagrus auratus"] == 3
    assert species_maxn["Notolabrus fucicola"] == 7


def test_get_maxn_summary_empty(ann_db):
    df = ann_db.get_maxn_summary(drop_id="NONEXISTENT")
    assert df.empty


# ── null_deployment_row ─────────────────────────────────────────────────────


def test_null_deployment_row_round_trip(ann_db):
    """The absence record survives storage and the MaxN summary.

    get_maxn_summary once dropped absence records via a NULL = NULL subquery;
    the sentinel plus the null-safe IS comparison keeps them visible.
    """
    from spyfish.config.base import NULL_DEPLOYMENT
    from spyfish.database.annotation_manager import null_deployment_row

    row = null_deployment_row(DROP, "citsci")
    assert row["scientific_name"] == NULL_DEPLOYMENT
    assert row["max_interval"] == 0
    assert row["time_of_max_seconds"] is None

    _seed(ann_db, [row])
    stored = ann_db.get_annotations_for_drop(DROP, "citsci")
    assert len(stored) == 1
    assert stored[0]["scientific_name"] == NULL_DEPLOYMENT

    summary = ann_db.get_maxn_summary(drop_id=DROP)
    assert len(summary) == 1
    assert summary.iloc[0]["maxn"] == 0


def test_null_deployment_row_biigle_replaceable(ann_db):
    """A volume-scoped external_id makes the row clear_synced-replaceable."""
    from spyfish.database.annotation_manager import null_deployment_row

    _seed(ann_db, [null_deployment_row(DROP, "expert", external_id="biigle_volume_7")])
    ann_db.clear_synced_annotations(DROP, "expert")
    assert ann_db.get_annotations_for_drop(DROP, "expert") == []


# ── Newest model per drop ───────────────────────────────────────────────────

OLD_MODEL = "cfd_binary_water_20260301"
NEW_MODEL = "species_20260822_192536"


def test_model_recency_key_reads_the_training_stamp():
    """Type first, stamp last; a date-only name sorts at midnight of its day;
    anything off the standard sorts as oldest."""
    from spyfish.database.annotation_manager import model_recency_key

    names = [
        "species_20260822_192536",
        "cfd_binary_water_20260301",
        "species_20260822_020313",
        "species_20260822",
        "no_stamp_model",
        "20260429_081503_species",  # stamp-first: not canonical, sorts oldest
    ]
    assert sorted(names, key=model_recency_key) == [
        "20260429_081503_species",
        "no_stamp_model",
        "cfd_binary_water_20260301",
        "species_20260822",
        "species_20260822_020313",
        "species_20260822_192536",
    ]


def test_init_db_renames_stamp_first_model_names(tmp_path):
    """`20260429_081503_species` (auto-promotion's old spelling) becomes
    `species_20260429_081503`; canonical names and other sources untouched."""
    import sqlite3

    path = tmp_path / "a.db"
    db = AnnotationDatabaseManager(db_path=str(path))
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO annotations (drop_id, scientific_name, max_interval, "
            "annotated_by, external_id) VALUES (?, ?, ?, ?, ?)",
            [
                (DROP, "Pagrus auratus", 2, "ml", "20260429_081503_species"),
                (DROP, "Pagrus auratus", 3, "ml", "species_20260822_192536"),
                (DROP, "Pagrus auratus", 1, "expert", "legacy"),
            ],
        )
    AnnotationDatabaseManager(db_path=str(path))  # init_db runs the migration
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT annotated_by, external_id FROM annotations ORDER BY id"
        ).fetchall()
    assert rows == [
        ("ml", "species_20260429_081503"),
        ("ml", "species_20260822_192536"),
        ("expert", "legacy"),
    ]
    del db


def test_init_db_moves_an_undated_model_name_out_of_annotated_by(tmp_path):
    import sqlite3

    path = tmp_path / "b.db"
    AnnotationDatabaseManager(db_path=str(path))
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO annotations (drop_id, scientific_name, max_interval, "
            "annotated_by, external_id) VALUES (?, ?, ?, ?, NULL)",
            (DROP, "Pagrus auratus", 2, "my_undated_model"),
        )
    AnnotationDatabaseManager(db_path=str(path))
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT annotated_by, external_id FROM annotations"
        ).fetchone()
    assert row == ("ml", "my_undated_model")


def test_maxn_summary_takes_the_newest_model_not_the_larger_count(ann_db):
    """The old model's bigger count must not win over the newer model."""
    _seed(
        ann_db,
        [
            _ann(max_interval=9, external_id=OLD_MODEL),
            _ann(max_interval=2, external_id=NEW_MODEL),
            _ann(species="Parika scaber", max_interval=4, external_id=OLD_MODEL),
        ],
    )
    df = ann_db.get_maxn_summary(drop_id=DROP)
    assert set(df["external_id"]) == {NEW_MODEL}
    assert df.set_index("scientific_name")["maxn"].to_dict() == {"Pagrus auratus": 2}


def test_maxn_summary_all_models_keeps_one_peak_per_model(ann_db):
    _seed(
        ann_db,
        [
            _ann(max_interval=9, external_id=OLD_MODEL),
            _ann(max_interval=2, external_id=NEW_MODEL),
        ],
    )
    df = ann_db.get_maxn_summary(drop_id=DROP, all_models=True)
    assert df.set_index("external_id")["maxn"].to_dict() == {OLD_MODEL: 9, NEW_MODEL: 2}


def test_newest_model_only_chooses_from_the_full_model_list():
    """A species slice must not fall back to an older model that saw it."""
    import pandas as pd

    from spyfish.database.annotation_manager import newest_model_only

    slice_ = pd.DataFrame(
        {
            "drop_id": [DROP, DROP],
            "annotated_by": ["ml", "expert"],
            "external_id": [OLD_MODEL, "12345"],
        }
    )
    models = pd.DataFrame(
        {"drop_id": [DROP, DROP], "external_id": [OLD_MODEL, NEW_MODEL]}
    )
    out = newest_model_only(slice_, models)
    assert list(out["annotated_by"]) == ["expert"]


def test_init_moves_a_model_name_out_of_annotated_by(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    ann_db = AnnotationDatabaseManager(db_path=db_path)
    _seed(
        ann_db,
        [
            _ann(annotated_by="binary_cfd_sweep_20260427_002337", external_id=None),
            _ann(annotated_by="expert", external_id="777"),
        ],
    )
    AnnotationDatabaseManager(db_path=db_path)  # re-open runs the migration
    rows = {
        r["annotated_by"]: r["external_id"]
        for r in ann_db.get_annotations_for_drop(DROP)
    }
    # The expert row's bare annotation id becomes a BIIGLE-owned key too (the
    # pipeline DB of the test config knows no volume for it, hence "unknown").
    assert rows == {
        "ml": "binary_cfd_sweep_20260427_002337",
        "expert": "biigle_volume_unknown",
    }


# ── Legacy expert rows survive a BIIGLE re-sync ─────────────────────────────


def test_clear_synced_leaves_legacy_expert_rows(ann_db):
    """The first BIIGLE sync on a drop with legacy expert data must not erase it.

    Before 2026-10-13 the clear matched `external_id IS NOT NULL`, and the
    legacy backfill writes `external_id = "legacy"`, which is not NULL. No drop
    had both at the time, so the deletion had never fired.
    """
    from spyfish.database.annotation_manager import (
        LEGACY_EXTERNAL_ID,
        biigle_external_id,
    )

    _seed(
        ann_db,
        [
            _ann(
                annotated_by="expert",
                external_id=LEGACY_EXTERNAL_ID,
                species="Legacy fish",
            ),
            _ann(
                annotated_by="expert",
                external_id=biigle_external_id(7),
                species="Biigle fish",
            ),
            _ann(annotated_by="expert", external_id=None, species="Manual Entry"),
        ],
    )
    ann_db.clear_synced_annotations(DROP, "expert")
    remaining = {
        r["scientific_name"] for r in ann_db.get_annotations_for_drop(DROP, "expert")
    }
    assert remaining == {"Legacy fish", "Manual Entry"}


def test_maxn_summary_keeps_legacy_and_biigle_expert_peaks_apart(ann_db):
    """Two expert accounts of one drop are two rows, not one merged peak."""
    from spyfish.database.annotation_manager import (
        LEGACY_EXTERNAL_ID,
        biigle_external_id,
    )

    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id=LEGACY_EXTERNAL_ID, max_interval=5),
            _ann(
                annotated_by="expert", external_id=biigle_external_id(7), max_interval=2
            ),
            # citsci external_id is a per-row subject id and must NOT split the peak
            _ann(annotated_by="citsci", external_id="1001", max_interval=1),
            _ann(annotated_by="citsci", external_id="1002", max_interval=3),
        ],
    )
    summary = ann_db.get_maxn_summary(drop_id=DROP)
    expert = summary[summary["annotated_by"] == "expert"].set_index("external_id")[
        "maxn"
    ]
    assert expert.to_dict() == {LEGACY_EXTERNAL_ID: 5, biigle_external_id(7): 2}
    citsci = summary[summary["annotated_by"] == "citsci"]
    assert len(citsci) == 1 and citsci.iloc[0]["maxn"] == 3


def test_annotation_source_labels_each_account():
    """Export label: legacy / biigle_volume_N for experts, model for ML, name otherwise."""
    import pandas as pd

    from spyfish.database.annotation_manager import annotation_source

    by = pd.Series(["expert", "expert", "expert", "ml", "ml", "citsci"])
    ext = pd.Series(
        ["legacy", "biigle_volume_7", None, "species_20260822", None, "1001"]
    )
    assert annotation_source(by, ext).tolist() == [
        "legacy",
        "biigle_volume_7",
        "expert",
        "species_20260822",
        "ml",
        "citsci",
    ]


def test_export_df_has_a_source_column(ann_db):
    from spyfish.config.wrapper import config
    from spyfish.database.annotation_manager import (
        LEGACY_EXTERNAL_ID,
        biigle_external_id,
    )

    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id=LEGACY_EXTERNAL_ID),
            _ann(annotated_by="expert", external_id=biigle_external_id(7)),
            _ann(annotated_by="ml", external_id="species_20260822"),
        ],
    )
    df = ann_db.get_all_annotations_export_df()
    assert "external_id" not in df.columns
    assert sorted(df[config.csv_source_column]) == [
        "biigle_volume_7",
        "legacy",
        "species_20260822",
    ]
    assert set(df[config.csv_annotated_by_column]) == {"expert", "ml"}


def test_init_migrates_numeric_biigle_ids_to_volume_keys(tmp_path, monkeypatch):
    """Pre-2026-10-13 expert rows hold a bare annotation id; rewrite to the volume."""
    import sqlite3

    from spyfish.config.wrapper import config
    from spyfish.database.annotation_manager import biigle_external_id

    pipeline = tmp_path / "pipeline.db"
    with sqlite3.connect(pipeline) as c:
        c.execute(
            "CREATE TABLE deployments (drop_id TEXT PRIMARY KEY, biigle_volume_id INTEGER)"
        )
        c.execute("INSERT INTO deployments VALUES (?, 4626)", (DROP,))
    monkeypatch.setattr(type(config), "db_path", property(lambda self: pipeline))

    ann_path = tmp_path / "ann.db"
    first = AnnotationDatabaseManager(db_path=str(ann_path))
    _seed(
        first,
        [
            _ann(annotated_by="expert", external_id="123456"),
            _ann(annotated_by="expert", external_id="legacy", species="Legacy fish"),
            _ann(
                drop_id="KSF_20240124_BUV_KSF_085_02",
                annotated_by="expert",
                external_id="99",
            ),
            _ann(annotated_by="citsci", external_id="1001"),
        ],
    )
    reopened = AnnotationDatabaseManager(db_path=str(ann_path))  # init_db runs again
    rows = {
        (r["drop_id"], r["scientific_name"], r["annotated_by"]): r["external_id"]
        for r in reopened.get_annotations_for_drop(DROP)
        + reopened.get_annotations_for_drop("KSF_20240124_BUV_KSF_085_02")
    }
    assert rows[(DROP, "Pagrus auratus", "expert")] == biigle_external_id(4626)
    assert rows[(DROP, "Legacy fish", "expert")] == "legacy"
    assert rows[(DROP, "Pagrus auratus", "citsci")] == "1001"
    # a drop the pipeline DB does not know still becomes BIIGLE-owned
    assert rows[
        ("KSF_20240124_BUV_KSF_085_02", "Pagrus auratus", "expert")
    ] == biigle_external_id("unknown")
    # idempotent: a third open changes nothing
    AnnotationDatabaseManager(db_path=str(ann_path))
    again = {
        r["scientific_name"]: r["external_id"]
        for r in reopened.get_annotations_for_drop(DROP, "expert")
    }
    assert again == {
        "Pagrus auratus": biigle_external_id(4626),
        "Legacy fish": "legacy",
    }


def test_maxn_summary_does_not_split_unmigrated_numeric_expert_ids(ann_db):
    """A read-only dashboard may see pre-migration rows; one peak, not one per box."""
    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id="101", max_interval=2),
            _ann(annotated_by="expert", external_id="102", max_interval=6),
            _ann(annotated_by="expert", external_id="103", max_interval=1),
        ],
    )
    summary = ann_db.get_maxn_summary(drop_id=DROP)
    assert len(summary) == 1
    assert summary.iloc[0]["maxn"] == 6


def test_clear_synced_annotations_with_a_volume_keeps_the_other_volume(ann_db):
    """A drop reviewed in two BIIGLE volumes: re-syncing one must not wipe the other."""
    from spyfish.database.annotation_manager import biigle_external_id

    _seed(
        ann_db,
        [
            _ann(
                max_interval=2,
                annotated_by="expert",
                external_id=biigle_external_id(111),
            ),
            _ann(
                max_interval=5,
                annotated_by="expert",
                external_id=biigle_external_id(222),
            ),
            _ann(max_interval=1, annotated_by="expert", external_id="legacy"),
        ],
    )
    ann_db.clear_synced_annotations(DROP, "expert", volume_id=111)
    left = sorted(
        r["external_id"] for r in ann_db.get_annotations_for_drop(DROP, "expert")
    )
    assert left == [biigle_external_id(222), "legacy"]
