"""
Extended DatabaseManager tests — covers methods not in test_database.py
or test_status_transitions.py.

  - update_deployment_fields: valid, invalid field, missing drop_id
  - upsert_sites: inserts, skips empty site_id, full replace
  - get_site: found, not found
  - sync_annotation_counts: cross-DB count propagation
  - get_max_priority: empty DB, after setting priorities
"""

import pandas as pd
import pytest

from spyfish.config.base import MlStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager

# ── update_deployment_fields ─────────────────────────────────────────────────


def test_update_deployment_fields_sets_priority(temp_db):
    temp_db.add_or_update_deployment(drop_id="KSF_20240124_BUV_KSF_085_01")
    result = temp_db.update_deployment_fields(
        "KSF_20240124_BUV_KSF_085_01", priority=10
    )
    assert result is True
    dep = temp_db.get_deployment("KSF_20240124_BUV_KSF_085_01")
    assert dep["priority"] == 10


def test_update_deployment_fields_invalid_field_raises(temp_db):
    temp_db.add_or_update_deployment(drop_id="KSF_20240124_BUV_KSF_085_01")
    with pytest.raises(ValueError, match="Unknown fields"):
        temp_db.update_deployment_fields(
            "KSF_20240124_BUV_KSF_085_01", bogus_field="bad"
        )


def test_update_deployment_fields_missing_drop_returns_false(temp_db):
    result = temp_db.update_deployment_fields("NONEXISTENT_DROP", priority=5)
    assert result is False


def test_update_deployment_fields_empty_is_noop(temp_db):
    temp_db.add_or_update_deployment(drop_id="KSF_20240124_BUV_KSF_085_01")
    result = temp_db.update_deployment_fields("KSF_20240124_BUV_KSF_085_01")
    assert result is True


# ── upsert_sites + get_site ──────────────────────────────────────────────────


def test_upsert_sites_and_get_site(temp_db):
    sites_df = pd.DataFrame(
        [
            {
                config.site_id_column: "KSF_085",
                config.site_name_column: "Kapiti South",
                config.link_to_marine_reserve_column: "Kapiti Island",
                config.protection_status_column: "Marine Reserve",
            }
        ]
    )
    temp_db.upsert_sites(sites_df)

    site = temp_db.get_site("KSF_085")
    assert site is not None
    assert site[config.site_name_column] == "Kapiti South"
    assert site[config.link_to_marine_reserve_column] == "Kapiti Island"


def test_upsert_sites_skips_empty_site_id(temp_db):
    sites_df = pd.DataFrame(
        [
            {
                config.site_id_column: "",
                config.site_name_column: "Ghost Site",
                config.link_to_marine_reserve_column: "",
                config.protection_status_column: "",
            }
        ]
    )
    temp_db.upsert_sites(sites_df)
    # No site should exist
    assert temp_db.get_site("") is None


def test_upsert_sites_full_replace(temp_db):
    """Second upsert should delete sites from the first call."""
    sites_v1 = pd.DataFrame(
        [
            {
                config.site_id_column: "OLD_001",
                config.site_name_column: "Old Site",
                config.link_to_marine_reserve_column: "",
                config.protection_status_column: "",
            },
        ]
    )
    temp_db.upsert_sites(sites_v1)
    assert temp_db.get_site("OLD_001") is not None

    sites_v2 = pd.DataFrame(
        [
            {
                config.site_id_column: "NEW_001",
                config.site_name_column: "New Site",
                config.link_to_marine_reserve_column: "",
                config.protection_status_column: "",
            },
        ]
    )
    temp_db.upsert_sites(sites_v2)
    assert temp_db.get_site("OLD_001") is None
    assert temp_db.get_site("NEW_001") is not None


def test_get_site_not_found(temp_db):
    assert temp_db.get_site("NONEXISTENT") is None


# ── get_max_priority ─────────────────────────────────────────────────────────


def test_get_max_priority_empty_db(temp_db):
    assert temp_db.get_max_priority() == 0


def test_get_max_priority_returns_highest(temp_db):
    temp_db.add_or_update_deployment(drop_id="KSF_20240124_BUV_KSF_085_01")
    temp_db.add_or_update_deployment(drop_id="KSF_20240124_BUV_KSF_085_02")
    temp_db.update_deployment_fields("KSF_20240124_BUV_KSF_085_01", priority=5)
    temp_db.update_deployment_fields("KSF_20240124_BUV_KSF_085_02", priority=20)
    assert temp_db.get_max_priority() == 20


# ── sync_annotation_counts ──────────────────────────────────────────────────


def test_sync_annotation_counts(tmp_path, monkeypatch):
    """Counts from annotation DB propagate to deployment DB."""
    from spyfish.config.wrapper import config as real_config

    monkeypatch.setattr(real_config, "_project_root", tmp_path)

    # Both DBs must be at the canonical paths so sync_annotation_counts
    # (which creates its own AnnotationDatabaseManager internally) finds them.
    (tmp_path / "process_files" / "db").mkdir(parents=True)
    from spyfish.database.manager import DatabaseManager

    db = DatabaseManager(db_path=str(real_config.db_path))
    ann_db = AnnotationDatabaseManager(db_path=str(real_config.annotations_db_path))

    drop_id = "KSF_20240124_BUV_KSF_085_01"
    db.add_or_update_deployment(drop_id=drop_id, ml_status=MlStatus.COMPLETE)

    # Add 2 ML + 1 expert annotation
    ann_db.add_annotations(
        [
            {
                "drop_id": drop_id,
                "scientific_name": "Pagrus auratus",
                "time_of_max": "00:00:05",
                "max_interval": 2,
                "annotated_by": "ml",
                "interval_annotation": "",
                "confidence_agreement": 0.9,
                "external_id": "yolov8n",
            },
            {
                "drop_id": drop_id,
                "scientific_name": "Notolabrus fucicola",
                "time_of_max": "00:00:15",
                "max_interval": 1,
                "annotated_by": "ml",
                "interval_annotation": "",
                "confidence_agreement": 0.8,
                "external_id": "yolov8n",
            },
            {
                "drop_id": drop_id,
                "scientific_name": "Pagrus auratus",
                "time_of_max": "00:00:05",
                "max_interval": 3,
                "annotated_by": "expert",
                "interval_annotation": "",
                "confidence_agreement": 1.0,
                "external_id": "biigle_42",
            },
        ]
    )

    db.sync_annotation_counts([drop_id])

    dep = db.get_deployment(drop_id)
    assert dep["ml_annotations"] == 2
    assert dep["expert_annotations"] == 1
    assert dep["citsci_annotations"] == 0
