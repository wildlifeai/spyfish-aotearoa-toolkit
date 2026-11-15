"""SQLite overhead guards: one transaction per ingest, schema init once per file.

The databases sit on a network filesystem on the HPC, where every connection
open and every commit is slow. These tests pin the counts so a per-row loop
cannot creep back in.
"""

import os
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from spyfish.config.base import (
    INGEST_ISSUE,
    CitSciStatus,
    IngestStatus,
    MlStatus,
    VideoPresence,
)
from spyfish.config.wrapper import config
from spyfish.database import annotation_manager as annotation_manager_module
from spyfish.database import manager as manager_module
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager, validation_error_row
from spyfish.orchestrator.ingest import _sync_deployments_to_db

DROPS = [f"KSF_20240124_BUV_KSF_085_{n:02d}" for n in range(1, 7)]


class _Counting:
    """Wraps ``sqlite3.connect`` and counts the calls."""

    def __init__(self):
        self.calls = 0
        self._real = sqlite3.connect

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._real(*args, **kwargs)


@pytest.fixture
def connect_counter():
    counter = _Counting()
    with (
        patch.object(manager_module.sqlite3, "connect", counter),
        patch.object(annotation_manager_module.sqlite3, "connect", counter),
    ):
        yield counter


def _deployment_rows():
    """Six rows covering the defaults, every optional field and a bad deployment."""
    return [
        {"drop_id": DROPS[0]},
        {
            "drop_id": DROPS[1],
            "ingest_status": IngestStatus.OK,
            "ml_status": MlStatus.READY,
            "video_path": f"media/{DROPS[1]}.mp4",
            "video_presence": VideoPresence.PRESENT,
            "sampling_start": 10,
            "sampling_end": 1810,
            "biigle_volume_id": "4626",
            "latitude": -36.5,
            "longitude": 175.1,
            "depth": 12.5,
            "nzmhcs_abiotic": "Sand",
            "nzmhcs_biotic": "Kelp forest",
        },
        {
            "drop_id": DROPS[2],
            "ingest_status": IngestStatus.EXCLUDED,
            "is_bad_deployment": True,
            "video_presence": VideoPresence.NO_VIDEO_BAD_DEP,
        },
        {"drop_id": DROPS[3], "ingest_status": IngestStatus.VALIDATION_ERROR},
        {"drop_id": DROPS[4], "ml_status": MlStatus.READY},
        {"drop_id": DROPS[5], "video_presence": VideoPresence.ARCHIVED},
    ]


def _reingest_rows():
    """The same drops again, with changed metadata and blanked coordinates."""
    rows = _deployment_rows()
    rows[1] = {
        "drop_id": DROPS[1],
        "ingest_status": IngestStatus.VALIDATION_ERROR,
        "ml_status": MlStatus.PENDING,  # must NOT overwrite the stored status
        "video_path": f"media/moved/{DROPS[1]}.mp4",
        "video_presence": VideoPresence.ARCHIVED,
        "sampling_start": 20,
        "sampling_end": 1820,
    }
    return rows


def _errors():
    return [
        validation_error_row(
            survey_id="KSF_20240124_BUV",
            drop_id=DROPS[3],
            error_type=IngestStatus.VALIDATION_ERROR,
            column_name="sampling_window",
            error_message="window too short",
        ),
        validation_error_row(
            survey_id="KSF_20240124_BUV",
            drop_id=DROPS[1],
            error_type=INGEST_ISSUE,
            column_name="sampling_window",
            error_message="starts late",
            invalid_value="20",
        ),
    ]


def _progress(db):
    """Pipeline progress a re-ingest must leave alone."""
    db.advance_status(DROPS[1], MlStatus.COLUMN, MlStatus.RUNNING)
    db.update_section_status(DROPS[4], MlStatus.COLUMN, MlStatus.COMPLETE)
    db.update_section_status(DROPS[4], CitSciStatus.COLUMN, CitSciStatus.CLIPS_UPLOADED)
    db.update_deployment_fields(DROPS[0], priority=7)


def _dump(db):
    """Everything the two write paths must agree on (timestamps left out)."""
    with db.get_connection() as conn:
        deployments = [
            {k: v for k, v in dict(r).items() if k not in ("created_at", "updated_at")}
            for r in conn.execute("SELECT * FROM deployments ORDER BY drop_id")
        ]
        errors = [
            {k: v for k, v in dict(r).items() if k != "created_at"}
            for r in conn.execute("SELECT * FROM validation_errors ORDER BY id")
        ]
    return deployments, errors


def _write_per_row(db, rows, errors):
    for error in errors:
        db.add_validation_errors([error])
    for row in rows:
        db.add_or_update_deployment(**row)


# ── Bulk upsert ──────────────────────────────────────────────────────────────


def test_bulk_upsert_writes_the_same_rows_as_the_per_row_path(tmp_path):
    per_row = DatabaseManager(db_path=str(tmp_path / "per_row.db"))
    bulk = DatabaseManager(db_path=str(tmp_path / "bulk.db"))

    _write_per_row(per_row, _deployment_rows(), _errors())
    bulk.add_or_update_deployments(_deployment_rows(), validation_errors=_errors())

    deployments, errors = _dump(bulk)
    assert (deployments, errors) == _dump(per_row)
    assert [d["drop_id"] for d in deployments] == DROPS
    assert len(errors) == 2


def test_bulk_reingest_preserves_section_statuses_like_the_per_row_path(tmp_path):
    per_row = DatabaseManager(db_path=str(tmp_path / "per_row.db"))
    bulk = DatabaseManager(db_path=str(tmp_path / "bulk.db"))

    _write_per_row(per_row, _deployment_rows(), [])
    bulk.add_or_update_deployments(_deployment_rows())
    for db in (per_row, bulk):
        _progress(db)

    _write_per_row(per_row, _reingest_rows(), _errors())
    bulk.add_or_update_deployments(_reingest_rows(), validation_errors=_errors())

    assert _dump(bulk) == _dump(per_row)
    after = bulk.get_deployments_by_ids(DROPS)
    # Section statuses and priority survive the re-ingest.
    assert after[DROPS[1]]["ml_status"] == MlStatus.RUNNING
    assert after[DROPS[4]]["ml_status"] == MlStatus.COMPLETE
    assert after[DROPS[4]]["citsci_status"] == CitSciStatus.CLIPS_UPLOADED
    assert after[DROPS[0]]["priority"] == 7
    # Metadata is refreshed.
    assert after[DROPS[1]]["ingest_status"] == IngestStatus.VALIDATION_ERROR
    assert after[DROPS[1]]["video_presence"] == VideoPresence.ARCHIVED
    assert after[DROPS[1]]["video_path"] == f"media/moved/{DROPS[1]}.mp4"
    assert after[DROPS[1]]["sampling_start"] == 20
    # Blanked coordinates, habitat and volume keep the recorded values.
    assert after[DROPS[1]]["latitude"] == -36.5
    assert after[DROPS[1]]["depth"] == 12.5
    assert after[DROPS[1]]["nzmhcs_abiotic"] == "Sand"
    assert after[DROPS[1]]["biigle_volume_id"] == "4626"


def test_bulk_upsert_is_all_or_nothing(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROPS[0], video_path="before.mp4")
    rows = [
        {"drop_id": DROPS[0], "video_path": "after.mp4"},
        {"drop_id": DROPS[1]},
        {"drop_id": DROPS[2], "ingest_status": None},  # NOT NULL column
    ]
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.add_or_update_deployments(rows, validation_errors=_errors())

    deployments, errors = _dump(temp_db)
    assert [d["drop_id"] for d in deployments] == [DROPS[0]]
    assert deployments[0]["video_path"] == "before.mp4"
    assert errors == []


def test_bulk_upsert_rejects_an_unknown_field_before_writing(temp_db):
    with pytest.raises(TypeError):
        temp_db.add_or_update_deployments(
            [{"drop_id": DROPS[0]}, {"drop_id": DROPS[1], "not_a_column": 1}]
        )
    assert temp_db.get_all_deployments_map() == {}


def test_bulk_upsert_of_nothing_opens_no_connection(temp_db, connect_counter):
    temp_db.add_or_update_deployments([])
    temp_db.add_or_update_deployments([], validation_errors=[])
    assert connect_counter.calls == 0


# ── Ingest ───────────────────────────────────────────────────────────────────


def _ingest_frame(n):
    """`n` deployment rows as the SharePoint CSV gives them.

    Row 0 has a sampling window that is far too short (an error), row 1 a
    blank one, row 2 is a bad deployment, row 3 has no video on S3, row 5 is
    one second short (an issue, not blocking); the rest are clean.
    """
    full = config.buv_video_duration_seconds
    one_second_short = full - config.sampling_end_buffer_seconds - 1
    records, known_files = [], set()
    for i in range(n):
        drop_id = f"KSF_20240124_BUV_KSF_{i + 1:03d}_01"
        key = config.get_video_s3_key(drop_id)
        start, end = 60.0, 60.0 + full
        if i == 0:
            start, end = 60.0, 100.0
        elif i == 1:
            start, end = float("nan"), float("nan")
        elif i == 5:
            end = start + one_second_short
        records.append(
            {
                config.drop_id_column: drop_id,
                config.csv_is_bad_deployment_column: "True" if i == 2 else "False",
                config.csv_video_file_link_column: key,
                config.csv_sampling_start_column: start,
                config.csv_sampling_end_column: end,
                config.latitude_column: -36.5,
                config.longitude_column: 175.1,
                config.depth_column: 10 + i,
                config.nzmhcs_abiotic_column: "",
                config.nzmhcs_biotic_column: "",
            }
        )
        if i != 3:
            known_files.add(key)
    mapping = {
        "drop_id_column": config.drop_id_column,
        "is_bad_deployment_column": config.csv_is_bad_deployment_column,
        "video_file_link_column": config.csv_video_file_link_column,
    }
    return pd.DataFrame(records), known_files, mapping


def _run_sync(db, n):
    df, known_files, mapping = _ingest_frame(n)
    _sync_deployments_to_db(
        df, db, set(), known_files, {key: "STANDARD" for key in known_files}, mapping
    )
    return df


def test_ingest_opens_one_connection_whatever_the_row_count(temp_db, connect_counter):
    counts = []
    for n in (5, 40):
        before = connect_counter.calls
        _run_sync(temp_db, n)
        counts.append(connect_counter.calls - before)
    assert counts == [1, 1]


def test_ingest_writes_what_the_per_row_calls_wrote(temp_db, tmp_path):
    """The batched ingest against a replay of it through the per-row methods."""

    class _Recorder:
        def __init__(self):
            self.rows, self.errors = [], []

        def add_or_update_deployments(self, deployments, validation_errors=None):
            self.rows = list(deployments)
            self.errors = list(validation_errors or [])

    recorder = _Recorder()
    df, known_files, mapping = _ingest_frame(8)
    _sync_deployments_to_db(
        df, recorder, set(), known_files, dict.fromkeys(known_files), mapping
    )
    per_row = DatabaseManager(db_path=str(tmp_path / "per_row.db"))
    _write_per_row(per_row, recorder.rows, recorder.errors)

    _sync_deployments_to_db(
        df, temp_db, set(), known_files, dict.fromkeys(known_files), mapping
    )

    deployments, errors = _dump(temp_db)
    assert (deployments, errors) == _dump(per_row)
    by_id = {d["drop_id"]: d for d in deployments}
    ids = list(df[config.drop_id_column])
    assert len(by_id) == 8
    assert by_id[ids[0]]["ingest_status"] == IngestStatus.VALIDATION_ERROR
    assert by_id[ids[1]]["ingest_status"] == IngestStatus.VALIDATION_ERROR
    assert by_id[ids[2]]["ingest_status"] == IngestStatus.EXCLUDED
    assert by_id[ids[3]]["video_presence"] == VideoPresence.ABSENT
    assert by_id[ids[3]]["ml_status"] == MlStatus.PENDING
    assert by_id[ids[4]]["ml_status"] == MlStatus.READY
    assert by_id[ids[5]]["ingest_status"] == IngestStatus.OK
    assert {(e["DropID"], e["ErrorType"]) for e in errors} == {
        (ids[0], IngestStatus.VALIDATION_ERROR),
        (ids[5], INGEST_ISSUE),
    }


def test_reingest_through_the_ingest_keeps_pipeline_progress(temp_db):
    db = temp_db
    df = _run_sync(db, 6)
    drop_id = df[config.drop_id_column][4]
    db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.RUNNING)
    _run_sync(db, 6)
    assert db.get_deployment(drop_id)["ml_status"] == MlStatus.RUNNING


def test_advance_status_uses_one_connection(temp_db, connect_counter):
    temp_db.add_or_update_deployment(drop_id=DROPS[0], ml_status=MlStatus.READY)
    before = connect_counter.calls
    temp_db.advance_status(DROPS[0], MlStatus.COLUMN, MlStatus.RUNNING)
    assert connect_counter.calls - before == 1
    assert temp_db.get_deployment(DROPS[0])["ml_status"] == MlStatus.RUNNING


# ── Schema init once per database file ───────────────────────────────────────


@pytest.mark.parametrize("manager_cls", [DatabaseManager, AnnotationDatabaseManager])
def test_schema_init_runs_once_per_path(tmp_path, connect_counter, manager_cls):
    path = str(tmp_path / "one.db")
    calls = []
    real_init = manager_cls.init_db

    def counting_init(self):
        calls.append(self.db_path)
        return real_init(self)

    with patch.object(manager_cls, "init_db", counting_init):
        first = manager_cls(db_path=path)
        after_first = connect_counter.calls
        for _ in range(5):
            again = manager_cls(db_path=path)
        assert calls == [first.db_path]
        # Repeat constructions open no connection at all.
        assert connect_counter.calls == after_first

        other = manager_cls(db_path=str(tmp_path / "two.db"))
        assert calls == [first.db_path, other.db_path]

    if manager_cls is DatabaseManager:
        assert again._deployments_columns == first._deployments_columns
        assert "ml_status" in again._deployments_columns
        again.validate_column("zoo_crops_status")


@pytest.mark.parametrize("manager_cls", [DatabaseManager, AnnotationDatabaseManager])
def test_schema_init_runs_again_for_a_recreated_file(tmp_path, manager_cls):
    path = tmp_path / "recreated.db"
    table = "deployments" if manager_cls is DatabaseManager else "annotations"

    def has_table():
        with sqlite3.connect(path) as conn:
            return bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)
                ).fetchone()
            )

    manager_cls(db_path=str(path))
    assert has_table()

    # Deleted: the next construction recreates it.
    os.rename(path, tmp_path / "moved_away.db")
    manager_cls(db_path=str(path))
    assert has_table()

    # Replaced by another file with no schema (a sync landing a new copy).
    replacement = tmp_path / "replacement.db"
    with sqlite3.connect(replacement) as conn:
        conn.execute("CREATE TABLE unrelated (x)")
    os.replace(replacement, path)
    manager_cls(db_path=str(path))
    assert has_table()

    # Present but empty, as a bare sqlite3.connect leaves it.
    os.rename(path, tmp_path / "moved_away_2.db")
    path.touch()
    manager_cls(db_path=str(path))
    assert has_table()


def test_read_only_manager_never_inits(tmp_path):
    path = str(tmp_path / "ro.db")
    DatabaseManager(db_path=path)
    with patch.object(DatabaseManager, "init_db") as init:
        reader = DatabaseManager(db_path=path, read_only=True)
    init.assert_not_called()
    assert "drop_id" in reader._deployments_columns
    with pytest.raises(sqlite3.OperationalError):
        DatabaseManager(db_path=str(tmp_path / "missing.db"), read_only=True)


def test_blank_video_link_falls_back_to_the_standard_key(temp_db):
    """A blank VideoFileLink arrives as NaN; str(NaN) is "nan", which used to be
    stored as the path, so a video at the standard key was marked absent
    (SLI_20260114_BUV_SLI_009_02). Found by convention: present, plus an issue."""

    class _Recorder:
        def add_or_update_deployments(self, deployments, validation_errors=None):
            self.rows = list(deployments)
            self.errors = list(validation_errors or [])

    df, known_files, mapping = _ingest_frame(8)
    link = config.csv_video_file_link_column
    df[link] = df[link].astype(object)
    df.loc[4, link] = float("nan")  # on S3 at the standard key
    df.loc[3, link] = float("nan")  # not on S3 at all
    rec = _Recorder()
    _sync_deployments_to_db(
        df, rec, set(), known_files, dict.fromkeys(known_files), mapping
    )

    by_id = {r["drop_id"]: r for r in rec.rows}
    ids = list(df[config.drop_id_column])
    assert by_id[ids[4]]["video_path"] == config.get_video_s3_key(ids[4])
    assert by_id[ids[4]]["video_presence"] == VideoPresence.PRESENT
    assert by_id[ids[4]]["ml_status"] == MlStatus.READY
    assert by_id[ids[3]]["video_path"] == config.get_video_s3_key(ids[3])
    assert by_id[ids[3]]["video_presence"] == VideoPresence.ABSENT
    link_issues = [
        e
        for e in rec.errors
        if e["ErrorType"] == INGEST_ISSUE and e["DropID"] == ids[4]
    ]
    assert len(link_issues) == 1
    assert not [
        e for e in rec.errors if e["DropID"] == ids[3]
    ]  # absent is its own signal
