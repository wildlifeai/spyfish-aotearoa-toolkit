import logging
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from spyfish.config.base import (
    SECTIONS,
    IngestStatus,
    InvalidTransitionError,
    MlStatus,
    VideoPresence,
)
from spyfish.config.wrapper import config


def parse_geo_value(value) -> Optional[float]:
    """Coerce a CSV coordinate/depth cell to float, or None.

    Returns None for blanks, non-numeric text and 0. Zero is explicitly allowed by
    the Latitude/Longitude range rules in config.yaml as a "not recorded" sentinel,
    but 0,0 is a real place in the Atlantic, storing it as a number would put
    deployments off the coast of Africa on any map. None is the honest answer.

    Note this must NOT go through str() like the text columns do, or a missing
    value becomes the string "nan" instead of NULL.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number == 0:  # NaN or the not-recorded sentinel
        return None
    return number


def _clean_protection_status(value) -> str:
    """Normalise a ProtectionStatus cell so variants collapse to one category.

    The source data carries "No protection", "No Protection" and "No protection "
    as three spellings of one thing, which renders as three separate bars on any
    chart. Whitespace is normalised here; case variants are resolved through the
    `protection_status_aliases` map in config.yaml, keyed by the lowercased form.
    """
    text = " ".join(str(value or "").split())
    if not text or text.lower() == "nan":
        return ""
    # Case-insensitive match against the canonical list first, so any casing of
    # a known status stores as the one spelling without an alias entry per
    # variant. The alias map is left for genuine synonyms, where the source
    # words differ rather than just their case.
    canonical = {known.lower(): known for known in config.known_protection_statuses}
    if text.lower() in canonical:
        return canonical[text.lower()]
    return config.protection_status_aliases.get(text.lower(), text)


def _add_column_if_missing(cursor, table: str, column: str, decl: str) -> None:
    """Idempotent ``ALTER TABLE ... ADD COLUMN`` for in-place schema migration.

    Re-raises anything that is not a duplicate-column error, so a genuine schema
    fault still surfaces instead of being swallowed. `table` and `column` are
    always code literals here, never user input.
    """
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


class DatabaseManager:
    """
    Core SQLite database manager for the Spyfish pipeline.
    Uses pure sqlite3 with dict-like row factories for simplicity.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = (
            str(config.db_path) if db_path is None else str(Path(db_path).absolute())
        )
        self.init_db()
        self._deployments_columns: frozenset[str] = self._read_deployments_columns()

    def _read_deployments_columns(self) -> frozenset[str]:
        """Reads the actual column names from the deployments table schema.

        Used by validate_column() so the injection whitelist is derived from
        the CREATE TABLE statement rather than a hand-maintained copy of it.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(deployments)")
            return frozenset(row["name"] for row in cursor.fetchall())

    def get_connection(self):
        """Returns a configured SQLite connection wrapped in contextlib.closing.

        ⚠️  DOES NOT auto-commit. Use this for SELECT queries. For DML
        (INSERT/UPDATE/DELETE), prefer get_writable_connection() which
        commits on exit and rolls back on exception.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return closing(conn)

    @contextmanager
    def get_writable_connection(self):
        """Yields a SQLite connection that auto-commits on clean exit.

        Use this for any DML (INSERT/UPDATE/DELETE). On exception, the
        transaction is rolled back and the exception re-raised.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Creates the core 'deployments' table if it does not exist."""
        logging.info(f"Initializing database at {self.db_path}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # We track the bare minimum needed for orchestration. Metadata lives in BUV Deployments.csv
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS deployments (
                    drop_id TEXT PRIMARY KEY,
                    video_path TEXT,
                    video_presence TEXT NOT NULL DEFAULT 'absent',
                    priority INTEGER NOT NULL DEFAULT 0,
                    ingest_status TEXT NOT NULL DEFAULT 'ok',
                    ml_status TEXT NOT NULL DEFAULT 'ml_pending',
                    citsci_status TEXT NOT NULL DEFAULT 'citsci_pending',
                    expert_status TEXT NOT NULL DEFAULT 'expert_pending',
                    reporting_status TEXT NOT NULL DEFAULT 'reporting_pending',
                    is_bad_deployment BOOLEAN NOT NULL DEFAULT 0,
                    sampling_start INTEGER,
                    sampling_end INTEGER,
                    ml_annotations INTEGER DEFAULT 0,
                    citsci_annotations INTEGER DEFAULT 0,
                    expert_annotations INTEGER DEFAULT 0,
                    biigle_volume_id TEXT,
                    training_biigle_volume_id INTEGER,
                    latitude REAL,
                    longitude REAL,
                    depth REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Idempotent migration for pre-existing DBs created before
            # training_biigle_volume_id was part of the schema.
            try:
                cursor.execute(
                    "ALTER TABLE deployments ADD COLUMN training_biigle_volume_id INTEGER"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

            # Where the drop actually landed, and how deep. Distinct from the
            # planned position on `sites`, comparing the two surfaces mis-sited
            # drops. Both coordinate pairs are in config `sensitive_columns`.
            _add_column_if_missing(cursor, "deployments", "latitude", "REAL")
            _add_column_if_missing(cursor, "deployments", "longitude", "REAL")
            _add_column_if_missing(cursor, "deployments", "depth", "REAL")

            # Idempotent rename of biigle_status → expert_status. The column
            # represents data-presence semantics (does this drop have expert
            # annotations from any source), not BIIGLE-pipeline-specific state.
            # See ExpertStatus docstring in config/base.py for rationale.
            cursor.execute("PRAGMA table_info(deployments)")
            cols = {row[1] for row in cursor.fetchall()}
            if "biigle_status" in cols and "expert_status" not in cols:
                cursor.execute(
                    "ALTER TABLE deployments RENAME COLUMN biigle_status TO expert_status"
                )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sites (
                    site_id TEXT PRIMARY KEY,
                    site_name TEXT,
                    link_to_marine_reserve TEXT,
                    protection_status TEXT,
                    region TEXT,
                    latitude REAL,
                    longitude REAL
                )
            """
            )

            # `region` is the only geographic grouping the pipeline has, there is
            # no region column on deployments and none derivable from a DropID.
            # latitude/longitude here are the PLANNED site position.
            _add_column_if_missing(cursor, "sites", "region", "TEXT")
            _add_column_if_missing(cursor, "sites", "latitude", "REAL")
            _add_column_if_missing(cursor, "sites", "longitude", "REAL")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    SurveyID TEXT,
                    DropID TEXT,
                    ErrorType TEXT,
                    FileName TEXT,
                    ColumnName TEXT,
                    ErrorMessage TEXT,
                    InvalidValue TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """
            )

            # Create a trigger to automatically update updated_at
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_deployments_timestamp
                AFTER UPDATE ON deployments
                BEGIN
                    UPDATE deployments SET updated_at = CURRENT_TIMESTAMP WHERE drop_id = NEW.drop_id;
                END;
            """
            )

            conn.commit()

    # ── Pipeline metadata KV store ───────────────────────────────────────────

    def get_metadata(self, key: str) -> Optional[str]:
        """Read a value from the pipeline_metadata table, or None if not set."""
        with self.get_connection() as conn:
            row = (
                conn.cursor()
                .execute("SELECT value FROM pipeline_metadata WHERE key = ?", (key,))
                .fetchone()
            )
            return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Upsert a key-value pair in the pipeline_metadata table."""
        with self.get_connection() as conn:
            conn.cursor().execute(
                "INSERT INTO pipeline_metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()

    def upsert_sites(self, sites_df) -> None:
        """Replace all site metadata from BUV Survey Sites DataFrame.

        Full replace (delete + insert) rather than upsert, sites are config data with no
        pipeline state, so removed sites should not linger in the DB.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sites")
            skipped = 0
            for _, row in sites_df.iterrows():
                site_id = str(row.get(config.site_id_column, "")).strip()
                if not site_id or site_id.lower() == "nan":
                    skipped += 1
                    continue
                cursor.execute(
                    """
                    INSERT INTO sites (
                        site_id, site_name, link_to_marine_reserve, protection_status,
                        region, latitude, longitude
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(site_id) DO UPDATE SET
                        site_name=excluded.site_name,
                        link_to_marine_reserve=excluded.link_to_marine_reserve,
                        protection_status=excluded.protection_status,
                        region=excluded.region,
                        latitude=excluded.latitude,
                        longitude=excluded.longitude
                    """,
                    (
                        site_id,
                        str(row.get(config.site_name_column, "")),
                        str(row.get(config.link_to_marine_reserve_column, "")),
                        _clean_protection_status(
                            row.get(config.protection_status_column)
                        ),
                        # Region is not on this sheet, it is resolved from
                        # Marine Reserves.csv via LinkToMarineReserve, so the
                        # caller passes it in on the row when available.
                        str(row.get(config.region_column, "")).strip(),
                        # Site coordinates are the PLANNED position and are named
                        # Targeted* upstream; the actual fix lives on deployments.
                        parse_geo_value(row.get(config.targeted_latitude_column)),
                        parse_geo_value(row.get(config.targeted_longitude_column)),
                    ),
                )
            if skipped:
                logging.warning(
                    f"Skipped {skipped} site rows with missing/empty site_id, check column mapping in config.yaml."
                )
            conn.commit()
        logging.info(f"Upserted {len(sites_df) - skipped} sites into DB.")

    def get_site(self, site_id: str) -> Optional[Dict[str, str]]:
        """Fetch site metadata by SiteID. Returns a config-keyed dict or None if not found."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            config.site_id_column: row["site_id"],
            config.site_name_column: row["site_name"],
            config.link_to_marine_reserve_column: row["link_to_marine_reserve"],
            config.protection_status_column: row["protection_status"],
            config.region_column: row["region"],
            config.latitude_column: row["latitude"],
            config.longitude_column: row["longitude"],
        }

    def add_or_update_deployment(
        self,
        drop_id: str,
        *,
        ingest_status: str = IngestStatus.OK,
        ml_status: str = MlStatus.PENDING,
        video_path: str = "",
        video_presence: str = VideoPresence.ABSENT,
        is_bad_deployment: bool = False,
        sampling_start: Optional[int] = None,
        sampling_end: Optional[int] = None,
        biigle_volume_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        depth: Optional[float] = None,
    ) -> None:
        """Insert a deployment, or update its metadata if it already exists.

        INSERT-only (ignored on conflict): ml_status, and the citsci/biigle/
        reporting status columns (set by SQL defaults on insert only).
        UPDATE-on-conflict: ingest_status, video_path, video_presence,
        is_bad_deployment, sampling_start, sampling_end, biigle_volume_id,
        latitude, longitude, depth.
        Annotation counts (ml/citsci/expert) are owned by sync_annotation_counts.

        latitude/longitude/depth are COALESCEd so a re-ingest from a CSV that has
        blanked them does not erase a previously recorded position.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO deployments (
                    drop_id, video_path, video_presence,
                    ingest_status, ml_status,
                    is_bad_deployment, sampling_start, sampling_end,
                    biigle_volume_id, latitude, longitude, depth
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drop_id) DO UPDATE SET
                    video_path=excluded.video_path,
                    video_presence=excluded.video_presence,
                    ingest_status=excluded.ingest_status,
                    is_bad_deployment=excluded.is_bad_deployment,
                    sampling_start=excluded.sampling_start,
                    sampling_end=excluded.sampling_end,
                    biigle_volume_id=COALESCE(excluded.biigle_volume_id, deployments.biigle_volume_id),
                    latitude=COALESCE(excluded.latitude, deployments.latitude),
                    longitude=COALESCE(excluded.longitude, deployments.longitude),
                    depth=COALESCE(excluded.depth, deployments.depth)
                """,
                (
                    drop_id,
                    video_path,
                    video_presence,
                    ingest_status,
                    ml_status,
                    is_bad_deployment,
                    sampling_start,
                    sampling_end,
                    biigle_volume_id,
                    latitude,
                    longitude,
                    depth,
                ),
            )
            conn.commit()

    def validate_column(self, column: str) -> None:
        """Rejects any column name not in the deployments table schema.

        Caller-supplied column names (section, prerequisite keys) MUST be
        checked against the real schema before being interpolated into SQL.
        The allowed set is derived from `PRAGMA table_info(deployments)` at
        init time, there is no hand-maintained list to drift from the schema.
        """
        if column not in self._deployments_columns:
            raise ValueError(
                f"Invalid column name {column!r}. "
                f"Allowed: {sorted(self._deployments_columns)}"
            )

    def update_section_status(self, drop_id: str, section: str, new_status: str):
        """Updates a specific status column for a deployment.

        When transitioning out of the section's ERROR value, clears that
        section's rows from validation_errors so a retry starts clean. This
        is the low-level setter, bypasses transition validation. Pipeline
        code should use `advance_status()` instead.
        """
        self.validate_column(section)
        status_cls = SECTIONS.get(section)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT {section} FROM deployments WHERE drop_id = ?", (drop_id,)
            )
            row = cursor.fetchone()
            current_status = row[section] if row else None

            cursor.execute(
                f"UPDATE deployments SET {section} = ? WHERE drop_id = ?",
                (new_status, drop_id),
            )

            # If we're leaving an error state, clear this section's errors.
            # `ingest_status` has no ERROR value so status_cls can be None.
            if (
                status_cls is not None
                and current_status == status_cls.ERROR
                and new_status != status_cls.ERROR
            ):
                cursor.execute(
                    "DELETE FROM validation_errors WHERE DropID = ? AND ErrorType = ?",
                    (drop_id, status_cls.ERROR),
                )
            conn.commit()

    def bulk_update_section_status(
        self,
        drop_ids: List[str],
        section: str,
        new_status: str,
        skip_if_in: Optional[List[str]] = None,
    ) -> int:
        """Bulk-set a section's status across many drops in one transaction.

        Designed for ingest paths that need to advance hundreds of drops at
        once (legacy CSV ingest, bootstrap orchestrator), calling
        `advance_status` in a Python loop is ~3 roundtrips per drop and
        gets slow fast.

        Bypasses transition validation by design; the caller is expected
        to have verified the move is correct in aggregate (e.g. "any
        drop with expert annotations → expert_complete"). For per-drop
        validated transitions, use `advance_status` in a loop.

        Args:
            drop_ids: drops to update. Empty list = no-op.
            section: column name (e.g. ExpertStatus.COLUMN).
            new_status: target value (e.g. ExpertStatus.COMPLETE).
            skip_if_in: optional list of current-status values to leave
                untouched (e.g. don't overwrite COMPLETE or SKIPPED).

        Returns:
            Number of rows actually changed.
        """
        if not drop_ids:
            return 0
        self.validate_column(section)
        status_cls = SECTIONS.get(section)

        drop_placeholders = ",".join("?" * len(drop_ids))
        where_skip = ""
        params: List[Any] = [new_status] + list(drop_ids) + [new_status]
        if skip_if_in:
            skip_placeholders = ",".join("?" * len(skip_if_in))
            where_skip = f" AND {section} NOT IN ({skip_placeholders})"
            params.extend(skip_if_in)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE deployments
                SET {section} = ?
                WHERE drop_id IN ({drop_placeholders})
                  AND {section} != ?
                  {where_skip}
                """,
                params,
            )
            n_changed = cursor.rowcount

            # If the target isn't ERROR, clear any error rows for these drops
            # in this section, mirrors `update_section_status`'s behaviour.
            if (
                status_cls is not None
                and new_status != status_cls.ERROR
                and n_changed > 0
            ):
                cursor.execute(
                    f"""
                    DELETE FROM validation_errors
                    WHERE DropID IN ({drop_placeholders})
                      AND ErrorType = ?
                    """,
                    list(drop_ids) + [status_cls.ERROR],
                )
            conn.commit()
        return n_changed

    def advance_status(self, drop_id: str, section: str, to_status: str) -> None:
        """Validated state-machine transition for any section.

        Looks up the status class from the SECTIONS registry, checks that
        the target transition is in VALID_TRANSITIONS, and delegates the
        write to `update_section_status`. Raises `InvalidTransitionError`
        for disallowed moves, `KeyError` for unknown drop_ids, and
        `ValueError` for unknown section column names.

        Usage:
            db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.COMPLETE)
        """
        if section not in SECTIONS:
            raise ValueError(f"Unknown section {section!r}. Known: {sorted(SECTIONS)}")
        status_cls = SECTIONS[section]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {section} FROM deployments WHERE drop_id = ?", (drop_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"No deployment found with drop_id={drop_id!r}")
            current = row[section]

        allowed = status_cls.VALID_TRANSITIONS.get(current, set())
        if to_status not in allowed:
            raise InvalidTransitionError(
                f"{drop_id}: invalid {status_cls.__name__} transition "
                f"{current!r} → {to_status!r}"
            )
        self.update_section_status(drop_id, section, to_status)

    def update_deployment_fields(self, drop_id: str, **fields) -> bool:
        """Update metadata columns on a deployment record. Returns False if drop_id not found."""
        allowed = {
            "ingest_status",
            "video_path",
            "video_presence",
            "priority",
            "sampling_start",
            "sampling_end",
            "is_bad_deployment",
            "biigle_volume_id",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Unknown fields: {invalid}. Allowed: {allowed}")
        if not fields:
            return True
        set_clause = ", ".join(f"{col} = ?" for col in fields)
        values = list(fields.values()) + [drop_id]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE deployments SET {set_clause} WHERE drop_id = ?", values
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_biigle_volume_id(self, drop_id: str, volume_id: str):
        """Sets the biigle_volume_id for a specific deployment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE deployments SET biigle_volume_id = ? WHERE drop_id = ?",
                (str(volume_id), drop_id),
            )
            conn.commit()

    def update_training_biigle_volume_id(self, drop_id: str, volume_id: int) -> None:
        """Sets the training_biigle_volume_id for a specific deployment.

        Called only after the training-frames batch has been successfully uploaded
        to a Biigle volume, a non-NULL value means "this drop's training frames
        are in Biigle volume <id>".
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE deployments SET training_biigle_volume_id = ? WHERE drop_id = ?",
                (int(volume_id), drop_id),
            )
            conn.commit()

    def get_training_biigle_volume_id(self, drop_id: str) -> Optional[int]:
        """Returns the training_biigle_volume_id for a drop, or None if not set."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT training_biigle_volume_id FROM deployments WHERE drop_id = ?",
                (drop_id,),
            )
            row = cursor.fetchone()
        if row is None or row["training_biigle_volume_id"] is None:
            return None
        return int(row["training_biigle_volume_id"])

    def get_drops_for_survey_with_video_window(self, survey_id: str) -> List[str]:
        """Returns drop_ids in `survey_id` that have a downloadable video AND a
        defined sampling window, the eligibility set for training-frame extraction.

        Filters:
          - drop_id starts with `{survey_id}_` (e.g. AHE_20250513_BUV → AHE_20250513_BUV_*)
          - video_presence = 'present' (excludes ABSENT, ARCHIVED, NO_VIDEO_BAD_DEP)
          - sampling_start AND sampling_end are both set

        Deliberately does NOT filter on is_bad_deployment or ingest_status,
        per-spec, bad/short deployments still produce useful training frames
        as long as they have a video and a sampling window.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                r"""
                SELECT drop_id FROM deployments
                WHERE drop_id LIKE ? || '\_%' ESCAPE '\'
                  AND video_presence = ?
                  AND sampling_start IS NOT NULL
                  AND sampling_end IS NOT NULL
                ORDER BY drop_id
                """,
                (survey_id, VideoPresence.PRESENT),
            )
            return [row["drop_id"] for row in cursor.fetchall()]

    def get_deployments_eligible(
        self,
        section: str,
        statuses: List[str],
        prerequisites: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns deployments eligible for processing by a pipeline stage.

        Filters to ingest_status='ok' (excludes bad/errored/removed deployments),
        checks that `section` is in `statuses`, and optionally checks additional
        prerequisites.

        ``prerequisites`` values may be a single string (column = value) or a
        list (column IN (...)). The list form supports cases like "BIIGLE can
        proceed when citsci is complete OR skipped".

        Always orders by priority DESC so high-priority drops are processed first.
        """
        self.validate_column(section)
        if not statuses:
            return []
        placeholders = ", ".join(["?"] * len(statuses))
        params: List[Any] = list(statuses)
        where = [f"{section} IN ({placeholders})", "ingest_status = 'ok'"]
        if prerequisites:
            for col, val in prerequisites.items():
                self.validate_column(col)
                if isinstance(val, (list, tuple, set)):
                    vals = list(val)
                    if not vals:
                        continue
                    in_placeholders = ", ".join(["?"] * len(vals))
                    where.append(f"{col} IN ({in_placeholders})")
                    params.extend(vals)
                else:
                    where.append(f"{col} = ?")
                    params.append(val)
        query = f"SELECT * FROM deployments WHERE {' AND '.join(where)} ORDER BY priority DESC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_biigle_volumes_awaiting_sync(
        self, expert_status: str
    ) -> List[Dict[str, Any]]:
        """Returns deployments that have a Biigle volume assigned AND are in the given expert_status.

        Used by the Biigle annotation-sync stage to find volumes that have been
        uploaded but not yet marked complete. Deliberately does NOT filter by
        `ingest_status='ok'`, a deployment can be excluded *after* its Biigle
        volume was created, and we still want to sync back any annotations the
        experts produced before it was flagged.
        """
        self.validate_column(
            "expert_status"
        )  # defense in depth even with hardcoded value
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT drop_id, biigle_volume_id
                FROM deployments
                WHERE biigle_volume_id IS NOT NULL
                  AND expert_status = ?
                """,
                (expert_status,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_max_priority(self) -> int:
        """Returns the current maximum priority value across all deployments (0 if none set)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(priority), 0) FROM deployments")
            return cursor.fetchone()[0]

    def get_deployment(self, drop_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific deployment by drop_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments WHERE drop_id = ?", (drop_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_deployments_by_ids(self, drop_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple deployments in a single query. Returns {drop_id: record}."""
        if not drop_ids:
            return {}
        placeholders = ", ".join(["?"] * len(drop_ids))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM deployments WHERE drop_id IN ({placeholders})", drop_ids
            )
            return {row["drop_id"]: dict(row) for row in cursor.fetchall()}

    def get_all_deployments_map(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all deployment records and return them as a dictionary {drop_id: record}."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM deployments ORDER BY priority DESC")
            return {row["drop_id"]: dict(row) for row in cursor.fetchall()}

    def clear_validation_errors(self):
        """Clears all validation errors from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM validation_errors")
            conn.commit()

    def add_validation_error(
        self,
        *,
        survey_id: str,
        drop_id: str,
        error_type: str,
        column_name: str,
        error_message: str,
        file_name: str = "",
        invalid_value: str = "",
    ) -> None:
        """Insert a single validation error with explicit named fields.

        Typed wrapper around `add_validation_errors` for the common case of
        recording one error at a time (e.g. an ML inference failure). Forces
        keyword-only arguments so the field mapping can't be accidentally
        positional.
        """
        self.add_validation_errors(
            [
                {
                    "SurveyID": survey_id,
                    "DropID": drop_id,
                    "ErrorType": error_type,
                    "FileName": file_name,
                    "ColumnName": column_name,
                    "ErrorMessage": error_message,
                    "InvalidValue": invalid_value,
                }
            ]
        )

    def add_validation_errors(self, errors: List[Dict[str, Any]]) -> None:
        """Bulk insert validation errors.

        Each dict must have keys: SurveyID, DropID, ErrorType, FileName,
        ColumnName, ErrorMessage, InvalidValue. For single-error inserts
        prefer `add_validation_error()` which validates the field names
        at the call site.
        """
        if not errors:
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO validation_errors (SurveyID, DropID, ErrorType, FileName, ColumnName, ErrorMessage, InvalidValue)
                VALUES (:SurveyID, :DropID, :ErrorType, :FileName, :ColumnName, :ErrorMessage, :InvalidValue)
            """,
                errors,
            )
            conn.commit()

    def get_all_validation_errors(self) -> List[Dict[str, Any]]:
        """Returns all stored validation errors (updated to not fetch global status)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT v.SurveyID, v.DropID, v.ErrorType, v.FileName, v.ColumnName, v.ErrorMessage, v.InvalidValue
                FROM validation_errors v
            """
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def sync_annotation_counts(self, drop_ids: Optional[List[str]] = None):
        """
        Aggregates counts from spyfish_annotations.db and updates the deployments table in spyfish_pipeline.db.
        If drop_ids is provided, only updates those specific deployments.
        """
        logging.info(
            f"Syncing annotation counts to main pipeline database{' (incremental)' if drop_ids else ''}..."
        )

        # Import to avoid circular dependencies if any
        from spyfish.database.annotation_manager import AnnotationDatabaseManager

        ann_db = AnnotationDatabaseManager()

        query = """
            SELECT drop_id, annotated_by as source, COUNT(*) as total
            FROM annotations
        """
        params = []
        if drop_ids:
            placeholders = ", ".join(["?"] * len(drop_ids))
            query += f" WHERE drop_id IN ({placeholders})"
            params.extend(drop_ids)

        query += " GROUP BY drop_id, annotated_by"

        with ann_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()

        counts_by_drop = {
            d: {"ml": 0, "expert": 0, "citsci": 0} for d in (drop_ids or [])
        }
        for row in results:
            drop_id = row["drop_id"]
            source = row["source"]
            count = row["total"]

            if drop_id not in counts_by_drop:
                counts_by_drop[drop_id] = {"ml": 0, "expert": 0, "citsci": 0}

            if source == "ml":
                counts_by_drop[drop_id]["ml"] = count
            elif source == "expert":
                counts_by_drop[drop_id]["expert"] = count
            elif source == "citsci":
                counts_by_drop[drop_id]["citsci"] = count

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for drop_id, counts in counts_by_drop.items():
                cursor.execute(
                    """
                    UPDATE deployments
                    SET ml_annotations = ?,
                        expert_annotations = ?,
                        citsci_annotations = ?
                    WHERE drop_id = ?
                """,
                    (counts["ml"], counts["expert"], counts["citsci"], drop_id),
                )
            conn.commit()

        # Advance section status for any drop that now has annotations.
        # Data presence overrides intent: a drop with annotations is complete
        # for that section regardless of prior pipeline state (including
        # SKIPPED, if someone produced annotations anyway, they're done).
        # Skip only drops already at COMPLETE for that section (idempotent).
        # The three (source → column → complete-value) tuples are part of
        # the fixed schema contract; hardcoded here to keep DatabaseManager
        # decoupled from the per-section status classes.
        section_map = [
            ("ml", "ml_status", "ml_complete"),
            ("citsci", "citsci_status", "citsci_complete"),
            ("expert", "expert_status", "expert_complete"),
        ]
        total_advanced = 0
        for source, section_col, complete_val in section_map:
            drops_with_data = [d for d, c in counts_by_drop.items() if c[source] > 0]
            if not drops_with_data:
                continue
            n = self.bulk_update_section_status(
                drops_with_data,
                section_col,
                complete_val,
                skip_if_in=[complete_val],
            )
            if n:
                logging.info(
                    f"sync_annotation_counts: advanced {section_col} → "
                    f"{complete_val} for {n} drop(s)."
                )
            total_advanced += n

        logging.info(
            f"Updated annotation counts for {len(counts_by_drop)} deployments "
            f"(advanced status on {total_advanced} drop-section(s))."
        )
