"""Connection management: one connection per thread, the store hands them out."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from ..errors import StoreError
from .migrations import MIGRATIONS, SCHEMA_VERSION

BUSY_TIMEOUT_MS = 5000


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


schema_migrations = """
CREATE TABLE IF NOT EXISTS schema_migrations(
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);"""


class Store:
    """Owner of the sqlite file; all repo functions take it as their first argument."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StoreError(f"cannot create data directory {self.path.parent}: {exc}") from exc
        self._local = threading.local()
        self._connect_check()

    # -- per-thread connection ------------------------------------------------

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
            except sqlite3.Error as exc:
                raise StoreError(f"cannot open database {self._path}: {exc}") from exc
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            apply_migrations(conn)
            self._local.conn = conn
        return conn

    def _connect_check(self) -> None:
        try:
            self.connection().execute("SELECT 1")
        except (sqlite3.Error, StoreError) as exc:  # pragma: no cover - disk-level failure
            raise StoreError(f"database check failed: {exc}") from exc

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @property
    def schema_version(self) -> int:
        row = (
            self.connection()
            .execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
            .fetchone()
        )
        return int(row[0])


def apply_migrations(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(schema_migrations)
        applied = {int(row[0]) for row in conn.execute("SELECT version FROM schema_migrations")}
        if applied and max(applied, default=0) > SCHEMA_VERSION:
            raise StoreError(
                f"database schema v{max(applied)} is newer than this hyrip"
                f" (v{SCHEMA_VERSION}) supports"
            )
        for version, sql in enumerate(MIGRATIONS, start=1):
            if version not in applied:
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, now_utc_iso()),
                )
        conn.commit()
    except sqlite3.Error as exc:
        raise StoreError(f"schema migration failed: {exc}") from exc
