from __future__ import annotations

from pathlib import Path
import sqlite3


_SQLITE_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "sqlite_validation"
    / "phase_1_core_hierarchy_schema.sql"
)


def apply_phase_1_core_schema(connection: sqlite3.Connection) -> None:
    """Apply the isolated SQLite validation schema to a SQLite connection."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SQLITE_SCHEMA_PATH.read_text(encoding="utf-8"))
