from __future__ import annotations

from pathlib import Path
import sqlite3


_MIGRATION_PATH = (
    Path(__file__).resolve().parent
    / "migrations"
    / "001_phase_1_core_hierarchy.sql"
)


def apply_phase_1_core_schema(connection: sqlite3.Connection) -> None:
    """Apply the isolated Phase 1 core hierarchy schema to a SQLite connection."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_MIGRATION_PATH.read_text(encoding="utf-8"))
