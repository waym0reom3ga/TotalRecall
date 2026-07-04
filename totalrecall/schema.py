"""SQLite schema definitions and initialization for TotalRecall."""

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL DEFAULT (strftime('%s', 'now')),
    session_id TEXT NOT NULL DEFAULT '',
    chunk_number INTEGER NOT NULL DEFAULT -1,
    layer INTEGER NOT NULL DEFAULT 0,
    input TEXT NOT NULL DEFAULT '',
    output TEXT NOT NULL DEFAULT '',
    error_log TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_number INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level INTEGER NOT NULL CHECK(level >= 1),
    tags TEXT NOT NULL DEFAULT '[]',
    information TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
    source_chunk_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags);
CREATE INDEX IF NOT EXISTS idx_memories_level_created ON memories(level DESC, created_at DESC);
"""


def init_db(db_dir: Path) -> sqlite3.Connection:
    """Initialize a single SQLite database with WAL mode and return the connection."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "totalrecall.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
