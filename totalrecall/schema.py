"""SQLite schema definitions and initialization for TotalRecall."""

import sqlite3
from pathlib import Path


MEMORY_LOG_SCHEMA = """
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
"""

TOTAL_RECALL_SCHEMA = """
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


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize a SQLite database with WAL mode and return the connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_log(db_dir: Path) -> sqlite3.Connection:
    """Initialize the raw command log database."""
    conn = init_db(db_dir / "memory_log.db")
    conn.executescript(MEMORY_LOG_SCHEMA)
    conn.commit()
    return conn


def init_total_recall(db_dir: Path) -> sqlite3.Connection:
    """Initialize the compressed memory database."""
    conn = init_db(db_dir / "total_recall.db")
    conn.executescript(TOTAL_RECALL_SCHEMA)
    conn.commit()
    return conn
