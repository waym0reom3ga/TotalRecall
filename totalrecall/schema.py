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

-- FTS5 index on information for semantic content search (fallback when tag matching fails)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    information,
    content='memories',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, information) VALUES (new.id, new.information);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF information ON memories
BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
    INSERT INTO memories_fts(rowid, information) VALUES (new.id, new.information);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories
BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
END;
"""


def init_db(db_dir: Path) -> sqlite3.Connection:
    """Initialize a single SQLite database with WAL mode and return the connection."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "totalrecall.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
