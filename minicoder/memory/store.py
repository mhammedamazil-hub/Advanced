"""Persistent storage engine for the brain.

Design principles (per spec §4):

* SQLite (WAL mode) is the single source of truth — the *smallest practical*
  combination; JSON is used for exports/interop only, no external DB needed.
* Durable across restarts, interrupted sleep cycles and crashes:
  - every write happens inside a transaction (atomic commit),
  - rotating backups are taken before destructive maintenance,
  - on open we run ``PRAGMA integrity_check`` and restore the newest healthy
    backup if the live database is corrupt.
* Every record carries **provenance** (where the knowledge came from) and
  **confidence/verification** fields so nothing generated-but-unverified is
  ever treated as established fact.
* Thread-safe: a re-entrant store lock serializes access so the dashboard's
  request threads and background sleep threads can share one connection.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from typing import Any, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    situation TEXT NOT NULL,
    objective TEXT NOT NULL,
    actions TEXT NOT NULL,          -- JSON list
    result TEXT NOT NULL,
    status TEXT NOT NULL,           -- success | failure | unverified
    mistakes TEXT NOT NULL,         -- JSON list
    lessons TEXT NOT NULL,          -- JSON list
    timestamp REAL NOT NULL,
    source TEXT NOT NULL,           -- provenance: user|execution|hypothesis|docs|training|tool
    confidence REAL NOT NULL,
    salience REAL NOT NULL,
    domain TEXT,
    created REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    consolidated INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    compressed TEXT,
    embedding TEXT                  -- JSON vector
);
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    definition TEXT NOT NULL,
    domain TEXT,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,           -- fact | hypothesis | outdated
    provenance TEXT NOT NULL,       -- JSON list of provenance strings
    support INTEGER NOT NULL DEFAULT 1,
    importance REAL NOT NULL DEFAULT 0.5,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    outdated INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    embedding TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    relation TEXT NOT NULL,         -- related_to|is_a|part_of|supports|contradicts|prerequisite
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src, dst, relation)
);
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    triggers TEXT NOT NULL,         -- JSON list of keywords
    steps TEXT NOT NULL,            -- JSON list of {action, args}
    trust TEXT NOT NULL,            -- untrusted | verified
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    provenance TEXT NOT NULL,
    created REAL NOT NULL,
    last_used REAL,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sleep_cycles (
    id TEXT PRIMARY KEY,
    started REAL NOT NULL,
    ended REAL,
    status TEXT NOT NULL,           -- running | interrupted | completed
    report TEXT
);
CREATE TABLE IF NOT EXISTS sleep_checkpoints (
    cycle_id TEXT NOT NULL,
    stage INTEGER NOT NULL,
    payload TEXT,
    PRIMARY KEY (cycle_id, stage)
);
CREATE TABLE IF NOT EXISTS learn_log (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    level INTEGER NOT NULL,         -- 1..5 of the safe learning hierarchy
    summary TEXT NOT NULL,
    details TEXT NOT NULL           -- JSON
);
CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    details TEXT NOT NULL           -- JSON
);
CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    a TEXT NOT NULL,                -- concept ids
    b TEXT NOT NULL,
    note TEXT NOT NULL,
    created REAL NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS archive (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,             -- episode | concept | skill | model | config
    payload TEXT NOT NULL,          -- full JSON snapshot (recoverable)
    reason TEXT NOT NULL,
    archived_at REAL NOT NULL
);
"""

SCHEMA_VERSION = "1"
MAX_BACKUPS = 3


def new_id(*parts: Any, length: int = 16) -> str:
    """Deterministic-ish unique id from arbitrary content + time + salt."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
    h.update(str(time.time_ns()).encode())
    h.update(os.urandom(8))
    return h.hexdigest()[:length]


def atomic_write_json(path: str, obj: Any) -> None:
    """Atomic JSON file write: temp file + fsync + os.replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Store:
    """SQLite-backed persistent memory store with backups and recovery."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = self._open_or_recover(path)
        with self.conn:
            self.conn.executescript(SCHEMA)
            # migration: skills gain a 'studied' flag (sleep skill-training)
            try:
                self.conn.execute(
                    "ALTER TABLE skills ADD COLUMN studied INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass                              # column already exists
            self.set_meta("schema_version", SCHEMA_VERSION)

    # ------------------------------------------------------------------ open
    def _connect(self, path: str, busy_ms: int = 5000) -> sqlite3.Connection:
        # check_same_thread=False so dashboard request threads can read;
        # writes are serialized through self.lock + WAL single-writer.
        conn = sqlite3.connect(path, timeout=busy_ms / 1000,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={busy_ms}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _healthy(self, conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            return row is not None and row[0] == "ok"
        except sqlite3.DatabaseError:
            return False

    def _open_or_recover(self, path: str) -> sqlite3.Connection:
        if os.path.exists(path):
            try:
                conn = self._connect(path, busy_ms=800)
                if self._healthy(conn):
                    conn.execute("PRAGMA busy_timeout=5000")
                    return conn
                conn.close()
            except sqlite3.DatabaseError:
                pass
            # Corrupt main db — restore newest healthy backup.
            for backup in self._backup_paths():
                if os.path.exists(backup):
                    try:
                        candidate = self._connect(backup, busy_ms=800)
                        if self._healthy(candidate):
                            candidate.close()
                            shutil.copyfile(backup, path)
                            return self._connect(path)
                        candidate.close()
                    except sqlite3.DatabaseError:
                        continue
            # Nothing recoverable: quarantine the corrupt file and start fresh.
            shutil.copyfile(path, path + f".corrupt-{int(time.time())}")
            os.unlink(path)
        return self._connect(path)

    def _backup_paths(self) -> list[str]:
        return [self.path + f".bak.{i}" for i in range(1, MAX_BACKUPS + 1)]

    # ------------------------------------------------------------- utilities
    def set_meta(self, key: str, value: Any) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?",
                                    (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return row[0]

    def backup(self) -> Optional[str]:
        """Take a rotating backup using the SQLite backup API (consistent)."""
        with self.lock:
            backups = self._backup_paths()
            for i in range(MAX_BACKUPS - 1, 0, -1):
                if os.path.exists(backups[i - 1]):
                    if i + 1 > MAX_BACKUPS:
                        os.unlink(backups[i - 1])
                    else:
                        os.replace(backups[i - 1], backups[i])
            target = self.path + ".bak.1"
            dest = sqlite3.connect(target)
            try:
                self.conn.backup(dest)
            finally:
                dest.close()
            return target

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, tuple(params))

    def transaction(self):
        """Explicit atomic transaction that also serializes thread access."""
        return _Transaction(self)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def checkpoint_wal(self) -> None:
        with self.lock:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


class _Transaction:
    def __init__(self, store: "Store"):
        self.store = store

    def __enter__(self):
        self.store.lock.acquire()
        self.store.conn.execute("BEGIN IMMEDIATE")
        return self.store.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.store.conn.commit()
            else:
                self.store.conn.rollback()
        finally:
            self.store.lock.release()
        return False
