"""MemorySystem — the unified facade over all memory subsystems.

Also provides whole-brain export/import for portability and safe sharing.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from ..config import BrainConfig
from .store import Store, atomic_write_json
from .working import WorkingMemory
from .episodic import EpisodicMemory, Episode
from .semantic import SemanticMemory, Concept
from .procedural import ProceduralMemory, Skill


class MemorySystem:
    def __init__(self, config: BrainConfig):
        self.config = config
        config.ensure_dirs()
        self.store = Store(config.db_path)
        self.working = WorkingMemory(
            config.memory, snapshot_path=config.memory_path("working", "current.json"))
        self.episodic = EpisodicMemory(self.store, config.memory)
        self.semantic = SemanticMemory(self.store, config.memory)
        self.procedural = ProceduralMemory(self.store, config.memory)

    # --------------------------------------------------------------- export
    def export_all(self, path: Optional[str] = None) -> str:
        """Export the whole long-term memory to one JSON file (atomic)."""
        path = path or self.config.memory_path(
            "exports", f"memory-export-{int(time.time())}.json")
        data = {
            "schema": 1,
            "exported_at": time.time(),
            "episodes": [dict(r) for r in
                         self.store.execute("SELECT * FROM episodes").fetchall()],
            "concepts": [dict(r) for r in
                         self.store.execute("SELECT * FROM concepts").fetchall()],
            "edges": [dict(r) for r in
                      self.store.execute("SELECT * FROM edges").fetchall()],
            "skills": [dict(r) for r in
                       self.store.execute("SELECT * FROM skills").fetchall()],
            "contradictions": [dict(r) for r in
                               self.store.execute("SELECT * FROM contradictions").fetchall()],
            "archive": [dict(r) for r in
                        self.store.execute("SELECT * FROM archive").fetchall()],
            "learn_log": [dict(r) for r in
                          self.store.execute("SELECT * FROM learn_log").fetchall()],
        }
        atomic_write_json(path, data)
        return path

    def import_all(self, path: str) -> dict:
        """Import a memory export; existing ids win (never overwrite silently:
        conflicting imports are skipped instead of duplicated)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") != 1:
            raise ValueError("unsupported export schema")
        counts = {"episodes": 0, "concepts": 0, "edges": 0, "skills": 0}
        with self.store.transaction() as cur:
            for ep in data.get("episodes", []):
                existing = cur.execute("SELECT 1 FROM episodes WHERE id=?",
                                       (ep["id"],)).fetchone()
                if existing:
                    continue
                cur.execute(
                    """INSERT INTO episodes (id, situation, objective, actions, result,
                       status, mistakes, lessons, timestamp, source, confidence,
                       salience, domain, created, last_accessed, access_count,
                       consolidated, archived, compressed, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ep["id"], ep["situation"], ep["objective"], ep["actions"],
                     ep["result"], ep["status"], ep["mistakes"], ep["lessons"],
                     ep["timestamp"], ep["source"], ep["confidence"], ep["salience"],
                     ep["domain"], ep["created"], ep["last_accessed"],
                     ep["access_count"], ep["consolidated"], ep["archived"],
                     ep.get("compressed"), ep.get("embedding")))
                counts["episodes"] += 1
            for c in data.get("concepts", []):
                existing = cur.execute("SELECT 1 FROM concepts WHERE id=?",
                                       (c["id"],)).fetchone()
                if existing:
                    continue
                cur.execute(
                    """INSERT INTO concepts (id, name, definition, domain, confidence,
                       status, provenance, support, importance, created, updated,
                       last_accessed, access_count, outdated, archived, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (c["id"], c["name"], c["definition"], c["domain"], c["confidence"],
                     c["status"], c["provenance"], c["support"], c["importance"],
                     c["created"], c["updated"], c["last_accessed"], c["access_count"],
                     c["outdated"], c["archived"], c.get("embedding")))
                counts["concepts"] += 1
            for e in data.get("edges", []):
                cur.execute(
                    "INSERT INTO edges(src, dst, relation, weight) VALUES (?,?,?,?) "
                    "ON CONFLICT(src, dst, relation) DO UPDATE SET "
                    "weight=MAX(weight, excluded.weight)",
                    (e["src"], e["dst"], e["relation"], e["weight"]))
                counts["edges"] += 1
            for s in data.get("skills", []):
                existing = cur.execute("SELECT 1 FROM skills WHERE id=?",
                                       (s["id"],)).fetchone()
                if existing:
                    continue
                cur.execute(
                    """INSERT INTO skills (id, name, description, triggers, steps,
                       trust, success_count, failure_count, provenance, created,
                       last_used, archived)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (s["id"], s["name"], s["description"], s["triggers"], s["steps"],
                     s["trust"], s["success_count"], s["failure_count"],
                     s["provenance"], s["created"], s.get("last_used"),
                     s.get("archived", 0)))
                counts["skills"] += 1
        return counts

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        return {
            "working": self.working.stats(),
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "procedural": self.procedural.stats(),
            "db_path": self.config.db_path,
            "db_bytes": os.path.getsize(self.config.db_path)
            if os.path.exists(self.config.db_path) else 0,
        }

    def backup(self) -> str:
        return self.store.backup() or ""

    def close(self) -> None:
        self.store.checkpoint_wal()
        self.store.close()
