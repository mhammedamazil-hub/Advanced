"""Episodic memory — what the brain has actually experienced.

Each episode captures situation, objective, actions, result, verification
status, mistakes, lessons, timestamps, source and confidence.  The brain must
always be able to distinguish successful, failed and *unverified* experiences.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import MemoryConfig
from .store import Store, new_id
from . import textsim

SUCCESS, FAILURE, UNVERIFIED = "success", "failure", "unverified"


@dataclass
class Episode:
    situation: str
    objective: str
    actions: list[str]
    result: str
    status: str = UNVERIFIED       # success | failure | unverified
    mistakes: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    source: str = "user"           # provenance: user|execution|hypothesis|docs|training|tool
    confidence: float = 0.5
    salience: float = 0.5
    domain: str = "general"
    timestamp: float = 0.0
    id: str = ""
    consolidated: bool = False
    archived: bool = False
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = new_id(self.situation, self.objective)
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.last_accessed:
            self.last_accessed = self.timestamp

    @property
    def verified(self) -> bool:
        return self.status == SUCCESS

    def searchable_text(self) -> str:
        return " ".join([
            self.situation, self.objective, self.result,
            " ".join(self.actions), " ".join(self.lessons),
            " ".join(self.mistakes), self.domain,
        ])


class EpisodicMemory:
    def __init__(self, store: Store, config: MemoryConfig):
        self.store = store
        self.config = config

    # -------------------------------------------------------------- record
    def record(self, episode: Episode) -> Episode:
        ep = episode
        embedding = textsim.embed(ep.searchable_text())
        self.store.execute(
            """INSERT INTO episodes
               (id, situation, objective, actions, result, status, mistakes,
                lessons, timestamp, source, confidence, salience, domain,
                created, last_accessed, access_count, consolidated, archived,
                embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ep.id, ep.situation, ep.objective, json.dumps(ep.actions),
             ep.result, ep.status, json.dumps(ep.mistakes), json.dumps(ep.lessons),
             ep.timestamp, ep.source, ep.confidence, ep.salience, ep.domain,
             ep.timestamp, ep.last_accessed, ep.access_count,
             int(ep.consolidated), int(ep.archived), json.dumps(embedding)),
        )
        self.store.conn.commit()
        return ep

    # -------------------------------------------------------------- recall
    def recall(self, query: str, k: Optional[int] = None,
               include_archived: bool = False) -> list[Episode]:
        k = k or self.config.episodic_recall_k
        qvec = textsim.embed(query)
        if include_archived:
            rows = self.store.execute(
                "SELECT * FROM episodes WHERE archived IN (?, 0)", (1,)).fetchall()
        else:
            rows = self.store.execute(
                "SELECT * FROM episodes WHERE archived=0").fetchall()
        now = time.time()
        scored = []
        for r in rows:
            ep = self._row_to_episode(r)
            emb = json.loads(r["embedding"]) if r["embedding"] else None
            rel = textsim.hybrid_similarity(query, ep.searchable_text(), qvec, emb)
            recency = 1.0 / (1.0 + (now - ep.timestamp) / 3600.0)
            score = (0.55 * rel + 0.15 * recency + 0.20 * ep.salience
                     + 0.10 * (1.0 if ep.verified else 0.3 if ep.status == FAILURE else 0.0))
            scored.append((score, r, ep, rel))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, r, ep, rel in scored[:k]:
            if rel <= 0.02 and len(rows) > 3:
                continue                     # not truly relevant
            self._touch(ep.id)
            out.append(ep)
        return out

    def _touch(self, ep_id: str) -> None:
        self.store.execute(
            "UPDATE episodes SET access_count=access_count+1, last_accessed=? WHERE id=?",
            (time.time(), ep_id),
        )
        self.store.conn.commit()

    def get(self, ep_id: str) -> Optional[Episode]:
        r = self.store.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
        return self._row_to_episode(r) if r else None

    def recent(self, n: int = 10, unconsolidated_only: bool = False) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE archived=0"
        if unconsolidated_only:
            sql += " AND consolidated=0"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        return [self._row_to_episode(r) for r in self.store.execute(sql, (n,)).fetchall()]

    def all_active(self) -> list[Episode]:
        rows = self.store.execute("SELECT * FROM episodes WHERE archived=0").fetchall()
        return [self._row_to_episode(r) for r in rows]

    # ------------------------------------------------------------ lifecycle
    def mark_consolidated(self, ep_ids: list[str]) -> None:
        with self.store.transaction() as cur:
            for ep_id in ep_ids:
                cur.execute("UPDATE episodes SET consolidated=1 WHERE id=?", (ep_id,))

    def archive(self, ep_id: str, reason: str, compressed: str) -> None:
        r = self.store.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
        if not r:
            return
        payload = dict(r)
        with self.store.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO archive(id, kind, payload, reason, archived_at) "
                "VALUES (?,?,?,?,?)",
                (ep_id, "episode", json.dumps(payload, default=str), reason, time.time()),
            )
            cur.execute("UPDATE episodes SET archived=1, compressed=? WHERE id=?",
                        (compressed, ep_id))

    def recover_archive(self, ep_id: str) -> bool:
        r = self.store.execute(
            "SELECT * FROM archive WHERE id=? AND kind='episode'", (ep_id,)).fetchone()
        if not r:
            return False
        with self.store.transaction() as cur:
            cur.execute("UPDATE episodes SET archived=0, compressed=NULL WHERE id=?", (ep_id,))
            cur.execute("DELETE FROM archive WHERE id=?", (ep_id,))
        return True

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        row = self.store.execute(
            """SELECT COUNT(*) c,
                      SUM(status='success') s,
                      SUM(status='failure') f,
                      SUM(status='unverified') u,
                      SUM(consolidated) cons,
                      SUM(archived) arch
               FROM episodes""").fetchone()
        return {
            "count": row["c"] or 0,
            "success": row["s"] or 0,
            "failure": row["f"] or 0,
            "unverified": row["u"] or 0,
            "consolidated": row["cons"] or 0,
            "archived": row["arch"] or 0,
        }

    def _row_to_episode(self, r) -> Episode:
        return Episode(
            id=r["id"], situation=r["situation"], objective=r["objective"],
            actions=json.loads(r["actions"]), result=r["result"], status=r["status"],
            mistakes=json.loads(r["mistakes"]), lessons=json.loads(r["lessons"]),
            timestamp=r["timestamp"], source=r["source"], confidence=r["confidence"],
            salience=r["salience"], domain=r["domain"] or "general",
            consolidated=bool(r["consolidated"]), archived=bool(r["archived"]),
            access_count=r["access_count"], last_accessed=r["last_accessed"],
        )
