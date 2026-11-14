"""Semantic memory — permanent, structured, connected knowledge.

Concepts are stored with provenance, confidence and verification status, and
are linked into a knowledge graph.  Retrieval uses hybrid text similarity plus
**spreading activation** over graph edges, so the system retrieves *related*
knowledge, not just keyword matches.

Facts and hypotheses are strictly separated (``status``), contradictions are
recorded (never silently overwritten), and outdated knowledge is marked, not
deleted.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import MemoryConfig
from .store import Store, new_id
from . import textsim

FACT, HYPOTHESIS, OUTDATED = "fact", "hypothesis", "outdated"


@dataclass
class Concept:
    name: str
    definition: str
    domain: str = "general"
    confidence: float = 0.5
    status: str = HYPOTHESIS       # fact | hypothesis | outdated
    provenance: list[str] = field(default_factory=lambda: ["hypothesis"])
    support: int = 1
    importance: float = 0.5
    id: str = ""
    created: float = 0.0
    updated: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    outdated: bool = False
    archived: bool = False
    score: float = 0.0             # transient retrieval score
    via: list[str] = field(default_factory=list)  # graph path that found it

    def __post_init__(self):
        if not self.id:
            self.id = new_id(self.name)
        now = time.time()
        self.created = self.created or now
        self.updated = self.updated or now
        self.last_accessed = self.last_accessed or now

    def searchable_text(self) -> str:
        return f"{self.name} {self.domain} {self.definition}"


class SemanticMemory:
    def __init__(self, store: Store, config: MemoryConfig):
        self.store = store
        self.config = config

    # -------------------------------------------------------------- writes
    def upsert(self, concept_static: Concept,
               provenance_note: Optional[str] = None) -> Concept:
        """Insert or update a concept.  Updates accumulate support and keep the
        highest-confidence version — knowledge is never silently overwritten:
        conflicting definitions for the same name produce a contradiction row.
        """
        name_key = concept_static.name.strip().lower()
        existing = self.store.execute(
            "SELECT * FROM concepts WHERE lower(name)=?", (name_key,)).fetchone()
        embedding = json.dumps(textsim.embed(concept_static.searchable_text()))
        now = time.time()
        prov = list(concept_static.provenance)
        if provenance_note and provenance_note not in prov:
            prov.append(provenance_note)
        if existing is None:
            c = concept_static
            c.provenance = prov
            self.store.execute(
                """INSERT INTO concepts
                   (id, name, definition, domain, confidence, status, provenance,
                    support, importance, created, updated, last_accessed,
                    access_count, outdated, archived, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (c.id, c.name, c.definition, c.domain, c.confidence, c.status,
                 json.dumps(c.provenance), c.support, c.importance, c.created,
                 c.updated, c.last_accessed, c.access_count, int(c.outdated),
                 int(c.archived), embedding),
            )
            self.store.conn.commit()
            return c
        # update path
        old_def = existing["definition"]
        new_def = concept_static.definition
        prov_old = json.loads(existing["provenance"])
        for p in prov:
            if p not in prov_old:
                prov_old.append(p)
        support = existing["support"] + 1
        confidence = max(existing["confidence"], concept_static.confidence)
        status = existing["status"]
        if existing["status"] != FACT and concept_static.status == FACT:
            status = FACT            # hypothesis promoted by verification
        # A conflicting definition is *recorded*, not overwritten.
        if new_def.strip() and new_def.strip() != old_def.strip():
            sim = textsim.jaccard(old_def, new_def)
            if sim < 0.4:
                self._record_contradiction(existing["id"], concept_static, new_def)
            if confidence >= existing["confidence"] and sim >= 0.3:
                old_def = new_def    # refine, keep history in contradictions
        self.store.execute(
            """UPDATE concepts SET definition=?, domain=?, confidence=?, status=?,
               provenance=?, support=?, importance=?, updated=?, embedding=?
               WHERE id=?""",
            (old_def, concept_static.domain or existing["domain"], confidence,
             status, json.dumps(prov_old), support,
             max(existing["importance"], concept_static.importance), now,
             embedding, existing["id"]),
        )
        self.store.conn.commit()
        return self.get_by_id(existing["id"])  # type: ignore[return-value]

    def mark_outdated(self, concept_id: str) -> None:
        self.store.execute(
            "UPDATE concepts SET outdated=1, status=?, updated=? WHERE id=?",
            (OUTDATED, time.time(), concept_id))
        self.store.conn.commit()

    def _record_contradiction(self, existing_id: str, new_concept: Concept,
                              new_def: str) -> None:
        self.store.execute(
            "INSERT OR IGNORE INTO contradictions(id, a, b, note, created) "
            "VALUES (?,?,?,?,?)",
            (new_id(existing_id, new_def)[:16], existing_id, existing_id,
             f"conflicting definitions for '{new_concept.name}': "
             f"new='{new_def[:120]}'", time.time()),
        )
        self.store.conn.commit()

    # ---------------------------------------------------------------- graph
    def link(self, a_id: str, b_id: str, relation: str = "related_to",
             weight: float = 1.0) -> None:
        if a_id == b_id:
            return
        self.store.execute(
            "INSERT INTO edges(src, dst, relation, weight) VALUES (?,?,?,?) "
            "ON CONFLICT(src, dst, relation) DO UPDATE SET weight=MAX(weight, excluded.weight)",
            (a_id, b_id, relation, float(weight)))
        self.store.conn.commit()

    def neighbors(self, concept_id: str) -> list[tuple[str, str, float]]:
        rows = self.store.execute(
            "SELECT src, dst, relation, weight FROM edges WHERE src=? OR dst=?",
            (concept_id, concept_id)).fetchall()
        out = []
        for r in rows:
            other = r["dst"] if r["src"] == concept_id else r["src"]
            out.append((other, r["relation"], r["weight"]))
        return out

    # --------------------------------------------------------------- query
    def query(self, text: str, k: Optional[int] = None,
              include_hypotheses: bool = True) -> list[Concept]:
        """Hybrid similarity + one hop of spreading activation along edges."""
        k = k or self.config.semantic_recall_k
        qvec = textsim.embed(text)
        rows = self.store.execute(
            "SELECT * FROM concepts WHERE archived=0").fetchall()
        if not rows:
            return []
        seeds: dict[str, tuple[float, object]] = {}
        for r in rows:
            emb = json.loads(r["embedding"]) if r["embedding"] else None
            sim = textsim.hybrid_similarity(text, f"{r['name']} {r['definition']}",
                                            qvec, emb)
            if sim > 0.05:
                seeds[r["id"]] = (sim, r)
        # spreading activation: 1 hop, attenuated by edge weight
        activated: dict[str, tuple[float, str]] = {}
        for seed_id, (sim, _r) in seeds.items():
            for other, rel, w in self.neighbors(seed_id):
                if other in seeds:
                    continue
                prev = activated.get(other, (0.0, rel))[0]
                activated[other] = (max(prev, sim * 0.5 * w), rel)
        results: list[Concept] = []
        for cid, (sim, r) in seeds.items():
            c = self._row_to_concept(r)
            if c.status == HYPOTHESIS and not include_hypotheses:
                continue
            c.score = min(1.0, sim * (0.8 + 0.2 * c.confidence)
                          * (0.7 if c.outdated else 1.0))
            c.via = ["direct"]
            results.append(c)
        for cid, (sim, rel) in activated.items():
            r = self.store.execute("SELECT * FROM concepts WHERE id=?", (cid,)).fetchone()
            if not r:
                continue
            c = self._row_to_concept(r)
            if c.status == HYPOTHESIS and not include_hypotheses:
                continue
            c.score = min(1.0, sim * (0.8 + 0.2 * c.confidence))
            c.via = [rel]
            results.append(c)
        results.sort(key=lambda c: c.score, reverse=True)
        out = results[:k]
        for c in out:
            self.store.execute(
                "UPDATE concepts SET access_count=access_count+1, last_accessed=? WHERE id=?",
                (time.time(), c.id))
        if out:
            self.store.conn.commit()
        return out

    def get_by_id(self, concept_id: str) -> Optional[Concept]:
        r = self.store.execute("SELECT * FROM concepts WHERE id=?", (concept_id,)).fetchone()
        return self._row_to_concept(r) if r else None

    def get_by_name(self, name: str) -> Optional[Concept]:
        r = self.store.execute(
            "SELECT * FROM concepts WHERE lower(name)=?", (name.strip().lower(),)).fetchone()
        return self._row_to_concept(r) if r else None

    def all_active(self) -> list[Concept]:
        rows = self.store.execute("SELECT * FROM concepts WHERE archived=0").fetchall()
        return [self._row_to_concept(r) for r in rows]

    def contradictions(self, unresolved_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM contradictions"
        if unresolved_only:
            sql += " WHERE resolved=0"
        return [dict(r) for r in self.store.execute(sql).fetchall()]

    # ---------------------------------------------------- organization utils
    def find_near_duplicates(self, threshold: float = 0.85) -> list[tuple[str, str, float]]:
        """Pairs of active concepts whose name+definition nearly match."""
        rows = self.store.execute(
            "SELECT id, name, definition FROM concepts WHERE archived=0").fetchall()
        pairs = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                sim = textsim.jaccard(a["name"] + " " + a["definition"],
                                      b["name"] + " " + b["definition"])
                if sim >= threshold:
                    pairs.append((a["id"], b["id"], sim))
        return pairs

    def merge(self, keep_id: str, drop_id: str) -> bool:
        """Merge ``drop_id`` into ``keep_id``: support adds up, edges move,
        the merged record keeps full provenance of both."""
        with self.store.transaction() as cur:
            a = cur.execute("SELECT * FROM concepts WHERE id=?", (keep_id,)).fetchone()
            b = cur.execute("SELECT * FROM concepts WHERE id=?", (drop_id,)).fetchone()
            if not a or not b:
                return False
            prov = json.loads(a["provenance"]) + [
                f"merged:{b['name']}", *json.loads(b["provenance"])]
            prov = list(dict.fromkeys(prov))
            keep_conf = max(a["confidence"], b["confidence"])
            keep_status = a["status"]
            for s in (a["status"], b["status"]):
                if s == FACT:
                    keep_status = FACT
            cur.execute(
                """UPDATE concepts SET support=?, confidence=?, status=?,
                   provenance=?, importance=?, access_count=access_count + ?, updated=?
                   WHERE id=?""",
                (a["support"] + b["support"], keep_conf, keep_status, json.dumps(prov),
                 max(a["importance"], b["importance"]), b["access_count"],
                 time.time(), keep_id),
            )
            for r in cur.execute(
                    "SELECT src, dst, relation, weight FROM edges WHERE src=? OR dst=?",
                    (drop_id, drop_id)).fetchall():
                other = r["dst"] if r["src"] == drop_id else r["src"]
                if other == keep_id:
                    continue
                cur.execute(
                    "INSERT INTO edges(src, dst, relation, weight) VALUES (?,?,?,?) "
                    "ON CONFLICT(src, dst, relation) DO UPDATE SET weight=MAX(weight, excluded.weight)",
                    (keep_id, other, r["relation"], r["weight"]))
            cur.execute("DELETE FROM edges WHERE src=? OR dst=?", (drop_id, drop_id))
            cur.execute(
                "INSERT OR REPLACE INTO archive(id, kind, payload, reason, archived_at) "
                "VALUES (?,?,?,?,?)",
                (drop_id, "concept", json.dumps(dict(b), default=str),
                 f"merged into {keep_id}", time.time()))
            cur.execute("DELETE FROM concepts WHERE id=?", (drop_id,))
        return True

    def archive(self, concept_id: str, reason: str) -> None:
        r = self.store.execute(
            "SELECT * FROM concepts WHERE id=?", (concept_id,)).fetchone()
        if not r:
            return
        with self.store.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO archive(id, kind, payload, reason, archived_at) "
                "VALUES (?,?,?,?,?)",
                (concept_id, "concept", json.dumps(dict(r), default=str),
                 reason, time.time()))
            cur.execute("UPDATE concepts SET archived=1 WHERE id=?", (concept_id,))

    def recover_archive(self, concept_id: str) -> bool:
        r = self.store.execute(
            "SELECT * FROM archive WHERE id=? AND kind='concept'", (concept_id,)).fetchone()
        if not r:
            return False
        with self.store.transaction() as cur:
            cur.execute("UPDATE concepts SET archived=0 WHERE id=?", (concept_id,))
            cur.execute("DELETE FROM archive WHERE id=?", (concept_id,))
        return True

    def stats(self) -> dict:
        row = self.store.execute(
            """SELECT COUNT(*) c,
                      SUM(status='fact') f,
                      SUM(status='hypothesis') h,
                      SUM(outdated) o,
                      SUM(archived) a
               FROM concepts""").fetchone()
        edges = self.store.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
        return {
            "count": row["c"] or 0, "facts": row["f"] or 0,
            "hypotheses": row["h"] or 0, "outdated": row["o"] or 0,
            "archived": row["a"] or 0, "edges": edges or 0,
        }

    def graph_snapshot(self) -> dict:
        """Export the knowledge graph (used by dashboard + archival)."""
        nodes = []
        for r in self.store.execute(
                "SELECT * FROM concepts WHERE archived=0").fetchall():
            c = self._row_to_concept(r).__dict__
            c.pop("embedding", None)
            nodes.append(c)
        edges = [dict(r) for r in self.store.execute("SELECT * FROM edges").fetchall()]
        return {"nodes": nodes, "edges": edges}

    def _row_to_concept(self, r) -> Concept:
        return Concept(
            id=r["id"], name=r["name"], definition=r["definition"],
            domain=r["domain"] or "general", confidence=r["confidence"],
            status=r["status"], provenance=json.loads(r["provenance"]),
            support=r["support"], importance=r["importance"],
            created=r["created"], updated=r["updated"],
            last_accessed=r["last_accessed"], access_count=r["access_count"],
            outdated=bool(r["outdated"]), archived=bool(r["archived"]),
        )
