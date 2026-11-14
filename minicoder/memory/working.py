"""Working memory — the brain's limited-capacity active workspace.

* Capacity-limited with priority-based retention.
* Item priority = recency, relevance (to the current focus) and importance.
* Automatic expiration (TTL), context compression, and *pinning* so critical
  task information cannot be silently evicted.
* State is snapshot-persisted (atomic writes) so a restart restores the
  workspace.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from ..config import MemoryConfig
from .store import atomic_write_json, new_id
from . import textsim


@dataclass
class MemoryItem:
    kind: str                      # task | code | reasoning | error | constraint | note | decision
    content: str
    importance: float = 0.5
    ttl: float = 3600.0
    pinned: bool = False
    tags: list[str] = field(default_factory=list)
    id: str = ""
    created: float = 0.0
    ttl_kind: str = "seconds"

    def __post_init__(self):
        if not self.id:
            self.id = new_id(self.kind, self.content)
        if not self.created:
            self.created = time.time()

    def expired(self, now: Optional[float] = None) -> bool:
        if self.pinned:
            return False
        now = now or time.time()
        return (now - self.created) > self.ttl


class WorkingMemory:
    def __init__(self, config: MemoryConfig, snapshot_path: Optional[str] = None):
        self.config = config
        self.snapshot_path = snapshot_path
        self.items: list[MemoryItem] = []
        self.compression_count = 0
        if snapshot_path:
            self._restore()

    # ------------------------------------------------------------- mutation
    def add(self, kind: str, content: str, *, importance: float = 0.5,
            ttl: Optional[float] = None, pinned: bool = False,
            tags: Optional[list[str]] = None) -> MemoryItem:
        item = MemoryItem(
            kind=kind, content=content, importance=float(max(0.0, min(1.0, importance))),
            ttl=ttl if ttl is not None else self.config.working_ttl_seconds,
            pinned=pinned, tags=tags or [],
        )
        # de-duplicate identical content
        for existing in self.items:
            if existing.content == content and existing.kind == kind:
                existing.created = time.time()          # refresh = recency boost
                existing.importance = max(existing.importance, item.importance)
                existing.pinned = existing.pinned or pinned
                self._persist()
                return existing
        self.items.append(item)
        self._enforce_capacity()
        self._persist()
        return item

    def pin(self, item_id: str, pinned: bool = True) -> bool:
        for it in self.items:
            if it.id == item_id:
                it.pinned = pinned
                self._persist()
                return True
        return False

    def remove(self, item_id: str) -> bool:
        before = len(self.items)
        self.items = [i for i in self.items if i.id != item_id]
        changed = len(self.items) != before
        if changed:
            self._persist()
        return changed

    def clear(self, *, keep_pinned: bool = True) -> None:
        self.items = [i for i in self.items if keep_pinned and i.pinned]
        self._persist()

    # ------------------------------------------------------------- priority
    def _recency(self, item: MemoryItem, now: float) -> float:
        age = now - item.created
        return 1.0 / (1.0 + age / 300.0)          # half-life ~5 minutes

    def _relevance(self, item: MemoryItem, focus: str, focus_tags: set[str]) -> float:
        score = 0.0
        if focus:
            score = max(score, textsim.hybrid_similarity(focus, item.content))
        if focus_tags and item.tags:
            overlap = len(focus_tags & set(t.lower() for t in item.tags))
            score = max(score, overlap / max(1, len(focus_tags)))
        return min(1.0, score)

    def score(self, item: MemoryItem, now: Optional[float] = None,
              focus: str = "", focus_tags: Optional[set[str]] = None) -> float:
        if item.pinned:
            return float("inf")
        now = now or time.time()
        cfg = self.config
        return (
            cfg.recency_weight * self._recency(item, now)
            + cfg.relevance_weight * self._relevance(item, focus, focus_tags or set())
            + cfg.importance_weight * item.importance
        )

    # ------------------------------------------------------------- capacity
    def _expire(self) -> None:
        now = time.time()
        self.items = [i for i in self.items if not i.expired(now)]

    def _enforce_capacity(self) -> None:
        """Evict lowest-priority items when over capacity (never evict pinned)."""
        self._expire()
        while len(self.items) > self.config.working_capacity:
            candidates = [i for i in self.items if not i.pinned]
            if not candidates:
                break                      # everything pinned: refuse to drop
            now = time.time()
            victim = min(candidates, key=lambda i: self.score(i, now))
            self.items.remove(victim)

    def compress(self, max_chars: int = 400,
                 summarizer: Optional[Callable[[list[str]], str]] = None) -> Optional[MemoryItem]:
        """Merge the lowest-priority unpinned items into one summary item.

        A model-provided ``summarizer`` can be plugged in; without one we use a
        deterministic extractive compression (first line of each item).
        """
        self._expire()
        unpinned = [i for i in self.items if not i.pinned]
        if len(unpinned) < 3:
            return None
        now = time.time()
        ordered = sorted(unpinned, key=lambda i: self.score(i, now))
        merge_set = ordered[: max(2, len(ordered) // 2)]
        texts = [i.content for i in merge_set]
        if summarizer:
            summary = summarizer(texts)[:max_chars]
        else:
            parts = []
            for t in texts:
                first = t.strip().splitlines()[0][:80]
                parts.append(f"- {first}")
            summary = ("[compressed %d items]\n" % len(parts)) + "\n".join(parts)
            summary = summary[:max_chars]
        tags = sorted({t for i in merge_set for t in i.tags})[:8]
        for it in merge_set:
            self.items.remove(it)
        merged = self.add("note", summary, importance=0.45, tags=tags)
        self.compression_count += 1
        self._persist()
        return merged

    # --------------------------------------------------------------- access
    def context(self, focus: str = "", max_chars: int = 2000,
                focus_tags: Optional[set[str]] = None) -> str:
        """Highest-priority items rendered for the model's context window."""
        self._expire()
        now = time.time()
        ranked = sorted(self.items, key=lambda i: self.score(i, now, focus, focus_tags),
                        reverse=True)
        out, used = [], 0
        for it in ranked:
            line = f"[{it.kind}] {it.content}"
            if used + len(line) > max_chars:
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out)

    def stats(self) -> dict:
        self._expire()
        pinned = sum(1 for i in self.items if i.pinned)
        return {
            "items": len(self.items),
            "capacity": self.config.working_capacity,
            "usage": len(self.items) / max(1, self.config.working_capacity),
            "pinned": pinned,
            "compressions": self.compression_count,
        }

    # ---------------------------------------------------------- persistence
    def _persist(self) -> None:
        if not self.snapshot_path:
            return
        atomic_write_json(self.snapshot_path, {
            "items": [asdict(i) for i in self.items],
            "compression_count": self.compression_count,
        })

    def _restore(self) -> None:
        import json, os
        if not os.path.exists(self.snapshot_path):
            return
        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.items = [MemoryItem(**kw) for kw in data.get("items", [])]
            self.compression_count = data.get("compression_count", 0)
            self._expire()
        except (OSError, ValueError, TypeError):
            self.items = []                 # corrupt snapshot counts as empty
