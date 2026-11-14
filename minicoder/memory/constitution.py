"""Constitution — the brain's standing rules.

Rules are semantic-memory concepts in the protected domain ``constitution``:

* hard-coded as facts with maximum importance and provenance ``user-rule``;
* pinned into working memory on every brain wake (always in the model's live
  context, hence "the brain always remembers its rules");
* immune to the sleep engine's forgetting stage (protected domain);
* contradictions with rules are recorded, never overwrite them.
"""
from __future__ import annotations

import re
import time

from .semantic import Concept, FACT
from .system import MemorySystem

DOMAIN = "constitution"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())[:40].strip("-")
    return slug or f"rule-{int(time.time())}"


def add_rule(memory: MemorySystem, text: str, source: str = "user") -> Concept:
    """Add (or reinforce) a standing rule."""
    name = f"rule: {_slug(text)}"
    c = Concept(
        name=name, definition=text.strip(), domain=DOMAIN,
        confidence=1.0, status=FACT, provenance=[f"{source}-rule"],
        importance=1.0)
    c = memory.semantic.upsert(c, provenance_note=f"{source}-rule")
    pin_rules(memory)
    return c


def list_rules(memory: MemorySystem) -> list[Concept]:
    return memory.semantic.by_domain(DOMAIN)


def pin_rules(memory: MemorySystem) -> int:
    """Ensure every rule sits pinned in working memory (always in context)."""
    pinned = 0
    for rule in list_rules(memory):
        memory.working.add("constraint", f"RULE: {rule.definition}",
                           importance=1.0, pinned=True,
                           tags=["rule", DOMAIN])
        pinned += 1
    return pinned


def rules_text(memory: MemorySystem, limit: int = 20) -> str:
    rules = list_rules(memory)[:limit]
    return "\n".join(f"- {r.definition}" for r in rules)
