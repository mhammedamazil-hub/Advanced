"""Metacognition — measurable self-monitoring, not simulated inner monologues.

Every introspective question is answered by a numeric, testable function:

* "What am I trying to accomplish?" → tracked goal stack.
* "What do I already know?"       → retrieval-support score from memory.
* "What is missing?"              → explicit gap list, updated by tool errors.
* "How confident am I?"           → confidence estimate combining retrieval
                                    support, verification status, contradictions.
* "Which strategy?"               → deterministic policy over measured signals.
* "Did my solution work?"         → sandbox verification status, not vibes.
* "What did I learn?"             → salience gate deciding long-term storage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:      # pragma: no cover
    from .memory.system import MemorySystem


@dataclass
class TaskState:
    goal: str
    created: float = field(default_factory=time.time)
    attempts: int = 0
    gaps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    strategy: str = "recall-first"
    resolved: bool = False


@dataclass
class Reflection:
    worked: bool
    verified: bool
    lesson: str
    record_long_term: bool
    salience: float
    confidence_delta: float


class MetaCognition:
    def __init__(self, memory: "MemorySystem"):
        self.memory = memory
        self.current: Optional[TaskState] = None
        self.history: list[TaskState] = []

    # ---------------------------------------------------------- goal tracking
    def begin_task(self, goal: str) -> TaskState:
        self.current = TaskState(goal=goal)
        return self.current

    def end_task(self, resolved: bool) -> Optional[TaskState]:
        if self.current:
            self.current.resolved = resolved
            self.history.append(self.current)
        state, self.current = self.current, None
        return state

    # -------------------------------------------------------- knowledge probe
    def knowledge_support(self, query: str) -> dict:
        """How much does long-term memory already know about this query?

        Returns measurable support: best-match scores + a combined support
        index in [0,1] used by confidence estimation and strategy selection.
        """
        sem = self.memory.semantic.query(query, k=3)
        epi = self.memory.episodic.recall(query, k=3)
        skills = self.memory.procedural.recall(query, k=2)
        sem_score = max((c.score for c in sem), default=0.0)
        verified_epi = sum(1 for e in epi if e.verified)
        support = min(1.0, 0.5 * sem_score + 0.3 * min(1.0, verified_epi / 2.0)
                      + 0.2 * (1.0 if skills else 0.0))
        return {
            "semantic_hits": len(sem), "semantic_best": round(sem_score, 3),
            "episodic_hits": len(epi), "verified_episodes": verified_epi,
            "skill_hits": len(skills), "support": round(support, 3),
            "concepts": [c.name for c in sem],
            "skills": [s.name for s in skills],
        }

    def find_gaps(self, query: str, required: Optional[list[str]] = None) -> list[str]:
        """Explicitly identify missing information (e.g. tokens with zero
        concept support, plus caller-declared requirements)."""
        gaps: list[str] = []
        support = self.knowledge_support(query)
        if support["support"] < 0.25:
            gaps.append(f"low prior knowledge about: {query[:80]}")
        for req in required or []:
            hits = self.memory.semantic.query(req, k=1)
            if not hits or hits[0].score < 0.15:
                gaps.append(f"unknown requirement: {req}")
        if self.current is not None:
            for g in gaps:
                if g not in self.current.gaps:
                    self.current.gaps.append(g)
        return gaps

    # ---------------------------------------------------------- confidence
    def estimate_confidence(self, query: str, *, verified: bool = False,
                            had_tool_errors: bool = False) -> float:
        support = self.knowledge_support(query)["support"]
        contradictions = len(self.memory.semantic.contradictions())
        conf = 0.35 + 0.30 * support
        if verified:
            conf += 0.30
        if had_tool_errors:
            conf -= 0.15
        conf -= min(0.15, 0.03 * contradictions)
        conf = max(0.0, min(1.0, conf))
        if self.current is not None:
            self.current.confidence = conf
        return round(conf, 3)

    # ----------------------------------------------------------- strategies
    def select_strategy(self, query: str) -> dict:
        """Deterministic policy over measured signals.

        * high support + a matching verified skill → execute the skill;
        * medium support                          → memory-guided generation;
        * low support                             → decompose + explore.
        """
        ks = self.knowledge_support(query)
        skills = self.memory.procedural.recall(query, k=1)
        verified_skill = skills and skills[0].trust == "verified"
        if ks["support"] >= 0.6 and verified_skill:
            strategy = "execute-skill"
        elif ks["support"] >= 0.45:
            strategy = "memory-guided"
        else:
            strategy = "decompose-explore"
        if self.current is not None:
            self.current.strategy = strategy
        return {"strategy": strategy, **ks}

    # --------------------------------------------------------- self evaluation
    def self_evaluate(self, objective: str, *, worked: bool, verified: bool,
                      mistakes: Optional[list[str]] = None,
                      lesson: str = "") -> Reflection:
        """Decide — measurably — whether an experience deserves long-term memory."""
        mistakes = mistakes or []
        salience = 0.2
        if verified and worked:
            salience = 0.7
        elif worked and not verified:
            salience = 0.4              # plausible but unproven
        elif mistakes:
            salience = 0.8              # failures with error info are gold
        else:
            salience = 0.5
        record = salience >= 0.4        # the gate
        if not lesson:
            if verified and worked:
                lesson = f"Verified approach for '{objective[:60]}' — reuse it."
            elif mistakes:
                lesson = f"Avoid: {mistakes[0][:80]}"
            else:
                lesson = f"Attempted '{objective[:60]}' without verification."
        prev = self.current.confidence if self.current else 0.0
        delta = (0.1 if verified and worked else -0.05 if mistakes else 0.0)
        return Reflection(worked=worked, verified=verified, lesson=lesson,
                          record_long_term=record, salience=salience,
                          confidence_delta=round(delta, 3))

    # ------------------------------------------------------------- reporting
    def report(self) -> dict:
        return {
            "current_task": self.current.goal if self.current else None,
            "confidence": self.current.confidence if self.current else None,
            "gaps": list(self.current.gaps) if self.current else [],
            "completed_tasks": len(self.history),
            "resolved": sum(1 for t in self.history if t.resolved),
        }
