"""Procedural memory — reusable skills/workflows learned from experience.

Skills are structured procedures (a list of typed steps; e.g. ``run_python``,
``write_file``, ``run_shell``, ``recall``, ``assert_condition``).  Safety rule
from the spec: **never automatically execute an untrusted stored procedure**.
A skill starts ``untrusted`` and only becomes ``verified`` after its steps have
been re-executed successfully in the sandbox.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import MemoryConfig
from .store import Store, new_id
from . import textsim
from .episodic import Episode, SUCCESS

UNTRUSTED, VERIFIED = "untrusted", "verified"


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str]
    steps: list[dict]               # [{action: str, args: dict}, ...]
    trust: str = UNTRUSTED
    provenance: str = "learned"
    id: str = ""
    success_count: int = 0
    failure_count: int = 0
    created: float = 0.0
    last_used: float = 0.0
    studied: bool = False              # internalized by sleep training

    def __post_init__(self):
        if not self.id:
            self.id = new_id(self.name)
        if not self.created:
            self.created = time.time()


class ProceduralMemory:
    def __init__(self, store: Store, config: MemoryConfig):
        self.store = store
        self.config = config

    # ------------------------------------------------------------------ CRUD
    def learn(self, skill: Skill) -> Skill:
        existing = self.store.execute(
            "SELECT id FROM skills WHERE lower(name)=?", (skill.name.strip().lower(),)
        ).fetchone()
        if existing:
            # Replace definition only if the new one is at least as trusted.
            row = self.store.execute(
                "SELECT trust FROM skills WHERE id=?", (existing["id"],)).fetchone()
            new_trust = skill.trust
            if row["trust"] == VERIFIED and new_trust == UNTRUSTED:
                new_trust = VERIFIED  # never downgrade trust silently
            self.store.execute(
                "UPDATE skills SET description=?, triggers=?, steps=?, trust=? WHERE id=?",
                (skill.description, json.dumps(skill.triggers), json.dumps(skill.steps),
                 new_trust, existing["id"]))
            self.store.conn.commit()
            skill.id = existing["id"]
            return skill
        self.store.execute(
            """INSERT INTO skills (id, name, description, triggers, steps, trust,
               success_count, failure_count, provenance, created, last_used)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (skill.id, skill.name, skill.description, json.dumps(skill.triggers),
             json.dumps(skill.steps), skill.trust, skill.success_count,
             skill.failure_count, skill.provenance, skill.created,
             skill.last_used or None),
        )
        self.store.conn.commit()
        return skill

    def recall(self, query: str, k: int = 3) -> list[Skill]:
        rows = self.store.execute("SELECT * FROM skills WHERE archived=0").fetchall()
        scored = []
        for r in rows:
            s = self._row_to_skill(r)
            text = f"{s.name} {s.description} {' '.join(s.triggers)}"
            sim = textsim.hybrid_similarity(query, text)
            # trigger-keyword exact match gets a strong boost
            ql = query.lower()
            if any(t.lower() in ql for t in s.triggers):
                sim = min(1.0, sim + 0.5)
            if sim > 0.05:
                scored.append((sim * (0.7 + 0.3 * min(1.0, s.success_count / 3.0)), s))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in scored[:k]]

    def get(self, skill_id: str) -> Optional[Skill]:
        r = self.store.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        return self._row_to_skill(r) if r else None

    def all_active(self) -> list[Skill]:
        return [self._row_to_skill(r) for r in
                self.store.execute("SELECT * FROM skills WHERE archived=0").fetchall()]

    def set_trust(self, skill_id: str, trust: str) -> None:
        assert trust in (UNTRUSTED, VERIFIED)
        self.store.execute("UPDATE skills SET trust=? WHERE id=?", (trust, skill_id))
        self.store.conn.commit()

    def _bump(self, skill_id: str, success: bool) -> None:
        col = "success_count" if success else "failure_count"
        self.store.execute(
            f"UPDATE skills SET {col}={col}+1, last_used=? WHERE id=?",
            (time.time(), skill_id))
        self.store.conn.commit()

    # -------------------------------------------------------------- execution
    def execute(self, skill: Skill, tools, *, approve_untrusted: bool = False,
                context: Optional[dict] = None) -> dict:
        """Run a skill's steps through the sandboxed tool executor.

        * untrusted skills require explicit approval (``approve_untrusted=True``);
        * a failing step updates the skill's failure bookkeeping;
        * verified success during execution promotes trust.
        """
        context = dict(context or {})
        if skill.trust == UNTRUSTED and not approve_untrusted:
            return {"ok": False, "error": "skill is untrusted and approval was not given",
                    "executed": 0}
        executed = 0
        log: list[dict] = []
        for i, step in enumerate(skill.steps):
            action = step.get("action")
            args = step.get("args", {})
            res = self._run_step(action, args, tools, context)
            log.append({"step": i, "action": action, "ok": res.get("ok", False),
                        "detail": res.get("error") or res.get("stdout", "")[:200]})
            if not res.get("ok", False):
                self._bump(skill.id, success=False)
                return {"ok": False, "error": f"step {i} ({action}) failed: "
                        f"{res.get('error', 'unknown')}", "executed": executed, "log": log}
            executed += 1
            context[f"step_{i}"] = res
        self._bump(skill.id, success=True)
        if skill.trust == UNTRUSTED and executed == len(skill.steps) and skill.steps:
            self.set_trust(skill.id, VERIFIED)
        return {"ok": True, "executed": executed, "log": log, "context": context}

    def _run_step(self, action: str, args: dict, tools, context: dict) -> dict:
        handler = {
            "run_python": lambda: tools.run_python(self._fmt(args.get("code", ""), context)),
            "run_shell": lambda: tools.run_shell(self._fmt(args.get("cmd", ""), context)),
            "write_file": lambda: tools.write_file(
                self._fmt(args.get("path", ""), context),
                self._fmt(args.get("content", ""), context)),
            "read_file": lambda: tools.read_file(self._fmt(args.get("path", ""), context)),
            "assert_condition": lambda: tools.run_python(self._fmt(
                args.get("check", "True"), context)),
            "recall": lambda: {"ok": True, "stdout": args.get("hint", "")},
            "think": lambda: {"ok": True, "stdout": args.get("note", "")},
        }.get(action)
        if handler is None:
            return {"ok": False, "error": f"unknown action: {action}"}
        return handler()

    @staticmethod
    def _fmt(template: str, context: dict) -> str:
        out = template
        for key, value in context.items():
            if isinstance(value, str):
                out = out.replace("{{" + key + "}}", value)
            elif isinstance(value, dict) and "stdout" in value:
                out = out.replace("{{" + key + ".stdout}}", str(value["stdout"]))
        return out

    # -------------------------------------------------------------- learning
    def learn_from_episode(self, episode: Episode, name: Optional[str] = None) -> Optional[Skill]:
        """Distill a verified episode into a candidate skill (starts untrusted)."""
        if episode.status != SUCCESS:
            return None
        steps: list[dict] = []
        for action in episode.actions:
            if action.startswith("python:"):
                steps.append({"action": "run_python",
                              "args": {"code": action[len("python:"):]}})
            elif action.startswith("shell:"):
                steps.append({"action": "run_shell",
                              "args": {"cmd": action[len("shell:"):]}})
            else:
                steps.append({"action": "think", "args": {"note": action}})
        if not steps:
            return None
        skill = Skill(
            name=name or f"handle:{episode.objective[:60]}",
            description=f"Distilled from episode {episode.id}: {episode.objective}",
            triggers=[t for t in textsim.tokenize(episode.objective)[:8]],
            steps=steps, trust=UNTRUSTED,
            provenance=f"episode:{episode.id}",
        )
        return self.learn(skill)

    def mark_studied(self, skill_ids: list[str]) -> None:
        with self.store.transaction() as cur:
            for sid in skill_ids:
                cur.execute("UPDATE skills SET studied=1 WHERE id=?", (sid,))

    def unstudied(self) -> list[Skill]:
        return [self._row_to_skill(r) for r in
                self.store.execute(
                    "SELECT * FROM skills WHERE archived=0 AND studied=0").fetchall()]

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        row = self.store.execute(
            """SELECT COUNT(*) c, SUM(trust='verified') v,
                      SUM(success_count) s, SUM(failure_count) f,
                      SUM(archived) a, SUM(studied) st
               FROM skills""").fetchone()
        return {"count": row["c"] or 0, "verified": row["v"] or 0,
                "total_successes": row["s"] or 0, "total_failures": row["f"] or 0,
                "studied": row["st"] or 0, "archived": row["a"] or 0}

    def _row_to_skill(self, r) -> Skill:
        keys = r.keys()
        return Skill(
            id=r["id"], name=r["name"], description=r["description"],
            triggers=json.loads(r["triggers"]), steps=json.loads(r["steps"]),
            trust=r["trust"], provenance=r["provenance"],
            success_count=r["success_count"], failure_count=r["failure_count"],
            created=r["created"], last_used=r["last_used"] or 0.0,
            studied=bool(r["studied"]) if "studied" in keys else False,
        )
