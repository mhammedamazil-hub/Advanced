"""The Sleep Engine — autonomous computational sleep with 7 stages:

    1. Memory replay        — surface informative recent experiences
    2. Consolidation        — turn experiences into durable concepts
    3. Pattern discovery    — generalize recurring situations (with causal care)
    4. Organization         — merge duplicates, link concepts, flag outdated
    5. Forgetting/compress  — importance-based retention, archive, never lose gold
    6. Learning/improvement — practice weak domains, verify, only keep verified gains
    7. Wake                 — report everything that changed

Mechanisms are *computational analogues* of biological sleep (replay,
consolidation, synaptic downscaling) — not a literal brain simulation.

Interruption safety: after every stage the engine writes a checkpoint row,
so a crashed/killed sleep cycle can be resumed exactly where it stopped.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from ..config import BrainConfig
from ..memory.system import MemorySystem
from ..memory.episodic import Episode, SUCCESS, FAILURE, UNVERIFIED
from ..memory.semantic import Concept, FACT, HYPOTHESIS
from ..memory.store import atomic_write_json, new_id
from ..memory import textsim
from ..tools.sandbox import Sandbox
from .. import curriculum

PracticeSolver = Callable[[curriculum.Task], str]

ERROR_SIG_RE = re.compile(
    r"([A-Za-z]+Error|[A-Za-z]+Exception|SyntaxError|timeout|assertion failed|failed)",
    re.IGNORECASE)


@dataclass
class SleepReport:
    cycle_id: str = ""
    started: float = 0.0
    ended: float = 0.0
    status: str = "running"          # running | interrupted | completed
    resumed_from: Optional[str] = None
    stage_timings: dict = field(default_factory=dict)
    replayed_episodes: list = field(default_factory=list)
    consolidated_episodes: list = field(default_factory=list)
    new_concepts: list = field(default_factory=list)
    merged_concepts: list = field(default_factory=list)
    pattern_concepts: list = field(default_factory=list)
    new_edges: int = 0
    marked_outdated: list = field(default_factory=list)
    archived: list = field(default_factory=list)
    practice: list = field(default_factory=list)
    skills_studied: list = field(default_factory=list)   # internalized by training
    weaknesses: dict = field(default_factory=dict)
    unresolved: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


SkillTrainer = Callable[[list[str], list[str]], dict]


class SleepEngine:
    def __init__(self, memory: MemorySystem, config: BrainConfig,
                 sandbox: Sandbox,
                 practice_solver: Optional[PracticeSolver] = None,
                 skill_trainer: Optional[SkillTrainer] = None):
        self.memory = memory
        self.config = config
        self.sandbox = sandbox
        self.practice_solver = practice_solver
        self.skill_trainer = skill_trainer    # (texts, skill_names) -> metrics
        self.report = SleepReport()

    # ------------------------------------------------------------- entrypoints
    def run(self, resume_cycle: Optional[str] = None) -> SleepReport:
        """Run a full sleep cycle (or resume an interrupted one)."""
        self.memory.backup()                         # safety: snapshot first
        if resume_cycle:
            self.report = self._load_report(resume_cycle)
            cycle_id = resume_cycle
            completed = self._completed_stages(cycle_id)
        else:
            self.report = SleepReport(
                cycle_id="sleep-" + uuid.uuid4().hex[:8], started=time.time())
            cycle_id = self.report.cycle_id
            completed = set()
            self.memory.store.execute(
                "INSERT OR REPLACE INTO sleep_cycles(id, started, status) VALUES (?,?,?)",
                (cycle_id, self.report.started, "running"))
            self.memory.store.conn.commit()
        stages = [
            (1, "replay", self._stage_replay),
            (2, "consolidation", self._stage_consolidation),
            (3, "pattern_discovery", self._stage_pattern_discovery),
            (4, "organization", self._stage_organization),
            (5, "forgetting", self._stage_forgetting),
            (6, "learning", self._stage_learning),
            (7, "wake", self._stage_wake),
        ]
        try:
            for num, name, fn in stages:
                if num in completed:
                    continue
                t0 = time.time()
                fn()
                self.report.stage_timings[name] = round(time.time() - t0, 3)
                self._checkpoint(cycle_id, num)
            if self.report.status == "running":
                self.report.status = "completed"
            self._finalize(cycle_id)
        except (KeyboardInterrupt, Exception) as exc:    # noqa: BLE001
            self.report.status = "interrupted"
            self.report.notes.append(f"interrupted at stage with error: {exc!r}")
            self._finalize(cycle_id)
            if isinstance(exc, KeyboardInterrupt):
                raise
        return self.report

    # ------------------------------------------------------------- stage 1-2
    def _stage_replay(self) -> None:
        """Score recent experiences: informative ones replay first."""
        episodes = self.memory.episodic.all_active()
        uncons = [e for e in episodes if not e.consolidated]
        now = time.time()

        def informativeness(e: Episode) -> float:
            score = 0.0
            if e.status == SUCCESS:
                score += 1.0
            elif e.status == FAILURE:
                score += 0.8                      # most informative: mistakes
            else:
                score += 0.2                      # unverified: least trustworthy
            score += min(0.3, 0.1 * len(e.mistakes))
            score += 0.4 * e.salience + 0.3 * e.confidence
            score += 0.5 / (1.0 + (now - e.timestamp) / 3600.0)
            return score

        ranked = sorted(uncons or episodes, key=informativeness, reverse=True)
        top = ranked[: self.config.sleep.replay_top_k]
        self.report.replayed_episodes = [
            {"id": e.id, "objective": e.objective, "status": e.status,
             "score": round(informativeness(e), 3)} for e in top]
        self._replay_set = top

    def _stage_consolidation(self) -> None:
        """Extract durable knowledge from replayed episodes into semantics."""
        replayed = getattr(self, "_replay_set", None)
        if replayed is None:                       # resumed: rebuild from report
            ids = [r["id"] for r in self.report.replayed_episodes]
            replayed = [e for e in (self.memory.episodic.get(i) for i in ids) if e]
        consolidated_ids: list[str] = []
        for ep in replayed:
            new = self._consolidate_episode(ep)
            if new:
                self.report.new_concepts.extend(new)
            # mark consolidated only when distillation actually extracted
            # knowledge OR the episode itself is verified; empty junk must stay
            # subject to the forgetting stage so memory cannot grow forever.
            if new or ep.status == SUCCESS:
                consolidated_ids.append(ep.id)
        if consolidated_ids:
            self.memory.episodic.mark_consolidated(consolidated_ids)
        self.report.consolidated_episodes = consolidated_ids

    def _consolidate_episode(self, ep: Episode) -> list[dict]:
        created = []
        source = f"episode:{ep.id}/{ep.source}"
        status = FACT if (ep.status == SUCCESS and ep.confidence >= 0.7) else HYPOTHESIS
        # 1. lessons -> concepts
        lessons = list(ep.lessons)
        for mistake in ep.mistakes:
            sig = self._error_signature(mistake)
            if sig:
                lessons.append(f"When '{sig}' occurs in {ep.domain}: "
                               f"check the immediate cause before changing code.")
        for i, lesson in enumerate(lessons):
            name = f"lesson: {lesson.strip()[:72]}"
            c = Concept(
                name=name, definition=lesson.strip(), domain=ep.domain,
                confidence=ep.confidence, status=status,
                provenance=[source, "sleep-consolidation"],
                importance=max(0.3, ep.salience))
            c = self.memory.semantic.upsert(c, provenance_note=source)
            created.append({"id": c.id, "name": c.name, "status": c.status})
            # 2. link to related existing concepts in the same domain
            self._link_related(c)
        # 3. a distillation concept for the objective itself (verified only)
        if ep.status == SUCCESS:
            name = f"solved: {ep.objective[:72]}"
            c = Concept(
                name=name,
                definition=f"Verified approach for '{ep.objective}': "
                           f"{ep.result[:200]}",
                domain=ep.domain, confidence=ep.confidence, status=FACT if ep.confidence >= 0.7 else HYPOTHESIS,
                provenance=[source, "sleep-consolidation"], importance=ep.salience)
            c = self.memory.semantic.upsert(c, provenance_note=source)
            created.append({"id": c.id, "name": c.name, "status": c.status})
            self._link_related(c)
        return created

    def _link_related(self, concept: Concept) -> None:
        for other in self.memory.semantic.all_active():
            if other.id == concept.id:
                continue
            sim = textsim.jaccard(concept.name + " " + concept.definition,
                                  other.name + " " + other.definition)
            if other.domain == concept.domain:
                sim = min(1.0, sim + 0.15)
            if sim >= 0.38:
                self.memory.semantic.link(concept.id, other.id, "related_to",
                                          weight=round(sim, 3))
                self.report.new_edges += 1

    # ------------------------------------------------------------- stage 3-4
    def _stage_pattern_discovery(self) -> None:
        """Generalize recurring (domain, error-signature) groups >= min size.

        A discovered pattern is a *correlation hypothesis* — the generated
        concept is explicitly tagged non-causal until verified by outcomes.
        """
        episodes = self.memory.episodic.all_active()
        groups: dict[tuple[str, str], list[Episode]] = {}
        for e in episodes:
            sigs = {self._error_signature(m) for m in e.mistakes}
            for sig in sigs:
                if not sig:
                    continue
                groups.setdefault((e.domain, sig), []).append(e)
        min_group = self.config.memory.pattern_min_group
        for (domain, sig), members in groups.items():
            if len(members) < min_group:
                continue
            lessons: list[str] = []
            for m in members:
                lessons.extend(m.lessons)
            general_lesson = ("; ".join(dict.fromkeys(lessons)))[:300] or \
                f"Repeated '{sig}' failures observed in {domain}."
            name = f"pattern: {sig} in {domain}"
            c = Concept(
                name=name,
                definition=(f"Across {len(members)} experiences, '{sig}' recurred in "
                            f"{domain}. General strategy: {general_lesson} "
                            f"[correlation, not proven causation; support={len(members)}]"),
                domain=domain, confidence=min(0.9, 0.4 + 0.1 * len(members)),
                status=HYPOTHESIS,
                provenance=["sleep-pattern-discovery"]
                + [f"episode:{m.id}/{m.source}" for m in members],
                support=len(members), importance=0.8)
            c = self.memory.semantic.upsert(c)
            self.report.pattern_concepts.append(
                {"id": c.id, "name": name, "members": len(members)})
            for m in members[:4]:
                for concept in self.memory.semantic.query(m.objective, k=1):
                    self.memory.semantic.link(c.id, concept.id, "supports", weight=0.6)

    def _stage_organization(self) -> None:
        """Merge duplicates, enforce version freshness, refresh links."""
        # merge near-duplicate concepts (highest confidence survives)
        for a, b, sim in self.memory.semantic.find_near_duplicates(threshold=0.82):
            ca = self.memory.semantic.get_by_id(a)
            cb = self.memory.semantic.get_by_id(b)
            if not ca or not cb:
                continue
            keep, drop = (ca, cb) if (ca.confidence, ca.support) >= (cb.confidence, cb.support) else (cb, ca)
            if self.memory.semantic.merge(keep.id, drop.id):
                self.report.merged_concepts.append(
                    {"kept": keep.name, "dropped": drop.name, "similarity": round(sim, 3)})
        # freshness: boring-date pattern like 'python 3.8' vs 'python 3.11'
        versioned: dict[str, list[Concept]] = {}
        rx = re.compile(r"^(.*?)[ _-]?v?(\d+(?:\.\d+)+)$")
        for c in self.memory.semantic.all_active():
            m = rx.match(c.name.strip())
            if m:
                versioned.setdefault(m.group(1).strip().lower(), []).append((m.group(2), c))
        for base, entries in versioned.items():
            if len(entries) < 2:
                continue
            def _ver_key(item):
                return [int(x) for x in item[0].split(".")]
            newest = max(entries, key=_ver_key)
            for ver, c in entries:
                if c is not newest[1] and not c.outdated:
                    self.memory.semantic.mark_outdated(c.id)
                    self.report.marked_outdated.append(c.name)

    # ------------------------------------------------------------- stage 5
    def _stage_forgetting(self) -> None:
        """Importance-based retention with decay; verified memories have a floor.

        Nothing important is deleted: low-retention items are *archived*
        (full payload preserved + recoverable), and active rows keep a small
        compressed summary.
        """
        cfg = self.config.memory
        now = time.time()
        tau = cfg.decay_tau_hours * 3600.0
        # episodes
        for e in self.memory.episodic.all_active():
            age_h = (now - e.timestamp) / 3600.0
            base = (0.30 * e.salience + 0.25 * e.confidence
                    + 0.15 * min(1.0, e.access_count / 5.0)
                    + (0.20 if e.status == SUCCESS else 0.05 if e.status == FAILURE else 0.0)
                    + (0.10 if e.consolidated else 0.0))
            retention = base * math.exp(-age_h / tau) if age_h > 0 else base
            if e.status == SUCCESS:
                retention = max(retention, cfg.verified_retention_floor)
            if retention < cfg.archive_epsilon and not e.consolidated:
                compressed = (f"Archived episode ({e.domain}/{e.status}): "
                              f"{e.objective[:100]} -> {e.result[:100]}")
                self.memory.episodic.archive(e.id, "retention decay", compressed)
                self.report.archived.append({"kind": "episode", "id": e.id,
                                             "retention": round(retention, 4)})
        # concepts: only weak, unsupported hypotheses may be archived;
        # protected domains (e.g. constitution rules) are immortal
        from ..memory.semantic import PROTECTED_DOMAINS
        for c in self.memory.semantic.all_active():
            if c.domain in PROTECTED_DOMAINS:
                continue
            age_h = (now - c.updated) / 3600.0
            base = (0.30 * c.importance + 0.25 * c.confidence
                    + 0.20 * min(1.0, c.access_count / 5.0)
                    + 0.15 * min(1.0, c.support / 4.0)
                    + (0.10 if c.status == FACT else 0.0))
            retention = base * math.exp(-age_h / tau) if age_h > 0 else base
            if c.status == FACT:
                retention = max(retention, cfg.verified_retention_floor)
            if retention < cfg.archive_epsilon and c.support < 2:
                self.memory.semantic.archive(c.id, "retention decay")
                self.report.archived.append({"kind": "concept", "id": c.id,
                                             "name": c.name,
                                             "retention": round(retention, 4)})

    # ------------------------------------------------------------- stage 6
    def _stage_learning(self) -> None:
        """Find weak domains, generate targeted practice, keep VERIFIED wins only."""
        episodes = self.memory.episodic.all_active()
        by_domain: dict[str, dict[str, int]] = {}
        for e in episodes:
            d = by_domain.setdefault(e.domain, {"success": 0, "failure": 0})
            if e.status == SUCCESS:
                d["success"] += 1
            elif e.status == FAILURE:
                d["failure"] += 1
        weaknesses = {
            dom: v["failure"] - v["success"]
            for dom, v in by_domain.items() if v["failure"] > v["success"]
        }
        self.report.weaknesses = dict(sorted(weaknesses.items(),
                                             key=lambda kv: kv[1], reverse=True))
        practice_pool = list(self.report.weaknesses.items())[:2]
        for domain, deficit in practice_pool:
            tasks = curriculum.practice_tasks_for(domain, k=2, seed=int(time.time()) % 10_000)
            for task in tasks:
                outcome = self._run_practice(task)
                self.report.practice.append(outcome)
        self._study_new_skills()

    def _study_new_skills(self) -> None:
        """Internalize newly added skills by actually TRAINING on them.

        Each unstudied skill is rendered into instruction-style training text
        (its trigger + its executable steps), then the skill adapter learns it:
        the training is benchmark-gated (level-4 rules), and the skill is
        marked ``studied`` only if the adapter survived the gate.
        """
        if self.skill_trainer is None:
            return
        new_skills = self.memory.procedural.unstudied()
        if not new_skills:
            return
        new_skills = new_skills[: self.config.sleep.skill_study_max]
        texts: list[str] = []
        names: list[str] = []
        ids: list[str] = []
        for s in new_skills:
            body_lines: list[str] = []
            for step in s.steps:
                action, args = step.get("action"), step.get("args", {})
                if action == "run_python":
                    body_lines.append(args.get("code", "").strip())
                elif action == "run_shell":
                    body_lines.append("# shell: " + args.get("cmd", "").strip())
                elif action == "write_file":
                    body_lines.append(f"# write {args.get('path','?')}\n"
                                      f"{args.get('content','')[:200]}")
                elif action == "think":
                    body_lines.append("# plan: " + args.get("note", "").strip())
            body = "\n".join(b for b in body_lines if b).strip() or "# (no code steps)"
            triggers = ", ".join(s.triggers[:6])
            texts.append(
                f"### Task: {s.name} — {s.description}\n"
                f"### Keywords: {triggers}\n"
                f"### Solution:\n{body}\n### End\n")
            names.append(s.name)
            ids.append(s.id)
        try:
            result = self.skill_trainer(texts, names)
        except Exception as exc:                    # noqa: BLE001
            self.report.notes.append(f"skill study failed: {exc!r}")
            self.report.unresolved.append("skill study failed; retry next sleep")
            return
        if result.get("decision") == "kept":
            self.memory.procedural.mark_studied(ids)
            self.report.skills_studied = names
            self.memory.episodic.record(Episode(
                situation="sleep stage 6", objective="study new skills",
                actions=[f"trained adapter on {len(ids)} skills"],
                result=f"internalized: {', '.join(names)}",
                status=SUCCESS, source="sleep-skill-training", confidence=0.85,
                salience=0.7, domain="meta-learning",
                lessons=[f"Skill(s) studied during sleep: {', '.join(names)}"]))
        else:
            self.report.notes.append(
                f"skill study adapter rejected by benchmark gate: {result.get('benchmark_before')} -> {result.get('benchmark_after')}")
            self.report.unresolved.append(f"skills not yet internalized: {', '.join(names)}")

    def _run_practice(self, task: curriculum.Task) -> dict:
        record = {"task_id": task.task_id, "domain": task.domain,
                  "status": "skipped"}
        if self.practice_solver is None:
            record["status"] = "queued"      # no solver wired; noted for next cycle
            return record
        try:
            code = self.practice_solver(task)[:4000]
        except Exception as exc:               # noqa: BLE001
            record["status"] = "error"
            record["error"] = repr(exc)
            return record
        res = self.sandbox.run_python(code, tests=task.tests)
        if res.get("tests_passed"):
            ep = Episode(
                situation=f"sleep practice in {task.domain}",
                objective=task.prompt, actions=[f"python:{code}"],
                result=res.get("stdout", "")[:300] or "tests passed",
                status=SUCCESS, lessons=[f"Practice verified solution for {task.task_id}"],
                source="sleep-practice", confidence=0.9, salience=0.6,
                domain=task.domain)
            self.memory.episodic.record(ep)
            self.memory.procedural.learn_from_episode(ep, name=f"solve:{task.task_id}")
            record["status"] = "verified"
        else:
            ep = Episode(
                situation=f"sleep practice in {task.domain}",
                objective=task.prompt, actions=[f"python:{code}"],
                result=(res.get("stderr") or res.get("error") or "failed")[:300],
                status=UNVERIFIED,
                mistakes=[(res.get("error") or "practice failed")[:200]],
                source="sleep-practice", confidence=0.2, salience=0.5,
                domain=task.domain)
            self.memory.episodic.record(ep)
            record["status"] = "unverified"
            record["error"] = res.get("error")
            self.report.unresolved.append(f"practice failed: {task.task_id}")
        return record

    # ------------------------------------------------------------- stage 7
    def _stage_wake(self) -> None:
        for contra in self.memory.semantic.contradictions():
            self.report.unresolved.append(f"contradiction: {contra['note'][:120]}")
        self.report.status = "completed"

    # ------------------------------------------------------------- internals
    @staticmethod
    def _error_signature(mistake: str) -> Optional[str]:
        m = ERROR_SIG_RE.search(mistake or "")
        return m.group(1).lower() if m else None

    def _checkpoint(self, cycle_id: str, stage: int) -> None:
        self.memory.store.execute(
            "INSERT OR REPLACE INTO sleep_checkpoints(cycle_id, stage, payload) "
            "VALUES (?,?,?)",
            (cycle_id, stage, json.dumps(self.report.to_dict(), default=str)))
        self.memory.store.conn.commit()

    def _completed_stages(self, cycle_id: str) -> set[int]:
        rows = self.memory.store.execute(
            "SELECT stage FROM sleep_checkpoints WHERE cycle_id=?",
            (cycle_id,)).fetchall()
        return {int(r["stage"]) for r in rows}

    def _load_report(self, cycle_id: str) -> SleepReport:
        rows = self.memory.store.execute(
            "SELECT stage, payload FROM sleep_checkpoints WHERE cycle_id=? "
            "ORDER BY stage DESC", (cycle_id,)).fetchall()
        if rows:
            data = json.loads(rows[0]["payload"])
            rep = SleepReport(**{k: v for k, v in data.items()
                                 if k in SleepReport.__dataclass_fields__})
        else:
            rep = SleepReport(cycle_id=cycle_id, started=time.time())
        rep.resumed_from = cycle_id
        return rep

    def _finalize(self, cycle_id: str) -> None:
        self.report.ended = time.time()
        out = self.report.to_dict()
        self.memory.store.execute(
            "INSERT OR REPLACE INTO sleep_cycles(id, started, ended, status, report) "
            "VALUES (?,?,?,?,?)",
            (cycle_id, self.report.started, self.report.ended,
             self.report.status, json.dumps(out, default=str)))
        self.memory.store.conn.commit()
        logs_dir = self.config.memory_path("sleep_logs")
        atomic_write_json(
            f"{logs_dir}/report-{cycle_id}.json", out)
