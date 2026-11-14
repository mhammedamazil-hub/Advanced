"""Brain — the orchestrator implementing the lifecycle:

    WAKE → PERCEIVE → THINK → LEARN → REMEMBER → SLEEP → CONSOLIDATE → IMPROVE → WAKE

The brain never starts with an empty mind: memory is persistent across
sessions, so wake() restores working memory and all long-term stores.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from .config import BrainConfig, SleepConfig
import os
import numpy as np
from .memory.system import MemorySystem
from .memory.episodic import Episode, SUCCESS, FAILURE, UNVERIFIED
from .metacog import MetaCognition
from .sleep.engine import SleepEngine, SleepReport
from .tools.sandbox import Sandbox
from .curriculum import Task, build_corpus
from .model.tokenizer import ByteBPETokenizer
from .model.transformer import MiniGPT, param_count
from .model.trainer import Trainer, CheckpointManager
from .model.lora import default_targets  # noqa: F401  (re-export surface)

AWAKE, THINKING, LEARNING, SLEEPING = "awake", "thinking", "learning", "sleeping"


@dataclass
class SolveResult:
    code: str
    via: str                     # memory-reuse | skill | model | model+memory-hints
    verified: bool
    confidence: float
    episode_id: str
    detail: str = ""


class Brain:
    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        self.memory = MemorySystem(self.config)
        self.sandbox = Sandbox(self.config.memory_path("sandbox"))
        self.metacog = MetaCognition(self.memory)
        self.model: Optional[MiniGPT] = None
        self.model_meta: Optional[dict] = None
        self.tokenizer: Optional[ByteBPETokenizer] = None
        self.checkpoints = CheckpointManager(self.config.checkpoints_dir)
        self.state = AWAKE
        self.last_activity = time.time()
        self.experiences_since_sleep = 0
        self.last_sleep_report: Optional[dict] = None
        self._sleep_thread: Optional[threading.Thread] = None
        self._sleep_error: Optional[str] = None
        self.birth = self.memory.store.get_meta("birth", None)
        if self.birth is None:
            self.birth = time.time()
            self.memory.store.set_meta("birth", self.birth)
            self.memory.store.conn.commit()
        self._rng = np.random.default_rng(self.config.train.seed)

    # ------------------------------------------------------------------ model
    def ensure_model(self, fresh: bool = False) -> MiniGPT:
        """Load the newest checkpoint if any; otherwise initialize fresh."""
        if self.model is not None and not fresh:
            return self.model
        try:
            self.model, meta = self.checkpoints.load()
            self.model_meta = meta
        except FileNotFoundError:
            self.model = MiniGPT(self.config.model)
            self.model_meta = {"name": "(fresh init)", "metrics": {}}
        if self.tokenizer is None:
            tok_path = self.config.memory_path("tokenizer.json")
            import os
            if os.path.exists(tok_path) and not fresh:
                self.tokenizer = ByteBPETokenizer.load(tok_path)
            else:
                self.tokenizer = ByteBPETokenizer()
                self.tokenizer.train(build_corpus(2, 0),
                                     target_vocab=self.config.model.vocab_size - 1)
                self.tokenizer.save(tok_path)
        return self.model

    def model_info(self) -> dict:
        info = {"available": self.model is not None}
        if self.model is not None:
            from .model.quantize import size_report
            info.update({
                "params": param_count(self.model),
                "config": {k: v for k, v in self.config.model.__dict__.items()},
                "checkpoint": (self.model_meta or {}).get("name"),
                "size": size_report(self.model),
            })
        return info

    # ---------------------------------------------------------------- perceive
    def perceive(self, kind: str, content: str, *, importance: float = 0.5,
                 tags: Optional[list[str]] = None, pinned: bool = False) -> None:
        self.memory.working.add(kind, content, importance=importance,
                                tags=tags, pinned=pinned)
        self.last_activity = time.time()

    # ------------------------------------------------------------------- think
    def prepare(self, goal: str, required: Optional[list[str]] = None) -> dict:
        """THINK: probe memory, identify gaps, pick a strategy (all measurable)."""
        self.metacog.begin_task(goal)
        gaps = self.metacog.find_gaps(goal, required=required)
        strategy = self.metacog.select_strategy(goal)
        self.perceive("task", goal, importance=0.8, tags=["active-task"])
        return {"gaps": gaps, **strategy}

    # ------------------------------------------------------------------- solve
    def solve(self, task: Task, max_new: int = 130, temperature: float = 0.0,
              record: bool = True) -> SolveResult:
        """Full THINK→ACT→VERIFY→LEARN cycle for one coding task."""
        self.state = THINKING
        self.prepare(task.prompt)
        # ---------- 1. memory exact reuse (verified episode for same objective)
        for ep in self.memory.episodic.recall(task.prompt, k=3):
            if ep.verified and ep.objective == task.prompt and ep.actions:
                for action in ep.actions:
                    if action.startswith("python:"):
                        code = action[len("python:"):]
                        res = self.sandbox.run_python(code, tests=task.tests)
                        verified = bool(res.get("tests_passed"))
                        out = self._finish_solve(task, code, "memory-reuse",
                                                 verified, record,
                                                 res.get("error"), attempt_ep=ep)
                        self.state = AWAKE
                        return out
        # ---------- 2. verified skill
        for skill in self.memory.procedural.recall(task.prompt, k=1):
            if skill.trust == "verified":
                res = self.memory.procedural.execute(skill, self.sandbox)
                ctx = res.get("context", {})
                code = ""
                for step in skill.steps:
                    if step.get("action") == "run_python":
                        code = step["args"].get("code", "")
                if res.get("ok") and code:
                    vres = self.sandbox.run_python(code, tests=task.tests)
                    verified = bool(vres.get("tests_passed"))
                    out = self._finish_solve(task, code, "skill", verified, record,
                                             vres.get("error"))
                    self.state = AWAKE
                    return out
        # ---------- 3. model generation (best-of-N, verification selects)
        model = self.ensure_model()
        hints = self._memory_hints(task.prompt)
        prompt_text = f"### Task: {task.prompt}\n### Solution:\n"
        prompt_ids = self.tokenizer.encode(prompt_text, add_eos=False)
        via = "model+memory-hints" if hints else "model"
        record_attempt = record
        best: Optional[SolveResult] = None
        errors: list[str] = []
        candidates = [(0.0,)] + [(0.7,)] * max(0, self.config.solve_samples - 1)
        rng = self._rng
        for (temp,) in candidates:
            ids = model.generate(prompt_ids, max_new=max_new, temperature=temp,
                                 top_k=0 if temp <= 0 else 24, stop_id=258,
                                 rng=rng if temp > 0 else None)
            code = self.tokenizer.decode(ids)
            res = self.sandbox.run_python(code, tests=task.tests)
            if res.get("tests_passed"):
                out = self._finish_solve(task, code, via, True, record_attempt,
                                         None)
                self.state = AWAKE
                return out
            errors.append((res.get("error") or "failed")[:120])
            record_attempt = False            # only one episode per task
        out = self._finish_solve(task, code, via, False, record,
                                 errors[0] if errors else "no output")
        self.state = AWAKE
        return out

    def _memory_hints(self, prompt: str) -> list[str]:
        concepts = self.memory.semantic.query(prompt, k=2)
        return [c.definition for c in concepts if c.score > 0.35]

    def _finish_solve(self, task: Task, code: str, via: str, verified: bool,
                      record: bool, error: Optional[str],
                      attempt_ep: Optional[Episode] = None) -> SolveResult:
        mistakes = [error] if (error and not verified) else []
        reflection = self.metacog.self_evaluate(
            task.prompt, worked=verified, verified=verified, mistakes=mistakes)
        conf = self.metacog.estimate_confidence(task.prompt, verified=verified,
                                                had_tool_errors=bool(mistakes))
        ep_id = attempt_ep.id if attempt_ep else ""
        if record and reflection.record_long_term and attempt_ep is None:
            status = SUCCESS if verified else (FAILURE if mistakes else UNVERIFIED)
            ep = Episode(
                situation=f"coding task in {task.domain}",
                objective=task.prompt,
                actions=[f"python:{code}"],
                result=("tests passed" if verified else (error or "unverified"))[:300],
                status=status,
                mistakes=mistakes,
                lessons=[reflection.lesson] if verified or mistakes else [],
                source="execution" if verified else "hypothesis",
                confidence=conf, salience=reflection.salience,
                domain=task.domain)
            self.memory.episodic.record(ep)
            self.experiences_since_sleep += 1
            ep_id = ep.id
            if verified:
                self.memory.procedural.learn_from_episode(ep)   # level-2 candidate
        self.metacog.end_task(resolved=verified)
        self.last_activity = time.time()
        return SolveResult(code=code, via=via, verified=verified,
                           confidence=conf, episode_id=ep_id,
                           detail=reflection.lesson)

    # ------------------------------------------------------------------- learn
    def train_base(self, steps: Optional[int] = None, seed: int = 0,
                   log: bool = False) -> dict:
        """Train the compact model on the synthetic curriculum corpus."""
        self.state = LEARNING
        model = self.ensure_model()
        corpus = build_corpus(samples_per_task=2, seed=seed)
        seqs = [self.tokenizer.encode(t) for t in corpus]
        trainer = Trainer(model, self.config.train)
        result = trainer.fit(seqs, steps=steps,
                             log_fn=(print if log else None))
        info = self.checkpoints.save(model, "base",
                                     metrics={"final_loss": result["final_loss"]})
        result["checkpoint"] = info.name
        result["corpus_samples"] = len(corpus)
        self.memory.store.execute(
            "INSERT INTO learn_log(id, timestamp, level, summary, details) "
            "VALUES (?,?,?,?,?)",
            (f"base-{int(time.time())}", time.time(), 4,
             f"base training: {result['final_loss']:.3f} loss",
             __import__("json").dumps(result, default=str)))
        self.memory.store.conn.commit()
        self.state = AWAKE
        return result

    # ------------------------------------------------------------------- sleep
    def sleep(self, resume_cycle: Optional[str] = None) -> SleepReport:
        """Blocking sleep cycle: replay → consolidate → … → improve → wake."""
        if self.state == SLEEPING:
            raise RuntimeError("already sleeping")
        self.state = SLEEPING
        engine = SleepEngine(self.memory, self.config, self.sandbox,
                             practice_solver=self._practice_solver)
        report = engine.run(resume_cycle=resume_cycle)
        self.last_sleep_report = report.to_dict()
        self.experiences_since_sleep = 0
        self.state = AWAKE
        return report

    def sleep_async(self) -> str:
        """Non-blocking sleep for the dashboard; returns cycle/thread state."""
        if self.state == SLEEPING:
            return "already sleeping"
        self.state = SLEEPING
        self._sleep_error = None

        def _runner():
            try:
                engine = SleepEngine(self.memory, self.config, self.sandbox,
                                     practice_solver=self._practice_solver)
                report = engine.run()
                self.last_sleep_report = report.to_dict()
            except Exception as exc:           # noqa: BLE001
                self._sleep_error = repr(exc)
            finally:
                self.experiences_since_sleep = 0
                self.state = AWAKE
        self._sleep_thread = threading.Thread(target=_runner, daemon=True)
        self._sleep_thread.start()
        return "sleep started"

    def _practice_solver(self, task: Task) -> str:
        """Used by sleep stage 6: solve practice tasks with current abilities."""
        result = self.solve(task, max_new=80, record=False)
        return result.code

    # -------------------------------------------------------------------- wake
    def wake(self) -> dict:
        self.state = AWAKE
        self.memory.working.clear(keep_pinned=True)
        self.perceive("note", f"woke at {time.strftime('%H:%M:%S')}",
                      importance=0.2)
        return {"state": self.state, "working": self.memory.working.stats()}

    def tick(self) -> Optional[str]:
        """Auto-sleep triggers: idleness or accumulated experiences."""
        cfg: SleepConfig = self.config.sleep
        if self.state == SLEEPING:
            return None
        if self.experiences_since_sleep >= cfg.new_experiences_trigger:
            return self.sleep_async()
        if time.time() - self.last_activity > cfg.idle_trigger_seconds:
            return self.sleep_async()
        return None

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "state": self.state,
            "uptime_s": round(time.time() - self.birth, 1),
            "experiences_since_sleep": self.experiences_since_sleep,
            "last_activity": self.last_activity,
            "sleep_error": self._sleep_error,
            "memory": self.memory.stats(),
            "metacognition": self.metacog.report(),
            "model": self.model_info(),
            "sandbox": self.sandbox.stats(),
            "last_sleep_report": self.last_sleep_report,
            "checkpoints": self.checkpoints.list(),
        }

    def close(self) -> None:
        self.memory.close()
