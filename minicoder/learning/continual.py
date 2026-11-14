"""Continual learning manager — the safe, gated learning hierarchy (spec §5):

    LEVEL 1  update external memory                  — always allowed
    LEVEL 2  learn reusable skills/procedures        — verified via sandbox
    LEVEL 3  generate targeted training examples     — verified before use
    LEVEL 4  train LoRA adapters                     — benchmark-gated, reversible
    LEVEL 5  update core model                       — strictest gate: prior
             checkpoint + benchmark comparison + regression test + rollback

Memory learning (levels 1-2) and neural weight learning (levels 4-5) are
separate processes with separate logs.  Every neural update leaves a
reproducible training record in ``learn_log``.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

import numpy as np

from ..config import BrainConfig
from ..curriculum import practice_tasks_for, build_corpus, PROVENANCE
from ..memory.system import MemorySystem
from ..memory.store import new_id, atomic_write_json
from ..model.tokenizer import ByteBPETokenizer
from ..model.transformer import MiniGPT
from ..model.lora import LoRAAdapter
from ..model.trainer import Trainer, CheckpointManager
from ..tools.sandbox import Sandbox

BenchmarkFn = Callable[[Optional[LoRAAdapter]], float]


class ContinualLearner:
    def __init__(self, config: BrainConfig, memory: MemorySystem,
                 sandbox: Sandbox, checkpoints: CheckpointManager):
        self.config = config
        self.memory = memory
        self.sandbox = sandbox
        self.checkpoints = checkpoints

    # ------------------------------------------------------------- logging
    def _log(self, level: int, summary: str, details: dict) -> None:
        self.memory.store.execute(
            "INSERT INTO learn_log(id, timestamp, level, summary, details) "
            "VALUES (?,?,?,?,?)",
            (new_id(summary), time.time(), level, summary,
             json.dumps(details, default=str)))
        self.memory.store.conn.commit()

    def history(self) -> list[dict]:
        rows = self.memory.store.execute(
            "SELECT * FROM learn_log ORDER BY timestamp DESC LIMIT 100").fetchall()
        return [{"id": r["id"], "timestamp": r["timestamp"], "level": r["level"],
                 "summary": r["summary"], "details": json.loads(r["details"])}
                for r in rows]

    # -------------------------------------------------------------- level 1
    def level1_update_memory(self, concepts: list) -> dict:
        """Write concepts through semantic memory (provenance preserved)."""
        written = 0
        for c in concepts:
            self.memory.semantic.upsert(c)
            written += 1
        self._log(1, f"memory update: {written} concepts", {"written": written})
        return {"written": written}

    # -------------------------------------------------------------- level 2
    def level2_learn_skills(self, min_confidence: float = 0.7) -> dict:
        """Distill verified episodes into reusable skills (start untrusted until
        re-executed in the sandbox)."""
        learned = 0
        for ep in self.memory.episodic.all_active():
            if not ep.verified or ep.confidence < min_confidence:
                continue
            if self.memory.procedural.learn_from_episode(ep):
                learned += 1
        self._log(2, f"skill distillation: {learned} candidate skills",
                  {"learned": learned})
        return {"learned": learned}

    # -------------------------------------------------------------- level 3
    def level3_generate_examples(self, domain: str, k: int = 4) -> dict:
        """Generate targeted training examples for a weak domain and *verify*
        each reference solution in the sandbox before it may be used."""
        tasks = practice_tasks_for(domain, k=k, seed=int(time.time()) % 100_000)
        verified = []
        for t in tasks:
            res = self.sandbox.run_python(t.reference, tests=t.tests)
            if res.get("tests_passed"):
                verified.append(t.instruction_pair())
        out = self.config.memory_path(
            "datasets", f"targeted-{domain}-{int(time.time())}.jsonl")
        atomic_write_json(out, {"domain": domain, "provenance": PROVENANCE,
                                "samples": verified})
        self._log(3, f"targeted examples for '{domain}': {len(verified)}/{len(tasks)} verified",
                  {"domain": domain, "verified": len(verified), "path": out})
        return {"verified": len(verified), "generated": len(tasks), "path": out}

    # -------------------------------------------------------------- level 4
    def level4_train_adapter(self, model: MiniGPT, tokenizer: ByteBPETokenizer,
                             benchmark: BenchmarkFn, *,
                             name: str = "adapter",
                             corpus_seed: int = 0) -> dict:
        """Train a LoRA adapter on the legal curriculum corpus.  The adapter is
        kept ONLY if the benchmark does not regress beyond tolerance."""
        corpus = build_corpus(samples_per_task=2, seed=corpus_seed)
        seqs = [tokenizer.encode(t) for t in corpus]
        cfg = self.config.lora
        adapter = LoRAAdapter(model, rank=cfg.rank, alpha=cfg.alpha, seed=1)
        before = benchmark(adapter)                       # adapter neutral ⇒ == base
        trainer = Trainer(model, self.config.train)
        result = trainer.fit(seqs, steps=cfg.train_steps, adapter=adapter,
                             adapter_only=True)
        after = benchmark(adapter)
        regress_ok = (after - before) >= -cfg.regression_tolerance
        decision = "kept" if regress_ok else "discarded"
        record = {"benchmark_before": before, "benchmark_after": after,
                  "regression_tolerance": cfg.regression_tolerance,
                  "adapter_params": adapter.param_count(), "decision": decision,
                  "train": result}
        if regress_ok:
            self.checkpoints.save(model, name, metrics={"benchmark": after},
                                  adapter=adapter)
        else:
            record["rolled_back"] = True
        self._log(4, f"adapter '{name}' {decision}: {before:.3f} -> {after:.3f}",
                  record)
        return record

    # -------------------------------------------------------------- level 5
    def level5_update_core(self, model: MiniGPT, tokenizer: ByteBPETokenizer,
                           benchmark: BenchmarkFn, *,
                           approved: bool = False, steps: int = 40,
                           corpus_seed: int = 0) -> dict:
        """The strictest gate.  Requires explicit approval.  Saves the previous
        checkpoint FIRST, trains with replay + EWC, compares benchmarks, and
        rolls back automatically on regression.  The only working checkpoint is
        never modified in place."""
        if not approved:
            self._log(5, "core update refused (not approved)", {})
            return {"refused": True, "reason": "approved=False"}
        prev = self.checkpoints.save(model, "pre-update", metrics={})
        before = benchmark(None)
        corpus = [tokenizer.encode(t) for t in build_corpus(2, corpus_seed)]
        trainer = Trainer(model, self.config.train)
        replay = corpus[: max(1, len(corpus) // 2)]
        fisher = trainer.fisher_estimate(corpus, batches=2, batch=4, seq=48)
        ewc = Trainer.ewc_state(model, fisher, lam=0.4)
        result = trainer.fit(corpus, steps=steps, replay_sequences=replay,
                             replay_ratio=0.3, ewc=ewc)
        after = benchmark(None)
        tol = self.config.lora.regression_tolerance
        improved = after >= before - tol
        record = {"before": before, "after": after, "tolerance": tol,
                  "train": result, "previous_checkpoint": prev.name}
        if improved:
            self.checkpoints.save(model, "core-update", metrics={"benchmark": after})
            record["decision"] = "kept"
        else:
            rolled = self.checkpoints.rollback()
            restored, _meta = self.checkpoints.load(rolled["name"])
            model.params = restored.params
            record["decision"] = "rolled_back"
            record["restored_checkpoint"] = rolled["name"]
        self._log(5, f"core update {record['decision']}: {before:.3f} -> {after:.3f}",
                  record)
        return record
