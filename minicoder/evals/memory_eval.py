"""Memory + consolidation + forgetting evaluations.

Each function returns numeric metrics and cleans up its own synthetic
"canary" records afterwards (canaries are eval artifacts, never real memories).
"""
from __future__ import annotations

import time
from typing import Optional

from ..memory.episodic import Episode, SUCCESS, FAILURE, UNVERIFIED
from ..memory.semantic import Concept, FACT, HYPOTHESIS
from ..memory.system import MemorySystem
from ..config import BrainConfig
from ..tools.sandbox import Sandbox

CANARY = "eval-canary"


def _purge(memory: MemorySystem) -> None:
    st = memory.store
    with st.transaction() as cur:
        for table, col in (("episodes", "source"), ("concepts", "provenance")):
            if table == "concepts":
                cur.execute("DELETE FROM concepts WHERE provenance LIKE ?",
                            (f"%{CANARY}%",))
            else:
                cur.execute("DELETE FROM episodes WHERE source=?", (CANARY,))
        cur.execute("DELETE FROM contradictions WHERE note LIKE ?",
                    (f"%{CANARY}%",))


# ---------------------------------------------------------------------------
def retrieval_precision(memory: MemorySystem, probes: int = 8) -> dict:
    """Plant known concepts, query with *paraphrases*, measure hit@3."""
    planted = []
    for i in range(probes):
        c = Concept(
            name=f"{CANARY} fact {i}: bubbling sorts swap neighbours",
            definition=(f"Synthetic verification fact {i}: bubble sort repeatedly "
                        f"swaps adjacent out-of-order elements; complexity O(n^2)."),
            domain="eval", confidence=0.9, status=FACT, provenance=[CANARY],
            importance=0.6)
        planted.append(memory.semantic.upsert(c))
    queries = [
        "how does bubble sort swap adjacent elements",
        "quadratic pairwise exchange sorting algorithm",
        "which sort compares neighbours and swaps them",
        "Complexity O(n^2) exchange sort",
        "adjacent swap sorting fact 0",
        "adjacent swap sorting fact 1",
        "synthetic verification fact 2 complexity",
        "bubble sort neighbour exchange",
    ][:probes]
    hits = 0
    hits_at1 = 0
    for q in queries:
        res = memory.semantic.query(q, k=3)
        if any(CANARY in c.name for c in res):
            hits += 1
        if res and CANARY in res[0].name:
            hits_at1 += 1
    _purge(memory)
    n = max(1, len(queries))
    return {"queries": len(queries), "hit@3": round(hits / n, 3),
            "hit@1": round(hits_at1 / n, 3)}


# ---------------------------------------------------------------------------
def retention_after_decay(memory: MemorySystem) -> dict:
    """Verified knowledge must survive decay; junk should be archived and
    must remain recoverable."""
    from ..sleep.engine import SleepEngine
    cfg = memory.config
    old = time.time() - 35 * 24 * 3600          # 35 simulated days
    important = Episode(
        situation="learned a verified fix", objective="fix ImportError by installing deps",
        actions=["shell: pip install deps"], result="fixed", status=SUCCESS,
        lessons=["verified fix"], source=CANARY, confidence=0.95, salience=0.9,
        domain="eval")
    important.timestamp = old
    junk = Episode(
        situation="idle chatter", objective="nothing of value",
        actions=["think: none"], result="ok", status=UNVERIFIED, mistakes=[],
        lessons=[], source=CANARY, confidence=0.1, salience=0.05, domain="eval")
    junk.timestamp = old
    memory.episodic.record(important)
    memory.episodic.record(junk)
    sandbox = Sandbox(cfg.memory_path("sandbox"))
    engine = SleepEngine(memory, cfg, sandbox)
    engine._stage_replay()
    engine._stage_forgetting()                    # directly exercise stage 5
    kept = memory.episodic.get(important.id)
    junk_after = memory.episodic.get(junk.id)
    recovered = memory.episodic.recover_archive(junk.id)
    junk_rec = memory.episodic.get(junk.id)
    _purge(memory)
    return {
        "important_survived": bool(kept and not kept.archived),
        "junk_archived": bool(junk_after and junk_after.archived),
        "archive_recoverable": bool(recovered and junk_rec and not junk_rec.archived),
    }


# ---------------------------------------------------------------------------
def consolidation_quality(memory: MemorySystem) -> dict:
    """Measure what a full sleep cycle changes: knowledge extracted, duplicates
    merged, retrieval of lesson knowledge before vs after."""
    from ..sleep.engine import SleepEngine
    cfg = memory.config
    before = memory.semantic.stats()
    # 4 similar experiences about the same bug → pattern discovery target
    for i in range(4):
        ep = Episode(
            situation=f"ci run {i}", objective=f"fix ci failure {i}",
            actions=["shell: pip install -r requirements.txt"],
            result="fixed by installing missing dependency",
            status=SUCCESS if i % 2 == 0 else FAILURE,
            mistakes=["ModuleNotFoundError: no module named requests"],
            lessons=["ensure CI installs dependencies from requirements.txt"],
            source=CANARY, confidence=0.85 if i % 2 == 0 else 0.5,
            salience=0.7, domain="eval")
        memory.episodic.record(ep)
    q = "ensure CI installs dependencies"
    pre = memory.semantic.query(q, k=3)
    pre_best = max((c.score for c in pre), default=0.0)
    sandbox = Sandbox(cfg.memory_path("sandbox"))
    engine = SleepEngine(memory, cfg, sandbox)
    report = engine.run()
    post = memory.semantic.query(q, k=3)
    post_best = max((c.score for c in post), default=0.0)
    after = memory.semantic.stats()
    patterns = report.pattern_concepts
    _purge(memory)
    return {
        "concepts_before": before["count"], "concepts_after": after["count"],
        "new_concepts": len(report.new_concepts),
        "patterns_found": len(patterns),
        "retrieval_best_before": round(pre_best, 3),
        "retrieval_best_after": round(post_best, 3),
        "report_status": report.status,
    }


# ---------------------------------------------------------------------------
def forgetting_resistance(model, trainer_factory, seq_a: list[list[int]],
                          seq_b: list[list[int]], steps: int = 24) -> dict:
    """Catastrophic-forgetting probe on the real NumPy model.

    Train on task A, measure loss A; then train on task B three ways in
    *copies* of the model: naive, replay-mixed, EWC.  Compare loss on A after.
    """
    import copy
    import numpy as np
    from ..model.transformer import MiniGPT

    def loss_on(m: MiniGPT, seqs) -> float:
        x = np.array([s[:-1] for s in seqs[:6]], dtype=np.int64)
        y = np.array([s[1:] for s in seqs[:6]], dtype=np.int64)
        loss, _ = m.loss_and_grads(x, y)
        return loss

    base = copy.deepcopy(model)
    t0 = trainer_factory(base)
    t0.fit(seq_a, steps=steps)
    a_before = loss_on(base, seq_a)
    out = {"loss_A_after_A": round(a_before, 4)}
    for label, kwargs in (
            ("naive", {}),
            ("replay", {"replay_sequences": seq_a, "replay_ratio": 0.4}),
            ("ewc", {})):
        m = copy.deepcopy(base)
        tr = trainer_factory(m)
        if label == "ewc":
            fisher = tr.fisher_estimate(seq_a, batches=2, batch=4,
                                        seq=min(24, len(seq_a[0]) - 1))
            kwargs["ewc"] = tr.ewc_state(m, fisher, lam=0.4)
        tr.fit(seq_b, steps=steps, **kwargs)
        out[f"loss_A_after_B_{label}"] = round(loss_on(m, seq_a), 4)
    out["forgetting_naive"] = round(out["loss_A_after_B_naive"] - a_before, 4)
    out["forgetting_replay"] = round(out["loss_A_after_B_replay"] - a_before, 4)
    out["forgetting_ewc"] = round(out["loss_A_after_B_ewc"] - a_before, 4)
    return out
