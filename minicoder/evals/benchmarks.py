"""Coding benchmarks.

Honesty note: with no network to fetch HumanEval/MBPP in this sandbox, we ship
**Mini-Bench**, an in-house proxy suite built from the curriculum taxonomy
with a *held-out parameterization* (seed 999) so it is never literally trained
on.  Methodology mirrors HumanEval: a candidate solution is extracted from
model output and verified by execution against unit tests in the sandbox.
Docs explain how to plug in real HumanEval/MBPP on a connected machine.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from ..curriculum import executable_tasks, Task
from ..tools.sandbox import Sandbox

BENCH_SEED = 999          # held-out: different from any training seed
MINI_BENCH: list[Task] = executable_tasks(seed=BENCH_SEED)

Solver = Callable[[Task], str]


def extract_code(text: str) -> str:
    """Extract the code part of a generation (between '### Solution:' and end)."""
    marker = "### Solution:"
    if marker in text:
        text = text.split(marker, 1)[1]
    for stop in ("### End", "### Task"):
        if stop in text:
            text = text.split(stop, 1)[0]
    lines = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and "Solution" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def evaluate_solver(solver: Solver, sandbox: Sandbox,
                    tasks: Optional[list[Task]] = None) -> dict:
    """Run a solver over the benchmark; every candidate is executed+tested."""
    tasks = tasks or MINI_BENCH
    records = []
    passed = 0
    for t in tasks:
        t0 = time.time()
        try:
            raw = solver(t)[:6000]
        except Exception as exc:            # noqa: BLE001
            raw = ""
            err = repr(exc)
        else:
            err = None
        code = extract_code(raw) if ("###" in raw) or err is None else raw
        res = sandbox.run_python(code, tests=t.tests, timeout=4.0)
        ok = bool(res.get("tests_passed")) and err is None
        passed += int(ok)
        records.append({
            "task_id": t.task_id, "domain": t.domain, "ok": ok,
            "error": err or res.get("error"),
            "code_chars": len(code), "duration_s": round(time.time() - t0, 3),
        })
    return {"score": passed / max(1, len(tasks)), "passed": passed,
            "total": len(tasks), "records": records}


def run_benchmark_suite(store, label: str, solver: Solver,
                        sandbox: Sandbox) -> dict:
    """Evaluate + persist result in the benchmarks table for the dashboard."""
    import json
    from ..memory.store import new_id
    result = evaluate_solver(solver, sandbox)
    store.execute(
        "INSERT INTO benchmarks(id, timestamp, label, score, details) VALUES (?,?,?,?,?)",
        (new_id(label), time.time(), label, result["score"],
         json.dumps(result)))
    store.conn.commit()
    return result
