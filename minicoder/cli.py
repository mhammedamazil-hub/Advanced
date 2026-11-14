"""MINICODER command-line interface.

Examples
--------
    python -m minicoder.cli demo --fast          # full lifecycle demo
    python -m minicoder.cli train --steps 120    # train compact model
    python -m minicoder.cli solve py-fizzbuzz    # solve a curriculum task
    python -m minicoder.cli sleep                # run a sleep cycle now
    python -m minicoder.cli dashboard --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .brain import Brain
from .config import BrainConfig
from .curriculum import executable_tasks, Task
from .memory.episodic import Episode


def _brain(args) -> Brain:
    cfg = BrainConfig(base_dir=args.data)
    return Brain(cfg)


def cmd_status(brain: Brain, _args) -> int:
    print(json.dumps(brain.status(), indent=2, default=str))
    return 0


def cmd_train(brain: Brain, args) -> int:
    brain.ensure_model()
    result = brain.train_base(steps=args.steps, log=True)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_solve(brain: Brain, args) -> int:
    brain.ensure_model()
    tasks = {t.task_id: t for t in executable_tasks(seed=0)}
    task = tasks.get(args.task_id)
    if task is None:
        task = Task(task_id=args.task_id, domain="general",
                    prompt=args.task_id, reference="", tests="assert True")
    res = brain.solve(task, temperature=args.temperature)
    print(json.dumps({"via": res.via, "verified": res.verified,
                      "confidence": res.confidence, "code": res.code,
                      "detail": res.detail}, indent=2))
    return 0 if res.verified else 1


def cmd_sleep(brain: Brain, args) -> int:
    report = brain.sleep(resume_cycle=args.resume)
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def cmd_perceive(brain: Brain, args) -> int:
    brain.perceive(args.kind, args.content, importance=args.importance)
    print("perceived.")
    return 0


def cmd_episode(brain: Brain, args) -> int:
    ep = Episode(situation=args.situation or "manual", objective=args.objective,
                 actions=args.action or [], result=args.result or "",
                 status=args.status, mistakes=args.mistake or [],
                 lessons=args.lesson or [], confidence=args.confidence,
                 source="user")
    brain.memory.episodic.record(ep)
    print(json.dumps({"recorded": ep.id}, indent=2))
    return 0


def cmd_recall(brain: Brain, args) -> int:
    out = {
        "episodes": [e.__dict__ | {"actions": None} for e in
                     brain.memory.episodic.recall(args.query)],
        "concepts": [c.__dict__ | {"score": round(c.score, 3)} for c in
                     brain.memory.semantic.query(args.query)],
        "skills": [s.__dict__ for s in brain.memory.procedural.recall(args.query)],
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_eval(brain: Brain, args) -> int:
    from .evals.benchmarks import run_benchmark_suite, MINI_BENCH
    from .dashboard.server import _model_solver
    brain.ensure_model()
    result = run_benchmark_suite(brain.memory.store, args.label or "mini-bench",
                                 _model_solver(brain), brain.sandbox)
    print(json.dumps({"score": result["score"], "passed": result["passed"],
                      "total": result["total"]}, indent=2))
    return 0


def cmd_export(brain: Brain, args) -> int:
    print(json.dumps({"path": brain.memory.export_all(args.path)}, indent=2))
    return 0


def cmd_import(brain: Brain, args) -> int:
    print(json.dumps(brain.memory.import_all(args.path), indent=2))
    return 0


def cmd_dashboard(brain: Brain, args) -> int:
    from .dashboard.server import serve
    serve(brain, host="0.0.0.0", port=args.port)
    return 0


def cmd_chat(brain: Brain, args) -> int:
    """Interactive chat REPL routed through the whole cognitive stack."""
    brain.ensure_model()
    print("🧠 MINICODER chat — intents: 'rule: …' | 'remember: …' | "
          "'learn skill: name | python code' | 'solve <task-id>' | 'sleep' | /exit")
    print(f"   constitution rules loaded: {len(brain.rules())}")
    if args.oneshot:
        out = brain.chat(args.oneshot)
        print(out["answer"])
        return 0
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("/exit", "/quit", "exit", "quit"):
            break
        if not line:
            continue
        out = brain.chat(line)
        conf = int(round((out.get("confidence") or 0) * 100))
        print(f"brain[{out['kind']}, conf {conf}%]> {out['answer']}")
        if out.get("sources"):
            print("   sources:", " · ".join(out["sources"])[:160])
    return 0


def cmd_rule(brain: Brain, args) -> int:
    print(json.dumps(brain.add_rule(args.text), indent=2))
    return 0


def cmd_rules(brain: Brain, args) -> int:
    rules = brain.rules()
    print(json.dumps({"rules": rules, "count": len(rules)}, indent=2))
    return 0


def cmd_learn(brain: Brain, args) -> int:
    from .learning.continual import ContinualLearner
    from .evals.benchmarks import evaluate_solver, MINI_BENCH
    from .demo import _adapter_solver, _pure_model_solver
    brain.ensure_model()
    cl = ContinualLearner(brain.config, brain.memory, brain.sandbox,
                          brain.checkpoints)

    def bench(adapter=None):
        solver = (_pure_model_solver(brain) if adapter is None
                  else _adapter_solver(brain, adapter))
        return evaluate_solver(solver, brain.sandbox,
                               tasks=MINI_BENCH[: args.eval_subset])["score"]

    if args.level == 3:
        out = cl.level3_generate_examples(args.domain)
    elif args.level == 4:
        out = cl.level4_train_adapter(brain.model, brain.tokenizer, bench,
                                      name=args.name or "adapter")
    elif args.level == 5:
        out = cl.level5_update_core(brain.model, brain.tokenizer, bench,
                                    approved=args.approved, steps=args.steps)
    else:
        out = cl.level2_learn_skills()
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_demo(brain: Brain, args) -> int:
    from .demo import run_demo
    report = run_demo(brain, fast=args.fast)
    print(json.dumps(report["summary"], indent=2, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="minicoder")
    parser.add_argument("--data", default="data", help="brain data directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("train"); p.add_argument("--steps", type=int, default=None)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("solve"); p.add_argument("task_id")
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("sleep"); p.add_argument("--resume", default=None)
    p.set_defaults(fn=cmd_sleep)

    p = sub.add_parser("perceive"); p.add_argument("content")
    p.add_argument("--kind", default="note"); p.add_argument("--importance", type=float, default=0.5)
    p.set_defaults(fn=cmd_perceive)

    p = sub.add_parser("episode"); p.add_argument("objective")
    p.add_argument("--situation", default=""); p.add_argument("--action", action="append")
    p.add_argument("--result", default=""); p.add_argument("--status", default="unverified",
        choices=["success", "failure", "unverified"])
    p.add_argument("--mistake", action="append"); p.add_argument("--lesson", action="append")
    p.add_argument("--confidence", type=float, default=0.5); p.set_defaults(fn=cmd_episode)

    p = sub.add_parser("recall"); p.add_argument("query"); p.set_defaults(fn=cmd_recall)

    p = sub.add_parser("eval"); p.add_argument("--label", default=None)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("export"); p.add_argument("path", nargs="?", default=None)
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("import"); p.add_argument("path"); p.set_defaults(fn=cmd_import)

    p = sub.add_parser("learn"); p.add_argument("level", type=int, choices=[2, 3, 4, 5])
    p.add_argument("--domain", default="python-basics"); p.add_argument("--name", default=None)
    p.add_argument("--approved", action="store_true"); p.add_argument("--steps", type=int, default=40)
    p.add_argument("--eval-subset", type=int, default=6)
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("dashboard"); p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("chat"); p.add_argument("--say", dest="oneshot", default=None,
        help="single message instead of REPL")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("rule"); p.add_argument("text"); p.set_defaults(fn=cmd_rule)
    sub.add_parser("rules").set_defaults(fn=cmd_rules)

    p = sub.add_parser("demo"); p.add_argument("--fast", action="store_true")
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    brain = _brain(args)
    try:
        return args.fn(brain, args)
    finally:
        brain.close()


if __name__ == "__main__":
    sys.exit(main())
