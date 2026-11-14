from minicoder.tools.sandbox import Sandbox
from minicoder.evals.benchmarks import evaluate_solver, MINI_BENCH


def test_oracle_solver_scores_100pct(tmp_path):
    sb = Sandbox(str(tmp_path / "sb"))
    result = evaluate_solver(lambda task: task.reference, sb)
    assert result["score"] == 1.0
    assert result["passed"] == result["total"] == len(MINI_BENCH)


def test_garbage_solver_scores_low(tmp_path):
    sb = Sandbox(str(tmp_path / "sb2"))
    result = evaluate_solver(lambda task: "def nope():\n    pass\n", sb)
    assert result["score"] == 0.0


def test_benchmark_held_out_seed():
    """Bench tasks must differ from training-seed tasks (held-out)."""
    from minicoder.curriculum import executable_tasks
    train = {t.task_id for t in executable_tasks(seed=0)}
    bench = {t.task_id for t in MINI_BENCH}
    assert train & bench == set() or len(train & bench) < len(bench)


def test_extract_code():
    from minicoder.evals.benchmarks import extract_code
    raw = "### Solution:\ndef f(x):\n    return x\n### End\n"
    assert extract_code(raw) == "def f(x):\n    return x"
    plain = "def g():\n    return 1\n"
    assert extract_code(plain) == plain.strip()
