from minicoder.config import BrainConfig
from minicoder.memory.system import MemorySystem
from minicoder.memory.semantic import Concept, FACT
from minicoder.metacog import MetaCognition


def make(tmp_path):
    mem = MemorySystem(BrainConfig(base_dir=str(tmp_path)))
    return mem, MetaCognition(mem)


def test_knowledge_support_measurable(tmp_path):
    mem, mc = make(tmp_path)
    mc.memory.semantic.upsert(Concept(
        name="binary search halves interval", status=FACT, confidence=0.9,
        definition="binary search halves the search interval each step"))
    before = mc.knowledge_support("quantum chromodynamics")
    after = mc.knowledge_support("binary search interval halving")
    assert after["support"] > before["support"]
    assert after["semantic_hits"] >= 1


def test_confidence_ranges_and_verification_effect(tmp_path):
    mem, mc = make(tmp_path)
    mc.begin_task("task")
    low = mc.estimate_confidence("unknown thing entirely", verified=False,
                                 had_tool_errors=True)
    high = mc.estimate_confidence("unknown thing entirely", verified=True)
    assert 0.0 <= low < high <= 1.0
    mc.end_task(resolved=True)


def test_strategy_policy(tmp_path):
    mem, mc = make(tmp_path)
    s1 = mc.select_strategy("brand new unknown topic zzz")
    assert s1["strategy"] in ("decompose-explore", "memory-guided")


def test_self_evaluation_gate(tmp_path):
    mem, mc = make(tmp_path)
    r_verified = mc.self_evaluate("obj", worked=True, verified=True)
    r_junk = mc.self_evaluate("obj", worked=False, verified=False, mistakes=[])
    assert r_verified.record_long_term and r_verified.salience > r_junk.salience
    r_fail = mc.self_evaluate("obj", worked=False, verified=False,
                              mistakes=["ImportError: foo"])
    assert r_fail.record_long_term                    # informative failures kept


def test_task_lifecycle(tmp_path):
    mem, mc = make(tmp_path)
    mc.begin_task("fix the bug")
    mc.find_gaps("fix the bug", required=["the traceback"])
    assert mc.current.gaps
    mc.end_task(resolved=True)
    rep = mc.report()
    assert rep["completed_tasks"] == 1 and rep["resolved"] == 1
