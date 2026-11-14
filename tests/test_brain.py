"""Integration test: whole-brain lifecycle on the real stack."""
import pytest

from minicoder.brain import Brain, AWAKE
from minicoder.config import BrainConfig
from minicoder.curriculum import executable_tasks
from minicoder.memory.episodic import SUCCESS


@pytest.fixture
def brain(tmp_path):
    cfg = BrainConfig(base_dir=str(tmp_path / "data"))
    cfg.train.total_steps = 40
    cfg.train.warmup_steps = 5
    cfg.train.seq_len = 48
    cfg.train.batch_size = 8
    cfg.sleep.new_experiences_trigger = 1000   # no auto-sleep surprises
    b = Brain(cfg)
    yield b
    b.close()


def test_persistence_across_sessions(tmp_path):
    cfg = BrainConfig(base_dir=str(tmp_path / "data"))
    b1 = Brain(cfg)
    b1.perceive("constraint", "always pin policies", pinned=True)
    b1.memory.semantic.upsert(
        __import__("minicoder.memory.semantic", fromlist=["Concept"]).Concept(
            name="session test marker", definition="persisted across restarts",
            confidence=0.9, status="fact"))
    b1.close()
    b2 = Brain(cfg)                              # new "session"
    assert any(i.content == "always pin policies" and i.pinned
               for i in b2.memory.working.items)
    hits = b2.memory.semantic.query("persisted across restarts")
    assert hits
    b2.close()


def test_solve_records_experience_and_verifies(brain):
    task = [t for t in executable_tasks(seed=0) if t.task_id.startswith("py-add")][0]
    brain.ensure_model()
    res = brain.solve(task)                       # likely model fails pre-training
    stats = brain.memory.episodic.stats()
    assert stats["count"] >= 1
    assert res.episode_id
    # now teach verified solution directly via experience, then re-solve
    brain.memory.episodic.record(
        __import__("minicoder.memory.episodic", fromlist=["Episode"]).Episode(
            situation="oracle", objective=task.prompt,
            actions=[f"python:{task.reference}"], result="tests passed",
            status=SUCCESS, confidence=0.95, salience=0.9, domain=task.domain))
    res2 = brain.solve(task)                      # should reuse verified memory
    assert res2.verified and res2.via == "memory-reuse"
    assert "add" in res2.code


def test_brain_sleep_cycle_changes_mind(brain):
    for i in range(3):
        brain.memory.episodic.record(
            __import__("minicoder.memory.episodic", fromlist=["Episode"]).Episode(
                situation="ci", objective=f"fix ci failure {i}",
                actions=["shell: pip install -r requirements.txt"],
                result="ok", status=SUCCESS,
                mistakes=["ModuleNotFoundError: x"],
                lessons=["install deps first"], confidence=0.9, salience=0.8,
                domain="debugging"))
    before = brain.memory.semantic.stats()["count"]
    report = brain.sleep()
    after = brain.memory.semantic.stats()["count"]
    assert report.status == "completed"
    assert after > before                          # knowledge grew during sleep
    assert brain.state == AWAKE
    assert brain.last_sleep_report["cycle_id"] == report.cycle_id


def test_status_schema(brain):
    s = brain.status()
    for key in ("state", "memory", "metacognition", "model", "sandbox",
                "checkpoints"):
        assert key in s
    assert {"working", "episodic", "semantic", "procedural"} <= set(s["memory"])
