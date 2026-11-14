import time

from minicoder.config import BrainConfig
from minicoder.memory.system import MemorySystem
from minicoder.memory.episodic import Episode, SUCCESS, FAILURE, UNVERIFIED
from minicoder.memory.semantic import FACT
from minicoder.sleep.engine import SleepEngine
from minicoder.tools.sandbox import Sandbox


def make(tmp_path):
    cfg = BrainConfig(base_dir=str(tmp_path))
    cfg.memory.archive_epsilon = 0.05
    mem = MemorySystem(cfg)
    sb = Sandbox(cfg.memory_path("sandbox"))
    return cfg, mem, SleepEngine(mem, cfg, sb)


def seed_experiences(mem):
    # 3 failures with the same signature → pattern discovery
    for i in range(3):
        mem.episodic.record(Episode(
            situation="ci", objective=f"fix ci {i}",
            actions=["shell: pip install -r requirements.txt"],
            result="ModuleNotFoundError persisted", status=FAILURE,
            mistakes=["ModuleNotFoundError: requests"],
            lessons=["install dependencies in ci first"],
            confidence=0.6, salience=0.7, domain="debugging"))
    mem.episodic.record(Episode(
        situation="task", objective="write reverse string fn",
        actions=["python:def reverse_string(s): return s[::-1]"],
        result="tests passed", status=SUCCESS,
        lessons=["slicing s[::-1] reverses strings in python"],
        confidence=0.95, salience=0.8, domain="python-strings"))
    junk = Episode(situation="noise", objective="nothing", actions=[],
                   result="ok", status=UNVERIFIED, confidence=0.05, salience=0.02,
                   domain="general")
    junk.timestamp = time.time() - 40 * 24 * 3600          # very old
    mem.episodic.record(junk)
    return junk


def test_full_sleep_cycle(tmp_path):
    cfg, mem, engine = make(tmp_path)
    junk = seed_experiences(mem)
    pre_concepts = mem.semantic.stats()["count"]
    report = engine.run()
    assert report.status == "completed"
    assert set(report.stage_timings) == {
        "replay", "consolidation", "pattern_discovery", "organization",
        "forgetting", "learning", "wake"}
    # replay prioritized the informative episodes
    assert len(report.replayed_episodes) >= 4
    # consolidation wrote lesson concepts and marked episodes
    assert len(report.new_concepts) > 0
    assert mem.episodic.stats()["consolidated"] >= 4
    # a lesson became searchable knowledge
    hits = mem.semantic.query("reverses strings in python", k=3)
    assert hits
    # pattern discovery: 3 × ModuleNotFoundError in debugging
    assert any("modulenotfounderror" in p["name"].lower()
               for p in report.pattern_concepts)
    pattern = mem.semantic.get_by_name(
        report.pattern_concepts[0]["name"])
    assert "not proven causation" in pattern.definition    # causal care enforced
    # forgetting: the old junk episode got archived (recoverable)
    assert mem.episodic.get(junk.id).archived is True
    assert any(a["id"] == junk.id for a in report.archived)
    assert mem.episodic.recover_archive(junk.id)
    # verified knowledge survives
    assert mem.semantic.stats()["count"] > pre_concepts - 1
    # report persisted
    row = mem.store.execute("SELECT status FROM sleep_cycles WHERE id=?",
                            (report.cycle_id,)).fetchone()
    assert row["status"] == "completed"


def test_consolidated_lessons_become_facts_when_verified(tmp_path):
    cfg, mem, engine = make(tmp_path)
    mem.episodic.record(Episode(
        situation="t", objective="verified lesson source",
        actions=["python:pass"], result="ok", status=SUCCESS,
        lessons=["deep lesson about verified behavior"], confidence=0.95,
        salience=0.9, domain="python"))
    report = engine.run()
    assert report.status == "completed"
    hits = mem.semantic.query("deep lesson about verified behavior")
    assert hits and hits[0].status == FACT


def test_resume_interrupted_cycle(tmp_path):
    cfg, mem, engine = make(tmp_path)
    junk = seed_experiences(mem)

    class Boom(Exception):
        pass

    original = engine._stage_organization

    def crash():
        raise Boom("simulated crash mid-sleep")

    engine._stage_organization = crash                      # noqa: monkeypatch-ish
    report = engine.run()
    assert report.status == "interrupted"
    completed = engine._completed_stages(report.cycle_id)
    assert {1, 2, 3} <= completed                           # stages 1-3 persisted
    # resume with fixed stage-4
    engine2 = SleepEngine(mem, cfg, Sandbox(cfg.memory_path("sandbox")))
    report2 = engine2.run(resume_cycle=report.cycle_id)
    assert report2.status == "completed"
    assert report2.resumed_from == report.cycle_id
    # consolidation stage was NOT re-run (episodes remained marked)
    assert report2.consolidated_episodes == report.consolidated_episodes
