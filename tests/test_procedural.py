from minicoder.config import BrainConfig
from minicoder.memory.system import MemorySystem
from minicoder.memory.procedural import Skill, UNTRUSTED, VERIFIED
from minicoder.memory.episodic import Episode, SUCCESS
from minicoder.tools.sandbox import Sandbox


def setup(tmp_path):
    mem = MemorySystem(BrainConfig(base_dir=str(tmp_path)))
    sb = Sandbox(mem.config.memory_path("sandbox"))
    return mem, sb


def add_skill():
    return Skill(
        name="add two numbers",
        description="compute a + b in python",
        triggers=["add", "sum"],
        steps=[{"action": "run_python", "args": {"code": "print(2 + 3)"}}],
        trust=UNTRUSTED)


def test_untrusted_skill_refused(tmp_path):
    mem, sb = setup(tmp_path)
    s = mem.procedural.learn(add_skill())
    res = mem.procedural.execute(s, sb)                    # no approval
    assert res["ok"] is False and "untrusted" in res["error"]
    assert res["executed"] == 0


def test_approved_execution_and_promotion(tmp_path):
    mem, sb = setup(tmp_path)
    s = mem.procedural.learn(add_skill())
    res = mem.procedural.execute(s, sb, approve_untrusted=True)
    assert res["ok"] is True and res["executed"] == 1
    assert mem.procedural.get(s.id).trust == VERIFIED      # promoted only after success
    assert mem.procedural.get(s.id).success_count == 1


def test_failing_step_recorded(tmp_path):
    mem, sb = setup(tmp_path)
    bad = Skill(name="bad skill", description="boom", triggers=[],
                steps=[{"action": "run_python", "args": {"code": "raise ValueError('x')"}}],
                trust=VERIFIED)
    mem.procedural.learn(bad)
    res = mem.procedural.execute(bad, sb)
    assert res["ok"] is False
    assert mem.procedural.get(mem.procedural.recall("bad skill")[0].id).failure_count == 1


def test_learn_from_verified_episode(tmp_path):
    mem, sb = setup(tmp_path)
    ep = Episode(situation="work", objective="reverse a string",
                 actions=["python:print('abc'[::-1])"], result="cba",
                 status=SUCCESS, confidence=0.9, domain="python")
    mem.episodic.record(ep)
    skill = mem.procedural.learn_from_episode(ep)
    assert skill is not None
    assert skill.trust == UNTRUSTED
    assert skill.provenance == f"episode:{ep.id}"
    res = mem.procedural.execute(skill, sb, approve_untrusted=True)
    assert res["ok"]                                    # replays verified action fine


def test_recall_by_trigger(tmp_path):
    mem, _ = setup(tmp_path)
    mem.procedural.learn(add_skill())
    hits = mem.procedural.recall("how to sum numbers")
    assert hits and hits[0].name == "add two numbers"
