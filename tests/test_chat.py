"""Chatbot interface: intents route correctly, open questions get
memory-grounded answers, rules show up in replies."""
import pytest

from minicoder.brain import Brain
from minicoder.config import BrainConfig


@pytest.fixture
def brain(tmp_path):
    cfg = BrainConfig(base_dir=str(tmp_path / "data"))
    cfg.sleep.new_experiences_trigger = 10_000
    b = Brain(cfg)
    yield b
    b.close()


def test_rule_intent(brain):
    out = brain.chat("rule: never lie about verification status")
    assert out["kind"] == "rule-added" and out["confidence"] == 1.0
    assert "never lie" in brain.rules()[0]


def test_remember_intent(brain):
    out = brain.chat("remember: prod db runs postgres 15 on port 5433")
    assert out["kind"] == "remembered"
    hits = brain.memory.episodic.recall("postgres 15 port 5433")
    assert hits and "postgres" in hits[0].objective


def test_learn_skill_intent(brain):
    out = brain.chat("learn skill: double numbers | "
                     "def double(x):\n    return x * 2")
    assert out["kind"] == "skill-added"
    skills = brain.memory.procedural.recall("double numbers")
    assert skills and skills[0].steps[0]["action"] == "run_python"
    assert skills[0].studied is False                    # not yet — sleep does it


def test_unknown_task_solve(brain):
    out = brain.chat("solve does-not-exist-42")
    assert out["kind"] == "error"


def test_open_question_uses_memory(brain):
    brain.add_rule("cite memory whenever possible")
    brain.chat("remember: binary search halves the interval each step")
    out = brain.chat("binary search")
    assert out["kind"] == "answer"
    assert isinstance(out["answer"], str)
    assert "sources" in out
    # the chat itself was recorded as an unverified episode
    assert any(e.domain == "chat" for e in
               brain.memory.episodic.recent(5))


def test_chat_sleep_intent(brain):
    out = brain.chat("sleep")
    assert out["kind"] == "sleep"
    import time
    time.sleep(0.5)
    while brain.state == "sleeping":
        time.sleep(0.1)
    assert brain.state == "awake"
