"""Constitution rules: persistent, pinned, protected from forgetting."""
import time

from minicoder.brain import Brain
from minicoder.config import BrainConfig
from minicoder.memory import constitution
from minicoder.memory.system import MemorySystem
from minicoder.sleep.engine import SleepEngine
from minicoder.tools.sandbox import Sandbox


def make_brain(tmp_path):
    cfg = BrainConfig(base_dir=str(tmp_path / "data"))
    cfg.sleep.new_experiences_trigger = 10_000
    return Brain(cfg)


def test_rule_added_is_pinned_in_working(tmp_path):
    brain = make_brain(tmp_path)
    brain.add_rule("never run destructive shell commands")
    wm = brain.memory.working
    assert any(i.pinned and "never run destructive" in i.content
               for i in wm.items)
    rules = brain.rules()
    assert rules == ["never run destructive shell commands"]
    brain.close()


def test_rules_survive_restart(tmp_path):
    brain = make_brain(tmp_path)
    brain.add_rule("always verify code in the sandbox")
    brain.close()
    brain2 = make_brain(tmp_path)                    # new session, same data dir
    assert brain2.rules() == ["always verify code in the sandbox"]
    assert any(i.pinned and "RULE:" in i.content
               for i in brain2.memory.working.items)  # pinned at wake
    brain2.close()


def test_rules_protected_from_forgetting(tmp_path):
    brain = make_brain(tmp_path)
    brain.add_rule("my permanent rule about safety")
    mem = brain.memory
    # age the rule 100 days — forgetting must still never archive it
    rule = constitution.list_rules(mem)[0]
    mem.store.execute("UPDATE concepts SET updated=?, created=? WHERE id=?",
                      (time.time() - 100 * 24 * 3600,
                       time.time() - 100 * 24 * 3600, rule.id))
    mem.store.conn.commit()
    engine = SleepEngine(mem, brain.config,
                         Sandbox(brain.config.memory_path("sandbox")))
    engine._stage_replay()
    engine._stage_forgetting()
    still = mem.semantic.get_by_id(rule.id)
    assert still is not None and still.archived is False
    brain.close()


def test_rule_reinforcement_not_duplicated(tmp_path):
    brain = make_brain(tmp_path)
    brain.add_rule("prefer simple solutions")
    brain.add_rule("prefer simple solutions")            # same → reinforce
    assert len(brain.rules()) == 1
    brain.close()
