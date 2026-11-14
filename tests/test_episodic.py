from minicoder.config import BrainConfig
from minicoder.memory.system import MemorySystem
from minicoder.memory.episodic import Episode, SUCCESS, FAILURE, UNVERIFIED


def make_mem(tmp_path) -> MemorySystem:
    return MemorySystem(BrainConfig(base_dir=str(tmp_path)))


def test_record_and_recall_ranking(tmp_path):
    mem = make_mem(tmp_path)
    mem.episodic.record(Episode(
        situation="repo work", objective="fix python import error",
        actions=["shell: pip install foo"], result="works now", status=SUCCESS,
        lessons=["imports need installs"], confidence=0.9, salience=0.8,
        domain="python"))
    mem.episodic.record(Episode(
        situation="lunch", objective="order sushi",
        actions=[], result="done", status=SUCCESS, confidence=0.5))
    hits = mem.episodic.recall("python import error")
    assert hits[0].objective == "fix python import error"


def test_status_distinction(tmp_path):
    mem = make_mem(tmp_path)
    for status in (SUCCESS, FAILURE, UNVERIFIED):
        mem.episodic.record(Episode(
            situation="s", objective=f"task-{status}", actions=[], result="r",
            status=status))
    stats = mem.episodic.stats()
    assert stats["success"] == 1 and stats["failure"] == 1 and stats["unverified"] == 1
    epi = mem.episodic.recall("task-success")[0]
    assert epi.verified is True


def test_archive_and_recover(tmp_path):
    mem = make_mem(tmp_path)
    ep = mem.episodic.record(Episode(
        situation="s", objective="old unverified thing", actions=[], result="ok",
        status=UNVERIFIED, confidence=0.1, salience=0.05))
    mem.episodic.archive(ep.id, "test", "compressed…")
    assert mem.episodic.get(ep.id).archived is True
    assert mem.episodic.recover_archive(ep.id) is True
    assert mem.episodic.get(ep.id).archived is False


def test_mark_consolidated(tmp_path):
    mem = make_mem(tmp_path)
    ep = mem.episodic.record(Episode(situation="s", objective="o", actions=[],
                                     result="r"))
    mem.episodic.mark_consolidated([ep.id])
    assert mem.episodic.get(ep.id).consolidated is True
    assert mem.episodic.recent(5, unconsolidated_only=True) == []


def test_export_import_roundtrip(tmp_path):
    mem = make_mem(tmp_path)
    mem.episodic.record(Episode(situation="s", objective="carry me across",
                                actions=["think: x"], result="ok",
                                status=SUCCESS, confidence=0.8))
    out = mem.export_all(str(tmp_path / "exp.json"))
    mem2 = MemorySystem(BrainConfig(base_dir=str(tmp_path / "second")))
    counts = mem2.import_all(out)
    assert counts["episodes"] == 1
    assert mem2.episodic.recall("carry me across")
    # re-import must not duplicate
    counts2 = mem2.import_all(out)
    assert counts2["episodes"] == 0
