import time

from minicoder.config import MemoryConfig
from minicoder.memory.working import WorkingMemory


def make_wm(cap=4, ttl=100.0):
    return WorkingMemory(MemoryConfig(working_capacity=cap, working_ttl_seconds=ttl))


def test_capacity_eviction_prefers_low_priority():
    wm = make_wm(cap=3)
    wm.add("note", "critical pin", importance=1.0, pinned=True)
    wm.add("note", "unimportant old", importance=0.05)
    time.sleep(0.01)
    wm.add("note", "moderate value item", importance=0.5)
    wm.add("note", "high value recent", importance=0.9)
    contents = [i.content for i in wm.items]
    assert len(wm.items) == 3
    assert "critical pin" in contents
    assert "unimportant old" not in contents        # evicted (lowest priority)
    assert "high value recent" in contents


def test_pinned_never_expires():
    wm = make_wm(ttl=0.001)
    wm.add("task", "pinned task", pinned=True)
    time.sleep(0.01)
    wm.add("note", "temp")
    stats = wm.stats()
    assert any(i.pinned for i in wm.items)
    assert stats["items"] >= 1


def test_ttl_expiration():
    wm = make_wm(ttl=0.02)
    wm.add("note", "short lived")
    time.sleep(0.05)
    wm.add("note", "trigger expire")
    assert all(i.content != "short lived" for i in wm.items)


def test_dedup_refresh():
    wm = make_wm()
    a = wm.add("note", "same thing", importance=0.3)
    time.sleep(0.005)
    b = wm.add("note", "same thing", importance=0.6)
    assert a.id == b.id
    assert len(wm.items) == 1
    assert b.importance == 0.6


def test_compress_merges_low_priority():
    wm = make_wm(cap=32)
    for i in range(6):
        wm.add("note", f"low note {i}", importance=0.1)
    wm.add("task", "important", importance=0.9, pinned=True)
    merged = wm.compress(max_chars=200)
    assert merged is not None
    assert "compressed" in merged.content
    assert any(i.pinned for i in wm.items)          # protected item kept


def test_context_orders_by_priority():
    wm = make_wm()
    wm.add("note", "the python import error context", importance=0.9)
    wm.add("note", "unrelated lunch note", importance=0.1)
    ctx = wm.context(focus="python import error", max_chars=1000)
    assert ctx.splitlines()[0].startswith("[note] the python import error context")


def test_snapshot_roundtrip(tmp_path):
    path = str(tmp_path / "wm.json")
    wm = WorkingMemory(MemoryConfig(), snapshot_path=path)
    wm.add("note", "persist me", importance=0.7)
    wm2 = WorkingMemory(MemoryConfig(), snapshot_path=path)
    assert any(i.content == "persist me" for i in wm2.items)
