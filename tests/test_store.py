import os

from minicoder.config import BrainConfig
from minicoder.memory.store import Store, atomic_write_json


def make_store(tmp_path) -> Store:
    return Store(str(tmp_path / "brain.db"))


def test_schema_created(tmp_path):
    st = make_store(tmp_path)
    tables = {r["name"] for r in st.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"episodes", "concepts", "edges", "skills", "sleep_cycles",
            "sleep_checkpoints", "learn_log", "benchmarks",
            "contradictions", "archive", "meta"} <= tables
    assert st.get_meta("schema_version") == "1"
    st.close()


def test_persistence_across_reopen(tmp_path):
    st = make_store(tmp_path)
    st.set_meta("hello", {"a": 1})
    st.close()
    st2 = make_store(tmp_path)
    assert st2.get_meta("hello") == {"a": 1}
    st2.close()


def test_backup_rotation_and_recovery(tmp_path):
    st = make_store(tmp_path)
    st.set_meta("v", 1)
    b1 = st.backup()
    assert os.path.exists(b1)
    st.set_meta("v", 2)
    b2 = st.backup()
    assert os.path.exists(b2)
    # simulate catastrophic corruption of the main file
    with open(st.path, "wb") as fh:
        fh.write(os.urandom(2048))
    st.close()
    st3 = make_store(tmp_path)          # must recover from newest healthy backup
    assert st3.get_meta("v") in (1, 2)  # recovered, not empty
    assert st3._healthy(st3.conn)
    st3.close()


def test_transaction_rollback(tmp_path):
    st = make_store(tmp_path)
    try:
        with st.transaction() as cur:
            cur.execute("INSERT INTO episodes(id, situation, objective, actions, "
                        "result, status, mistakes, lessons, timestamp, source, "
                        "confidence, salience, created, last_accessed) VALUES "
                        "('x','a','b','[]','r','success','[]','[]',0,'t',0.5,0.5,0,0)")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    row = st.execute("SELECT COUNT(*) c FROM episodes").fetchone()
    assert row["c"] == 0                    # rolled back atomically
    st.close()


def test_atomic_write_json_no_partials(tmp_path):
    path = str(tmp_path / "out.json")
    atomic_write_json(path, {"a": 1})
    assert os.path.exists(path)
    import json
    assert json.load(open(path)) == {"a": 1}
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
    assert leftovers == []
