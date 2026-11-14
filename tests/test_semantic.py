from minicoder.config import BrainConfig
from minicoder.memory.system import MemorySystem
from minicoder.memory.semantic import Concept, FACT, HYPOTHESIS


def make_mem(tmp_path) -> MemorySystem:
    return MemorySystem(BrainConfig(base_dir=str(tmp_path)))


def test_upsert_accumulates_support_and_promotes(tmp_path):
    mem = make_mem(tmp_path)
    c = Concept(name="python decorators", definition="wrap functions",
                confidence=0.4, status=HYPOTHESIS, provenance=["docs"])
    mem.semantic.upsert(c)
    c2 = Concept(name="Python Decorators", definition="wrap functions with @ syntax",
                 confidence=0.9, status=FACT, provenance=["verified"])
    mem.semantic.upsert(c2)
    got = mem.semantic.get_by_name("python decorators")
    assert got.support == 2
    assert got.status == FACT                       # promoted, never downgraded
    assert sorted(got.provenance) == ["docs", "verified"]


def test_conflicting_definitions_recorded_not_overwritten(tmp_path):
    mem = make_mem(tmp_path)
    mem.semantic.upsert(Concept(name="default db port", definition="5432",
                                confidence=0.9, status=FACT, provenance=["config-a"]))
    mem.semantic.upsert(Concept(name="default db port", definition="9999 totally",
                                confidence=0.3, status=HYPOTHESIS,
                                provenance=["untrusted-note"]))
    got = mem.semantic.get_by_name("default db port")
    assert got.definition == "5432"                  # high-confidence version kept
    assert mem.semantic.contradictions()


def test_knowledge_graph_activation(tmp_path):
    mem = make_mem(tmp_path)
    a = mem.semantic.upsert(Concept(name="binary search", definition="halves interval",
                                    domain="algorithms", confidence=0.9, status=FACT))
    b = mem.semantic.upsert(Concept(name="log n complexity",
                                    definition="logarithmic growth rate",
                                    domain="algorithms", confidence=0.9, status=FACT))
    mem.semantic.link(a.id, b.id, "supports")
    hits = mem.semantic.query("halves interval search", k=5)
    names = {c.name for c in hits}
    assert "binary search" in names
    assert "log n complexity" in names               # reached via spreading activation
    via = {c.name: c.via for c in hits}
    assert via.get("log n complexity") == ["supports"]


def test_merge_duplicates(tmp_path):
    mem = make_mem(tmp_path)
    a = mem.semantic.upsert(Concept(name="rest api design",
                                    definition="use nouns for resources and verbs via http methods",
                                    confidence=0.8))
    b = mem.semantic.upsert(Concept(name="rest api naming",
                                    definition="use nouns for resources and verbs via http methods",
                                    confidence=0.5))
    pairs = mem.semantic.find_near_duplicates(threshold=0.6)
    assert any({p[0], p[1]} == {a.id, b.id} for p in pairs)
    assert mem.semantic.merge(a.id, b.id)
    assert mem.semantic.get_by_id(b.id) is None
    assert mem.semantic.get_by_id(a.id).support == 2   # supports summed
    assert any(p.startswith("merged:") for p in mem.semantic.get_by_id(a.id).provenance)


def test_query_excludes_hypotheses_when_requested(tmp_path):
    mem = make_mem(tmp_path)
    mem.semantic.upsert(Concept(name="hypothesis thing unique word xyzabc",
                                definition="unproven guess xyzabc", status=HYPOTHESIS,
                                confidence=0.3))
    mem.semantic.upsert(Concept(name="fact thing unique word xyzabc",
                                definition="proven fact xyzabc", status=FACT,
                                confidence=0.9))
    hits = mem.semantic.query("xyzabc", k=5, include_hypotheses=False)
    assert all(c.status == FACT for c in hits)
