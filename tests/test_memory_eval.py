from minicoder.config import BrainConfig
from minicoder.memory.system import MemorySystem
from minicoder.evals import memory_eval


def test_retrieval_precision(tmp_path):
    mem = MemorySystem(BrainConfig(base_dir=str(tmp_path)))
    res = memory_eval.retrieval_precision(mem, probes=6)
    assert res["hit@3"] >= 0.8                      # retrieval works on paraphrases


def test_retention_decay_rules(tmp_path):
    mem = MemorySystem(BrainConfig(base_dir=str(tmp_path)))
    res = memory_eval.retention_after_decay(mem)
    assert res["important_survived"] is True
    assert res["junk_archived"] is True
    assert res["archive_recoverable"] is True


def test_consolidation_quality_improves_retrieval(tmp_path):
    mem = MemorySystem(BrainConfig(base_dir=str(tmp_path)))
    res = memory_eval.consolidation_quality(mem)
    assert res["report_status"] == "completed"
    assert res["patterns_found"] >= 1
    assert res["concepts_after"] > res["concepts_before"]
    assert res["retrieval_best_after"] >= res["retrieval_best_before"]
