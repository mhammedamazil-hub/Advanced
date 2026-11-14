import numpy as np

from minicoder.config import ModelConfig, TrainConfig
from minicoder.model.transformer import MiniGPT
from minicoder.model.trainer import Trainer, CheckpointManager
from minicoder.model.lora import LoRAAdapter


def tiny_model():
    cfg = ModelConfig(vocab_size=40, d_model=24, n_layer=1, n_head=4,
                      n_kv_head=2, d_ff=32, max_seq=32)
    return MiniGPT(cfg, seed=0)


def tokens(seed=0, n=24, length=16, vocab=40):
    rng = np.random.default_rng(seed)
    # learnable pattern: incrementing token stream, not pure noise
    return [list((np.arange(length) + i) % vocab) for i in range(n)]


def test_training_reduces_loss():
    m = tiny_model()
    cfg = TrainConfig(total_steps=30, warmup_steps=3, batch_size=4, seq_len=8,
                      lr=5e-3)
    tr = Trainer(m, cfg)
    result = tr.fit(tokens(), steps=30)
    assert result["final_loss"] < result["start_loss"] - 0.05
    assert len(tr.history) == 30


def test_adapter_training_only_updates_lora(tmp_path):
    m = tiny_model()
    before = {k: v.copy() for k, v in m.params.items()}
    cfg = TrainConfig(total_steps=5, warmup_steps=1, batch_size=4, seq_len=8)
    tr = Trainer(m, cfg)
    adapter = LoRAAdapter(m, rank=2, alpha=2.0, seed=0)
    tr.fit(tokens(), steps=5, adapter=adapter, adapter_only=True)
    for k in m.params:
        assert np.array_equal(m.params[k], before[k])   # base frozen
    assert any(np.abs(B).sum() > 0 for A, B, _ in adapter.lora_weights.values())


def test_replay_and_ewc_run(tmp_path):
    m = tiny_model()
    cfg = TrainConfig(total_steps=6, warmup_steps=1, batch_size=4, seq_len=8)
    tr = Trainer(m, cfg)
    seq_a = tokens(seed=0)
    fisher = tr.fisher_estimate(seq_a, batches=2, batch=4, seq=8)
    assert fisher and all(np.all(v >= 0) for v in fisher.values())
    ewc = Trainer.ewc_state(m, fisher)
    out = tr.fit(tokens(seed=7), steps=6, replay_sequences=seq_a, ewc=ewc)
    assert np.isfinite(out["final_loss"])


def test_checkpoints_save_load_rollback(tmp_path):
    m = tiny_model()
    mgr = CheckpointManager(str(tmp_path / "ckpt"), keep=3)
    m.params["te"][0, 0] = 1.234
    mgr.save(m, "first", metrics={"loss": 1.0})
    m.params["te"][0, 0] = 5.678
    mgr.save(m, "second", metrics={"loss": 0.9})
    assert len(mgr.list()) == 2
    restored, meta = mgr.load()
    assert meta["name"] == "second"
    assert restored.params["te"][0, 0] == 5.678
    prev = mgr.rollback()
    assert prev["name"] == "first"
    restored, _ = mgr.load()
    assert restored.params["te"][0, 0] == 1.234
    # rollback refusing to drop the only checkpoint
    try:
        mgr.rollback()
        assert False, "should refuse"
    except RuntimeError:
        pass
