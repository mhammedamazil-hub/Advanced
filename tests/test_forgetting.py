import numpy as np
import pytest

from minicoder.config import ModelConfig, TrainConfig
from minicoder.model.transformer import MiniGPT
from minicoder.model.trainer import Trainer
from minicoder.evals.memory_eval import forgetting_resistance


def _factory(m):
    cfg = TrainConfig(total_steps=10, warmup_steps=2, batch_size=4, seq_len=8,
                      lr=4e-3)
    return Trainer(m, cfg)


def _seqs(base, vocab=36, n=12, length=12):
    return [[(base + i + j) % vocab for j in range(length)] for i in range(n)]


@pytest.mark.slow
def test_forgetting_resistance_replay_helps():
    cfg = ModelConfig(vocab_size=36, d_model=24, n_layer=1, n_head=4,
                      n_kv_head=2, d_ff=32, max_seq=32)
    model = MiniGPT(cfg, seed=0)
    out = forgetting_resistance(model, _factory, _seqs(0), _seqs(20), steps=10)
    # EWC/replay should forget less than naive on this synthetic probe
    assert out["forgetting_replay"] <= out["forgetting_naive"] + 1e-6
