"""Skill studying during sleep: new skills become real training material
(level-4 gated LoRA), and verified studying marks skills as studied."""
import pytest

from minicoder.brain import Brain
from minicoder.config import BrainConfig, ModelConfig
from minicoder.curriculum import build_corpus
from minicoder.memory.procedural import Skill, VERIFIED


@pytest.fixture
def brain(tmp_path):
    cfg = BrainConfig(base_dir=str(tmp_path / "data"))
    cfg.sleep.new_experiences_trigger = 10_000
    cfg.sleep.skill_train_steps = 6          # tiny for the test
    cfg.sleep.skill_study_max = 4
    cfg.train.batch_size = 4
    cfg.train.seq_len = 32
    cfg.model = ModelConfig(vocab_size=260, d_model=24, n_layer=1, n_head=2,
                            n_kv_head=1, d_ff=32, max_seq=48)
    b = Brain(cfg)
    yield b
    b.close()


def _patch_tiny_model(brain):
    """Give the test brain a tiny model + tiny tokenizer so training is fast."""
    from minicoder.model.transformer import MiniGPT
    from minicoder.model.tokenizer import ByteBPETokenizer
    tok = ByteBPETokenizer()
    # tokenizer vocab must fit the tiny model vocab (64): no merges trained
    brain.tokenizer = tok
    brain.model = MiniGPT(brain.config.model, seed=0)


def test_sleep_studies_new_skills(brain, monkeypatch):
    _patch_tiny_model(brain)
    skill = Skill(
        name="compute average", description="average of a list of numbers",
        triggers=["average", "mean"],
        steps=[{"action": "run_python",
                "args": {"code": "def avg(xs):\n    return sum(xs) / len(xs)"}}],
        trust=VERIFIED)
    brain.memory.procedural.learn(skill)

    # fast gate: fake benchmark that stays neutral → adapter kept
    from minicoder.learning import continual as cl_mod
    orig = cl_mod.ContinualLearner.level4_train_adapter

    def fast_level4(self, model, tokenizer, benchmark, **kw):
        return orig(self, model, tokenizer,
                    lambda adapter=None: 0.5,          # neutral gate ⇒ kept
                    name=kw.get("name", "skill-study"),
                    texts=kw.get("texts"), steps=kw.get("steps"))
    monkeypatch.setattr(cl_mod.ContinualLearner, "level4_train_adapter",
                        fast_level4)

    report = brain.sleep()
    assert report.status == "completed"
    assert report.skills_studied == ["compute average"]
    stored = brain.memory.procedural.recall("average of a list")[0]
    assert stored.studied is True
    # an episode records the studying act
    assert any(e.domain == "meta-learning"
               for e in brain.memory.episodic.recent(3))
    stats = brain.memory.procedural.stats()
    assert stats["studied"] >= 1


def test_no_skills_no_study(brain):
    report = brain.sleep()
    assert report.skills_studied == []
