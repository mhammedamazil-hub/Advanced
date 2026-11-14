import numpy as np

from minicoder.config import ModelConfig
from minicoder.model.transformer import MiniGPT, param_count
from minicoder.model.tokenizer import ByteBPETokenizer


def micro(seed=0):
    cfg = ModelConfig(vocab_size=48, d_model=16, n_layer=2, n_head=4,
                      n_kv_head=2, d_ff=24, max_seq=16, dtype="float64")
    return MiniGPT(cfg, seed=seed)


def test_param_count():
    m = micro()
    n = param_count(m)
    assert n > 0
    # hand-computed expectation for micro(): te 48·16 + pe 16·16 + gf 16
    # per layer: g1 16 + Wq 256 + Wk 128 + Wv 128 + Wo 256 + g2 16
    #            + W1 16·24 + W2 24·16 = 1568
    assert n == 48 * 16 + 16 * 16 + 16 + 2 * 1568


def test_forward_shapes_and_causality():
    m = micro()
    x = np.random.randint(0, 48, size=(2, 10))
    logits, _ = m.forward(x)
    assert logits.shape == (2, 10, 48)
    # causality: changing a *later* token must not change earlier logits
    x2 = x.copy()
    x2[:, 7] = (x2[:, 7] + 1) % 48
    logits2, _ = m.forward(x2)
    assert np.allclose(logits[:, :7], logits2[:, :7])
    assert not np.allclose(logits[:, 8:], logits2[:, 8:])


def test_gradient_check_numerical():
    """Manual backprop must match finite differences (the real correctness gate)."""
    m = micro(seed=42)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 48, size=(2, 6))
    y = rng.integers(0, 48, size=(2, 6))
    loss, grads = m.loss_and_grads(x, y)
    assert np.isfinite(loss)
    checked = 0
    eps = 1e-5
    rng2 = np.random.default_rng(1)
    for name in ("te", "l0.Wq", "l0.Wk", "l0.Wo", "l1.W1", "l1.W2", "gf"):
        arr = m.params[name]
        for _ in range(4):
            idx = tuple(rng2.integers(0, s) for s in arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _ = m.loss_and_grads(x, y)
            arr[idx] = orig - eps
            lm, _ = m.loss_and_grads(x, y)
            arr[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name][idx]
            assert abs(num - ana) < 1e-5 + 1e-4 * abs(num), \
                f"{name}{idx}: numeric {num} vs analytic {ana}"
            checked += 1
    assert checked >= 28


def test_lora_forward_changes_output_and_grads_flow():
    from minicoder.model.lora import LoRAAdapter
    m = micro(seed=3)
    adapter = LoRAAdapter(m, rank=2, alpha=2.0, seed=0)
    # neutral at init (B = 0)
    x = np.random.randint(0, 48, size=(1, 5))
    y = np.random.randint(0, 48, size=(1, 5))
    l0, _ = m.loss_and_grads(x, y)
    l1, _ = m.loss_and_grads(x, y, adapter=adapter)
    assert abs(l0 - l1) < 1e-10                     # zero-init ⇒ neutral
    # perturb A/B and grads must target only adapter when adapter_only
    for name, (A, B, s) in adapter.lora_weights.items():
        adapter.lora_weights[name] = (A, B + 0.1, s)
    _, grads = m.loss_and_grads(x, y, adapter=adapter, adapter_only=True)
    assert grads
    assert all(k.startswith("lora:") for k in grads)
    assert any(np.abs(v).sum() > 0 for v in grads.values())


def test_generate_deterministic_when_greedy():
    m = micro(seed=7)
    ids = [5, 10, 15, 20]                      # within micro vocab (48)
    out1 = m.generate(ids, max_new=5, temperature=0.0, stop_id=47)
    out2 = m.generate(ids, max_new=5, temperature=0.0, stop_id=47)
    assert out1 == out2 and len(out1) <= 5


def test_save_load_roundtrip(tmp_path):
    m = micro(seed=5)
    path = str(tmp_path / "m.npz")
    m.save_with_manifest(path)
    m2 = MiniGPT.load(path)
    for k in m.params:
        assert np.array_equal(m.params[k], m2.params[k])
    assert m2.config.vocab_size == 48 and m2.config.n_layer == 2


def test_quantization_roundtrip_close():
    from minicoder.model.quantize import quantize_model, dequantize_model, size_report
    m = micro(seed=9)
    x = np.random.randint(0, 48, size=(1, 6))
    y = np.random.randint(0, 48, size=(1, 6))
    l0, _ = m.loss_and_grads(x, y)
    packed = quantize_model(m)
    m2 = micro(seed=9)
    dequantize_model(packed, m2)
    l1, _ = m2.loss_and_grads(x, y)
    # int8 round trip: small deviation, still finite
    assert np.isfinite(l1) and abs(l0 - l1) < 0.05
    rep = size_report(m)
    assert rep["int8_bytes"] < rep["fp32_bytes"]
