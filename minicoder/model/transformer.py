"""MiniGPT — a compact code language model in pure NumPy.

Architecture features (per spec §6, all *implemented* and testable):
* efficient Transformer, pre-norm residual stream, RMSNorm
* Grouped Query Attention (n_kv_head < n_head → KV heads shared)
* weight tying (input embedding == output projection) when configured
* GELU feed-forward networks
* optional LoRA adapters at any projection (parameter-efficient fine-tuning)
* int8 quantization via ``quantize.py``

The default configuration is ~0.32M parameters: small enough to *genuinely
train* on this CPU-only 2-core sandbox for the proof-of-learning demo.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np

from ..config import ModelConfig


# ---------------------------------------------------------------------------
# primitive ops with manual gradients
# ---------------------------------------------------------------------------

def _rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-6):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    r = 1.0 / np.sqrt(ms + eps)
    return x * r * g, (r, x)


def _rmsnorm_backward(dy: np.ndarray, g: np.ndarray, r: np.ndarray,
                      x: np.ndarray):
    reduce_axes = tuple(range(dy.ndim - 1))
    dg = np.sum(dy * x * r, axis=reduce_axes)
    dot = np.sum(dy * g * x, axis=-1, keepdims=True)
    dx = g * r * dy - (r ** 3) * x * dot / x.shape[-1]
    return dx, dg


_GELU_C = math.sqrt(2.0 / math.pi)


def _gelu(x: np.ndarray) -> np.ndarray:
    u = _GELU_C * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(u))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    u = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _GELU_C * (1.0 + 3.0 * 0.044715 * x * x)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def param_count(model: "MiniGPT") -> int:
    return int(sum(p.size for p in model.params.values()))


class MiniGPT:
    def __init__(self, config: ModelConfig, seed: int = 0):
        self.config = config
        rng = np.random.default_rng(seed)
        c = config
        assert c.n_head % c.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.head_dim = c.d_model // c.n_head
        self.kv_dim = self.head_dim * c.n_kv_head
        self.group = c.n_head // c.n_kv_head
        std = 0.02
        self.params: dict[str, np.ndarray] = {
            "te": (rng.standard_normal((c.vocab_size, c.d_model)) * std).astype(c.dtype),
            "pe": (rng.standard_normal((c.max_seq, c.d_model)) * std).astype(c.dtype),
            "gf": np.ones(c.d_model, dtype=c.dtype),
        }
        if not c.weight_tying:
            self.params["lm_head"] = (rng.standard_normal((c.d_model, c.vocab_size)) * std).astype(c.dtype)
        for i in range(c.n_layer):
            p = f"l{i}."
            self.params[p + "g1"] = np.ones(c.d_model, dtype=c.dtype)
            self.params[p + "Wq"] = (rng.standard_normal((c.d_model, c.d_model)) * std).astype(c.dtype)
            self.params[p + "Wk"] = (rng.standard_normal((c.d_model, self.kv_dim)) * std).astype(c.dtype)
            self.params[p + "Wv"] = (rng.standard_normal((c.d_model, self.kv_dim)) * std).astype(c.dtype)
            self.params[p + "Wo"] = (rng.standard_normal((c.d_model, c.d_model)) * std / math.sqrt(2 * c.n_layer)).astype(c.dtype)
            self.params[p + "g2"] = np.ones(c.d_model, dtype=c.dtype)
            self.params[p + "W1"] = (rng.standard_normal((c.d_model, c.d_ff)) * std).astype(c.dtype)
            self.params[p + "W2"] = (rng.standard_normal((c.d_ff, c.d_model)) * std / math.sqrt(2 * c.n_layer)).astype(c.dtype)

    # ------------------------------------------------------------ weights
    def _eff(self, name: str, adapter: Optional[dict]) -> np.ndarray:
        w = self.params[name]
        if adapter and name in adapter.lora_weights:
            A, B, scale = adapter.lora_weights[name]
            return w + scale * (B @ A)
        return w

    # ------------------------------------------------------------ forward
    def forward(self, tokens: np.ndarray, adapter=None):
        """tokens: (B, T) int64 → logits (B, T, V), cache for backward."""
        c = self.config
        B, T = tokens.shape
        tok_emb = self.params["te"][tokens]                    # (B,T,d)
        pos_emb = self.params["pe"][:T]                        # (T,d)
        X = tok_emb + pos_emb
        caches: list[dict] = []
        for i in range(c.n_layer):
            p = f"l{i}."
            xn, norm1 = _rmsnorm(X, self.params[p + "g1"])
            attn_out, attn_cache = self._attention(xn, p, adapter,
                                                   causal_offset=0)
            Y = X + attn_out
            xn2, norm2 = _rmsnorm(Y, self.params[p + "g2"])
            W1, W2 = self._eff(p + "W1", adapter), self._eff(p + "W2", adapter)
            pre = xn2 @ W1
            act = _gelu(pre)
            ffn_out = act @ W2
            X = Y + ffn_out
            caches.append({"norm1": norm1, "attn": attn_cache, "norm2": norm2,
                           "xn2": xn2, "pre": pre, "act": act, "X_in": X - ffn_out - attn_out})
        xf, normf = _rmsnorm(X, self.params["gf"])
        xf2 = xf.reshape(B * T, c.d_model)
        if c.weight_tying:
            Wout = self.params["te"].T
        else:
            Wout = self.params["lm_head"]
        logits = xf2 @ Wout                                    # (B*T, V)
        cache = {"tokens": tokens, "pos_emb": pos_emb, "layers": caches,
                 "normf": normf, "xf": xf, "B": B, "T": T,
                 "final_input": X}
        return logits.reshape(B, T, c.vocab_size), cache

    def _attention(self, Xn: np.ndarray, prefix: str, adapter,
                   causal_offset: int = 0):
        c = self.config
        B, T, d = Xn.shape
        H, KVH, hd = c.n_head, c.n_kv_head, self.head_dim
        Wq = self._eff(prefix + "Wq", adapter)
        Wk = self._eff(prefix + "Wk", adapter)
        Wv = self._eff(prefix + "Wv", adapter)
        Wo = self._eff(prefix + "Wo", adapter)
        Xn2 = Xn.reshape(B * T, d)
        Q = (Xn2 @ Wq).reshape(B, T, H, hd).transpose(0, 2, 1, 3)       # (B,H,T,hd)
        K = (Xn2 @ Wk).reshape(B, T, KVH, hd).transpose(0, 2, 1, 3)     # (B,KVH,T,hd)
        V = (Xn2 @ Wv).reshape(B, T, KVH, hd).transpose(0, 2, 1, 3)
        Ke = np.repeat(K, self.group, axis=1)                            # (B,H,T,hd)
        Ve = np.repeat(V, self.group, axis=1)
        scale = 1.0 / math.sqrt(hd)
        S = (Q @ Ke.transpose(0, 1, 3, 2)) * scale                       # (B,H,T,T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        S = np.where(mask, -1e18, S)
        P = _softmax_rows(S)                                             # (B,H,T,T)
        A = P @ Ve                                                       # (B,H,T,hd)
        A2 = A.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
        out = A2.reshape(B * T, H * hd) @ Wo
        out = out.reshape(B, T, d)
        cache = {"Xn": Xn, "Q": Q, "Ke": Ke, "Ve": Ve, "P": P,
                 "A2": A2, "scale": scale, "prefix": prefix}
        return out, cache

    # ------------------------------------------------------------ loss+grads
    def loss_and_grads(self, x: np.ndarray, y: np.ndarray, adapter=None,
                       trainable: Optional[set[str]] = None,
                       adapter_only: bool = False):
        """x,y: (B,T) int64 (y is next-token).  Returns (loss, grads).

        ``trainable`` limits which base-parameter names receive gradients;
        ``adapter_only`` additionally returns gradients for LoRA A/B.
        """
        c = self.config
        B, T = x.shape
        logits, cache = self.forward(x, adapter=adapter)
        V = c.vocab_size
        logits2 = logits.reshape(B * T, V)
        targets = y.reshape(B * T)
        lsm = logits2 - logits2.max(axis=-1, keepdims=True)
        lsm = lsm - np.log(np.exp(lsm).sum(axis=-1, keepdims=True))
        rows = np.arange(B * T)
        loss_per = -lsm[rows, targets]
        loss = float(loss_per.mean())
        dlogits = np.exp(lsm)
        dlogits[rows, targets] -= 1.0
        dlogits /= (B * T)

        grads: dict[str, np.ndarray] = {}
        xf = cache["xf"].reshape(B * T, c.d_model)
        if c.weight_tying:
            dWout = "te"
            grads[dWout] = dlogits.T @ xf                        # embed side added later
            dxf2 = dlogits @ self.params[dWout]
        else:
            grads["lm_head"] = dlogits.T @ xf
            dxf2 = dlogits @ self.params["lm_head"]
        dX, dgf = _rmsnorm_backward(dxf2.reshape(B, T, c.d_model),
                                    self.params["gf"],
                                    cache["normf"][0], cache["normf"][1])
        grads["gf"] = dgf
        # backward through layers
        for i in reversed(range(c.n_layer)):
            p = f"l{i}."
            lc = cache["layers"][i]
            # ----- FFN branch -----
            dY = dX
            xn2, pre, act = lc["xn2"], lc["pre"], lc["act"]
            W1 = self._eff(p + "W1", adapter)
            W2 = self._eff(p + "W2", adapter)
            Bp, Tp, ff = act.shape
            dW2 = act.reshape(Bp * Tp, ff).T @ dY.reshape(Bp * Tp, -1)
            dact = (dY.reshape(Bp * Tp, -1) @ W2.T).reshape(Bp, Tp, ff)
            dpre = dact * _gelu_grad(pre)
            xn2_2d = xn2.reshape(Bp * Tp, -1)
            dW1 = xn2_2d.T @ dpre.reshape(Bp * Tp, ff)
            dxn2 = (dpre.reshape(Bp * Tp, ff) @ W1.T).reshape(Bp, Tp, -1)
            self._route_weight_grad(p + "W1", dW1, grads, adapter,
                                    trainable, adapter_only)
            self._route_weight_grad(p + "W2", dW2, grads, adapter,
                                    trainable, adapter_only)
            dY2, dg2 = _rmsnorm_backward(dxn2, self.params[p + "g2"],
                                         lc["norm2"][0], lc["norm2"][1])
            grads[p + "g2"] = dg2
            dX = dY + dY2        # residual merge
            # ----- attention branch -----
            dattn = dX
            atc = lc["attn"]
            dXn, aw = self._attention_backward(dattn, atc, p, adapter)
            for name, g in aw.items():
                self._route_weight_grad(name, g, grads, adapter,
                                        trainable, adapter_only)
            dXatt, dg1 = _rmsnorm_backward(dXn, self.params[p + "g1"],
                                           lc["norm1"][0], lc["norm1"][1])
            grads[p + "g1"] = dg1
            dX = dX + dXatt      # residual merge
        # input embeddings
        grads_te_in = np.zeros_like(self.params["te"])
        np.add.at(grads_te_in, cache["tokens"].reshape(-1),
                  dX.reshape(B * T, c.d_model))
        if "te" in grads:
            grads["te"] = grads["te"] + grads_te_in
        else:
            grads["te"] = grads_te_in
        grads["pe"] = np.zeros_like(self.params["pe"])
        grads["pe"][:T] = dX.sum(axis=0)
        if adapter_only:
            grads = {k: v for k, v in grads.items() if k.startswith("lora:")}
        elif trainable is not None:
            grads = {k: v for k, v in grads.items()
                     if k in trainable or k.startswith("lora:")}
        return loss, grads

    def _route_weight_grad(self, name: str, dW: np.ndarray, grads: dict,
                           adapter, trainable, adapter_only: bool) -> None:
        if adapter is not None and name in adapter.lora_weights:
            A, B, scale = adapter.lora_weights[name]
            grads["lora:B:" + name] = scale * (dW @ A.T)
            grads["lora:A:" + name] = scale * (B.T @ dW)
            if adapter_only:
                return
        if trainable is None or name in trainable:
            if adapter_only:
                return
            grads[name] = dW

    def _attention_backward(self, dO: np.ndarray, atc: dict, prefix: str,
                            adapter):
        c = self.config
        B, T, d = dO.shape
        H, KVH, hd = c.n_head, c.n_kv_head, self.head_dim
        Wo = self._eff(prefix + "Wo", adapter)
        A2 = atc["A2"]                                              # (B,T,d)
        dWo = A2.reshape(B * T, d).T @ dO.reshape(B * T, d)
        dA2 = (dO.reshape(B * T, d) @ Wo.T).reshape(B, T, H, hd)
        dA = dA2.transpose(0, 2, 1, 3)                              # (B,H,T,hd)
        P, Ve, Ke, Q = atc["P"], atc["Ve"], atc["Ke"], atc["Q"]
        dVe = P.transpose(0, 1, 3, 2) @ dA
        dP = dA @ Ve.transpose(0, 1, 3, 2)
        dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
        dS = dS * atc["scale"]
        dQ = dS @ Ke                                                # (B,H,T,hd)
        dKe = dS.transpose(0, 1, 3, 2) @ Q                          # (B,H,T,hd)
        dK = dKe.reshape(B, KVH, self.group, T, hd).sum(axis=2)     # (B,KVH,T,hd)
        dV = dVe.reshape(B, KVH, self.group, T, hd).sum(axis=2)
        dQ2 = dQ.transpose(0, 2, 1, 3).reshape(B * T, d)
        dK2 = dK.transpose(0, 2, 1, 3).reshape(B * T, KVH * hd)
        dV2 = dV.transpose(0, 2, 1, 3).reshape(B * T, KVH * hd)
        Xn = atc["Xn"]
        Xn2 = Xn.reshape(B * T, d)
        Wq = self._eff(prefix + "Wq", adapter)
        Wk = self._eff(prefix + "Wk", adapter)
        Wv = self._eff(prefix + "Wv", adapter)
        grads = {
            prefix + "Wo": dWo,
            prefix + "Wq": Xn2.T @ dQ2,
            prefix + "Wk": Xn2.T @ dK2,
            prefix + "Wv": Xn2.T @ dV2,
        }
        dXn = dQ2 @ Wq.T + dK2 @ Wk.T + dV2 @ Wv.T
        return dXn.reshape(B, T, d), grads

    # ------------------------------------------------------------ generate
    def forward_last_logits(self, tokens: list[int], adapter=None) -> np.ndarray:
        arr = np.array(tokens[-self.config.max_seq:], dtype=np.int64)[None, :]
        logits, _ = self.forward(arr, adapter=adapter)
        return logits[0, -1]

    def generate(self, prompt_ids: list[int], max_new: int = 64,
                 temperature: float = 0.8, top_k: int = 24,
                 stop_id: int = 258, adapter=None,
                 rng: Optional[np.random.Generator] = None) -> list[int]:
        rng = rng or np.random.default_rng()
        ids = list(prompt_ids)
        for _ in range(max_new):
            logits = self.forward_last_logits(ids, adapter=adapter)
            if temperature <= 0:
                nxt = int(np.argmax(logits))
            else:
                logits = logits / temperature
                if top_k > 0:
                    idx = np.argpartition(logits, -top_k)[-top_k:]
                    mask = np.full_like(logits, -1e18)
                    mask[idx] = logits[idx]
                    logits = mask
                probs = _softmax_rows(logits[None, :])[0]
                probs = probs / probs.sum()
                nxt = int(rng.choice(len(probs), p=probs))
            ids.append(nxt)
            if nxt == stop_id:
                break
        return ids[len(prompt_ids):]

    # ------------------------------------------------------------ storage
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **self.params)

    @classmethod
    def load(cls, path: str, config: Optional[ModelConfig] = None) -> "MiniGPT":
        data = np.load(path, allow_pickle=False)
        if config is None:
            manifest = path + ".manifest.json"
            if os.path.exists(manifest):
                with open(manifest, "r", encoding="utf-8") as fh:
                    config = ModelConfig(**json.load(fh)["config"])
            else:
                te = data["te"]
                config = ModelConfig(
                    vocab_size=int(te.shape[0]), d_model=int(te.shape[1]),
                    n_layer=sum(1 for k in data.files if k.endswith("g1")),
                    weight_tying="lm_head" not in data)
        model = cls(config)
        for k in data.files:
            model.params[k] = data[k]
        return model

    def save_with_manifest(self, path: str, extra: Optional[dict] = None) -> None:
        self.save(path)
        manifest = {"config": self.config.__dict__, "params": param_count(self)}
        if extra:
            manifest.update(extra)
        with open(path + ".manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
