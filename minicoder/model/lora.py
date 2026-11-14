"""LoRA adapters — parameter-efficient fine-tuning (spec §5, level 4).

An adapter holds low-rank A/B matrices for a chosen set of projection weights.
While training an adapter the *base weights are frozen*, so the working model
checkpoint can never be damaged; merging back is a deliberate, gated action.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .transformer import MiniGPT


def default_targets(model: MiniGPT) -> list[str]:
    """All attention/FFN projection matrices of every layer."""
    names = []
    for i in range(model.config.n_layer):
        p = f"l{i}."
        for w in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            names.append(p + w)
    return names


class LoRAAdapter:
    def __init__(self, model: MiniGPT, rank: int = 4, alpha: float = 8.0,
                 targets: Iterable[str] | None = None, seed: int = 0):
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.targets = list(targets) if targets else default_targets(model)
        rng = np.random.default_rng(seed)
        self.lora_weights: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for name in self.targets:
            base = model.params[name]
            if base.ndim != 2:
                continue
            rows, cols = base.shape
            A = rng.standard_normal((rank, cols)) * 0.01
            B = np.zeros((rows, rank))          # zero-init ⇒ adapter starts neutral
            self.lora_weights[name] = (A.astype(base.dtype),
                                       B.astype(base.dtype), self.scale)

    # ------------------------------------------------------------------ io
    def param_count(self) -> int:
        return int(sum(A.size + B.size for A, B, _ in self.lora_weights.values()))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        flat: dict[str, np.ndarray] = {}
        for name, (A, B, _scale) in self.lora_weights.items():
            flat["A::" + name] = A
            flat["B::" + name] = B
        np.savez_compressed(path, **flat)
        with open(path + ".manifest.json", "w", encoding="utf-8") as fh:
            json.dump({"rank": self.rank, "alpha": self.alpha,
                       "targets": list(self.lora_weights)}, fh, indent=2)

    @classmethod
    def load(cls, path: str, model: MiniGPT) -> "LoRAAdapter":
        with open(path + ".manifest.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        adapter = cls(model, rank=meta["rank"], alpha=meta["alpha"],
                      targets=[])
        data = np.load(path, allow_pickle=False)
        adapter.lora_weights = {}
        for name in meta["targets"]:
            A = data["A::" + name]
            B = data["B::" + name]
            adapter.lora_weights[name] = (A, B, adapter.scale)
        return adapter

    def merge_into(self, model: MiniGPT) -> None:
        """Fold the adapter into base weights (used only by the gated level-5
        update after benchmark comparison)."""
        for name, (A, B, scale) in self.lora_weights.items():
            model.params[name] = model.params[name] + scale * (B @ A)
