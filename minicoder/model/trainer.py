"""Training loop for MiniGPT: AdamW, cosine LR, grad clipping, replay mixing,
EWC-lite regularization, and a checkpoint manager with rollback.

Replay mixing and EWC implement the *catastrophic-forgetting prevention*
mandated by spec §5 (Elastic Weight Consolidation, replay-based learning).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import TrainConfig
from .transformer import MiniGPT

PAD_TARGET = 256      # tokenizer.PAD — targets equal to PAD produce no gradient


class Trainer:
    def __init__(self, model: MiniGPT, config: TrainConfig):
        self.model = model
        self.config = config
        self.opt_state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.step = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------- schedule
    def lr(self, step: int) -> float:
        c = self.config
        if step < c.warmup_steps:
            return c.lr * (step + 1) / max(1, c.warmup_steps)
        progress = (step - c.warmup_steps) / max(1, c.total_steps - c.warmup_steps)
        cosine = 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
        return c.min_lr + (c.lr - c.min_lr) * float(cosine)

    # -------------------------------------------------------------- batching
    def make_batch(self, stream: np.ndarray, batch: int, seq: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        hi = len(stream) - seq - 1
        starts = rng.integers(0, max(1, hi), size=batch)
        x = np.stack([stream[s:s + seq] for s in starts]).astype(np.int64)
        y = np.stack([stream[s + 1:s + seq + 1] for s in starts]).astype(np.int64)
        return x, y

    def stream_from_tokens(self, sequences: list[list[int]]) -> np.ndarray:
        flat: list[int] = []
        for s in sequences:
            flat.extend(s)
        return np.array(flat or [0, 1], dtype=np.int64)

    # -------------------------------------------------------------- training
    def train_step(self, x: np.ndarray, y: np.ndarray, adapter=None,
                   trainable: Optional[set[str]] = None,
                   adapter_only: bool = False,
                   ewc: Optional[dict] = None) -> float:
        loss, grads = self.model.loss_and_grads(
            x, y, adapter=adapter, trainable=trainable, adapter_only=adapter_only)
        if ewc is not None:                                # EWC-lite penalty
            lam = ewc.get("lambda", 0.4)
            fisher, star = ewc["fisher"], ewc["params_star"]
            for name, g in list(grads.items()):
                if name in fisher and name in self.model.params:
                    pen = lam * fisher[name] * (self.model.params[name] - star[name])
                    grads[name] = g + pen
        # global-norm gradient clip
        total = 0.0
        for g in grads.values():
            total += float((g * g).sum())
        norm = np.sqrt(total) or 1.0
        clip = self.config.grad_clip
        if norm > clip:
            scale = clip / norm
            for k in grads:
                grads[k] = g_nan_safe(grads[k] * scale)
        lr = self.lr(self.step)
        self._adamw_update(grads, lr, adapter)
        self.step += 1
        self.history.append({"step": self.step, "loss": round(loss, 5),
                             "lr": lr, "t": time.time()})
        return loss

    def _adamw_update(self, grads: dict, lr: float, adapter=None) -> None:
        c = self.config
        for name, g in grads.items():
            target = self._resolve_param(name, adapter)
            if target is None:
                continue
            arr, setter = target
            m, v = self.opt_state.get(name, (np.zeros_like(arr), np.zeros_like(arr)))
            m = c.adam_beta1 * m + (1 - c.adam_beta1) * g
            v = c.adam_beta2 * v + (1 - c.adam_beta2) * (g * g)
            self.opt_state[name] = (m, v)
            mhat = m / (1 - c.adam_beta1 ** self.step) if self.step > 0 else m
            vhat = v / (1 - c.adam_beta2 ** self.step) if self.step > 0 else v
            update = lr * (mhat / (np.sqrt(vhat) + c.adam_eps))
            if arr.ndim >= 2:
                update = update + lr * c.weight_decay * arr
            setter(arr - update)

    def _resolve_param(self, name: str, adapter):
        """Map a gradient name to the array it updates (base weight or LoRA)."""
        if name.startswith("lora:"):
            assert adapter is not None, "LoRA grads require an adapter"
            lkind = name[5]                                # 'A' or 'B'
            real = name[len("lora:X:"):]
            A, B, scale = adapter.lora_weights[real]
            arr = A if lkind == "A" else B

            def setter(new_arr, real=real, lkind=lkind, A=A, B=B, scale=scale):
                if lkind == "A":
                    adapter.lora_weights[real] = (new_arr, B, scale)
                else:
                    adapter.lora_weights[real] = (A, new_arr, scale)
            return arr, setter
        if name not in self.model.params:
            return None
        arr = self.model.params[name]

        def setter(new_arr, name=name):
            self.model.params[name] = new_arr
        return arr, setter

    # ------------------------------------------------------------ procedures
    def fit(self, sequences: list[list[int]], steps: Optional[int] = None,
            batch: Optional[int] = None, seq: Optional[int] = None,
            adapter=None, adapter_only: bool = False,
            replay_sequences: Optional[list[list[int]]] = None,
            replay_ratio: float = 0.3, ewc: Optional[dict] = None,
            log_fn=None) -> dict:
        """Train on token streams with optional replay mixing + EWC."""
        steps = steps or self.config.total_steps
        batch = batch or self.config.batch_size
        seq = seq or self.config.seq_len
        rng = np.random.default_rng(self.config.seed)
        main_stream = self.stream_from_tokens(sequences)
        replay_stream = (self.stream_from_tokens(replay_sequences)
                         if replay_sequences else None)
        start = time.time()
        for i in range(steps):
            if replay_stream is not None and len(replay_stream) > seq + 2 \
                    and rng.random() < replay_ratio:
                x, y = self.make_batch(replay_stream, batch, seq, rng)
            else:
                x, y = self.make_batch(main_stream, batch, seq, rng)
            loss = self.train_step(x, y, adapter=adapter,
                                   adapter_only=adapter_only, ewc=ewc)
            if log_fn and (i % 20 == 0 or i == steps - 1):
                log_fn(f"step {self.step}/{steps} loss={loss:.4f} lr={self.lr(self.step):.5f}")
        return {"steps": steps, "final_loss": self.history[-1]["loss"],
                "start_loss": self.history[-steps]["loss"] if steps <= len(self.history) else None,
                "duration_s": round(time.time() - start, 2),
                "history_tail": self.history[-min(10, steps):]}

    # ------------------------------------------------------------------ EWC
    def fisher_estimate(self, sequences: list[list[int]], batches: int = 4,
                        batch: int = 8, seq: int = 64) -> dict[str, np.ndarray]:
        """Diagonal Fisher ≈ E[grad²] over probe batches (EWC-lite)."""
        stream = self.stream_from_tokens(sequences)
        rng = np.random.default_rng(self.config.seed + 999)
        fisher: dict[str, np.ndarray] = {}
        for _ in range(batches):
            x, y = self.make_batch(stream, batch, seq, rng)
            _, grads = self.model.loss_and_grads(x, y)
            for name, g in grads.items():
                if name not in self.model.params:
                    continue
                sq = g * g
                fisher[name] = fisher[name] + sq if name in fisher else sq
        for name in fisher:
            fisher[name] = fisher[name] / batches
        return fisher

    @staticmethod
    def ewc_state(model: MiniGPT, fisher: dict[str, np.ndarray],
                  lam: float = 0.4) -> dict:
        return {"lambda": lam, "fisher": fisher,
                "params_star": {k: v.copy() for k, v in model.params.items()}}


def g_nan_safe(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)


# ---------------------------------------------------------------------------
# Checkpoint manager: versioned, with rollback.  Never mutate the only
# working checkpoint.
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    name: str
    step: int
    created: float
    metrics: dict = field(default_factory=dict)
    parent: Optional[str] = None


class CheckpointManager:
    def __init__(self, root: str, keep: int = 5):
        self.root = root
        self.keep = keep
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.json")

    def _load_index(self) -> list[dict]:
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return []

    def _save_index(self, items: list[dict]) -> None:
        from ..memory.store import atomic_write_json
        atomic_write_json(self.index_path, items)

    def save(self, model: MiniGPT, name: str, metrics: Optional[dict] = None,
             adapter=None) -> CheckpointInfo:
        items = self._load_index()
        names = {it["name"] for it in items}
        if name in names:
            name = f"{name}-{int(time.time())}"
        path = os.path.join(self.root, name)
        os.makedirs(path, exist_ok=True)
        model.save_with_manifest(os.path.join(path, "model.npz"),
                                 extra={"metrics": metrics or {}})
        meta = {"name": name, "step": getattr(model, "step", 0),
                "created": time.time(), "metrics": metrics or {},
                "parent": items[-1]["name"] if items else None}
        if adapter is not None:
            adapter.save(os.path.join(path, "adapter.npz"))
            meta["has_adapter"] = True
        items.append(meta)
        # pruning (never prune the only checkpoint!)
        while len(items) > self.keep and len(items) > 1:
            old = items.pop(0)
            import shutil
            shutil.rmtree(os.path.join(self.root, old["name"]), ignore_errors=True)
        self._save_index(items)
        return CheckpointInfo(**{k: meta[k] for k in ("name", "step", "created",
                                                      "metrics", "parent")})

    def latest(self) -> Optional[dict]:
        items = self._load_index()
        return items[-1] if items else None

    def list(self) -> list[dict]:
        return self._load_index()

    def load(self, name: Optional[str] = None) -> tuple[MiniGPT, dict]:
        items = self._load_index()
        if not items:
            raise FileNotFoundError("no checkpoints available")
        meta = items[-1] if name is None else next(
            (it for it in items if it["name"] == name), None)
        if meta is None:
            raise FileNotFoundError(f"checkpoint not found: {name}")
        path = os.path.join(self.root, meta["name"], "model.npz")
        return MiniGPT.load(path), meta

    def rollback(self) -> dict:
        """Remove the newest checkpoint and return the previous one."""
        items = self._load_index()
        if len(items) < 2:
            raise RuntimeError("rollback refused: it would remove the only checkpoint")
        newest = items.pop()
        import shutil
        shutil.rmtree(os.path.join(self.root, newest["name"]), ignore_errors=True)
        self._save_index(items)
        return items[-1]
