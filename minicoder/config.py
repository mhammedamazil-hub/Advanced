"""Global configuration for the MINICODER brain.

All knobs live here so experiments are reproducible: one config object fully
describes a brain instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class MemoryConfig:
    # Working memory
    working_capacity: int = 32          # max items held at once
    working_ttl_seconds: float = 3600.0  # default time-to-live for items
    recency_weight: float = 0.35
    relevance_weight: float = 0.40
    importance_weight: float = 0.25
    # Episodic / semantic retention (sleep stage 5)
    decay_tau_hours: float = 24.0 * 7    # exponential decay time constant
    archive_epsilon: float = 0.05        # retention below this -> archive
    verified_retention_floor: float = 0.35  # verified memories never drop below
    # Recall sizes
    episodic_recall_k: int = 5
    semantic_recall_k: int = 5
    # Pattern discovery
    pattern_min_group: int = 3


@dataclass
class ModelConfig:
    """Compact transformer configuration (defaults ~0.32M params)."""
    vocab_size: int = 800
    d_model: int = 96
    n_layer: int = 3
    n_head: int = 4
    n_kv_head: int = 2            # Grouped Query Attention: 4 query heads share 2 KV heads
    d_ff: int = 256
    max_seq: int = 128
    weight_tying: bool = True     # shared input/output embedding
    dropout: float = 0.0          # kept 0 for CPU reproducibility
    dtype: str = "float64"


@dataclass
class TrainConfig:
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 20
    total_steps: int = 200
    batch_size: int = 16
    seq_len: int = 96
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    seed: int = 1234


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    train_steps: int = 60
    lr: float = 5e-3
    regression_tolerance: float = 0.02  # max allowed benchmark drop


@dataclass
class SleepConfig:
    replay_top_k: int = 30
    max_stages: int = 7
    new_experiences_trigger: int = 8   # auto-sleep when this many unconsolidated
    idle_trigger_seconds: float = 900.0


@dataclass
class BrainConfig:
    base_dir: str = "data"
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    # best-of-N generation attempts per task (verification picks the winner)
    solve_samples: int = 3

    # ---- paths ---------------------------------------------------------
    @property
    def memory_dir(self) -> str:
        return os.path.join(self.base_dir, "memory")

    def memory_path(self, *parts: str) -> str:
        return os.path.join(self.memory_dir, *parts)

    @property
    def db_path(self) -> str:
        return self.memory_path("brain.db")

    @property
    def checkpoints_dir(self) -> str:
        return self.memory_path("checkpoints")

    def ensure_dirs(self) -> None:
        for sub in (
            "working", "episodic", "semantic", "procedural",
            "concepts", "knowledge_graph", "archived", "sleep_logs",
            "checkpoints", "sandbox", "exports", "datasets",
        ):
            os.makedirs(self.memory_path(sub), exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = BrainConfig()
