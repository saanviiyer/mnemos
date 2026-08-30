"""Typed configs. YAML in, dataclasses out, no dict-spelunking downstream."""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class MemoryConfig:
    kind: str = "none"                 # none | product_key | slot | surprise | knn
    layers: List[int] = field(default_factory=list)
    placement: str = "parallel"        # parallel | replace_mlp
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    init_std: float = 0.02
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass
class DataConfig:
    kind: str = "associative_recall"
    seq_len: int = 128
    vocab_size: int = 256
    params: Dict[str, Any] = field(default_factory=dict)
    path: str | None = None            # for kind: text


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 8
    log_every: int = 25
    ckpt_every: int = 0        # 0 disables mid-run checkpointing
    resume: bool = False       # opt-in: pick up from ckpt_last.pt if present
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs/default"
    compile: bool = False


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- (de)serialisation -----------------------------------------------------
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        d = copy.deepcopy(d or {})
        model_d = d.get("model", {}) or {}
        mem_d = model_d.pop("memory", {}) or {}
        cfg = ExperimentConfig(
            name=d.get("name", "unnamed"),
            model=ModelConfig(**model_d, memory=MemoryConfig(**mem_d)),
            data=DataConfig(**(d.get("data", {}) or {})),
            train=TrainConfig(**(d.get("train", {}) or {})),
        )
        cfg.validate()
        return cfg

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return ExperimentConfig.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def validate(self) -> "ExperimentConfig":
        m = self.model
        if m.d_model % m.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if m.memory.placement not in {"parallel", "replace_mlp"}:
            raise ValueError("memory.placement must be 'parallel' or 'replace_mlp'")
        for li in m.memory.layers:
            if not 0 <= li < m.n_layers:
                raise ValueError(f"memory.layers index {li} outside [0, {m.n_layers})")
        if m.memory.kind == "none" and m.memory.layers:
            raise ValueError("memory.kind is 'none' but memory.layers is non-empty")
        if m.memory.kind != "none" and not m.memory.layers:
            raise ValueError(f"memory.kind is {m.memory.kind!r} but no memory.layers given")
        if self.data.seq_len > m.max_seq_len:
            raise ValueError("data.seq_len exceeds model.max_seq_len")
        if self.data.kind != "text" and self.data.vocab_size > m.vocab_size:
            raise ValueError("data.vocab_size exceeds model.vocab_size")
        if m.memory.kind == "surprise":
            # Two chunks silently freeze the momentum and forgetting rates: the first
            # chunk multiplies a zero state and the last chunk's write is never read,
            # so autograd never sees either rate. The run trains and reports nothing
            # unusual, which is why this is an error rather than a note.
            chunk = int(m.memory.params.get("chunk_size", 32))
            n_chunks = -(-self.data.seq_len // max(chunk, 1))
            if n_chunks < 3:
                raise ValueError(
                    f"surprise memory gets {n_chunks} chunk(s) at seq_len "
                    f"{self.data.seq_len} and chunk_size {chunk}; its momentum and "
                    "decay rates are unlearnable below 3 chunks. Lower chunk_size.")
        return self
