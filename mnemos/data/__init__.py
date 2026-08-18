from __future__ import annotations

from ..config import DataConfig
from .synthetic import TASKS, SyntheticTask, build_task
from .text import ByteTextDataset


def build_dataset(cfg: DataConfig, split: str = "train"):
    if cfg.kind == "text":
        if not cfg.path:
            raise ValueError("data.kind == 'text' requires data.path")
        return ByteTextDataset(cfg.path, cfg.seq_len, split=split, **cfg.params)
    # synthetic tasks are generated on the fly; the split is just a different seed
    return build_task(cfg.kind, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, **cfg.params)


__all__ = ["build_dataset", "build_task", "ByteTextDataset", "SyntheticTask", "TASKS"]
