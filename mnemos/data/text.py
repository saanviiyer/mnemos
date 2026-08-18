"""Byte-level text data. No tokenizer, no download, no vocab file.

Byte level keeps the loop honest at this scale: a 256-token vocabulary means the
embedding table is not the dominant parameter, so a memory module's parameter share
is visible rather than buried under embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


class ByteTextDataset:
    vocab_size = 256

    def __init__(self, path: str | Path, seq_len: int, split: str = "train",
                 val_frac: float = 0.1):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Point data.path at any .txt file, or use a synthetic task."
            )
        raw = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).long()
        if raw.numel() < seq_len + 2:
            raise ValueError(f"{path} has {raw.numel()} bytes, need > {seq_len + 2}")
        cut = int(raw.numel() * (1 - val_frac))
        self.data = raw[:cut] if split == "train" else raw[cut:]
        self.seq_len = seq_len
        self.split = split

    def batch(self, batch_size: int, g: torch.Generator) -> Tuple[torch.Tensor, ...]:
        hi = self.data.numel() - self.seq_len - 1
        ix = torch.randint(0, max(hi, 1), (batch_size,), generator=g)
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix])
        return x, y, torch.ones_like(x, dtype=torch.bool)
