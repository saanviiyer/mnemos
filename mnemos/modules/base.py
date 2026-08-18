"""Contract every memory module obeys.

A memory module is a *read* into the residual stream: it takes the hidden states
of a layer and returns a delta of the same shape. That shape choice is what makes
the causal control cheap - setting ``ablate = True`` zeroes the delta, so you can
measure what the model loses when the memory read is removed without touching any
other parameter.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    #: modules whose state should survive across batches (a datastore) set this True.
    #: modules whose state is per-sequence (a recurrent bank) leave it False and get
    #: reset by the model at the top of every forward pass.
    persistent_state: bool = False
    kind: str = "base"

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.ablate = False
        self._diag: Dict[str, float] = {}

    # -- subclasses implement these -------------------------------------------------
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset_memory(self, batch_size: int, device, dtype) -> None:
        """Clear per-sequence state. Called by the model unless persistent_state."""

    def flops_per_token(self, seq_len: int) -> float:
        return 0.0

    def memory_param_count(self) -> int:
        """Parameters that exist only because of the memory (used for matched controls)."""
        return sum(p.numel() for p in self.parameters())

    # -- shared plumbing ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablate:
            self._diag = {"read_norm": 0.0, "ablated": 1.0}
            return torch.zeros_like(x)
        out = self._forward(x)
        with torch.no_grad():
            self._diag["read_norm"] = out.float().pow(2).sum(-1).sqrt().mean().item()
            self._diag["ablated"] = 0.0
        return out

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._diag)
