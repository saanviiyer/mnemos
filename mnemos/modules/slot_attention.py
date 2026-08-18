"""Recurrent slot memory read by cross-attention (LM2 / RMT family).

A bank of n_slots vectors is carried across the sequence. The sequence is cut into
chunks; each chunk *reads* the bank as it stood before the chunk began and then
*writes* a gated summary of itself back. Reading strictly before writing is what
keeps the module causal - a token can never see its own chunk's write, let alone
a later one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_memory
from .base import MemoryModule


@register_memory("slot")
class SlotMemory(MemoryModule):
    persistent_state = False

    def __init__(
        self,
        d_model: int,
        n_slots: int = 64,
        d_mem: int | None = None,
        n_heads: int = 4,
        chunk_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.n_slots = n_slots
        self.d_mem = d_mem or d_model
        self.n_heads = n_heads
        self.head_dim = self.d_mem // n_heads
        if self.d_mem % n_heads != 0:
            raise ValueError("d_mem must be divisible by n_heads")
        self.chunk_size = chunk_size

        self.slot_init = nn.Parameter(torch.randn(n_slots, self.d_mem) * 0.02)
        self.q = nn.Linear(d_model, self.d_mem, bias=False)
        self.k = nn.Linear(self.d_mem, self.d_mem, bias=False)
        self.v = nn.Linear(self.d_mem, self.d_mem, bias=False)
        self.out = nn.Linear(self.d_mem, d_model, bias=False)
        nn.init.zeros_(self.out.weight)

        # write path: chunk summary -> candidate slot content, plus input/forget gates
        self.write_q = nn.Linear(d_model, self.d_mem, bias=False)
        self.write_gate = nn.Linear(d_model + self.d_mem, 2 * self.d_mem)
        self.drop = nn.Dropout(dropout)
        self._state: torch.Tensor | None = None

    def reset_memory(self, batch_size: int, device, dtype) -> None:
        self._state = self.slot_init.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        ).contiguous()

    def _read(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(mem).view(b, self.n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(mem).view(b, self.n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, t, self.d_mem)
        return self.out(self.drop(o))

    def _write(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # attention from slots (queries) to the chunk (keys/values): each slot pulls
        # what it cares about out of the chunk.
        b, t, _ = x.shape
        cand_k = self.write_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        slot_q = mem.view(b, self.n_slots, self.n_heads, self.head_dim).transpose(1, 2)
        cand = F.scaled_dot_product_attention(slot_q, cand_k, cand_k)
        cand = cand.transpose(1, 2).reshape(b, self.n_slots, self.d_mem)
        summary = x.mean(dim=1, keepdim=True).expand(-1, self.n_slots, -1)
        gates = self.write_gate(torch.cat([summary, mem], dim=-1))
        forget, inp = gates.chunk(2, dim=-1)
        return torch.sigmoid(forget) * mem + torch.sigmoid(inp) * torch.tanh(cand)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if self._state is None or self._state.shape[0] != b:
            self.reset_memory(b, x.device, x.dtype)
        mem = self._state
        reads, gate_mass = [], []
        for start in range(0, t, self.chunk_size):
            chunk = x[:, start : start + self.chunk_size]
            reads.append(self._read(chunk, mem))          # read the pre-chunk bank
            new_mem = self._write(chunk, mem)             # then write
            with torch.no_grad():
                gate_mass.append((new_mem - mem).float().abs().mean().item())
            mem = new_mem
        self._state = mem
        self._diag["write_delta"] = sum(gate_mass) / max(len(gate_mass), 1)
        self._diag["n_chunks"] = float(len(gate_mass))
        return torch.cat(reads, dim=1)

    def flops_per_token(self, seq_len: int) -> float:
        read = 2 * (self.d_model * self.d_mem) + 2 * 2 * self.n_slots * self.d_mem
        write = (2 * self.d_model * self.d_mem + 4 * self.n_slots * self.d_mem) / max(
            self.chunk_size, 1
        )
        return read + write
