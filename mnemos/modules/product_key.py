"""Product-key memory (Lample et al. 2019; Meta memory layers, 2024).

A dense table of n_keys^2 value slots is searched in O(n_keys) instead of O(n_keys^2)
by factorising each key into two halves and taking a top-k on each half separately.
This is the module that actually makes a model "large memory": parameters grow with
the table while per-token FLOPs stay near constant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_memory
from .base import MemoryModule


@register_memory("product_key")
class ProductKeyMemory(MemoryModule):
    def __init__(
        self,
        d_model: int,
        n_keys: int = 64,
        n_heads: int = 4,
        topk: int = 8,
        half_dim: int = 32,
        v_dim: int | None = None,
        dropout: float = 0.0,
        value_init_std: float = 0.02,
        track_usage: bool = True,
    ):
        super().__init__(d_model)
        self.n_keys = n_keys
        self.n_slots = n_keys * n_keys
        self.n_heads = n_heads
        self.topk = min(topk, n_keys)
        self.half_dim = half_dim
        self.v_dim = v_dim or d_model
        self.track_usage = track_usage

        self.query = nn.Linear(d_model, n_heads * 2 * half_dim, bias=False)
        self.query_norm = nn.LayerNorm(2 * half_dim)
        # per-head keys: (H, 2, n_keys, half_dim)
        self.keys = nn.Parameter(torch.randn(n_heads, 2, n_keys, half_dim) * (half_dim ** -0.5))
        self.values = nn.Parameter(torch.randn(self.n_slots, self.v_dim) * value_init_std)
        self.out = nn.Linear(n_heads * self.v_dim, d_model, bias=False)
        nn.init.zeros_(self.out.weight)  # start as a no-op read, let the gate open with training
        self.drop = nn.Dropout(dropout)
        self.register_buffer("usage", torch.zeros(self.n_slots), persistent=False)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        n = b * t
        q = self.query(x).view(n, self.n_heads, 2, self.half_dim)
        q = self.query_norm(q.reshape(n, self.n_heads, 2 * self.half_dim)).view(
            n, self.n_heads, 2, self.half_dim
        )

        # score each half independently: (N, H, 2, n_keys)
        scores = torch.einsum("nhpd,hpkd->nhpk", q, self.keys)
        s, i = scores.topk(self.topk, dim=-1)                     # (N, H, 2, k)
        s1, s2 = s[:, :, 0], s[:, :, 1]                           # (N, H, k)
        i1, i2 = i[:, :, 0], i[:, :, 1]

        cand = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).view(n, self.n_heads, -1)   # (N,H,k*k)
        cand_idx = (i1.unsqueeze(-1) * self.n_keys + i2.unsqueeze(-2)).view(n, self.n_heads, -1)
        top_s, top_pos = cand.topk(self.topk, dim=-1)              # (N, H, k)
        slot = cand_idx.gather(-1, top_pos)                        # (N, H, k)

        w = F.softmax(top_s.float(), dim=-1).to(x.dtype)
        vals = F.embedding(slot.reshape(-1), self.values).view(n, self.n_heads, self.topk, self.v_dim)
        read = (vals * w.unsqueeze(-1)).sum(dim=2)                 # (N, H, v_dim)
        out = self.out(self.drop(read.reshape(n, self.n_heads * self.v_dim))).view(b, t, self.d_model)

        if self.track_usage:
            with torch.no_grad():
                flat = slot.reshape(-1)
                hits = torch.bincount(flat, minlength=self.n_slots).float()
                self.usage.mul_(0.99).add_(hits / max(flat.numel(), 1))
                p = self.usage / self.usage.sum().clamp_min(1e-9)
                self._diag["slot_entropy_bits"] = float(
                    -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
                )
                self._diag["slot_entropy_frac"] = self._diag["slot_entropy_bits"] / (
                    torch.log2(torch.tensor(float(self.n_slots))).item()
                )
                self._diag["slots_touched_frac"] = float((hits > 0).float().mean().item())
        return out

    def flops_per_token(self, seq_len: int) -> float:
        h, k, hd = self.n_heads, self.topk, self.half_dim
        q_proj = 2 * self.d_model * h * 2 * hd
        search = 2 * h * 2 * self.n_keys * hd
        gather = h * k * self.v_dim
        o_proj = 2 * h * self.v_dim * self.d_model
        return q_proj + search + gather + o_proj
