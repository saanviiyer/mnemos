"""Surprise-gated fast weights (Titans-style test-time memory).

The memory is a weight matrix W that keeps learning *during* the forward pass,
including at eval time. It is trained on an associative objective - store v_t at
key k_t - and updated by the gradient of that objective, with momentum ("surprise
carries over") and decay ("forgetting").

Titans uses an MLP memory and backpropagates through it. This implementation uses a
linear memory, for which the gradient of ||W k - v||^2 is analytic:

    dL/dW = 2 (W k - v) k^T

so the update is exact and closes in a single matmul, with no inner autograd graph
and no per-token Python loop. That is the delta rule. The MLP variant is a real
extension, not a shortcut this module is hiding: see README "Extending".

Causality: chunked. Every token reads W as it stood before its own chunk, so no
token can read a write derived from itself or from anything later.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register_memory
from .base import MemoryModule


@register_memory("surprise")
class SurpriseMemory(MemoryModule):
    persistent_state = False

    def __init__(
        self,
        d_model: int,
        d_key: int = 64,
        d_val: int = 64,
        chunk_size: int = 32,
        inner_lr: float = 0.1,
        momentum: float = 0.9,
        decay: float = 0.01,
        learn_rates: bool = True,
    ):
        super().__init__(d_model)
        self.d_key, self.d_val, self.chunk_size = d_key, d_val, chunk_size
        self.k_proj = nn.Linear(d_model, d_key, bias=False)
        self.v_proj = nn.Linear(d_model, d_val, bias=False)
        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.out = nn.Linear(d_val, d_model, bias=False)
        nn.init.zeros_(self.out.weight)
        self.norm = nn.LayerNorm(d_val)

        def _logit(p):
            p = min(max(p, 1e-4), 1 - 1e-4)
            return torch.log(torch.tensor(p / (1 - p)))

        self.raw_lr = nn.Parameter(torch.log(torch.tensor(inner_lr)), requires_grad=learn_rates)
        self.raw_momentum = nn.Parameter(_logit(momentum), requires_grad=learn_rates)
        self.raw_decay = nn.Parameter(_logit(decay), requires_grad=learn_rates)
        self._W: torch.Tensor | None = None
        self._S: torch.Tensor | None = None

    def reset_memory(self, batch_size: int, device, dtype) -> None:
        self._W = torch.zeros(batch_size, self.d_val, self.d_key, device=device, dtype=dtype)
        self._S = torch.zeros_like(self._W)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if self._W is None or self._W.shape[0] != b:
            self.reset_memory(b, x.device, x.dtype)
        lr = torch.exp(self.raw_lr)
        eta = torch.sigmoid(self.raw_momentum)
        alpha = torch.sigmoid(self.raw_decay)

        k_all = self.k_proj(x)
        v_all = self.v_proj(x)
        q_all = self.q_proj(x)
        # unit-norm keys/queries keep the delta rule numerically well behaved
        k_all = k_all / k_all.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q_all = q_all / q_all.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        W, S = self._W, self._S
        reads, surprise = [], []
        for start in range(0, t, self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            q, k, v = q_all[:, sl], k_all[:, sl], v_all[:, sl]
            reads.append(torch.bmm(q, W.transpose(1, 2)))       # read pre-chunk W
            err = torch.bmm(k, W.transpose(1, 2)) - v           # (B, C, d_val)
            grad = torch.bmm(err.transpose(1, 2), k) / max(k.shape[1], 1)
            with torch.no_grad():
                surprise.append(err.float().pow(2).mean().item())
            S = eta * S - lr * grad
            W = (1 - alpha) * W + S
        self._W, self._S = W, S
        self._diag["surprise"] = sum(surprise) / max(len(surprise), 1)
        self._diag["mem_norm"] = float(W.detach().float().norm(dim=(1, 2)).mean().item())
        y = torch.cat(reads, dim=1)
        return self.out(self.norm(y))

    def flops_per_token(self, seq_len: int) -> float:
        proj = 2 * 3 * self.d_model * self.d_key
        read = 2 * self.d_key * self.d_val
        write = 3 * 2 * self.d_key * self.d_val
        return proj + read + write + 2 * self.d_val * self.d_model
