"""Non-parametric kNN memory over past activations (Memorizing Transformers).

A FIFO datastore of (key, value) pairs harvested from earlier *segments*. Retrieval
is exact top-k by dot product - no ANN index, no extra dependency - which is the
right trade at the scales this repo targets and swappable at larger ones.

Causality: the datastore is written **after** the forward pass returns, so nothing
in the current segment can retrieve itself. Across segments the memory is by design
persistent: that is the point of the module, and it is why ``persistent_state`` is
True and the model does not silently clear it between batches.

**One projection, not three.** Stored keys and values are detached, so a separate
``k_proj`` and ``v_proj`` feeding only the datastore would receive no gradient ever
and be trained by nothing -- the first version of this module had exactly that, and
the ragged optimizer state it produced is what surfaced it. Memorizing Transformers
does not have the problem because its kNN memory reuses the enclosing attention
layer's own key and value projections, which that layer trains. This module has no
enclosing attention to borrow from, so it borrows from itself: the query projection
also produces the stored keys, and the stored values are the hidden states directly.
Every parameter here is now on the gradient path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_memory
from .base import MemoryModule


@register_memory("knn")
class KNNMemory(MemoryModule):
    persistent_state = True

    def __init__(
        self,
        d_model: int,
        capacity: int = 8192,
        d_key: int = 64,
        topk: int = 16,
        dropout: float = 0.0,
        write_stride: int = 1,
    ):
        super().__init__(d_model)
        self.capacity, self.topk, self.d_key = capacity, topk, d_key
        self.write_stride = max(write_stride, 1)
        # one projection, used for the live query and for the keys written to the
        # store, so the gradient from retrieval trains the thing that built the index
        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out.weight)
        self.gate = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)
        self.register_buffer("store_k", torch.zeros(0, d_key), persistent=False)
        self.register_buffer("store_v", torch.zeros(0, d_model), persistent=False)
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def clear_datastore(self) -> None:
        self.store_k = self.store_k.new_zeros(0, self.d_key)
        self.store_v = self.store_v.new_zeros(0, self.d_model)
        self._pending = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = F.normalize(self.q_proj(x), dim=-1)
        # the same projection supplies the keys the store is indexed by, and the
        # stored values are the hidden states themselves
        self._pending = (q.detach().reshape(-1, self.d_key),
                         x.detach().reshape(-1, self.d_model))

        n_store = self.store_k.shape[0]
        if n_store == 0:
            # An empty datastore reads as zero, but the read must stay attached to the
            # autograd graph: returning a bare zeros_like would make the very first
            # optimiser step fail with "does not require grad".
            self._diag.update(store_size=0.0, retrieval_hit=0.0)
            return self.out(x) * 0.0

        q = q.reshape(b * t, self.d_key)
        sims = q @ self.store_k.t()                              # (B*T, N)
        kk = min(self.topk, n_store)
        top_s, top_i = sims.topk(kk, dim=-1)
        w = F.softmax(top_s.float() / max(self.d_key ** 0.5, 1e-6), dim=-1).to(x.dtype)
        vals = self.store_v[top_i.reshape(-1)].view(b * t, kk, self.d_model)
        read = (vals * w.unsqueeze(-1)).sum(dim=1).view(b, t, self.d_model)
        self._diag.update(store_size=float(n_store), retrieval_hit=float(top_s.mean().item()))
        return torch.sigmoid(self.gate) * self.out(self.drop(read))

    @torch.no_grad()
    def commit(self) -> None:
        """Flush the last forward pass into the datastore. Call after the step."""
        if self._pending is None:
            return
        k, v = self._pending
        self._pending = None
        if self.write_stride > 1:
            k, v = k[:: self.write_stride], v[:: self.write_stride]
        store_k = torch.cat([self.store_k.to(k.device, k.dtype), k], dim=0)
        store_v = torch.cat([self.store_v.to(v.device, v.dtype), v], dim=0)
        if store_k.shape[0] > self.capacity:
            store_k = store_k[-self.capacity :]
            store_v = store_v[-self.capacity :]
        self.store_k, self.store_v = store_k, store_v

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, seq_len: int) -> float:
        n = float(min(self.capacity, max(self.store_k.shape[0], 1)))
        return 2 * self.d_model * self.d_key + 2 * n * self.d_key \
            + self.topk * self.d_model + 2 * self.d_model * self.d_model
