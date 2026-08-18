"""A decoder-only transformer with memory modules bolted into named layers."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import CausalSelfAttention, RMSNorm, SwiGLU, init_linear_
from .modules.base import MemoryModule
from .registry import build_memory


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, memory: Optional[MemoryModule], placement: str):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_base)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.placement = placement
        self.memory = memory
        self.mlp = None if (memory is not None and placement == "replace_mlp") else SwiGLU(
            cfg.d_model, cfg.d_ff, cfg.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        branch = torch.zeros_like(x)
        if self.mlp is not None:
            branch = branch + self.mlp(h)
        if self.memory is not None:
            branch = branch + self.memory(h)
        return x + branch


class MemoryLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        mem_layers = set(cfg.memory.layers)
        blocks = []
        for i in range(cfg.n_layers):
            mem = None
            if cfg.memory.kind != "none" and i in mem_layers:
                mem = build_memory(cfg.memory.kind, cfg.d_model, **cfg.memory.params)
            blocks.append(Block(cfg, mem, cfg.memory.placement))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_linear_(self, cfg.init_std)
        for blk in self.blocks:                    # re-zero the memory read gates
            if blk.memory is not None and hasattr(blk.memory, "out"):
                nn.init.zeros_(blk.memory.out.weight)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    # -- memory plumbing ------------------------------------------------------
    @property
    def memories(self) -> List[MemoryModule]:
        return [b.memory for b in self.blocks if b.memory is not None]

    def reset_memory(self, batch_size: int, device=None, dtype=None, force: bool = False) -> None:
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        for m in self.memories:
            if force or not m.persistent_state:
                m.reset_memory(batch_size, device, dtype)
            if force and hasattr(m, "clear_datastore"):
                m.clear_datastore()

    def set_ablate(self, flag: bool) -> None:
        for m in self.memories:
            m.ablate = flag

    def commit_memory(self) -> None:
        """Let persistent memories absorb the batch that just ran."""
        for m in self.memories:
            if hasattr(m, "commit"):
                m.commit()

    def memory_diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, b in enumerate(self.blocks):
            if b.memory is None:
                continue
            for k, v in b.memory.diagnostics().items():
                out[f"mem{i}/{k}"] = v
        return out

    # -- forward --------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        b, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        self.reset_memory(b, idx.device, self.embed.weight.dtype)
        x = self.drop(self.embed(idx))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss_tok = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view(b, t)
        if loss_mask is None:
            loss = loss_tok.mean()
        else:
            m = loss_mask.to(loss_tok.dtype)
            loss = (loss_tok * m).sum() / m.sum().clamp_min(1.0)
        return logits, loss

    # -- accounting -----------------------------------------------------------
    def param_count(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)

    def memory_param_count(self) -> int:
        return sum(m.memory_param_count() for m in self.memories)

    def non_embedding_param_count(self) -> int:
        emb = self.embed.weight.numel()
        head = 0 if self.cfg.tie_embeddings else self.head.weight.numel()
        return self.param_count() - emb - head

    def flops_per_token(self, seq_len: Optional[int] = None) -> float:
        seq_len = seq_len or self.cfg.max_seq_len
        total = 0.0
        for blk in self.blocks:
            total += blk.attn.flops_per_token(seq_len)
            if blk.mlp is not None:
                total += blk.mlp.flops_per_token()
            if blk.memory is not None:
                total += blk.memory.flops_per_token(seq_len)
        total += 2 * self.cfg.d_model * self.cfg.vocab_size
        return total

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(window)
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
