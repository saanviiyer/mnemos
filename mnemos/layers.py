"""Transformer primitives: RMSNorm, RoPE, causal attention, SwiGLU."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """Returns (cos, sin) of shape (1, 1, seq_len, head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)       # (T, head_dim)
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, Dh)."""
    t = x.shape[-2]
    return x * cos[..., :t, :] + _rotate_half(x) * sin[..., :t, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_base = rope_base
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = build_rope_cache(t, self.head_dim, self.rope_base, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)

    def flops_per_token(self, seq_len: int) -> float:
        d = self.n_heads * self.head_dim
        proj = 2 * 4 * d * d                       # qkv + out projections
        attn = 2 * 2 * d * seq_len / 2             # causal: half the score/value work
        return proj + attn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.d_model, self.d_ff = d_model, d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))

    def flops_per_token(self) -> float:
        return 2 * 3 * self.d_model * self.d_ff

    def param_count(self) -> int:
        return 3 * self.d_model * self.d_ff


def init_linear_(module: nn.Module, std: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
