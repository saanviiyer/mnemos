"""Matched controls.

A memory module adds parameters *and* compute *and* a new pathway. If you compare it
against the model you started with, all three are confounded and you cannot say which
one bought the improvement. Two controls separate them:

* **matched baseline** - no memory, but the feed-forward width is widened until the
  parameter count (or the per-token FLOPs) matches the memory model. If the memory
  model no longer wins, the win was capacity.
* **read ablation** - the same trained memory model, evaluated with the memory read
  forced to zero. If nothing changes, the pathway is decorative.

They disagree in an informative way for sparse memories: product-key memory adds a lot
of parameters for very few FLOPs, so a parameter-matched dense baseline is *given more
compute* than the memory model, and a FLOP-matched one is given far fewer parameters.
Report both numbers. Reporting whichever one flatters the method is the failure mode
this module exists to make inconvenient.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, Literal, Tuple

from .config import ExperimentConfig
from .model import MemoryLM

Match = Literal["params", "flops"]


def _measure(cfg: ExperimentConfig, seq_len: int) -> Tuple[int, float]:
    m = MemoryLM(cfg.model)
    return m.param_count(), m.flops_per_token(seq_len)


def matched_baseline_config(
    cfg: ExperimentConfig, match: Match = "params", max_scale: int = 64
) -> Tuple[ExperimentConfig, Dict[str, float]]:
    """Return a memory-free config whose budget matches ``cfg`` on ``match``."""
    if cfg.model.memory.kind == "none":
        raise ValueError("config has no memory; there is nothing to match")
    seq_len = cfg.data.seq_len
    target_params, target_flops = _measure(cfg, seq_len)
    target = target_params if match == "params" else target_flops

    def budget_for(d_ff: int) -> float:
        trial = copy.deepcopy(cfg)
        trial.model.memory.kind = "none"
        trial.model.memory.layers = []
        trial.model.memory.params = {}
        trial.model.d_ff = d_ff
        p, f = _measure(trial, seq_len)
        return p if match == "params" else f

    lo, hi = 1, max(cfg.model.d_ff * max_scale, cfg.model.d_ff + 8)
    if budget_for(hi) < target:
        raise ValueError(
            f"cannot reach the {match} budget by widening d_ff up to {hi}; "
            "match on n_layers or raise max_scale"
        )
    while lo < hi:                                    # smallest d_ff that reaches target
        mid = (lo + hi) // 2
        if budget_for(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    best = lo

    out = copy.deepcopy(cfg)
    out.name = f"{cfg.name}-matched_{match}"
    out.model.memory.kind = "none"
    out.model.memory.layers = []
    out.model.memory.params = {}
    out.model.d_ff = best
    out.train.out_dir = f"{cfg.train.out_dir}-matched_{match}"
    out.validate()

    got_p, got_f = _measure(out, seq_len)
    report = {
        "match_on": match,
        "d_ff_memory_model": cfg.model.d_ff,
        "d_ff_baseline": best,
        "params_memory_model": float(target_params),
        "params_baseline": float(got_p),
        "flops_memory_model": float(target_flops),
        "flops_baseline": float(got_f),
        "param_ratio": got_p / max(target_params, 1),
        "flop_ratio": got_f / max(target_flops, 1e-9),
    }
    return out, report


def budget_table(cfg: ExperimentConfig) -> Dict[str, float]:
    """What the memory actually costs, before any training happens."""
    m = MemoryLM(cfg.model)
    seq_len = cfg.data.seq_len
    bare = copy.deepcopy(cfg)
    bare.model.memory.kind, bare.model.memory.layers, bare.model.memory.params = "none", [], {}
    b = MemoryLM(bare.model)
    return {
        "params_with_memory": m.param_count(),
        "params_without_memory": b.param_count(),
        "params_from_memory": m.memory_param_count(),
        "param_overhead_x": m.param_count() / max(b.param_count(), 1),
        "flops_per_token_with_memory": m.flops_per_token(seq_len),
        "flops_per_token_without_memory": b.flops_per_token(seq_len),
        "flop_overhead_x": m.flops_per_token(seq_len) / max(b.flops_per_token(seq_len), 1e-9),
    }
