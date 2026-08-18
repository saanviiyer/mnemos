"""Synthetic probes that separate memory from pattern-matching.

Every task returns ``(x, y, mask)`` where ``y`` is ``x`` shifted left by one and
``mask`` marks the positions that are actually being scored. Masking matters: on a
recall task, most positions are free tokens the model can guess from the marginal,
so an unmasked loss dilutes the signal you care about by an order of magnitude and
makes a memoryless baseline look competitive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

PAD, SEP, BOS = 0, 1, 2
N_SPECIAL = 3

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class TaskSpec:
    vocab_size: int
    seq_len: int


class SyntheticTask:
    """Base class: subclasses implement ``_sequences``, which returns (B, T) tokens
    and a (B, T) boolean mask over *target* positions."""

    name = "base"

    def __init__(self, seq_len: int, vocab_size: int, **params):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.params = params

    def _sequences(self, batch_size: int, g: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def batch(self, batch_size: int, g: torch.Generator) -> Batch:
        toks, tgt_mask = self._sequences(batch_size, g)
        x, y = toks[:, :-1], toks[:, 1:]
        mask = tgt_mask[:, 1:]
        return x.contiguous(), y.contiguous(), mask.contiguous()

    # helper
    def _rand(self, shape, low, high, g):
        return torch.randint(low, high, shape, generator=g)


class AssociativeRecall(SyntheticTask):
    """key-value pairs, then queries. The canonical memory probe (MQAR)."""

    name = "associative_recall"

    def __init__(self, seq_len: int, vocab_size: int, n_pairs: int = 16, n_queries: int = 4,
                 n_keys: int | None = None, n_values: int | None = None):
        super().__init__(seq_len, vocab_size)
        self.n_pairs, self.n_queries = n_pairs, n_queries
        budget = vocab_size - N_SPECIAL
        self.n_keys = n_keys or max(n_pairs, budget // 2)
        self.n_values = n_values or (budget - self.n_keys)
        if self.n_keys < n_pairs:
            raise ValueError("need at least n_pairs distinct keys")
        if self.n_keys + self.n_values + N_SPECIAL > vocab_size:
            raise ValueError("key/value ranges overflow vocab_size")
        self.key0 = N_SPECIAL
        self.val0 = N_SPECIAL + self.n_keys
        need = 1 + 2 * n_pairs + 1 + 2 * n_queries
        if need > seq_len:
            raise ValueError(f"seq_len {seq_len} too short for task (needs {need})")

    def _sequences(self, batch_size: int, g: torch.Generator):
        b, p, q = batch_size, self.n_pairs, self.n_queries
        # distinct keys per sequence via per-row permutation of the key range
        perm = torch.argsort(torch.rand(b, self.n_keys, generator=g), dim=-1)[:, :p]
        keys = perm + self.key0                                          # (B, p)
        vals = self._rand((b, p), self.val0, self.val0 + self.n_values, g)

        qi = self._rand((b, q), 0, p, g)
        qk = keys.gather(1, qi)
        qv = vals.gather(1, qi)

        body = torch.stack([keys, vals], dim=-1).view(b, 2 * p)
        tail = torch.stack([qk, qv], dim=-1).view(b, 2 * q)
        core = torch.cat(
            [torch.full((b, 1), BOS), body, torch.full((b, 1), SEP), tail], dim=1
        )
        toks = torch.full((b, self.seq_len), PAD, dtype=torch.long)
        toks[:, : core.shape[1]] = core

        mask = torch.zeros(b, self.seq_len, dtype=torch.bool)
        start = 1 + 2 * p + 1
        mask[:, start + 1 : start + 2 * q : 2] = True                    # value positions only
        return toks, mask


class Needle(SyntheticTask):
    """One fact buried in filler, retrieved after a long gap."""

    name = "needle"

    def __init__(self, seq_len: int, vocab_size: int, n_facts: int = 1):
        super().__init__(seq_len, vocab_size)
        self.n_facts = n_facts
        budget = vocab_size - N_SPECIAL
        self.n_keys = budget // 2
        self.key0, self.val0 = N_SPECIAL, N_SPECIAL + budget // 2
        self.n_values = vocab_size - self.val0

    def _sequences(self, batch_size: int, g: torch.Generator):
        b, t = batch_size, self.seq_len
        # filler is drawn from the value range only, so a key token appears exactly
        # once per sequence and the probe has a unique answer.
        filler = self._rand((b, t), self.val0, self.vocab_size, g)
        toks = filler
        toks[:, 0] = BOS
        keys = self._rand((b, self.n_facts), self.key0, self.key0 + self.n_keys, g)
        vals = self._rand((b, self.n_facts), self.val0, self.vocab_size, g)
        probe_len = 2 + 2 * self.n_facts
        span = t - probe_len - 2 * self.n_facts - 1
        if span <= 1:
            raise ValueError("seq_len too short for needle task")
        for f in range(self.n_facts):
            pos = torch.randint(1, span, (b,), generator=g)
            toks.scatter_(1, pos.unsqueeze(1), keys[:, f : f + 1])
            toks.scatter_(1, (pos + 1).unsqueeze(1), vals[:, f : f + 1])
        tail = torch.stack([keys, vals], dim=-1).view(b, 2 * self.n_facts)
        toks[:, t - 2 * self.n_facts - 1] = SEP
        toks[:, t - 2 * self.n_facts :] = tail
        mask = torch.zeros(b, t, dtype=torch.bool)
        mask[:, t - 2 * self.n_facts + 1 :: 2] = True
        return toks, mask


class Copy(SyntheticTask):
    """Reproduce a prefix after a separator. Tests bulk retention, not selection."""

    name = "copy"

    def __init__(self, seq_len: int, vocab_size: int, span: int | None = None):
        super().__init__(seq_len, vocab_size)
        self.span = span or (seq_len - 2) // 2
        if 2 * self.span + 2 > seq_len:
            raise ValueError("span too long for seq_len")

    def _sequences(self, batch_size: int, g: torch.Generator):
        b, t, s = batch_size, self.seq_len, self.span
        src = self._rand((b, s), N_SPECIAL, self.vocab_size, g)
        core = torch.cat([torch.full((b, 1), BOS), src, torch.full((b, 1), SEP), src], dim=1)
        toks = torch.full((b, t), PAD, dtype=torch.long)
        toks[:, : core.shape[1]] = core
        mask = torch.zeros(b, t, dtype=torch.bool)
        mask[:, s + 2 : 2 * s + 2] = True
        return toks, mask


class InductionHead(SyntheticTask):
    """After a marker seen once before, emit the token that followed it."""

    name = "induction"

    def __init__(self, seq_len: int, vocab_size: int):
        super().__init__(seq_len, vocab_size)

    def _sequences(self, batch_size: int, g: torch.Generator):
        b, t = batch_size, self.seq_len
        toks = self._rand((b, t), N_SPECIAL + 1, self.vocab_size, g)
        toks[:, 0] = BOS
        marker = N_SPECIAL
        first = torch.randint(1, t // 2, (b,), generator=g)
        toks.scatter_(1, first.unsqueeze(1), torch.full((b, 1), marker))
        toks[:, t - 2] = marker
        answer = toks.gather(1, (first + 1).unsqueeze(1))
        toks[:, t - 1 : t] = answer
        mask = torch.zeros(b, t, dtype=torch.bool)
        mask[:, t - 1] = True
        return toks, mask


TASKS = {c.name: c for c in [AssociativeRecall, Needle, Copy, InductionHead]}


def build_task(kind: str, seq_len: int, vocab_size: int, **params) -> SyntheticTask:
    if kind not in TASKS:
        raise KeyError(f"unknown synthetic task {kind!r}; known: {sorted(TASKS)}")
    return TASKS[kind](seq_len=seq_len, vocab_size=vocab_size, **params)
