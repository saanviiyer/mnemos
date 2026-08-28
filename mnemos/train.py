"""Training loop. Steps, not epochs, because the synthetic tasks are infinite."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from .config import ExperimentConfig
from .data import build_dataset
from .evaluate import evaluate
from .model import MemoryLM


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    prog = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    floor = cfg.lr * cfg.min_lr_frac
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * prog))


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1-D params (norms, gates, learned inner rates) are not weight-decayed
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


class Trainer:
    def __init__(self, cfg: ExperimentConfig, model: Optional[MemoryLM] = None):
        self.cfg = cfg.validate()
        torch.manual_seed(cfg.train.seed)
        self.device = pick_device(cfg.train.device)
        self.model = (model or MemoryLM(cfg.model)).to(self.device)
        self.train_data = build_dataset(cfg.data, "train")
        self.val_data = build_dataset(cfg.data, "val")
        self.opt = build_optimizer(self.model, cfg.train.lr, cfg.train.weight_decay)
        self.g_train = torch.Generator().manual_seed(cfg.train.seed)
        self.g_val = torch.Generator().manual_seed(cfg.train.seed + 10_000)
        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.start_step = 0
        if cfg.train.resume:
            self._resume()
        # A resumed run continues its own log; a fresh one must not append to a
        # previous attempt's.
        self.log_path = self.out_dir / "metrics.jsonl"
        if self.start_step == 0:
            self._rotate_log(self.log_path)
        cfg.save(self.out_dir / "config.yaml")

    @staticmethod
    def _rotate_log(path: Path) -> Path:
        """Move a previous run's metrics aside instead of appending to them.

        Re-running into an existing out_dir used to interleave two runs in one
        metrics.jsonl, so anything reading the eval curve saw a saw-tooth stitched
        from both attempts. Old attempts are preserved as metrics.jsonl.1, .2, ...
        """
        if path.exists() and path.stat().st_size > 0:
            i = 1
            while path.with_suffix(path.suffix + f".{i}").exists():
                i += 1
            path.rename(path.with_suffix(path.suffix + f".{i}"))
        return path

    # -- interruption survival ---------------------------------------------
    @property
    def last_ckpt(self) -> Path:
        return self.out_dir / "ckpt_last.pt"

    def _resume(self) -> None:
        """Restore model, optimizer, step counter and data order from ckpt_last.pt.

        The data generators are restored too. Without that, a resumed run replays the
        batches the killed run already saw, which is a silent extra epoch over the
        early data rather than a continuation.
        """
        if not self.last_ckpt.exists():
            return
        try:
            # CPU, deliberately. Adam keeps its per-parameter step as a tensor, and
            # load_state_dict then places it beside its parameter. Restoring those
            # straight onto MPS put a run in a state where a step counter read back as
            # zero and AdamW divided by 1 - beta1**0 == 0. Loading on CPU and letting
            # load_state_dict do the placement keeps the step counters where the
            # non-capturable Adam path expects them.
            ck = torch.load(self.last_ckpt, map_location="cpu", weights_only=False)
        except Exception as exc:                       # a half-written checkpoint
            print(f"[mnemos] ignoring unreadable {self.last_ckpt}: {exc}")
            return
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        # A parameter that goes several steps without a gradient keeps a stale step
        # count, and one that reads zero makes AdamW divide by zero on the next step.
        # Repair rather than crash: a step of zero means "never updated", which is
        # exactly what a fresh state says.
        for st in self.opt.state.values():
            step = st.get("step")
            if step is not None and float(step) < 1.0:
                st["step"] = torch.ones_like(step) if torch.is_tensor(step) else 1.0
        self.g_train.set_state(ck["g_train"].cpu() if torch.is_tensor(ck["g_train"]) else ck["g_train"])
        self.start_step = int(ck["step"]) + 1
        self.step = self.start_step
        print(f"[mnemos] resuming {self.out_dir.name} at step {self.start_step}")

    def save_last(self) -> None:
        """Atomic mid-run checkpoint: write beside, then rename over."""
        tmp = self.last_ckpt.with_suffix(".pt.tmp")
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "g_train": self.g_train.get_state(), "step": self.step,
                    "config": self.cfg.to_dict()}, tmp)
        tmp.replace(self.last_ckpt)

    # -- helpers -----------------------------------------------------------
    def _to_device(self, batch):
        # Deliberately blocking. A non_blocking copy from unpinned host memory lets the
        # source tensor be freed before the transfer completes, which on MPS silently
        # delivers garbage token ids a few batches in a hundred: the loss goes negative,
        # gradients blow up, and nothing errors. Batches here are tiny, so the async
        # copy bought nothing and cost correctness.
        return tuple(t.to(self.device) for t in batch)

    def _log(self, rec: Dict) -> None:
        rec = {"step": self.step, **rec}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def summary(self) -> Dict[str, float]:
        m = self.model
        return {
            "params_total": m.param_count(),
            "params_non_embedding": m.non_embedding_param_count(),
            "params_memory": m.memory_param_count(),
            "flops_per_token": m.flops_per_token(self.cfg.data.seq_len),
            "device": str(self.device),
            "memory_kind": self.cfg.model.memory.kind,
        }

    # -- main loop ---------------------------------------------------------
    def train(self) -> Dict[str, float]:
        tc = self.cfg.train
        self._log({"event": "start", **self.summary()})
        t0 = time.time()
        last: Dict[str, float] = {}
        self.model.train()
        for step in range(self.start_step, tc.steps):
            self.step = step
            for gp in self.opt.param_groups:
                gp["lr"] = lr_at(step, tc)
            self.opt.zero_grad(set_to_none=True)
            acc_loss = 0.0
            for _ in range(tc.grad_accum):
                x, y, mask = self._to_device(self.train_data.batch(tc.batch_size, self.g_train))
                _, loss = self.model(x, y, mask)
                value = loss.item()
                if not math.isfinite(value) or value < 0.0:
                    raise RuntimeError(
                        f"non-finite or negative loss ({value}) at step {step}. "
                        "Cross-entropy cannot be negative: suspect corrupted inputs "
                        "or a device transfer race, not a bad learning rate."
                    )
                (loss / tc.grad_accum).backward()
                acc_loss += value / tc.grad_accum
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            self.opt.step()
            self.model.commit_memory()

            if step % tc.log_every == 0 or step == tc.steps - 1:
                rec = {"event": "train", "loss": acc_loss, "lr": lr_at(step, tc),
                       "grad_norm": float(gnorm), "elapsed_s": round(time.time() - t0, 2)}
                rec.update(self.model.memory_diagnostics())
                self._log(rec)
            if tc.eval_every and (step + 1) % tc.eval_every == 0:
                last = self.evaluate()
                self.model.train()
            if tc.ckpt_every and (step + 1) % tc.ckpt_every == 0:
                self.save_last()
        if not last:
            last = self.evaluate()
        self.save()
        self._log({"event": "end", **last, "wall_s": round(time.time() - t0, 2)})
        return last

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        g = torch.Generator().manual_seed(self.cfg.train.seed + 10_000)
        res = evaluate(self.model, self.val_data, self.cfg.train.eval_batches,
                       self.cfg.train.batch_size, g, self.device)
        self._log({"event": "eval", **res})
        return res

    def save(self, name: str = "ckpt.pt") -> Path:
        path = self.out_dir / name
        torch.save({"model": self.model.state_dict(), "config": self.cfg.to_dict(),
                    "step": self.step}, path)
        return path


def load_checkpoint(path: str | Path, device: str = "auto"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ck["config"])
    model = MemoryLM(cfg.model)
    model.load_state_dict(ck["model"])
    return model.to(pick_device(device)), cfg
