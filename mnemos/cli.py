"""mnemos <train|eval|ablate|budget|list> - the whole surface area."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from .ablate import budget_table, matched_baseline_config
from .config import ExperimentConfig
from .data import TASKS, build_dataset
from .evaluate import evaluate, read_ablation
from .registry import available
from .train import Trainer, load_checkpoint, pick_device


def _overrides(cfg: ExperimentConfig, pairs: list[str]) -> ExperimentConfig:
    """--set train.steps=50 model.d_model=64"""
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        node: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        cur = getattr(node, parts[-1])
        val: Any
        if isinstance(cur, bool):
            val = raw.lower() in {"1", "true", "yes"}
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, (list, dict)):
            val = json.loads(raw)
        else:
            val = raw
        setattr(node, parts[-1], val)
    return cfg.validate()


def _load(args) -> ExperimentConfig:
    return _overrides(ExperimentConfig.load(args.config), args.set)


def _dump(obj: Dict, path: Path | None) -> None:
    text = json.dumps(obj, indent=2, default=float)
    print(text)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def cmd_train(args) -> None:
    cfg = _load(args)
    tr = Trainer(cfg)
    print(json.dumps(tr.summary(), indent=2, default=float))
    res = tr.train()
    record = {"config": cfg.name, **tr.summary(), **res}
    _dump(record, Path(args.out) if args.out else Path(tr.out_dir) / "final.json")
    if args.out:                       # keep the run directory self-contained too
        (Path(tr.out_dir) / "final.json").write_text(json.dumps(record, indent=2, default=float))


def cmd_eval(args) -> None:
    model, cfg = load_checkpoint(args.ckpt, args.device)
    data = build_dataset(cfg.data, "val")
    res = evaluate(model, data, args.batches, cfg.train.batch_size,
                   torch.Generator().manual_seed(cfg.train.seed + 10_000),
                   pick_device(args.device))
    _dump(res, Path(args.out) if args.out else None)


def cmd_ablate(args) -> None:
    """Train the memory model, its matched baselines, and read-ablate the result."""
    cfg = _load(args)
    report: Dict[str, Any] = {"name": cfg.name, "budget": budget_table(cfg)}

    tr = Trainer(cfg)
    report["memory_model"] = {**tr.summary(), **tr.train()}
    report["read_ablation"] = read_ablation(
        tr.model, tr.val_data, cfg.train.eval_batches, cfg.train.batch_size,
        cfg.train.seed + 10_000, tr.device
    )

    for match in args.match:
        base_cfg, match_report = matched_baseline_config(cfg, match)
        base = Trainer(base_cfg)
        report[f"baseline_{match}"] = {**match_report, **base.summary(), **base.train()}
        report[f"gain_over_{match}_matched"] = {
            "delta_val_loss": report[f"baseline_{match}"]["val_loss"]
            - report["memory_model"]["val_loss"],
            "delta_val_acc": report["memory_model"]["val_acc"]
            - report[f"baseline_{match}"]["val_acc"],
        }
    _dump(report, Path(args.out) if args.out else Path(cfg.train.out_dir) / "ablation.json")


def cmd_budget(args) -> None:
    _dump(budget_table(_load(args)), Path(args.out) if args.out else None)


def cmd_list(args) -> None:
    print("memory kinds :", ", ".join(available() + ["none"]))
    print("data tasks   :", ", ".join(sorted(TASKS) + ["text"]))
    print("device       :", pick_device("auto"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mnemos", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_config(p):
        p.add_argument("--config", required=True)
        p.add_argument("--set", nargs="*", default=[], help="dotted overrides, key=value")
        p.add_argument("--out", default=None)
        return p

    with_config(sub.add_parser("train", help="train one model")).set_defaults(fn=cmd_train)
    p = with_config(sub.add_parser("ablate", help="train + matched baselines + read ablation"))
    p.add_argument("--match", nargs="*", default=["params", "flops"],
                   choices=["params", "flops"])
    p.set_defaults(fn=cmd_ablate)
    with_config(sub.add_parser("budget", help="parameter and FLOP cost of the memory")).set_defaults(
        fn=cmd_budget
    )

    e = sub.add_parser("eval", help="evaluate a checkpoint")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--batches", type=int, default=16)
    e.add_argument("--device", default="auto")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    sub.add_parser("list", help="registered memory kinds and tasks").set_defaults(fn=cmd_list)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
