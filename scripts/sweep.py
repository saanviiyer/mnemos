#!/usr/bin/env python3
"""Resumable multi-seed sweep driver.

Rerun the exact same command after an interruption and it continues: every
(arm, seed) owns a directory, a finished run leaves ``final.json`` behind and is
skipped, and a run killed mid-flight restarts from its last ``ckpt_last.pt``
with the data order intact. Nothing is recomputed and nothing is overwritten.

    python scripts/sweep.py --scale small --seeds 0,1,2 --dry-run
    nohup python scripts/sweep.py --scale small --seeds 0,1,2 > sweep.log 2>&1 &

Each memory arm is read-ablated straight after training, while its checkpoint is
already in memory, so the causal control comes with error bars rather than being a
one-off afterthought.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.ablate import budget_table, matched_baseline_config          # noqa: E402
from mnemos.config import ExperimentConfig                               # noqa: E402
from mnemos.evaluate import read_ablation                                # noqa: E402
from mnemos.train import Trainer                                         # noqa: E402

# --- scales -------------------------------------------------------------------
# "tiny" reproduces RESULTS.md. "small" is the scale-up: 2x width, 1.5x depth,
# 3.5x sequence, 2.7x pairs to remember, 5x steps.
def _pkm(n_keys: int, v_dim: int = 128) -> dict:
    return {"kind": "product_key",
            "params": dict(n_keys=n_keys, n_heads=4, topk=8, half_dim=32, v_dim=v_dim)}


SCALES: Dict[str, dict] = {
    "tiny": dict(
        vocab_size=64, d_model=128, n_layers=4, n_heads=4, d_ff=256,
        seq_len=36, n_pairs=12, n_queries=4,
        steps=3000, batch_size=128, lr=3e-3, warmup=200,
        eval_every=750, eval_batches=16, log_every=100, ckpt_every=500,
        memory_layer=2, controls_for=["product_key"],
        arms={
            "none": {"kind": "none", "params": {}},
            "product_key": _pkm(32, v_dim=64),
            "slot": {"kind": "slot", "params": dict(n_slots=32, n_heads=4, chunk_size=4)},
            "surprise": {"kind": "surprise",
                         "params": dict(d_key=64, d_val=64, chunk_size=4, inner_lr=0.5)},
            "knn": {"kind": "knn", "params": dict(capacity=8192, d_key=64, topk=16)},
        },
    ),
    "small": dict(
        vocab_size=128, d_model=256, n_layers=6, n_heads=4, d_ff=512,
        seq_len=128, n_pairs=32, n_queries=8,
        steps=15000, batch_size=128, lr=1.5e-3, warmup=800,
        eval_every=1500, eval_batches=16, log_every=250, ckpt_every=500,
        memory_layer=3, controls_for=["product_key"],
        arms={
            "none": {"kind": "none", "params": {}},
            "product_key": _pkm(64),
            "slot": {"kind": "slot", "params": dict(n_slots=64, n_heads=4, chunk_size=16)},
            "surprise": {"kind": "surprise",
                         "params": dict(d_key=128, d_val=128, chunk_size=16, inner_lr=0.5)},
            "knn": {"kind": "knn", "params": dict(capacity=16384, d_key=128, topk=16)},
        },
    ),
    # The question neither laptop tier asks: what happens as the memory table grows
    # relative to the dense parameters. Everything else is held fixed and only n_keys
    # moves, from a table worth 0.3% of the dense weights to one worth 160% of them.
    # Each point carries its own matched controls, which is where the sparse-memory
    # asymmetry becomes impossible to ignore: matching 33M memory parameters with dense
    # width means a d_ff of ~3750 against a baseline of 1024.
    "cluster": dict(
        vocab_size=256, d_model=512, n_layers=8, n_heads=8, d_ff=1024,
        seq_len=256, n_pairs=64, n_queries=16,
        steps=30000, batch_size=128, lr=1e-3, warmup=1500,
        eval_every=3000, eval_batches=32, log_every=500, ckpt_every=500,
        memory_layer=4,
        controls_for=["pkm_k32", "pkm_k64", "pkm_k128", "pkm_k256", "pkm_k512"],
        arms={
            "none": {"kind": "none", "params": {}},
            "pkm_k32": _pkm(32),      # 1k slots,   0.1M memory params
            "pkm_k64": _pkm(64),      # 4k slots,   0.5M
            "pkm_k128": _pkm(128),    # 16k slots,  2.1M
            "pkm_k256": _pkm(256),    # 65k slots,  8.4M
            "pkm_k512": _pkm(512),    # 262k slots, 33.5M
        },
    ),
}


def make_config(scale: str, arm: str, seed: int, out_root: Path,
                steps: int | None = None) -> ExperimentConfig:
    s = SCALES[scale]
    spec = s["arms"][arm]
    kind, params = spec["kind"], spec["params"]
    cfg = ExperimentConfig.from_dict({
        "name": f"{scale}-{arm}-s{seed}",
        "model": {
            "vocab_size": s["vocab_size"], "d_model": s["d_model"],
            "n_layers": s["n_layers"], "n_heads": s["n_heads"], "d_ff": s["d_ff"],
            "max_seq_len": s["seq_len"],
            "memory": {
                "kind": kind,
                "layers": [] if kind == "none" else [s["memory_layer"]],
                "placement": "parallel",
                "params": params,
            },
        },
        "data": {"kind": "associative_recall", "seq_len": s["seq_len"],
                 "vocab_size": s["vocab_size"],
                 "params": {"n_pairs": s["n_pairs"], "n_queries": s["n_queries"]}},
        "train": {
            "steps": steps or s["steps"], "batch_size": s["batch_size"], "lr": s["lr"],
            "warmup": s["warmup"], "eval_every": s["eval_every"],
            "eval_batches": s["eval_batches"], "log_every": s["log_every"],
            "ckpt_every": s["ckpt_every"], "resume": True, "seed": seed,
            "out_dir": str(out_root / f"{arm}_s{seed}"),
        },
    })
    return cfg


def plan(scale: str, seeds: List[int], arms: List[str], out_root: Path,
         steps: int | None) -> List[ExperimentConfig]:
    """Seed-major: every arm and its matched controls at seed 0, then seed 1, ...

    Ordering matters because the sweep is expected to be interrupted. Arm-major
    order spends the first half of the budget putting three seeds on three arms and
    none on the rest, so a sweep stopped at the halfway point yields nothing
    comparable. Seed-major order means every prefix of the run list is a complete
    sweep at some seed count.
    """
    runs = []
    for seed in seeds:
        for arm in arms:
            runs.append(make_config(scale, arm, seed, out_root, steps))
        bases = [b for b in SCALES[scale].get("controls_for", []) if b in arms]
        for base_arm in bases:
            base = make_config(scale, base_arm, seed, out_root, steps)
            # With a single control base there is nothing to disambiguate, so the name
            # stays short; with several the base arm has to be in the name or the
            # directories collide.
            tag = "" if len(bases) == 1 else f"_{base_arm}"
            for match in ("params", "flops"):
                ctrl, _ = matched_baseline_config(base, match)
                ctrl.name = f"{scale}-matched_{match}{tag}-s{seed}"
                ctrl.train.out_dir = str(out_root / f"matched_{match}{tag}_s{seed}")
                ctrl.train.resume = True
                runs.append(ctrl)
    return runs


def run_one(cfg: ExperimentConfig) -> Dict:
    out = Path(cfg.train.out_dir)
    final = out / "final.json"
    if final.exists():
        print(f"[skip] {cfg.name} already finished", flush=True)
        return json.loads(final.read_text())

    t0 = time.time()
    tr = Trainer(cfg)
    res = tr.train()
    record = {"name": cfg.name, "seed": cfg.train.seed,
              "memory_kind": cfg.model.memory.kind, "d_ff": cfg.model.d_ff,
              **tr.summary(), **res, "wall_s": round(time.time() - t0, 1)}

    if cfg.model.memory.kind != "none":
        ra = read_ablation(tr.model, tr.val_data, cfg.train.eval_batches,
                           cfg.train.batch_size, cfg.train.seed + 10_000, tr.device)
        (out / "ablation.json").write_text(json.dumps(ra, indent=2, default=float))
        record["ablation_delta_loss"] = ra["delta_loss"]
        record["ablation_delta_acc"] = ra["delta_acc"]

    final.write_text(json.dumps(record, indent=2, default=float))
    print(f"[done] {cfg.name}: val_loss {record['val_loss']:.4f} "
          f"val_acc {record['val_acc']:.4f} in {record['wall_s']:.0f}s", flush=True)
    return record


def summarise(records: List[Dict], out_root: Path) -> None:
    by_arm: Dict[str, List[Dict]] = {}
    for r in records:
        key = r["name"].split("-")[1]
        by_arm.setdefault(key, []).append(r)

    def agg(vals):
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rows = []
    for arm, rs in by_arm.items():
        loss_m, loss_s = agg([r["val_loss"] for r in rs])
        acc_m, acc_s = agg([r["val_acc"] for r in rs])
        dl = [r["ablation_delta_loss"] for r in rs if "ablation_delta_loss" in r]
        dl_m, dl_s = agg(dl)
        rows.append({"arm": arm, "n_seeds": len(rs),
                     "params": rs[0]["params_total"],
                     "flops_per_token": rs[0]["flops_per_token"],
                     "val_loss_mean": loss_m, "val_loss_std": loss_s,
                     "val_acc_mean": acc_m, "val_acc_std": acc_s,
                     "ablation_delta_loss_mean": dl_m, "ablation_delta_loss_std": dl_s})
    rows.sort(key=lambda r: r["val_loss_mean"])
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2, default=float))

    print("\n| arm | seeds | params | FLOPs/token | val loss | val acc | ablation delta loss |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        d = ("n/a" if r["ablation_delta_loss_mean"] is None
             else f"{r['ablation_delta_loss_mean']:+.4f} ± {r['ablation_delta_loss_std']:.4f}")
        print(f"| `{r['arm']}` | {r['n_seeds']} | {r['params']/1e6:.2f}M | "
              f"{r['flops_per_token']/1e6:.2f}M | "
              f"{r['val_loss_mean']:.4f} ± {r['val_loss_std']:.4f} | "
              f"{r['val_acc_mean']:.4f} ± {r['val_acc_std']:.4f} | {d} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="small", choices=sorted(SCALES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--steps", type=int, default=None, help="override the scale's steps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = (list(SCALES[args.scale]["arms"]) if args.arms == "all"
            else args.arms.split(","))
    out_root = Path(args.out or f"runs/sweep_{args.scale}")
    out_root.mkdir(parents=True, exist_ok=True)

    runs = plan(args.scale, seeds, arms, out_root, args.steps)
    manifest = [{"name": c.name, "out_dir": c.train.out_dir, "steps": c.train.steps,
                 "memory_kind": c.model.memory.kind, "d_ff": c.model.d_ff,
                 "seed": c.train.seed} for c in runs]
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    done = sum(1 for c in runs if (Path(c.train.out_dir) / "final.json").exists())
    print(f"{len(runs)} runs planned at scale {args.scale} "
          f"({len(arms)} arms x {len(seeds)} seeds + matched controls); "
          f"{done} already finished", flush=True)
    bt = budget_table(make_config(args.scale, "product_key", seeds[0], out_root, args.steps))
    print("product-key budget:", json.dumps({k: round(v, 4) for k, v in bt.items()}), flush=True)
    for m in manifest:
        print(f"  {m['name']:34s} steps={m['steps']:>6} d_ff={m['d_ff']:>5}", flush=True)
    if args.dry_run:
        return

    records = [run_one(c) for c in runs]
    summarise(records, out_root)


if __name__ == "__main__":
    main()
