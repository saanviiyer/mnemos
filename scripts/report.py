#!/usr/bin/env python3
"""Turn runs/bench/*.json into the markdown tables in RESULTS.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["tiny_none", "tiny_pkm", "tiny_slot", "tiny_surprise", "tiny_knn"]
LABEL = {"tiny_none": "none (control)", "tiny_pkm": "product_key", "tiny_slot": "slot",
         "tiny_surprise": "surprise", "tiny_knn": "knn"}


def fmt(n: float) -> str:
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}k"


def main(bench: Path) -> None:
    rows = []
    for name in ORDER:
        p = bench / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        # the memory can live in any block, so find the read_norm key rather than
        # hardcoding the layer index the shipped configs happen to use
        rn = next((v for k, v in r.items() if k.endswith("/read_norm")), None)
        rows.append((LABEL[name], r["params_total"], r["flops_per_token"],
                     r["val_loss"], r["val_acc"], rn))
    print("| memory | params | FLOPs/token | val loss | val acc | read norm |")
    print("|---|---|---|---|---|---|")
    for label, p, f, loss, acc, rn in rows:
        rn_s = "n/a" if rn is None else f"{rn:.3f}"
        print(f"| `{label}` | {fmt(p)} | {fmt(f)} | {loss:.3f} | {acc:.3f} | {rn_s} |")

    abl = bench / "pkm_ablation.json"
    if abl.exists():
        a = json.loads(abl.read_text())
        print("\n| arm | d_ff | params | FLOPs/token | val loss | val acc |")
        print("|---|---|---|---|---|---|")
        for key, label in [("memory_model", "product_key"),
                           ("baseline_params", "matched on params"),
                           ("baseline_flops", "matched on FLOPs")]:
            m = a.get(key)
            if not m:
                continue
            d_ff = m.get("d_ff_baseline", m.get("d_ff_memory_model", "-"))
            print(f"| {label} | {d_ff} | {fmt(m['params_total'])} | {fmt(m['flops_per_token'])} "
                  f"| {m['val_loss']:.3f} | {m['val_acc']:.3f} |")
        read_ablation_line(a["read_ablation"])
    else:
        solo = bench / "pkm_read_ablation.json"
        if solo.exists():
            read_ablation_line(json.loads(solo.read_text()))


def read_ablation_line(ra: dict) -> None:
    on, off = ra["memory_on"], ra["memory_off"]
    print(f"\nRead ablation, {int(on['scored_tokens'])} scored tokens: "
          f"val loss {on['val_loss']:.4f} -> {off['val_loss']:.4f} "
          f"(delta {ra['delta_loss']:+.4f}), "
          f"val acc {on['val_acc']:.4f} -> {off['val_acc']:.4f} "
          f"(delta {ra['delta_acc']:+.5f}).")
    diag = {k.split("/")[-1]: round(v, 3) for k, v in ra["diagnostics"].items()
            if not k.endswith("ablated")}
    print(f"Utilisation while the read was on: {diag}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/bench"))
