#!/usr/bin/env bash
# Every memory kind, a handful of steps each. Should finish in a couple of minutes on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in configs/tiny_*.yaml; do
  echo "=== $cfg ==="
  python -m mnemos.cli train --config "$cfg" \
    --set train.steps=20 train.eval_batches=2 train.log_every=10 \
          "train.out_dir=runs/smoke/$(basename "$cfg" .yaml)"
done
