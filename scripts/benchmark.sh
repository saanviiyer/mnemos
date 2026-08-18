#!/usr/bin/env bash
# The headline comparison: every memory kind on the same task, same budget-free
# settings, plus the matched-baseline + read-ablation report for product-key memory.
# About 45 minutes on an M-series laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/bench
for cfg in configs/tiny_none.yaml configs/tiny_pkm.yaml configs/tiny_slot.yaml \
           configs/tiny_surprise.yaml configs/tiny_knn.yaml; do
  n=$(basename "$cfg" .yaml)
  echo "=== $n ==="
  python -m mnemos.cli train --config "$cfg" --set "train.out_dir=runs/bench/$n" \
    --out "runs/bench/$n.json"
done
echo "=== product-key ablation (matched baselines + read ablation) ==="
python -m mnemos.cli ablate --config configs/tiny_pkm.yaml \
  --set "train.out_dir=runs/bench/pkm_ablate" --out runs/bench/pkm_ablation.json
