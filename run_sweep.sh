#!/usr/bin/env bash
# Relaunch this after any interruption. Finished runs are skipped, a run killed
# mid-flight resumes from its last checkpoint.
cd "$(dirname "$0")"
exec python3 -u scripts/sweep.py --scale small --seeds 0,1,2 --steps 10000 \
  --out runs/sweep_small
