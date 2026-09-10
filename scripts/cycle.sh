#!/usr/bin/env bash
# Full retraining cycle: collect demos -> fine-tune SmolVLA -> evaluate. ~90 min on an RTX 5070 Laptop.
# Usage: scripts/cycle.sh [n_demos] [n_eval_episodes]     (env: STEPS, BATCH_SIZE pass through to train.sh)
# Refuses to run if data/rebot_pick_place or outputs/train/smolvla_rebot exist: move or delete them first
# (e.g. `mv data/rebot_pick_place data/rebot_pick_place_old`) so a previous run is never silently overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
DEMOS="${1:-200}"
EVAL_N="${2:-50}"
for d in data/rebot_pick_place outputs/train/smolvla_rebot outputs/eval/smolvla_rebot; do
  [ -e "$d" ] && { echo "refusing to overwrite $d; move it aside first" >&2; exit 1; }
done
mkdir -p outputs/logs
echo "=== collect start $(date)"
pixi run python scripts/collect.py --episodes "$DEMOS" > outputs/logs/collect.log 2>&1
echo "=== collect done $(date): $(grep -c success outputs/logs/collect.log) successes, $(grep -c failed outputs/logs/collect.log) failed"
echo "=== train start $(date)"
scripts/train.sh > outputs/logs/train.log 2>&1
echo "=== train done $(date)"
echo "=== eval start $(date)"
scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model "$EVAL_N" > outputs/logs/eval.log 2>&1
echo "=== eval done $(date)"
grep -o "'pc_success': [0-9.]*" outputs/logs/eval.log | tail -1
