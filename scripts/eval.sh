#!/usr/bin/env bash
# Evaluate a checkpoint in the Genesis env with stock lerobot-eval.
# Usage: scripts/eval.sh [checkpoint_dir] [n_episodes]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/train/smolvla_rebot/checkpoints/last/pretrained_model}"
N="${2:-50}"
# The lerobot-eval console script does not put the repo root on sys.path; the plugin import needs it.
export PYTHONPATH="${PYTHONPATH:-}:."
# smolvla_base was pretrained with fixed camera{1,2,3} slots; the checkpoint's config.json
# (loaded verbatim by make_policy) still declares those keys, so lerobot_eval's pre-load
# feature validator rejects our env's observation.images.front/wrist before the checkpoint's
# saved policy_preprocessor.json rename ever runs. Passing --rename_map here (matching
# scripts/train.sh) renames the env features and, per lerobot/policies/factory.py, skips
# that validator. camera3 is simply left unused.
exec pixi run lerobot-eval \
  --env.type=rebot \
  --env.discover_packages_path=rebot_sim \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --eval.n_episodes="$N" \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --output_dir="${OUTPUT_DIR:-outputs/eval/smolvla_rebot}"
