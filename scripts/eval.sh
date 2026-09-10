#!/usr/bin/env bash
# Evaluate a checkpoint in the Genesis env with stock lerobot-eval.
# Usage: scripts/eval.sh [checkpoint_dir] [n_episodes] [extra lerobot-eval flags...]
#   e.g. scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 5 --env.show_viewer=true
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/train/smolvla_rebot/checkpoints/last/pretrained_model}"
N="${2:-50}"
shift $(( $# > 2 ? 2 : $# ))
# The lerobot-eval console script does not put the repo root on sys.path; the plugin import needs it.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
# Same rename as scripts/train.sh (see there): the checkpoint keeps smolvla_base's camera1/2/3 slot names.
# Genesis cannot run inside AsyncVectorEnv subprocesses; batch size must stay 1.
exec pixi run lerobot-eval \
  --env.type=rebot \
  --env.discover_packages_path=rebot_sim \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --eval.n_episodes="$N" \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --output_dir="${OUTPUT_DIR:-outputs/eval/smolvla_rebot}" \
  "$@"
