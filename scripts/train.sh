#!/usr/bin/env bash
# Fine-tune SmolVLA on the collected oracle dataset. ~8 GB VRAM: batch 8 bf16; drop to 4 on OOM.
#
# `lerobot/smolvla_base` was pretrained with 3 fixed camera slots named
# observation.images.camera{1,2,3}. `--policy.path` loads that pretrained
# config verbatim (its input_features are not rebuilt from our dataset), so
# without a rename_map the visual-feature validator rejects our dataset's
# `observation.images.front`/`.wrist` keys. --rename_map both renames our
# keys to match and (per lerobot/policies/factory.py) skips that validator.
# The unmapped camera3 slot is simply unused (empty_cameras=0 means it is
# dropped, not zero-padded). observation.state/action dims (7 here vs the
# base model's 6) are NOT affected by this: cfg.output_features is rebuilt
# from the dataset unconditionally, and state/action tensors are padded to
# max_state_dim/max_action_dim (32) at runtime regardless of declared shape.
set -euo pipefail
cd "$(dirname "$0")/.."
exec pixi run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/rebot_pick_place \
  --dataset.root=data/rebot_pick_place \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size="${BATCH_SIZE:-8}" \
  --steps="${STEPS:-20000}" \
  --save_freq=5000 \
  --output_dir=outputs/train/smolvla_rebot \
  --job_name=smolvla_rebot \
  --wandb.enable=false \
  "$@"
