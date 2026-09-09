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
# Genesis's gs.init() unconditionally calls torch.set_default_device("cuda") (see
# genesis/__init__.py, "Update torch default dtype and device, just in case") because ~30
# call sites inside genesis itself create tensors (e.g. torch.zeros(...)) without an explicit
# device= and rely on that default to land on the GPU alongside the sim state. That global
# default leaks into lerobot_eval.py, which has several bare torch.tensor(...)/torch.arange(...)
# calls (no device=) that assume they land on cpu to match the rest of its cpu-resident rollout
# bookkeeping (built via torch.from_numpy, always cpu). Two crashes result:
#   TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy
#   the tensor to host memory first.                                    (rollout() progress bar)
#   RuntimeError: Expected all tensors to be on the same device, but found at least two
#   devices, cuda:0 and cpu!                                       (eval_policy() done-mask calc)
# We can't fix this in rebot_sim (it's Genesis's behavior) and can't reset torch's global
# default device back to cpu (that would break Genesis's own unqualified tensor ops during
# env.step()). Instead we monkeypatch torch.tensor/torch.arange to force device="cpu" only
# when called with no explicit device *from within the lerobot.scripts.lerobot_eval module
# itself* (checked via the caller's frame) -- this leaves every other caller, including
# Genesis and the policy/preprocessor code (which explicitly places tensors via .to(device)
# regardless of the global default), completely untouched.
exec pixi run python -c '
import sys, torch
sys.argv = ["lerobot-eval"] + sys.argv[1:]

import lerobot.scripts.lerobot_eval as _eval_mod

def _cpu_when_called_from_lerobot_eval(orig_fn):
    def wrapper(*args, **kwargs):
        if "device" not in kwargs and sys._getframe(1).f_globals.get("__name__") == "lerobot.scripts.lerobot_eval":
            kwargs["device"] = "cpu"
        return orig_fn(*args, **kwargs)
    return wrapper

# Only patch torch.tensor/torch.arange to default to cpu (see above) *after* make_env()
# returns: building the Genesis env lazily imports several genesis submodules that use
# @torch.jit.script, and TorchScript needs real, inspectable source for torch.arange/
# torch.tensor to compile those -- our wrapper closures break that. Patching after env
# construction avoids ever touching those names during Genesis's lazy JIT compilation.
_orig_make_env = _eval_mod.make_env
def _make_env_then_patch(*args, **kwargs):
    envs = _orig_make_env(*args, **kwargs)
    torch.tensor = _cpu_when_called_from_lerobot_eval(torch.tensor)
    torch.arange = _cpu_when_called_from_lerobot_eval(torch.arange)
    return envs
_eval_mod.make_env = _make_env_then_patch

sys.exit(_eval_mod.main())
' \
  --env.type=rebot \
  --env.discover_packages_path=rebot_sim \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --eval.n_episodes="$N" \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --output_dir="${OUTPUT_DIR:-outputs/eval/smolvla_rebot}"
