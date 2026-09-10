# reBot Arm B601-DM — Genesis VLA testbed

Genesis simulation of the [Seeed reBot Arm B601-DM](https://github.com/Seeed-Projects/reBot-DevArm), turned into a
testbed for vision-language-action policies. The task is fixed: **pick up the red cube and place it in the green
square**. The cube's xy position and yaw are randomized on every reset; the target zone is fixed. A classical
`pyroboplan` oracle solves the task to collect a demonstration dataset, and that dataset is used to fine-tune
[SmolVLA](https://huggingface.co/lerobot/smolvla_base).

## Quick start

```bash
pixi install

# Viewer / smoke test
pixi run python sim.py            # interactive viewer, drives the arm through waypoints
pixi run python sim.py --headless # no viewer, saves frame.png, asserts waypoint tracking

# Tests (16 tests, ~2 min: Genesis JIT-builds its kernels + a 20-seed oracle success-rate check)
pixi run test

# Collect an oracle demonstration dataset (writes a LeRobot v3 dataset, recreating --root)
pixi run python scripts/collect.py --episodes 200 --root data/rebot_pick_place

# Fine-tune SmolVLA on the collected dataset (batch 8 uses ~3.2 GB VRAM at ~4.5 step/s on an RTX 5070 Laptop)
BATCH_SIZE=8 STEPS=20000 scripts/train.sh

# Evaluate a checkpoint in-sim
scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 50

# Watch the policy live in the Genesis viewer (extra args are passed through to lerobot-eval)
scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 5 --env.show_viewer=true

# Watch the oracle (scripted expert) live in the Genesis viewer, nothing saved
pixi run python scripts/watch_oracle.py --episodes 5 --seed 0
```

## Observation / action space

- Observations: `pixels/front` and `pixels/wrist`, both 256x256 RGB, plus `agent_pos`, a 7-d vector (6 joint
  positions + gripper opening).
- Actions: 7-d — 6 absolute joint targets + gripper opening — applied at 10 Hz.

## Layout

`rebot_sim/`
- `__init__.py` — registers `gym_rebot/RebotPickPlace-v0` and, when `lerobot` is installed, the `rebot` EnvConfig.
- `scene.py` — shared constants (joint limits, task geometry, colors) and Genesis scene builders (room, arm, cube,
  target zone) used by the viewer, the gym env, and the oracle.
- `env.py` — `RebotPickPlaceEnv`, the gymnasium env: reset/step, camera rendering, success check.
- `oracle.py` — classical pick-and-place planner: Pinocchio IK (`pyroboplan`) + RRT-Connect for free-space moves,
  Cartesian straight-line segments for the grasp/place approach and retreat. Every episode first visits a fixed
  tilted "hover" pose from which the wrist camera sees the whole cube range, so demos contain a canonical
  look-then-reach frame for the VLA to localize from.
- `lerobot_env.py` — `RebotEnv` `EnvConfig` so `lerobot-eval --env.type=rebot` can drive the gym env.

`scripts/`
- `collect.py` — runs the oracle through the env and writes successful episodes as a LeRobot v3 dataset.
- `train.sh` — fine-tunes `lerobot/smolvla_base` on the collected dataset.
- `eval.sh` — evaluates a checkpoint in the Genesis env via stock `lerobot-eval`.

## Dataset

`scripts/collect.py` collected 200 successful episodes (12,110 frames, 10 fps, LeRobot v3 format, ~29 MB) to
`data/rebot_pick_place`; the oracle succeeded on 200/200 attempted seeds.

## Baseline results

| Policy                | Success rate                          | How |
|------------------------|----------------------------------------|-----|
| Oracle (pyroboplan)     | 20/20 (tests), 200/200 (collection)    | `pixi run test`, `scripts/collect.py` |
| SmolVLA (fine-tuned)    | 19/50 = 38% (200 demos, 20k steps)     | `scripts/train.sh` + `scripts/eval.sh` |

Success requires the cube to have been lifted at least 4 cm at some point, to rest inside the zone, and the
gripper to have released it (`LIFT_HEIGHT` in `rebot_sim/env.py`), so shoving the cube into the zone does not
count. The SmolVLA number is from `scripts/eval.sh` on `checkpoints/last` (seeds 1000+, disjoint from the
training seeds 0-199); the 10k-step checkpoint scored 3/10 on a quick interim check.

## Notes

- `assets/rebot_arm_dm/` is `Rebot_Arm_description/DM/` from upstream (CERN-OHL-W v2, see LICENSE there). Local
  change: added `<inertial>` to both finger links (upstream omits them; Genesis' MuJoCo parser rejects massless
  moving bodies).
- 8 dofs: joint1..6 + finger_left/finger_right. Genesis ignores the URDF `<mimic>`, so drive right = -left.
- joint2/joint3 limits are [-3.14, 0]; "up" is negative.
- Pinocchio needs absolute mesh paths (`oracle.load_model` rewrites the URDF's `../meshes` before parsing) and
  there is no SRDF for this arm, so the oracle filters out adjacent (same/parent/sibling joint) collision pairs
  itself instead of loading one.
- Genesis defaults robot meshes to one convex hull per mesh, which turns each gripper finger into a wedge that
  ejects the cube instead of pinching it; `scene.add_arm` passes `decompose_robot_error_threshold=0.15` (Genesis'
  own default for non-robot rigid bodies) to restore flat, parallel finger pads.
- `gs.init()` installs a global torch default-device mode (CUDA) that rewrites device-less calls like
  `torch.as_tensor` in third-party code and breaks lerobot's processors; `scene.init_genesis` removes that mode
  right after Genesis installs it.
- `lerobot/smolvla_base` hardcodes 3 camera input slots named `camera1/2/3`; `train.sh` and `eval.sh` both pass
  `--rename_map` to map our `front`/`wrist` keys onto `camera1`/`camera2` (the unused `camera3` slot is simply
  dropped).
- The `lerobot-eval` console script doesn't put the repo root on `sys.path`, so the `rebot_sim` plugin import
  would fail without it; `eval.sh` exports `PYTHONPATH=.` before invoking it.
- A few tunables are marked `ponytail:` in the code and may need retuning against real hardware: `TCP_OFFSET`
  and `GRIP_CLOSED` in `oracle.py`, and the PD gains in `scene.set_arm_gains`.
- `pixi run test` takes ~2 min: most of that is Genesis JIT-building its kernels plus the 20-seed oracle
  success-rate test.
