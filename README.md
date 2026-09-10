# reBot Arm B601-DM — Genesis VLA testbed

Genesis simulation of the [Seeed reBot Arm B601-DM](https://github.com/Seeed-Projects/reBot-DevArm), turned into a
testbed for vision-language-action policies. The task is fixed: **pick up the red cube and place it in the green
square**. The cube's xy position and yaw are randomized on every reset; the target zone is fixed. A classical
`pyroboplan` oracle solves the task to collect a demonstration dataset, and that dataset is used to fine-tune
[SmolVLA](https://huggingface.co/lerobot/smolvla_base).

## Quick start

```bash
pixi install
pixi run test                     # 19 tests, ~2 min (Genesis JIT-builds kernels + a 20-seed oracle check)
pixi run python sim.py            # bare viewer smoke test: drives the arm through waypoints, no policy
```

Everything below runs from the repo root. The first Genesis scene build in a process takes ~45 s (finger mesh
decomposition); subsequent resets are fast.

## Watching the policies

Both commands open the Genesis viewer plus a second window showing the two policy cameras side by side
(front | wrist), refreshed on every observation. Viewer controls: left-drag orbits, scroll zooms, right-drag pans.

```bash
# Oracle (scripted expert). Nothing is saved. Prints success/fail per seed.
pixi run python scripts/watch_oracle.py --episodes 5 --seed 0

# Trained SmolVLA. OUTPUT_DIR keeps it from overwriting the recorded benchmark in outputs/eval/smolvla_rebot.
OUTPUT_DIR=outputs/eval/watch scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 5 --env.show_viewer=true
```

- The cube's xy/yaw come from the episode seed. `lerobot-eval` uses `seed + episode_index` with `--seed=1000` by
  default (training used seeds 0-199). To replay one specific layout: `... 1 --env.show_viewer=true --seed=1003`.
  From the recorded 84% run, seeds 1008, 1010, 1011, 1012, 1025, 1027, 1031, 1035 failed; the rest of 1000-1049
  succeeded. SmolVLA samples noise per action chunk, so the same seed can still go either way on replay.
- Each episode ends on success or after 300 steps (30 s at 10 Hz); the next episode resets in the same window.
- Without `--env.show_viewer=true`, `eval.sh` runs headless and still writes an mp4 per episode to
  `<OUTPUT_DIR>/videos/rebot_0/eval_episode_<i>.mp4` (front camera) plus `eval_info.json` (`pc_success`).
- Any extra `eval.sh` arguments after `[checkpoint] [n_episodes]` go straight to `lerobot-eval`, e.g.
  `--policy.n_action_steps=10` to replan the action chunk more often (default 50).

## Training a policy

The pipeline is collect -> train -> eval. Each step is its own script; `scripts/cycle.sh` chains them.

```bash
# 1. Collect oracle demonstrations as a LeRobot v3 dataset (~3 s/episode). Recreates --root from scratch.
pixi run python scripts/collect.py --episodes 200 --root data/rebot_pick_place --seed 0

# 2. Fine-tune lerobot/smolvla_base (~75 min for 20k steps, ~3.2 GB VRAM at batch 8 on an RTX 5070 Laptop).
#    Checkpoints land in outputs/train/smolvla_rebot/checkpoints/{005000,...,last}/pretrained_model.
BATCH_SIZE=8 STEPS=20000 scripts/train.sh

# 3. Evaluate a checkpoint on 50 held-out seeds (~2 min headless).
scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 50

# All three in one go (refuses to overwrite an existing dataset/checkpoint dir; move them aside first).
for d in data/rebot_pick_place outputs/train/smolvla_rebot outputs/eval/smolvla_rebot; do mv "$d" "${d}_old"; done
mkdir -p outputs/logs && scripts/cycle.sh 200 50 > outputs/logs/cycle.log 2>&1 &
tail -f outputs/logs/cycle.log         # stage markers + final pc_success; per-stage logs in outputs/logs/
```

`train.sh` and `eval.sh` forward extra arguments to `lerobot-train` / `lerobot-eval`, so any lerobot flag the
script does not already set can be appended (see `lerobot-train --help`).

### Tuning knobs

Which script you must re-run depends on what you change:

| Change | Where | Then re-run |
|---|---|---|
| Number of demos, collection seeds | `collect.py --episodes/--seed` | collect, train, eval |
| Camera pose / FOV, lighting, room, cube/zone geometry | `rebot_sim/scene.py` (`CAM_*`, `LIGHT_DIR`, `AMBIENT`, `CUBE_*_RANGE`, `ZONE_XY`), `rebot_sim/env.py` (`WRIST_CAM_*`, `image_size`) | collect, train, eval (observations change) |
| Oracle behaviour: hover pose, approach height, gripper hold length, grip force | `rebot_sim/oracle.py` (`HOVER`, `HOVER_PITCH`, `APPROACH`, `GRIPPER_SETTLE_STEPS`, `GRIP_CLOSED`) | collect, train, eval |
| Training length, batch size, LR, save frequency | `STEPS`, `BATCH_SIZE` env vars or lerobot flags to `train.sh` | train, eval |
| Action chunking at inference | `--policy.n_action_steps=<k>` to `eval.sh` (k <= 50) | eval only |
| Episode length, success threshold | `lerobot_env.py` (`episode_length`), `env.py` (`LIFT_HEIGHT`) | eval only (dataset unaffected) |

Anything that changes what the cameras see or what the oracle does invalidates the dataset and checkpoints;
keep old runs by renaming their directories (this repo keeps the first cycle as `*_oldcam`).

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
- `watch_oracle.py` — runs the oracle in the Genesis viewer, nothing saved.
- `cycle.sh` — collect -> train -> eval in one go, with per-stage logs under `outputs/logs/`.

## Dataset

`scripts/collect.py` collected 200 successful episodes (15,527 frames, 10 fps, LeRobot v3 format, ~62 MB) to
`data/rebot_pick_place`; the oracle succeeded on 200/200 attempted seeds.

## Baseline results

| Policy                | Success rate                          | How |
|------------------------|----------------------------------------|-----|
| Oracle (pyroboplan)     | 20/20 (tests), 200/200 (collection)    | `pixi run test`, `scripts/collect.py` |
| SmolVLA (fine-tuned)    | 42/50 = 84% (200 demos, 20k steps)     | `scripts/train.sh` + `scripts/eval.sh` |

Success requires the cube to have been lifted at least 4 cm at some point, to rest inside the zone, and the
gripper to have released it (`LIFT_HEIGHT` in `rebot_sim/env.py`), so shoving the cube into the zone does not
count. The SmolVLA number is from `scripts/eval.sh` on `checkpoints/last` (seeds 1000+, disjoint from the
training seeds 0-199); the 10k-step checkpoint scored 3/10 on a quick interim check.

## Notes

- The 84% replaced an earlier 38% from the same recipe. Three changes between them: front camera moved from
  2.5 m to 1.3 m (cube went from a few pixels to clearly visible), the oracle now visits a fixed hover pose so the
  wrist camera sees the cube before any cube-dependent motion, and near-overhead lighting removed a wall shadow
  across half the table. They were changed together, so their individual contributions are not separated.
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
