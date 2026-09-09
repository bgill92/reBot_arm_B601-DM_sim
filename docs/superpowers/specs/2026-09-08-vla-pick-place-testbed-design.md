# VLA pick-and-place testbed — design

Date: 2026-09-08

## Goal

Turn this Genesis sim of the reBot Arm B601-DM into a testbed for fine-tuning
vision-language-action (VLA) models. Task: pick up a cube from a random spot on
the table and place it in a fixed marked zone. Demonstrations come from a
classical pick-and-place oracle (pyroboplan). Training and evaluation use
LeRobot with SmolVLA.

Decisions made:

| Question | Decision |
|---|---|
| VLA stack | LeRobot + SmolVLA (~450M params, fits 8 GB VRAM) |
| Scope | Sim-only. Keep obs/action spec real-robot-compatible, no hardware work. |
| Oracle | pyroboplan (Pinocchio IK + RRT-Connect + Cartesian planner) |
| Cameras | Fixed third-person + wrist cam on `link6`, 256x256 RGB |
| Action space | 7-d absolute joint positions: q1..q6 + gripper opening |
| Variation | Random cube xy/yaw, fixed target zone, one instruction |
| Architecture | Gymnasium env shared by oracle and VLA; stock LeRobot CLIs |

Hardware: RTX 5070 Laptop 8 GB, 16 CPU cores, CUDA 12.8, Python 3.12, pixi.

## pyroboplan viability (verified 2026-09-08)

Smoke test on `assets/rebot_arm_dm/urdf/ReBot_Arm_DM.urdf` with pyroboplan
1.3.1 / pin 4.0.0 passed. Two patches required:

1. Pinocchio rejects relative mesh paths (`../meshes/...`). Load the URDF as a
   string, replace `../meshes` with the absolute meshes dir, build with
   `pin.buildModelFromXML` + `pin.buildGeomFromUrdfString(..., COLLISION)`.
2. No SRDF, so `addAllCollisionPairs()` flags the home pose as self-colliding.
   Remove pairs whose geometries sit on the same joint, parent/child joints,
   or sibling joints. Leaves 32 pairs; home pose is then collision-free.

After patches: `DifferentialIk` on `link6` converges; bidirectional
RRT-Connect plans between configs.

Dependency cost: pyroboplan pins `pin==4.0.0`, `matplotlib==3.10.9`,
`scipy<=1.17.1`, `toppra==0.6.3`, `meshcat`, `drake` (44 MB, only used by
`trajectory_optimization`, which we do not use). Current lock has matplotlib
3.11.1 and scipy 1.18.1; pixi should downgrade. Fallback: install pyroboplan
with `--no-deps` plus `pin` and `toppra`.

## 1. Task and environment — `rebot_sim/env.py`

`RebotPickPlaceEnv(gymnasium.Env)` wraps the Genesis scene.

Scene (built from `rebot_sim/scene.py`, extracted from `sim.py`):

- Existing room, table, arm at `ARM_POS`.
- Cube: 3 cm rigid box, red.
- Target zone: 10x10 cm flat green box, fixed, visual only (no collision).
  Fixed position on the table, e.g. (0.25, -0.25) relative to arm base.
- Cameras: `front` fixed at the existing viewer pose; `wrist` attached to
  `link6` via `cam.attach(link, offset_T)`. Both 256x256 RGB, no GUI.

`reset(seed)`:

- Cube xy uniform over the reachable region in front of the arm, roughly
  x in [0.15, 0.35], y in [-0.15, 0.15] relative to base; yaw uniform.
- Arm to `HOME`, gripper open. Step the sim ~20 times to settle.

Observation dict:

- `observation.images.front`, `observation.images.wrist`: uint8 HxWx3.
- `observation.state`: float32[7] = q1..q6, gripper opening (left finger).
- Instruction (constant, stored as dataset task): `"pick up the red cube and
  place it in the green square"`.

`step(action)`:

- `action`: float32[7] absolute joint targets, gripper opening in [0, 0.05].
- Writes `control_dofs_position` for arm dofs and `[g, -g]` for fingers, then
  runs 10 sim substeps (dt=0.01) so control rate is 10 Hz.
- Reward 1.0 on success else 0.0. `terminated` on success. `truncated` after
  300 steps.
- Success: cube center within zone xy +-5 cm, cube z at table height
  (+-1 cm), gripper opening > 0.03.

Registered as `gym_rebot/RebotPickPlace-v0` via `gymnasium.register` at
package import.

## 2. Oracle — `rebot_sim/oracle.py`

Pinocchio model built once per process using the two patches above. Table
added to the collision model as an `hppfcl.Box`. Cube added as a box whose
collision pair is enabled before grasp and disabled once grasped.

Waypoints per episode, all top-down grasps (gripper z pointing down), read
from privileged sim state (cube pose):

1. pre-grasp: cube position + 8 cm z
2. grasp: cube position, then close gripper
3. lift: grasp + 8 cm z
4. pre-place: zone center + 8 cm z
5. place: zone center, then open gripper
6. retreat: place + 8 cm z

Segment planning:

- Free-space moves (home->1, 3->4, 5->6): `DifferentialIk` for the goal
  config, `RRTPlanner` (RRT-Connect, bidirectional), `path_shortcutting`,
  then `CubicPolynomialTrajectory` sampled at 10 Hz.
- Vertical moves (1->2, 2->3, 4->5): `CartesianPlanner` straight-line in
  SE3, IK per sample, sampled at 10 Hz.

Output: list of 7-d actions. Gripper open = 0.05, closed = value tuned to
grip the 3 cm cube (start at 0.012; a `GRIP_CLOSED` constant with a
`ponytail:` note since real hardware will need tuning).

Grasp verification: after close + lift, cube z must rise by >4 cm, else the
episode is a failure. Env success flag is the final check. Failed episodes
are discarded by the collector.

## 3. Data collection — `scripts/collect.py`

```
for i in range(N):
    obs = env.reset(seed=i)
    actions = oracle.plan(env)          # uses privileged cube pose
    for a in actions:
        record frame {obs, a, task}
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc: break
    if term: dataset.save_episode() else discard
```

Dataset: `LeRobotDataset.create(repo_id="local/rebot_pick_place", fps=10,
features={images..., observation.state, action})`, LeRobot v3 format, images
encoded as video. Task string = instruction.

Initial target: 200 successful episodes (~100 frames each, ~20k frames).
Planner is CPU-bound and single-threaded; expect minutes to a couple of
hours. Sanity check with `lerobot-dataset-viz`.

## 4. Training and evaluation

Train (`scripts/train.sh`):

```
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/rebot_pick_place \
  --policy.device=cuda --batch_size=8 --steps=20000
```

SmolVLA accepts two cameras (pads the third). Drop batch size to 4 if 8 GB
OOMs.

Eval (`scripts/eval.sh`): register `RebotEnvConfig(EnvConfig)` in
`rebot_sim/lerobot_env.py` so stock `lerobot-eval --env.type=rebot
--policy.path=<checkpoint>` runs 50 episodes and reports success rate plus
videos. Fallback if the LeRobot env plugin path fights: a ~60-line
`scripts/eval.py` calling `policy.select_action` in the gym loop.

Metric: env `terminated` rate over 50 seeds. Oracle success rate is the
ceiling.

## 5. Layout, dependencies, phases

```
rebot_sim/
  __init__.py     # gym registration
  scene.py        # room/table/arm/cube/zone builders + constants
  env.py          # RebotPickPlaceEnv
  oracle.py       # pinocchio load, collision filtering, planner pipeline
  lerobot_env.py  # EnvConfig subclass for lerobot-eval
scripts/
  collect.py
  train.sh
  eval.sh
tests/
  test_env.py     # headless: obs shapes, teleport cube -> success
  test_oracle.py  # headless: 20 seeds, >=90% success
sim.py            # viewer, thin wrapper over rebot_sim.scene
docs/superpowers/specs/
```

Dependencies added to `pixi.toml` pypi-dependencies: `pyroboplan`,
`lerobot[smolvla]`, `gymnasium`, `pytest`. Risk: LeRobot's torch range vs
conda `pytorch-gpu`, and pyroboplan's exact pins. Mitigation: a pixi
`[feature.train]` environment holding lerobot if a single env fails to solve.

Phases (one commit each, each leaves a passing test):

0. Deps solve; `pixi run python sim.py --headless` still passes.
1. Scene refactor, cube, zone, cameras, gym env. `tests/test_env.py`.
2. Oracle. `tests/test_oracle.py`.
3. Collect 200 episodes; visualize.
4. Train SmolVLA; eval 50 episodes; record baseline success rate in README.

Out of scope: sim-to-real, multiple cubes, language variation, randomized
target zone, batched Genesis envs.
