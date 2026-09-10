# Handoff — VLA pick-and-place testbed

Last updated: 2026-09-10. Everything is on `main` (`vla-testbed` was fast-forward merged the same day and is
now a redundant pointer). Remote: `github.com/bgill92/reBot_arm_B601-DM_sim`. Working tree clean; all 19 tests
pass (`pixi run test`, ~2 min). Unrelated local branch `nyx` (path-traced rendering) is not pushed.

## Status in one paragraph

Sim, oracle, dataset collection, SmolVLA fine-tuning and in-sim eval all work end to end. The current policy
scores **84%** (42/50) on held-out seeds. That number came from one cycle in which three things changed at once
(closer front camera, oracle hover pose, overhead lighting), so nothing is known yet about which of them mattered
or why the remaining 8 episodes fail. Those two questions are the next steps.

## What exists

Genesis sim of the reBot Arm B601-DM turned into a VLA testbed. Task: pick up the red cube, place it in the
green square (cube xy/yaw random, zone fixed). See `README.md` for usage and `docs/superpowers/specs/` +
`docs/superpowers/plans/` for the original design/plan.

| Piece | File | Status |
|---|---|---|
| Scene constants/builders | `rebot_sim/scene.py` | done |
| Gym env (`gym_rebot/RebotPickPlace-v0`) | `rebot_sim/env.py` | done; obs `pixels/{front,wrist}` 256², `agent_pos`[7]; action 7-d abs joints + gripper @10 Hz; 300-step TimeLimit |
| Oracle (pyroboplan IK + RRT-Connect + Cartesian) | `rebot_sim/oracle.py` | done; 20/20 test seeds, 200/200 during collection |
| LeRobot EnvConfig plugin (`--env.type=rebot`) | `rebot_sim/lerobot_env.py` | done |
| Collector → LeRobot v3 | `scripts/collect.py` | done |
| Train / eval wrappers | `scripts/train.sh`, `scripts/eval.sh` | done |
| Full cycle script | `scripts/cycle.sh [n_demos] [n_eval]` | done; refuses to overwrite existing dataset/checkpoint dirs |
| Live viewers | `scripts/watch_oracle.py`, `scripts/eval.sh ... --env.show_viewer=true` | done; Genesis viewer + tkinter front\|wrist camera window |

## Results so far

Second cycle, 2026-09-10, after moving the front camera closer (2.5 m → 1.3 m), adding the oracle's fixed
tilted hover pose (wrist camera sees the cube before any cube-dependent motion) and overhead lighting.
Success went **38% → 84%** with the same 200 demos / 20k steps recipe. First-cycle artifacts kept as `*_oldcam`.

| Artifact | Location (git-ignored) | Numbers |
|---|---|---|
| Dataset | `data/rebot_pick_place` | 200 episodes, 15,527 frames, 10 fps, 62 MB, seeds 0–199, 200/200 oracle success |
| SmolVLA checkpoints | `outputs/train/smolvla_rebot/checkpoints/{005000,010000,015000,020000,last}` | 20k steps, batch 8, 75 min, loss 1.31 → 0.057 |
| Final eval | `outputs/eval/smolvla_rebot/eval_info.json` + videos | **42/50 = 84%** (lift-required success metric, eval seeds 1000+); failures at episodes 8, 10, 11, 12, 25, 27, 31, 35 |
| First cycle (old camera, no hover) | `data/rebot_pick_place_oldcam`, `outputs/{train,eval}/smolvla_rebot_oldcam` | 19/50 = 38%, 12,110 frames, loss 2.89 → 0.05 |
| Interim eval | `outputs/eval/ckpt10k` | 3/10 at 10k steps, first cycle, before the lift latch was added |
| Cycle logs | `outputs/logs/{cycle,collect,train,eval}.log` (chain script now tracked as `scripts/cycle.sh`) | full collect → train → eval chain |
| Smoke artifacts | `outputs/train/smoke`, `outputs/eval/smoke` | throwaway |

## Non-obvious things learned (all fixed in code, documented in README "Notes")

- pyroboplan `DifferentialIk.solve` and `CartesianPlanner.generate` mutate the start config **in place** → copy before calling (`oracle.py`).
- pyroboplan `CartesianPlanner` with trapezoidal scaling hits Slerp alpha=1.0000000000000002 → `use_trapezoidal_scaling=False`.
- Genesis default single-convex-hull finger meshes form a wedge that ejects the cube → `decompose_robot_error_threshold=0.15` on the arm URDF (`scene.add_arm`), +45 s cold build.
- `gs.init` installs a global torch default-device (CUDA) mode; it rewrites device-less `torch.as_tensor` in lerobot processors → `torch.set_default_device(None)` right after init (`scene.init_genesis`). Order-sensitive: must run before lerobot builds its processors (true for lerobot-eval, which makes the env first).
- `lerobot/smolvla_base` config hardcodes `observation.images.camera1/2/3`; `make_policy` only rebuilds input features when empty → both train and eval pass `--rename_map` front→camera1, wrist→camera2; camera3 is dropped (`empty_cameras=0`).
- `lerobot-eval` console script needs `PYTHONPATH=.` to import the `rebot_sim` plugin (`eval.sh` handles it). `--eval.batch_size` must stay 1 (Genesis + AsyncVectorEnv don't mix).
- Pinocchio rejects the URDF's relative mesh paths; no SRDF → adjacent-pair collision filter.
- A stray `~/.local/lib/python3.12/site-packages/cmeel*` shadowed pinocchio → `PYTHONNOUSERSITE=1` in `pixi.toml` activation env.
- `is_success` originally accepted a shove; now latches `_lifted` when cube z > table + 4 cm (`LIFT_HEIGHT`).

## Known gaps (from the original review, still open)

1. Cube is not in the oracle's Pinocchio collision model (spec §2 wanted it); free-space RRT could sweep through it. Fine at current ranges.
2. Oracle plans a retreat segment that is never executed/recorded: `collect.py` breaks on `terminated`, which fires during the release hold. Either drop the segment or record through it (needs re-collection).
3. `GRIPPER_SETTLE_STEPS=12` → 24 identical hold actions per ~90-frame demo; watch for "policy freezes at grasp" in eval videos. 8 also works.
4. No `close()` on the env; Genesis scenes leak in-process (harmless at n_envs=1).
5. `slow` pytest marker defined but not excluded by default.
6. `env` rejects `render_mode=` kwarg despite advertising `rgb_array`.

## Next steps (in priority order)

### 1. Classify the 8 failures (~30 min, no compute)

Watch `outputs/eval/smolvla_rebot/videos/rebot_0/eval_episode_{8,10,11,12,25,27,31,35}.mp4` (seeds 1008, 1010,
1011, 1012, 1025, 1027, 1031, 1035) and bin each into: missed grasp (fingers close beside the cube), grasp then drop,
placed outside the zone, never released, froze mid-episode, or timed out. Note the cube's spawn position for each;
if the misses cluster at one edge of `CUBE_X_RANGE`/`CUBE_Y_RANGE`, that points at wrist-camera coverage rather than
policy capacity. Record the tally here. The fix differs per bin: missed grasp → more demos or a lower
`n_action_steps`; drop → `GRIP_CLOSED`/`GRIPPER_SETTLE_STEPS`; freeze → the repeated hold frames (gap 3 below).

### 2. Ablate the 38% → 84% jump (3 cycles, ~90 min each, GPU)

Same recipe (200 demos, 20k steps, 50 eval episodes), one change reverted per run:

| Run | Revert | How |
|---|---|---|
| A: no hover | oracle hover segment | in `oracle.py` `plan_episode`, delete segment 0 and start segment 1 from `q0` instead of `q_hover` |
| B: old camera | front camera pose | `scene.py`: `CAM_POS=(1.6, -1.8, 1.5)`, `CAM_LOOKAT=(-0.1, 0.0, TABLE_HEIGHT+0.25)` |
| C: old lighting | overhead light | `scene.py`: `LIGHT_DIR=(-1,-1,-1)`, `AMBIENT=(0.1,0.1,0.1)` |

For each: move `data/rebot_pick_place` and `outputs/{train,eval}/smolvla_rebot` aside (e.g. suffix `_ablA`), run
`scripts/cycle.sh 200 50`, note `pc_success`, restore the code. At n=50 and p≈0.84 the binomial standard error is ~5 pts, so only
differences above ~15 pts are meaningful; if all three land near 84%, the changes were redundant and any one
would have sufficed. Consider a second seed range (`--seed=2000`) for the winner to firm up the number.

### 3. Raise the ceiling (after 1 and 2)

- More demos (`scripts/cycle.sh 500 50`; collection is ~3 s/episode) and/or longer training (`STEPS=50000`).
- `--policy.n_action_steps=10` (or 25) at eval only: cheap, replans the chunk more often.
- `GRIPPER_SETTLE_STEPS=8` in `oracle.py` to cut repeated hold frames (needs re-collection).
- Randomize the arm start pose or the zone to reduce near-duplicate early frames (needs re-collection).
- Put the cube into the oracle's collision model (gap 1) if failures show the arm sweeping through it.

## How to resume

```bash
git checkout main
pixi run test                                                                     # 19 passed, ~2 min
scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 50  # reproduces 84%, ~2 min
OUTPUT_DIR=outputs/eval/watch scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 5 --env.show_viewer=true
```
Full retrain: `scripts/cycle.sh 200 50` after moving the current dataset/checkpoint dirs aside (~90 min:
collect 10, train 75, eval 2). See README "Training a policy" for the individual steps and tuning knobs.
