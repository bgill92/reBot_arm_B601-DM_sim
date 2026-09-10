# Handoff — VLA pick-and-place testbed

Last updated: 2026-09-10. Branch `vla-testbed` (26 commits on top of `main` @ `05c1a7f`), **not merged**.
Working tree clean; all 19 tests pass (`pixi run test`, ~1–2.5 min).

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

## Known gaps / follow-ups (from the final review, not done)

1. Cube is not in the oracle's Pinocchio collision model (spec §2 wanted it); free-space RRT could sweep through it. Fine at current ranges.
2. Oracle plans a retreat segment that is never executed/recorded: `collect.py` breaks on `terminated`, which fires during the release hold. Either drop the segment or record through it (needs re-collection).
3. `GRIPPER_SETTLE_STEPS=12` → 24 identical hold actions per ~90-frame demo; watch for "policy freezes at grasp" in eval videos. 8 also works.
4. No `close()` on the env; Genesis scenes leak in-process (harmless at n_envs=1).
5. `slow` pytest marker defined but not excluded by default.
6. `env` rejects `render_mode=` kwarg despite advertising `rgb_array`.
7. Second-cycle failure videos (8 episodes listed above) not yet classified (miss grasp vs drop vs place miss).

## Suggested next steps to raise the 84%

- More demos (`--episodes 500+`; collection ~6 s/episode), longer training (`STEPS=50000`), and/or `GRIPPER_SETTLE_STEPS=8`.
- Randomize the arm start pose or the zone to reduce near-duplicate early frames.
- Inspect the 8 failure videos in `outputs/eval/smolvla_rebot/videos/rebot_0/` to classify failure modes.
- Try `--policy.n_action_steps` smaller than 50 (chunk replan more often) at eval.

## How to resume

```bash
git checkout vla-testbed
pixi run test                                              # 19 passed
scripts/eval.sh outputs/train/smolvla_rebot/checkpoints/last/pretrained_model 50
```
Full training reproduction: `pixi run python scripts/collect.py --episodes 200` (~20 min) → `scripts/train.sh` (~75 min) → `scripts/eval.sh` (~5 min).
Decision pending from the user: merge `vla-testbed` into `main`, open a PR, or keep the branch.
