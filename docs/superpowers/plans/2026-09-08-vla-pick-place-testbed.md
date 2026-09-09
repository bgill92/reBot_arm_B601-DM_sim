# VLA Pick-and-Place Testbed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Genesis reBot Arm sim into a gymnasium env, generate pick-and-place demos with a pyroboplan oracle, write them as a LeRobot dataset, fine-tune SmolVLA, and evaluate with stock `lerobot-eval`.

**Architecture:** One `gymnasium.Env` (`rebot_sim/env.py`) wraps the Genesis scene and is the only thing the oracle and the VLA touch. The oracle (`rebot_sim/oracle.py`) plans in a Pinocchio model of the same URDF and emits 7-d joint-target actions. `scripts/collect.py` runs oracle through env into `LeRobotDataset`. `rebot_sim/lerobot_env.py` registers an `EnvConfig` so `lerobot-eval --env.type=rebot --env.discover_packages_path=rebot_sim` works unchanged.

**Tech Stack:** Python 3.12, pixi, Genesis 1.3.3, pyroboplan 1.3.1 (Pinocchio 4.0 + coal), gymnasium ≥1.1, LeRobot 0.6.1 with SmolVLA, PyTorch 2.11 CUDA 12.9, pytest.

Spec: `docs/superpowers/specs/2026-09-08-vla-pick-place-testbed-design.md`.

---

## Facts verified before planning (do not re-derive)

- **URDF frames.** `joint6` child is `link6`. Fixed `end_joint` puts `end_link` at `xyz="0 0 0.15539"` along `link6` +z. So `link6` +z is the tool approach axis and `end_link` origin sits ~15.5 cm out, near the fingertips. Fingers slide along `end_link` y (= `link6` −y). `finger_left` range `[0, 0.05]`, `finger_right` `[-0.05, 0]`; closed = 0.
- **Joint limits** (rad): j1 ±2.8, j2 [−3.14, 0], j3 [−3.14, 0], j4 [−1.87, 1.57], j5 ±1.57, j6 ±3.14.
- **Pinocchio** rejects the URDF's relative mesh paths. Load as string, replace `../meshes` with absolute path, `pin.buildModelFromXML` + `pin.buildGeomFromUrdfString(model, xml, pin.GeometryType.COLLISION)`. Pin 4 ships collision shapes as the `coal` module (`coal.Box`).
- **Self-collision pairs.** No SRDF. After `addAllCollisionPairs()` the home pose "collides". Remove pairs whose geometries sit on the same, parent/child, or sibling joints. Leaves 32 pairs, home is collision-free. Verified `DifferentialIk` on `link6` converges and bidirectional RRT-Connect plans.
- **pyroboplan shapes.** `CubicPolynomialTrajectory(t, q)` wants `q` as `(ndof, N)`; `.generate(dt)` returns `(t_vec, q, qd, qdd, qddd)` with `q` shaped `(ndof, N)`. `CartesianPlanner.generate(q_init, dt)` returns `(success, t_vec, q)` with `q` `(nq, N)`. `RRTPlanner.plan(q_start, q_goal)` returns a list of configs or `None`. `shortcut_path(model, collision_model, q_path, max_iters, max_step_size)`.
- **Genesis.** `gs._initialized` flag guards double `gs.init`. `entity.get_pos()`/`get_quat()` return torch tensors, quats are **wxyz**. `cam.render(rgb=True)[0]` is HxWx3 uint8. `scene.add_camera(res, pos, lookat, fov, GUI=False)`, `cam.set_pose(pos=, lookat=, up=)`. `gs.morphs.Box(size, pos, fixed, collision)`. `set_dofs_position(pos, dofs, zero_velocity=True)`.
- **LeRobot 0.6.1.** Requires `torch>=2.7,<2.12`, `numpy<2.3`, `gymnasium>=1.1.1`. Current lock has torch 2.13 / numpy 2.5.2, so pixi must downgrade (conda-forge has `pytorch-gpu 2.11.0 cuda129`). `lerobot-eval` expects gym obs `{"pixels": {cam: HxWx3}, "agent_pos": float32[...]}`, reads `info["is_success"]`, `env.call("task_description")`, and `env.call("_max_episode_steps")` (from gymnasium `TimeLimit`, i.e. register with `max_episode_steps`). `--env.discover_packages_path=<pkg>` imports `<pkg>` before parsing so `@EnvConfig.register_subclass("rebot")` and `gymnasium.register` run. `EnvConfig.features_map` maps env keys (`"pixels/front"`, `"agent_pos"`, `"action"`) to policy keys (`observation.images.front`, `observation.state`, `action`).
- **LeRobotDataset.create(repo_id, fps, features, root=None, robot_type=None, use_videos=True)**. Video feature dict: `{"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}`. State/action: `{"dtype": "float32", "shape": (7,), "names": [...]}`. `add_frame(frame)` needs a `"task"` key. Then `save_episode()`, finally `finalize()`.
- **SmolVLA** config: `chunk_size=50`, `n_action_steps=50`, `empty_cameras=0`, `resize_imgs_with_padding=(512, 512)`. Fine-tune entrypoint: `lerobot-train --policy.path=lerobot/smolvla_base`.

## File structure

| Path | Responsibility |
|---|---|
| `rebot_sim/__init__.py` | gym registration; import `lerobot_env` if lerobot present |
| `rebot_sim/scene.py` | constants (paths, joints, limits, room, table, cube, zone, cams) and Genesis builders; `init_genesis()` |
| `rebot_sim/env.py` | `RebotPickPlaceEnv(gym.Env)`: reset/step/obs/success |
| `rebot_sim/oracle.py` | Pinocchio model load + collision filter, `plan_episode(env)` → list of 7-d actions |
| `rebot_sim/lerobot_env.py` | `RebotEnv(EnvConfig)` for `lerobot-eval` |
| `scripts/collect.py` | oracle → `LeRobotDataset` |
| `scripts/train.sh`, `scripts/eval.sh` | CLI wrappers |
| `tests/conftest.py` | session-scoped env fixture (Genesis builds are slow) |
| `tests/test_scene.py`, `tests/test_env.py`, `tests/test_oracle.py`, `tests/test_collect.py`, `tests/test_lerobot_env.py` | headless pytest |
| `sim.py` | viewer; imports from `rebot_sim.scene` |
| `README.md` | usage + baseline results |

All commands run from the repo root via `pixi run ...`. Every test is headless (no viewer).

---

### Task 0: Dependencies

**Files:**
- Modify: `pixi.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Edit `pixi.toml`**

Replace the whole file with:

```toml
[workspace]
name = "reBot_arm_B601-DM_sim"
channels = ["conda-forge"]
platforms = [{ name = "linux-64-cuda", platform = "linux-64", cuda = "12.8", glibc = "2.35" }]

[dependencies]
python = "3.12.*"
pytorch-gpu = ">=2.7,<2.12"
torchvision = "*"
cuda-version = ">=12.8"
numpy = "<2.3"
ffmpeg = "*"
pytest = "*"

[pypi-dependencies]
genesis-world = ">=1.3.3"
pyroboplan = ">=1.3.1"
gymnasium = ">=1.1.1"
lerobot = { version = ">=0.6.1", extras = ["smolvla"] }

[tasks]
test = "pytest -x -q tests"
sim = "python sim.py"
collect = "python scripts/collect.py"
```

- [ ] **Step 2: Solve and install**

Run: `pixi install`
Expected: solves and installs. If the solver fails on `pyroboplan`'s exact pins (`matplotlib==3.10.9`, `scipy<=1.17.1`, `drake`), fall back to installing it without deps:

```toml
[pypi-dependencies]
genesis-world = ">=1.3.3"
pin = "==4.0.0"
toppra = "==0.6.3"
gymnasium = ">=1.1.1"
lerobot = { version = ">=0.6.1", extras = ["smolvla"] }
```

then `pixi run pip install --no-deps pyroboplan==1.3.1` and add `pixi run pip install --no-deps pyroboplan==1.3.1` as a `[tasks] setup` entry. Record which path was taken in the README (Task 11).

- [ ] **Step 3: Verify the stack imports**

Run:
```bash
pixi run python -c "import torch, numpy, genesis, pyroboplan, gymnasium, lerobot; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, lerobot.__version__)"
```
Expected: prints a `2.11.x True 2.2.x 0.6.1`-like line, no ImportError.

- [ ] **Step 4: Verify the existing sim still runs**

Run: `pixi run python sim.py --headless`
Expected: `wrote frame.png` and `final joint error vs HOME: 0.0xxx rad`, exit 0.

- [ ] **Step 5: Ignore generated outputs**

Append to `.gitignore`:
```
frame.png
outputs/
data/
```

- [ ] **Step 6: Commit**

```bash
git add pixi.toml pixi.lock .gitignore
git commit -m "Add pyroboplan, gymnasium, lerobot[smolvla], pytest deps"
```

---

### Task 1: Extract scene module from `sim.py`

**Files:**
- Create: `rebot_sim/__init__.py`
- Create: `rebot_sim/scene.py`
- Modify: `sim.py`
- Create: `tests/test_scene.py`

- [ ] **Step 1: Write the failing test**

`tests/test_scene.py`:
```python
import numpy as np

from rebot_sim import scene


def test_constants_consistent():
    assert scene.URDF.exists()
    assert len(scene.ARM_JOINTS) == 6
    assert len(scene.HOME) == 6
    assert np.all(scene.JOINT_LOWER <= scene.HOME) and np.all(scene.HOME <= scene.JOINT_UPPER)


def test_local_to_world_puts_point_on_table():
    p = scene.local_to_world((0.0, 0.0))
    assert np.allclose(p, scene.ARM_POS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_scene.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rebot_sim'`.

- [ ] **Step 3: Create the package and scene module**

`rebot_sim/__init__.py` (registration comes in Task 5; keep empty for now):
```python
"""reBot Arm B601-DM pick-and-place sim package."""
```

`rebot_sim/scene.py`:
```python
"""Constants and Genesis scene builders shared by the viewer, the gym env, and the oracle."""

from pathlib import Path

import genesis as gs
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets" / "rebot_arm_dm"
URDF = ASSET_DIR / "urdf" / "ReBot_Arm_DM.urdf"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["finger_left", "finger_right"]  # right mimics left with multiplier -1
GRIPPER_MAX = 0.05
JOINT_LOWER = np.array([-2.8, -3.14, -3.14, -1.87, -1.57, -3.14])
JOINT_UPPER = np.array([2.8, 0.0, 0.0, 1.57, 1.57, 3.14])
# joint2/joint3 limits are [-3.14, 0], so "up" is negative.
HOME = np.array([0.0, -1.57, -1.57, 0.0, 0.0, 0.0])

# Room: 4 m x 4 m, 2.6 m high, arm on a table at the center.
ROOM_W, ROOM_D, ROOM_H = 4.0, 4.0, 2.6
WALL_T = 0.05
TABLE_HEIGHT = 0.75
TABLE_W, TABLE_D, TABLE_T = 1.2, 0.8, 0.04
LEG_S = 0.05
ARM_POS = np.array([-(TABLE_W / 2 - 0.1), 0.0, TABLE_HEIGHT])  # near the back (-x) edge; arm reaches +x
CAM_POS = (1.6, -1.8, 1.5)
CAM_LOOKAT = (-0.1, 0.0, TABLE_HEIGHT + 0.25)

# Task objects. xy ranges are in the arm-base frame (x forward, y left).
CUBE_SIZE = 0.03
CUBE_X_RANGE = (0.15, 0.35)
CUBE_Y_RANGE = (-0.15, 0.15)
ZONE_SIZE = 0.10
ZONE_XY = (0.25, -0.25)

COLOR_FLOOR = (0.55, 0.45, 0.35)
COLOR_WALL = (0.85, 0.85, 0.8)
COLOR_WOOD = (0.6, 0.42, 0.25)
COLOR_CUBE = (0.9, 0.1, 0.1)
COLOR_ZONE = (0.1, 0.8, 0.1)


def init_genesis() -> None:
    if not gs._initialized:
        gs.init(backend=gs.gpu)


def local_to_world(xy) -> np.ndarray:
    """Arm-base-frame xy -> world xyz on the table surface."""
    return ARM_POS + np.array([xy[0], xy[1], 0.0])


def _box(scene: gs.Scene, size, pos, color, **kw):
    return scene.add_entity(
        gs.morphs.Box(size=size, pos=tuple(pos), **kw),
        surface=gs.surfaces.Default(color=color),
    )


def build_room(scene: gs.Scene) -> None:
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=COLOR_FLOOR))
    hw, hd, hh = ROOM_W / 2, ROOM_D / 2, ROOM_H / 2
    _box(scene, (ROOM_W, WALL_T, ROOM_H), (0, hd, hh), COLOR_WALL, fixed=True)
    _box(scene, (ROOM_W, WALL_T, ROOM_H), (0, -hd, hh), COLOR_WALL, fixed=True)
    _box(scene, (WALL_T, ROOM_D, ROOM_H), (-hw, 0, hh), COLOR_WALL, fixed=True)
    _box(scene, (WALL_T, ROOM_D, ROOM_H), (hw, 0, hh), COLOR_WALL, fixed=True)

    _box(scene, (TABLE_W, TABLE_D, TABLE_T), (0, 0, TABLE_HEIGHT - TABLE_T / 2), COLOR_WOOD, fixed=True)
    lx, ly = TABLE_W / 2 - LEG_S, TABLE_D / 2 - LEG_S
    leg_h = TABLE_HEIGHT - TABLE_T
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(scene, (LEG_S, LEG_S, leg_h), (sx * lx, sy * ly, leg_h / 2), COLOR_WOOD, fixed=True)


def add_arm(scene: gs.Scene):
    return scene.add_entity(gs.morphs.URDF(file=str(URDF), fixed=True, pos=tuple(ARM_POS)))


def add_cube(scene: gs.Scene):
    pos = local_to_world((CUBE_X_RANGE[0], 0.0)) + np.array([0, 0, CUBE_SIZE / 2])
    return _box(scene, (CUBE_SIZE,) * 3, pos, COLOR_CUBE)


def add_zone(scene: gs.Scene):
    pos = local_to_world(ZONE_XY) + np.array([0, 0, 0.001])
    return _box(scene, (ZONE_SIZE, ZONE_SIZE, 0.002), pos, COLOR_ZONE, fixed=True, collision=False)


def set_arm_gains(arm, arm_dofs, finger_dofs) -> None:
    # ponytail: rough PD gains, tune against real Damiao 4340P/4310 MIT-mode response if needed.
    arm.set_dofs_kp(np.array([400, 400, 400, 100, 100, 100]), arm_dofs)
    arm.set_dofs_kv(np.array([40, 40, 40, 10, 10, 10]), arm_dofs)
    arm.set_dofs_kp(np.array([200, 200]), finger_dofs)
    arm.set_dofs_kv(np.array([10, 10]), finger_dofs)
```

- [ ] **Step 4: Rewrite `sim.py` on top of the module**

Replace `sim.py` with:
```python
"""Genesis simulation of the Seeed reBot Arm B601-DM.

Usage:
    pixi run python sim.py            # interactive viewer
    pixi run python sim.py --headless # no viewer, saves frame.png
"""

import argparse
import sys

import genesis as gs
import numpy as np

from rebot_sim import scene as S

WAYPOINTS = [
    S.HOME,
    np.array([0.8, -1.0, -2.0, 0.5, 0.5, 1.0]),
    np.array([-0.8, -2.0, -1.0, -0.5, -0.5, -1.0]),
    S.HOME,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    S.init_genesis()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(camera_pos=S.CAM_POS, camera_lookat=S.CAM_LOOKAT, camera_fov=40),
        show_viewer=not args.headless,
    )
    S.build_room(scene)
    arm = S.add_arm(scene)
    S.add_cube(scene)
    S.add_zone(scene)
    cam = scene.add_camera(res=(1280, 720), pos=S.CAM_POS, lookat=S.CAM_LOOKAT, fov=40, GUI=False)
    scene.build()

    arm_dofs = [arm.get_joint(n).dofs_idx_local[0] for n in S.ARM_JOINTS]
    finger_dofs = [arm.get_joint(n).dofs_idx_local[0] for n in S.FINGER_JOINTS]
    S.set_arm_gains(arm, arm_dofs, finger_dofs)
    arm.set_dofs_position(S.HOME, arm_dofs)

    def gripper(opening: float) -> None:
        opening = float(np.clip(opening, 0.0, S.GRIPPER_MAX))
        arm.control_dofs_position(np.array([opening, -opening]), finger_dofs)

    steps_per_segment = 150
    for i in range(len(WAYPOINTS) - 1):
        a, b = WAYPOINTS[i], WAYPOINTS[i + 1]
        for t in range(steps_per_segment):
            arm.control_dofs_position(a + (b - a) * t / steps_per_segment, arm_dofs)
            gripper(S.GRIPPER_MAX if i % 2 == 0 else 0.0)
            scene.step()
        if args.headless and i == 0:
            import imageio

            imageio.imwrite("frame.png", cam.render(rgb=True)[0])
            print("wrote frame.png")

    for _ in range(100):  # settle
        arm.control_dofs_position(S.HOME, arm_dofs)
        scene.step()
    q = arm.get_dofs_position(arm_dofs)
    err = np.abs(np.asarray(q.cpu()) - S.HOME).max()
    print(f"final joint error vs HOME: {err:.4f} rad")
    assert err < 0.05, "arm did not track waypoints"

    if not args.headless:
        while scene.viewer.is_alive():
            arm.control_dofs_position(S.HOME, arm_dofs)
            scene.step()
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and the viewer script**

Run: `pixi run pytest tests/test_scene.py -q && pixi run python sim.py --headless`
Expected: `2 passed`, then `wrote frame.png`, `final joint error vs HOME: ...`, exit 0. Open `frame.png` and confirm a red cube and green square sit on the table in front of the arm.

- [ ] **Step 6: Commit**

```bash
git add rebot_sim sim.py tests/test_scene.py
git commit -m "Extract scene builders into rebot_sim.scene; add cube and target zone"
```

---

### Task 2: Gym env — construction, reset, observation

**Files:**
- Create: `rebot_sim/env.py`
- Create: `tests/conftest.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py` (one Genesis scene per test session; building takes ~10 s):
```python
import pytest

from rebot_sim.env import RebotPickPlaceEnv


@pytest.fixture(scope="session")
def env():
    return RebotPickPlaceEnv()
```

`tests/test_env.py`:
```python
import numpy as np

from rebot_sim import scene as S


def test_reset_obs_shapes(env):
    obs, info = env.reset(seed=0)
    assert obs["pixels"]["front"].shape == (256, 256, 3)
    assert obs["pixels"]["wrist"].shape == (256, 256, 3)
    assert obs["pixels"]["front"].dtype == np.uint8
    assert obs["agent_pos"].shape == (7,) and obs["agent_pos"].dtype == np.float32
    assert info["is_success"] is False
    assert env.observation_space.contains(obs)


def test_reset_places_cube_in_range_and_is_seeded(env):
    env.reset(seed=3)
    c1 = env.cube_pos()
    env.reset(seed=3)
    c2 = env.cube_pos()
    assert np.allclose(c1, c2, atol=1e-3)
    local = c1 - S.ARM_POS
    assert S.CUBE_X_RANGE[0] - 0.01 <= local[0] <= S.CUBE_X_RANGE[1] + 0.01
    assert S.CUBE_Y_RANGE[0] - 0.01 <= local[1] <= S.CUBE_Y_RANGE[1] + 0.01
    assert abs(local[2] - S.CUBE_SIZE / 2) < 0.005  # resting on the table


def test_wrist_camera_sees_something(env):
    obs, _ = env.reset(seed=0)
    assert obs["pixels"]["wrist"].std() > 5  # not a flat image
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rebot_sim.env'`.

- [ ] **Step 3: Implement the env (reset + obs only)**

`rebot_sim/env.py`:
```python
"""Gymnasium env: pick up the red cube and place it in the green square."""

import genesis as gs
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from rebot_sim import scene as S

# Wrist camera, expressed in the link6 frame (z = tool axis, fingers slide along y).
# Mounted off to +x so the fingers do not block the view, looking at the fingertips.
WRIST_CAM_OFFSET = np.array([0.06, 0.0, 0.03])
WRIST_CAM_LOOKAT = np.array([0.0, 0.0, 0.20])
WRIST_CAM_UP = np.array([1.0, 0.0, 0.0])


class RebotPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    TASK = "pick up the red cube and place it in the green square"

    def __init__(self, image_size: int = 256, substeps: int = 10, show_viewer: bool = False):
        S.init_genesis()
        self.substeps = substeps
        self.task_description = self.TASK
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            viewer_options=gs.options.ViewerOptions(camera_pos=S.CAM_POS, camera_lookat=S.CAM_LOOKAT, camera_fov=40),
            show_viewer=show_viewer,
        )
        S.build_room(self.scene)
        self.arm = S.add_arm(self.scene)
        self.cube = S.add_cube(self.scene)
        self.zone = S.add_zone(self.scene)
        res = (image_size, image_size)
        self.cam_front = self.scene.add_camera(res=res, pos=S.CAM_POS, lookat=S.CAM_LOOKAT, fov=40, GUI=False)
        self.cam_wrist = self.scene.add_camera(res=res, pos=S.CAM_POS, lookat=S.CAM_LOOKAT, fov=70, GUI=False)
        self.scene.build()

        self.arm_dofs = [self.arm.get_joint(n).dofs_idx_local[0] for n in S.ARM_JOINTS]
        self.finger_dofs = [self.arm.get_joint(n).dofs_idx_local[0] for n in S.FINGER_JOINTS]
        S.set_arm_gains(self.arm, self.arm_dofs, self.finger_dofs)
        self.link6 = self.arm.get_link("link6")
        self.zone_world = S.local_to_world(S.ZONE_XY)

        img = spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"front": img, "wrist": img}),
                "agent_pos": spaces.Box(-np.inf, np.inf, (7,), np.float32),
            }
        )
        self.action_space = spaces.Box(
            np.append(S.JOINT_LOWER, 0.0).astype(np.float32),
            np.append(S.JOINT_UPPER, S.GRIPPER_MAX).astype(np.float32),
            dtype=np.float32,
        )

    # ----- state helpers -----
    def cube_pos(self) -> np.ndarray:
        return self.cube.get_pos().cpu().numpy().reshape(3)

    def cube_quat(self) -> np.ndarray:
        """wxyz."""
        return self.cube.get_quat().cpu().numpy().reshape(4)

    def joint_pos(self) -> np.ndarray:
        return self.arm.get_dofs_position(self.arm_dofs).cpu().numpy().reshape(6)

    def gripper_opening(self) -> float:
        return float(self.arm.get_dofs_position(self.finger_dofs).cpu().numpy().reshape(2)[0])

    def _set_gripper_target(self, opening: float) -> None:
        g = float(np.clip(opening, 0.0, S.GRIPPER_MAX))
        self.arm.control_dofs_position(np.array([g, -g]), self.finger_dofs)

    # ----- gym API -----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.arm.set_dofs_position(S.HOME, self.arm_dofs, zero_velocity=True)
        self.arm.set_dofs_position(np.array([S.GRIPPER_MAX, -S.GRIPPER_MAX]), self.finger_dofs, zero_velocity=True)
        self.arm.control_dofs_position(S.HOME, self.arm_dofs)
        self._set_gripper_target(S.GRIPPER_MAX)

        x = self.np_random.uniform(*S.CUBE_X_RANGE)
        y = self.np_random.uniform(*S.CUBE_Y_RANGE)
        yaw = self.np_random.uniform(0.0, np.pi / 2)
        pos = S.local_to_world((x, y)) + np.array([0.0, 0.0, S.CUBE_SIZE / 2 + 0.002])
        quat = Rotation.from_euler("z", yaw).as_quat(scalar_first=True)
        self.cube.set_pos(pos)
        self.cube.set_quat(quat)
        for _ in range(20):  # settle
            self.scene.step()
        return self._get_obs(), {"is_success": False}

    def _update_wrist_cam(self) -> None:
        p = self.link6.get_pos().cpu().numpy().reshape(3)
        q = self.link6.get_quat().cpu().numpy().reshape(4)
        R = Rotation.from_quat(q, scalar_first=True).as_matrix()
        self.cam_wrist.set_pose(pos=p + R @ WRIST_CAM_OFFSET, lookat=p + R @ WRIST_CAM_LOOKAT, up=R @ WRIST_CAM_UP)

    def _get_obs(self) -> dict:
        self._update_wrist_cam()
        front = np.asarray(self.cam_front.render(rgb=True)[0])[..., :3]
        wrist = np.asarray(self.cam_wrist.render(rgb=True)[0])[..., :3]
        state = np.append(self.joint_pos(), self.gripper_opening()).astype(np.float32)
        return {"pixels": {"front": front, "wrist": wrist}, "agent_pos": state}

    def render(self):
        return np.asarray(self.cam_front.render(rgb=True)[0])[..., :3]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_env.py -q`
Expected: `3 passed`. If `test_wrist_camera_sees_something` fails (flat image), the camera is looking into the gripper body: try `WRIST_CAM_OFFSET = np.array([0.0, -0.06, 0.03])` and `WRIST_CAM_UP = np.array([0.0, -1.0, 0.0])`. To eyeball it, run:

```bash
pixi run python -c "
import imageio; from rebot_sim.env import RebotPickPlaceEnv
e = RebotPickPlaceEnv(); o, _ = e.reset(seed=0)
imageio.imwrite('front.png', o['pixels']['front']); imageio.imwrite('wrist.png', o['pixels']['wrist'])"
```
and open `wrist.png`. It should show the fingers at the bottom and the table below. Delete the PNGs afterwards.

- [ ] **Step 5: Commit**

```bash
git add rebot_sim/env.py tests/conftest.py tests/test_env.py
git commit -m "Add RebotPickPlaceEnv reset and observations"
```

---

### Task 3: Gym env — step, success, registration

**Files:**
- Modify: `rebot_sim/env.py`
- Modify: `rebot_sim/__init__.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_env.py`:
```python
import gymnasium as gym


def test_step_tracks_joint_target(env):
    env.reset(seed=0)
    target = S.HOME + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(15):
        obs, r, term, trunc, info = env.step(np.append(target, S.GRIPPER_MAX))
    assert abs(obs["agent_pos"][0] - target[0]) < 0.05
    assert r == 0.0 and term is False and info["is_success"] is False


def test_success_when_cube_teleported_into_zone(env):
    env.reset(seed=0)
    env.cube.set_pos(env.zone_world + np.array([0.0, 0.0, S.CUBE_SIZE / 2 + 0.002]))
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.append(S.HOME, S.GRIPPER_MAX))
    assert term is True and r == 1.0 and info["is_success"] is True


def test_registered_env_has_time_limit():
    import rebot_sim  # noqa: F401  registers the id

    e = gym.make("gym_rebot/RebotPickPlace-v0")
    assert e.spec.max_episode_steps == 300
    assert e.unwrapped.task_description == e.unwrapped.TASK
    e.close()
```

Note: `test_registered_env_has_time_limit` builds a second Genesis scene. That is fine (Genesis supports several scenes per process) but slow; keep it last.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_env.py -q`
Expected: 3 new failures (`AttributeError: ... has no attribute 'step'`, `gym.error.NameNotFound`).

- [ ] **Step 3: Add step and success to `rebot_sim/env.py`**

Add these methods to `RebotPickPlaceEnv` (after `reset`):
```python
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(7)
        q = np.clip(action[:6], S.JOINT_LOWER, S.JOINT_UPPER)
        self.arm.control_dofs_position(q, self.arm_dofs)
        self._set_gripper_target(float(action[6]))
        for _ in range(self.substeps):
            self.scene.step()
        self._step_count += 1
        success = self.is_success()
        return self._get_obs(), float(success), success, False, {"is_success": success}

    def is_success(self) -> bool:
        c = self.cube_pos()
        in_zone = np.all(np.abs(c[:2] - self.zone_world[:2]) <= S.ZONE_SIZE / 2)
        on_table = abs(c[2] - (S.TABLE_HEIGHT + S.CUBE_SIZE / 2)) < 0.01
        released = self.gripper_opening() > 0.03
        return bool(in_zone and on_table and released)
```

- [ ] **Step 4: Register the env**

`rebot_sim/__init__.py`:
```python
"""reBot Arm B601-DM pick-and-place sim package.

Importing this package registers `gym_rebot/RebotPickPlace-v0` and, when lerobot is
installed, the `rebot` EnvConfig used by `lerobot-eval`.
"""

from gymnasium.envs.registration import register

register(
    id="gym_rebot/RebotPickPlace-v0",
    entry_point="rebot_sim.env:RebotPickPlaceEnv",
    max_episode_steps=300,
)

try:
    from rebot_sim import lerobot_env  # noqa: F401
except ImportError:  # lerobot not installed or lerobot_env not written yet
    pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_env.py -q`
Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add rebot_sim/env.py rebot_sim/__init__.py tests/test_env.py
git commit -m "Add env step, success check, and gym registration"
```

---

### Task 4: Oracle — Pinocchio model, collision filtering, IK

**Files:**
- Create: `rebot_sim/oracle.py`
- Create: `tests/test_oracle.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_oracle.py`:
```python
import numpy as np
import pinocchio as pin

from rebot_sim import oracle, scene as S


def test_model_loads_reduced_and_home_is_collision_free():
    model, coll = oracle.load_model()
    assert model.nq == 6
    assert [model.names[i] for i in range(1, model.njoints)] == S.ARM_JOINTS
    assert "table" in [g.name for g in coll.geometryObjects]
    assert not oracle.in_collision(model, coll, S.HOME)


def test_ik_reaches_top_down_pose_above_cube_region():
    model, coll = oracle.load_model()
    target = oracle.top_down_pose(np.array([0.25, 0.0, S.CUBE_SIZE / 2 + oracle.APPROACH]), yaw=0.0)
    q = oracle.solve_ik(model, coll, target, S.HOME)
    assert q is not None
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    T = data.oMf[model.getFrameId(oracle.TCP_FRAME)]
    assert np.allclose(T.translation, target.translation, atol=2e-3)
    assert np.allclose(T.rotation[:, 0], [0, 0, -1], atol=1e-2)  # tool axis points down
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_oracle.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rebot_sim.oracle'`.

- [ ] **Step 3: Implement model loading, collision filter, pose helper, IK**

`rebot_sim/oracle.py`:
```python
"""Classical pick-and-place oracle: Pinocchio model + pyroboplan IK/RRT, emits 7-d joint actions."""

import coal
import numpy as np
import pinocchio as pin
from pyroboplan.core.utils import check_collisions_at_state
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.cartesian_planner import CartesianPlanner, CartesianPlannerOptions
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.polynomial import CubicPolynomialTrajectory

from rebot_sim import scene as S

# end_link sits 15.5 cm along link6 +z, near the fingertips. Its local x axis is the tool axis
# (points out of the gripper) and its y axis is the finger sliding axis.
TCP_FRAME = "end_link"
# ponytail: extra offset along the tool axis from end_link to the fingertip midpoint; tune if
# the gripper grasps too high/low. Positive moves the grasp deeper.
TCP_OFFSET = 0.0
APPROACH = 0.08
GRIP_OPEN = S.GRIPPER_MAX
GRIP_CLOSED = 0.012  # ponytail: per-finger opening that squeezes the 3 cm cube; tune on hardware.
CONTROL_DT = 0.1  # 10 Hz, matches env substeps * sim dt
JOINT_SPEED = 1.0  # rad/s used to time joint-space segments
GRIPPER_SETTLE_STEPS = 8


def load_model():
    """Build the 6-dof arm model with fingers locked and the table as an obstacle."""
    xml = S.URDF.read_text().replace("../meshes", str(S.ASSET_DIR / "meshes"))
    full = pin.buildModelFromXML(xml)
    full_coll = pin.buildGeomFromUrdfString(full, xml, pin.GeometryType.COLLISION)
    lock = [full.getJointId(n) for n in S.FINGER_JOINTS]
    model, coll = pin.buildReducedModel(full, full_coll, lock, pin.neutral(full))

    # Table in the arm-base frame: the base sits on the table top at world ARM_POS.
    table_center = np.array([-S.ARM_POS[0], 0.0, -S.TABLE_T / 2])
    coll.addGeometryObject(
        pin.GeometryObject("table", 0, pin.SE3(np.eye(3), table_center), coal.Box(S.TABLE_W, S.TABLE_D, S.TABLE_T))
    )
    coll.addAllCollisionPairs()
    _remove_adjacent_pairs(model, coll)
    return model, coll


def _remove_adjacent_pairs(model, coll) -> None:
    """Drop pairs on the same / parent-child / sibling joints; there is no SRDF for this arm."""

    def adjacent(j1, j2):
        return j1 == j2 or model.parents[j1] == j2 or model.parents[j2] == j1 or model.parents[j1] == model.parents[j2]

    keep = [
        cp
        for cp in coll.collisionPairs
        if not adjacent(coll.geometryObjects[cp.first].parentJoint, coll.geometryObjects[cp.second].parentJoint)
    ]
    coll.removeAllCollisionPairs()
    for cp in keep:
        coll.addCollisionPair(cp)


def in_collision(model, coll, q) -> bool:
    return check_collisions_at_state(model, coll, q, model.createData(), coll.createData())


def top_down_pose(p_local: np.ndarray, yaw: float) -> pin.SE3:
    """TCP pose (arm-base frame) with the tool axis pointing down and fingers closing along `yaw`.

    end_link x = tool axis -> world -z. end_link y = finger axis -> rotated by `yaw` in the xy plane.
    """
    x_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return pin.SE3(R, np.asarray(p_local, dtype=float) + np.array([0.0, 0.0, TCP_OFFSET]))


def solve_ik(model, coll, target: pin.SE3, q_init: np.ndarray, seed: int = 0):
    ik = DifferentialIk(
        model,
        data=model.createData(),
        collision_model=coll,
        options=DifferentialIkOptions(max_iters=500, max_retries=20, rng_seed=seed),
    )
    return ik.solve(TCP_FRAME, target, init_state=q_init)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_oracle.py -q`
Expected: `2 passed`. If the collision test fails because `table` collides with `base_link`, both live on joint 0 so `_remove_adjacent_pairs` should already drop that pair; print `[(coll.geometryObjects[cp.first].name, coll.geometryObjects[cp.second].name) for cp in coll.collisionPairs]` and the offending pair to debug.

- [ ] **Step 5: Commit**

```bash
git add rebot_sim/oracle.py tests/test_oracle.py
git commit -m "Add oracle model loading, collision filtering, and top-down IK"
```

---

### Task 5: Oracle — full pick-and-place episode plan

**Files:**
- Modify: `rebot_sim/oracle.py`
- Modify: `tests/test_oracle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_oracle.py`:
```python
import pytest


def run_oracle_episode(env, seed):
    env.reset(seed=seed)
    actions = oracle.plan_episode(env)
    assert len(actions) > 0
    term = False
    for a in actions:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return term


def test_oracle_succeeds_on_one_seed(env):
    assert run_oracle_episode(env, seed=0)


@pytest.mark.slow
def test_oracle_success_rate(env):
    successes = sum(run_oracle_episode(env, seed=s) for s in range(20))
    assert successes >= 18, f"oracle success {successes}/20"
```

Register the marker in a new `pytest.ini`:
```ini
[pytest]
markers =
    slow: long-running end-to-end checks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_oracle.py -q -k one_seed`
Expected: FAIL with `AttributeError: module 'rebot_sim.oracle' has no attribute 'plan_episode'`.

- [ ] **Step 3: Implement segment planners and `plan_episode`**

Append to `rebot_sim/oracle.py`:
```python
class _Planner:
    """Holds one Pinocchio model per process; env is only read for cube pose and joint state."""

    def __init__(self):
        self.model, self.coll = load_model()

    def joint_path(self, q_start, q_goal, seed):
        """Collision-free joint-space path, shortcut, resampled at CONTROL_DT. Returns (N, 6) or None."""
        opts = RRTPlannerOptions(
            max_step_size=0.1,
            max_connection_dist=1.0,
            rrt_connect=True,
            bidirectional_rrt=True,
            max_planning_time=5.0,
            rng_seed=seed,
        )
        path = RRTPlanner(self.model, self.coll, opts).plan(q_start, q_goal)
        if path is None:
            return None
        path = shortcut_path(self.model, self.coll, path, max_iters=100, max_step_size=0.05)
        q_wp = np.array(path).T  # (6, K)
        dists = np.linalg.norm(np.diff(q_wp, axis=1), axis=0)
        t_wp = np.concatenate([[0.0], np.cumsum(np.maximum(dists / JOINT_SPEED, CONTROL_DT))])
        _, q, *_ = CubicPolynomialTrajectory(t_wp, q_wp).generate(CONTROL_DT)
        return q.T

    def cartesian_path(self, q_start, target: pin.SE3):
        """Straight-line TCP move from the current pose to `target`. Returns (N, 6) or None."""
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q_start)
        start = data.oMf[self.model.getFrameId(TCP_FRAME)]
        ik = DifferentialIk(
            self.model,
            data=data,
            collision_model=self.coll,
            options=DifferentialIkOptions(max_iters=200, max_retries=3, rng_seed=0),
        )
        planner = CartesianPlanner(
            self.model,
            TCP_FRAME,
            [start, target],
            ik,
            options=CartesianPlannerOptions(max_linear_velocity=0.2, max_linear_acceleration=0.5),
        )
        ok, _, q = planner.generate(q_start, CONTROL_DT)
        return q.T if ok else None


_planner = None


def _get_planner() -> _Planner:
    global _planner
    if _planner is None:
        _planner = _Planner()
    return _planner


def _with_gripper(q_path, opening) -> list[np.ndarray]:
    return [np.append(q, opening).astype(np.float32) for q in q_path]


def _hold(q, opening, n=GRIPPER_SETTLE_STEPS) -> list[np.ndarray]:
    return [np.append(q, opening).astype(np.float32)] * n


def plan_episode(env, seed: int = 0) -> list[np.ndarray]:
    """Plan a full pick-and-place from the env's current (privileged) state.

    Returns the list of 7-d actions (6 joint targets + gripper opening) to feed to env.step,
    or an empty list if any segment fails to plan.
    """
    P = _get_planner()
    cube_local = env.cube_pos() - S.ARM_POS
    from scipy.spatial.transform import Rotation

    yaw = Rotation.from_quat(env.cube_quat(), scalar_first=True).as_euler("xyz")[2]
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # cube is symmetric every 90 deg
    zone_local = env.zone_world - S.ARM_POS
    up = np.array([0.0, 0.0, APPROACH])

    grasp = top_down_pose(cube_local, yaw)
    pre_grasp = top_down_pose(cube_local + up, yaw)
    place = top_down_pose(zone_local + np.array([0.0, 0.0, S.CUBE_SIZE / 2]), 0.0)
    pre_place = top_down_pose(zone_local + np.array([0.0, 0.0, S.CUBE_SIZE / 2]) + up, 0.0)

    q0 = env.joint_pos()
    q_pre_grasp = solve_ik(P.model, P.coll, pre_grasp, q0, seed)
    q_pre_place = solve_ik(P.model, P.coll, pre_place, q0, seed)
    if q_pre_grasp is None or q_pre_place is None:
        return []

    actions: list[np.ndarray] = []

    # 1. free-space move above the cube
    seg = P.joint_path(q0, q_pre_grasp, seed)
    if seg is None:
        return []
    actions += _with_gripper(seg, GRIP_OPEN)

    # 2. straight down, close
    seg = P.cartesian_path(q_pre_grasp, grasp)
    if seg is None:
        return []
    actions += _with_gripper(seg, GRIP_OPEN)
    q_grasp = seg[-1]
    actions += _hold(q_grasp, GRIP_CLOSED)

    # 3. straight up (fingers closed)
    seg = P.cartesian_path(q_grasp, pre_grasp)
    if seg is None:
        return []
    actions += _with_gripper(seg, GRIP_CLOSED)
    q_lift = seg[-1]

    # 4. free-space move above the zone
    seg = P.joint_path(q_lift, q_pre_place, seed)
    if seg is None:
        return []
    actions += _with_gripper(seg, GRIP_CLOSED)

    # 5. straight down, open
    seg = P.cartesian_path(q_pre_place, place)
    if seg is None:
        return []
    actions += _with_gripper(seg, GRIP_CLOSED)
    q_place = seg[-1]
    actions += _hold(q_place, GRIP_OPEN)

    # 6. retreat
    seg = P.cartesian_path(q_place, pre_place)
    if seg is None:
        return []
    actions += _with_gripper(seg, GRIP_OPEN)
    return actions
```

Move the `from scipy.spatial.transform import Rotation` import to the module top with the other imports.

- [ ] **Step 4: Run the single-seed test**

Run: `pixi run pytest tests/test_oracle.py -q -k one_seed -s`
Expected: `1 passed`.

Debugging ladder if it fails (work top to bottom, one at a time, re-run after each):
1. `plan_episode` returned `[]`: print which segment returned `None`. IK failures on `pre_grasp` usually mean the cube region is outside reach for a top-down pose; shrink `CUBE_X_RANGE` upper bound to 0.32. RRT failures: raise `max_planning_time` to 10.
2. Cube never lifts (check `env.cube_pos()[2]` after segment 3): grasp too high → raise `TCP_OFFSET` in 5 mm steps; fingers pass through cube without contact → lower `GRIP_CLOSED` to 0.010; cube pushed away on descent → gripper not fully open, increase `GRIPPER_SETTLE_STEPS`.
3. Cube lifted but dropped mid-move: raise `finger` kp in `set_arm_gains` to 400, or slow `JOINT_SPEED` to 0.6.
4. Cube placed outside zone: PD tracking lag. Append `_hold(q_place, GRIP_CLOSED, 5)` before opening so the arm settles.
5. `is_success` false because gripper reads < 0.03 at the end: the retreat segment keeps `GRIP_OPEN`, so verify `_set_gripper_target` is receiving `action[6]` unclipped.

- [ ] **Step 5: Run the 20-seed success-rate test**

Run: `pixi run pytest tests/test_oracle.py -q -k success_rate -s`
Expected: `1 passed` (≥18/20). If between 15 and 17, apply the ladder above to the failing seeds (print the seed in `run_oracle_episode`).

- [ ] **Step 6: Commit**

```bash
git add rebot_sim/oracle.py tests/test_oracle.py pytest.ini
git commit -m "Add pyroboplan pick-and-place episode planner"
```

---

### Task 6: Collector → LeRobotDataset

**Files:**
- Create: `scripts/collect.py`
- Create: `tests/test_collect.py`

- [ ] **Step 1: Write the failing test**

`tests/test_collect.py`:
```python
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect  # noqa: E402


def test_collect_two_episodes(env, tmp_path):
    root = tmp_path / "ds"
    n_ok = collect.collect(env, n_episodes=2, root=root, repo_id="local/test_rebot", seed0=0)
    assert n_ok == 2
    ds = LeRobotDataset("local/test_rebot", root=root)
    assert ds.num_episodes == 2
    assert ds.fps == 10
    frame = ds[0]
    assert frame["observation.state"].shape == (7,)
    assert frame["action"].shape == (7,)
    assert frame["observation.images.front"].shape == (3, 256, 256)
    assert frame["observation.images.wrist"].shape == (3, 256, 256)
    assert frame["task"] == env.TASK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_collect.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect'`.

- [ ] **Step 3: Implement `scripts/collect.py`**

```python
"""Run the oracle through the env and write successful episodes as a LeRobot v3 dataset.

Usage:
    pixi run python scripts/collect.py --episodes 200 --root data/rebot_pick_place
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import rebot_sim  # noqa: F401
from rebot_sim import oracle
from rebot_sim.env import RebotPickPlaceEnv

STATE_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def features(image_size: int) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.front": img,
        "observation.images.wrist": img,
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    }


def collect(env: RebotPickPlaceEnv, n_episodes: int, root: Path, repo_id: str, seed0: int = 0) -> int:
    """Collect until `n_episodes` successes. Returns the number saved."""
    if root.exists():
        shutil.rmtree(root)
    image_size = env.observation_space["pixels"]["front"].shape[0]
    ds = LeRobotDataset.create(repo_id=repo_id, fps=10, features=features(image_size), root=root, robot_type="rebot_arm_b601_dm")
    saved, seed = 0, seed0
    while saved < n_episodes:
        obs, _ = env.reset(seed=seed)
        actions = oracle.plan_episode(env, seed=seed)
        term = False
        for a in actions:
            ds.add_frame(
                {
                    "observation.images.front": obs["pixels"]["front"],
                    "observation.images.wrist": obs["pixels"]["wrist"],
                    "observation.state": obs["agent_pos"],
                    "action": np.asarray(a, dtype=np.float32),
                    "task": env.TASK,
                }
            )
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if term:
            ds.save_episode()
            saved += 1
            print(f"seed {seed}: success ({saved}/{n_episodes})")
        else:
            ds.clear_episode_buffer()
            print(f"seed {seed}: failed, discarded")
        seed += 1
    ds.finalize()
    return saved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--root", type=Path, default=Path("data/rebot_pick_place"))
    p.add_argument("--repo-id", default="local/rebot_pick_place")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    env = RebotPickPlaceEnv()
    n = collect(env, args.episodes, args.root, args.repo_id, args.seed)
    print(f"saved {n} episodes to {args.root}")


if __name__ == "__main__":
    main()
```

`clear_episode_buffer()` is the LeRobot 0.6.1 method that drops the in-progress episode and its temp images.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_collect.py -q`
Expected: `1 passed`. Video encoding needs ffmpeg on PATH (conda `ffmpeg` from Task 0). If `add_frame` complains about image dtype/shape, it wants HxWx3 uint8, which `_get_obs` already returns.

- [ ] **Step 5: Commit**

```bash
git add scripts/collect.py tests/test_collect.py
git commit -m "Add oracle data collector writing LeRobotDataset"
```

---

### Task 7: LeRobot EnvConfig plugin for `lerobot-eval`

**Files:**
- Create: `rebot_sim/lerobot_env.py`
- Create: `tests/test_lerobot_env.py`

- [ ] **Step 1: Write the failing test**

`tests/test_lerobot_env.py`:
```python
import numpy as np
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import preprocess_observation

import rebot_sim  # noqa: F401


def test_rebot_env_config_registered_and_builds_vector_env():
    cfg = EnvConfig.get_choice_class("rebot")()
    assert cfg.gym_id == "gym_rebot/RebotPickPlace-v0"
    envs = cfg.create_envs(n_envs=1)
    vec = envs["rebot"][0]
    obs, info = vec.reset(seed=[0])
    batch = preprocess_observation(obs)
    assert batch["observation.images.front"].shape == (1, 3, 256, 256)
    assert batch["observation.images.wrist"].shape == (1, 3, 256, 256)
    assert batch["observation.state"].shape == (1, 7)
    assert vec.call("_max_episode_steps")[0] == 300
    assert vec.call("task_description")[0] == "pick up the red cube and place it in the green square"
    a = np.tile(vec.single_action_space.sample(), (1, 1))
    _, _, _, _, info = vec.step(a)
    assert "is_success" in info
    vec.close()
```

`create_envs(n_envs, use_async_envs=False)` is the `EnvConfig` method `lerobot-eval` calls; it returns `{env_type: {task_id: VectorEnv}}`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_lerobot_env.py -q`
Expected: FAIL with `ValueError`/`KeyError` mentioning `rebot` not registered.

- [ ] **Step 3: Implement the config**

`rebot_sim/lerobot_env.py`:
```python
"""LeRobot EnvConfig so `lerobot-eval --env.type=rebot --env.discover_packages_path=rebot_sim` works."""

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@EnvConfig.register_subclass("rebot")
@dataclass
class RebotEnv(EnvConfig):
    task: str | None = "RebotPickPlace-v0"
    fps: int = 10
    episode_length: int = 300
    image_size: int = 256
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "pixels/front": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
            "pixels/wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/front": f"{OBS_IMAGES}.front",
            "pixels/wrist": f"{OBS_IMAGES}.wrist",
        }
    )

    def __post_init__(self):
        shape = (self.image_size, self.image_size, 3)
        self.features["pixels/front"] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
        self.features["pixels/wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)

    @property
    def gym_kwargs(self) -> dict:
        return {"image_size": self.image_size, "max_episode_steps": self.episode_length}
```

`EnvConfig.package_name` is `gym_rebot` and `gym_id` is `gym_rebot/RebotPickPlace-v0`; the id is already registered by `rebot_sim/__init__.py`, so LeRobot never tries to import a `gym_rebot` package.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_lerobot_env.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add rebot_sim/lerobot_env.py tests/test_lerobot_env.py
git commit -m "Register rebot EnvConfig for lerobot-eval"
```

---

### Task 8: Collect the dataset

**Files:**
- Output: `data/rebot_pick_place/` (git-ignored)

- [ ] **Step 1: Collect 200 episodes**

Run: `pixi run python scripts/collect.py --episodes 200 --root data/rebot_pick_place`
Expected: per-seed lines ending with `saved 200 episodes to data/rebot_pick_place`. Note the number of discarded seeds; if more than ~15% fail, stop and apply the Task 5 debugging ladder before training.

- [ ] **Step 2: Inspect**

Run:
```bash
pixi run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/rebot_pick_place', root='data/rebot_pick_place')
print(ds.num_episodes, ds.num_frames, ds.fps, list(ds.features))"
pixi run lerobot-dataset-viz --repo-id local/rebot_pick_place --root data/rebot_pick_place --episode-index 0
```
Expected: `200 <~20000> 10 [...]`, and a rerun window showing both camera streams with the arm picking and placing. (`lerobot-dataset-viz` flag names come from its draccus config; run with `--help` if they differ.)

No commit (data is ignored).

---

### Task 9: Train SmolVLA

**Files:**
- Create: `scripts/train.sh`

- [ ] **Step 1: Write the script**

`scripts/train.sh`:
```bash
#!/usr/bin/env bash
# Fine-tune SmolVLA on the collected oracle dataset. ~8 GB VRAM: batch 8 bf16; drop to 4 on OOM.
set -euo pipefail
cd "$(dirname "$0")/.."
exec pixi run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/rebot_pick_place \
  --dataset.root=data/rebot_pick_place \
  --batch_size="${BATCH_SIZE:-8}" \
  --steps="${STEPS:-20000}" \
  --save_freq=5000 \
  --output_dir=outputs/train/smolvla_rebot \
  --job_name=smolvla_rebot \
  --wandb.enable=false \
  "$@"
```

`chmod +x scripts/train.sh`.

- [ ] **Step 2: Smoke-run 20 steps**

Run: `STEPS=20 scripts/train.sh --output_dir=outputs/train/smoke`
Expected: downloads `lerobot/smolvla_base`, prints loss lines, writes `outputs/train/smoke/checkpoints/`. On CUDA OOM: `BATCH_SIZE=4 STEPS=20 scripts/train.sh ...`. If it complains about a mismatched flag name, `pixi run lerobot-train --help` and fix the script.

- [ ] **Step 3: Full run**

Run: `scripts/train.sh`
Expected: finishes 20k steps (hours on a laptop 5070; leave it). Final checkpoint at `outputs/train/smolvla_rebot/checkpoints/last/pretrained_model`.

- [ ] **Step 4: Commit the script**

```bash
git add scripts/train.sh
git commit -m "Add SmolVLA fine-tuning script"
```

---

### Task 10: Evaluate

**Files:**
- Create: `scripts/eval.sh`

- [ ] **Step 1: Write the script**

`scripts/eval.sh`:
```bash
#!/usr/bin/env bash
# Evaluate a checkpoint in the Genesis env with stock lerobot-eval.
# Usage: scripts/eval.sh [checkpoint_dir] [n_episodes]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/train/smolvla_rebot/checkpoints/last/pretrained_model}"
N="${2:-50}"
exec pixi run lerobot-eval \
  --env.type=rebot \
  --env.discover_packages_path=rebot_sim \
  --policy.path="$CKPT" \
  --policy.device=cuda \
  --eval.n_episodes="$N" \
  --eval.batch_size=1 \
  --output_dir=outputs/eval/smolvla_rebot
```

`chmod +x scripts/eval.sh`.

- [ ] **Step 2: Smoke-run on the 20-step checkpoint**

Run: `scripts/eval.sh outputs/train/smoke/checkpoints/last/pretrained_model 2`
Expected: 2 episodes roll out (likely 0% success), `outputs/eval/smolvla_rebot/eval_info.json` written and videos under `outputs/eval/smolvla_rebot/videos/`. Failure modes:
- `rebot` not a registered env type → the plugin path did not load; check `--env.discover_packages_path=rebot_sim` spelling and that `rebot_sim/__init__.py` imports `lerobot_env`.
- Feature name mismatch between policy and env (e.g. policy expects `observation.images.camera1`) → SmolVLA base was trained with other camera names; `lerobot-eval` has `--rename_map` for this, e.g. `--rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'`. Fine-tuning from `--policy.path` normally adopts the dataset's names, so this should not trigger; if it does, apply the same map in `train.sh`.

- [ ] **Step 3: Full eval**

Run: `scripts/eval.sh`
Expected: 50 episodes, `pc_success` printed and in `eval_info.json`. Record the number.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval.sh
git commit -m "Add lerobot-eval wrapper for the rebot env"
```

---

### Task 11: README and full test pass

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `pixi run pytest -q tests`
Expected: all pass (the slow oracle test included).

- [ ] **Step 2: Update README**

Replace `README.md` with:
````markdown
# reBot Arm B601-DM — Genesis VLA testbed

Genesis simulation of the [Seeed reBot Arm B601-DM](https://github.com/Seeed-Projects/reBot-DevArm)
set up as a pick-and-place testbed for fine-tuning vision-language-action models.

Task: *pick up the red cube and place it in the green square*. Cube position and yaw are
randomized each episode; the zone is fixed.

```bash
pixi install
pixi run python sim.py                 # viewer
pixi run python sim.py --headless      # saves frame.png, asserts tracking
pixi run pytest -q tests               # headless tests (slow oracle test included)

pixi run python scripts/collect.py --episodes 200   # oracle demos -> data/rebot_pick_place
scripts/train.sh                                    # fine-tune SmolVLA (BATCH_SIZE, STEPS env vars)
scripts/eval.sh [checkpoint] [n_episodes]           # lerobot-eval in the sim
```

## Layout

- `rebot_sim/scene.py` — constants and Genesis builders (room, table, arm, cube, zone).
- `rebot_sim/env.py` — `gym_rebot/RebotPickPlace-v0`. Obs: `pixels/front`, `pixels/wrist` (256x256 RGB),
  `agent_pos` (6 joints + gripper). Action: 7 absolute joint targets + gripper opening, 10 Hz.
- `rebot_sim/oracle.py` — pyroboplan (Pinocchio IK + RRT-Connect + Cartesian) pick-and-place planner.
- `rebot_sim/lerobot_env.py` — LeRobot `EnvConfig` (`--env.type=rebot --env.discover_packages_path=rebot_sim`).
- `scripts/collect.py` — runs the oracle, writes a LeRobot v3 dataset, drops failed episodes.

## Baseline

| Policy | Episodes | Success |
|---|---|---|
| Oracle (pyroboplan) | 20 | fill in from tests/test_oracle.py |
| SmolVLA fine-tuned, 200 demos, 20k steps | 50 | fill in from scripts/eval.sh |

## Notes

- `assets/rebot_arm_dm/` is `Rebot_Arm_description/DM/` from upstream (CERN-OHL-W v2, see LICENSE there).
  Local change: added `<inertial>` to both finger links (upstream omits them; Genesis' MuJoCo parser rejects massless moving bodies).
- 8 dofs: joint1..6 + finger_left/finger_right. Genesis ignores the URDF `<mimic>`, so drive right = -left.
- joint2/joint3 limits are [-3.14, 0]; "up" is negative.
- Pinocchio needs absolute mesh paths and has no SRDF for this arm; `oracle.load_model` patches both.
- Tunables marked `ponytail:` in `oracle.py` (`TCP_OFFSET`, `GRIP_CLOSED`) and `scene.py` (PD gains).
````

Fill in the two success numbers from Task 5 step 5 and Task 10 step 3. If Task 0 needed the `--no-deps` fallback, add a line under the install command.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document VLA testbed workflow and baseline results"
```

---

## Self-review

- **Spec coverage.** Env (Task 2–3), oracle (4–5), collection (6, 8), training (9), eval via EnvConfig plugin (7, 10), layout/deps/README (0, 11). Spec's "grasp verification after lift" is covered by the env success flag plus the debugging ladder rather than a separate check; the collector discards any non-successful episode, which is the behavior the spec wants.
- **Placeholders.** None; the two README "fill in" cells are instructions to paste measured numbers.
- **Type consistency.** `env.cube_pos()`, `env.cube_quat()`, `env.joint_pos()`, `env.gripper_opening()`, `env.zone_world`, `env.TASK`, `env.task_description` are defined in Task 2 and used in Tasks 3, 5, 6, 7. `oracle.load_model`, `in_collision`, `top_down_pose`, `solve_ik`, `plan_episode`, `TCP_FRAME`, `APPROACH` defined in Tasks 4–5 and used in tests and `collect.py`. Obs keys `pixels/front`, `pixels/wrist`, `agent_pos` match `features_map` in Task 7 and `preprocess_observation`. Dataset keys `observation.images.front/wrist`, `observation.state`, `action` match SmolVLA expectations.
