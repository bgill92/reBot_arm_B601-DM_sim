"""Constants and Genesis scene builders shared by the viewer, the gym env, and the oracle."""

from pathlib import Path

import genesis as gs
import numpy as np
import torch

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
# Front camera: 3/4 view ~1.3 m from the workspace center so the cube fills more than a few pixels
# at 256^2 while the whole arm stays in frame through reach and lift.
CAM_LOOKAT = (-0.25, 0.0, TABLE_HEIGHT + 0.05)  # cube-range center in world coords, just above the table
CAM_POS = (0.45, -0.9, TABLE_HEIGHT + 0.65)

# Task objects. xy ranges are in the arm-base frame (x forward, y left).
CUBE_SIZE = 0.03
CUBE_X_RANGE = (0.15, 0.35)
CUBE_Y_RANGE = (-0.15, 0.15)
ZONE_SIZE = 0.10
ZONE_XY = (0.25, -0.25)
ZONE_THICKNESS = 0.002
ZONE_Z_OFFSET = 0.001  # Lift the marker slightly to avoid z-fighting with the table top.

COLOR_FLOOR = (0.55, 0.45, 0.35)
COLOR_WALL = (0.85, 0.85, 0.8)
COLOR_WOOD = (0.6, 0.42, 0.25)
COLOR_CUBE = (0.9, 0.1, 0.1)
COLOR_ZONE = (0.1, 0.8, 0.1)


# `gs._initialized` is a private Genesis flag; the library exposes no public
# equivalent to check whether gs.init() has already run.
def init_genesis() -> None:
    if not gs._initialized:
        gs.init(backend=gs.gpu)
        # gs.init installs a global torch default-device mode (CUDA). That mode also rewrites
        # device-less calls like torch.as_tensor in third-party code (lerobot's processors) and
        # breaks lerobot-eval. Genesis places its own tensors explicitly, so removing the mode
        # is safe. Passing None removes the mode instead of swapping in a CPU one.
        torch.set_default_device(None)


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
    # decompose_robot_error_threshold: Genesis defaults robots to one convex hull per mesh, which turns
    # each gripper finger into a wedge (the crossed rack juts inward at the base) that squirts the cube
    # out of the jaws instead of pinching it. Convex decomposition restores the flat, parallel pads.
    # 0.15 is Genesis' own default threshold for rigid objects; only robots default to no decomposition.
    return scene.add_entity(
        gs.morphs.URDF(file=str(URDF), fixed=True, pos=tuple(ARM_POS), decompose_robot_error_threshold=0.15)
    )


def add_cube(scene: gs.Scene):
    pos = local_to_world((CUBE_X_RANGE[0], 0.0)) + np.array([0, 0, CUBE_SIZE / 2])
    return _box(scene, (CUBE_SIZE,) * 3, pos, COLOR_CUBE)


def add_zone(scene: gs.Scene):
    pos = local_to_world(ZONE_XY) + np.array([0, 0, ZONE_Z_OFFSET])
    # collision=False: this is a visual-only marker, so the cube and fingers pass through it.
    return _box(scene, (ZONE_SIZE, ZONE_SIZE, ZONE_THICKNESS), pos, COLOR_ZONE, fixed=True, collision=False)


def set_arm_gains(arm, arm_dofs, finger_dofs) -> None:
    # ponytail: rough PD gains, tune against real Damiao 4340P/4310 MIT-mode response if needed.
    arm.set_dofs_kp(np.array([400, 400, 400, 100, 100, 100]), arm_dofs)
    arm.set_dofs_kv(np.array([40, 40, 40, 10, 10, 10]), arm_dofs)
    arm.set_dofs_kp(np.array([200, 200]), finger_dofs)
    arm.set_dofs_kv(np.array([10, 10]), finger_dofs)
