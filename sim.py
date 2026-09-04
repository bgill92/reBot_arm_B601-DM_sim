"""Genesis simulation of the Seeed reBot Arm B601-DM.

Usage:
    pixi run python sim.py            # interactive viewer
    pixi run python sim.py --headless # no viewer, saves frame.png
"""

import argparse
import sys

import numpy as np
import genesis as gs

URDF = "assets/rebot_arm_dm/urdf/ReBot_Arm_DM.urdf"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["finger_left", "finger_right"]  # right mimics left with multiplier -1
GRIPPER_MAX = 0.05

# Joint-space waypoints (rad). joint2/joint3 limits are [-3.14, 0], so "up" is negative.
HOME = np.array([0.0, -1.57, -1.57, 0.0, 0.0, 0.0])
WAYPOINTS = [
    HOME,
    np.array([0.8, -1.0, -2.0, 0.5, 0.5, 1.0]),
    np.array([-0.8, -2.0, -1.0, -0.5, -0.5, -1.0]),
    HOME,
]


# Room: 4 m x 4 m, 2.6 m high, arm on a table at the center.
ROOM_W, ROOM_D, ROOM_H = 4.0, 4.0, 2.6
WALL_T = 0.05
TABLE_HEIGHT = 0.75
TABLE_W, TABLE_D, TABLE_T = 1.2, 0.8, 0.04
LEG_S = 0.05
ARM_POS = (-(TABLE_W / 2 - 0.1), 0.0, TABLE_HEIGHT)  # near the back (-x) edge; arm reaches +x
CAM_POS = (1.6, -1.8, 1.5)
CAM_LOOKAT = (-0.1, 0.0, TABLE_HEIGHT + 0.25)


def build_room(scene: gs.Scene) -> None:
    def box(size, pos, color):
        scene.add_entity(
            gs.morphs.Box(size=size, pos=pos, fixed=True),
            surface=gs.surfaces.Default(color=color),
        )

    floor = (0.55, 0.45, 0.35)
    wall = (0.85, 0.85, 0.8)
    wood = (0.6, 0.42, 0.25)

    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=floor))
    hw, hd, hh = ROOM_W / 2, ROOM_D / 2, ROOM_H / 2
    box((ROOM_W, WALL_T, ROOM_H), (0, hd, hh), wall)  # back
    box((ROOM_W, WALL_T, ROOM_H), (0, -hd, hh), wall)  # front
    box((WALL_T, ROOM_D, ROOM_H), (-hw, 0, hh), wall)  # left
    box((WALL_T, ROOM_D, ROOM_H), (hw, 0, hh), wall)  # right

    box((TABLE_W, TABLE_D, TABLE_T), (0, 0, TABLE_HEIGHT - TABLE_T / 2), wood)
    lx, ly = TABLE_W / 2 - LEG_S, TABLE_D / 2 - LEG_S
    leg_h = TABLE_HEIGHT - TABLE_T
    for sx in (-1, 1):
        for sy in (-1, 1):
            box((LEG_S, LEG_S, leg_h), (sx * lx, sy * ly, leg_h / 2), wood)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=CAM_POS, camera_lookat=CAM_LOOKAT, camera_fov=40
        ),
        show_viewer=not args.headless,
    )
    build_room(scene)
    arm = scene.add_entity(
        gs.morphs.URDF(file=URDF, fixed=True, pos=ARM_POS)
    )
    cam = scene.add_camera(
        res=(1280, 720), pos=CAM_POS, lookat=CAM_LOOKAT, fov=40, GUI=False
    )
    scene.build()

    arm_dofs = [arm.get_joint(n).dofs_idx_local[0] for n in ARM_JOINTS]
    finger_dofs = [arm.get_joint(n).dofs_idx_local[0] for n in FINGER_JOINTS]

    # ponytail: rough PD gains, tune against real Damiao 4340P/4310 MIT-mode response if needed.
    arm.set_dofs_kp(np.array([400, 400, 400, 100, 100, 100]), arm_dofs)
    arm.set_dofs_kv(np.array([40, 40, 40, 10, 10, 10]), arm_dofs)
    arm.set_dofs_kp(np.array([200, 200]), finger_dofs)
    arm.set_dofs_kv(np.array([10, 10]), finger_dofs)

    arm.set_dofs_position(HOME, arm_dofs)

    def gripper(opening: float) -> None:
        opening = float(np.clip(opening, 0.0, GRIPPER_MAX))
        arm.control_dofs_position(np.array([opening, -opening]), finger_dofs)

    steps_per_segment = 150
    for i in range(len(WAYPOINTS) - 1):
        a, b = WAYPOINTS[i], WAYPOINTS[i + 1]
        for t in range(steps_per_segment):
            s = t / steps_per_segment
            arm.control_dofs_position(a + (b - a) * s, arm_dofs)
            gripper(GRIPPER_MAX if i % 2 == 0 else 0.0)
            scene.step()
        if args.headless and i == 0:
            rgb, _, _, _ = cam.render(rgb=True)
            import imageio

            imageio.imwrite("frame.png", rgb)
            print("wrote frame.png")

    for _ in range(100):  # settle
        arm.control_dofs_position(HOME, arm_dofs)
        scene.step()
    q = arm.get_dofs_position(arm_dofs)
    err = np.abs(np.asarray(q.cpu()) - HOME).max()
    print(f"final joint error vs HOME: {err:.4f} rad")
    assert err < 0.05, "arm did not track waypoints"

    if not args.headless:
        while scene.viewer.is_alive():
            arm.control_dofs_position(HOME, arm_dofs)
            scene.step()
    sys.exit(0)


if __name__ == "__main__":
    main()
