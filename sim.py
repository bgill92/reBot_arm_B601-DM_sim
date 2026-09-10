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
        vis_options=S.vis_options(),
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
