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
