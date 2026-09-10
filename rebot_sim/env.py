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

LIFT_HEIGHT = 0.04  # Cube must rise this far above the table at some point to count as picked up.


class _CamWindow:
    """One tkinter window showing front | wrist side by side, refreshed on every observation."""

    # ponytail: tkinter, not Genesis camera GUI=True. lerobot pulls opencv-python-headless, which
    # shadows genesis-world's opencv-python, so cv2.imshow raises. Swap for GUI=True if that changes.
    def __init__(self) -> None:
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title("rebot cameras: front | wrist")
        self._label = tk.Label(self._root)
        self._label.pack()

    def show(self, front: np.ndarray, wrist: np.ndarray) -> None:
        from PIL import Image, ImageTk

        img = ImageTk.PhotoImage(Image.fromarray(np.hstack([front, wrist])))
        self._label.configure(image=img)
        self._label.image = img  # tkinter keeps only a weak ref
        self._root.update()


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
            vis_options=S.vis_options(),
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
        self._cam_window = _CamWindow() if show_viewer else None

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

    # Both fingers are driven symmetrically, so the left finger alone is the opening.
    def gripper_opening(self) -> float:
        return float(self.arm.get_dofs_position(self.finger_dofs).cpu().numpy().reshape(2)[0])

    def _set_gripper_target(self, opening: float) -> None:
        g = float(np.clip(opening, 0.0, S.GRIPPER_MAX))
        self.arm.control_dofs_position(np.array([g, -g]), self.finger_dofs)

    # ----- gym API -----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.arm.set_dofs_position(S.HOME, self.arm_dofs, zero_velocity=True)
        self.arm.set_dofs_position(np.array([S.GRIPPER_MAX, -S.GRIPPER_MAX]), self.finger_dofs, zero_velocity=True)
        self.arm.control_dofs_position(S.HOME, self.arm_dofs)
        self._set_gripper_target(S.GRIPPER_MAX)

        x = self.np_random.uniform(*S.CUBE_X_RANGE)
        y = self.np_random.uniform(*S.CUBE_Y_RANGE)
        # The cube has 4-fold symmetry, so yaw in [0, pi/2) covers every distinct grasp orientation.
        yaw = self.np_random.uniform(0.0, np.pi / 2)
        # Spawn 2 mm above the table so the settle steps drop it into clean contact.
        pos = S.local_to_world((x, y)) + np.array([0.0, 0.0, S.CUBE_SIZE / 2 + 0.002])
        quat = Rotation.from_euler("z", yaw).as_quat(scalar_first=True)
        self.cube.set_pos(pos)
        self.cube.set_quat(quat)
        self._lifted = False
        for _ in range(20):  # settle
            self.scene.step()
        return self._get_obs(), {"is_success": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(7)
        q = np.clip(action[:6], S.JOINT_LOWER, S.JOINT_UPPER)
        self.arm.control_dofs_position(q, self.arm_dofs)
        self._set_gripper_target(float(action[6]))
        for _ in range(self.substeps):
            self.scene.step()
        if self.cube_pos()[2] - S.TABLE_HEIGHT - S.CUBE_SIZE / 2 > LIFT_HEIGHT:
            self._lifted = True
        success = self.is_success()
        return self._get_obs(), float(success), success, False, {"is_success": success}

    def is_success(self) -> bool:
        """Success = the cube was lifted at least LIFT_HEIGHT above the table at some point,
        now rests inside the zone, and the gripper has released it."""
        c = self.cube_pos()
        in_zone = np.all(np.abs(c[:2] - self.zone_world[:2]) <= S.ZONE_SIZE / 2)
        on_table = abs(c[2] - (S.TABLE_HEIGHT + S.CUBE_SIZE / 2)) < 0.01
        released = self.gripper_opening() > 0.03
        return bool(in_zone and on_table and released and self._lifted)

    def _update_wrist_cam(self) -> None:
        p = self.link6.get_pos().cpu().numpy().reshape(3)
        q = self.link6.get_quat().cpu().numpy().reshape(4)
        R = Rotation.from_quat(q, scalar_first=True).as_matrix()
        self.cam_wrist.set_pose(pos=p + R @ WRIST_CAM_OFFSET, lookat=p + R @ WRIST_CAM_LOOKAT, up=R @ WRIST_CAM_UP)

    def _render_front(self) -> np.ndarray:
        return np.asarray(self.cam_front.render(rgb=True)[0])[..., :3]

    def _get_obs(self) -> dict:
        self._update_wrist_cam()
        front = self._render_front()
        wrist = np.asarray(self.cam_wrist.render(rgb=True)[0])[..., :3]
        if self._cam_window is not None:
            self._cam_window.show(front, wrist)
        state = np.append(self.joint_pos(), self.gripper_opening()).astype(np.float32)
        return {"pixels": {"front": front, "wrist": wrist}, "agent_pos": state}

    def render(self):
        return self._render_front()
