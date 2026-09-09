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
from scipy.spatial.transform import Rotation

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
