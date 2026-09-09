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
# ponytail: extra grasp depth along the tool axis; end_link already sits at the fingertip plane, so
# 0 grasps at the cube centre. Tune if the gripper grasps too high/low. Positive moves it deeper.
TCP_OFFSET = 0.0
APPROACH = 0.08
GRIP_OPEN = S.GRIPPER_MAX
GRIP_CLOSED = 0.012  # ponytail: per-finger opening that squeezes the 3 cm cube; tune on hardware.
CONTROL_DT = 0.1  # 10 Hz, matches env substeps * sim dt
JOINT_SPEED = 1.0  # rad/s used to time joint-space segments
# The arm has a few mm of gravity droop at the grasp pose, so hold still long enough for both the
# fingers to seat on the cube and the PD to settle before moving again. 8 also works; 12 is margin.
GRIPPER_SETTLE_STEPS = 12


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
    return pin.SE3(R, np.asarray(p_local, dtype=float) - np.array([0.0, 0.0, TCP_OFFSET]))


def solve_ik(model, coll, target: pin.SE3, q_init: np.ndarray, seed: int = 0):
    q_init = np.array(q_init, dtype=float)  # pyroboplan's DifferentialIk mutates init_state in place.
    ik = DifferentialIk(
        model,
        data=model.createData(),
        collision_model=coll,
        options=DifferentialIkOptions(max_iters=500, max_retries=20, rng_seed=seed),
    )
    return ik.solve(TCP_FRAME, target, init_state=q_init)


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
        # .copy(): oMf is a live reference into `data`, which the IK solver below shares and overwrites,
        # so an uncopied start pose would drift mid-plan and skew the waypoint spacing.
        start = data.oMf[self.model.getFrameId(TCP_FRAME)].copy()
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
            # ponytail: linear (not trapezoidal) time scaling - pyroboplan's trapezoidal profile returns
            # alpha = 1.0000000000000002 at the final sample, which scipy Slerp rejects with a ValueError.
            # 0.2 m/s (pyroboplan defaults to 1.0) spreads the 8 cm approach over ~5 waypoints, so the PD
            # tracks an actual straight line instead of jumping between the two endpoints.
            options=CartesianPlannerOptions(use_trapezoidal_scaling=False, max_linear_velocity=0.2),
        )
        # np.array: generate() feeds q_start to DifferentialIk, which mutates its init_state in place.
        ok, _, q = planner.generate(np.array(q_start, dtype=float), CONTROL_DT)
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
    action = np.append(q, opening).astype(np.float32)
    return [action.copy() for _ in range(n)]


def _fail(stage: str) -> list:
    print(f"oracle: {stage} planning failed")
    return []


def plan_episode(env, seed: int = 0) -> list[np.ndarray]:
    """Plan a full pick-and-place from the env's current (privileged) state.

    Returns the list of 7-d actions (6 joint targets + gripper opening) to feed to env.step,
    or an empty list if any segment fails to plan.
    """
    planner = _get_planner()
    cube_local = env.cube_pos() - S.ARM_POS
    yaw = Rotation.from_quat(env.cube_quat(), scalar_first=True).as_euler("xyz")[2]
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # cube is symmetric every 90 deg
    zone_local = env.zone_world - S.ARM_POS
    up = np.array([0.0, 0.0, APPROACH])

    grasp = top_down_pose(cube_local, yaw)
    pre_grasp = top_down_pose(cube_local + up, yaw)
    place = top_down_pose(zone_local + np.array([0.0, 0.0, S.CUBE_SIZE / 2]), 0.0)
    pre_place = top_down_pose(zone_local + np.array([0.0, 0.0, S.CUBE_SIZE / 2]) + up, 0.0)

    q0 = env.joint_pos()
    q_pre_grasp = solve_ik(planner.model, planner.coll, pre_grasp, q0, seed)
    q_pre_place = solve_ik(planner.model, planner.coll, pre_place, q0, seed)
    if q_pre_grasp is None or q_pre_place is None:
        return _fail("pre-grasp/pre-place IK")

    actions: list[np.ndarray] = []

    # 1. free-space move above the cube
    seg = planner.joint_path(q0, q_pre_grasp, seed)
    if seg is None:
        return _fail("move to pre-grasp")
    actions += _with_gripper(seg, GRIP_OPEN)

    # 2. straight down, close
    seg = planner.cartesian_path(q_pre_grasp, grasp)
    if seg is None:
        return _fail("descend to grasp")
    actions += _with_gripper(seg, GRIP_OPEN)
    q_grasp = seg[-1]
    actions += _hold(q_grasp, GRIP_CLOSED)

    # 3. straight up (fingers closed)
    seg = planner.cartesian_path(q_grasp, pre_grasp)
    if seg is None:
        return _fail("lift")
    actions += _with_gripper(seg, GRIP_CLOSED)
    q_lift = seg[-1]

    # 4. free-space move above the zone
    seg = planner.joint_path(q_lift, q_pre_place, seed)
    if seg is None:
        return _fail("move to pre-place")
    actions += _with_gripper(seg, GRIP_CLOSED)

    # 5. straight down, open
    seg = planner.cartesian_path(q_pre_place, place)
    if seg is None:
        return _fail("descend to place")
    actions += _with_gripper(seg, GRIP_CLOSED)
    q_place = seg[-1]
    actions += _hold(q_place, GRIP_OPEN)

    # 6. retreat
    seg = planner.cartesian_path(q_place, pre_place)
    if seg is None:
        return _fail("retreat")
    actions += _with_gripper(seg, GRIP_OPEN)
    return actions
