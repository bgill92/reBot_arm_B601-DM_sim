import pytest
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


def test_in_collision_detects_arm_through_table():
    model, coll = oracle.load_model()
    q = np.array([0.0, -2.9, -1.6, 0.0, 0.0, 0.0])  # shoulder swung back and down, wrist below the table top
    assert oracle.in_collision(model, coll, q)


def run_oracle_episode(env, seed):
    env.reset(seed=seed)
    actions = oracle.plan_episode(env, seed=seed)
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
