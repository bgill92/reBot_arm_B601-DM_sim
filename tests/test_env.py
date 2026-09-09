import gymnasium as gym
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


def test_reset_starts_at_home_with_gripper_open(env):
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs["agent_pos"][:6], S.HOME, atol=0.05)
    assert abs(obs["agent_pos"][6] - S.GRIPPER_MAX) < 0.005
