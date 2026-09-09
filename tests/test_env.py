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
