import numpy as np
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import preprocess_observation

import rebot_sim  # noqa: F401


def test_rebot_env_config_registered_and_builds_vector_env():
    cfg = EnvConfig.get_choice_class("rebot")()
    assert cfg.gym_id == "gym_rebot/RebotPickPlace-v0"
    envs = cfg.create_envs(n_envs=1)
    vec = envs["rebot"][0]
    try:
        obs, info = vec.reset(seed=[0])
        batch = preprocess_observation(obs)
        assert batch["observation.images.front"].shape == (1, 3, 256, 256)
        assert batch["observation.images.wrist"].shape == (1, 3, 256, 256)
        assert batch["observation.state"].shape == (1, 7)
        assert vec.call("_max_episode_steps")[0] == 300
        assert vec.call("task_description")[0] == "pick up the red cube and place it in the green square"
        a = np.tile(vec.single_action_space.sample(), (1, 1))
        _, _, _, _, info = vec.step(a)
        assert "is_success" in info
    finally:
        vec.close()
