"""reBot Arm B601-DM pick-and-place sim package.

Importing this package registers `gym_rebot/RebotPickPlace-v0` and, when lerobot is
installed, the `rebot` EnvConfig used by `lerobot-eval`.
"""

import importlib.util

from gymnasium.envs.registration import register

register(
    id="gym_rebot/RebotPickPlace-v0",
    entry_point="rebot_sim.env:RebotPickPlaceEnv",
    max_episode_steps=300,
)

# lerobot is optional: the env and oracle work without it; only lerobot-eval needs the config.
if importlib.util.find_spec("lerobot") is not None:
    from rebot_sim import lerobot_env  # noqa: F401
