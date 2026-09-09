"""reBot Arm B601-DM pick-and-place sim package.

Importing this package registers `gym_rebot/RebotPickPlace-v0` and, when lerobot is
installed, the `rebot` EnvConfig used by `lerobot-eval`.
"""

from gymnasium.envs.registration import register

register(
    id="gym_rebot/RebotPickPlace-v0",
    entry_point="rebot_sim.env:RebotPickPlaceEnv",
    max_episode_steps=300,
)

try:
    from rebot_sim import lerobot_env  # noqa: F401
except ImportError:  # lerobot not installed or lerobot_env not written yet
    pass
