"""LeRobot EnvConfig so `lerobot-eval --env.type=rebot --env.discover_packages_path=rebot_sim` works."""

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@EnvConfig.register_subclass("rebot")
@dataclass
class RebotEnv(EnvConfig):
    task: str | None = "RebotPickPlace-v0"
    fps: int = 10
    episode_length: int = 300
    image_size: int = 256
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "pixels/front": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
            "pixels/wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/front": f"{OBS_IMAGES}.front",
            "pixels/wrist": f"{OBS_IMAGES}.wrist",
        }
    )

    def __post_init__(self):
        shape = (self.image_size, self.image_size, 3)
        self.features["pixels/front"] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
        self.features["pixels/wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)

    @property
    def gym_kwargs(self) -> dict:
        return {"image_size": self.image_size, "max_episode_steps": self.episode_length}
