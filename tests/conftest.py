import pytest

from rebot_sim.env import RebotPickPlaceEnv


@pytest.fixture(scope="session")
def env():
    return RebotPickPlaceEnv()
