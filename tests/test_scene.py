import numpy as np

from rebot_sim import scene


def test_constants_consistent():
    assert scene.URDF.exists()
    assert len(scene.ARM_JOINTS) == 6
    assert len(scene.HOME) == 6
    assert np.all(scene.JOINT_LOWER <= scene.HOME) and np.all(scene.HOME <= scene.JOINT_UPPER)


def test_local_to_world_puts_point_on_table():
    p = scene.local_to_world((0.0, 0.0))
    assert np.allclose(p, scene.ARM_POS)
