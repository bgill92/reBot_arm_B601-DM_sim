import sys
from pathlib import Path

import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect  # noqa: E402


def test_collect_two_episodes(env, tmp_path):
    root = tmp_path / "ds"
    n_ok = collect.collect(env, n_episodes=2, root=root, repo_id="local/test_rebot", seed0=0)
    assert n_ok == 2
    ds = LeRobotDataset("local/test_rebot", root=root)
    assert ds.num_episodes == 2
    assert ds.fps == 10
    frame = ds[0]
    assert frame["observation.state"].shape == (7,)
    assert frame["action"].shape == (7,)
    assert frame["observation.images.front"].shape == (3, 256, 256)
    assert frame["observation.images.wrist"].shape == (3, 256, 256)
    assert frame["task"] == env.TASK


def test_collect_refuses_to_delete_non_dataset_dir(env, tmp_path):
    root = tmp_path / "notds"
    root.mkdir()
    keep = root / "keep.txt"
    keep.write_text("do not delete")
    with pytest.raises(SystemExit):
        collect.collect(env, 1, root, "local/x")
    assert keep.exists()


def test_collect_gives_up_after_max_attempts(env, tmp_path, monkeypatch):
    monkeypatch.setattr(collect.oracle, "plan_episode", lambda env, seed=0: [])
    n = collect.collect(env, n_episodes=2, root=tmp_path / "ds2", repo_id="local/y", max_attempts=3)
    assert n == 0
