"""Run the oracle through the env and write successful episodes as a LeRobot v3 dataset.

Recreates `--root` from scratch, deleting a previous dataset there.

Usage:
    pixi run python scripts/collect.py --episodes 200 --root data/rebot_pick_place
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Running as `python scripts/collect.py` puts scripts/ (not the repo root) on sys.path,
# so rebot_sim would not be importable without this; pytest avoids it via pytest.ini's
# `pythonpath = .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebot_sim  # noqa: E402, F401
from rebot_sim import oracle
from rebot_sim.env import RebotPickPlaceEnv

STATE_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def features(image_size: int) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.front": img,
        "observation.images.wrist": img,
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    }


def collect(env: RebotPickPlaceEnv, n_episodes: int, root: Path, repo_id: str, seed0: int = 0) -> int:
    """Collect until `n_episodes` successes. Returns the number saved."""
    if root.exists():
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"{root} exists and does not look like a LeRobot dataset; refusing to delete it.")
        shutil.rmtree(root)
    image_size = env.observation_space["pixels"]["front"].shape[0]
    fps = env.metadata["render_fps"]
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features(image_size), root=root, robot_type="rebot_arm_b601_dm")
    saved, seed = 0, seed0
    while saved < n_episodes:
        obs, _ = env.reset(seed=seed)
        actions = oracle.plan_episode(env, seed=seed)
        term = False
        for a in actions:
            ds.add_frame(
                {
                    "observation.images.front": obs["pixels"]["front"],
                    "observation.images.wrist": obs["pixels"]["wrist"],
                    "observation.state": obs["agent_pos"],
                    "action": np.asarray(a, dtype=np.float32),
                    "task": env.TASK,
                }
            )
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if term:
            ds.save_episode()
            saved += 1
            print(f"seed {seed}: success ({saved}/{n_episodes})")
        else:
            ds.clear_episode_buffer()
            print(f"seed {seed}: failed, discarded")
        seed += 1
    ds.finalize()
    return saved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--root", type=Path, default=Path("data/rebot_pick_place"))
    p.add_argument("--repo-id", default="local/rebot_pick_place")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    env = RebotPickPlaceEnv()
    n = collect(env, args.episodes, args.root, args.repo_id, args.seed)
    print(f"saved {n} episodes to {args.root}")


if __name__ == "__main__":
    main()
