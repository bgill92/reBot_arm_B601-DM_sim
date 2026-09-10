"""Run the oracle in the Genesis viewer, one seed per episode, nothing saved.

Usage:
    pixi run python scripts/watch_oracle.py --episodes 5 --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebot_sim  # noqa: E402, F401
from rebot_sim import oracle
from rebot_sim.env import RebotPickPlaceEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="no viewer (smoke test)")
    args = p.parse_args()
    env = RebotPickPlaceEnv(show_viewer=not args.headless)
    for seed in range(args.seed, args.seed + args.episodes):
        env.reset(seed=seed)
        term = False
        for a in oracle.plan_episode(env, seed=seed):
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        print(f"seed {seed}: {'success' if term else 'fail'}")


if __name__ == "__main__":
    main()
