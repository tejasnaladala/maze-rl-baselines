"""Quick verification: does wall-following still hit 100% on the
main-sweep test maze distribution (no is_solvable filter, seed_offset=10M)?
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from experiment_lib_v2 import make_maze, ego_features, ACTIONS, WALL, HAZARD
from maze_env_helpers import main_sweep_test_seeds, step_env
from launch_wall_following import WallFollower, DFSAgent
from launch_wall_follow_egofeats import EgoOnlyWallFollower

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def run_full_grid_wall_follower(side: str, size: int, seed: int) -> float:
    """Wall-follower with full grid access on main-sweep test mazes."""
    test_seeds = main_sweep_test_seeds(seed, 50)
    solved = 0
    for ms in test_seeds:
        maze = make_maze(size, ms)
        agent = WallFollower(side=side)
        agent.reset()
        ax, ay = 1, 1
        gx, gy = size - 2, size - 2
        max_steps = 4 * size * size
        for step in range(max_steps):
            action = agent.act(maze, (ay, ax))
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, size)
            ax, ay = new_ax, new_ay
            if (ax, ay) == (gx, gy):
                solved += 1
                break
    return solved / 50


def run_ego_wall_follower(side: str, size: int, seed: int) -> float:
    """Ego-only wall-follower on main-sweep test mazes."""
    test_seeds = main_sweep_test_seeds(seed, 50)
    solved = 0
    for ms in test_seeds:
        maze = make_maze(size, ms)
        agent = EgoOnlyWallFollower(side=side)
        agent.reset()
        ax, ay = 1, 1
        gx, gy = size - 2, size - 2
        max_steps = 4 * size * size
        action_hist: list = []
        for step in range(max_steps):
            obs = np.array(ego_features(maze, ax, ay, gx, gy, size, action_hist), dtype=np.float32)
            action = agent.act(obs)
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, size)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
    return solved / 50


def main() -> None:
    print("=== Wall-follower verification on main-sweep test distribution ===")
    print(f"  (no is_solvable filter, seed_offset=10_000_000)")
    print()
    for size in (9, 13, 17):
        rates_full = [run_full_grid_wall_follower("left", size, s) for s in SEEDS]
        rates_ego = [run_ego_wall_follower("left", size, s) for s in SEEDS]
        print(f"  Size {size}x{size}:")
        print(f"    WallFollowerLeft (full grid):    {100*np.mean(rates_full):>5.1f}% (sd {100*np.std(rates_full):.1f})")
        print(f"    EgoWallFollowerLeft (ego-only):  {100*np.mean(rates_ego):>5.1f}% (sd {100*np.std(rates_ego):.1f})")


if __name__ == "__main__":
    main()
