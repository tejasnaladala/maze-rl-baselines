"""Canonical harness validation (per Codex Round 4 review).

Tests whether our custom test harness in launch_policy_distillation.py and
launch_cross_env_transfer.py reproduces the main-sweep baselines. If
Random != ~31.7% and NoBackRandom != ~52.2% at 9x9, the harness is invalid
and the distillation result is confounded.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze, is_solvable, ego_features, ACTIONS, WALL, HAZARD,
    NoBacktrackRandomAgent, RandomAgent, FeatureQAgent,
    set_all_seeds, run_experiment,
)
from maze_env_helpers import get_obs, step_env


SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
NUM_TEST = 50


def custom_harness_test(agent_name: str, seed: int, match_main: bool = True) -> float:
    """If match_main=True: use the main-sweep test-maze distribution (no is_solvable filter, seed_offset=10_000_000).
    If False: use the old launcher's filtered distribution.
    """
    import random as _random
    main_rng = _random.Random(seed)
    set_all_seeds(seed, deterministic=False)
    if agent_name == "Random":
        agent = RandomAgent()
    elif agent_name == "NoBackRandom":
        agent = NoBacktrackRandomAgent()
    elif agent_name == "FeatureQ":
        agent = FeatureQAgent()
        _ = run_experiment(agent, "FeatureQ", MAZE_SIZE, 100, 0, seed)
    else:
        raise ValueError(agent_name)

    solved = 0
    test_seeds: list[int] = []
    if match_main:
        # Match main sweep: maze_seed = rng.randint(0, 10M) + 10M for test phase
        for _ in range(NUM_TEST):
            ms = main_rng.randint(0, 10_000_000) + 10_000_000
            test_seeds.append(ms)
    else:
        rng_np = np.random.default_rng(seed + 1_000_000)
        while len(test_seeds) < NUM_TEST:
            s = int(rng_np.integers(0, 10**9))
            if is_solvable(make_maze(MAZE_SIZE, seed=s), MAZE_SIZE):
                test_seeds.append(s)

    for s in test_seeds:
        maze = make_maze(MAZE_SIZE, seed=s)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        if hasattr(agent, "reset_for_new_maze"):
            agent.reset_for_new_maze()
        action_hist: list = []
        max_steps = 4 * MAZE_SIZE * MAZE_SIZE
        for step in range(max_steps):
            obs = get_obs(maze, ax, ay, gx, gy, MAZE_SIZE, action_hist)
            if hasattr(agent, "eval_action"):
                action = agent.eval_action(obs)
            else:
                action = agent.act(obs, step)
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, MAZE_SIZE)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
    return solved / NUM_TEST


def main() -> None:
    print("=" * 60)
    print("Canonical Harness Validation — per Codex Round 4")
    print("=" * 60)
    print("Reference (main sweep): Random=31.7%, NoBackRandom=52.2%, FeatureQ_v2=35.3%")
    print()

    print("--- Mode A: match_main=True (main sweep test-maze distribution) ---")
    for agent_name in ("Random", "NoBackRandom", "FeatureQ"):
        rates = [custom_harness_test(agent_name, seed, match_main=True) for seed in SEEDS]
        arr = np.array(rates) * 100
        print(f"  {agent_name:<15} n=20  mean={arr.mean():>5.1f}%  sd={arr.std():>4.1f}  "
              f"range={arr.min():>3.0f}-{arr.max():>3.0f}")

    print()
    print("--- Mode B: match_main=False (old launcher's filtered distribution) ---")
    for agent_name in ("Random", "NoBackRandom"):
        rates = [custom_harness_test(agent_name, seed, match_main=False) for seed in SEEDS]
        arr = np.array(rates) * 100
        print(f"  {agent_name:<15} n=20  mean={arr.mean():>5.1f}%  sd={arr.std():>4.1f}")


if __name__ == "__main__":
    main()
