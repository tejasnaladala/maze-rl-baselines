"""Cross-size transfer matrix (per Codex review).

Tests generalization claim: train on size N, test on different sizes.
The hypothesis: trained agents overfit to their training-distribution
maze size; structural priors (NoBack, Random, Wall-follower) generalize
trivially because they're size-invariant.

Train sizes: {9, 13}
Test sizes: {9, 13, 17, 21, 25}
Agents: MLP_DQN, FeatureQ_v2, DoubleDQN, NoBackRandom, Random
20 seeds.

5 agents x 2 train_sizes x 5 test_sizes x 20 seeds = 1000 runs
(but only train once per (agent, train_size, seed), test on 5 sizes -> 200 trainings).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    NoBacktrackRandomAgent, RandomAgent, FeatureQAgent, MLPDQNAgent, DoubleDQNAgent,
    run_experiment, make_maze, ego_features, is_solvable,
    OBS_DIM, NUM_ACTIONS, ACTIONS, WALL,
    load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)
from maze_env_helpers import get_obs, step_env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
TRAIN_SIZES = [9, 13]
TEST_SIZES = [9, 13, 17, 21, 25]
NUM_TRAIN = 100
NUM_TEST = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_cross_env_transfer"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"

DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def test_at_size(agent, test_size: int, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + test_size * 10000 + 1_000_000)
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < n_test:
        s = int(rng.integers(0, 10**9))
        if is_solvable(make_maze(test_size, seed=s), test_size):
            test_seeds.append(s)
    for s in test_seeds:
        maze = make_maze(test_size, seed=s)
        ax, ay = 1, 1
        gx, gy = test_size - 2, test_size - 2
        if hasattr(agent, "reset_for_new_maze"):
            agent.reset_for_new_maze()
        max_steps = 4 * test_size * test_size
        action_hist: list = []
        step = 0
        for step in range(max_steps):
            obs = get_obs(maze, ax, ay, gx, gy, test_size, action_hist)
            if hasattr(agent, "eval_action"):
                action = agent.eval_action(obs)
            else:
                action = agent.act(obs, step + NUM_TRAIN * 1000)
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, test_size)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1
    return {"n_eps": n_test, "solved": solved,
            "success_rate": solved / n_test,
            "mean_steps": total_steps / n_test}


def make_agent(name: str):
    if name == "Random": return RandomAgent()
    if name == "NoBackRandom": return NoBacktrackRandomAgent()
    if name == "FeatureQ_v2": return FeatureQAgent()
    if name == "MLP_DQN": return MLPDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)
    if name == "DoubleDQN": return DoubleDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)
    raise ValueError(name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = ["Random", "NoBackRandom", "FeatureQ_v2", "MLP_DQN", "DoubleDQN"]
    total_trainings = len(AGENTS) * len(TRAIN_SIZES) * len(SEEDS)
    done = len(completed)
    print(f"\nCross-size transfer: {total_trainings} trainings -> {total_trainings*5} test cells")
    print(f"Train sizes: {TRAIN_SIZES}, Test sizes: {TEST_SIZES}")
    print(f"Code hash: {code_hash()}\n")

    n = done
    for agent_name in AGENTS:
        for train_size in TRAIN_SIZES:
            for seed in SEEDS:
                key = f"{agent_name}__train{train_size}__seed{seed}"
                if key in completed:
                    continue
                print(f"  [{n}/{total_trainings}] train {agent_name} on {train_size}x{train_size} s={seed}...",
                      end=" ", flush=True)
                t0 = time.time()
                set_all_seeds(seed, deterministic=False)
                agent = make_agent(agent_name)
                # Train (only relevant for learning agents)
                if hasattr(agent, "learn"):
                    _ = run_experiment(agent, agent_name, train_size,
                                       NUM_TRAIN, 0, seed)
                # Test on each size
                test_results = {}
                for test_size in TEST_SIZES:
                    test_results[f"test_{test_size}"] = test_at_size(
                        agent, test_size, NUM_TEST, seed
                    )
                elapsed = time.time() - t0

                run_file = OUT_DIR / f"{agent_name}_train{train_size}_{seed}.json"
                atomic_save([{
                    "agent_name": agent_name,
                    "train_size": train_size,
                    "seed": seed,
                    "phase": "transfer_test",
                    "wall_time_s": elapsed,
                    "code_hash": code_hash(),
                    "test_results": test_results,
                }], run_file)

                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                n += 1
                summary = " ".join(
                    f"t{s}={100*test_results[f'test_{s}']['success_rate']:.0f}%"
                    for s in TEST_SIZES
                )
                print(f"done ({elapsed:.0f}s) {summary}")

    print(f"\nCross-size transfer complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
