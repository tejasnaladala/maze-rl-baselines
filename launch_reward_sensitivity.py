"""Reward sensitivity sweep beyond K4 (per Codex review).

Tests our claim that the headline finding is robust to reward design,
not an artifact of our specific shaping.

6 reward configurations at 9x9, all 5 main agents, 20 seeds:
  - 'sparse': only +10 on goal (no per-step penalty, no shaping)
  - 'sparse_step': sparse + small per-step penalty (-0.01)
  - 'no_distance': step penalty + revisit penalty, NO distance shaping
  - 'no_revisit': step penalty + distance shaping, NO revisit penalty
  - 'inverted_distance': REVERSE the distance signal (negative when closer)
  - 'doubled_shaping': 2x distance + revisit penalty (test for over-shaping)

5 agents x 6 configs x 20 seeds = 600 runs.

This experiment is decisive: if NoBackRandom dominates ACROSS all 6
reward configs (as expected, since it's reward-blind), the K4 result
is shown to generalize. If MLP_DQN improves dramatically under any one
config, the paper's framing needs adjustment.
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
    make_maze, ego_features, is_solvable, OBS_DIM, NUM_ACTIONS, ACTIONS, WALL, HAZARD,
    load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
NUM_TRAIN = 100
NUM_TEST = 50

REWARD_CONFIGS = {
    "sparse":            dict(goal=10.0, step=0.0,  distance=0.0,   revisit=0.0,  hazard=0.0,  wall=0.0),
    "sparse_step":       dict(goal=10.0, step=-0.01, distance=0.0,  revisit=0.0,  hazard=0.0,  wall=0.0),
    "no_distance":       dict(goal=10.0, step=-0.04, distance=0.0,  revisit=-0.1, hazard=-1.0, wall=-0.3),
    "no_revisit":        dict(goal=10.0, step=-0.04, distance=-0.04, revisit=0.0, hazard=-1.0, wall=-0.3),
    "inverted_distance": dict(goal=10.0, step=-0.04, distance=+0.04, revisit=-0.1, hazard=-1.0, wall=-0.3),
    "doubled_shaping":   dict(goal=10.0, step=-0.04, distance=-0.08, revisit=-0.2, hazard=-1.0, wall=-0.3),
}

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_reward_sensitivity"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"

DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def step_with_reward_config(maze, pos, last_pos, action, goal, visited, cfg):
    """Custom step function honoring an arbitrary reward config dict."""
    n = len(maze)
    dr, dc = DELTAS[action]
    new_pos = (pos[0] + dr, pos[1] + dc)
    cell = maze[new_pos[0]][new_pos[1]] if (0 <= new_pos[0] < n and 0 <= new_pos[1] < n) else WALL
    reward = cfg["step"]
    next_pos = pos
    done = False
    bumped_wall = False
    hit_hazard = False

    if cell == WALL:
        reward += cfg["wall"]
        bumped_wall = True
    elif cell == HAZARD:
        reward += cfg["hazard"]
        hit_hazard = True
        next_pos = new_pos
    else:
        next_pos = new_pos
        if next_pos == goal:
            reward += cfg["goal"]
            done = True

    # Distance shaping
    if cfg["distance"] != 0.0 and not done:
        old_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        new_dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
        if new_dist < old_dist:
            reward += cfg["distance"]
        elif new_dist > old_dist:
            reward -= cfg["distance"]

    # Revisit penalty
    if cfg["revisit"] != 0.0 and next_pos in visited and not done:
        reward += cfg["revisit"]

    visited.add(next_pos)
    return next_pos, reward, done


def run_single(agent, agent_name: str, maze_size: int, seed: int, cfg: dict,
               num_train: int, num_test: int) -> dict:
    """Train + test with a custom reward config."""
    rng = np.random.default_rng(seed)
    n_solved_train = 0
    n_solved_test = 0
    train_rewards = []
    test_rewards = []
    test_steps = []

    # Training phase
    for ep in range(num_train):
        s = int(rng.integers(0, 10**9))
        maze = make_maze(maze_size, seed=s)
        if not is_solvable(maze, maze_size):
            continue
        pos = (1, 1)
        n = len(maze)
        goal = (n - 2, n - 2)
        visited = {pos}
        if hasattr(agent, "reset_for_new_maze"):
            agent.reset_for_new_maze(maze)
        max_steps = 4 * maze_size * maze_size
        ep_reward = 0.0
        for step in range(max_steps):
            obs = ego_features(maze, pos, goal)
            action = agent.act(obs, step)
            next_pos, reward, done = step_with_reward_config(
                maze, pos, pos, action, goal, visited, cfg
            )
            next_obs = ego_features(maze, next_pos, goal)
            if hasattr(agent, "learn"):
                agent.learn(obs, action, reward, next_obs, done)
            pos = next_pos
            ep_reward += reward
            if done:
                n_solved_train += 1
                break
        train_rewards.append(ep_reward)

    # Test phase (greedy)
    for ep in range(num_test):
        s = int(rng.integers(0, 10**9))
        maze = make_maze(maze_size, seed=s)
        if not is_solvable(maze, maze_size):
            continue
        pos = (1, 1)
        n = len(maze)
        goal = (n - 2, n - 2)
        visited = {pos}
        if hasattr(agent, "reset_for_new_maze"):
            agent.reset_for_new_maze(maze)
        max_steps = 4 * maze_size * maze_size
        ep_reward = 0.0
        steps = 0
        for step in range(max_steps):
            obs = ego_features(maze, pos, goal)
            if hasattr(agent, "eval_action"):
                action = agent.eval_action(obs)
            else:
                action = agent.act(obs, step + num_train * max_steps)
            next_pos, reward, done = step_with_reward_config(
                maze, pos, pos, action, goal, visited, cfg
            )
            pos = next_pos
            ep_reward += reward
            steps = step + 1
            if done:
                n_solved_test += 1
                break
        test_rewards.append(ep_reward)
        test_steps.append(steps)

    return {
        "n_train_solved": n_solved_train,
        "n_test_solved": n_solved_test,
        "test_success_rate": n_solved_test / num_test,
        "mean_test_reward": float(np.mean(test_rewards)) if test_rewards else 0.0,
        "mean_test_steps": float(np.mean(test_steps)) if test_steps else 0.0,
    }


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
    total = len(AGENTS) * len(REWARD_CONFIGS) * len(SEEDS)
    done = len(completed)
    print(f"\nReward sensitivity: {total} runs, {done} done")
    print(f"Configs: {list(REWARD_CONFIGS.keys())}")
    print(f"Code hash: {code_hash()}\n")

    for cfg_name, cfg in REWARD_CONFIGS.items():
        for agent_name in AGENTS:
            for seed in SEEDS:
                composite = f"{cfg_name}__{agent_name}"
                key = run_key(composite, MAZE_SIZE, seed)
                if key in completed:
                    continue
                print(f"  [{done}/{total}] {composite} s={seed}...", end=" ", flush=True)
                t0 = time.time()
                set_all_seeds(seed, deterministic=False)
                agent = make_agent(agent_name)
                result = run_single(agent, agent_name, MAZE_SIZE, seed, cfg,
                                    NUM_TRAIN, NUM_TEST)
                elapsed = time.time() - t0

                run_file = OUT_DIR / f"{composite}_{MAZE_SIZE}_{seed}.json"
                atomic_save([{
                    "agent_name": composite,
                    "base_agent": agent_name,
                    "reward_config": cfg_name,
                    "reward_params": cfg,
                    "maze_size": MAZE_SIZE,
                    "seed": seed,
                    "phase": "test",
                    "wall_time_s": elapsed,
                    "code_hash": code_hash(),
                    **result,
                }], run_file)

                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done += 1
                print(f"done ({elapsed:.0f}s) test={100*result['test_success_rate']:.0f}%")

    print(f"\nReward sensitivity complete. {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
