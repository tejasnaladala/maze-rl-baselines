"""Modern baseline suite for the main maze harness.

Targets the NeurIPS reviewer concern: 'your DQN baselines may simply be
under-tuned'. Sweeps multiple algorithms with multiple hyperparameter
configurations on the SAME main-sweep evaluation harness used by the
headline tables.

Suite (compact, runnable in 6-10 hours on RTX 5070 Ti):
- PPO with 3 HP configs (lr in {1e-4, 3e-4, 1e-3}) x 10 seeds = 30 runs
- DQN_500K with 3 HP configs (lr in {1e-4, 5e-4, 1e-3}) x 10 seeds = 30 runs
- A2C with default x 10 seeds = 10 runs

Total: 70 runs at 9x9 with 500K env steps each. Tests on the same
main-sweep test distribution as MLP_DQN headline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze, ego_features, ACTIONS, WALL, HAZARD, NUM_ACTIONS, OBS_DIM,
    load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)
from maze_env_helpers import main_sweep_test_seeds
from launch_ppo_shaped import MazeShapedEnv

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
MAZE_SIZE = 9
NUM_TEST = 50
BUDGET = 500_000

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_modern_baselines"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"

CONFIGS = [
    ("PPO_lr1e-4", PPO, dict(learning_rate=1e-4)),
    ("PPO_lr3e-4", PPO, dict(learning_rate=3e-4)),
    ("PPO_lr1e-3", PPO, dict(learning_rate=1e-3)),
    ("DQN_lr1e-4", DQN, dict(learning_rate=1e-4)),
    ("DQN_lr5e-4", DQN, dict(learning_rate=5e-4)),
    ("DQN_lr1e-3", DQN, dict(learning_rate=1e-3)),
    ("A2C_default", A2C, dict(learning_rate=7e-4)),
]


def train_and_test(agent_cls, hp: dict, seed: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    train_env = MazeShapedEnv(size=MAZE_SIZE, seed=seed)

    common = dict(verbose=0, seed=seed, device=DEVICE)
    if agent_cls is PPO:
        model = PPO("MlpPolicy", train_env, n_steps=512, batch_size=64, n_epochs=4,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    **hp, **common)
    elif agent_cls is DQN:
        model = DQN("MlpPolicy", train_env, buffer_size=20_000, learning_starts=1000,
                    batch_size=64, gamma=0.99, target_update_interval=300,
                    train_freq=4, gradient_steps=1,
                    exploration_fraction=0.2, exploration_final_eps=0.05,
                    **hp, **common)
    elif agent_cls is A2C:
        model = A2C("MlpPolicy", train_env, n_steps=5, gamma=0.99,
                    gae_lambda=1.0, ent_coef=0.01, **hp, **common)

    t0 = time.time()
    model.learn(total_timesteps=BUDGET, progress_bar=False)
    train_time = time.time() - t0

    test_seeds = main_sweep_test_seeds(seed, NUM_TEST)
    solved = 0
    total_steps = 0
    for ms in test_seeds:
        grid = make_maze(MAZE_SIZE, ms)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        action_hist: list = []
        for step in range(4 * MAZE_SIZE * MAZE_SIZE):
            obs = np.array(ego_features(grid, ax, ay, gx, gy, MAZE_SIZE, action_hist), dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            dx, dy = ACTIONS[a]
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE and grid[ny][nx] != WALL:
                ax, ay = nx, ny
            action_hist.append(a)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1
    return {
        "n_eps": NUM_TEST,
        "solved": solved,
        "success_rate": solved / NUM_TEST,
        "mean_steps": total_steps / NUM_TEST,
        "train_time_s": train_time,
        "total_timesteps": BUDGET,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total = len(CONFIGS) * len(SEEDS)
    done = 0
    print(f"Modern baseline suite: {total} runs (7 configs x 10 seeds)")
    print(f"Code hash: {code_hash()}\n")

    for name, cls, hp in CONFIGS:
        for seed in SEEDS:
            key = run_key(name, MAZE_SIZE, seed)
            if key in completed:
                done += 1
                continue
            print(f"  [{done}/{total}] {name} s={seed}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = train_and_test(cls, hp, seed)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            elapsed = time.time() - t0
            run_file = OUT_DIR / f"{name}_{MAZE_SIZE}_{seed}.json"
            atomic_save([{
                "agent_name": name,
                "maze_size": MAZE_SIZE, "seed": seed,
                "phase": "test",
                "wall_time_s": elapsed,
                "code_hash": code_hash(),
                **result,
            }], run_file)
            completed.add(key); save_checkpoint(CHECKPOINT_FILE, completed)
            done += 1
            print(f"done ({elapsed:.0f}s) test={100*result['success_rate']:.0f}%")

    print(f"\nModern baselines complete: {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
