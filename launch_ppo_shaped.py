"""SB3 PPO with shaped reward on the EXACT main-sweep maze env.

Closes Codex's remaining concern: "CountPPO at 0.5% is on sparse reward,
not comparable to MLP_DQN at 19% on shaped reward."

This launcher trains PPO with the same shaped reward, same horizon, same
24-d obs, same start/goal as the main run_experiment. Tests on the same
main-sweep test distribution (seed_offset=10M).

10 seeds × 9x9 × 500K env steps = 10 runs total. ~20-30 min on H200.
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

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]  # 10 seeds
MAZE_SIZE = 9
NUM_TEST = 50
BUDGET = 500_000

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_ppo_shaped"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


class MazeShapedEnv(gym.Env):
    """Reproduces run_experiment's training maze distribution + shaped reward."""

    def __init__(self, size: int, seed: int):
        super().__init__()
        self.size = size
        import random as _random
        self.train_rng = _random.Random(seed)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.max_steps = 4 * size * size
        self._reset_maze()

    def _reset_maze(self):
        # Match run_experiment: maze_seed = rng.randint(0, 10M) + seed_offset
        s = self.train_rng.randint(0, 10_000_000)
        self.grid = make_maze(self.size, s)
        self.ax, self.ay = 1, 1
        self.gx, self.gy = self.size - 2, self.size - 2
        self.action_hist: list = []
        self.steps = 0
        self.visited = {(1, 1)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            import random as _random
            self.train_rng = _random.Random(seed)
        self._reset_maze()
        obs = np.array(
            ego_features(self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist),
            dtype=np.float32,
        )
        return obs, {}

    def step(self, action):
        action = int(action)
        dx, dy = ACTIONS[action]
        nx, ny = self.ax + dx, self.ay + dy
        reward = -0.02  # main sweep step cost
        done = False
        truncated = False
        prev_dist = abs(self.ax - self.gx) + abs(self.ay - self.gy)
        if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] != WALL:
            self.ax, self.ay = nx, ny
            new_dist = abs(self.ax - self.gx) + abs(self.ay - self.gy)
            if new_dist < prev_dist:
                reward += 0.08
            elif new_dist > prev_dist:
                reward -= 0.04
            if (self.ax, self.ay) in self.visited:
                reward -= 0.1
            self.visited.add((self.ax, self.ay))
            if self.grid[self.ay][self.ax] == HAZARD:
                reward = -1.0
            if (self.ax, self.ay) == (self.gx, self.gy):
                reward = 10.0
                done = True
        else:
            reward = -0.3
        self.action_hist.append(action)
        self.steps += 1
        if self.steps >= self.max_steps and not done:
            truncated = True
        obs = np.array(
            ego_features(self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist),
            dtype=np.float32,
        )
        return obs, reward, done, truncated, {}


def train_and_test(seed: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    train_env = MazeShapedEnv(size=MAZE_SIZE, seed=seed)
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=0, seed=seed, device=DEVICE,
    )
    t0 = time.time()
    model.learn(total_timesteps=BUDGET, progress_bar=False)
    train_time = time.time() - t0

    # Test on main-sweep test distribution
    test_seeds = main_sweep_test_seeds(seed, NUM_TEST)
    solved = 0
    total_steps = 0
    for ms in test_seeds:
        grid = make_maze(MAZE_SIZE, ms)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        action_hist: list = []
        max_steps = 4 * MAZE_SIZE * MAZE_SIZE
        for step in range(max_steps):
            obs = np.array(
                ego_features(grid, ax, ay, gx, gy, MAZE_SIZE, action_hist),
                dtype=np.float32,
            )
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
        "n_eps": NUM_TEST, "solved": solved,
        "success_rate": solved / NUM_TEST,
        "mean_steps": total_steps / NUM_TEST,
        "train_time_s": train_time,
        "total_timesteps": BUDGET,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total = len(SEEDS)
    done = 0
    print(f"PPO + shaped reward on main harness: {total} runs")
    print(f"Code hash: {code_hash()}\n")
    for seed in SEEDS:
        key = run_key("PPO_shaped_500K", MAZE_SIZE, seed)
        if key in completed:
            continue
        print(f"  [{done}/{total}] PPO_shaped s={seed}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = train_and_test(seed)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        elapsed = time.time() - t0
        run_file = OUT_DIR / f"PPO_shaped_500K_{MAZE_SIZE}_{seed}.json"
        atomic_save([{
            "agent_name": "PPO_shaped_500K",
            "maze_size": MAZE_SIZE, "seed": seed,
            "phase": "test",
            "wall_time_s": elapsed,
            "code_hash": code_hash(),
            **result,
        }], run_file)
        completed.add(key); save_checkpoint(CHECKPOINT_FILE, completed)
        done += 1
        print(f"done ({elapsed:.0f}s) test={100*result['success_rate']:.0f}%")
    print(f"\nPPO shaped complete: {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
