"""SB3 focused A1 closure: DQN_500K at 9x9 and 13x13 × 20 seeds = 40 runs.

PPO_500K data already complete (20 seeds each at 9x9 and 13x13) from the
earlier launch_budget_matched_sb3.py. This adds the DQN counterpart so
both modern RL algorithms have matched 500K env-step comparisons at the
headline sizes — fully closes A1 attack.
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

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 13]
NUM_TEST = 50
BUDGET = 500_000  # matches PPO_large

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_budget_matched_sb3"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


class MazeGymEnv(gym.Env):
    """Gym wrapper around our maze for SB3."""

    def __init__(self, size: int, seed: int):
        super().__init__()
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.max_steps = 4 * size * size
        self._reset_maze()

    def _reset_maze(self):
        while True:
            s = int(self.rng.integers(0, 10**9))
            m = make_maze(self.size, seed=s)
            from experiment_lib_v2 import is_solvable
            if is_solvable(m, self.size):
                self.grid = m
                break
        self.ax, self.ay = 1, 1
        self.gx, self.gy = self.size - 2, self.size - 2
        self.action_hist: list = []
        self.steps = 0
        self.visited = {(1, 1)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
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
        reward = -0.04
        done = False
        truncated = False
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            reward += -0.3
        else:
            cell = self.grid[ny][nx]
            if cell == WALL:
                reward += -0.3
            elif cell == HAZARD:
                reward += -1.0
                self.ax, self.ay = nx, ny
            else:
                self.ax, self.ay = nx, ny
                if (self.ax, self.ay) == (self.gx, self.gy):
                    reward += 10.0
                    done = True
                else:
                    old_d = abs(self.ax - dx - self.gx) + abs(self.ay - dy - self.gy)
                    new_d = abs(self.ax - self.gx) + abs(self.ay - self.gy)
                    if new_d < old_d:
                        reward += 0.08
                    elif new_d > old_d:
                        reward -= 0.04
                    if (self.ax, self.ay) in self.visited:
                        reward -= 0.1
                    self.visited.add((self.ax, self.ay))
        self.action_hist.append(action)
        self.steps += 1
        if self.steps >= self.max_steps and not done:
            truncated = True
        obs = np.array(
            ego_features(self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist),
            dtype=np.float32,
        )
        return obs, reward, done, truncated, {}


def train_dqn(seed: int, size: int, total_steps: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    train_env = MazeGymEnv(size=size, seed=seed)
    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=5e-4,
        buffer_size=20_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=300,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=0, seed=seed, device=DEVICE,
    )
    train_start = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - train_start

    # Deterministic test on fresh mazes
    test_env = MazeGymEnv(size=size, seed=seed + 1_000_000)
    solved = 0
    total_steps_test = 0
    for ep in range(NUM_TEST):
        obs, _ = test_env.reset(seed=seed + ep * 7919 + 1_000_000)
        done = False
        truncated = False
        ep_steps = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, truncated, _ = test_env.step(int(action))
            ep_steps += 1
        if done:
            solved += 1
        total_steps_test += ep_steps
    return {
        "n_eps": NUM_TEST,
        "solved": solved,
        "success_rate": solved / NUM_TEST,
        "mean_steps": total_steps_test / NUM_TEST,
        "train_time_s": train_time,
        "total_timesteps": total_steps,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    total = len(MAZE_SIZES) * len(SEEDS)
    done = 0  # we count within our scope only
    print(f"\nSB3 focused DQN_large: {total} runs (DQN_500K @ {MAZE_SIZES}x{MAZE_SIZES} × {len(SEEDS)} seeds)")
    print(f"Code hash: {code_hash()}\n")

    for size in MAZE_SIZES:
        for seed in SEEDS:
            agent_name = "DQN_large"
            key = run_key(agent_name, size, seed)
            if key in completed:
                continue
            print(f"  [{done}/{total}] {agent_name} {size}x{size} s={seed}...",
                  end=" ", flush=True)
            t0 = time.time()
            try:
                result = train_dqn(seed, size, BUDGET)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            elapsed = time.time() - t0

            run_file = OUT_DIR / f"{agent_name}_{size}_{seed}.json"
            atomic_save([{
                "agent_name": agent_name,
                "maze_size": size,
                "seed": seed,
                "phase": "test",
                "wall_time_s": elapsed,
                "code_hash": code_hash(),
                "total_env_steps": BUDGET,
                **result,
            }], run_file)
            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done += 1
            print(f"done ({elapsed:.0f}s) test={100*result['success_rate']:.0f}%")

    print(f"\nSB3 focused DQN_large complete. {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
