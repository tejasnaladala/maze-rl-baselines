"""Count-based intrinsic motivation PPO sweep on the audited 9x9 harness.

Direct response to the Codex MCP adversarial audit (R8) finding that the
modern-baseline class is easy to dismiss without an exploration baseline:
"the easiest dismissal is 'you only showed that vanilla reward-driven RL
fails on a generator where exploration is obviously the bottleneck.' If
intrinsic motivation closes the gap, your current claim is too strong and
you learn something real. If it still stalls near Random, the package
becomes materially harder to wave away as under-baselined."

We add a count-based intrinsic reward bonus to the same shaped-reward
PPO run from launch_ppo_shaped.py. State key is the agent (x, y) cell;
bonus = beta / sqrt(N(s)). Counts persist across episodes within a seed
to capture cross-episode novelty.

If this also stalls near Random (32.7%), the under-baselined critique is
materially defeated. If it closes the gap toward distillation (97.4%),
the central paper claim needs revision and the result is even more
interesting.

n=20 seeds at 9x9, 500K env steps, beta in {0.05, 0.1, 0.3} (one beta per
sub-sweep selected after a 5-seed pilot at beta=0.1).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    ACTIONS,
    HAZARD,
    NUM_ACTIONS,
    OBS_DIM,
    WALL,
    atomic_save,
    code_hash,
    ego_features,
    load_checkpoint,
    make_maze,
    run_key,
    save_checkpoint,
    set_all_seeds,
)
from maze_env_helpers import main_sweep_test_seeds

DEVICE = "cpu"  # SB3 PPO with MlpPolicy is CPU-bound; per session measurements ~400s/seed
print(f"Device: {DEVICE}")

# 20 seeds for n=20 statistical power per Codex audit recommendation.
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
NUM_TEST = 50
BUDGET = 500_000

# beta is the intrinsic-reward coefficient. beta=0.1 is the standard default
# for count-based bonuses in small gridworld settings.
BETA = 0.1

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_count_ppo_audited"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


class MazeShapedCountEnv(gym.Env):
    """Same shaped-reward env as launch_ppo_shaped.py + count-based bonus."""

    def __init__(self, size: int, seed: int, beta: float = BETA) -> None:
        super().__init__()
        self.size = size
        import random as _random

        self.train_rng = _random.Random(seed)
        self.beta = beta
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.max_steps = 4 * size * size
        # Visit counts persist across episodes within a single env (seed) lifetime.
        # State key = (maze_seed, ax, ay) so visits are per-maze-instance.
        self.visit_counts: dict[tuple, int] = {}
        self._reset_maze()

    def _reset_maze(self) -> None:
        s = self.train_rng.randint(0, 10_000_000)
        self.current_maze_seed = s
        self.grid = make_maze(self.size, s)
        self.ax, self.ay = 1, 1
        self.gx, self.gy = self.size - 2, self.size - 2
        self.action_hist: list = []
        self.steps = 0
        self.visited = {(1, 1)}

    def _intrinsic_bonus(self) -> float:
        key = (self.current_maze_seed, self.ax, self.ay)
        n = self.visit_counts.get(key, 0)
        self.visit_counts[key] = n + 1
        # Standard count-based bonus: beta / sqrt(N(s)). N+1 in denominator to
        # avoid div-by-zero on first visit.
        return float(self.beta / math.sqrt(n + 1))

    def reset(self, seed=None, options=None):
        if seed is not None:
            import random as _random

            self.train_rng = _random.Random(seed)
        self._reset_maze()
        obs = np.array(
            ego_features(
                self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist
            ),
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

        # Add intrinsic count-based bonus on every step. This is the only
        # difference from launch_ppo_shaped.py.
        reward += self._intrinsic_bonus()

        self.action_hist.append(action)
        self.steps += 1
        if self.steps >= self.max_steps and not done:
            truncated = True
        obs = np.array(
            ego_features(
                self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist
            ),
            dtype=np.float32,
        )
        return obs, reward, done, truncated, {}


def train_and_test(seed: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    train_env = MazeShapedCountEnv(size=MAZE_SIZE, seed=seed, beta=BETA)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
        device=DEVICE,
    )
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
        max_steps = 4 * MAZE_SIZE * MAZE_SIZE
        for step in range(max_steps):
            obs = np.array(
                ego_features(grid, ax, ay, gx, gy, MAZE_SIZE, action_hist), dtype=np.float32
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
        "n_eps": NUM_TEST,
        "solved": solved,
        "success_rate": solved / NUM_TEST,
        "mean_steps": total_steps / NUM_TEST,
        "train_time_s": train_time,
        "total_timesteps": BUDGET,
        "intrinsic_beta": BETA,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    name = "PPO_countbased_shaped"
    print(f"Count-based PPO sweep: {len(SEEDS)} seeds at 9x9, beta={BETA}, {BUDGET} env steps each")
    print(f"Code hash: {code_hash()}\n")
    done = 0
    for seed in SEEDS:
        key = run_key(name, MAZE_SIZE, seed)
        if key in completed:
            done += 1
            continue
        print(f"  [{done}/{len(SEEDS)}] {name} s={seed}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = train_and_test(seed)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        elapsed = time.time() - t0
        run_file = OUT_DIR / f"{name}_{MAZE_SIZE}_{seed}.json"
        atomic_save(
            [
                {
                    "agent_name": name,
                    "maze_size": MAZE_SIZE,
                    "seed": seed,
                    "phase": "test",
                    "wall_time_s": elapsed,
                    "code_hash": code_hash(),
                    **result,
                }
            ],
            run_file,
        )
        completed.add(key)
        save_checkpoint(CHECKPOINT_FILE, completed)
        done += 1
        print(f"done ({elapsed:.0f}s) test={100 * result['success_rate']:.0f}%")

    print(f"\nCount-based PPO complete: {len(SEEDS)} seeds in {OUT_DIR}")


if __name__ == "__main__":
    main()
