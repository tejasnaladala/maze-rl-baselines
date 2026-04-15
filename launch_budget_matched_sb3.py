"""Tier 2: Budget-matched SB3 baselines (PPO, DQN, A2C).

The original exp1b gave SB3 agents 100K-500K env steps while FeatureQ/Tabular got ~10K.
This launcher runs PPO, DQN, A2C at matched env-step budgets so the comparison is fair:
  - budget_small: 10_000 steps (matched to FeatureQ)
  - budget_mid:   100_000 steps
  - budget_large: 500_000 steps

3 agents x 3 budgets x 3 sizes x 20 seeds = 540 runs
"""

import sys, json, random, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze, ego_features, ACTIONS, WALL, HAZARD, NUM_ACTIONS, OBS_DIM,
    ExpResult, load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, DQN, A2C
    HAVE_SB3 = True
except ImportError:
    HAVE_SB3 = False
    print("WARNING: stable-baselines3 not installed. pip install stable-baselines3 gymnasium")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 13, 21]
NUM_TEST = 50

BUDGETS = {
    'small':  10_000,
    'mid':    100_000,
    'large':  500_000,
}

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_budget_matched_sb3'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


class MazeGymEnv(gym.Env):
    """Gymnasium wrapper for SB3 agents."""
    metadata = {'render_modes': []}

    def __init__(self, maze_size: int, seed: int,
                 reward_shaping: bool = True, visit_penalty: bool = True):
        super().__init__()
        self.size = maze_size
        self.base_seed = seed
        self._rng = random.Random(seed)
        self.reward_shaping = reward_shaping
        self.visit_penalty = visit_penalty
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.max_steps = max(300, 4 * maze_size * maze_size)
        self._reset_internal(seed_offset=0)

    def _reset_internal(self, seed_offset: int = 0):
        maze_seed = self._rng.randint(0, 10_000_000) + seed_offset
        self.grid = make_maze(self.size, maze_seed)
        self.ax, self.ay = 1, 1
        self.gx, self.gy = self.size - 2, self.size - 2
        self.action_hist: list[int] = []
        self.visited = {(1, 1)}
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal(seed_offset=0)
        obs = np.asarray(
            ego_features(self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist),
            dtype=np.float32,
        )
        return obs, {}

    def step(self, action: int):
        dx, dy = ACTIONS[int(action)]
        nx, ny = self.ax + dx, self.ay + dy
        reward = -0.02
        done = False
        truncated = False
        prev_dist = abs(self.ax - self.gx) + abs(self.ay - self.gy)

        if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] != WALL:
            self.ax, self.ay = nx, ny
            if self.reward_shaping:
                new_dist = abs(self.ax - self.gx) + abs(self.ay - self.gy)
                if new_dist < prev_dist:
                    reward += 0.08
                elif new_dist > prev_dist:
                    reward -= 0.04
            if self.visit_penalty and (self.ax, self.ay) in self.visited:
                reward -= 0.1
            self.visited.add((self.ax, self.ay))
            if self.grid[self.ay][self.ax] == HAZARD:
                reward = -1.0
            if self.ax == self.gx and self.ay == self.gy:
                reward = 10.0
                done = True
        else:
            reward = -0.3

        self.action_hist.append(int(action))
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        obs = np.asarray(
            ego_features(self.grid, self.ax, self.ay, self.gx, self.gy, self.size, self.action_hist),
            dtype=np.float32,
        )
        return obs, reward, done, truncated, {}


def test_rollouts(model, maze_size: int, seed: int, num_episodes: int = NUM_TEST) -> list[dict]:
    """Zero-shot rollouts of a trained model on unseen mazes."""
    env = MazeGymEnv(maze_size, seed + 10_000_000)
    records = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(int(action))
            ep_reward += r
            steps += 1
        records.append({
            'phase': 'test', 'episode': ep,
            'reward': float(ep_reward), 'steps': int(steps),
            'solved': bool(done and not truncated),
        })
    return records


def train_and_test(algo_name: str, maze_size: int, seed: int, total_timesteps: int) -> list[dict]:
    train_env = MazeGymEnv(maze_size, seed)
    common_kwargs = dict(
        policy='MlpPolicy', env=train_env, verbose=0, device=DEVICE, seed=seed,
    )
    if algo_name == 'PPO':
        model = PPO(**common_kwargs, n_steps=256, batch_size=64, n_epochs=4)
    elif algo_name == 'DQN':
        model = DQN(**common_kwargs, buffer_size=20000, learning_starts=1000,
                    batch_size=64, target_update_interval=300)
    elif algo_name == 'A2C':
        model = A2C(**common_kwargs, n_steps=16)
    else:
        raise ValueError(algo_name)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return test_rollouts(model, maze_size, seed)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    configs = [(a, b, budget) for a in ['PPO', 'DQN', 'A2C']
               for b, budget in BUDGETS.items()]
    total_runs = len(SEEDS) * len(MAZE_SIZES) * len(configs)
    done_count = len(completed)

    print(f"\nTier 2: Budget-matched SB3 — {total_runs} runs, {done_count} done")
    print(f"Agents: PPO, DQN, A2C x Budgets: {BUDGETS}")
    print(f"Output: {OUT_DIR}")

    for algo, budget_name, budget in configs:
        for maze_size in MAZE_SIZES:
            for seed in SEEDS:
                run_tag = f"{algo}_{budget_name}"
                key = run_key(run_tag, maze_size, seed)
                if key in completed:
                    continue

                print(f"  [{done_count}/{total_runs}] {run_tag} {maze_size}x{maze_size} s={seed}...",
                      end=" ", flush=True)
                t0 = time.time()

                set_all_seeds(seed, deterministic=False)
                try:
                    test_records = train_and_test(algo, maze_size, seed, budget)
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

                records = []
                for r in test_records:
                    records.append({
                        'agent_name': run_tag, 'maze_size': maze_size, 'seed': seed,
                        'phase': r['phase'], 'episode': r['episode'],
                        'reward': r['reward'], 'steps': r['steps'], 'solved': r['solved'],
                        'synops': 0,
                        'config': {'algo': algo, 'budget_name': budget_name, 'total_timesteps': budget,
                                   'code_hash': code_hash()},
                        'lib_version': 'v2',
                    })

                run_file = OUT_DIR / f'{run_tag}_{maze_size}_{seed}.json'
                atomic_save(records, run_file)

                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done_count += 1

                succ = sum(r['solved'] for r in test_records) / len(test_records) * 100
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s) test={succ:.0f}%")

    print(f"\nTier 2 SB3 budget-matched complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
