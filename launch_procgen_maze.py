"""Tier A.1 — Procgen Maze cross-environment replication.

Tests whether the "NoBackRandom > trained RL" finding generalizes to
the canonical procedural RL benchmark (Cobbe et al. 2019, ICML).

Procgen Maze:
  - Procedurally generated mazes (200 distinct level seeds in train, infinite test)
  - Pixel observations (64x64x3) with built-in CNN-friendly features
  - Discrete action space (15 actions; we restrict to nav-only effective)
  - Continuous reward, dense or sparse modes

Agents:
  - Random          (uniform random)
  - NoBackRandom    (non-backtracking, our headline)
  - PPO_500K        (SB3 PPO at 500K env steps -- standard procgen budget)
  - PPO_2M          (SB3 PPO at 2M env steps -- generous budget)
  - DQN_500K        (SB3 DQN at 500K env steps)

5 agents x 20 seeds = 100 runs.
Test: 100 unseen procgen levels (test-distribution) per agent per seed.

Outputs to raw_results/exp_procgen_maze/.

REQUIRES: pip install procgen stable-baselines3 gymnasium
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    load_checkpoint,
    save_checkpoint,
    run_key,
    atomic_save,
    set_all_seeds,
    code_hash,
)

try:
    import gymnasium as gym
    import procgen  # noqa: F401  -- registers procgen-* envs
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    HAVE_DEPS = True
except ImportError as e:
    print(f"WARNING: missing dep ({e}). Install with: pip install procgen stable-baselines3 gymnasium")
    HAVE_DEPS = False
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
NUM_TEST_EPS = 100
TRAIN_LEVELS = 200       # standard easy-mode procgen
TEST_LEVELS = 0          # 0 means use full distribution (unseen levels)
ENV_NAME = "procgen-maze-v0"

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_procgen_maze"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


def make_env(seed: int, num_levels: int, start_level: int = 0, num_envs: int = 1) -> gym.Env:
    """Create a vectorized procgen Maze env."""
    def _init():
        env = gym.make(
            ENV_NAME,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode="easy",
            use_backgrounds=False,
            restrict_themes=True,
            paint_vel_info=False,
        )
        return env
    return DummyVecEnv([_init for _ in range(num_envs)])


def eval_random_agent(seed: int, n_eps: int, no_backtrack: bool = False) -> dict:
    """Evaluate a random or non-backtracking random walk on procgen maze test set."""
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_NAME, num_levels=0, start_level=0, distribution_mode="easy",
                   use_backgrounds=False, restrict_themes=True)

    n_actions = env.action_space.n
    solved = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep * 7919)
        last_action = -1
        ep_steps = 0
        ep_reward = 0.0
        max_steps = 500
        done = False
        while not done and ep_steps < max_steps:
            if no_backtrack and last_action != -1:
                # Procgen action 0..14; nav actions are 1,3,5,7 (left, down, up, right roughly)
                # We forbid the *opposite* of last action where defined.
                opposite = {1: 7, 7: 1, 3: 5, 5: 3}.get(last_action, -1)
                choices = [a for a in range(n_actions) if a != opposite]
                action = int(rng.choice(choices))
            else:
                action = int(rng.integers(0, n_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_action = action
            ep_reward += float(reward)
            ep_steps += 1
        if ep_reward > 0.5:  # procgen maze gives reward 10 on success
            solved += 1
        total_steps += ep_steps
        total_reward += ep_reward
    env.close()
    return {
        "n_eps": n_eps,
        "solved": solved,
        "success_rate": solved / n_eps,
        "mean_reward": total_reward / n_eps,
        "mean_steps": total_steps / n_eps,
    }


def train_and_eval_sb3(
    agent_class, seed: int, total_steps: int, n_test: int,
    learning_rate: float = 5e-4
) -> dict:
    """Train SB3 PPO/DQN on train levels, eval on unseen test levels."""
    set_all_seeds(seed, deterministic=False)  # keep speed
    train_env = make_env(seed=seed, num_levels=TRAIN_LEVELS, start_level=0)
    train_env = VecMonitor(train_env)

    if agent_class is PPO:
        model = PPO(
            "CnnPolicy", train_env,
            learning_rate=learning_rate,
            n_steps=256,
            batch_size=64,
            n_epochs=3,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0, seed=seed, device=DEVICE,
        )
    else:  # DQN
        model = DQN(
            "CnnPolicy", train_env,
            learning_rate=learning_rate,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500,
            verbose=0, seed=seed, device=DEVICE,
        )

    train_start = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - train_start

    # Evaluate on test distribution (unseen levels)
    test_env = gym.make(ENV_NAME, num_levels=0, start_level=TRAIN_LEVELS,
                        distribution_mode="easy", use_backgrounds=False, restrict_themes=True)
    solved = 0
    total_reward = 0.0
    total_steps_test = 0
    for ep in range(n_test):
        obs, _ = test_env.reset(seed=seed + ep * 7919 + 1_000_000)
        ep_reward = 0.0
        ep_steps = 0
        done = False
        while not done and ep_steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(int(action))
            done = terminated or truncated
            ep_reward += float(reward)
            ep_steps += 1
        if ep_reward > 0.5:
            solved += 1
        total_reward += ep_reward
        total_steps_test += ep_steps
    test_env.close()
    train_env.close()
    return {
        "n_eps": n_test,
        "solved": solved,
        "success_rate": solved / n_test,
        "mean_reward": total_reward / n_test,
        "mean_steps": total_steps_test / n_test,
        "train_time_s": train_time,
        "total_timesteps": total_steps,
    }


def run_one(agent_name: str, seed: int) -> dict:
    """Dispatch to the right evaluator."""
    if agent_name == "Random":
        return eval_random_agent(seed, NUM_TEST_EPS, no_backtrack=False)
    elif agent_name == "NoBackRandom":
        return eval_random_agent(seed, NUM_TEST_EPS, no_backtrack=True)
    elif agent_name == "PPO_500K":
        return train_and_eval_sb3(PPO, seed, total_steps=500_000, n_test=NUM_TEST_EPS)
    elif agent_name == "PPO_2M":
        return train_and_eval_sb3(PPO, seed, total_steps=2_000_000, n_test=NUM_TEST_EPS)
    elif agent_name == "DQN_500K":
        return train_and_eval_sb3(DQN, seed, total_steps=500_000, n_test=NUM_TEST_EPS)
    else:
        raise ValueError(f"unknown agent {agent_name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = ["Random", "NoBackRandom", "PPO_500K", "PPO_2M", "DQN_500K"]
    total = len(AGENTS) * len(SEEDS)
    done = len(completed)

    print(f"\nProcgen Maze: {total} runs, {done} done")
    print(f"Code hash: {code_hash()}")
    print(f"Train levels: {TRAIN_LEVELS}, test eps: {NUM_TEST_EPS}\n")

    for agent_name in AGENTS:
        for seed in SEEDS:
            key = run_key(agent_name, 0, seed)  # size irrelevant for procgen
            if key in completed:
                continue

            print(f"  [{done}/{total}] {agent_name} s={seed}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = run_one(agent_name, seed)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            elapsed = time.time() - t0

            run_file = OUT_DIR / f"{agent_name}_{seed}.json"
            atomic_save([{
                "agent_name": agent_name,
                "env": ENV_NAME,
                "seed": seed,
                "phase": "test",
                "wall_time_s": elapsed,
                "code_hash": code_hash(),
                **result,
            }], run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done += 1
            print(f"done ({elapsed:.0f}s) success={100*result['success_rate']:.0f}%")

    print(f"\nProcgen Maze complete. {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
