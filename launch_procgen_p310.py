"""Procgen Maze launcher (Python 3.10 conda env required).

Run via: /venv/p310/bin/python launch_procgen_p310.py

Setup:
  - 200 train levels + 0 test levels (held-out distribution)
  - Easy mode, no backgrounds, restricted themes (matches Cobbe 2019)
  - Agents: Random, NoBackRandom, PPO@500K, PPO@1M, DQN@500K
  - 5 agents x 20 seeds x 1 env = 100 runs
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

import gymnasium as gym
import procgen  # noqa: F401
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
NUM_TEST_EPS = 100
TRAIN_LEVELS = 200
ENV_NAME = "procgen-maze-v0"

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_procgen_maze"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


def _make_env(num_levels: int, start_level: int = 0) -> gym.Env:
    return gym.make(
        ENV_NAME,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="easy",
        use_backgrounds=False,
        restrict_themes=True,
        paint_vel_info=False,
    )


def make_train_vec(seed: int, num_envs: int = 1) -> gym.Env:
    return VecMonitor(DummyVecEnv([lambda: _make_env(TRAIN_LEVELS, 0)] * num_envs))


def eval_random(seed: int, no_back: bool = False) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    env = _make_env(num_levels=0, start_level=TRAIN_LEVELS)
    n_actions = env.action_space.n
    solved = 0
    total_steps = 0
    total_reward = 0.0
    for ep in range(NUM_TEST_EPS):
        obs, _ = env.reset(seed=seed + ep * 7919 + 1_000_000)
        last_action = -1
        ep_reward = 0.0
        done = False
        ep_steps = 0
        max_steps = 500
        while not done and ep_steps < max_steps:
            if no_back and last_action != -1:
                # Procgen actions: 0..14 (we forbid LR-opposite)
                opp = {1: 7, 7: 1, 3: 5, 5: 3}.get(last_action, -1)
                choices = [a for a in range(n_actions) if a != opp]
                action = int(rng.choice(choices))
            else:
                action = int(rng.integers(0, n_actions))
            obs, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            last_action = action
            ep_reward += float(reward)
            ep_steps += 1
        if ep_reward > 0.5:
            solved += 1
        total_steps += ep_steps
        total_reward += ep_reward
    env.close()
    return {
        "n_eps": NUM_TEST_EPS,
        "solved": solved,
        "success_rate": solved / NUM_TEST_EPS,
        "mean_reward": total_reward / NUM_TEST_EPS,
        "mean_steps": total_steps / NUM_TEST_EPS,
    }


def train_and_eval_sb3(agent_class, seed: int, total_steps: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    train_env = make_train_vec(seed)

    if agent_class is PPO:
        model = PPO(
            "CnnPolicy", train_env,
            learning_rate=5e-4, n_steps=256, batch_size=64, n_epochs=3,
            gamma=0.999, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=0, seed=seed, device=DEVICE,
        )
    else:
        model = DQN(
            "CnnPolicy", train_env,
            learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
            batch_size=64, gamma=0.99, target_update_interval=500,
            verbose=0, seed=seed, device=DEVICE,
        )

    train_start = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    train_time = time.time() - train_start

    test_env = _make_env(num_levels=0, start_level=TRAIN_LEVELS)
    solved = 0
    total_reward = 0.0
    total_steps_test = 0
    for ep in range(NUM_TEST_EPS):
        obs, _ = test_env.reset(seed=seed + ep * 7919 + 1_000_000)
        ep_reward = 0.0
        done = False
        ep_steps = 0
        while not done and ep_steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = test_env.step(int(action))
            done = bool(term or trunc)
            ep_reward += float(reward)
            ep_steps += 1
        if ep_reward > 0.5:
            solved += 1
        total_reward += ep_reward
        total_steps_test += ep_steps
    test_env.close()
    train_env.close()
    return {
        "n_eps": NUM_TEST_EPS,
        "solved": solved,
        "success_rate": solved / NUM_TEST_EPS,
        "mean_reward": total_reward / NUM_TEST_EPS,
        "mean_steps": total_steps_test / NUM_TEST_EPS,
        "train_time_s": train_time,
        "total_timesteps": total_steps,
    }


def run_one(agent_name: str, seed: int) -> dict:
    if agent_name == "Random":
        return eval_random(seed, no_back=False)
    if agent_name == "NoBackRandom":
        return eval_random(seed, no_back=True)
    if agent_name == "PPO_500K":
        return train_and_eval_sb3(PPO, seed, total_steps=500_000)
    if agent_name == "PPO_1M":
        return train_and_eval_sb3(PPO, seed, total_steps=1_000_000)
    if agent_name == "DQN_500K":
        return train_and_eval_sb3(DQN, seed, total_steps=500_000)
    raise ValueError(f"unknown agent {agent_name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = ["Random", "NoBackRandom", "PPO_500K", "PPO_1M", "DQN_500K"]
    total = len(AGENTS) * len(SEEDS)
    done = len(completed)
    print(f"\nProcgen Maze: {total} runs, {done} done")
    print(f"Code hash: {code_hash()}\n")

    for agent_name in AGENTS:
        for seed in SEEDS:
            key = run_key(agent_name, 0, seed)
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
