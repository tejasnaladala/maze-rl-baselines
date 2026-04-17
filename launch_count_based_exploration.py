"""KILLER EXPERIMENT (per Codex adversarial review):

  Count-based exploration baselines (Strehl & Littman 2008, Bellemare 2016).

If a tabular state-count bonus closes the gap to NoBackRandom, our claim
that 'random walks beat trained RL' is too narrow -- intrinsic motivation
methods would already be the strong baseline. If count-based bonuses do
NOT close the gap, our paper survives the strongest hostile critique.

This is the lethal reviewer check. Run it and see.

Setup:
  - PPO + state-count bonus: r' = r + beta / sqrt(N(s))
  - PPO + NGU-lite: episodic novelty via state-visit counts within episode
  - 2 agents x 20 seeds x 9x9 = 40 runs
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze,
    ego_features,
    is_solvable,
    OBS_DIM,
    NUM_ACTIONS,
    load_checkpoint,
    save_checkpoint,
    run_key,
    atomic_save,
    set_all_seeds,
    code_hash,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
TOTAL_ENV_STEPS = 200_000
NUM_TEST_EPS = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_count_exploration"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.policy_head(z), self.value_head(z)


def discretize_obs(obs: np.ndarray) -> tuple:
    """Hash the 24-dim feature vector to a discrete state ID for counting."""
    return tuple(np.round(obs * 10).astype(np.int32).tolist())


def collect_rollout(policy, count_table: Counter, n_steps: int, maze_size: int,
                    rng: np.random.Generator, beta: float = 1.0,
                    episodic: bool = False) -> dict:
    """Roll out PPO with count-based intrinsic reward."""
    obs_buf = []
    act_buf = []
    rew_buf = []
    val_buf = []
    logp_buf = []
    done_buf = []

    maze = make_maze(maze_size, seed=int(rng.integers(0, 10**9)))
    while not is_solvable(maze):
        maze = make_maze(maze_size, seed=int(rng.integers(0, 10**9)))
    pos = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)
    ep_steps = 0
    max_ep_steps = 4 * maze_size * maze_size
    episode_counts: Counter = Counter()

    for _ in range(n_steps):
        obs = ego_features(maze, pos, goal)
        obs_t = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        a = int(action.item())
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        new_pos = (pos[0] + dr, pos[1] + dc)
        if (0 <= new_pos[0] < maze.shape[0] and 0 <= new_pos[1] < maze.shape[1]
                and maze[new_pos] != 1):
            pos = new_pos
        ep_steps += 1

        ext = -0.04
        done = False
        if pos == goal:
            ext = 10.0
            done = True
        elif ep_steps >= max_ep_steps:
            done = True

        # Intrinsic bonus
        state_id = discretize_obs(obs)
        if episodic:
            episode_counts[state_id] += 1
            n = episode_counts[state_id]
        else:
            count_table[state_id] += 1
            n = count_table[state_id]
        intrinsic = beta / np.sqrt(max(1, n))

        obs_buf.append(obs)
        act_buf.append(a)
        rew_buf.append(ext + float(intrinsic))
        val_buf.append(float(value.item()))
        logp_buf.append(float(logp.item()))
        done_buf.append(done)

        if done:
            maze = make_maze(maze_size, seed=int(rng.integers(0, 10**9)))
            while not is_solvable(maze):
                maze = make_maze(maze_size, seed=int(rng.integers(0, 10**9)))
            pos = (1, 1)
            goal = (maze.shape[0] - 2, maze.shape[1] - 2)
            ep_steps = 0
            episode_counts = Counter()

    return {
        "obs": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.int64),
        "rewards": np.array(rew_buf, dtype=np.float32),
        "values": np.array(val_buf, dtype=np.float32),
        "logps": np.array(logp_buf, dtype=np.float32),
        "dones": np.array(done_buf, dtype=bool),
    }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0.0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae
    return advantages


def ppo_update(policy, opt, batch, n_epochs=4, batch_size=64, clip=0.2):
    obs = torch.from_numpy(batch["obs"]).to(DEVICE)
    actions = torch.from_numpy(batch["actions"]).to(DEVICE)
    old_logps = torch.from_numpy(batch["logps"]).to(DEVICE)
    advantages = torch.from_numpy(batch["advantages"]).to(DEVICE)
    returns = torch.from_numpy(batch["returns"]).to(DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            logits, values = policy(obs[b])
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(actions[b])
            ratio = (new_logps - old_logps[b]).exp()
            surr1 = ratio * advantages[b]
            surr2 = ratio.clamp(1 - clip, 1 + clip) * advantages[b]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(-1), returns[b])
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()


def train_count_ppo(seed: int, total_steps: int, episodic: bool = False) -> dict:
    set_all_seeds(seed, deterministic=False)
    rng = np.random.default_rng(seed)
    policy = PPOPolicy(OBS_DIM, NUM_ACTIONS, hidden=64).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    count_table: Counter = Counter()

    rollout_size = 2048
    steps_done = 0
    while steps_done < total_steps:
        batch = collect_rollout(
            policy, count_table, rollout_size, MAZE_SIZE, rng,
            beta=1.0, episodic=episodic
        )
        adv = compute_gae(batch["rewards"], batch["values"], batch["dones"])
        ret = adv + batch["values"]
        batch["advantages"] = adv
        batch["returns"] = ret
        ppo_update(policy, opt, batch)
        steps_done += rollout_size

    return _test_policy(policy, seed)


def _test_policy(policy, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    policy.train(False)
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < NUM_TEST_EPS:
        s = int(rng.integers(0, 10**9))
        maze = make_maze(MAZE_SIZE, seed=s)
        if is_solvable(maze):
            test_seeds.append(s)

    for s in test_seeds:
        maze = make_maze(MAZE_SIZE, seed=s)
        pos = (1, 1)
        goal = (maze.shape[0] - 2, maze.shape[1] - 2)
        max_steps = 4 * MAZE_SIZE * MAZE_SIZE
        step = 0
        for step in range(max_steps):
            obs = ego_features(maze, pos, goal)
            with torch.no_grad():
                logits, _ = policy(torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0))
                action = int(logits.argmax(dim=-1).item())
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < maze.shape[0] and 0 <= new_pos[1] < maze.shape[1]
                    and maze[new_pos] != 1):
                pos = new_pos
            if pos == goal:
                solved += 1
                break
        total_steps += step + 1
    policy.train(True)
    return {
        "n_eps": NUM_TEST_EPS,
        "solved": solved,
        "success_rate": solved / NUM_TEST_EPS,
        "mean_steps": total_steps / NUM_TEST_EPS,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = [
        ("CountPPO_global", lambda s: train_count_ppo(s, TOTAL_ENV_STEPS, episodic=False)),
        ("CountPPO_episodic", lambda s: train_count_ppo(s, TOTAL_ENV_STEPS, episodic=True)),
    ]
    total = len(AGENTS) * len(SEEDS)
    done = len(completed)
    print(f"\nCount-based exploration: {total} runs, {done} done")
    print(f"Code hash: {code_hash()}\n")

    for agent_name, train_fn in AGENTS:
        for seed in SEEDS:
            key = run_key(agent_name, MAZE_SIZE, seed)
            if key in completed:
                continue
            print(f"  [{done}/{total}] {agent_name} s={seed}...", end=" ", flush=True)
            t0 = time.time()
            result = train_fn(seed)
            elapsed = time.time() - t0

            run_file = OUT_DIR / f"{agent_name}_{MAZE_SIZE}_{seed}.json"
            atomic_save([{
                "agent_name": agent_name,
                "maze_size": MAZE_SIZE,
                "seed": seed,
                "phase": "test",
                "wall_time_s": elapsed,
                "code_hash": code_hash(),
                "total_env_steps": TOTAL_ENV_STEPS,
                **result,
            }], run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done += 1
            print(f"done ({elapsed:.0f}s) test={100*result['success_rate']:.0f}%")

    print(f"\nCount-based exploration complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
