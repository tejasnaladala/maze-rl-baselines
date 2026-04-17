"""Tier B.2 - RND and ICM exploration baselines.

Defends against the 'smarter exploration would win' reviewer critique.
RND (Burda 2019) and ICM (Pathak 2017) are SOTA neural exploration methods.

Both augment a base PPO agent with intrinsic reward bonuses:
- RND: bonus = ||target_net(s) - predictor_net(s)||^2 (novelty by neural distillation)
- ICM:  bonus = ||forward_model(s, a) - phi(s')||^2  (novelty by forward prediction error)

If RND/ICM-augmented PPO can't beat NoBackRandom (52.2% at 9x9), the paper's
finding generalizes to SOTA exploration too.

Setup:
- 2 agents (RND, ICM) x 20 seeds x 9x9 = 40 runs
- 200K env steps each
"""

from __future__ import annotations

import sys
import time
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

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_exploration_baselines"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


def mlp(in_dim: int, out_dim: int, hidden: int = 64, n_layers: int = 2) -> nn.Module:
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def set_inference_mode(module: nn.Module, training: bool) -> None:
    """Toggle module training mode (replaces .train()/.train(False) for hook compat)."""
    module.train(training)


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


def collect_rollout(policy, intrinsic_fn, n_steps: int, maze_size: int,
                    rng: np.random.Generator) -> dict:
    obs_buf = []
    act_buf = []
    rew_ext_buf = []
    rew_int_buf = []
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
        next_obs = ego_features(maze, pos, goal)

        ext_reward = -0.04
        done = False
        if pos == goal:
            ext_reward = 10.0
            done = True
        elif ep_steps >= max_ep_steps:
            done = True

        int_reward = intrinsic_fn(obs, a, next_obs)

        obs_buf.append(obs)
        act_buf.append(a)
        rew_ext_buf.append(ext_reward)
        rew_int_buf.append(float(int_reward))
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

    return {
        "obs": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.int64),
        "rewards_ext": np.array(rew_ext_buf, dtype=np.float32),
        "rewards_int": np.array(rew_int_buf, dtype=np.float32),
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

    N = len(obs)
    for _ in range(n_epochs):
        idx = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch_size):
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


def train_rnd(seed: int, total_steps: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    rng = np.random.default_rng(seed)

    policy = PPOPolicy(OBS_DIM, NUM_ACTIONS, hidden=64).to(DEVICE)
    target = mlp(OBS_DIM, 32, hidden=64).to(DEVICE)
    predictor = mlp(OBS_DIM, 32, hidden=64).to(DEVICE)
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    pred_opt = torch.optim.Adam(predictor.parameters(), lr=1e-4)

    def intrinsic_fn(obs, _action, _next_obs):
        with torch.no_grad():
            o = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
            return float(((target(o) - predictor(o)) ** 2).sum().item())

    rollout_size = 2048
    steps_done = 0
    while steps_done < total_steps:
        batch = collect_rollout(policy, intrinsic_fn, rollout_size, MAZE_SIZE, rng)
        rew_combined = batch["rewards_ext"] + 0.5 * batch["rewards_int"]
        adv = compute_gae(rew_combined, batch["values"], batch["dones"])
        ret = adv + batch["values"]
        batch["advantages"] = adv
        batch["returns"] = ret

        ppo_update(policy, opt, batch)

        obs_t = torch.from_numpy(batch["obs"]).to(DEVICE)
        with torch.no_grad():
            tgt = target(obs_t)
        pred = predictor(obs_t)
        loss = F.mse_loss(pred, tgt)
        pred_opt.zero_grad()
        loss.backward()
        pred_opt.step()

        steps_done += rollout_size

    return _test_policy(policy, seed)


def train_icm(seed: int, total_steps: int) -> dict:
    set_all_seeds(seed, deterministic=False)
    rng = np.random.default_rng(seed)

    policy = PPOPolicy(OBS_DIM, NUM_ACTIONS, hidden=64).to(DEVICE)
    feature = mlp(OBS_DIM, 32, hidden=64).to(DEVICE)
    forward_m = mlp(32 + NUM_ACTIONS, 32, hidden=64).to(DEVICE)
    inverse_m = mlp(32 + 32, NUM_ACTIONS, hidden=64).to(DEVICE)

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    icm_opt = torch.optim.Adam(
        list(feature.parameters()) + list(forward_m.parameters()) + list(inverse_m.parameters()),
        lr=1e-4
    )

    def intrinsic_fn(obs, action, next_obs):
        with torch.no_grad():
            o = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
            no = torch.from_numpy(next_obs).float().to(DEVICE).unsqueeze(0)
            phi = feature(o)
            phi_next = feature(no)
            a_one = F.one_hot(torch.tensor([action], device=DEVICE), NUM_ACTIONS).float()
            pred = forward_m(torch.cat([phi, a_one], dim=-1))
            return float(((pred - phi_next) ** 2).sum().item())

    rollout_size = 2048
    steps_done = 0
    while steps_done < total_steps:
        batch = collect_rollout(policy, intrinsic_fn, rollout_size, MAZE_SIZE, rng)
        rew_combined = batch["rewards_ext"] + 0.5 * batch["rewards_int"]
        adv = compute_gae(rew_combined, batch["values"], batch["dones"])
        ret = adv + batch["values"]
        batch["advantages"] = adv
        batch["returns"] = ret

        ppo_update(policy, opt, batch)

        obs_t = torch.from_numpy(batch["obs"]).to(DEVICE)
        a_t = torch.from_numpy(batch["actions"]).to(DEVICE)
        next_obs_t = torch.cat([obs_t[1:], obs_t[:1]], dim=0)
        phi = feature(obs_t)
        phi_next = feature(next_obs_t)
        a_one = F.one_hot(a_t, NUM_ACTIONS).float()
        pred_phi = forward_m(torch.cat([phi, a_one], dim=-1))
        pred_a = inverse_m(torch.cat([phi, phi_next], dim=-1))
        fwd_loss = F.mse_loss(pred_phi, phi_next.detach())
        inv_loss = F.cross_entropy(pred_a, a_t)
        icm_loss = fwd_loss + inv_loss
        icm_opt.zero_grad()
        icm_loss.backward()
        icm_opt.step()

        steps_done += rollout_size

    return _test_policy(policy, seed)


def _test_policy(policy, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    set_inference_mode(policy, False)
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
                obs_t = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
                logits, _ = policy(obs_t)
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
    set_inference_mode(policy, True)
    return {
        "n_eps": NUM_TEST_EPS,
        "solved": solved,
        "success_rate": solved / NUM_TEST_EPS,
        "mean_steps": total_steps / NUM_TEST_EPS,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = [("RND", train_rnd), ("ICM", train_icm)]
    total = len(AGENTS) * len(SEEDS)
    done = len(completed)
    print(f"\nExploration baselines: {total} runs, {done} done")
    print(f"Code hash: {code_hash()}\n")

    for agent_name, train_fn in AGENTS:
        for seed in SEEDS:
            key = run_key(agent_name, MAZE_SIZE, seed)
            if key in completed:
                continue
            print(f"  [{done}/{total}] {agent_name} s={seed}...", end=" ", flush=True)
            t0 = time.time()
            result = train_fn(seed, TOTAL_ENV_STEPS)
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

    print(f"\nExploration baselines complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
