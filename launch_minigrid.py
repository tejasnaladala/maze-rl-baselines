"""Tier 3: MiniGrid cross-environment generalization.

Does the "Random matches or beats neural RL" finding hold on MiniGrid environments,
which are partially observable and procedurally generated?

Environments tested:
  - MiniGrid-FourRooms-v0       (navigation, 4 rooms connected by doorways)
  - MiniGrid-MultiRoom-N2-S4-v0 (doors + 2 rooms)
  - MiniGrid-DoorKey-5x5-v0     (pick up key, unlock door, reach goal)
  - MiniGrid-Unlock-v0          (simpler: unlock door, no navigation)

Observation: the standard compact 7x7x3 partial view, flattened.
Action space: 6 actions (left/right/forward/pickup/drop/toggle).

Agents: Random, NoBackRandom, FeatureQ (with discretized feature hash), MLP_DQN, DoubleDQN.
(PPO/DQN via SB3 are in a separate launcher.)

4 envs x 5 agents x 20 seeds = 400 runs.
"""

import sys, json, random, time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    ReplayBuffer, set_all_seeds, code_hash,
    load_checkpoint, save_checkpoint, run_key, atomic_save,
)

try:
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers envs
    HAVE_MINIGRID = True
except ImportError:
    HAVE_MINIGRID = False
    print("WARNING: minigrid not installed. pip install minigrid")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
NUM_TRAIN_EPS = 100
NUM_TEST_EPS = 50
MAX_STEPS_PER_EP = 500

ENV_IDS = [
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-Unlock-v0",
]

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_minigrid'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def flatten_obs(obs_dict) -> np.ndarray:
    """MiniGrid returns {'image': HxWxC, 'direction': int, 'mission': str}. Flatten to 1D."""
    img = obs_dict['image']
    if img.ndim == 3:
        img = img.flatten()
    return np.concatenate([img.astype(np.float32) / 10.0,
                           np.array([obs_dict['direction'] / 3.0], dtype=np.float32)])


class MiniGridRandomAgent:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self._rng = random.Random(0)
    def act(self, obs, step): return self._rng.randint(0, self.num_actions - 1)
    def eval_action(self, obs): return self._rng.randint(0, self.num_actions - 1)
    def learn(self, *args, **kwargs): pass
    def reset_for_new_maze(self): pass
    def get_synops(self): return 0
    def seed(self, s): self._rng = random.Random(s)


class MiniGridNoBackRandomAgent:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self._rng = random.Random(0)
        self._last = -1
    def act(self, obs, step):
        # MiniGrid actions: 0=left, 1=right, 2=forward, others
        # Avoid immediate reverse: (left, right) are opposites, forward has no opposite
        choices = list(range(self.num_actions))
        if self._last == 0 and 1 in choices:
            choices.remove(1)
        elif self._last == 1 and 0 in choices:
            choices.remove(0)
        a = self._rng.choice(choices)
        self._last = a
        return a
    def eval_action(self, obs): return self.act(obs, 0)
    def learn(self, *args, **kwargs): pass
    def reset_for_new_maze(self): self._last = -1
    def get_synops(self): return 0
    def seed(self, s): self._rng = random.Random(s)


class MiniGridMLPDQN:
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128,
                 lr: float = 5e-4, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 10000,
                 target_update: int = 500, buffer_size: int = 20000,
                 batch_size: int = 64, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        ).to(self.device)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        ).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update
        self.steps = 0
        self.synops = 0

    def act(self, obs, step):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return q.argmax(dim=1).item()

    def eval_action(self, obs):
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return q.argmax(dim=1).item()

    def learn(self, obs, action, reward, next_obs, done):
        self.replay.push(obs, action, reward, next_obs, float(done))
        if len(self.replay) < self.batch_size: return
        o, a, r, no, d = self.replay.sample(self.batch_size)
        o, a, r, no, d = o.to(self.device), a.to(self.device), r.to(self.device), no.to(self.device), d.to(self.device)
        q = self.net(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(no).max(dim=1).values
            target = r + self.gamma * nq * (1 - d)
        loss = nn.SmoothL1Loss()(q, target)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())

    def reset_for_new_maze(self): pass
    def get_synops(self): return self.synops


def run_minigrid_episode(env, agent, learn: bool, max_steps: int) -> dict:
    obs_dict, info = env.reset()
    obs = flatten_obs(obs_dict)
    ep_reward = 0.0
    steps = 0
    solved = False

    for step in range(max_steps):
        if not learn and hasattr(agent, 'eval_action'):
            action = agent.eval_action(obs)
        else:
            action = agent.act(obs, step)

        next_obs_dict, reward, terminated, truncated, _ = env.step(int(action))
        next_obs = flatten_obs(next_obs_dict)
        done = terminated or truncated
        ep_reward += float(reward)
        steps += 1
        if reward > 0:
            solved = True  # MiniGrid gives positive reward only on goal

        if learn:
            agent.learn(obs, int(action), float(reward), next_obs, done)

        obs = next_obs
        if done:
            break

    return {'reward': ep_reward, 'steps': steps, 'solved': solved}


def run_minigrid_experiment(agent, agent_name: str, env_id: str, seed: int,
                            num_train: int, num_test: int) -> list[dict]:
    env = gym.make(env_id)
    results = []
    if hasattr(agent, 'seed'):
        agent.seed(seed)

    # Deterministic seed per-episode
    env.reset(seed=seed)
    for ep in range(num_train):
        env.reset(seed=seed + ep)
        r = run_minigrid_episode(env, agent, learn=True, max_steps=MAX_STEPS_PER_EP)
        r.update({'phase': 'train', 'episode': ep, 'agent_name': agent_name,
                  'env_id': env_id, 'seed': seed, 'synops': agent.get_synops()})
        results.append(r)

    for ep in range(num_test):
        env.reset(seed=seed + 1_000_000 + ep)
        r = run_minigrid_episode(env, agent, learn=False, max_steps=MAX_STEPS_PER_EP)
        r.update({'phase': 'test', 'episode': ep, 'agent_name': agent_name,
                  'env_id': env_id, 'seed': seed, 'synops': agent.get_synops()})
        results.append(r)

    env.close()
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    # Probe each env to get obs_dim / num_actions
    env_meta: dict = {}
    for env_id in ENV_IDS:
        env = gym.make(env_id)
        obs_dict, _ = env.reset(seed=0)
        obs = flatten_obs(obs_dict)
        env_meta[env_id] = {'obs_dim': len(obs), 'num_actions': int(env.action_space.n)}
        env.close()

    agent_specs: list[tuple[str, callable]] = []
    for env_id, meta in env_meta.items():
        agent_specs.append((f"Random@{env_id}", lambda m=meta: MiniGridRandomAgent(m['num_actions'])))
        agent_specs.append((f"NoBackRand@{env_id}", lambda m=meta: MiniGridNoBackRandomAgent(m['num_actions'])))
        agent_specs.append((f"MLP_DQN@{env_id}", lambda m=meta: MiniGridMLPDQN(
            obs_dim=m['obs_dim'], num_actions=m['num_actions'], device=DEVICE)))

    total_runs = len(SEEDS) * len(agent_specs)
    done_count = len(completed)

    print(f"\nTier 3: MiniGrid cross-env — {total_runs} runs, {done_count} done")
    print(f"Envs: {ENV_IDS}")
    print(f"Code hash: {code_hash()}\n")

    for agent_tag, make_agent in agent_specs:
        for seed in SEEDS:
            key = run_key(agent_tag, 0, seed)
            if key in completed:
                continue

            print(f"  [{done_count}/{total_runs}] {agent_tag} s={seed}...", end=" ", flush=True)
            t0 = time.time()

            set_all_seeds(seed, deterministic=False)  # MiniGrid needs non-deterministic cuDNN for speed
            agent = make_agent()
            env_id = agent_tag.split("@", 1)[1]
            results = run_minigrid_experiment(agent, agent_tag, env_id, seed,
                                              NUM_TRAIN_EPS, NUM_TEST_EPS)

            run_file = OUT_DIR / f'{agent_tag.replace("@", "_at_").replace("-", "_")}_{seed}.json'
            atomic_save(results, run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done_count += 1

            test = [r for r in results if r['phase'] == 'test']
            test_success = sum(r['solved'] for r in test) / len(test) * 100
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nTier 3 MiniGrid complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
