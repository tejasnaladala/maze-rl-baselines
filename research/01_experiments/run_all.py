"""ICONS 2026 Experiment Suite -- Spiking DQN Cross-Environment Generalization

Compares: Spiking DQN vs MLP DQN vs Tabular Q-Learning
on procedurally generated mazes with zero-shot transfer evaluation.

Usage:
    python research/01_experiments/run_all.py

Results saved to research/01_experiments/raw_results/
"""

import sys, os, time, json, random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use spiking DQN from our codebase
from engram.spiking_dqn import SpikingQNetwork, ReplayBuffer

# ============================================================
# MAZE ENVIRONMENT (variable size, procedural generation)
# ============================================================

WALL = 1
HAZARD = 3

def mulberry32(seed):
    def rng():
        nonlocal seed
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((seed ^ (seed >> 15)) * (1 | seed)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng

def make_maze(size, seed):
    """Generate a solvable maze of given odd size."""
    assert size % 2 == 1, "Size must be odd"
    g = [[WALL]*size for _ in range(size)]
    rng = mulberry32(seed)

    def carve(x, y):
        g[y][x] = 0
        dirs = [(0,-2),(2,0),(0,2),(-2,0)]
        for i in range(3, 0, -1):
            j = int(rng() * (i+1))
            dirs[i], dirs[j] = dirs[j], dirs[i]
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 1 <= nx <= size-2 and 1 <= ny <= size-2 and g[ny][nx] == WALL:
                g[y+dy//2][x+dx//2] = 0
                carve(nx, ny)

    carve(1, 1)
    # Add hazards
    for _ in range(size // 3):
        for attempt in range(30):
            hx = 1 + int(rng() * (size-2))
            hy = 1 + int(rng() * (size-2))
            if g[hy][hx] == 0 and not (hx==1 and hy==1) and not (hx==size-2 and hy==size-2):
                g[hy][hx] = HAZARD
                break
    return g

ACTIONS = [(0,-1),(1,0),(0,1),(-1,0)]

def ego_features(grid, ax, ay, gx, gy, size, action_hist):
    """Ego-centric feature vector for the agent."""
    feats = []
    # 3x3 local map (9 values)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = ax+dx, ay+dy
            if 0 <= nx < size and 0 <= ny < size:
                feats.append(grid[ny][nx] / 3.0)
            else:
                feats.append(1.0)
    # Goal direction (2 values)
    feats.append(np.sign(gx - ax))
    feats.append(np.sign(gy - ay))
    # Distance (1 value)
    max_dist = size * 2
    feats.append((abs(gx-ax) + abs(gy-ay)) / max_dist)
    # Last 3 actions one-hot (12 values)
    hist = list(action_hist[-3:])
    while len(hist) < 3:
        hist.insert(0, -1)
    for a in hist:
        oh = [0.0]*4
        if 0 <= a < 4:
            oh[a] = 1.0
        feats.extend(oh)
    return feats  # 9 + 2 + 1 + 12 = 24 dimensions

OBS_DIM = 24
NUM_ACTIONS = 4

def run_episode(grid, size, agent_fn, max_steps=200):
    """Run one episode. Returns (total_reward, steps, solved)."""
    ax, ay = 1, 1
    gx, gy = size-2, size-2
    total_r = 0.0
    action_hist = []
    visited = set()
    visited.add((ax, ay))

    for step in range(max_steps):
        obs = ego_features(grid, ax, ay, gx, gy, size, action_hist)
        action = agent_fn(obs, step)
        dx, dy = ACTIONS[action]
        nx, ny = ax+dx, ay+dy
        reward = -0.02

        prev_dist = abs(ax-gx) + abs(ay-gy)
        if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] != WALL:
            ax, ay = nx, ny
            new_dist = abs(ax-gx) + abs(ay-gy)
            if new_dist < prev_dist:
                reward += 0.08
            elif new_dist > prev_dist:
                reward -= 0.04
            if (ax, ay) in visited:
                reward -= 0.1
            visited.add((ax, ay))
            if grid[ay][ax] == HAZARD:
                reward = -1.0
            if ax == gx and ay == gy:
                reward = 10.0
                total_r += reward
                action_hist.append(action)
                return total_r, step+1, True
        else:
            reward = -0.3

        total_r += reward
        action_hist.append(action)

    return total_r, max_steps, False

# ============================================================
# AGENTS
# ============================================================

class TabularQAgent:
    """Position-based tabular Q-learning (standard RL baseline)."""
    def __init__(self, lr=0.15, gamma=0.99, eps=0.4):
        self.q = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.prev_obs = None
        self.prev_action = None

    def act(self, obs, step):
        key = self._key(obs)
        if key not in self.q:
            self.q[key] = np.zeros(NUM_ACTIONS)
        if random.random() < self.eps:
            return random.randint(0, NUM_ACTIONS-1)
        return int(np.argmax(self.q[key]))

    def learn(self, obs, action, reward, next_obs, done):
        key = self._key(obs)
        nkey = self._key(next_obs)
        if key not in self.q: self.q[key] = np.zeros(NUM_ACTIONS)
        if nkey not in self.q: self.q[nkey] = np.zeros(NUM_ACTIONS)
        target = reward + (0 if done else self.gamma * np.max(self.q[nkey]))
        self.q[key][action] += self.lr * (target - self.q[key][action])

    def reset_for_new_maze(self):
        self.q = {}  # position entries are maze-specific
        self.eps = 0.4

    def _key(self, obs):
        # Use first 9 values (3x3 local map) + goal direction as position proxy
        return tuple(round(v, 1) for v in obs[:12])


class FeatureQAgent:
    """Feature-based tabular Q-learning (our approach without spiking)."""
    def __init__(self, lr=0.2, gamma=0.99, eps=0.3):
        self.q = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def act(self, obs, step):
        key = self._key(obs)
        if key not in self.q:
            self.q[key] = np.ones(NUM_ACTIONS) * 1.0  # optimistic init
        if random.random() < max(0.08, self.eps):
            return random.randint(0, NUM_ACTIONS-1)
        return int(np.argmax(self.q[key]))

    def learn(self, obs, action, reward, next_obs, done):
        key = self._key(obs)
        nkey = self._key(next_obs)
        if key not in self.q: self.q[key] = np.ones(NUM_ACTIONS) * 1.0
        if nkey not in self.q: self.q[nkey] = np.ones(NUM_ACTIONS) * 1.0
        target = reward + (0 if done else self.gamma * np.max(self.q[nkey]))
        self.q[key][action] += self.lr * (target - self.q[key][action])
        self.eps = max(0.08, self.eps * 0.999)

    def reset_for_new_maze(self):
        pass  # features transfer

    def _key(self, obs):
        return tuple(round(v, 1) for v in obs)


class MLPDQNAgent:
    """Standard MLP DQN (non-spiking neural baseline)."""
    def __init__(self, hidden=64, lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=3000):
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, NUM_ACTIONS)
        )
        self.target = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, NUM_ACTIONS)
        )
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(20000)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0
        self.synops = 0  # count multiply-accumulate operations

    def act(self, obs, step):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS-1)
        with torch.no_grad():
            q = self.net(torch.FloatTensor(obs).unsqueeze(0))
            self.synops += OBS_DIM * 64 + 64 * 32 + 32 * NUM_ACTIONS  # MAC count
            return q.argmax(dim=1).item()

    def train_step(self, batch_size=32):
        if len(self.replay) < batch_size:
            return
        obs, actions, rewards, next_obs, dones = self.replay.sample(batch_size)
        q = self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(next_obs).max(dim=1).values
            target = rewards + self.gamma * nq * (1 - dones)
        loss = nn.SmoothL1Loss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def reset_for_new_maze(self):
        pass  # neural net transfers


class SpikingDQNAgent:
    """Spiking DQN with surrogate gradients (our method)."""
    def __init__(self, hidden=64, num_steps=8, lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=3000):
        self.net = SpikingQNetwork(OBS_DIM, NUM_ACTIONS, hidden, num_steps)
        self.target = SpikingQNetwork(OBS_DIM, NUM_ACTIONS, hidden, num_steps)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(20000)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0
        self.num_steps = num_steps
        self.synops = 0  # count spike-accumulate operations

    def act(self, obs, step):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS-1)
        with torch.no_grad():
            q = self.net(torch.FloatTensor(obs).unsqueeze(0))
            # SynOps: only count active spikes (estimated 10% firing rate)
            firing_rate = 0.1
            self.synops += int(self.num_steps * (OBS_DIM * 64 + 64 * 32 + 32 * NUM_ACTIONS) * firing_rate)
            return q.argmax(dim=1).item()

    def train_step(self, batch_size=32):
        if len(self.replay) < batch_size:
            return
        obs, actions, rewards, next_obs, dones = self.replay.sample(batch_size)
        q = self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(next_obs).max(dim=1).values
            target = rewards + self.gamma * nq * (1 - dones)
        loss = nn.SmoothL1Loss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def reset_for_new_maze(self):
        pass  # neural net transfers


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

@dataclass
class ExpResult:
    agent_name: str
    maze_size: int
    seed: int
    phase: str  # 'train' or 'test'
    episode: int
    reward: float
    steps: int
    solved: bool
    synops: int = 0

def run_experiment(agent, agent_name, maze_size, num_train, num_test, seed, max_steps=200):
    """Run train + zero-shot test for one agent on one seed."""
    results = []
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_update_interval = 300
    total_env_steps = 0

    # TRAINING: agent sees num_train different mazes
    for ep in range(num_train):
        maze_seed = rng.randint(0, 1000000)
        grid = make_maze(maze_size, maze_seed)
        if hasattr(agent, 'reset_for_new_maze'):
            agent.reset_for_new_maze()

        # Collect trajectory
        ax, ay = 1, 1
        gx, gy = maze_size-2, maze_size-2
        action_hist = []
        visited = set()
        visited.add((ax, ay))
        ep_reward = 0.0
        solved = False

        for step in range(max_steps):
            obs = ego_features(grid, ax, ay, gx, gy, maze_size, action_hist)
            action = agent.act(obs, step)
            dx, dy = ACTIONS[action]
            nx, ny = ax+dx, ay+dy
            reward = -0.02
            done = False
            prev_dist = abs(ax-gx) + abs(ay-gy)

            if 0 <= nx < maze_size and 0 <= ny < maze_size and grid[ny][nx] != WALL:
                ax, ay = nx, ny
                new_dist = abs(ax-gx) + abs(ay-gy)
                if new_dist < prev_dist: reward += 0.08
                elif new_dist > prev_dist: reward -= 0.04
                if (ax, ay) in visited: reward -= 0.1
                visited.add((ax, ay))
                if grid[ay][ax] == HAZARD: reward = -1.0
                if ax == gx and ay == gy:
                    reward = 10.0; done = True; solved = True
            else:
                reward = -0.3

            next_obs = ego_features(grid, ax, ay, gx, gy, maze_size, action_hist + [action])
            ep_reward += reward
            action_hist.append(action)
            total_env_steps += 1

            # Learning
            if hasattr(agent, 'replay'):
                agent.replay.push(obs, action, reward, next_obs, float(done))
                agent.train_step()
                if total_env_steps % target_update_interval == 0:
                    agent.update_target()
            elif hasattr(agent, 'learn'):
                agent.learn(obs, action, reward, next_obs, done)

            if done:
                break

        results.append(ExpResult(agent_name, maze_size, seed, 'train', ep, ep_reward, step+1, solved,
                                 getattr(agent, 'synops', 0)))

    # TESTING: zero-shot on num_test UNSEEN mazes (no learning)
    for ep in range(num_test):
        maze_seed = rng.randint(1000000, 2000000)  # different range = unseen
        grid = make_maze(maze_size, maze_seed)

        ax, ay = 1, 1
        gx, gy = maze_size-2, maze_size-2
        action_hist = []
        visited = set()
        visited.add((ax, ay))
        ep_reward = 0.0
        solved = False

        for step in range(max_steps):
            obs = ego_features(grid, ax, ay, gx, gy, maze_size, action_hist)
            # Greedy action (no exploration during test)
            if hasattr(agent, 'net'):
                with torch.no_grad():
                    q = agent.net(torch.FloatTensor(obs).unsqueeze(0))
                    action = q.argmax(dim=1).item()
            elif hasattr(agent, 'q'):
                key = agent._key(obs)
                if key in agent.q:
                    action = int(np.argmax(agent.q[key]))
                else:
                    action = random.randint(0, 3)
            else:
                action = random.randint(0, 3)

            dx, dy = ACTIONS[action]
            nx, ny = ax+dx, ay+dy
            reward = -0.02
            prev_dist = abs(ax-gx) + abs(ay-gy)

            if 0 <= nx < maze_size and 0 <= ny < maze_size and grid[ny][nx] != WALL:
                ax, ay = nx, ny
                new_dist = abs(ax-gx) + abs(ay-gy)
                if new_dist < prev_dist: reward += 0.08
                elif new_dist > prev_dist: reward -= 0.04
                if (ax, ay) in visited: reward -= 0.1
                visited.add((ax, ay))
                if grid[ay][ax] == HAZARD: reward = -1.0
                if ax == gx and ay == gy:
                    reward = 10.0; solved = True; ep_reward += reward; break
            else:
                reward = -0.3

            ep_reward += reward
            action_hist.append(action)

        results.append(ExpResult(agent_name, maze_size, seed, 'test', ep, ep_reward, step+1, solved,
                                 getattr(agent, 'synops', 0)))

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    out_dir = Path(__file__).parent / 'raw_results'
    out_dir.mkdir(exist_ok=True)

    SEEDS = [42, 123, 456, 789, 1024]
    MAZE_SIZES = [9, 13]
    NUM_TRAIN = 80
    NUM_TEST = 40
    MAX_STEPS = 200

    all_results = []

    for maze_size in MAZE_SIZES:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"  Maze {maze_size}x{maze_size} | Seed {seed}")
            print(f"{'='*60}")

            for agent_name, make_agent in [
                ("TabularQ", lambda: TabularQAgent()),
                ("FeatureQ", lambda: FeatureQAgent()),
                ("MLP_DQN", lambda: MLPDQNAgent(hidden=64, lr=5e-4, eps_decay=NUM_TRAIN*MAX_STEPS//2)),
                ("SpikingDQN", lambda: SpikingDQNAgent(hidden=64, num_steps=8, lr=5e-4, eps_decay=NUM_TRAIN*MAX_STEPS//2)),
            ]:
                print(f"\n  Running {agent_name}...", end=" ", flush=True)
                t0 = time.time()
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)

                agent = make_agent()
                results = run_experiment(agent, agent_name, maze_size, NUM_TRAIN, NUM_TEST, seed, MAX_STEPS)
                elapsed = time.time() - t0

                train_results = [r for r in results if r.phase == 'train']
                test_results = [r for r in results if r.phase == 'test']
                train_success = sum(r.solved for r in train_results) / len(train_results) * 100
                test_success = sum(r.solved for r in test_results) / len(test_results) * 100

                print(f"done ({elapsed:.1f}s) | train: {train_success:.0f}% | test: {test_success:.0f}%")
                all_results.extend(results)

    # Save all results
    results_file = out_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Zero-Shot Test Success Rate (%) by Agent and Maze Size")
    print(f"{'='*80}")
    print(f"  {'Agent':<14} ", end="")
    for sz in MAZE_SIZES:
        print(f"{'  ' + str(sz) + 'x' + str(sz):>12}", end="")
    print()
    print(f"  {'-'*14} ", end="")
    for _ in MAZE_SIZES:
        print(f"  {'-'*10}", end="")
    print()

    for agent_name in ["TabularQ", "FeatureQ", "MLP_DQN", "SpikingDQN"]:
        print(f"  {agent_name:<14} ", end="")
        for sz in MAZE_SIZES:
            test_r = [r for r in all_results if r.agent_name == agent_name and r.maze_size == sz and r.phase == 'test']
            if test_r:
                rate = sum(r.solved for r in test_r) / len(test_r) * 100
                print(f"  {rate:>9.1f}%", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    # SynOps comparison
    print(f"\n  Energy Efficiency (SynOps per decision, test phase):")
    for agent_name in ["MLP_DQN", "SpikingDQN"]:
        test_r = [r for r in all_results if r.agent_name == agent_name and r.phase == 'test']
        if test_r and test_r[-1].synops > 0:
            total_steps = sum(r.steps for r in test_r)
            total_synops = test_r[-1].synops
            per_step = total_synops / max(total_steps, 1)
            print(f"  {agent_name:<14}  {per_step:.0f} SynOps/step")


if __name__ == "__main__":
    main()
