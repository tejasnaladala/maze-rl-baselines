"""Shared infrastructure for all Approach C experiments.

Contains: maze generation, feature extraction, 10 agent implementations,
experiment runner, result I/O, and energy measurement utilities.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate


# ============================================================
# SPIKING Q-NETWORK (inlined -- no Rust dependency needed)
# ============================================================

class SpikingQNetwork(nn.Module):
    """Spiking Q-Network with surrogate gradients, non-spiking LI output."""
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64, num_steps: int = 8):
        super().__init__()
        self.num_steps = num_steps
        self.num_actions = num_actions
        sg = surrogate.atan(alpha=2.0)
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.lif1 = snn.Leaky(beta=0.9, learn_beta=True, spike_grad=sg, reset_mechanism='subtract')
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.lif2 = snn.Leaky(beta=0.85, learn_beta=True, spike_grad=sg, reset_mechanism='subtract')
        self.fc_out = nn.Linear(hidden // 2, num_actions)
        self.li_out = snn.Leaky(beta=0.95, learn_beta=True, spike_grad=sg, reset_mechanism='none', output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.li_out.init_leaky()
        max_mem = torch.full((batch, self.num_actions), -1e9, device=x.device)
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _, mem_out = self.li_out(self.fc_out(spk2), mem_out)
            max_mem = torch.max(max_mem, mem_out)
        return max_mem


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
# CONSTANTS
# ============================================================

WALL = 1
HAZARD = 3
ACTIONS: list[tuple[int, int]] = [(0, -1), (1, 0), (0, 1), (-1, 0)]
NUM_ACTIONS = 4
OBS_DIM = 24  # ego-centric feature vector size

# ============================================================
# MAZE GENERATION
# ============================================================

def mulberry32(seed: int):
    """Seedable PRNG for deterministic maze generation."""
    def rng() -> float:
        nonlocal seed
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((seed ^ (seed >> 15)) * (1 | seed)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng


def make_maze(size: int, seed: int) -> list[list[int]]:
    """Generate a solvable maze using recursive backtracking. Size must be odd."""
    assert size % 2 == 1, f"Size must be odd, got {size}"
    g = [[WALL] * size for _ in range(size)]
    rng = mulberry32(seed)

    def carve(x: int, y: int) -> None:
        g[y][x] = 0
        dirs: list[tuple[int, int]] = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        for i in range(3, 0, -1):
            j = int(rng() * (i + 1))
            dirs[i], dirs[j] = dirs[j], dirs[i]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= size - 2 and 1 <= ny <= size - 2 and g[ny][nx] == WALL:
                g[y + dy // 2][x + dx // 2] = 0
                carve(nx, ny)

    carve(1, 1)
    # Add hazards
    for _ in range(size // 3):
        for attempt in range(30):
            hx = 1 + int(rng() * (size - 2))
            hy = 1 + int(rng() * (size - 2))
            if g[hy][hx] == 0 and not (hx == 1 and hy == 1) and not (hx == size - 2 and hy == size - 2):
                g[hy][hx] = HAZARD
                break
    return g


def make_dense_maze(size: int, seed: int) -> list[list[int]]:
    """Maze with extra walls added (tight corridors)."""
    g = make_maze(size, seed)
    rng = mulberry32(seed + 999)
    added = 0
    for _ in range(size * 2):
        x = 1 + int(rng() * (size - 2))
        y = 1 + int(rng() * (size - 2))
        if g[y][x] == 0 and not (x == 1 and y == 1) and not (x == size - 2 and y == size - 2):
            # Only add wall if it doesn't block the path
            g[y][x] = WALL
            if is_solvable(g, size):
                added += 1
                if added >= size // 2:
                    break
            else:
                g[y][x] = 0  # revert
    return g


def make_sparse_maze(size: int, seed: int) -> list[list[int]]:
    """Maze with walls removed (open space)."""
    g = make_maze(size, seed)
    rng = mulberry32(seed + 777)
    for _ in range(size * 3):
        x = 1 + int(rng() * (size - 2))
        y = 1 + int(rng() * (size - 2))
        if g[y][x] == WALL:
            g[y][x] = 0
    return g


def is_solvable(grid: list[list[int]], size: int) -> bool:
    """BFS check that start (1,1) can reach goal (size-2, size-2)."""
    visited = set()
    queue: list[tuple[int, int]] = [(1, 1)]
    visited.add((1, 1))
    gx, gy = size - 2, size - 2
    while queue:
        x, y = queue.pop(0)
        if x == gx and y == gy:
            return True
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] != WALL and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def ego_features(
    grid: list[list[int]], ax: int, ay: int, gx: int, gy: int,
    size: int, action_hist: list[int],
) -> list[float]:
    """24-dim ego-centric feature vector."""
    feats: list[float] = []
    # 3x3 local map (9 values)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < size and 0 <= ny < size:
                feats.append(grid[ny][nx] / 3.0)
            else:
                feats.append(1.0)
    # Goal direction (2 values)
    feats.append(float(np.sign(gx - ax)))
    feats.append(float(np.sign(gy - ay)))
    # Distance (1 value)
    max_dist = size * 2
    feats.append((abs(gx - ax) + abs(gy - ay)) / max_dist)
    # Last 3 actions one-hot (12 values)
    hist = list(action_hist[-3:])
    while len(hist) < 3:
        hist.insert(0, -1)
    for a in hist:
        oh = [0.0] * 4
        if 0 <= a < 4:
            oh[a] = 1.0
        feats.extend(oh)
    return feats


def ego_features_ablated(
    grid: list[list[int]], ax: int, ay: int, gx: int, gy: int,
    size: int, action_hist: list[int], ablation: str,
) -> list[float]:
    """Feature vector with one component removed for ablation study."""
    if ablation == 'full':
        return ego_features(grid, ax, ay, gx, gy, size, action_hist)

    feats: list[float] = []

    # 3x3 map or 4-direction walls
    if ablation == 'no_3x3_map':
        # Only 4-direction walls (4 values instead of 9)
        for dx, dy in ACTIONS:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < size and 0 <= ny < size:
                feats.append(float(grid[ny][nx] == WALL))
            else:
                feats.append(1.0)
    elif ablation == 'walls_only':
        for dx, dy in ACTIONS:
            nx, ny = ax + dx, ay + dy
            feats.append(1.0 if (nx < 0 or nx >= size or ny < 0 or ny >= size or grid[ny][nx] == WALL) else 0.0)
    else:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < size and 0 <= ny < size:
                    feats.append(grid[ny][nx] / 3.0)
                else:
                    feats.append(1.0)

    # Goal direction
    if ablation != 'no_goal_dir' and ablation != 'walls_only':
        feats.append(float(np.sign(gx - ax)))
        feats.append(float(np.sign(gy - ay)))
    elif ablation == 'walls_only':
        feats.append(float(np.sign(gx - ax)))
        feats.append(float(np.sign(gy - ay)))

    # Distance
    if ablation != 'no_distance':
        max_dist = size * 2
        feats.append((abs(gx - ax) + abs(gy - ay)) / max_dist)

    # Action history
    if ablation != 'no_action_hist':
        hist = list(action_hist[-3:])
        while len(hist) < 3:
            hist.insert(0, -1)
        for a in hist:
            oh = [0.0] * 4
            if 0 <= a < 4:
                oh[a] = 1.0
            feats.extend(oh)

    return feats


# ============================================================
# RESULT DATA
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
    wall_time_s: float = 0.0
    config: dict = field(default_factory=dict)


def save_results(results: list[ExpResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_results(path: Path) -> list[ExpResult]:
    with open(path) as f:
        data = json.load(f)
    return [ExpResult(**d) for d in data]


# ============================================================
# AGENT PROTOCOL
# ============================================================

class Agent(Protocol):
    def act(self, obs: list[float], step: int) -> int: ...
    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None: ...
    def reset_for_new_maze(self) -> None: ...
    def get_synops(self) -> int: ...


# ============================================================
# AGENT 1: RANDOM
# ============================================================

class RandomAgent:
    def act(self, obs: list[float], step: int) -> int:
        return random.randint(0, NUM_ACTIONS - 1)
    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None:
        pass
    def reset_for_new_maze(self) -> None:
        pass
    def get_synops(self) -> int:
        return 0


# ============================================================
# AGENT 2: TABULAR Q (position-based)
# ============================================================

class TabularQAgent:
    def __init__(self, lr: float = 0.15, gamma: float = 0.99, eps: float = 0.4) -> None:
        self.q: dict[tuple, np.ndarray] = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def act(self, obs: list[float], step: int) -> int:
        key = self._key(obs)
        if key not in self.q:
            self.q[key] = np.zeros(NUM_ACTIONS)
        if random.random() < self.eps:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q[key]))

    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None:
        key = self._key(obs)
        nkey = self._key(next_obs)
        if key not in self.q:
            self.q[key] = np.zeros(NUM_ACTIONS)
        if nkey not in self.q:
            self.q[nkey] = np.zeros(NUM_ACTIONS)
        target = reward + (0 if done else self.gamma * float(np.max(self.q[nkey])))
        self.q[key][action] += self.lr * (target - self.q[key][action])

    def reset_for_new_maze(self) -> None:
        self.q = {}
        self.eps = 0.4

    def get_synops(self) -> int:
        return 0

    def _key(self, obs: list[float]) -> tuple:
        return tuple(round(v, 1) for v in obs[:12])


# ============================================================
# AGENT 3: FEATURE Q (ego-centric, persists across mazes)
# ============================================================

class FeatureQAgent:
    def __init__(self, lr: float = 0.2, gamma: float = 0.99, eps: float = 0.3) -> None:
        self.q: dict[tuple, np.ndarray] = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def act(self, obs: list[float], step: int) -> int:
        key = self._key(obs)
        if key not in self.q:
            self.q[key] = np.ones(NUM_ACTIONS) * 1.0
        if random.random() < max(0.08, self.eps):
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q[key]))

    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None:
        key = self._key(obs)
        nkey = self._key(next_obs)
        if key not in self.q:
            self.q[key] = np.ones(NUM_ACTIONS) * 1.0
        if nkey not in self.q:
            self.q[nkey] = np.ones(NUM_ACTIONS) * 1.0
        target = reward + (0 if done else self.gamma * float(np.max(self.q[nkey])))
        self.q[key][action] += self.lr * (target - self.q[key][action])
        self.eps = max(0.08, self.eps * 0.999)

    def reset_for_new_maze(self) -> None:
        pass  # features transfer

    def get_synops(self) -> int:
        return 0

    def _key(self, obs: list[float]) -> tuple:
        return tuple(round(v, 1) for v in obs)


# ============================================================
# AGENT 4: MLP DQN
# ============================================================

class MLPDQNAgent:
    def __init__(self, hidden: int = 64, lr: float = 5e-4, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 5000,
                 target_update: int = 300, buffer_size: int = 20000,
                 batch_size: int = 64, device: str = 'auto') -> None:
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
        self.target = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps = 0
        self.synops = 0
        self._macs_per_forward = OBS_DIM * hidden + hidden * (hidden // 2) + (hidden // 2) * NUM_ACTIONS

    def act(self, obs: list[float], step: int) -> int:
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.synops += self._macs_per_forward
            return q.argmax(dim=1).item()

    def eval_action(self, obs: list[float]) -> int:
        """Deterministic greedy action for test phase -- no epsilon, no state mutation."""
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return q.argmax(dim=1).item()

    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None:
        self.replay.push(obs, action, reward, next_obs, float(done))
        if len(self.replay) < self.batch_size:
            return
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

    def reset_for_new_maze(self) -> None:
        pass

    def get_synops(self) -> int:
        return self.synops


# ============================================================
# AGENT 5: SPIKING DQN
# ============================================================

class SpikingDQNAgent:
    def __init__(self, hidden: int = 64, num_steps: int = 8, lr: float = 5e-4,
                 gamma: float = 0.99, eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 5000, target_update: int = 300, buffer_size: int = 20000,
                 batch_size: int = 64, device: str = 'auto') -> None:
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = SpikingQNetwork(OBS_DIM, NUM_ACTIONS, hidden, num_steps).to(self.device)
        self.target = SpikingQNetwork(OBS_DIM, NUM_ACTIONS, hidden, num_steps).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.target_update = target_update
        self.num_steps = num_steps
        self.steps = 0
        self.synops = 0
        self._firing_rate = 0.1
        self._ops_per_step = int(num_steps * (OBS_DIM * hidden + hidden * (hidden // 2) + (hidden // 2) * NUM_ACTIONS) * self._firing_rate)

    def act(self, obs: list[float], step: int) -> int:
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.synops += self._ops_per_step
            return q.argmax(dim=1).item()

    def eval_action(self, obs: list[float]) -> int:
        """Deterministic greedy action for test phase."""
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return q.argmax(dim=1).item()

    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None:
        self.replay.push(obs, action, reward, next_obs, float(done))
        if len(self.replay) < self.batch_size:
            return
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

    def reset_for_new_maze(self) -> None:
        pass

    def get_synops(self) -> int:
        return self.synops


# ============================================================
# AGENT 6: DOUBLE DQN
# ============================================================

class DoubleDQNAgent:
    """Double DQN: uses online network for action selection, target for evaluation."""
    def __init__(self, hidden: int = 64, lr: float = 5e-4, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 5000,
                 target_update: int = 300, buffer_size: int = 20000,
                 batch_size: int = 64, device: str = 'auto') -> None:
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
        self.target = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps = 0
        self.synops = 0
        self._macs = OBS_DIM * hidden + hidden * (hidden // 2) + (hidden // 2) * NUM_ACTIONS

    def act(self, obs: list[float], step: int) -> int:
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.synops += self._macs
            return q.argmax(dim=1).item()

    def eval_action(self, obs: list[float]) -> int:
        """Deterministic greedy action for test phase."""
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return q.argmax(dim=1).item()

    def learn(self, obs: list[float], action: int, reward: float, next_obs: list[float], done: bool) -> None:
        self.replay.push(obs, action, reward, next_obs, float(done))
        if len(self.replay) < self.batch_size:
            return
        o, a, r, no, d = self.replay.sample(self.batch_size)
        o, a, r, no, d = o.to(self.device), a.to(self.device), r.to(self.device), no.to(self.device), d.to(self.device)
        q = self.net(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = self.net(no).argmax(dim=1)
            nq = self.target(no).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * nq * (1 - d)
        loss = nn.SmoothL1Loss()(q, target)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())

    def reset_for_new_maze(self) -> None:
        pass

    def get_synops(self) -> int:
        return self.synops


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    agent: Agent,
    agent_name: str,
    maze_size: int,
    num_train: int,
    num_test: int,
    seed: int,
    max_steps: int = 0,  # 0 = auto-scale by maze size
    reward_shaping: bool = True,
    visit_penalty: bool = True,
    feature_fn=None,
) -> list[ExpResult]:
    """Run train + zero-shot test for one agent on one seed."""
    if feature_fn is None:
        feature_fn = ego_features

    # FIX 6: Scale max_steps by maze size (Codex review)
    if max_steps <= 0:
        max_steps = max(300, 4 * maze_size * maze_size)

    results: list[ExpResult] = []
    rng = random.Random(seed)

    def run_phase(phase: str, num_eps: int, learn: bool, seed_offset: int) -> None:
        for ep in range(num_eps):
            maze_seed = rng.randint(0, 10_000_000) + seed_offset
            grid = make_maze(maze_size, maze_seed)
            if hasattr(agent, 'reset_for_new_maze') and phase == 'train':
                agent.reset_for_new_maze()

            ax, ay = 1, 1
            gx, gy = maze_size - 2, maze_size - 2
            action_hist: list[int] = []
            visited = {(1, 1)}
            ep_reward = 0.0
            solved = False

            for step in range(max_steps):
                obs = feature_fn(grid, ax, ay, gx, gy, maze_size, action_hist)
                # FIX 3: Use deterministic eval for test phase (Codex review)
                if not learn and hasattr(agent, 'eval_action'):
                    action = agent.eval_action(obs)
                else:
                    action = agent.act(obs, step)
                dx, dy = ACTIONS[action]
                nx, ny = ax + dx, ay + dy
                reward = -0.02
                done = False
                prev_dist = abs(ax - gx) + abs(ay - gy)

                if 0 <= nx < maze_size and 0 <= ny < maze_size and grid[ny][nx] != WALL:
                    ax, ay = nx, ny
                    if reward_shaping:
                        new_dist = abs(ax - gx) + abs(ay - gy)
                        if new_dist < prev_dist:
                            reward += 0.08
                        elif new_dist > prev_dist:
                            reward -= 0.04
                    if visit_penalty and (ax, ay) in visited:
                        reward -= 0.1
                    visited.add((ax, ay))
                    if grid[ay][ax] == HAZARD:
                        reward = -1.0
                    if ax == gx and ay == gy:
                        reward = 10.0
                        done = True
                        solved = True
                else:
                    reward = -0.3

                next_obs = feature_fn(grid, ax, ay, gx, gy, maze_size, action_hist + [action])
                ep_reward += reward
                action_hist.append(action)

                if learn:
                    agent.learn(obs, action, reward, next_obs, done)

                if done:
                    break

            results.append(ExpResult(
                agent_name=agent_name, maze_size=maze_size, seed=seed,
                phase=phase, episode=ep, reward=ep_reward,
                steps=step + 1, solved=solved,
                synops=agent.get_synops(),
            ))

    # Train phase
    run_phase('train', num_train, learn=True, seed_offset=0)
    # Test phase (zero-shot, deterministic greedy, no learning, no state mutation)
    run_phase('test', num_test, learn=False, seed_offset=10_000_000)

    return results


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def load_checkpoint(path: Path) -> set[str]:
    """Load set of completed run keys from checkpoint file."""
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(sorted(completed), f)


def run_key(agent_name: str, maze_size: int, seed: int) -> str:
    return f"{agent_name}_{maze_size}_{seed}"
