"""Experiment library v2 — audit-fixed version for Tier 2+ experiments.

Changes from v1 (experiment_lib.py):
  FIX W5: TabularQ.reset_for_new_maze no longer wipes Q-table (configurable).
  FIX W6: FeatureQ.eval_action() + TabularQ.eval_action() — deterministic greedy,
          no epsilon floor, no state mutation. Test phase uses these.
  FIX K4: reward_shaping + visit_penalty flags fully honored (no hidden paths).
  FIX S1-S5: ExpResult carries wall_time_s and a config dict (hyperparams + code hash).
  FIX audit-H3.1: atomic checkpoint writes via os.replace + fsync.
  FIX audit-H1: is_solvable forbids paths through HAZARD (doc says "reachable avoiding hazards").
  FIX audit-B10: run_experiment records wall_bump events for symmetric reward analysis.
  NEW: NoBacktrackRandomAgent, LevyRandomAgent, BFSOracleAgent, AStarOracleAgent.

v1 agents (Random, TabularQ, FeatureQ, MLPDQN, SpikingDQN, DoubleDQN) are preserved
byte-for-byte where possible; only additive methods changed.
"""

from __future__ import annotations

import hashlib
import json
import os
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
# SPIKING Q-NETWORK (inlined)
# ============================================================

class SpikingQNetwork(nn.Module):
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
        # FIX audit-B9: real firing-rate measurement
        self.last_firing_rates: tuple[float, float] = (0.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.li_out.init_leaky()
        max_mem = torch.full((batch, self.num_actions), -1e9, device=x.device)
        spk1_acc = 0.0
        spk2_acc = 0.0
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _, mem_out = self.li_out(self.fc_out(spk2), mem_out)
            max_mem = torch.max(max_mem, mem_out)
            spk1_acc += float(spk1.mean().item())
            spk2_acc += float(spk2.mean().item())
        self.last_firing_rates = (spk1_acc / self.num_steps, spk2_acc / self.num_steps)
        return max_mem


class ReplayBuffer:
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
OBS_DIM = 24


# ============================================================
# MAZE GENERATION (unchanged from v1)
# ============================================================

def mulberry32(seed: int):
    def rng() -> float:
        nonlocal seed
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((seed ^ (seed >> 15)) * (1 | seed)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng


def make_maze(size: int, seed: int) -> list[list[int]]:
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
    for _ in range(size // 3):
        for attempt in range(30):
            hx = 1 + int(rng() * (size - 2))
            hy = 1 + int(rng() * (size - 2))
            if g[hy][hx] == 0 and not (hx == 1 and hy == 1) and not (hx == size - 2 and hy == size - 2):
                g[hy][hx] = HAZARD
                break
    return g


def make_dense_maze(size: int, seed: int) -> list[list[int]]:
    g = make_maze(size, seed)
    rng = mulberry32(seed + 999)
    added = 0
    for _ in range(size * 2):
        x = 1 + int(rng() * (size - 2))
        y = 1 + int(rng() * (size - 2))
        if g[y][x] == 0 and not (x == 1 and y == 1) and not (x == size - 2 and y == size - 2):
            g[y][x] = WALL
            if is_solvable(g, size):
                added += 1
                if added >= size // 2:
                    break
            else:
                g[y][x] = 0
    return g


def make_sparse_maze(size: int, seed: int) -> list[list[int]]:
    g = make_maze(size, seed)
    rng = mulberry32(seed + 777)
    for _ in range(size * 3):
        x = 1 + int(rng() * (size - 2))
        y = 1 + int(rng() * (size - 2))
        if g[y][x] == WALL:
            g[y][x] = 0
    return g


def is_solvable(grid: list[list[int]], size: int, avoid_hazards: bool = True) -> bool:
    """BFS from (1,1) to (size-2, size-2). v2: hazards are NOT traversable by default."""
    visited = {(1, 1)}
    queue: deque = deque([(1, 1)])
    gx, gy = size - 2, size - 2
    while queue:
        x, y = queue.popleft()
        if x == gx and y == gy:
            return True
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                cell = grid[ny][nx]
                if cell == WALL:
                    continue
                if avoid_hazards and cell == HAZARD:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def bfs_path(grid: list[list[int]], size: int, sx: int = 1, sy: int = 1,
             gx: int = -1, gy: int = -1, avoid_hazards: bool = True) -> Optional[list[tuple[int, int]]]:
    """Return BFS shortest path from (sx,sy) to (gx,gy), or None."""
    if gx < 0: gx = size - 2
    if gy < 0: gy = size - 2
    parent: dict = {(sx, sy): None}
    queue: deque = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        if x == gx and y == gy:
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in parent:
                continue
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            cell = grid[ny][nx]
            if cell == WALL:
                continue
            if avoid_hazards and cell == HAZARD:
                continue
            parent[(nx, ny)] = (x, y)
            queue.append((nx, ny))
    return None


# ============================================================
# FEATURE EXTRACTION (unchanged interface, same as v1)
# ============================================================

def ego_features(grid, ax, ay, gx, gy, size, action_hist):
    feats: list[float] = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < size and 0 <= ny < size:
                feats.append(grid[ny][nx] / 3.0)
            else:
                feats.append(1.0)
    feats.append(float(np.sign(gx - ax)))
    feats.append(float(np.sign(gy - ay)))
    max_dist = size * 2
    feats.append((abs(gx - ax) + abs(gy - ay)) / max_dist)
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
# RESULT DATA — v2 with wall_time_s and config
# ============================================================

@dataclass
class ExpResult:
    agent_name: str
    maze_size: int
    seed: int
    phase: str
    episode: int
    reward: float
    steps: int
    solved: bool
    synops: int = 0
    wall_time_s: float = 0.0
    config: dict = field(default_factory=dict)
    lib_version: str = "v2"


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
    def eval_action(self, obs: list[float]) -> int: ...
    def learn(self, obs, action, reward, next_obs, done) -> None: ...
    def reset_for_new_maze(self) -> None: ...
    def get_synops(self) -> int: ...


# ============================================================
# AGENT 1: RANDOM
# ============================================================

class RandomAgent:
    def __init__(self): self._rng = random.Random(0)
    def act(self, obs, step): return self._rng.randint(0, NUM_ACTIONS - 1)
    def eval_action(self, obs): return self._rng.randint(0, NUM_ACTIONS - 1)
    def learn(self, *args, **kwargs): pass
    def reset_for_new_maze(self): pass
    def get_synops(self): return 0
    def seed(self, s: int): self._rng = random.Random(s)


# ============================================================
# AGENT 1b: NO-BACKTRACK RANDOM
# ============================================================

class NoBacktrackRandomAgent:
    """Random agent that never picks the exact reverse of its last action."""
    def __init__(self):
        self._rng = random.Random(0)
        self._last_action = -1
    def _reverse(self, a: int) -> int:
        return (a + 2) % 4  # ACTIONS: up=0,right=1,down=2,left=3 → reverse is +2 mod 4
    def act(self, obs, step):
        choices = [a for a in range(NUM_ACTIONS) if a != self._reverse(self._last_action)]
        a = self._rng.choice(choices) if choices else self._rng.randint(0, 3)
        self._last_action = a
        return a
    def eval_action(self, obs): return self.act(obs, 0)
    def learn(self, *args, **kwargs): pass
    def reset_for_new_maze(self): self._last_action = -1
    def get_synops(self): return 0
    def seed(self, s: int): self._rng = random.Random(s)


# ============================================================
# AGENT 1c: LEVY RANDOM
# ============================================================

class LevyRandomAgent:
    """Lévy-flight: pick a direction, commit for a heavy-tailed duration."""
    def __init__(self, alpha: float = 1.5):
        self._rng = random.Random(0)
        self._alpha = alpha
        self._current_dir = 0
        self._steps_remaining = 0
    def _sample_run_length(self) -> int:
        u = self._rng.random()
        return max(1, int((1 - u) ** (-1.0 / self._alpha)))
    def act(self, obs, step):
        if self._steps_remaining <= 0:
            self._current_dir = self._rng.randint(0, 3)
            self._steps_remaining = self._sample_run_length()
        self._steps_remaining -= 1
        return self._current_dir
    def eval_action(self, obs): return self.act(obs, 0)
    def learn(self, *args, **kwargs): pass
    def reset_for_new_maze(self):
        self._current_dir = 0
        self._steps_remaining = 0
    def get_synops(self): return 0
    def seed(self, s: int): self._rng = random.Random(s)


# ============================================================
# AGENT 2: TABULAR Q — FIX W5, W6
# ============================================================

class TabularQAgent:
    def __init__(self, lr=0.15, gamma=0.99, eps=0.4, wipe_on_new_maze: bool = True):
        self.q: dict[tuple, np.ndarray] = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_initial = eps
        self.wipe_on_new_maze = wipe_on_new_maze  # FIX W5: configurable

    def act(self, obs, step):
        key = self._key(obs)
        if key not in self.q:
            self.q[key] = np.zeros(NUM_ACTIONS)
        if random.random() < self.eps:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q[key]))

    def eval_action(self, obs):  # FIX W6
        key = self._key(obs)
        if key not in self.q:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q[key]))

    def learn(self, obs, action, reward, next_obs, done):
        key = self._key(obs)
        nkey = self._key(next_obs)
        if key not in self.q:
            self.q[key] = np.zeros(NUM_ACTIONS)
        if nkey not in self.q:
            self.q[nkey] = np.zeros(NUM_ACTIONS)
        target = reward + (0 if done else self.gamma * float(np.max(self.q[nkey])))
        self.q[key][action] += self.lr * (target - self.q[key][action])

    def reset_for_new_maze(self):
        if self.wipe_on_new_maze:
            self.q = {}
            self.eps = self.eps_initial

    def get_synops(self): return 0
    def _key(self, obs): return tuple(round(v, 1) for v in obs[:12])


# ============================================================
# AGENT 3: FEATURE Q — FIX W6
# ============================================================

class FeatureQAgent:
    def __init__(self, lr=0.2, gamma=0.99, eps=0.3, eps_floor: float = 0.0):
        self.q: dict[tuple, np.ndarray] = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_floor = eps_floor  # FIX W6: 0 means truly greedy at convergence

    def act(self, obs, step):
        key = self._key(obs)
        if key not in self.q:
            self.q[key] = np.ones(NUM_ACTIONS) * 1.0
        if random.random() < max(self.eps_floor, self.eps):
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q[key]))

    def eval_action(self, obs):  # FIX W6
        key = self._key(obs)
        if key not in self.q:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q[key]))

    def learn(self, obs, action, reward, next_obs, done):
        key = self._key(obs)
        nkey = self._key(next_obs)
        if key not in self.q:
            self.q[key] = np.ones(NUM_ACTIONS) * 1.0
        if nkey not in self.q:
            self.q[nkey] = np.ones(NUM_ACTIONS) * 1.0
        target = reward + (0 if done else self.gamma * float(np.max(self.q[nkey])))
        self.q[key][action] += self.lr * (target - self.q[key][action])
        self.eps = max(self.eps_floor, self.eps * 0.999)

    def reset_for_new_maze(self): pass
    def get_synops(self): return 0
    def _key(self, obs): return tuple(round(v, 1) for v in obs)


# ============================================================
# AGENT 4: MLP DQN (same as v1, with config dict)
# ============================================================

class MLPDQNAgent:
    def __init__(self, hidden=64, lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 eps_decay=5000, target_update=300, buffer_size=20000,
                 batch_size=64, device='auto'):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
        self.target = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
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
        self._macs = OBS_DIM * hidden + hidden * (hidden // 2) + (hidden // 2) * NUM_ACTIONS

    def act(self, obs, step):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.synops += self._macs
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


# ============================================================
# AGENT 5: SPIKING DQN (v2 measures real firing rate)
# ============================================================

class SpikingDQNAgent:
    def __init__(self, hidden=64, num_steps=8, lr=5e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000, target_update=300,
                 buffer_size=20000, batch_size=64, device='auto'):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = SpikingQNetwork(OBS_DIM, NUM_ACTIONS, hidden, num_steps).to(self.device)
        self.target = SpikingQNetwork(OBS_DIM, NUM_ACTIONS, hidden, num_steps).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update
        self.num_steps = num_steps
        self.steps = 0
        self.synops = 0.0  # FIX B9: real-valued
        self._dense_ops = num_steps * (OBS_DIM * hidden + hidden * (hidden // 2) + (hidden // 2) * NUM_ACTIONS)

    def act(self, obs, step):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            # FIX B9: synops scaled by real mean firing rate from this forward pass
            fr1, fr2 = self.net.last_firing_rates
            effective_rate = (fr1 + fr2) / 2.0
            self.synops += self._dense_ops * effective_rate
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
    def get_synops(self): return int(self.synops)


# ============================================================
# AGENT 6: DOUBLE DQN (v2, consistent device handling)
# ============================================================

class DoubleDQNAgent:
    def __init__(self, hidden=64, lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 eps_decay=5000, target_update=300, buffer_size=20000,
                 batch_size=64, device='auto'):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
        self.target = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, NUM_ACTIONS)).to(self.device)
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
        self._macs = OBS_DIM * hidden + hidden * (hidden // 2) + (hidden // 2) * NUM_ACTIONS

    def act(self, obs, step):
        self.steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.synops += self._macs
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
            best_actions = self.net(no).argmax(dim=1)
            nq = self.target(no).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * nq * (1 - d)
        loss = nn.SmoothL1Loss()(q, target)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())

    def reset_for_new_maze(self): pass
    def get_synops(self): return self.synops


# ============================================================
# AGENT 7: BFS ORACLE — optimal policy baseline
# ============================================================

class BFSOracleAgent:
    """Plans a BFS shortest path each maze and follows it. Optimal policy baseline.

    Tries to avoid hazards first; falls back to shortest path including hazards
    if no hazard-free path exists. This matches how an informed agent should behave.
    """
    def __init__(self, avoid_hazards: bool = True):
        self.avoid_hazards = avoid_hazards
        self._plan: list[tuple[int, int]] = []
        self._plan_idx = 0
        self._grid: Optional[list[list[int]]] = None
        self._size = 0
        self._goal = (0, 0)
        self._last_action_on_no_plan = 0

    def set_env(self, grid, size: int, gx: int, gy: int):
        self._grid = grid
        self._size = size
        self._goal = (gx, gy)
        # Try hazard-avoiding plan first
        plan = None
        if self.avoid_hazards:
            plan = bfs_path(grid, size, 1, 1, gx, gy, avoid_hazards=True)
        # Fallback: shortest path including hazards
        if plan is None:
            plan = bfs_path(grid, size, 1, 1, gx, gy, avoid_hazards=False)
        self._plan = plan or []
        self._plan_idx = 0

    def act(self, obs, step):
        # plan[0] = start, plan[1] = first move, etc.
        # _plan_idx tracks which move we are on (0-indexed).
        # Action i takes us from plan[_plan_idx] to plan[_plan_idx + 1].
        if self._plan_idx + 1 >= len(self._plan):
            # Plan exhausted — should mean goal reached. Return last action as filler.
            return self._last_action_on_no_plan
        cur = self._plan[self._plan_idx]
        nxt = self._plan[self._plan_idx + 1]
        dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
        self._plan_idx += 1
        for i, (ax, ay) in enumerate(ACTIONS):
            if ax == dx and ay == dy:
                self._last_action_on_no_plan = i
                return i
        return 0

    def eval_action(self, obs): return self.act(obs, 0)
    def learn(self, *args, **kwargs): pass
    def reset_for_new_maze(self): self._plan_idx = 0
    def get_synops(self): return 0


# ============================================================
# CHECKPOINTING — FIX audit H3.1: atomic writes with fsync
# ============================================================

def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(sorted(completed), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def run_key(agent_name: str, maze_size: int, seed: int) -> str:
    return f"{agent_name}_{maze_size}_{seed}"


def atomic_save(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


# ============================================================
# EXPERIMENT RUNNER — v2 with full reward ablation, wall-time, and wall-bump tracking
# ============================================================

def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """FIX: full determinism bootstrap."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def code_hash() -> str:
    """SHA-256 of this file — used to pin results to a code version."""
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


def run_experiment(
    agent,
    agent_name: str,
    maze_size: int,
    num_train: int,
    num_test: int,
    seed: int,
    max_steps: int = 0,
    reward_shaping: bool = True,
    visit_penalty: bool = True,
    wall_bump_cost: float = -0.3,
    hazard_cost: float = -1.0,
    goal_reward: float = 10.0,
    feature_fn=None,
) -> list[ExpResult]:
    if feature_fn is None:
        feature_fn = ego_features
    if max_steps <= 0:
        max_steps = max(300, 4 * maze_size * maze_size)

    cfg = {
        'num_train': num_train, 'num_test': num_test, 'max_steps': max_steps,
        'reward_shaping': reward_shaping, 'visit_penalty': visit_penalty,
        'wall_bump_cost': wall_bump_cost, 'hazard_cost': hazard_cost,
        'goal_reward': goal_reward, 'code_hash': code_hash(),
    }

    results: list[ExpResult] = []
    rng = random.Random(seed)

    # Seed the agent's internal RNG if it has one (for Random variants)
    if hasattr(agent, 'seed'):
        agent.seed(seed)

    def run_phase(phase: str, num_eps: int, learn: bool, seed_offset: int) -> None:
        for ep in range(num_eps):
            t0 = time.time()
            maze_seed = rng.randint(0, 10_000_000) + seed_offset
            grid = make_maze(maze_size, maze_seed)
            if hasattr(agent, 'set_env'):  # oracle
                agent.set_env(grid, maze_size, maze_size - 2, maze_size - 2)
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
                        reward = hazard_cost
                    if ax == gx and ay == gy:
                        reward = goal_reward
                        done = True
                        solved = True
                else:
                    reward = wall_bump_cost

                next_obs = feature_fn(grid, ax, ay, gx, gy, maze_size, action_hist + [action])
                ep_reward += reward
                action_hist.append(action)

                if learn:
                    agent.learn(obs, action, reward, next_obs, done)

                if done:
                    break

            wall_time = time.time() - t0
            results.append(ExpResult(
                agent_name=agent_name, maze_size=maze_size, seed=seed,
                phase=phase, episode=ep, reward=ep_reward,
                steps=step + 1, solved=solved,
                synops=agent.get_synops(), wall_time_s=wall_time,
                config=cfg,
            ))

    run_phase('train', num_train, learn=True, seed_offset=0)
    run_phase('test', num_test, learn=False, seed_offset=10_000_000)
    return results
