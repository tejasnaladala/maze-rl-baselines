"""Tier 4b: Memory-augmented agents (DRQN) to test partial-observability hypothesis.

One explanation for the "Random beats DQN" finding is that the 24-dim ego features
create state aliasing (two different global states producing the same feature vector),
turning the MDP into a POMDP. A recurrent agent should be able to use history to
disambiguate. If DRQN still loses to Random, partial observability is NOT the cause.

Agents: DRQN (LSTM-augmented DQN)
Sizes: 9, 13, 21 (3 sizes for budget)
Seeds: 20
Total: 60 runs

Each DRQN run is ~2x slower than MLP_DQN due to LSTM + sequence replay.
"""

import sys, json, random, time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    OBS_DIM, NUM_ACTIONS, ACTIONS, WALL, HAZARD,
    make_maze, ego_features,
    ExpResult, load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9]  # time-constrained: 9x9 only (20 runs at ~141s each = ~47 min)
NUM_TRAIN = 100
NUM_TEST = 50

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_memory_agents'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


class DRQNNetwork(nn.Module):
    """DQN with LSTM memory layer."""
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64, lstm_hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden, num_actions)
        self.lstm_hidden = lstm_hidden

    def forward(self, x: torch.Tensor, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, seq = x.shape[0], x.shape[1]
        out = torch.relu(self.fc1(x))
        if hidden is None:
            h0 = torch.zeros(1, batch, self.lstm_hidden, device=x.device)
            c0 = torch.zeros(1, batch, self.lstm_hidden, device=x.device)
            hidden = (h0, c0)
        lstm_out, hidden_out = self.lstm(out, hidden)
        q = self.fc_out(lstm_out[:, -1, :])
        return q, hidden_out


class DRQNAgent:
    """DRQN with sequence replay and stored hidden states per sequence."""
    def __init__(self, hidden=64, lstm_hidden=64, lr=5e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000, seq_len=8,
                 buffer_size=5000, batch_size=16, target_update=300, device='auto'):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device if device != 'auto' else 'cpu')
        self.net = DRQNNetwork(OBS_DIM, NUM_ACTIONS, hidden, lstm_hidden).to(self.device)
        self.target = DRQNNetwork(OBS_DIM, NUM_ACTIONS, hidden, lstm_hidden).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        self.replay: deque = deque(maxlen=buffer_size)
        self.obs_history: list = []
        self.hidden = None

    def act(self, obs, step):
        self.steps += 1
        self.obs_history.append(list(obs))
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.steps / self.eps_decay)
        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q, self.hidden = self.net(obs_t, self.hidden)
            return int(q.argmax(dim=1).item())

    def eval_action(self, obs):
        self.obs_history.append(list(obs))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q, self.hidden = self.net(obs_t, self.hidden)
            return int(q.argmax(dim=1).item())

    def learn(self, obs, action, reward, next_obs, done):
        seq_start = max(0, len(self.obs_history) - self.seq_len)
        obs_seq = self.obs_history[seq_start:]
        while len(obs_seq) < self.seq_len:
            obs_seq.insert(0, [0.0] * OBS_DIM)

        next_history = self.obs_history + [list(next_obs)]
        next_start = max(0, len(next_history) - self.seq_len)
        next_seq = next_history[next_start:]
        while len(next_seq) < self.seq_len:
            next_seq.insert(0, [0.0] * OBS_DIM)

        self.replay.append((obs_seq, action, reward, next_seq, float(done)))

        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)
        obs_seqs, actions, rewards, next_seqs, dones = zip(*batch)

        obs_t = torch.FloatTensor(np.array(obs_seqs)).to(self.device)
        next_t = torch.FloatTensor(np.array(next_seqs)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q, _ = self.net(obs_t)
        q_selected = q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q, _ = self.target(next_t)
            target = rewards_t + self.gamma * next_q.max(dim=1).values * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_selected, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())

    def reset_for_new_maze(self):
        self.obs_history = []
        self.hidden = None

    def get_synops(self):
        return 0


def run_drqn_experiment(agent, agent_name: str, maze_size: int, seed: int):
    max_steps = max(300, 4 * maze_size * maze_size)
    # FIX Phase 1 audit: write full config to each result record for data lineage
    run_cfg = {
        'num_train': NUM_TRAIN, 'num_test': NUM_TEST, 'max_steps': max_steps,
        'reward_shaping': True, 'visit_penalty': True,
        'wall_bump_cost': -0.3, 'hazard_cost': -1.0, 'goal_reward': 10.0,
        'lstm_hidden': 64, 'seq_len': 8,
        'code_hash': code_hash(),
    }
    results: list[dict] = []
    rng = random.Random(seed)

    def run_phase(phase, num_eps, learn, seed_offset):
        for ep in range(num_eps):
            t0 = time.time()
            maze_seed = rng.randint(0, 10_000_000) + seed_offset
            grid = make_maze(maze_size, maze_seed)
            agent.reset_for_new_maze()
            ax, ay = 1, 1
            gx, gy = maze_size - 2, maze_size - 2
            action_hist: list[int] = []
            visited = {(1, 1)}
            ep_reward = 0.0
            solved = False

            for step in range(max_steps):
                obs = ego_features(grid, ax, ay, gx, gy, maze_size, action_hist)
                if not learn:
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
                    new_dist = abs(ax - gx) + abs(ay - gy)
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
                if learn:
                    agent.learn(obs, action, reward, next_obs, done)
                if done: break

            wall = time.time() - t0
            results.append({
                'agent_name': agent_name, 'maze_size': maze_size, 'seed': seed,
                'phase': phase, 'episode': ep, 'reward': ep_reward,
                'steps': step + 1, 'solved': solved, 'synops': 0,
                'wall_time_s': wall, 'lib_version': 'v2',
                'config': run_cfg,
            })

    run_phase('train', NUM_TRAIN, learn=True, seed_offset=0)
    run_phase('test', NUM_TEST, learn=False, seed_offset=10_000_000)
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(MAZE_SIZES)
    done_count = len(completed)

    print(f"\nTier 4b: DRQN memory agent — {total_runs} runs, {done_count} done")
    print(f"Code hash: {code_hash()}\n")

    agent_name = 'DRQN'
    for maze_size in MAZE_SIZES:
        for seed in SEEDS:
            key = run_key(agent_name, maze_size, seed)
            if key in completed:
                continue

            print(f"  [{done_count}/{total_runs}] {agent_name} {maze_size}x{maze_size} s={seed}...",
                  end=" ", flush=True)
            t0 = time.time()

            # Phase 1 audit fix: deterministic=True for bit-reproducibility.
            # Previously False for throughput; LSTM kernels are slow under
            # cudnn.deterministic, but the difference is ~10% wall-time
            # and we gain reviewer-defensible reproducibility.
            set_all_seeds(seed, deterministic=True)
            agent = DRQNAgent(device=DEVICE, eps_decay=NUM_TRAIN * 200)
            results = run_drqn_experiment(agent, agent_name, maze_size, seed)

            run_file = OUT_DIR / f'{agent_name}_{maze_size}_{seed}.json'
            atomic_save(results, run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done_count += 1

            test = [r for r in results if r['phase'] == 'test']
            test_success = sum(r['solved'] for r in test) / len(test) * 100
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nTier 4b DRQN complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
