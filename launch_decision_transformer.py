"""Tier B.1 - Decision Transformer baseline.

Modern offline-RL alternative to DQN. Defends against the 'but transformers'
reviewer critique. Uses BFS-Oracle trajectories as the offline dataset.

Decision Transformer (Chen et al. 2021, NeurIPS) reformulates RL as
sequence modeling on (return-to-go, state, action) tuples.

Setup:
- Collect ~100 BFS-optimal trajectories from training mazes (per seed)
- Train Decision Transformer with hidden=128, n_layers=3, n_heads=4
- Test on unseen mazes via return-conditioned generation
- 1 agent x 20 seeds x 9x9 = 20 runs
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
    BFSOracleAgent,
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
NUM_TRAIN_TRAJS = 100
NUM_TEST_EPS = 50
MAX_LEN = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_decision_transformer"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128,
                 n_layers: int = 3, n_heads: int = 4, max_len: int = 50):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.max_len = max_len

        self.state_emb = nn.Linear(state_dim, hidden)
        self.action_emb = nn.Embedding(n_actions, hidden)
        self.return_emb = nn.Linear(1, hidden)
        self.pos_emb = nn.Embedding(max_len * 3, hidden)
        self.ln = nn.LayerNorm(hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(hidden, n_actions)

    def forward(self, returns_to_go: torch.Tensor, states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        B, T = returns_to_go.shape[:2]

        r_emb = self.return_emb(returns_to_go)
        s_emb = self.state_emb(states)
        a_emb = self.action_emb(actions)

        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2)
        seq = stacked.reshape(B, T * 3, self.hidden)

        positions = torch.arange(T * 3, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.pos_emb(positions)
        seq = self.ln(seq)

        mask = torch.triu(
            torch.ones(T * 3, T * 3, device=seq.device, dtype=torch.bool), diagonal=1
        )
        out = self.transformer(seq, mask=mask)

        state_positions = out[:, 1::3]
        action_logits = self.action_head(state_positions)
        return action_logits


def collect_oracle_trajectories(maze_size: int, num_trajs: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    trajs: list[dict] = []
    attempts = 0
    while len(trajs) < num_trajs and attempts < num_trajs * 5:
        attempts += 1
        maze_seed = int(rng.integers(0, 10**9))
        maze = make_maze(maze_size, seed=maze_seed)
        if not is_solvable(maze):
            continue
        oracle = BFSOracleAgent()
        states, actions, rewards = [], [], []
        oracle.reset_for_new_maze(maze)
        pos = (1, 1)
        goal = (maze.shape[0] - 2, maze.shape[1] - 2)
        max_steps = 4 * maze_size * maze_size
        for step in range(max_steps):
            obs = ego_features(maze, pos, goal)
            states.append(obs.copy())
            action = oracle.act(obs, step)
            actions.append(action)
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < maze.shape[0] and 0 <= new_pos[1] < maze.shape[1]
                    and maze[new_pos] != 1):
                pos = new_pos
            if pos == goal:
                rewards.append(10.0)
                break
            else:
                rewards.append(-0.04)
        if pos == goal:
            trajs.append({"states": states, "actions": actions, "rewards": rewards})
    return trajs


def train_dt(model: DecisionTransformer, trajs: list[dict], epochs: int = 30,
             batch_size: int = 32, lr: float = 1e-4) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    sequences = []
    for tr in trajs:
        T = len(tr["states"])
        if T < 2:
            continue
        states = np.array(tr["states"], dtype=np.float32)
        actions = np.array(tr["actions"], dtype=np.int64)
        rewards = np.array(tr["rewards"], dtype=np.float32)
        rtg = np.array([rewards[i:].sum() for i in range(T)], dtype=np.float32)
        sequences.append((states, actions, rtg))

    if not sequences:
        return

    for epoch in range(epochs):
        np.random.shuffle(sequences)
        for batch_start in range(0, len(sequences), batch_size):
            batch = sequences[batch_start:batch_start + batch_size]
            max_T = min(MAX_LEN, max(len(s[0]) for s in batch))

            B = len(batch)
            S = np.zeros((B, max_T, OBS_DIM), dtype=np.float32)
            A = np.zeros((B, max_T), dtype=np.int64)
            R = np.zeros((B, max_T, 1), dtype=np.float32)
            mask = np.zeros((B, max_T), dtype=np.float32)
            for i, (s, a, r) in enumerate(batch):
                T = min(len(s), max_T)
                S[i, :T] = s[:T]
                A[i, :T] = a[:T]
                R[i, :T, 0] = r[:T]
                mask[i, :T] = 1.0

            S_t = torch.from_numpy(S).to(DEVICE)
            A_t = torch.from_numpy(A).to(DEVICE)
            R_t = torch.from_numpy(R).to(DEVICE)
            mask_t = torch.from_numpy(mask).to(DEVICE)

            logits = model(R_t, S_t, A_t)
            loss = F.cross_entropy(
                logits.reshape(-1, NUM_ACTIONS), A_t.reshape(-1), reduction="none"
            )
            loss = (loss * mask_t.reshape(-1)).sum() / mask_t.sum().clamp(min=1.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def test_dt(model: DecisionTransformer, maze_size: int, n_test: int,
            seed: int, target_return: float = 9.0) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    model.eval()
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < n_test:
        s = int(rng.integers(0, 10**9))
        maze = make_maze(maze_size, seed=s)
        if is_solvable(maze):
            test_seeds.append(s)

    for s in test_seeds:
        maze = make_maze(maze_size, seed=s)
        pos = (1, 1)
        goal = (maze.shape[0] - 2, maze.shape[1] - 2)
        max_steps = 4 * maze_size * maze_size

        states_hist: list = []
        actions_hist: list = []
        rtg_hist: list = [target_return]

        step = 0
        for step in range(max_steps):
            obs = ego_features(maze, pos, goal)
            states_hist.append(obs)

            T = len(states_hist)
            if T > MAX_LEN:
                states_hist = states_hist[-MAX_LEN:]
                actions_hist = actions_hist[-(MAX_LEN - 1):]
                rtg_hist = rtg_hist[-MAX_LEN:]
                T = MAX_LEN

            S = np.array(states_hist, dtype=np.float32)
            A = np.array(actions_hist + [0], dtype=np.int64)
            R = np.array(rtg_hist, dtype=np.float32).reshape(-1, 1)

            S_t = torch.from_numpy(S).unsqueeze(0).to(DEVICE)
            A_t = torch.from_numpy(A).unsqueeze(0).to(DEVICE)
            R_t = torch.from_numpy(R).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(R_t, S_t, A_t)
            action = int(logits[0, -1].argmax().item())
            actions_hist.append(action)

            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < maze.shape[0] and 0 <= new_pos[1] < maze.shape[1]
                    and maze[new_pos] != 1):
                pos = new_pos

            if pos == goal:
                solved += 1
                rtg_hist.append(rtg_hist[-1] - 10.0)
                break
            else:
                rtg_hist.append(rtg_hist[-1] + 0.04)

        total_steps += step + 1

    return {
        "n_eps": n_test,
        "solved": solved,
        "success_rate": solved / n_test,
        "mean_steps": total_steps / n_test,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    total = len(SEEDS)
    done = len(completed)
    print(f"\nDecision Transformer: {total} runs, {done} done")
    print(f"Code hash: {code_hash()}\n")

    for seed in SEEDS:
        key = run_key("DecisionTransformer", MAZE_SIZE, seed)
        if key in completed:
            continue

        print(f"  [{done}/{total}] DT s={seed}...", end=" ", flush=True)
        t0 = time.time()

        set_all_seeds(seed, deterministic=False)
        trajs = collect_oracle_trajectories(MAZE_SIZE, NUM_TRAIN_TRAJS, seed)
        if not trajs:
            print("no oracle trajectories collected, skipping")
            continue

        model = DecisionTransformer(
            state_dim=OBS_DIM, n_actions=NUM_ACTIONS,
            hidden=128, n_layers=3, n_heads=4, max_len=MAX_LEN
        ).to(DEVICE)

        train_dt(model, trajs, epochs=30, batch_size=32, lr=1e-4)
        result = test_dt(model, MAZE_SIZE, NUM_TEST_EPS, seed)

        elapsed = time.time() - t0
        run_file = OUT_DIR / f"DecisionTransformer_{MAZE_SIZE}_{seed}.json"
        atomic_save([{
            "agent_name": "DecisionTransformer",
            "maze_size": MAZE_SIZE,
            "seed": seed,
            "phase": "test",
            "wall_time_s": elapsed,
            "code_hash": code_hash(),
            "n_train_trajs": len(trajs),
            **result,
        }], run_file)

        completed.add(key)
        save_checkpoint(CHECKPOINT_FILE, completed)
        done += 1
        print(f"done ({elapsed:.0f}s) test={100*result['success_rate']:.0f}%")

    print(f"\nDecision Transformer complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
