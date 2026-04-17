"""KILLER EXPERIMENT (per Codex adversarial review):

  Policy Distillation from MULTIPLE teachers into neural networks.

Distill from 3 different teachers and test on unseen mazes:
  - BFSOracle (perfect knowledge): if MLP can't match, real function
    approx failure proven.
  - NoBackRandom (exploration prior, 52.2% baseline): if MLP can't match,
    exploration is not the only issue.
  - FeatureQ_v2 (tabular learner, 35.3%): tests whether tabular
    decisions transfer to neural.

Either outcome is a NeurIPS-grade finding:
  - If all 3 distilled MLPs match their teachers -> failure is EXPLORATION
  - If distilled MLP < teacher -> neural function approx genuinely fails
  - Mixed results -> teacher-specific failure modes

Setup:
  - Roll out teacher for ~100-500 successful episodes per seed
  - Train MLP (h64) and LSTM (h64) on demos via cross-entropy
  - Test on 50 unseen mazes per seed
  - 3 teachers x 2 architectures x 20 seeds x 9x9 = 120 runs
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
    NoBacktrackRandomAgent,
    BFSOracleAgent,
    FeatureQAgent,
    RandomAgent,
    run_experiment,
    OBS_DIM,
    NUM_ACTIONS,
    ACTIONS,
    load_checkpoint,
    save_checkpoint,
    run_key,
    atomic_save,
    set_all_seeds,
    code_hash,
)
from maze_env_helpers import get_obs, step_env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
NUM_DEMO_EPISODES = 500   # NoBack rollouts as supervised demonstrations
NUM_TEST_EPS = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_policy_distillation"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, seq_len: int = 8):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_actions)
        self.hidden_dim = hidden
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, hx=None) -> tuple[torch.Tensor, tuple]:
        # x: (B, T, obs_dim)
        z = self.encoder(x).relu()
        out, hx_new = self.lstm(z, hx)
        return self.head(out), hx_new


def make_teacher(name: str):
    if name == "BFSOracle": return BFSOracleAgent()
    if name == "NoBackRandom": return NoBacktrackRandomAgent()
    if name == "FeatureQ_v2": return FeatureQAgent()
    raise ValueError(name)


def collect_demos(teacher_name: str, num_episodes: int, maze_size: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll out teacher and return (states, actions) tensors. For learning
    teachers (FeatureQ), train them first via a separate run_experiment call."""
    rng = np.random.default_rng(seed)

    # For learning teachers, pre-train on a few episodes to get a good policy
    if teacher_name == "FeatureQ_v2":
        teacher = FeatureQAgent()
        # Quick training run to bring the teacher up to speed
        _ = run_experiment(teacher, "teacher_pretrain", maze_size, 100, 0, seed)
    else:
        teacher = make_teacher(teacher_name)

    all_states = []
    all_actions = []
    collected = 0
    attempts = 0
    while collected < num_episodes and attempts < num_episodes * 10:
        attempts += 1
        s = int(rng.integers(0, 10**9))
        maze = make_maze(maze_size, seed=s)
        if not is_solvable(maze, maze_size):
            continue
        if hasattr(teacher, "reset_for_new_maze"):
            teacher.reset_for_new_maze()
        ax, ay = 1, 1
        gx, gy = maze_size - 2, maze_size - 2
        max_steps = 4 * maze_size * maze_size
        ep_states = []
        ep_actions = []
        action_hist: list = []
        for step in range(max_steps):
            obs = get_obs(maze, ax, ay, gx, gy, maze_size, action_hist)
            ep_states.append(obs.copy())
            if hasattr(teacher, "eval_action"):
                action = teacher.eval_action(obs)
            else:
                action = teacher.act(obs, step)
            ep_actions.append(action)
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, maze_size)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                all_states.extend(ep_states)
                all_actions.extend(ep_actions)
                collected += 1
                break
    return (
        np.array(all_states, dtype=np.float32) if all_states else np.zeros((0, OBS_DIM), dtype=np.float32),
        np.array(all_actions, dtype=np.int64) if all_actions else np.zeros((0,), dtype=np.int64),
    )


def train_mlp(states: np.ndarray, actions: np.ndarray, epochs: int = 50,
              batch_size: int = 256, lr: float = 1e-3) -> MLPPolicy:
    model = MLPPolicy(OBS_DIM, NUM_ACTIONS, hidden=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    s_t = torch.from_numpy(states).to(DEVICE)
    a_t = torch.from_numpy(actions).to(DEVICE)
    n = len(states)
    if n == 0:
        return model
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            logits = model(s_t[b])
            loss = F.cross_entropy(logits, a_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_lstm(states: np.ndarray, actions: np.ndarray, epochs: int = 30,
               seq_len: int = 8, batch_size: int = 64, lr: float = 1e-3) -> LSTMPolicy:
    model = LSTMPolicy(OBS_DIM, NUM_ACTIONS, hidden=64, seq_len=seq_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(states)
    if n < seq_len + 1:
        return model
    # Build overlapping sequences
    s_t = torch.from_numpy(states).to(DEVICE)
    a_t = torch.from_numpy(actions).to(DEVICE)
    n_seq = n - seq_len
    for _ in range(epochs):
        idx = torch.randperm(n_seq, device=DEVICE)
        for i in range(0, n_seq, batch_size):
            b = idx[i:i + batch_size]
            seq_states = torch.stack([s_t[j:j + seq_len] for j in b.tolist()])
            seq_actions = torch.stack([a_t[j + 1:j + 1 + seq_len] for j in b.tolist()])
            logits, _ = model(seq_states)
            loss = F.cross_entropy(
                logits.reshape(-1, NUM_ACTIONS), seq_actions.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def test_mlp(model: MLPPolicy, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    model.train(False)
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < n_test:
        s = int(rng.integers(0, 10**9))
        if is_solvable(make_maze(MAZE_SIZE, seed=s), MAZE_SIZE):
            test_seeds.append(s)
    for s in test_seeds:
        maze = make_maze(MAZE_SIZE, seed=s)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        max_steps = 4 * MAZE_SIZE * MAZE_SIZE
        action_hist: list = []
        step = 0
        for step in range(max_steps):
            obs = get_obs(maze, ax, ay, gx, gy, MAZE_SIZE, action_hist)
            with torch.no_grad():
                logits = model(torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0))
                action = int(logits.argmax(dim=-1).item())
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, MAZE_SIZE)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1
    return {"n_eps": n_test, "solved": solved, "success_rate": solved / n_test,
            "mean_steps": total_steps / n_test}


def test_lstm(model: LSTMPolicy, n_test: int, seed: int, seq_len: int = 8) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    model.train(False)
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < n_test:
        s = int(rng.integers(0, 10**9))
        if is_solvable(make_maze(MAZE_SIZE, seed=s), MAZE_SIZE):
            test_seeds.append(s)
    for s in test_seeds:
        maze = make_maze(MAZE_SIZE, seed=s)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        max_steps = 4 * MAZE_SIZE * MAZE_SIZE
        history: list = []
        action_hist: list = []
        step = 0
        for step in range(max_steps):
            obs = get_obs(maze, ax, ay, gx, gy, MAZE_SIZE, action_hist)
            history.append(obs)
            if len(history) > seq_len:
                history = history[-seq_len:]
            seq = np.array(history, dtype=np.float32)
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(seq).to(DEVICE).unsqueeze(0))
                action = int(logits[0, -1].argmax(dim=-1).item())
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, MAZE_SIZE)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1
    return {"n_eps": n_test, "solved": solved, "success_rate": solved / n_test,
            "mean_steps": total_steps / n_test}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    TEACHERS = ["BFSOracle", "NoBackRandom", "FeatureQ_v2"]
    STUDENTS = ["MLP", "LSTM"]
    total = len(TEACHERS) * len(STUDENTS) * len(SEEDS)
    done = len(completed)
    print(f"\nPolicy Distillation: {total} runs, {done} done")
    print(f"Teachers: {TEACHERS}, Students: {STUDENTS}")
    print(f"Code hash: {code_hash()}\n")

    for teacher_name in TEACHERS:
        for student_name in STUDENTS:
            agent_name = f"Distilled{student_name}_from_{teacher_name}"
            for seed in SEEDS:
                key = run_key(agent_name, MAZE_SIZE, seed)
                if key in completed:
                    continue
                print(f"  [{done}/{total}] {agent_name} s={seed}...", end=" ", flush=True)
                t0 = time.time()

                set_all_seeds(seed, deterministic=False)

                # 1. Collect demos from teacher
                states, actions = collect_demos(teacher_name, NUM_DEMO_EPISODES,
                                                MAZE_SIZE, seed)
                n_demos = len(states)

                # 2. Train student
                if student_name == "MLP":
                    model = train_mlp(states, actions, epochs=50, batch_size=256, lr=1e-3)
                    result = test_mlp(model, NUM_TEST_EPS, seed)
                else:  # LSTM
                    model = train_lstm(states, actions, epochs=30, seq_len=8, batch_size=64, lr=1e-3)
                    result = test_lstm(model, NUM_TEST_EPS, seed, seq_len=8)

                elapsed = time.time() - t0
                run_file = OUT_DIR / f"{agent_name}_{MAZE_SIZE}_{seed}.json"
                atomic_save([{
                    "agent_name": agent_name,
                    "teacher": teacher_name,
                    "student": student_name,
                    "maze_size": MAZE_SIZE,
                    "seed": seed,
                    "phase": "test",
                    "wall_time_s": elapsed,
                    "code_hash": code_hash(),
                    "n_demonstrations": n_demos,
                    "n_demo_eps_target": NUM_DEMO_EPISODES,
                    **result,
                }], run_file)

                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done += 1
                print(f"done ({elapsed:.0f}s) demos={n_demos} test={100*result['success_rate']:.0f}%")

    print(f"\nPolicy distillation complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
