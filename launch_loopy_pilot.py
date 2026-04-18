"""Loopy-maze pilot — research validity check.

5 agents x 5 seeds x 9x9 mazes, 50 test eps each. Tests whether the headline
(heuristic strong / neural RL weak) survives when the spanning-tree property
is broken via Wilson + add_loops.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from experiment_lib_v2 import (
    BFSOracleAgent, MLPDQNAgent, NoBacktrackRandomAgent, RandomAgent,
    WALL, ACTIONS, ego_features, set_all_seeds, code_hash, atomic_save,
)
import experiment_lib_v2 as lib
from launch_wall_follow_egofeats import EgoOnlyWallFollower
from maze_env_helpers import step_env
from loopy_maze import make_wilson_maze, add_loops

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456, 789, 1024]
MAZE_SIZE = 9
NUM_TEST = 50
NUM_TRAIN_DQN = 100
N_EXTRA = MAZE_SIZE // 2  # floor(9*0.5) = 4
OUT_DIR = Path(__file__).parent / "raw_results" / "exp_loopy_pilot"


def loopy_maze(size: int, maze_seed: int) -> list:
    return add_loops(make_wilson_maze(size, maze_seed), size, maze_seed ^ 0x5A5A, N_EXTRA)


def is_solvable_open(maze, size: int) -> bool:
    from collections import deque
    visited, q = {(1, 1)}, deque([(1, 1)])
    gx, gy = size - 2, size - 2
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return True
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited and maze[ny][nx] != WALL:
                visited.add((nx, ny)); q.append((nx, ny))
    return False


def solvable_seeds(size: int, seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(seed + 1_000_000)
    out: list[int] = []
    while len(out) < n:
        s = int(rng.integers(0, 10**9))
        if is_solvable_open(loopy_maze(size, s), size):
            out.append(s)
    return out


def trace_episode(maze, agent, kind: str, size: int) -> tuple[bool, int]:
    if hasattr(agent, "reset"): agent.reset()
    if hasattr(agent, "reset_for_new_maze"): agent.reset_for_new_maze()
    gx, gy = size - 2, size - 2
    if hasattr(agent, "set_env"): agent.set_env(maze, size, gx, gy)
    ax, ay = 1, 1
    action_hist: list[int] = []
    max_steps = 4 * size * size
    for step in range(max_steps):
        obs = np.array(ego_features(maze, ax, ay, gx, gy, size, action_hist), dtype=np.float32)
        if kind == "ego":
            action = agent.act(obs)
        elif hasattr(agent, "eval_action"):
            action = agent.eval_action(obs.tolist())
        else:
            action = agent.act(obs.tolist(), step)
        new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, size)
        ax, ay = new_ax, new_ay
        action_hist.append(action)
        if (ax, ay) == (gx, gy):
            return True, step + 1
    return False, max_steps


def run_inference(name: str, factory, kind: str, size: int, seed: int, n_test: int) -> dict:
    test_seeds = solvable_seeds(size, seed, n_test)
    solved, total_steps = 0, 0
    for s in test_seeds:
        agent = factory()
        if hasattr(agent, "seed"): agent.seed(seed)
        ok, steps = trace_episode(loopy_maze(size, s), agent, kind, size)
        if ok: solved += 1
        total_steps += steps
    return {"n_eps": n_test, "solved": solved,
            "success_rate": solved / n_test, "mean_steps": total_steps / n_test}


def run_dqn(size: int, seed: int, n_train: int, n_test: int) -> dict:
    set_all_seeds(seed)
    agent = MLPDQNAgent(hidden=64, device=DEVICE)
    original = lib.make_maze
    lib.make_maze = lambda s_size, s_seed: loopy_maze(s_size, s_seed)
    try:
        results = lib.run_experiment(
            agent=agent, agent_name="MLP_DQN_h64", maze_size=size,
            num_train=n_train, num_test=n_test, seed=seed,
            reward_shaping=True, visit_penalty=True,
        )
    finally:
        lib.make_maze = original
    test = [r for r in results if r.phase == "test"]
    n = len(test)
    if n == 0:
        return {"n_eps": 0, "solved": 0, "success_rate": 0.0, "mean_steps": 0.0}
    solved = sum(1 for r in test if r.solved)
    return {"n_eps": n, "solved": solved,
            "success_rate": solved / n, "mean_steps": sum(r.steps for r in test) / n}


def save_run(agent_name: str, seed: int, result: dict, elapsed: float, extra: dict | None = None) -> None:
    base = {
        "agent_name": agent_name, "maze_size": MAZE_SIZE, "seed": seed, "phase": "test",
        "extra_passages": N_EXTRA, "wall_time_s": elapsed, "code_hash": code_hash(),
    }
    if extra: base.update(extra)
    base.update(result)
    atomic_save([base], OUT_DIR / f"{agent_name}_{MAZE_SIZE}_{seed}.json")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nLoopy pilot: {len(SEEDS)} seeds x {MAZE_SIZE}x{MAZE_SIZE} (extra_passages={N_EXTRA})")
    print(f"Code hash: {code_hash()} | Device: {DEVICE}\n")

    INFERENCE = [
        ("BFSOracle",           lambda: BFSOracleAgent(avoid_hazards=False), "neural"),
        ("EgoWallFollowerLeft", lambda: EgoOnlyWallFollower(side="left"),    "ego"),
        ("NoBackRandom",        lambda: NoBacktrackRandomAgent(),            "neural"),
        ("Random",              lambda: RandomAgent(),                       "neural"),
    ]

    for name, factory, kind in INFERENCE:
        for seed in SEEDS:
            t0 = time.time()
            res = run_inference(name, factory, kind, MAZE_SIZE, seed, NUM_TEST)
            elapsed = time.time() - t0
            save_run(name, seed, res, elapsed)
            print(f"  {name:>22s} s={seed:>5d}  test={100*res['success_rate']:>5.1f}%  "
                  f"steps={res['mean_steps']:>6.1f}  ({elapsed:.1f}s)")

    print()
    for seed in SEEDS:
        t0 = time.time()
        res = run_dqn(MAZE_SIZE, seed, NUM_TRAIN_DQN, NUM_TEST)
        elapsed = time.time() - t0
        save_run("MLP_DQN_h64", seed, res, elapsed, {"num_train": NUM_TRAIN_DQN})
        print(f"  {'MLP_DQN_h64':>22s} s={seed:>5d}  test={100*res['success_rate']:>5.1f}%  "
              f"steps={res['mean_steps']:>6.1f}  ({elapsed:.1f}s)")

    print("\n=== LOOPY PILOT SUMMARY (mean +/- sd over 5 seeds) ===")
    print(f"{'agent':<22s}  {'success%':>14s}  {'mean_steps':>14s}")
    print("-" * 56)
    summary_rows = []
    for name in [a[0] for a in INFERENCE] + ["MLP_DQN_h64"]:
        rates, steps = [], []
        for seed in SEEDS:
            with open(OUT_DIR / f"{name}_{MAZE_SIZE}_{seed}.json") as f:
                row = json.load(f)[0]
            rates.append(row["success_rate"]); steps.append(row["mean_steps"])
        ra, sa = np.array(rates), np.array(steps)
        summary_rows.append({"agent": name, "success_rate_mean": float(ra.mean()),
                             "success_rate_sd": float(ra.std()),
                             "mean_steps_mean": float(sa.mean()),
                             "mean_steps_sd": float(sa.std())})
        print(f"{name:<22s}  {100*ra.mean():>5.1f} +/- {100*ra.std():>4.1f}  "
              f"{sa.mean():>7.1f} +/- {sa.std():>4.1f}")
    atomic_save(summary_rows, OUT_DIR / "summary.json")
    print(f"\nResults in {OUT_DIR}")


if __name__ == "__main__":
    main()
