"""INFORMATION PARITY AUDIT (per Codex review):

Wall-following is currently 100% because it sees the FULL maze grid.
This is unfair vs neural agents which see only ego-features (24-dim).

This launcher restricts WallFollower/DFS to ONLY the ego_features
representation that neural agents use. If the heuristic STILL works,
the gap to neural truly is about learning, not information.

If the heuristic FAILS under the restricted observation, we honestly
report 'wall-following needs full grid; neural agents have less info'.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze, is_solvable, ego_features, ACTIONS, WALL,
    load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)
from maze_env_helpers import step_env

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 11, 13, 17, 21]
NUM_TEST_EPS = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_wall_follow_egofeats"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


# Ego_features layout (24-dim): the first 9 cells are the 3x3 local view (8 walls + center).
# The 4 walls around the agent are at indices: NW, N, NE, W, *, E, SW, S, SE -> [0..8]
# We need: north neighbor (index 1), east (5), south (7), west (3).
# In ego_features each value is grid[ny][nx]/3.0 with WALL=1 -> 1/3.

WALL_NORM = 1.0 / 3.0


def is_open_via_ego(obs: np.ndarray, action: int) -> bool:
    """Check if the cell in the 'action' direction is open per ego_features.
    ACTIONS = [(0,-1), (1,0), (0,1), (-1,0)] = [up, right, down, left]
    Ego 3x3 has indices: [0=NW, 1=N, 2=NE, 3=W, 4=center, 5=E, 6=SW, 7=S, 8=SE]
    """
    # Map action index to ego-features index
    ego_idx = {0: 1, 1: 5, 2: 7, 3: 3}[action]
    return abs(obs[ego_idx] - WALL_NORM) > 0.05  # not a wall


class EgoOnlyWallFollower:
    """Wall-following using ONLY ego_features observation, no map access."""
    def __init__(self, side: str = "left"):
        self.side = side
        self.facing = 1  # default: facing right (action=1 in lib's ACTIONS)

    def reset(self):
        self.facing = 1

    def act(self, obs):
        # left-hand rule via ACTIONS = [(0,-1),(1,0),(0,1),(-1,0)] order.
        # If facing 0 (up): left-side is action 3 (left), right-side is action 1 (right)
        TURN_TABLE = {
            0: (3, 0, 1, 2),  # facing up: left/forward/right/back
            1: (0, 1, 2, 3),  # facing right
            2: (1, 2, 3, 0),  # facing down
            3: (2, 3, 0, 1),  # facing left
        }
        left, fwd, right, back = TURN_TABLE[self.facing]
        if self.side == "left":
            options = [left, fwd, right, back]
        else:
            options = [right, fwd, left, back]
        for a in options:
            if is_open_via_ego(obs, a):
                self.facing = a
                return a
        return back


class EgoOnlyDFS:
    """DFS that only sees ego_features. NO map memory beyond what
    a neural net could maintain. Uses an internal map by tracking
    (action_history) coords -- this is the kind of memory an LSTM agent
    would need to learn implicitly.
    """
    def __init__(self):
        self.visited: set = set()
        self.path: list = []
        self.pos: tuple = (0, 0)

    def reset(self):
        self.visited = set([(0, 0)])
        self.path = []
        self.pos = (0, 0)

    def act(self, obs):
        # Try unvisited neighbors first via ego
        for a in (0, 1, 2, 3):  # up, right, down, left
            if not is_open_via_ego(obs, a):
                continue
            dx, dy = ACTIONS[a]
            nb = (self.pos[0] + dx, self.pos[1] + dy)
            if nb not in self.visited:
                self.path.append((self.pos, a))
                self.pos = nb
                self.visited.add(nb)
                return a
        # Backtrack
        if self.path:
            prev_pos, last_action = self.path.pop()
            opp = {0: 2, 2: 0, 1: 3, 3: 1}[last_action]
            self.pos = prev_pos
            return opp
        # Fallback: random open direction
        for a in (0, 1, 2, 3):
            if is_open_via_ego(obs, a):
                return a
        return 0


def run_one(agent_class, agent_kwargs: dict, maze_size: int,
            seed: int, n_test: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < n_test:
        s = int(rng.integers(0, 10**9))
        if is_solvable(make_maze(maze_size, seed=s), maze_size):
            test_seeds.append(s)

    for s in test_seeds:
        maze = make_maze(maze_size, seed=s)
        ax, ay = 1, 1
        gx, gy = maze_size - 2, maze_size - 2
        agent = agent_class(**agent_kwargs)
        agent.reset()
        action_hist: list = []
        max_steps = 4 * maze_size * maze_size
        step = 0
        for step in range(max_steps):
            obs = np.array(ego_features(maze, ax, ay, gx, gy, maze_size, action_hist),
                           dtype=np.float32)
            action = agent.act(obs)
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, maze_size)
            ax, ay = new_ax, new_ay
            action_hist.append(action)
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1
    return {"n_eps": n_test, "solved": solved,
            "success_rate": solved / n_test,
            "mean_steps": total_steps / n_test}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = [
        ("EgoWallFollowerLeft",  EgoOnlyWallFollower, {"side": "left"}),
        ("EgoWallFollowerRight", EgoOnlyWallFollower, {"side": "right"}),
        ("EgoDFSAgent",          EgoOnlyDFS,          {}),
    ]
    total = len(AGENTS) * len(MAZE_SIZES) * len(SEEDS)
    done = len(completed)
    print(f"\nEgo-only Wall/DFS (information parity audit): {total} runs, {done} done")
    print(f"Code hash: {code_hash()}\n")

    for agent_name, agent_cls, kwargs in AGENTS:
        for size in MAZE_SIZES:
            for seed in SEEDS:
                key = run_key(agent_name, size, seed)
                if key in completed:
                    continue
                print(f"  [{done}/{total}] {agent_name} {size}x{size} s={seed}...",
                      end=" ", flush=True)
                t0 = time.time()
                result = run_one(agent_cls, kwargs, size, seed, NUM_TEST_EPS)
                elapsed = time.time() - t0
                run_file = OUT_DIR / f"{agent_name}_{size}_{seed}.json"
                atomic_save([{
                    "agent_name": agent_name,
                    "maze_size": size,
                    "seed": seed,
                    "phase": "test",
                    "wall_time_s": elapsed,
                    "code_hash": code_hash(),
                    **result,
                }], run_file)
                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done += 1
                print(f"done ({elapsed:.1f}s) test={100*result['success_rate']:.0f}%")

    print(f"\nEgo-only wall-following complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
