"""Wall-following + DFS oracle baselines (per Codex review):

These are SIMPLE deterministic exploration heuristics that should be
included as oracle decomposition baselines:

  - WallFollowerLeft: classic 'left-hand rule' — always try to keep the
    wall on the left. Provably solves any simply-connected maze given enough time.
  - WallFollowerRight: mirror of above.
  - DFSAgent: depth-first search with memoization (memory-explicit).

These are NOT trained — they're hand-coded baselines that complement
NoBackRandom and Random as the structural-prior reference set.

If wall-following beats NoBackRandom, the headline becomes 'simple
deterministic priors win', not 'non-backtracking specifically'.
If NoBackRandom beats wall-following, our specific claim survives.

3 agents x 20 seeds x 5 sizes = 300 runs. Fast (no GPU needed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze,
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

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 11, 13, 17, 21]
NUM_TEST_EPS = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_wall_following"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"

# Action encoding: 0=up, 1=down, 2=left, 3=right
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# For a given facing direction, give (left_dir, forward_dir, right_dir, back_dir)
# Index by facing direction (action that we just took).
TURN_TABLE = {
    0: (2, 0, 3, 1),  # facing up -> left/up/right/down
    1: (3, 1, 2, 0),  # facing down
    2: (1, 2, 0, 3),  # facing left
    3: (0, 3, 1, 2),  # facing right
}


def is_open(maze, pos: tuple, action: int) -> bool:
    dr, dc = DELTAS[action]
    new_r, new_c = pos[0] + dr, pos[1] + dc
    n = len(maze)
    if not (0 <= new_r < n and 0 <= new_c < n):
        return False
    return maze[new_r][new_c] != 1


class WallFollower:
    def __init__(self, side: str = "left"):
        self.side = side
        self.facing = 3  # default: facing right

    def reset(self):
        self.facing = 3

    def act(self, maze, pos: tuple) -> int:
        """Try (left, forward, right, back) order for left-hand rule.
        Try (right, forward, left, back) for right-hand rule.
        """
        left, fwd, right, back = TURN_TABLE[self.facing]
        if self.side == "left":
            options = [left, fwd, right, back]
        else:
            options = [right, fwd, left, back]
        for a in options:
            if is_open(maze, pos, a):
                self.facing = a
                return a
        return back


class DFSAgent:
    """Depth-first search with explicit memory: visit unvisited neighbors,
    backtrack when no unvisited neighbor exists.
    """

    def __init__(self):
        self.visited: set = set()
        self.path: list = []  # Stack of (pos, action_taken_to_arrive)

    def reset(self):
        self.visited = set()
        self.path = []

    def act(self, maze, pos: tuple) -> int:
        self.visited.add(pos)
        # Try unvisited neighbors first (in NESW order)
        for a in (0, 3, 1, 2):
            if is_open(maze, pos, a):
                dr, dc = DELTAS[a]
                neighbor = (pos[0] + dr, pos[1] + dc)
                if neighbor not in self.visited:
                    self.path.append((pos, a))
                    return a
        # No unvisited neighbor: backtrack
        if self.path:
            prev_pos, last_action = self.path.pop()
            # Reverse the action
            opp = {0: 1, 1: 0, 2: 3, 3: 2}[last_action]
            return opp
        # Fallback: random
        for a in (0, 1, 2, 3):
            if is_open(maze, pos, a):
                return a
        return 0


def run_one(agent_class, agent_kwargs: dict, maze_size: int, seed: int,
            n_test: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_000)
    solved = 0
    total_steps = 0
    test_seeds: list[int] = []
    while len(test_seeds) < n_test:
        s = int(rng.integers(0, 10**9))
        m = make_maze(maze_size, seed=s)
        if is_solvable(m, maze_size):
            test_seeds.append(s)

    for s in test_seeds:
        maze = make_maze(maze_size, seed=s)
        n = len(maze)
        pos = (1, 1)
        goal = (n - 2, n - 2)
        max_steps = 4 * maze_size * maze_size
        agent = agent_class(**agent_kwargs)
        agent.reset()
        step = 0
        for step in range(max_steps):
            action = agent.act(maze, pos)
            dr, dc = DELTAS[action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < n and 0 <= new_pos[1] < n
                    and maze[new_pos[0]][new_pos[1]] != 1):
                pos = new_pos
            if pos == goal:
                solved += 1
                break
        total_steps += step + 1

    return {"n_eps": n_test, "solved": solved, "success_rate": solved / n_test,
            "mean_steps": total_steps / n_test}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = [
        ("WallFollowerLeft", WallFollower, {"side": "left"}),
        ("WallFollowerRight", WallFollower, {"side": "right"}),
        ("DFSAgent", DFSAgent, {}),
    ]
    total = len(AGENTS) * len(MAZE_SIZES) * len(SEEDS)
    done = len(completed)
    print(f"\nWall-following + DFS: {total} runs, {done} done")
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

    print(f"\nWall-following + DFS complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
