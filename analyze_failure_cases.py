"""Failure case visualization: cherry-pick mazes where neural agent fails
but a simple heuristic succeeds, and render side-by-side.

Output: paper_figures/fig_failure_cases.png + fig_failure_cases.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    make_maze, is_solvable, BFSOracleAgent, NoBacktrackRandomAgent,
    RandomAgent, MLPDQNAgent, run_experiment, ego_features, ACTIONS, WALL, HAZARD,
    OBS_DIM, NUM_ACTIONS,
)


ROOT = Path(__file__).parent
OUT_FIG = ROOT / "paper_figures" / "fig_failure_cases.png"
OUT_PDF = OUT_FIG.with_suffix(".pdf")
MAZE_SIZE = 9
MAX_STEPS = 4 * MAZE_SIZE * MAZE_SIZE


def trace_agent(maze, agent_name: str, agent, gx: int, gy: int, n: int) -> tuple:
    """Return (trajectory list of (ax, ay), solved, steps)."""
    if hasattr(agent, "reset_for_new_maze"):
        agent.reset_for_new_maze()
    ax, ay = 1, 1
    visited = [(ax, ay)]
    action_hist: list = []
    solved = False
    for step in range(MAX_STEPS):
        obs = np.array(ego_features(maze, ax, ay, gx, gy, n, action_hist), dtype=np.float32)
        if hasattr(agent, "eval_action"):
            action = agent.eval_action(obs)
        else:
            action = agent.act(obs, step)
        dx, dy = ACTIONS[action]
        nx, ny = ax + dx, ay + dy
        if 0 <= nx < n and 0 <= ny < n and maze[ny][nx] != WALL:
            ax, ay = nx, ny
        action_hist.append(action)
        visited.append((ax, ay))
        if (ax, ay) == (gx, gy):
            solved = True
            break
    return visited, solved, step + 1


def render_one(ax_plot, maze, traj, title: str, color: str, n: int):
    """Render one maze panel."""
    # Background: walls=black, hazards=red, open=white
    grid = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            if maze[y][x] == WALL:
                grid[y, x] = 0  # black
            elif maze[y][x] == HAZARD:
                grid[y, x] = 0.5  # mid (red overlay below)
            else:
                grid[y, x] = 1  # white
    ax_plot.imshow(grid, cmap="gray", origin="upper", vmin=0, vmax=1)
    # Hazard overlay
    for y in range(n):
        for x in range(n):
            if maze[y][x] == HAZARD:
                ax_plot.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                            facecolor="red", alpha=0.5))
    # Trajectory
    if len(traj) > 1:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax_plot.plot(xs, ys, "-", color=color, linewidth=1.2, alpha=0.7)
        ax_plot.plot(xs[0], ys[0], "o", color="green", markersize=6,
                     markeredgecolor="black")  # start
    # Goal
    ax_plot.plot(n - 2, n - 2, "*", color="gold", markersize=14,
                 markeredgecolor="black")
    ax_plot.set_title(title, fontsize=9)
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])


def main() -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # Find 6 mazes where MLP_DQN fails but NoBackRandom succeeds (or vice versa)
    rng = np.random.default_rng(42)
    failure_cases = []
    nobacks = []
    walls = []
    attempts = 0
    while len(failure_cases) < 6 and attempts < 200:
        attempts += 1
        s = int(rng.integers(0, 10**8))
        maze = make_maze(MAZE_SIZE, seed=s)
        if not is_solvable(maze, MAZE_SIZE):
            continue

        # Train an MLP_DQN quickly on different seeds to get a representative one
        if not failure_cases:
            mlp = MLPDQNAgent(hidden=64, eps_decay=20000)
            _ = run_experiment(mlp, "MLP_DQN", MAZE_SIZE, 100, 0, 42)
        # else: reuse the trained MLP

        # Run all 4 agents on the same maze
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2

        traj_mlp, solved_mlp, _ = trace_agent(maze, "MLP_DQN", mlp, gx, gy, MAZE_SIZE)

        nb = NoBacktrackRandomAgent()
        traj_nb, solved_nb, _ = trace_agent(maze, "NoBackRandom", nb, gx, gy, MAZE_SIZE)

        if not solved_mlp and solved_nb:
            failure_cases.append((s, maze, traj_mlp, traj_nb))

    if not failure_cases:
        print("No clear failure cases found in sample.")
        return

    # Plot grid: 2 rows x 3 cols, top row MLP_DQN failures, bottom NoBack successes
    fig, axes = plt.subplots(2, len(failure_cases), figsize=(3 * len(failure_cases), 6))
    if len(failure_cases) == 1:
        axes = axes.reshape(2, 1)
    for j, (s, maze, traj_mlp, traj_nb) in enumerate(failure_cases):
        render_one(axes[0, j], maze, traj_mlp,
                   f"MLP_DQN  fails  (seed={s})  steps={len(traj_mlp)}",
                   "blue", MAZE_SIZE)
        render_one(axes[1, j], maze, traj_nb,
                   f"NoBackRandom  solves  steps={len(traj_nb)}",
                   "orange", MAZE_SIZE)
    fig.suptitle("Side-by-side: same maze, different agents (9x9)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Found {len(failure_cases)} failure cases")


if __name__ == "__main__":
    main()
