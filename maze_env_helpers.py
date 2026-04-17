"""Helper functions for custom launchers using the maze env outside run_experiment.

Provides a uniform interface that matches experiment_lib_v2's conventions:
- Position: (ax, ay) where ax=column, ay=row
- Action encoding: matches ACTIONS in lib (dx, dy format)
- ego_features called with proper signature
"""

from __future__ import annotations

import numpy as np

from experiment_lib_v2 import (
    ego_features,
    ACTIONS,
    WALL,
    HAZARD,
    is_solvable,
    make_maze,
)


def make_solvable_maze(size: int, seed: int) -> tuple:
    """Returns (maze, ax, ay, gx, gy, size).
    Loops on different seeds until is_solvable() returns True.
    """
    s = seed
    while True:
        m = make_maze(size, seed=s)
        if is_solvable(m, size):
            return (m, 1, 1, size - 2, size - 2, size)
        s += 1


def get_obs(maze, ax: int, ay: int, gx: int, gy: int, size: int,
            action_hist: list) -> np.ndarray:
    """Compute ego_features and return as numpy array."""
    return np.array(ego_features(maze, ax, ay, gx, gy, size, action_hist),
                    dtype=np.float32)


def step_env(maze, ax: int, ay: int, action: int, size: int) -> tuple:
    """One env step. Returns (new_ax, new_ay, cell_value, hit_wall, hit_hazard).
    cell_value is the type of cell entered (or current if didn't move).
    """
    dx, dy = ACTIONS[action]
    nx, ny = ax + dx, ay + dy
    if not (0 <= nx < size and 0 <= ny < size):
        return ax, ay, WALL, True, False
    cell = maze[ny][nx]
    if cell == WALL:
        return ax, ay, WALL, True, False
    if cell == HAZARD:
        return nx, ny, HAZARD, False, True
    return nx, ny, 0, False, False


def reward_fn(ax: int, ay: int, new_ax: int, new_ay: int, gx: int, gy: int,
              hit_wall: bool, hit_hazard: bool, visited: set,
              shaping: bool = True, visit_penalty: bool = True,
              wall_cost: float = -0.3, hazard_cost: float = -1.0,
              goal_reward: float = 10.0, step_cost: float = -0.04,
              shape_coef: float = 0.04) -> tuple:
    """Compute reward + done given the step outcome. Mirrors run_experiment."""
    reward = step_cost
    done = False
    if hit_wall:
        reward += wall_cost
    elif hit_hazard:
        reward += hazard_cost
    elif (new_ax, new_ay) == (gx, gy):
        reward += goal_reward
        done = True
    else:
        if shaping:
            old_d = abs(ax - gx) + abs(ay - gy)
            new_d = abs(new_ax - gx) + abs(new_ay - gy)
            if new_d < old_d:
                reward += shape_coef
            elif new_d > old_d:
                reward -= shape_coef
        if visit_penalty and (new_ax, new_ay) in visited:
            reward -= 0.1
    visited.add((new_ax, new_ay))
    return reward, done
