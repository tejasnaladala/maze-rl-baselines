"""Loopy-maze generators for adversarial-reviewer audit.

* ``make_wilson_maze(size, seed)`` — Wilson's algorithm (loop-erased random
  walk), produces a UNIFORM SPANNING TREE maze (still simply-connected).
* ``add_loops(grid, size, seed, n_extra_passages)`` — knocks out N interior
  walls that have open cells on opposite sides, producing real shortcuts and
  breaking the spanning-tree property.

Grid format matches ``experiment_lib_v2.make_maze``: odd size, ``grid[y][x]``,
WALL=1, open=0. Start (1,1), goal (size-2, size-2).
"""

from __future__ import annotations

import random
from typing import List

from experiment_lib_v2 import WALL

Grid = List[List[int]]
_DIRS = [(0, -2), (2, 0), (0, 2), (-2, 0)]


def make_wilson_maze(size: int, seed: int) -> Grid:
    """Wilson's algorithm — uniform spanning tree maze on the odd sublattice."""
    if size % 2 == 0 or size < 5:
        raise ValueError(f"Size must be odd and >= 5, got {size}")

    rng = random.Random(seed)
    grid: Grid = [[WALL] * size for _ in range(size)]
    cells = [(x, y) for y in range(1, size - 1, 2) for x in range(1, size - 1, 2)]
    in_tree: set[tuple[int, int]] = set()

    start = cells[0]
    grid[start[1]][start[0]] = 0
    in_tree.add(start)

    for cell in cells[1:]:
        if cell in in_tree:
            continue
        # Loop-erased random walk from cell until we hit the tree
        path = [cell]
        path_idx = {cell: 0}
        cur = cell
        while cur not in in_tree:
            valid = [(dx, dy) for dx, dy in _DIRS
                     if 1 <= cur[0] + dx <= size - 2 and 1 <= cur[1] + dy <= size - 2]
            dx, dy = rng.choice(valid)
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in path_idx:
                cut = path_idx[nxt]
                for stale in path[cut + 1:]:
                    del path_idx[stale]
                del path[cut + 1:]
            else:
                path_idx[nxt] = len(path)
                path.append(nxt)
            cur = nxt
        # Carve the path into the tree
        for i in range(len(path) - 1):
            cx, cy = path[i]
            nx, ny = path[i + 1]
            grid[cy][cx] = 0
            grid[ny][nx] = 0
            grid[(cy + ny) // 2][(cx + nx) // 2] = 0
            in_tree.add(path[i])
        in_tree.add(path[-1])
    return grid


def add_loops(grid: Grid, size: int, seed: int, n_extra_passages: int) -> Grid:
    """Knock out ``n_extra_passages`` random interior walls that have open
    cells on opposite sides. Returns a NEW grid (does not mutate input).
    """
    if n_extra_passages <= 0:
        return [row[:] for row in grid]
    rng = random.Random(seed ^ 0xA1B2C3D4)
    candidates: list[tuple[int, int]] = []
    for y in range(2, size - 2):
        for x in range(2, size - 2):
            if grid[y][x] != WALL:
                continue
            horiz = grid[y][x - 1] != WALL and grid[y][x + 1] != WALL
            vert = grid[y - 1][x] != WALL and grid[y + 1][x] != WALL
            if horiz or vert:
                candidates.append((x, y))
    if not candidates:
        return [row[:] for row in grid]
    rng.shuffle(candidates)
    knock_n = min(n_extra_passages, len(candidates))
    new_grid: Grid = [row[:] for row in grid]
    for x, y in candidates[:knock_n]:
        new_grid[y][x] = 0
    return new_grid


__all__ = ["make_wilson_maze", "add_loops"]
