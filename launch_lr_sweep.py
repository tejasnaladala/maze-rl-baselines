"""Tier A.5 — Learning rate sweep (closes A4 attack).

A4 reviewer claim: "Your MLP_DQN uses lr=5e-4. Did you try lr=1e-4 or 1e-3?"

This launcher tests MLP_DQN at lr in {1e-4, 5e-4, 1e-3, 3e-3} x 10 seeds x 9x9.
40 runs total. Designed to fit on RTX 5070 Ti in ~30 min.

Outputs to raw_results/exp_lr_sweep/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    MLPDQNAgent,
    run_experiment,
    load_checkpoint,
    save_checkpoint,
    run_key,
    atomic_save,
    set_all_seeds,
    code_hash,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
MAZE_SIZE = 9
NUM_TRAIN = 100
NUM_TEST = 50
LRS = [1e-4, 5e-4, 1e-3, 3e-3]

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_lr_sweep"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    total = len(LRS) * len(SEEDS)
    done = len(completed)
    print(f"\nLR sweep (A4 closure): {total} runs, {done} done")
    print(f"LRs: {LRS}, seeds: {len(SEEDS)}, size: {MAZE_SIZE}x{MAZE_SIZE}")
    print(f"Code hash: {code_hash()}\n")

    for lr in LRS:
        for seed in SEEDS:
            agent_name = f"MLP_DQN_lr{lr:.0e}".replace("e-0", "e-")
            key = run_key(agent_name, MAZE_SIZE, seed)
            if key in completed:
                continue

            print(f"  [{done}/{total}] {agent_name} s={seed}...",
                  end=" ", flush=True)
            t0 = time.time()

            set_all_seeds(seed, deterministic=False)
            agent = MLPDQNAgent(
                hidden=64, lr=lr, device=DEVICE, eps_decay=NUM_TRAIN * 200
            )
            results = run_experiment(agent, agent_name, MAZE_SIZE,
                                     NUM_TRAIN, NUM_TEST, seed)

            run_file = OUT_DIR / f"{agent_name}_{MAZE_SIZE}_{seed}.json"
            atomic_save([{
                "agent_name": r.agent_name,
                "maze_size": r.maze_size,
                "seed": r.seed,
                "phase": r.phase,
                "episode": r.episode,
                "reward": r.reward,
                "steps": r.steps,
                "solved": r.solved,
                "synops": r.synops,
                "wall_time_s": r.wall_time_s,
                "config": {**r.config, "lr": lr},
                "lib_version": r.lib_version,
            } for r in results], run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done += 1

            test = [r for r in results if r.phase == "test"]
            success = sum(r.solved for r in test) / len(test) * 100
            print(f"done ({time.time()-t0:.1f}s) test={success:.0f}%")

    print(f"\nLR sweep complete. {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
