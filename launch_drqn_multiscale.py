"""Tier A.4 — DRQN multi-scale extension.

Already have DRQN at 9x9 (n=20, mean=19.0%). This launcher extends to
larger scales to bullet-proof the A3 attack defense across maze sizes.

DRQN det at sizes {13, 17, 21} x 20 seeds = 60 runs.

Outputs append to raw_results/exp_memory_agents/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    load_checkpoint,
    save_checkpoint,
    run_key,
    atomic_save,
    set_all_seeds,
    code_hash,
    run_experiment,
)

# Reuse the DRQN agent class from launch_memory_agents
sys.path.insert(0, str(Path(__file__).parent))
from launch_memory_agents import DRQNAgent  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [13, 17, 21]
NUM_TRAIN = 100
NUM_TEST = 50

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_memory_agents"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    total = len(MAZE_SIZES) * len(SEEDS)
    done_in_scope = sum(
        1 for k in completed
        if k.startswith("DRQN_") and any(f"_{s}_" in k for s in MAZE_SIZES)
    )
    print(f"\nDRQN multi-scale: {total} runs, {done_in_scope} done")
    print(f"Sizes: {MAZE_SIZES}, code hash: {code_hash()}\n")

    n = done_in_scope
    for maze_size in MAZE_SIZES:
        for seed in SEEDS:
            key = run_key("DRQN", maze_size, seed)
            if key in completed:
                continue

            print(f"  [{n}/{total}] DRQN {maze_size}x{maze_size} s={seed}...",
                  end=" ", flush=True)
            t0 = time.time()

            set_all_seeds(seed, deterministic=True)
            agent = DRQNAgent(
                hidden=64, seq_len=8, device=DEVICE, eps_decay=NUM_TRAIN * 200
            )
            results = run_experiment(agent, "DRQN", maze_size, NUM_TRAIN, NUM_TEST, seed)

            run_file = OUT_DIR / f"DRQN_{maze_size}_{seed}.json"
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
                "config": r.config,
                "lib_version": r.lib_version,
            } for r in results], run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            n += 1
            test = [r for r in results if r.phase == "test"]
            test_success = sum(r.solved for r in test) / len(test) * 100
            print(f"done ({time.time()-t0:.1f}s) test={test_success:.0f}%")

    print(f"\nDRQN multi-scale complete in {OUT_DIR}")


if __name__ == "__main__":
    main()
