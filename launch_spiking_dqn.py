"""Tier 1: SpikingDQN add-back sweep.

Separate launcher for SpikingDQN only, using the SAME protocol as launch_h200.py
(original experiment_lib.py) so results are merge-compatible with the 600-run Tier 0 dataset.

20 seeds x 6 maze sizes = 120 runs. Checkpointed, resume-safe, atomic per-run writes.
"""

import sys, json, random, time, shutil
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib import (
    SpikingDQNAgent,
    run_experiment, ExpResult,
    load_checkpoint, save_checkpoint, run_key,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 11, 13, 17, 21, 25]
NUM_TRAIN = 100
NUM_TEST = 50

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_spiking_dqn'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def atomic_save(data: list[dict], path: Path) -> None:
    """Atomic JSON write using os.replace (Windows- and POSIX-safe)."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(MAZE_SIZES)
    done_count = len(completed)

    print(f"\nTier 1: SpikingDQN sweep — {total_runs} total runs, {done_count} already completed")
    print(f"Seeds: {len(SEEDS)}, Sizes: {MAZE_SIZES}")
    print(f"Output: {OUT_DIR}\n")

    agent_name = 'SpikingDQN'

    for maze_size in MAZE_SIZES:
        for seed in SEEDS:
            key = run_key(agent_name, maze_size, seed)
            if key in completed:
                continue

            print(f"  [{done_count}/{total_runs}] {agent_name} {maze_size}x{maze_size} seed={seed}...",
                  end=" ", flush=True)
            t0 = time.time()

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if DEVICE == 'cuda':
                torch.cuda.manual_seed(seed)

            agent = SpikingDQNAgent(hidden=64, num_steps=8, device=DEVICE, eps_decay=NUM_TRAIN * 200)
            results = run_experiment(agent, agent_name, maze_size, NUM_TRAIN, NUM_TEST, seed)

            run_file = OUT_DIR / f'{agent_name}_{maze_size}_{seed}.json'
            atomic_save([{
                'agent_name': r.agent_name, 'maze_size': r.maze_size, 'seed': r.seed,
                'phase': r.phase, 'episode': r.episode, 'reward': r.reward,
                'steps': r.steps, 'solved': r.solved, 'synops': r.synops,
            } for r in results], run_file)

            completed.add(key)
            save_checkpoint(CHECKPOINT_FILE, completed)
            done_count += 1

            test_results = [r for r in results if r.phase == 'test']
            test_success = sum(r.solved for r in test_results) / len(test_results) * 100
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nTier 1 complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
