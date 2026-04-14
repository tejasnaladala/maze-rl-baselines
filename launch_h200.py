"""H200 Launch Script -- runs the full Approach C experiment suite.

Usage:
    python research/01_experiments/launch_h200.py

Detects CUDA automatically. Checkpoints after each (agent, size, seed).
Resume by re-running -- completed runs are skipped.
"""

import sys, os, json, random, time, tempfile, shutil
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib import (
    make_maze, ego_features, is_solvable,
    RandomAgent, TabularQAgent, FeatureQAgent, MLPDQNAgent, SpikingDQNAgent, DoubleDQNAgent,
    run_experiment, ExpResult, save_results, load_checkpoint, save_checkpoint, run_key,
    OBS_DIM, NUM_ACTIONS,
)

# Detect device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Configuration
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 11, 13, 17, 21, 25]
NUM_TRAIN = 100
NUM_TEST = 50

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_h200'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def make_agents():
    """Create all agents with CUDA support."""
    return [
        ('Random', lambda: RandomAgent()),
        ('TabularQ', lambda: TabularQAgent()),
        ('FeatureQ', lambda: FeatureQAgent()),
        ('MLP_DQN', lambda: MLPDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
        ('SpikingDQN', lambda: SpikingDQNAgent(hidden=64, num_steps=8, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
        ('DoubleDQN', lambda: DoubleDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
    ]


def atomic_save(data: list[dict], path: Path) -> None:
    """Atomic JSON write -- write to temp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f)
    shutil.move(str(tmp), str(path))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(MAZE_SIZES) * len(make_agents())
    done_count = len(completed)

    print(f"\nApproach C: {total_runs} total runs, {done_count} already completed")
    print(f"Seeds: {len(SEEDS)}, Sizes: {MAZE_SIZES}, Agents: {len(make_agents())}")
    print(f"Output: {OUT_DIR}\n")

    for maze_size in MAZE_SIZES:
        for seed in SEEDS:
            for agent_name, make_agent in make_agents():
                key = run_key(agent_name, maze_size, seed)
                if key in completed:
                    continue

                print(f"  [{done_count}/{total_runs}] {agent_name} {maze_size}x{maze_size} seed={seed}...", end=" ", flush=True)
                t0 = time.time()

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if DEVICE == 'cuda':
                    torch.cuda.manual_seed(seed)

                agent = make_agent()
                results = run_experiment(agent, agent_name, maze_size, NUM_TRAIN, NUM_TEST, seed)

                # Save per-run results
                run_file = OUT_DIR / f'{agent_name}_{maze_size}_{seed}.json'
                atomic_save([{
                    'agent_name': r.agent_name, 'maze_size': r.maze_size, 'seed': r.seed,
                    'phase': r.phase, 'episode': r.episode, 'reward': r.reward,
                    'steps': r.steps, 'solved': r.solved, 'synops': r.synops,
                } for r in results], run_file)

                # Update checkpoint
                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done_count += 1

                test_results = [r for r in results if r.phase == 'test']
                test_success = sum(r.solved for r in test_results) / len(test_results) * 100
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nAll {total_runs} runs complete. Results in {OUT_DIR}")


if __name__ == '__main__':
    main()
