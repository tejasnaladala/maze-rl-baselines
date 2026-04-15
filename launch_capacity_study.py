"""Phase 3B: Network-capacity sensitivity study.

Tests reviewer attack A5: "Neural RL might match Random if networks were larger."

Trains MLP_DQN with four network sizes:
  - hidden = 32 (half size, sanity)
  - hidden = 64 (default, matches main table)
  - hidden = 128 (2x)
  - hidden = 256 (4x — the reviewer's demand)

At 2 scales (9x9, 13x13) x 20 seeds x 4 capacities = 160 runs.

If MLP(256) matches or beats NoBackRandom, the "function approximation is the
problem" story weakens. If it still loses, the reviewer critique is dead.

Written to raw_results/exp_capacity_study/.
"""

import sys, json, random, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    MLPDQNAgent,
    run_experiment, load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 13]
NUM_TRAIN = 100
NUM_TEST = 50

CAPACITIES = [32, 64, 128, 256]

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_capacity_study'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(MAZE_SIZES) * len(CAPACITIES)
    done_count = len(completed)

    print(f"\nPhase 3B: Capacity sensitivity — {total_runs} runs, {done_count} done")
    print(f"Capacities: {CAPACITIES}")
    print(f"Sizes: {MAZE_SIZES}, Seeds: {len(SEEDS)}")
    print(f"Code hash: {code_hash()}\n")

    # Smaller sizes + capacities first so we have full coverage fast
    for capacity in CAPACITIES:
        for maze_size in MAZE_SIZES:
            for seed in SEEDS:
                agent_name = f"MLP_DQN_h{capacity}"
                key = run_key(agent_name, maze_size, seed)
                if key in completed:
                    continue

                print(f"  [{done_count}/{total_runs}] {agent_name} {maze_size}x{maze_size} s={seed}...",
                      end=" ", flush=True)
                t0 = time.time()

                set_all_seeds(seed, deterministic=False)  # keep speed
                agent = MLPDQNAgent(
                    hidden=capacity, device=DEVICE, eps_decay=NUM_TRAIN * 200
                )
                results = run_experiment(agent, agent_name, maze_size, NUM_TRAIN, NUM_TEST, seed)

                run_file = OUT_DIR / f'{agent_name}_{maze_size}_{seed}.json'
                atomic_save([{
                    'agent_name': r.agent_name, 'maze_size': r.maze_size, 'seed': r.seed,
                    'phase': r.phase, 'episode': r.episode, 'reward': r.reward,
                    'steps': r.steps, 'solved': r.solved, 'synops': r.synops,
                    'wall_time_s': r.wall_time_s, 'config': {**r.config, 'hidden': capacity},
                    'lib_version': r.lib_version,
                } for r in results], run_file)

                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done_count += 1

                test = [r for r in results if r.phase == 'test']
                test_success = sum(r.solved for r in test) / len(test) * 100
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nPhase 3B complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
