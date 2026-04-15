"""Phase 3A: V2 clean re-run of FeatureQ and TabularQ at all 6 sizes.

The V1 dataset has a W6 bug where FeatureQ and TabularQ use `act()` at test
time (with epsilon floor) instead of deterministic greedy. V2 adds `eval_action()`
to both, which produces different (typically higher) test success rates.

This launcher re-runs FeatureQ and TabularQ under V2 so the main table has
clean, bug-free numbers for the tabular baselines at ALL 6 sizes.

2 agents x 6 sizes x 20 seeds = 240 runs.
Light agents (no neural net) so runtime is ~10-20s per run on 5070 Ti.
Total: ~60-80 minutes wall clock.

Writes to raw_results/exp_v2_tabular/ (separate from the V1 results so we
don't contaminate the insurance backup).
"""

import sys, json, random, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    FeatureQAgent, TabularQAgent,
    run_experiment, load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 11, 13, 17, 21, 25]
NUM_TRAIN = 100
NUM_TEST = 50

AGENTS = [
    ('FeatureQ_v2', lambda: FeatureQAgent()),
    ('TabularQ_v2', lambda: TabularQAgent()),
]

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_v2_tabular'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(MAZE_SIZES) * len(AGENTS)
    done_count = len(completed)

    print(f"\nPhase 3A: V2 clean tabular re-run — {total_runs} runs, {done_count} done")
    print(f"Agents: {[a for a, _ in AGENTS]}")
    print(f"Sizes: {MAZE_SIZES}, Seeds: {len(SEEDS)}")
    print(f"Code hash: {code_hash()}\n")

    for agent_name, make_agent in AGENTS:
        for maze_size in MAZE_SIZES:
            for seed in SEEDS:
                key = run_key(agent_name, maze_size, seed)
                if key in completed:
                    continue

                print(f"  [{done_count}/{total_runs}] {agent_name} {maze_size}x{maze_size} s={seed}...",
                      end=" ", flush=True)
                t0 = time.time()

                set_all_seeds(seed, deterministic=True)
                agent = make_agent()
                results = run_experiment(agent, agent_name, maze_size, NUM_TRAIN, NUM_TEST, seed)

                run_file = OUT_DIR / f'{agent_name}_{maze_size}_{seed}.json'
                atomic_save([{
                    'agent_name': r.agent_name, 'maze_size': r.maze_size, 'seed': r.seed,
                    'phase': r.phase, 'episode': r.episode, 'reward': r.reward,
                    'steps': r.steps, 'solved': r.solved, 'synops': r.synops,
                    'wall_time_s': r.wall_time_s, 'config': r.config,
                    'lib_version': r.lib_version,
                } for r in results], run_file)

                completed.add(key)
                save_checkpoint(CHECKPOINT_FILE, completed)
                done_count += 1

                test = [r for r in results if r.phase == 'test']
                test_success = sum(r.solved for r in test) / len(test) * 100
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nPhase 3A complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
