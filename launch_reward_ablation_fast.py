"""Tier 2 FAST: focused K4 reward ablation for the 2-hour window.

Minimum viable K4 decisive experiment:
  - 2 reward configs: `full` (baseline) vs `vanilla` (no shaping, no visit penalty)
  - 5 agents: Random, NoBackRandom, FeatureQ, MLP_DQN, DoubleDQN
  - 1 size: 9x9 (the best-populated existing size)
  - 20 seeds

= 200 runs at ~30s avg = ~100 min.

This binary test answers the decisive question: does the "Random >= trained" effect
survive the removal of reward shaping? If yes, the paper's thesis is robust.
If no, we reframe honestly as "reward shaping miscalibrated".
"""

import sys, json, random, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    RandomAgent, NoBacktrackRandomAgent, FeatureQAgent,
    MLPDQNAgent, DoubleDQNAgent,
    run_experiment, load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
NUM_TRAIN = 100
NUM_TEST = 50

REWARD_CONFIGS = {
    'full':    {'reward_shaping': True,  'visit_penalty': True},
    'vanilla': {'reward_shaping': False, 'visit_penalty': False},
}

# Canonical agent names - match the Tier 4 launcher's spelling
AGENTS = [
    ('Random',       lambda: RandomAgent()),
    ('NoBackRandom', lambda: NoBacktrackRandomAgent()),
    ('FeatureQ',     lambda: FeatureQAgent()),
    ('MLP_DQN',      lambda: MLPDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
    ('DoubleDQN',    lambda: DoubleDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
]

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_reward_ablation_fast'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(AGENTS) * len(REWARD_CONFIGS)
    done_count = len(completed)

    print(f"\nTier 2 FAST: reward ablation @ 9x9 — {total_runs} runs, {done_count} done")
    print(f"Configs: {list(REWARD_CONFIGS)}")
    print(f"Agents: {[a for a,_ in AGENTS]}")
    print(f"Seeds: {len(SEEDS)}, Size: {MAZE_SIZE}x{MAZE_SIZE}")
    print(f"Code hash: {code_hash()}\n")

    # Iterate size -> config -> agent -> seed so that fast agents and all configs
    # get interleaved early. Gives us useful partial data at any point.
    for agent_name, make_agent in AGENTS:
        for cfg_name, reward_kwargs in REWARD_CONFIGS.items():
            for seed in SEEDS:
                run_tag = f"{cfg_name}__{agent_name}"
                key = run_key(run_tag, MAZE_SIZE, seed)
                if key in completed:
                    continue

                print(f"  [{done_count}/{total_runs}] {cfg_name} {agent_name} s={seed}...",
                      end=" ", flush=True)
                t0 = time.time()

                set_all_seeds(seed, deterministic=True)
                agent = make_agent()
                results = run_experiment(
                    agent, run_tag, MAZE_SIZE, NUM_TRAIN, NUM_TEST, seed,
                    **reward_kwargs,
                )

                run_file = OUT_DIR / f'{run_tag}_{MAZE_SIZE}_{seed}.json'
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

    print(f"\nTier 2 FAST complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
