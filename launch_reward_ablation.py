"""Tier 2: Reward-function ablation.

Tests whether the "Random beats learners" finding is caused by asymmetric reward shaping.
If Random only wins under visit_penalty=True + reward_shaping=True, the paper's thesis
reduces to "our reward was miscalibrated" — we must reframe.
If Random still wins under the vanilla {-0.02 step, -0.3 wall, +10 goal} reward,
the finding is robust to reward choice and we can publish the stronger claim.

Five reward configurations x 5 agents x 3 sizes x 20 seeds = 1500 runs.
"""

import sys, json, random, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    RandomAgent, NoBacktrackRandomAgent, TabularQAgent, FeatureQAgent,
    MLPDQNAgent, DoubleDQNAgent,
    run_experiment, load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZES = [9, 13, 21]
NUM_TRAIN = 100
NUM_TEST = 50

# Five reward configurations spanning the space
REWARD_CONFIGS = {
    'full':          {'reward_shaping': True,  'visit_penalty': True},
    'no_visit':      {'reward_shaping': True,  'visit_penalty': False},
    'no_shape':      {'reward_shaping': False, 'visit_penalty': True},
    'vanilla':       {'reward_shaping': False, 'visit_penalty': False},
    'vanilla_noham': {'reward_shaping': False, 'visit_penalty': False,
                      'wall_bump_cost': -0.02, 'hazard_cost': -0.02},
}

AGENTS = [
    ('Random',     lambda: RandomAgent()),
    ('NoBackRand', lambda: NoBacktrackRandomAgent()),
    ('FeatureQ',   lambda: FeatureQAgent()),
    ('MLP_DQN',    lambda: MLPDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
    ('DoubleDQN',  lambda: DoubleDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)),
]

OUT_DIR = Path(__file__).parent / 'raw_results' / 'exp_reward_ablation'
CHECKPOINT_FILE = OUT_DIR / 'checkpoint.json'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)
    total_runs = len(SEEDS) * len(MAZE_SIZES) * len(AGENTS) * len(REWARD_CONFIGS)
    done_count = len(completed)

    print(f"\nTier 2: Reward ablation — {total_runs} runs, {done_count} already done")
    print(f"Configs: {list(REWARD_CONFIGS.keys())}")
    print(f"Agents: {[a for a, _ in AGENTS]}")
    print(f"Sizes: {MAZE_SIZES}, Seeds: {len(SEEDS)}")
    print(f"Output: {OUT_DIR}")
    print(f"Code hash: {code_hash()}\n")

    for cfg_name, reward_kwargs in REWARD_CONFIGS.items():
        for maze_size in MAZE_SIZES:
            for seed in SEEDS:
                for agent_name, make_agent in AGENTS:
                    run_tag = f"{cfg_name}_{agent_name}"
                    key = run_key(run_tag, maze_size, seed)
                    if key in completed:
                        continue

                    print(f"  [{done_count}/{total_runs}] {cfg_name} {agent_name} {maze_size}x{maze_size} s={seed}...",
                          end=" ", flush=True)
                    t0 = time.time()

                    set_all_seeds(seed, deterministic=True)
                    agent = make_agent()
                    results = run_experiment(
                        agent, run_tag, maze_size, NUM_TRAIN, NUM_TEST, seed,
                        **reward_kwargs,
                    )

                    run_file = OUT_DIR / f'{run_tag}_{maze_size}_{seed}.json'
                    atomic_save([{
                        'agent_name': r.agent_name, 'maze_size': r.maze_size, 'seed': r.seed,
                        'phase': r.phase, 'episode': r.episode, 'reward': r.reward,
                        'steps': r.steps, 'solved': r.solved, 'synops': r.synops,
                        'wall_time_s': r.wall_time_s, 'config': r.config, 'lib_version': r.lib_version,
                    } for r in results], run_file)

                    completed.add(key)
                    save_checkpoint(CHECKPOINT_FILE, completed)
                    done_count += 1

                    test = [r for r in results if r.phase == 'test']
                    test_success = sum(r.solved for r in test) / len(test) * 100
                    elapsed = time.time() - t0
                    print(f"done ({elapsed:.1f}s) test={test_success:.0f}%")

    print(f"\nTier 2 reward ablation complete. {total_runs} runs in {OUT_DIR}")


if __name__ == '__main__':
    main()
