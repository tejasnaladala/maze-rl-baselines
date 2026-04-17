"""Reward sensitivity sweep beyond K4 (per Codex review).

Tests our claim that the headline finding is robust to reward design,
not an artifact of our specific shaping.

6 reward configurations at 9x9, all 5 main agents, 20 seeds:
  - 'full':           default (everything on, matches main sweep)
  - 'vanilla':        no_distance + no_revisit (matches K4 vanilla)
  - 'no_distance':    distance shaping off, revisit penalty on
  - 'no_revisit':     distance shaping on, revisit penalty off
  - 'no_wall_cost':   walls free (wall_bump_cost=0)
  - 'no_hazard_cost': hazards free (hazard_cost=0)

5 agents x 6 configs x 20 seeds = 600 runs.

Uses run_experiment from experiment_lib_v2 directly (no custom step logic),
guaranteeing we test exactly the same step semantics as the main sweep.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (
    NoBacktrackRandomAgent, RandomAgent, FeatureQAgent, MLPDQNAgent, DoubleDQNAgent,
    run_experiment,
    load_checkpoint, save_checkpoint, run_key, atomic_save,
    set_all_seeds, code_hash,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
MAZE_SIZE = 9
NUM_TRAIN = 100
NUM_TEST = 50

REWARD_CONFIGS = {
    "full":           dict(reward_shaping=True,  visit_penalty=True,  wall_bump_cost=-0.3, hazard_cost=-1.0),
    "vanilla":        dict(reward_shaping=False, visit_penalty=False, wall_bump_cost=-0.3, hazard_cost=-1.0),
    "no_distance":    dict(reward_shaping=False, visit_penalty=True,  wall_bump_cost=-0.3, hazard_cost=-1.0),
    "no_revisit":     dict(reward_shaping=True,  visit_penalty=False, wall_bump_cost=-0.3, hazard_cost=-1.0),
    "no_wall_cost":   dict(reward_shaping=True,  visit_penalty=True,  wall_bump_cost=0.0,  hazard_cost=-1.0),
    "no_hazard_cost": dict(reward_shaping=True,  visit_penalty=True,  wall_bump_cost=-0.3, hazard_cost=0.0),
}

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_reward_sensitivity"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


def make_agent(name: str):
    if name == "Random": return RandomAgent()
    if name == "NoBackRandom": return NoBacktrackRandomAgent()
    if name == "FeatureQ_v2": return FeatureQAgent()
    if name == "MLP_DQN": return MLPDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)
    if name == "DoubleDQN": return DoubleDQNAgent(hidden=64, device=DEVICE, eps_decay=NUM_TRAIN * 200)
    raise ValueError(name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    AGENTS = ["Random", "NoBackRandom", "FeatureQ_v2", "MLP_DQN", "DoubleDQN"]
    total = len(AGENTS) * len(REWARD_CONFIGS) * len(SEEDS)
    done = len(completed)
    print(f"\nReward sensitivity: {total} runs, {done} done")
    print(f"Configs: {list(REWARD_CONFIGS.keys())}")
    print(f"Code hash: {code_hash()}\n")

    for cfg_name, cfg in REWARD_CONFIGS.items():
        for agent_name in AGENTS:
            for seed in SEEDS:
                composite = f"{cfg_name}__{agent_name}"
                key = run_key(composite, MAZE_SIZE, seed)
                if key in completed:
                    continue
                print(f"  [{done}/{total}] {composite} s={seed}...", end=" ", flush=True)
                t0 = time.time()
                set_all_seeds(seed, deterministic=False)
                agent = make_agent(agent_name)
                results = run_experiment(
                    agent, composite, MAZE_SIZE, NUM_TRAIN, NUM_TEST, seed, **cfg
                )

                run_file = OUT_DIR / f"{composite}_{MAZE_SIZE}_{seed}.json"
                atomic_save([{
                    "agent_name": r.agent_name,
                    "base_agent": agent_name,
                    "reward_config": cfg_name,
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
                done += 1
                test = [r for r in results if r.phase == "test"]
                succ = sum(r.solved for r in test) / len(test) * 100
                print(f"done ({time.time()-t0:.0f}s) test={succ:.0f}%")

    print(f"\nReward sensitivity complete. {total} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
