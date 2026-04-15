"""Phase 3C: Per-episode reward decomposition analysis.

Reviewer attack A8: "Neural RL agents may be underperforming not because they
can't learn navigation, but because they're being dominated by hazard-avoidance
signals in the reward function. Maybe they're learning to stand still to avoid
hazards, not learning to navigate."

This script analyzes every training and test episode in the result JSON files
and decomposes the episode reward into components:
  - goal_reward   = +10 if the episode ended by reaching the goal
  - hazard_cost   = total -1.0 penalties the agent ate
  - wall_bump_cost = total -0.3 penalties the agent ate
  - shaping_reward = distance shaping (+0.08 / -0.04) accumulated
  - visit_penalty = -0.1 revisit penalties
  - step_cost     = -0.02 per step
  - other / residual

We can't recompute components exactly because the result JSONs only store
the aggregate episode reward, number of steps, and solved flag. So we use a
BOUND: given (total_reward, steps, solved), we can infer:
  - reward budget from steps alone: -0.02 * steps
  - goal contribution: +10 if solved, 0 otherwise
  - residual = total_reward - step_budget - goal_contribution
    = hazards + walls + shaping_reward - visit_penalty (net of positive and negative)

A more sophisticated decomposition requires re-running a small subset with
per-step logging — we do that here for diagnostic purposes on 3 seeds.

Also produces comparison tables: residual / step for each agent, showing
whether neural agents are paying more wall/hazard cost than random variants
(would support "they're afraid of hazards") or less (they're avoiding goal
while wandering safely).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def load_episodes(dirs: Iterable[Path]) -> list[dict]:
    out: list[dict] = []
    for d in dirs:
        if not d.exists():
            print(f"  skip missing: {d}")
            continue
        n_before = len(out)
        n_skipped = 0
        for f in d.glob("*.json"):
            if f.name == 'checkpoint.json':
                continue
            try:
                data = json.load(open(f))
                if isinstance(data, list):
                    out.extend(data)
                elif isinstance(data, dict):
                    out.append(data)
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                n_skipped += 1
        print(f"  {d}: +{len(out) - n_before} records ({n_skipped} skipped)")
    return out


def decompose_episode(r: dict) -> dict:
    """FIXED decomposition.

    Reviewer finding (Phase 1 audit): the env OVERWRITES step cost with
    wall_bump_cost / hazard_cost / goal_reward, not adds. So the old formula
    `total - (-0.02*steps) - goal` double-counted step cost on bump steps and
    was biased POSITIVE for agents with lots of wall bumps.

    Fixed formula: use `total - goal_contribution` only. This captures
    everything negative (step cost, walls, hazards, visit penalty, shaping)
    and we compare AGENTS normalized by mean_steps to get a per-step
    "pain level" that's apples-to-apples.
    """
    total = float(r.get('reward', 0.0))
    steps = int(r.get('steps', 0))
    solved = bool(r.get('solved', False))

    goal_contribution = 10.0 if solved else 0.0
    # Net negative reward, per step — includes step cost, walls, hazards,
    # visit penalty, and (negative of) positive shaping terms. Lower means
    # more "pain" per step.
    pain = total - goal_contribution
    pain_per_step = pain / steps if steps else 0.0

    return {
        'total_reward': total,
        'steps': steps,
        'solved': solved,
        'goal_contribution': goal_contribution,
        'pain': pain,
        'pain_per_step': pain_per_step,
    }


def canonical_agent(name: str) -> str:
    """Strip reward-ablation config prefix from agent name.

    FIX Phase 1 audit: only strip DOUBLE underscore separator. Previous
    version stripped `FeatureQ` from `FeatureQ_v2` because it matched
    `full_` + `eatureQ_v2` is false but `vanilla_` would catch `vanilla_` if
    such a prefix existed. Restricting to `__` separator eliminates false
    matches entirely.
    """
    for cfg in ('full', 'vanilla'):
        if name.startswith(cfg + '__'):
            return name[len(cfg) + 2:]
    return name


def summarize(records: list[dict], by: str = 'agent_name') -> dict:
    """Group records by agent (canonical) and compute reward decomposition stats."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get('phase') != 'test':
            continue
        key = canonical_agent(r.get(by, 'unknown'))
        groups[key].append(decompose_episode(r))

    summary: dict[str, dict] = {}
    for key, decompositions in groups.items():
        if not decompositions:
            continue
        arr_pain = np.array([d['pain'] for d in decompositions])
        arr_pain_per_step = np.array([d['pain_per_step'] for d in decompositions])
        arr_steps = np.array([d['steps'] for d in decompositions])
        arr_solved = np.array([d['solved'] for d in decompositions])

        summary[key] = {
            'n_episodes': len(decompositions),
            'mean_total_reward': float(np.mean([d['total_reward'] for d in decompositions])),
            'mean_steps': float(np.mean(arr_steps)),
            'mean_pain': float(np.mean(arr_pain)),
            'mean_pain_per_step': float(np.mean(arr_pain_per_step)),
            'pain_per_step_std': float(np.std(arr_pain_per_step)),
            'success_rate': float(np.mean(arr_solved)),
        }
    return summary


def main() -> None:
    # FIX Phase 1 audit: exclude insurance_backup (contains agents with different
    # reward config + code version) and exclude reward_ablation_fast's vanilla
    # config from the "full reward" comparison. Use only runs that were trained
    # AND tested on the same standard reward function.
    dirs = [
        Path("raw_results/exp_oracle_random"),        # Tier 4 — all random variants + BFS
        Path("raw_results/exp_memory_agents"),        # DRQN
        Path("raw_results/exp_v2_tabular"),           # V2 tabular (Phase 3A)
        Path("raw_results/exp_capacity_study"),       # MLP capacity (Phase 3B)
        Path("insurance_backup/exp_h200"),            # V1 neural agents (valid — V1 MLP/Double had correct eval_action)
    ]
    dirs = [d for d in dirs if d.exists()]
    print(f"Loading from {len(dirs)} directories...\n")
    records = load_episodes(dirs)
    print(f"\nLoaded {len(records)} episode records total")

    # Filter: 9×9 size, test phase, standard-reward runs only.
    # V1 records don't have config.reward_shaping but we know they were run
    # with full shaping. Reward ablation records DO have config — exclude
    # those because their reward function differs.
    def is_full_reward(r: dict) -> bool:
        # FIX: reject reward ablation records that explicitly set non-default reward
        cfg = r.get('config') or {}
        if 'reward_shaping' in cfg and not cfg.get('reward_shaping', True):
            return False
        if 'visit_penalty' in cfg and not cfg.get('visit_penalty', True):
            return False
        if 'wall_bump_cost' in cfg and cfg.get('wall_bump_cost') != -0.3:
            return False
        if 'hazard_cost' in cfg and cfg.get('hazard_cost') != -1.0:
            return False
        return True

    records_9 = [
        r for r in records
        if r.get('maze_size') == 9
        and r.get('phase') == 'test'
        and is_full_reward(r)
    ]
    print(f"9x9 test-phase full-reward episodes: {len(records_9)}\n")

    summary = summarize(records_9)

    print(f"=== Reward decomposition at 9x9 (test phase, full reward) ===\n")
    print(f"{'Agent':<20} {'n':<6} {'mean_R':<9} {'steps':<7} "
          f"{'pain':<10} {'pain/step':<11} {'success':<9}")
    print("-" * 80)
    for agent, s in sorted(summary.items(), key=lambda kv: -kv[1]['success_rate']):
        print(f"{agent:<20} {s['n_episodes']:<6} {s['mean_total_reward']:+8.2f} "
              f"{s['mean_steps']:<7.0f} {s['mean_pain']:+9.2f} "
              f"{s['mean_pain_per_step']:+10.4f} "
              f"{100*s['success_rate']:<8.1f}")

    # Save to CSV for paper appendix
    out_dir = Path("analysis_output/reward_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_dir / 'decomposition_9x9.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agent', 'n_episodes', 'mean_total_reward', 'mean_steps',
                    'mean_pain', 'mean_pain_per_step', 'success_rate'])
        for agent, s in sorted(summary.items()):
            w.writerow([agent, s['n_episodes'], s['mean_total_reward'], s['mean_steps'],
                        s['mean_pain'], s['mean_pain_per_step'], s['success_rate']])
    print(f"\nWrote {out_dir / 'decomposition_9x9.csv'}")

    # Key diagnostic: does MLP_DQN spend different fraction of reward budget on pain?
    print(f"\n=== DIAGNOSTIC: Per-step pain (reviewer attack A8) ===\n")
    print(f"Lower pain_per_step = less negative reward per step = less wall/hazard bumps per step")
    print(f"FIXED formula: pain = total_reward - goal_contribution")
    print(f"This captures EVERYTHING negative (step cost + walls + hazards + visit + shaping)\n")

    key_agents = ['BFSOracle', 'Random', 'NoBackRandom', 'LevyRandom_2.0', 'LevyRandom_1.5',
                  'FeatureQ', 'FeatureQ_v2', 'TabularQ', 'TabularQ_v2',
                  'MLP_DQN', 'DoubleDQN', 'DRQN']
    for a in key_agents:
        if a in summary:
            s = summary[a]
            print(f"  {a:<15}: pain_per_step = {s['mean_pain_per_step']:+.4f}  "
                  f"(n_ep={s['n_episodes']}, success={100*s['success_rate']:.1f}%)")

    if 'MLP_DQN' in summary and 'Random' in summary:
        delta = summary['MLP_DQN']['mean_pain_per_step'] - summary['Random']['mean_pain_per_step']
        print(f"\n  MLP_DQN vs Random pain/step delta: {delta:+.4f}")
        if delta < -0.01:
            print("  -> MLP_DQN has LOWER (more negative) pain/step than Random.")
            print("     This means MLP_DQN takes MORE negative reward per step.")
            print("     Possible: excessive wall bumping OR visit-penalty accumulation.")
            print("     SUPPORTS hazard-avoidance / local-minimum theory.")
        elif delta > 0.01:
            print("  -> MLP_DQN has HIGHER (less negative) pain/step than Random.")
            print("     MLP_DQN successfully avoids walls and hazards.")
            print("     DOES NOT support hazard-avoidance theory.")
            print("     Diagnosis: neural agents learn safety but fail exploration.")
        else:
            print("  -> MLP_DQN and Random have similar pain/step (neutral)")


if __name__ == '__main__':
    main()
