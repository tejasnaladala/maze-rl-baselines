"""final_analysis.py — run this after all experiments complete.

Merges every raw_results subdirectory, runs the fixed stats_pipeline, and
produces the publication-grade tables, figures, and the headline JSON that
`reproduce.py` will pin.

Usage:
    python final_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stats_pipeline import (
    load_all_results, canonical_agent, per_seed_success, per_seed_values,
    bootstrap_mean_ci, paired_bootstrap_diff, cohens_d, cohens_h,
    mann_whitney_u, holm_bonferroni, summary_table, pairwise_vs_reference,
    export_csv, export_latex_summary,
)

RESULT_DIRS = [
    "insurance_backup/exp_h200",                # 503 runs, original Tier 0 (V1)
    "raw_results/exp_h200",                     # if we resumed Tier 0 later
    "raw_results/exp_oracle_random",            # Tier 4a — BFS + random variants
    "raw_results/exp_reward_ablation_fast",     # Tier 2 fast — K4 test
    # raw_results/exp_reward_ablation moved to attic/exp_reward_ablation_orphan (orphaned)
    "raw_results/exp_memory_agents",            # Tier 4b — DRQN deterministic rerun
    "raw_results/exp_v2_tabular",               # Phase 3A — V2 FeatureQ/TabularQ
    "raw_results/exp_capacity_study",           # Phase 3B — MLP capacity sweep
    "raw_results/exp_spiking_dqn",              # Tier 1 — neuromorphic
    "raw_results/exp_budget_matched_sb3",       # Tier 2b — PPO/DQN/A2C
    "raw_results/exp_minigrid",                 # Tier 3 — cross-env
]

OUT_DIR = Path("analysis_output/final")


def section(name: str) -> None:
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in RESULT_DIRS if Path(d).exists()]
    section(f"Loading results from {len(dirs)} directories")
    for d in dirs:
        n = len(list(d.glob("*.json"))) - (1 if (d / 'checkpoint.json').exists() else 0)
        print(f"  {str(d):55s} {n:5d} files")
    results = load_all_results(dirs)
    print(f"\n  TOTAL records: {len(results)}")

    agents = sorted(set(canonical_agent(r.get('agent_name', '')) for r in results if r.get('agent_name')))
    sizes = sorted(set(r.get('maze_size') for r in results if r.get('maze_size')))
    print(f"  Canonical agents: {agents}")
    print(f"  Maze sizes: {sizes}")

    # ========================================================================
    # Headline Table 1: Per (agent, size) success rates with bootstrap CIs
    # ========================================================================
    section("Table 1: Per-agent success rates (test phase, 95% bootstrap CI)")
    primary_agents = [
        'BFSOracle',
        'NoBackRandom', 'LevyRandom_2.0', 'LevyRandom_1.5', 'Random',
        'FeatureQ_v2', 'TabularQ_v2',     # V2 clean (Phase 3A)
        'FeatureQ', 'TabularQ',           # V1 (for legacy comparison only)
        'MLP_DQN', 'DoubleDQN', 'DRQN',   # neural Q-learners
        'SpikingDQN',
    ]
    primary_sizes = [9, 11, 13, 17, 21, 25]
    summary = summary_table(results, primary_sizes, primary_agents, n_resamples=10000)
    export_csv(summary, OUT_DIR / 'table1_summary.csv')
    export_latex_summary(summary, OUT_DIR / 'table1_summary.tex')

    print(f"{'Agent':<18} {'Size':<5} {'n':<4} {'mean':<8} {'95% CI':<22}")
    for r in sorted(summary, key=lambda x: (x.size, x.agent)):
        print(f"{r.agent:<18} {r.size:<5} {r.n_seeds:<4} "
              f"{100*r.mean:5.1f}%   [{100*r.ci_lo:5.1f}, {100*r.ci_hi:5.1f}]")

    # ========================================================================
    # Headline Table 2: Pairwise vs Random with Cohen's d + Holm correction
    # ========================================================================
    section("Table 2: Pairwise vs Random (paired bootstrap, Holm-corrected)")
    pairwise = pairwise_vs_reference(
        results, primary_sizes, primary_agents, 'Random', n_resamples=10000
    )
    export_csv(pairwise, OUT_DIR / 'table2_vs_random.csv')

    print(f"{'Size':<5} {'Agent':<18} {'n':<4} {'agent%':<8} {'ref%':<8} "
          f"{'diff%':<10} {'d':<7} {'p_holm':<10} sig")
    for r in sorted(pairwise, key=lambda x: (x['size'], x['agent'])):
        p = r.get('bootstrap_p_holm', 1.0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{r['size']:<5} {r['agent']:<18} {r['n']:<4} "
              f"{100*r['agent_mean']:5.1f}%  {100*r['ref_mean']:5.1f}%  "
              f"{100*r['diff']:+6.1f}%  {r['cohens_d']:+5.2f}  {p:<9.4f} {sig}")

    # ========================================================================
    # Headline Table 3: Reward ablation (K4 decisive test)
    # ========================================================================
    section("Table 3: K4 Reward Ablation (9x9 only)")
    # Reward ablation records have agent_name like "full__Random" or "vanilla__MLP_DQN"
    # after passing through canonical_agent, it becomes "full::Random" etc.
    ablation_records = [r for r in results
                        if '::' in canonical_agent(r.get('agent_name', ''))
                        and r.get('maze_size') == 9]
    if ablation_records:
        print(f"  {len(ablation_records)} ablation records at 9x9")
        # Group by config + agent
        by_cfg_agent = defaultdict(list)
        for r in ablation_records:
            cfg_agent = canonical_agent(r.get('agent_name', ''))
            if r.get('phase') != 'test':
                continue
            seed = r['seed']
            by_cfg_agent[(cfg_agent, seed)].append(bool(r.get('solved', False)))

        per_key: dict = defaultdict(list)
        for (cfg_agent, seed), vals in by_cfg_agent.items():
            if vals:
                per_key[cfg_agent].append(sum(vals) / len(vals))

        print(f"{'Config::Agent':<30} {'n':<4} {'mean':<8} {'95% CI':<22}")
        for ca in sorted(per_key):
            vals = per_key[ca]
            mean, lo, hi = bootstrap_mean_ci(vals, n_resamples=5000)
            print(f"{ca:<30} {len(vals):<4} {100*mean:5.1f}%   "
                  f"[{100*lo:5.1f}, {100*hi:5.1f}]")

        # K4 decisive test: full_MLP vs vanilla_MLP — does MLP recover when reward is vanilla?
        print("\n  K4 decisive pairwise tests (full vs vanilla, paired on seeds):")
        for base_agent in ['Random', 'NoBackRandom', 'FeatureQ', 'MLP_DQN', 'DoubleDQN']:
            full_vals = per_key.get(f'full::{base_agent}', [])
            vanilla_vals = per_key.get(f'vanilla::{base_agent}', [])
            if len(full_vals) >= 3 and len(vanilla_vals) >= 3 and len(full_vals) == len(vanilla_vals):
                diff, lo, hi, p = paired_bootstrap_diff(vanilla_vals, full_vals, n_resamples=5000)
                d = cohens_d(vanilla_vals, full_vals)
                print(f"    {base_agent:<15} full={100*np.mean(full_vals):5.1f}% "
                      f"vanilla={100*np.mean(vanilla_vals):5.1f}% "
                      f"delta={100*diff:+5.1f}% [{100*lo:+5.1f}, {100*hi:+5.1f}] "
                      f"d={d:+5.2f} p={p:.4f}")
    else:
        print("  (no reward ablation data yet)")

    # ========================================================================
    # Headline JSON: for reproduce.py verification
    # ========================================================================
    section("Headline JSON (for reproduce.py pinning)")
    headline: dict = {}
    for r in summary:
        headline.setdefault(r.agent, {})[str(r.size)] = {
            'n_seeds': r.n_seeds,
            'mean_success': r.mean,
            'ci_lo': r.ci_lo,
            'ci_hi': r.ci_hi,
        }
    (OUT_DIR / 'headline.json').write_text(json.dumps(headline, indent=2, sort_keys=True))
    print(f"  Headline written to {OUT_DIR / 'headline.json'}")

    section("DONE")
    print(f"  All outputs in {OUT_DIR}")
    print(f"  Run 'python reproduce.py freeze --out manifest.json' to pin results")


if __name__ == '__main__':
    main()
