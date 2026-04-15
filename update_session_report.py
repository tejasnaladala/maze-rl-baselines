"""Regenerate the SESSION_REPORT.md tables from current raw_results.

Safe to run repeatedly — idempotent. Reads all result dirs, computes canonical
stats, and writes Markdown tables into SESSION_REPORT_tables.md which can be
pasted into the main report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from stats_pipeline import (
    load_all_results, canonical_agent, per_seed_success, per_seed_values,
    bootstrap_mean_ci, paired_bootstrap_diff, cohens_d,
    summary_table, pairwise_vs_reference,
)

RESULT_DIRS = [
    Path("insurance_backup/exp_h200"),
    Path("raw_results/exp_oracle_random"),
    Path("raw_results/exp_reward_ablation_fast"),
    Path("raw_results/exp_memory_agents"),
    Path("raw_results/exp_spiking_dqn"),
    Path("raw_results/exp_budget_matched_sb3"),
    Path("raw_results/exp_minigrid"),
]

OUT = Path("SESSION_REPORT_tables.md")


def format_ci(mean: float, lo: float, hi: float) -> str:
    return f"{100*mean:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def main() -> None:
    dirs = [d for d in RESULT_DIRS if d.exists()]
    results = load_all_results(dirs)
    print(f"Loaded {len(results)} records from {len(dirs)} dirs")

    lines = [
        f"# SESSION_REPORT — auto-regenerated tables",
        f"",
        f"Loaded {len(results)} records from {len(dirs)} directories.",
        f"",
    ]

    # Unified main table
    agents = sorted(set(canonical_agent(r.get('agent_name', '')) for r in results if r.get('agent_name')))
    sizes = sorted(set(r.get('maze_size', 0) for r in results if r.get('maze_size')))
    sizes = [s for s in sizes if s > 0]
    primary_agents = [a for a in agents if '::' not in a]

    summary = summary_table(results, sizes, primary_agents, n_resamples=5000)

    # Markdown main table
    lines.append("## Main table (all agents × all sizes)")
    lines.append("")
    lines.append("| Agent | " + " | ".join(f"{s}×{s}" for s in sizes) + " |")
    lines.append("|" + "---|" * (len(sizes) + 1))
    by_agent_size: dict[tuple, Any] = {(r.agent, r.size): r for r in summary}
    rank_agents = ['BFSOracle', 'NoBackRandom', 'LevyRandom_2.0', 'LevyRandom_1.5', 'Random',
                   'FeatureQ', 'MLP_DQN', 'DoubleDQN', 'SpikingDQN', 'DRQN', 'TabularQ']
    for a in rank_agents:
        cells = [a]
        for s in sizes:
            r = by_agent_size.get((a, s))
            if r is None or r.n_seeds < 3:
                cells.append("—")
            else:
                cells.append(f"{100*r.mean:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Pairwise vs Random
    pairwise = pairwise_vs_reference(results, sizes, primary_agents, 'Random', n_resamples=5000)
    lines.append("## Pairwise vs Random (Holm-Bonferroni corrected)")
    lines.append("")
    lines.append("| Size | Agent | n | agent% | random% | delta | Cohen's d | p_holm | sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(pairwise, key=lambda x: (x['size'], x['agent'])):
        p = r.get('bootstrap_p_holm', 1.0)
        sig = '\\*\\*\\*' if p < 0.001 else '\\*\\*' if p < 0.01 else '\\*' if p < 0.05 else ''
        lines.append(
            f"| {r['size']} | {r['agent']} | {r['n']} | "
            f"{100*r['agent_mean']:.1f}% | {100*r['ref_mean']:.1f}% | "
            f"{100*r['diff']:+.1f}% | {r['cohens_d']:+.2f} | {p:.4f} | {sig} |"
        )
    lines.append("")

    # Reward ablation K4 table (if present)
    ablation_agents = [a for a in agents if '::' in a]
    if ablation_agents:
        lines.append("## K4 Reward Ablation (reward_ablation_fast @ 9×9)")
        lines.append("")
        per_key: dict[str, list[float]] = defaultdict(list)
        for r in results:
            name = canonical_agent(r.get('agent_name', ''))
            if '::' not in name or r.get('maze_size') != 9 or r.get('phase') != 'test':
                continue
            seed = r['seed']
            per_key_entry = (name, seed)
            per_key[per_key_entry[0]].append(None)  # placeholder
        # Properly aggregate: (name, seed) -> list of solved
        by_name_seed: dict[tuple, list[bool]] = defaultdict(list)
        for r in results:
            name = canonical_agent(r.get('agent_name', ''))
            if '::' not in name or r.get('maze_size') != 9 or r.get('phase') != 'test':
                continue
            by_name_seed[(name, r['seed'])].append(bool(r.get('solved', False)))
        per_name: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for (name, seed), vals in by_name_seed.items():
            per_name[name].append((seed, sum(vals) / len(vals)))

        lines.append("| Config :: Agent | n seeds | mean success | 95% CI |")
        lines.append("|---|---|---|---|")
        for name in sorted(per_name):
            vals = [v for _, v in sorted(per_name[name])]
            if len(vals) < 3:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, n_resamples=5000)
            lines.append(f"| {name} | {len(vals)} | {100*mean:.1f}% | [{100*lo:.1f}, {100*hi:.1f}] |")
        lines.append("")

        lines.append("### K4 paired tests: vanilla − full per agent (Δ > 0 means vanilla helps)")
        lines.append("")
        lines.append("| Agent | full mean | vanilla mean | Δ | Cohen's d | p |")
        lines.append("|---|---|---|---|---|---|")
        for base in ['Random', 'NoBackRandom', 'FeatureQ', 'MLP_DQN', 'DoubleDQN']:
            full = dict(sorted(per_name.get(f'full::{base}', [])))
            vanilla = dict(sorted(per_name.get(f'vanilla::{base}', [])))
            common = sorted(set(full) & set(vanilla))
            if len(common) < 3:
                continue
            f_vals = [full[s] for s in common]
            v_vals = [vanilla[s] for s in common]
            diff, lo, hi, p = paired_bootstrap_diff(v_vals, f_vals, n_resamples=5000)
            d = cohens_d(v_vals, f_vals)
            lines.append(
                f"| {base} | {100*sum(f_vals)/len(f_vals):.1f}% | {100*sum(v_vals)/len(v_vals):.1f}% | "
                f"{100*diff:+.1f}% | {d:+.2f} | {p:.4f} |"
            )
        lines.append("")

    # DRQN summary (if present)
    drqn_summary = summary_table(results, [9, 11, 13, 17, 21, 25], ['DRQN'], n_resamples=5000)
    if drqn_summary:
        lines.append("## DRQN (partial-observability control)")
        lines.append("")
        lines.append("| Size | n seeds | mean success | 95% CI |")
        lines.append("|---|---|---|---|")
        for r in drqn_summary:
            lines.append(f"| {r.size} | {r.n_seeds} | {100*r.mean:.1f}% | [{100*r.ci_lo:.1f}, {100*r.ci_hi:.1f}] |")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
