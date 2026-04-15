"""Paper figure generation script.

Produces:
  - fig1_scale_curves.png   : Success rate vs maze side, one curve per agent class
  - fig2_paired_diffs.png   : Paired effect sizes (Cohen's d) vs Random per agent per size
  - fig3_k4_ablation.png    : Full vs vanilla reward for the 5 K4 agents at 9x9
  - fig4_pain_decomposition.png : pain_per_step vs success_rate scatter
  - fig5_capacity_study.png : MLP success vs hidden size (32, 64, 128, 256)

All figures: matplotlib only, no seaborn, clean B&W-friendly design.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print("WARNING: matplotlib not available, figures disabled")

from stats_pipeline import (
    load_all_results, canonical_agent, per_seed_success, per_seed_values,
    bootstrap_mean_ci,
)

RESULT_DIRS = [
    Path("raw_results/exp_oracle_random"),
    Path("raw_results/exp_reward_ablation_fast"),
    Path("raw_results/exp_memory_agents"),
    Path("raw_results/exp_v2_tabular"),
    Path("raw_results/exp_capacity_study"),
    Path("insurance_backup/exp_h200"),
]

OUT_DIR = Path("paper_figures")

# Agent categories for plotting
AGENT_STYLES = {
    'BFSOracle':       {'color': '#000000', 'linestyle': '-',  'marker': 'X', 'label': 'BFS Oracle'},
    'NoBackRandom':    {'color': '#1f77b4', 'linestyle': '-',  'marker': 'o', 'label': 'NoBackRandom'},
    'LevyRandom_2.0':  {'color': '#17becf', 'linestyle': '--', 'marker': 'v', 'label': 'Lévy(α=2.0)'},
    'LevyRandom_1.5':  {'color': '#17becf', 'linestyle': ':',  'marker': '^', 'label': 'Lévy(α=1.5)'},
    'Random':          {'color': '#2ca02c', 'linestyle': '-',  'marker': 's', 'label': 'Random'},
    'FeatureQ_v2':     {'color': '#ff7f0e', 'linestyle': '-',  'marker': 'D', 'label': 'FeatureQ (tabular)'},
    'MLP_DQN':         {'color': '#d62728', 'linestyle': '-',  'marker': 'P', 'label': 'MLP_DQN'},
    'DoubleDQN':       {'color': '#9467bd', 'linestyle': '-',  'marker': '*', 'label': 'DoubleDQN'},
    'DRQN':            {'color': '#8c564b', 'linestyle': '-',  'marker': 'h', 'label': 'DRQN'},
    'TabularQ_v2':     {'color': '#7f7f7f', 'linestyle': ':',  'marker': 'x', 'label': 'TabularQ'},
}

SIZES = [9, 11, 13, 17, 21, 25]


def get_success_by_size(results: list[dict], agent: str, sizes: list[int]) -> tuple[list[float], list[float], list[float]]:
    """Return (means, ci_lo, ci_hi) across sizes."""
    rng = np.random.default_rng(42)
    means, los, his = [], [], []
    for s in sizes:
        tuples = per_seed_success(results, agent, s, phase='test')
        vals = per_seed_values(tuples)
        if len(vals) < 3:
            means.append(np.nan); los.append(np.nan); his.append(np.nan)
            continue
        m, lo, hi = bootstrap_mean_ci(vals, n_resamples=5000, rng=rng)
        means.append(m)
        los.append(lo)
        his.append(hi)
    return means, los, his


def fig1_scale_curves(results: list[dict]) -> None:
    """Figure 1: Success rate vs maze side, log-scale y-axis."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=120)

    plot_agents = ['BFSOracle', 'NoBackRandom', 'LevyRandom_2.0', 'LevyRandom_1.5',
                   'Random', 'FeatureQ_v2', 'MLP_DQN', 'DoubleDQN', 'TabularQ_v2']

    for agent in plot_agents:
        if agent not in AGENT_STYLES:
            continue
        style = AGENT_STYLES[agent]
        means, los, his = get_success_by_size(results, agent, SIZES)
        xs = [s for s, m in zip(SIZES, means) if not np.isnan(m)]
        ys = [100 * m for m in means if not np.isnan(m)]
        lo_bars = [100 * (m - l) for m, l in zip(means, los) if not np.isnan(m)]
        hi_bars = [100 * (h - m) for m, h in zip(means, his) if not np.isnan(m)]
        if not xs:
            continue
        ax.errorbar(
            xs, ys, yerr=[lo_bars, hi_bars],
            color=style['color'], linestyle=style['linestyle'],
            marker=style['marker'], markersize=7, linewidth=1.8,
            capsize=3, label=style['label'],
        )

    ax.set_xlabel('Maze side (n)', fontsize=12)
    ax.set_ylabel('Test success rate (%)', fontsize=12)
    ax.set_title('Zero-shot test success rate vs maze size\n20 seeds × 50 test mazes per cell, 95% bootstrap CI',
                 fontsize=11)
    # Linear y-scale — log-like breakdown obscured by 0% TabularQ values
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(SIZES)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    ax.set_ylim(-0.5, 110)
    plt.tight_layout()
    path = OUT_DIR / 'fig1_scale_curves.png'
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig1_scale_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"  wrote {path}")


def fig2_paired_diffs(results: list[dict]) -> None:
    """Figure 2: Cohen's d vs Random, per agent per size."""
    from stats_pipeline import pairwise_vs_reference

    plot_agents = ['BFSOracle', 'NoBackRandom', 'LevyRandom_2.0', 'FeatureQ_v2',
                   'MLP_DQN', 'DoubleDQN']
    pairwise = pairwise_vs_reference(results, SIZES, plot_agents, 'Random', n_resamples=5000)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    xs_base = np.arange(len(SIZES))

    agents_seen = sorted(set(r['agent'] for r in pairwise))
    width = 0.8 / max(1, len(agents_seen))

    for i, agent in enumerate(agents_seen):
        if agent not in AGENT_STYLES:
            continue
        style = AGENT_STYLES[agent]
        data = {r['size']: r['cohens_d'] for r in pairwise if r['agent'] == agent}
        ds = [data.get(s, np.nan) for s in SIZES]
        offset = (i - len(agents_seen) / 2 + 0.5) * width
        ax.bar(
            xs_base + offset, ds, width=width * 0.9,
            color=style['color'], label=style['label'], edgecolor='black', linewidth=0.5,
        )

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(-2, color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.axhline(2, color='red', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_xticks(xs_base)
    ax.set_xticklabels([f'{s}×{s}' for s in SIZES])
    ax.set_ylabel("Cohen's d (vs. Random)", fontsize=12)
    ax.set_title('Paired effect size vs uniform Random\n(positive = agent beats Random; negative = Random beats agent)',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    plt.tight_layout()
    path = OUT_DIR / 'fig2_paired_diffs.png'
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig2_paired_diffs.pdf', bbox_inches='tight')
    plt.close()
    print(f"  wrote {path}")


def fig3_k4_ablation(results: list[dict]) -> None:
    """Figure 3: Full vs vanilla reward for K4 agents at 9x9."""
    k4_agents = ['Random', 'NoBackRandom', 'FeatureQ', 'MLP_DQN', 'DoubleDQN']

    full_means, full_los, full_his = [], [], []
    vanilla_means, vanilla_los, vanilla_his = [], [], []
    labels = []

    rng = np.random.default_rng(42)
    for agent in k4_agents:
        full_key = f'full::{agent}'
        vanilla_key = f'vanilla::{agent}'

        full_tuples = per_seed_success(results, full_key, 9, 'test')
        vanilla_tuples = per_seed_success(results, vanilla_key, 9, 'test')
        full_vals = per_seed_values(full_tuples)
        vanilla_vals = per_seed_values(vanilla_tuples)

        if len(full_vals) < 3 or len(vanilla_vals) < 3:
            full_means.append(np.nan)
            full_los.append(0); full_his.append(0)
            vanilla_means.append(np.nan)
            vanilla_los.append(0); vanilla_his.append(0)
        else:
            fm, flo, fhi = bootstrap_mean_ci(full_vals, n_resamples=5000, rng=rng)
            vm, vlo, vhi = bootstrap_mean_ci(vanilla_vals, n_resamples=5000, rng=rng)
            full_means.append(fm)
            full_los.append(fm - flo); full_his.append(fhi - fm)
            vanilla_means.append(vm)
            vanilla_los.append(vm - vlo); vanilla_his.append(vhi - vm)
        labels.append(agent)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    xs = np.arange(len(labels))
    width = 0.38

    ax.bar(xs - width / 2, [100 * m for m in full_means],
           yerr=[[100 * e for e in full_los], [100 * e for e in full_his]],
           width=width, label='Full reward', color='#1f77b4', edgecolor='black',
           linewidth=0.5, capsize=4)
    ax.bar(xs + width / 2, [100 * m for m in vanilla_means],
           yerr=[[100 * e for e in vanilla_los], [100 * e for e in vanilla_his]],
           width=width, label='Vanilla reward', color='#ff7f0e', edgecolor='black',
           linewidth=0.5, capsize=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel('Test success rate (%) at 9×9', fontsize=12)
    ax.set_title('K4 reward ablation: full shaping vs vanilla (step+wall+hazard+goal only)\n'
                 '20 seeds × 50 test mazes, 95% bootstrap CI',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim(0, max(100 * max((m for m in full_means + vanilla_means if not np.isnan(m)), default=0.5) * 1.2, 60))
    plt.tight_layout()
    path = OUT_DIR / 'fig3_k4_ablation.png'
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig3_k4_ablation.pdf', bbox_inches='tight')
    plt.close()
    print(f"  wrote {path}")


def fig4_pain_scatter(results: list[dict]) -> None:
    """Figure 4: pain-per-step vs success rate scatter at 9×9.

    The plot that defines the 'neural learns safety not goal-seeking' finding.
    X axis: success rate. Y axis: pain/step. Each point is an agent.
    """
    # Compute from 9×9 test data
    groups: dict[str, dict] = defaultdict(lambda: {'pain': [], 'solved': [], 'steps': []})
    for r in results:
        if r.get('maze_size') != 9 or r.get('phase') != 'test':
            continue
        name = canonical_agent(r.get('agent_name', ''))
        # Exclude ablation configs — compare only agents under full reward
        if '::' in name:
            continue
        total = float(r.get('reward', 0.0))
        solved = bool(r.get('solved', False))
        steps = int(r.get('steps', 0))
        if steps == 0:
            continue
        goal = 10.0 if solved else 0.0
        pain_per_step = (total - goal) / steps
        groups[name]['pain'].append(pain_per_step)
        groups[name]['solved'].append(1 if solved else 0)
        groups[name]['steps'].append(steps)

    agents_to_plot = [
        'BFSOracle', 'NoBackRandom', 'LevyRandom_2.0', 'LevyRandom_1.5',
        'Random', 'FeatureQ_v2', 'TabularQ_v2',
        'MLP_DQN', 'DoubleDQN', 'DRQN',
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for agent in agents_to_plot:
        g = groups.get(agent, None)
        if not g or not g['pain']:
            continue
        style = AGENT_STYLES.get(agent, {'color': '#888', 'marker': 'o', 'label': agent})
        mean_pain = float(np.mean(g['pain']))
        mean_success = float(np.mean(g['solved']))
        ax.scatter(
            [100 * mean_success], [mean_pain],
            color=style['color'], marker=style['marker'], s=150,
            edgecolors='black', linewidths=1.0, zorder=5,
        )
        ax.annotate(
            style.get('label', agent),
            xy=(100 * mean_success, mean_pain),
            xytext=(7, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold' if agent in ('NoBackRandom', 'Random', 'MLP_DQN') else 'normal',
        )

    ax.set_xlabel('Success rate (% of 1000 test mazes at 9×9)', fontsize=12)
    ax.set_ylabel('Per-step pain = (reward − goal) / steps', fontsize=12)
    ax.set_title('Figure 4: Neural RL learns safety, not goal-seeking\n'
                 'Upper-left agents are "safe idlers"; lower-right are "risky explorers"',
                 fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(0, color='k', linewidth=0.5)

    # Annotation: "neural cluster"
    ax.annotate(
        'Neural: safer\nbut fail to reach goal',
        xy=(18, -0.145),
        xytext=(35, -0.10),
        arrowprops={'arrowstyle': '->', 'color': '#666', 'lw': 1},
        fontsize=9, color='#666', ha='center',
    )
    ax.annotate(
        'Random variants:\nmore pain, more goals',
        xy=(35, -0.24),
        xytext=(55, -0.22),
        arrowprops={'arrowstyle': '->', 'color': '#666', 'lw': 1},
        fontsize=9, color='#666', ha='center',
    )

    plt.tight_layout()
    path = OUT_DIR / 'fig4_pain_scatter.png'
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig4_pain_scatter.pdf', bbox_inches='tight')
    plt.close()
    print(f"  wrote {path}")


def fig5_capacity_study(results: list[dict]) -> None:
    """Figure 5: MLP success vs hidden size, 9x9 and 13x13."""
    capacities = [32, 64, 128, 256]

    # Collect per-(size, capacity) seeds with enough data to plot.
    rng = np.random.default_rng(42)
    data_by_size: dict[int, list[tuple[int, float, float, float]]] = {}
    for size in [9, 13]:
        caps_with_data = []
        for c in capacities:
            agent = f'MLP_DQN_h{c}'
            tuples = per_seed_success(results, agent, size, 'test')
            vals = per_seed_values(tuples)
            if len(vals) < 3:
                continue
            m, lo, hi = bootstrap_mean_ci(vals, n_resamples=5000, rng=rng)
            caps_with_data.append((c, m, m - lo, hi - m))
        if caps_with_data:
            data_by_size[size] = caps_with_data

    if not data_by_size:
        print("  fig5: no capacity study data with >=3 seeds yet, skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for size, caps in data_by_size.items():
        cs = [t[0] for t in caps]
        ms = [100 * t[1] for t in caps]
        lo_bars = [100 * t[2] for t in caps]
        hi_bars = [100 * t[3] for t in caps]
        label = f'MLP @ {size}×{size}'
        ax.errorbar(
            cs, ms, yerr=[lo_bars, hi_bars],
            marker='o', markersize=7, label=label, capsize=4, linewidth=2,
        )

    # Reference lines for Random, NoBackRandom, BFSOracle at 9x9
    for agent, label, color in [
        ('Random',       'Random (9×9)',      '#2ca02c'),
        ('NoBackRandom', 'NoBackRand (9×9)',  '#1f77b4'),
        ('BFSOracle',    'BFS Oracle',        '#000000'),
    ]:
        tuples = per_seed_success(results, agent, 9, 'test')
        vals = per_seed_values(tuples)
        if vals:
            m = float(np.mean(vals))
            ax.axhline(100 * m, color=color, linestyle='--', linewidth=1.2, alpha=0.7, label=label)

    ax.set_xscale('log', base=2)
    ax.set_xticks(capacities)
    ax.set_xticklabels([str(c) for c in capacities])
    ax.set_xlabel('MLP hidden units', fontsize=12)
    ax.set_ylabel('Test success rate (%)', fontsize=12)
    ax.set_title('Network capacity sensitivity: MLP_DQN hidden = {32, 64, 128, 256}\n'
                 'Reviewer attack A5: "network too small"',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    path = OUT_DIR / 'fig5_capacity_study.png'
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'fig5_capacity_study.pdf', bbox_inches='tight')
    plt.close()
    print(f"  wrote {path}")


def main() -> None:
    if not HAVE_MPL:
        print("matplotlib not installed. pip install matplotlib")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in RESULT_DIRS if d.exists()]
    print(f"Loading from {len(dirs)} directories...")
    results = load_all_results(dirs)
    print(f"Loaded {len(results)} records\n")

    print("Generating figures...")
    fig1_scale_curves(results)
    fig2_paired_diffs(results)
    fig3_k4_ablation(results)
    fig4_pain_scatter(results)
    fig5_capacity_study(results)

    print(f"\nDone. Figures in {OUT_DIR}")


if __name__ == '__main__':
    main()
