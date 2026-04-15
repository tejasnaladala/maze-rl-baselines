"""Statistical analysis pipeline for Engram publication-grade results.

Reads raw JSON result files from any experiment tier and produces:
  - Per-agent / per-size / per-config success-rate tables with bootstrap 95% CIs
  - Paired-bootstrap significance tests (Random vs each trained agent)
  - Mann-Whitney U with Cohen's d effect size
  - Holm-Bonferroni correction for family-wise error rate
  - Power analysis (given observed effects, what n do we need?)
  - Export to CSV and LaTeX for paper tables

Zero dependence on specific experiment structure — operates on result dicts
with keys {agent_name, maze_size, seed, phase, solved}.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy import stats as sstats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("WARNING: scipy not available, Mann-Whitney U and power analysis disabled")


# ============================================================
# LOAD RESULTS
# ============================================================

def load_all_results(result_dirs: list[Path]) -> list[dict]:
    """Load all per-run JSON files from one or more result directories."""
    all_results: list[dict] = []
    for d in result_dirs:
        if not d.exists():
            print(f"WARNING: {d} not found, skipping")
            continue
        for f in sorted(d.glob("*.json")):
            if f.name == 'checkpoint.json':
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    all_results.extend(data)
                elif isinstance(data, dict):
                    all_results.append(data)
            except Exception as e:
                print(f"  skip {f.name}: {e}")
    return all_results


# FIX Codex-S2: harmonize alternate spellings of the same agent. Without this,
# a mid-experiment rename or a launcher-to-launcher naming drift makes stats treat
# the same agent as two different ones.
AGENT_ALIAS = {
    'NoBackRand':        'NoBackRandom',
    'NoBackRandom':      'NoBackRandom',
    'NoBacktrackRandom': 'NoBackRandom',
    'Levy':              'LevyRandom_1.5',
    'LevyRandom_1.5':    'LevyRandom_1.5',
    'LevyRandom_2.0':    'LevyRandom_2.0',
}


def canonical_agent(name: str) -> str:
    """Strip launcher prefixes and alias-normalize an agent name.
    Examples:
      'full__MLP_DQN'  -> 'full::MLP_DQN' (reward ablation fast: double-underscore sep)
      'full_MLP_DQN'   -> 'full::MLP_DQN' (old single-underscore form)
      'NoBackRand'     -> 'NoBackRandom'
      'MLP_DQN'        -> 'MLP_DQN'
    """
    # reward-ablation launchers prepend '{config}__' or '{config}_' to agent name
    for cfg in ('full', 'no_visit', 'no_shape', 'vanilla_noham', 'vanilla'):
        for sep in ('__', '_'):
            prefix = cfg + sep
            if name.startswith(prefix):
                base = name[len(prefix):]
                return f"{cfg}::{AGENT_ALIAS.get(base, base)}"
    return AGENT_ALIAS.get(name, name)


def per_seed_success(results: list[dict], agent: str, size: int, phase: str = 'test') -> list[tuple[int, float]]:
    """Return per-seed test success rates as sorted (seed, rate) tuples.

    FIX Codex-S2: return SORTED (seed, rate) tuples so callers can explicitly align
    on common seeds. The old API returned values in dict-insertion order, which
    meant paired bootstrap could silently compare mismatched pairs.
    """
    by_seed: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        if canonical_agent(r.get('agent_name', '')) != agent:
            continue
        if r.get('maze_size') != size or r.get('phase') != phase:
            continue
        by_seed[r['seed']].append(bool(r.get('solved', False)))
    return sorted([(s, sum(v) / len(v)) for s, v in by_seed.items() if v])


def per_seed_values(seed_rate_tuples: list[tuple[int, float]]) -> list[float]:
    """Extract just the success rates, for functions that need a list of floats."""
    return [r for _, r in seed_rate_tuples]


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_mean_ci(values: list[float], n_resamples: int = 10000,
                      confidence: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean. Returns (mean, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(values) == 0:
        return (float('nan'), float('nan'), float('nan'))
    arr = np.asarray(values, dtype=np.float64)
    means = np.empty(n_resamples, dtype=np.float64)
    n = len(arr)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    alpha = (1 - confidence) / 2
    return (float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def paired_bootstrap_diff(a: list[float], b: list[float], n_resamples: int = 10000,
                          rng: np.random.Generator | None = None) -> tuple[float, float, float, float]:
    """Paired bootstrap on a - b. Returns (mean_diff, ci_lo, ci_hi, p_value).
    p_value is the two-sided proportion of resamples where sign(resample_mean) != sign(obs_mean)."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap requires equal-length samples: {len(a)} vs {len(b)}")
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0)
    obs = float(diffs.mean())
    resamples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resamples[i] = diffs[idx].mean()
    ci_lo = float(np.quantile(resamples, 0.025))
    ci_hi = float(np.quantile(resamples, 0.975))
    # two-sided p: proportion of resamples on opposite side of 0 from observation
    if obs >= 0:
        p = 2 * float((resamples <= 0).mean())
    else:
        p = 2 * float((resamples >= 0).mean())
    p = min(1.0, p)
    return obs, ci_lo, ci_hi, p


# ============================================================
# EFFECT SIZES
# ============================================================

def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d with pooled standard deviation. Negative d means a < b."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0
    n1, n2 = len(a_arr), len(b_arr)
    v1, v2 = a_arr.var(ddof=1), b_arr.var(ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled_sd)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


# ============================================================
# HYPOTHESIS TESTS
# ============================================================

def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Returns (U statistic, two-sided p-value). Falls back to NaN if scipy missing."""
    if not HAVE_SCIPY:
        return (float('nan'), float('nan'))
    if len(a) < 1 or len(b) < 1:
        return (float('nan'), float('nan'))
    res = sstats.mannwhitneyu(a, b, alternative='two-sided')
    return (float(res.statistic), float(res.pvalue))


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [1.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        multiplier = n - rank
        adj = min(1.0, pvalues[idx] * multiplier)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


# ============================================================
# POWER ANALYSIS
# ============================================================

def required_sample_size(effect_h: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Required n per group for two-sample proportion test at given Cohen's h."""
    if not HAVE_SCIPY:
        return -1
    if abs(effect_h) < 1e-6:
        return 10**9  # impossible to detect zero effect
    z_alpha = sstats.norm.ppf(1 - alpha / 2)
    z_power = sstats.norm.ppf(power)
    n = ((z_alpha + z_power) / effect_h) ** 2
    return int(math.ceil(n))


# ============================================================
# SUMMARY TABLES
# ============================================================

@dataclass
class AgentSummary:
    agent: str
    size: int
    n_seeds: int
    mean: float
    ci_lo: float
    ci_hi: float
    std: float
    min: float
    max: float


def summary_table(results: list[dict], sizes: list[int], agents: list[str],
                  phase: str = 'test', n_resamples: int = 10000) -> list[AgentSummary]:
    rng = np.random.default_rng(42)
    out: list[AgentSummary] = []
    for size in sizes:
        for agent in agents:
            tuples = per_seed_success(results, agent, size, phase)
            vals = per_seed_values(tuples)
            if not vals:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, n_resamples=n_resamples, rng=rng)
            out.append(AgentSummary(
                agent=agent, size=size, n_seeds=len(vals),
                mean=mean, ci_lo=lo, ci_hi=hi,
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                min=float(min(vals)), max=float(max(vals)),
            ))
    return out


def pairwise_vs_reference(results: list[dict], sizes: list[int], agents: list[str],
                          reference: str, phase: str = 'test',
                          n_resamples: int = 10000) -> list[dict]:
    """All agents vs a reference agent, paired over COMMON seeds.

    FIX Codex-S2: explicitly intersect seeds before pairing. The old code relied
    on list-length equality + dict-insertion order, which silently paired
    mismatched seeds on any dataset with incomplete runs.
    """
    out: list[dict] = []
    pvals_raw: list[float] = []
    meta: list[dict] = []
    for size in sizes:
        ref_tuples = per_seed_success(results, reference, size, phase)
        if not ref_tuples:
            continue
        ref_map = dict(ref_tuples)

        for agent in agents:
            if agent == reference:
                continue
            a_tuples = per_seed_success(results, agent, size, phase)
            if not a_tuples:
                continue
            a_map = dict(a_tuples)

            common = sorted(set(a_map) & set(ref_map))
            if len(common) < 3:
                print(f"  SKIP {agent} vs {reference} @ size={size}: "
                      f"only {len(common)} common seeds (need >= 3)")
                continue
            if len(common) != len(a_tuples) or len(common) != len(ref_tuples):
                print(f"  NOTE {agent} vs {reference} @ size={size}: "
                      f"using {len(common)} common seeds "
                      f"(agent has {len(a_tuples)}, ref has {len(ref_tuples)})")

            a_vals = [a_map[s] for s in common]
            ref_vals = [ref_map[s] for s in common]

            diff, lo, hi, p = paired_bootstrap_diff(a_vals, ref_vals, n_resamples=n_resamples)
            d = cohens_d(a_vals, ref_vals)
            h = cohens_h(np.mean(a_vals), np.mean(ref_vals))
            u, mwu_p = mann_whitney_u(a_vals, ref_vals)
            row = {
                'size': size, 'agent': agent, 'reference': reference,
                'n': len(common),
                'agent_mean': float(np.mean(a_vals)),
                'ref_mean': float(np.mean(ref_vals)),
                'diff': diff, 'ci_lo': lo, 'ci_hi': hi,
                'bootstrap_p': p, 'mann_whitney_u': u, 'mwu_p': mwu_p,
                'cohens_d': d, 'cohens_h': h,
            }
            out.append(row)
            pvals_raw.append(p)
            meta.append(row)
    adjusted = holm_bonferroni(pvals_raw)
    for row, ap in zip(meta, adjusted):
        row['bootstrap_p_holm'] = ap
    return out


# ============================================================
# EXPORT
# ============================================================

def export_csv(rows: list[dict] | list[AgentSummary], path: Path) -> None:
    import csv
    if not rows:
        return
    if isinstance(rows[0], AgentSummary):
        fieldnames = list(asdict(rows[0]).keys())
        data_rows = [asdict(r) for r in rows]
    else:
        fieldnames = list(rows[0].keys())
        data_rows = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in data_rows:
            w.writerow(r)


def export_latex_summary(rows: list[AgentSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(r.size for r in rows))
    agents = sorted(set(r.agent for r in rows))
    table = {(r.agent, r.size): r for r in rows}
    lines = [
        "\\begin{tabular}{l" + "r" * len(sizes) + "}",
        "\\toprule",
        "Agent & " + " & ".join(f"{s}$\\times${s}" for s in sizes) + " \\\\",
        "\\midrule",
    ]
    for agent in agents:
        cells = [agent]
        for s in sizes:
            r = table.get((agent, s))
            if r is None:
                cells.append("--")
            else:
                cells.append(f"{100*r.mean:.1f}~[{100*r.ci_lo:.1f}, {100*r.ci_hi:.1f}]")
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines))


# ============================================================
# CLI
# ============================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True, help='Result directories')
    parser.add_argument('--out', default='analysis_output', help='Output directory')
    parser.add_argument('--reference', default='Random', help='Reference agent for pairwise')
    parser.add_argument('--phase', default='test')
    parser.add_argument('--n_resamples', type=int, default=10000)
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dirs]
    out_dir = Path(args.out)
    print(f"Loading from {len(dirs)} directories...")
    results = load_all_results(dirs)
    print(f"Loaded {len(results)} records")

    # FIX Codex-nit: enumerate canonical (alias-normalized) agent names so that
    # e.g. 'NoBackRand' and 'NoBackRandom' from two different launchers are treated
    # as the same agent, and so reward-ablation's 'full_MLP_DQN' is split into
    # '{config}::{agent}' buckets.
    agents = sorted(set(canonical_agent(r.get('agent_name', '')) for r in results if r.get('agent_name')))
    sizes = sorted(set(r.get('maze_size') for r in results if r.get('maze_size')))
    print(f"Agents: {agents}")
    print(f"Sizes: {sizes}")

    summary = summary_table(results, sizes, agents, phase=args.phase, n_resamples=args.n_resamples)
    export_csv(summary, out_dir / 'summary.csv')
    export_latex_summary(summary, out_dir / 'summary.tex')
    print(f"Summary: {len(summary)} rows -> {out_dir / 'summary.csv'}")

    pairwise = pairwise_vs_reference(results, sizes, agents, args.reference,
                                     phase=args.phase, n_resamples=args.n_resamples)
    export_csv(pairwise, out_dir / f'pairwise_vs_{args.reference}.csv')
    print(f"Pairwise vs {args.reference}: {len(pairwise)} rows -> {out_dir}/pairwise_vs_{args.reference}.csv")

    sig = sum(1 for r in pairwise if r.get('bootstrap_p_holm', 1.0) < 0.05)
    print(f"\nSignificant (Holm-corrected p < 0.05) differences vs {args.reference}: {sig}/{len(pairwise)}")


if __name__ == '__main__':
    main()
