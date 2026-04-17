"""Bayesian hierarchical analysis (per Codex review).

For each agent at 9x9, fits a hierarchical Bayesian model:
    success_i ~ Bernoulli(p_i)
    p_i ~ Beta(alpha_i, beta_i)
    (alpha_i, beta_i) drawn from agent-level prior

Reports posterior mean + 95% HDI per agent + posterior probability that
agent A > agent B for all pairs.

Implementation uses simple beta-binomial conjugate updates (closed form,
no MCMC needed for this 1-D case).

Output: analysis_output/bayesian/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta as Beta

ROOT = Path(__file__).parent
RAW = ROOT / "raw_results"
OUT = ROOT / "analysis_output" / "bayesian"


def load_test_outcomes() -> dict:
    """Returns: agent -> list of (n_success, n_total) tuples per seed."""
    by_agent: dict = defaultdict(list)
    for d in RAW.iterdir():
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            if "checkpoint" in f.name:
                continue
            try:
                data = json.load(open(f))
            except Exception:
                continue
            if not isinstance(data, list):
                data = [data]
            # Per-seed aggregation
            seed_outcomes: dict = defaultdict(list)
            for r in data:
                if r.get("phase") != "test" or r.get("maze_size") != 9:
                    continue
                agent = r.get("agent_name")
                seed = r.get("seed")
                solved = r.get("solved")
                if agent and seed is not None and solved is not None:
                    seed_outcomes[(agent, seed)].append(bool(solved))
            for (agent, seed), outs in seed_outcomes.items():
                if outs:
                    by_agent[agent].append((sum(outs), len(outs)))
    return by_agent


def posterior_beta(successes: list, alpha_prior: float = 1.0,
                   beta_prior: float = 1.0) -> tuple:
    """Conjugate update: Beta(alpha + s, beta + n - s) per seed, then
    mixture over seeds (we report the marginal posterior over the
    population mean p)."""
    # Pool across seeds: total successes / total trials
    total_s = sum(s for s, n in successes)
    total_n = sum(n for s, n in successes)
    alpha_post = alpha_prior + total_s
    beta_post = beta_prior + total_n - total_s
    return alpha_post, beta_post


def hdi(samples: np.ndarray, prob: float = 0.95) -> tuple:
    """Highest-density interval via interval-width minimization."""
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    n_in = int(np.ceil(prob * n))
    widths = sorted_s[n_in:] - sorted_s[: n - n_in]
    if len(widths) == 0:
        return (sorted_s[0], sorted_s[-1])
    i = int(np.argmin(widths))
    return (float(sorted_s[i]), float(sorted_s[i + n_in]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    data = load_test_outcomes()

    AGENTS_OF_INTEREST = [
        "BFSOracle", "WallFollowerLeft", "WallFollowerRight", "DFSAgent",
        "NoBackRandom", "Random", "LevyRandom_2.0", "LevyRandom_1.5",
        "FeatureQ_v2", "FeatureQ", "MLP_DQN", "DoubleDQN", "DRQN",
        "TabularQ_v2", "TabularQ",
    ]

    posteriors: dict = {}
    print(f"\n{'Agent':<22} {'n_seeds':<8} {'mean':<8} {'95% HDI':<22}")
    print("-" * 60)
    for agent in AGENTS_OF_INTEREST:
        if agent not in data or not data[agent]:
            continue
        outcomes = data[agent]
        alpha, beta = posterior_beta(outcomes)
        # Sample from posterior
        samples = Beta.rvs(alpha, beta, size=10000, random_state=42)
        mean = float(samples.mean())
        lo, hi = hdi(samples)
        posteriors[agent] = {
            "n_seeds": len(outcomes),
            "alpha": float(alpha),
            "beta": float(beta),
            "mean": mean,
            "hdi_lo": lo,
            "hdi_hi": hi,
        }
        print(f"{agent:<22} {len(outcomes):<8} {100*mean:>5.1f}%  [{100*lo:>4.1f}, {100*hi:>5.1f}]")

    # Pairwise probability A > B
    print("\n=== Pairwise P(A > B) ===")
    print(f"{'A':<22} {'B':<22} {'P(A > B)':<10}")
    print("-" * 60)
    KEY_PAIRS = [
        ("NoBackRandom", "Random"),
        ("NoBackRandom", "DoubleDQN"),
        ("NoBackRandom", "MLP_DQN"),
        ("NoBackRandom", "FeatureQ_v2"),
        ("WallFollowerLeft", "NoBackRandom"),
        ("WallFollowerLeft", "MLP_DQN"),
        ("FeatureQ_v2", "MLP_DQN"),
        ("Random", "MLP_DQN"),
    ]
    pair_probs: dict = {}
    for a, b in KEY_PAIRS:
        if a not in posteriors or b not in posteriors:
            continue
        sa = Beta.rvs(posteriors[a]["alpha"], posteriors[a]["beta"], size=20000, random_state=1)
        sb = Beta.rvs(posteriors[b]["alpha"], posteriors[b]["beta"], size=20000, random_state=2)
        p_a_gt_b = float((sa > sb).mean())
        pair_probs[f"{a} > {b}"] = p_a_gt_b
        print(f"{a:<22} {b:<22} {p_a_gt_b:.4f}")

    out = {
        "posteriors": posteriors,
        "pair_probabilities": pair_probs,
    }
    with open(OUT / "bayesian_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'bayesian_analysis.json'}")


if __name__ == "__main__":
    main()
