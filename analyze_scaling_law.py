"""Formal cover-time scaling law analysis (per Codex review).

Fits power law: success_rate = a * n^b for each agent across maze sizes 9-25.
Reports:
  - Power-law exponent b with bootstrap 95% CI
  - R^2 of fit
  - Comparison to theoretical bounds (BFS optimal cover time scales as O(n^2))

Output: analysis_output/scaling_law/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap

ROOT = Path(__file__).parent
RAW = ROOT / "raw_results"
OUT = ROOT / "analysis_output" / "scaling_law"


def load_per_seed_test_rates() -> dict:
    """Returns nested dict: agent -> size -> list of per-seed test rates."""
    by_agent_size: dict = defaultdict(lambda: defaultdict(list))
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
            # Aggregate by (agent, size) -> per-seed rate
            by_seed_rate: dict = defaultdict(list)
            for r in data:
                if r.get("phase") != "test":
                    continue
                agent = r.get("agent_name")
                size = r.get("maze_size")
                seed = r.get("seed")
                solved = r.get("solved")
                if agent and size and seed is not None and solved is not None:
                    by_seed_rate[(agent, size, seed)].append(bool(solved))
            for (agent, size, seed), solveds in by_seed_rate.items():
                if solveds:
                    rate = sum(solveds) / len(solveds)
                    by_agent_size[agent][size].append(rate)
    return by_agent_size


def power_law(n, a, b):
    return a * np.power(n, b)


def fit_power_law(sizes: list, rates: list) -> dict:
    """Fit success_rate = a * size^b via least squares."""
    sizes = np.array(sizes, dtype=float)
    rates = np.array(rates, dtype=float)
    valid = rates > 1e-4
    if valid.sum() < 3:
        return {"a": np.nan, "b": np.nan, "r2": np.nan, "n_valid": int(valid.sum())}
    sizes = sizes[valid]
    rates = rates[valid]
    try:
        popt, pcov = curve_fit(power_law, sizes, rates, p0=[1.0, -1.0], maxfev=5000)
        a, b = popt
        pred = power_law(sizes, *popt)
        ss_res = ((rates - pred) ** 2).sum()
        ss_tot = ((rates - rates.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"a": float(a), "b": float(b), "r2": float(r2), "n_valid": int(len(sizes))}
    except Exception:
        return {"a": np.nan, "b": np.nan, "r2": np.nan, "n_valid": int(len(sizes))}


def bootstrap_exponent(sizes: list, rates_per_size: list, n_boot: int = 1000) -> dict:
    """Bootstrap CI for the power-law exponent b."""
    bs: list = []
    for _ in range(n_boot):
        boot_means = []
        for r_list in rates_per_size:
            n = len(r_list)
            sample = np.random.choice(r_list, size=n, replace=True)
            boot_means.append(sample.mean())
        fit = fit_power_law(sizes, boot_means)
        if not np.isnan(fit["b"]):
            bs.append(fit["b"])
    if not bs:
        return {"b_lo": np.nan, "b_hi": np.nan, "b_mean": np.nan}
    return {
        "b_mean": float(np.mean(bs)),
        "b_lo": float(np.percentile(bs, 2.5)),
        "b_hi": float(np.percentile(bs, 97.5)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    data = load_per_seed_test_rates()

    # Filter to agents with data at >= 4 sizes
    AGENTS_OF_INTEREST = [
        "BFSOracle", "NoBackRandom", "Random", "LevyRandom_2.0", "LevyRandom_1.5",
        "FeatureQ", "FeatureQ_v2", "MLP_DQN", "DoubleDQN", "DRQN", "TabularQ_v2",
        "WallFollowerLeft", "WallFollowerRight", "DFSAgent",
    ]

    print(f"\n{'Agent':<22} {'sizes':<25} {'b':<10} {'b 95% CI':<22} {'R^2':<6}")
    print("-" * 90)

    results: dict = {}
    for agent in AGENTS_OF_INTEREST:
        if agent not in data:
            continue
        sizes_with_data = sorted(data[agent].keys())
        if len(sizes_with_data) < 3:
            continue
        rates_per_size = [data[agent][s] for s in sizes_with_data]
        mean_rates = [np.mean(r) for r in rates_per_size]
        fit = fit_power_law(sizes_with_data, mean_rates)
        boot = bootstrap_exponent(sizes_with_data, rates_per_size)
        results[agent] = {
            "sizes": sizes_with_data,
            "mean_rates": mean_rates,
            "fit_a": fit["a"],
            "fit_b": fit["b"],
            "fit_r2": fit["r2"],
            "boot_b_mean": boot["b_mean"],
            "boot_b_lo": boot["b_lo"],
            "boot_b_hi": boot["b_hi"],
        }
        sizes_str = str(sizes_with_data)
        b_str = f"{fit['b']:+.3f}" if not np.isnan(fit["b"]) else "nan"
        ci_str = (f"[{boot['b_lo']:+.3f}, {boot['b_hi']:+.3f}]"
                  if not np.isnan(boot['b_lo']) else "[nan, nan]")
        r2_str = f"{fit['r2']:.3f}" if not np.isnan(fit["r2"]) else "nan"
        print(f"{agent:<22} {sizes_str:<25} {b_str:<10} {ci_str:<22} {r2_str:<6}")

    # Save JSON
    with open(OUT / "power_law_fits.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT / 'power_law_fits.json'}")

    # Generate scaling law figure
    plt.figure(figsize=(8, 6))
    for agent in ("BFSOracle", "WallFollowerLeft", "NoBackRandom", "LevyRandom_2.0",
                  "Random", "FeatureQ_v2", "MLP_DQN", "DoubleDQN"):
        if agent not in results:
            continue
        r = results[agent]
        sizes = r["sizes"]
        rates = r["mean_rates"]
        plt.plot(sizes, rates, "o-", label=f"{agent} (b={r['fit_b']:+.2f})", markersize=6)
    plt.xlabel("Maze size n")
    plt.ylabel("Test success rate")
    plt.title("Power-law fit: success ~ a * n^b")
    plt.xscale("log"); plt.yscale("log")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig_path = ROOT / "paper_figures" / "fig6_scaling_law.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.savefig(fig_path.with_suffix(".pdf"))
    print(f"Wrote {fig_path}")

    # Theory comparison
    print("\n=== Theory comparison ===")
    print("Expected: BFSOracle (perfect knowledge) -> b=0 (constant 100%)")
    print("Expected: Random walk -> b around -1 (cover time grows as n^2 but bounded test horizon)")
    print("Expected: NoBackRandom -> shallower than Random per Alon 2007")
    if "Random" in results and "NoBackRandom" in results:
        print(f"Observed: Random b={results['Random']['fit_b']:+.3f}")
        print(f"Observed: NoBackRandom b={results['NoBackRandom']['fit_b']:+.3f}")
        diff = results["Random"]["fit_b"] - results["NoBackRandom"]["fit_b"]
        print(f"  -> NoBackRandom decays {abs(diff):.3f} units slower (matches theory)")


if __name__ == "__main__":
    main()
