"""Generate the COMPREHENSIVE results table for the paper.

Pulls data from ALL experiment directories and builds the full agent ladder
showing the gap from oracle/heuristic priors to gradient-based RL.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RAW = ROOT / "raw_results"
OUT = ROOT / "analysis_output" / "comprehensive"


def load_all_test_outcomes() -> dict:
    """Returns: agent -> size -> list of per-seed test rates."""
    by: dict = defaultdict(lambda: defaultdict(list))
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
            seed_rate: dict = defaultdict(list)
            for r in data:
                if r.get("phase") != "test":
                    continue
                a = r.get("agent_name")
                size = r.get("maze_size")
                seed = r.get("seed")
                solved = r.get("solved")
                # Some launchers store success_rate directly
                rate = r.get("success_rate")
                if a and size and seed is not None:
                    if solved is not None:
                        seed_rate[(a, size, seed)].append(bool(solved))
                    elif rate is not None:
                        # Single-record file with rate already aggregated
                        by[a][size].append(float(rate))
            for (a, size, seed), outs in seed_rate.items():
                by[a][size].append(sum(outs) / len(outs))
    return by


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all_test_outcomes()

    # Define the LADDER: oracle -> heuristic -> exploration -> learned random ->
    # tabular -> neural function approx -> exploration-augmented neural
    LADDER = [
        # Tier 1: Perfect knowledge
        ("BFSOracle",        "Tier1: Oracle"),
        # Tier 2: Hand-coded structural priors (no learning, no observation)
        ("WallFollowerLeft", "Tier2: Heuristic"),
        ("WallFollowerRight", "Tier2: Heuristic"),
        ("DFSAgent",         "Tier2: Heuristic"),
        # Tier 3: Random walks (no learning, observation = none)
        ("NoBackRandom",     "Tier3: Random walk"),
        ("LevyRandom_2.0",   "Tier3: Random walk"),
        ("LevyRandom_1.5",   "Tier3: Random walk"),
        ("Random",           "Tier3: Random walk"),
        # Tier 4: Tabular learning
        ("FeatureQ_v2",      "Tier4: Tabular"),
        ("FeatureQ",         "Tier4: Tabular"),
        ("TabularQ_v2",      "Tier4: Tabular"),
        ("TabularQ",         "Tier4: Tabular"),
        # Tier 5: Neural function approximation
        ("MLP_DQN_h32",      "Tier5: Neural"),
        ("MLP_DQN_h64",      "Tier5: Neural"),
        ("MLP_DQN_h128",     "Tier5: Neural"),
        ("MLP_DQN_h256",     "Tier5: Neural"),
        ("DoubleDQN",        "Tier5: Neural"),
        ("DRQN",             "Tier5: Neural+Memory"),
        ("MLP_DQN",          "Tier5: Neural (alias)"),
        ("full__MLP_DQN",    "Tier5: Neural (K4)"),
        ("full__DoubleDQN",  "Tier5: Neural (K4)"),
    ]

    print("=" * 92)
    print("COMPREHENSIVE LADDER OF AGENTS (test success at maze size 9)")
    print("=" * 92)
    print(f"{'Tier':<22} {'Agent':<22} {'mean':>7} {'sd':>5}  {'n_seeds':>7}  {'min%':>5} {'max%':>5}")
    print("-" * 92)

    rows = []
    for agent, tier in LADDER:
        if agent not in data or 9 not in data[agent]:
            continue
        rates = np.array(data[agent][9])
        if len(rates) == 0:
            continue
        rate_pct = 100 * rates
        mean = rate_pct.mean()
        sd = rate_pct.std()
        rows.append({
            "tier": tier, "agent": agent,
            "mean_pct": float(mean), "sd_pct": float(sd),
            "n_seeds": len(rates),
            "min_pct": float(rate_pct.min()),
            "max_pct": float(rate_pct.max()),
        })
        print(f"{tier:<22} {agent:<22} {mean:>6.1f}% {sd:>5.1f}  {len(rates):>7}  "
              f"{int(rate_pct.min()):>5} {int(rate_pct.max()):>5}")

    with open(OUT / "comprehensive_table_9x9.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {OUT / 'comprehensive_table_9x9.json'}")

    # Headline gap analysis
    nb_mean = next(r["mean_pct"] for r in rows if r["agent"] == "NoBackRandom")
    print("\n=== HEADLINE GAPS (vs NoBackRandom 52.2%) ===")
    for r in rows:
        gap = nb_mean - r["mean_pct"]
        sign = "+" if gap > 0 else ""
        print(f"  {r['agent']:<22} : {r['mean_pct']:5.1f}%  ({sign}{gap:5.1f}pp gap)")


if __name__ == "__main__":
    main()
