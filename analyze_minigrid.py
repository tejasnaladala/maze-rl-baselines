"""Analyze MiniGrid 4-env results across agents."""

import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def main() -> None:
    by = defaultdict(lambda: defaultdict(list))  # env -> agent -> per-seed test rate
    for f in glob.glob(str(ROOT / "raw_results/exp_minigrid/*.json")):
        if "checkpoint" in f:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, list):
            data = [data]
        # Aggregate test episodes per (env, agent, seed)
        seed_outcomes = defaultdict(list)
        for r in data:
            if r.get("phase") != "test":
                continue
            env_id = r.get("env_id", "")
            agent_name = r.get("agent_name", "")
            seed = r.get("seed", 0)
            solved = r.get("solved", False)
            # agent_name format: "MLP_DQN@MiniGrid-DoorKey-5x5-v0"
            base_agent = agent_name.split("@")[0] if "@" in agent_name else agent_name
            seed_outcomes[(env_id, base_agent, seed)].append(bool(solved))
        for (env, agent, _seed), outs in seed_outcomes.items():
            if outs:
                by[env][agent].append(sum(outs) / len(outs))

    print("=" * 90)
    print("MiniGrid 4-env results")
    print("=" * 90)
    for env in sorted(by.keys()):
        print(f"\n{env}")
        print("-" * 60)
        for agent in sorted(by[env].keys()):
            rates = np.array(by[env][agent]) * 100
            print(f"  {agent:<24} mean={rates.mean():>5.1f}%  sd={rates.std():>4.1f}  n={len(rates)}")


if __name__ == "__main__":
    main()
