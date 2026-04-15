"""progress_monitor.py — one-shot status of all running experiments.

Scans raw_results/, shows progress per tier, ETA estimate, and any failures.
Usage: python progress_monitor.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).parent / 'raw_results'

TIERS = {
    'exp_h200 (Tier 0)':                  600,
    'exp_oracle_random (Tier 4a)':        600,
    'exp_reward_ablation_fast (T2fast)':  200,
    'exp_reward_ablation (Tier 2 slow)':  1500,
    'exp_memory_agents (Tier 4b)':        20,
    'exp_memory_agents_nondet (4b-nd)':   20,
    'exp_v2_tabular (Phase 3A)':          240,
    'exp_capacity_study (Phase 3B)':      160,
    'exp_spiking_dqn (Tier 1)':           120,
    'exp_budget_matched_sb3 (T2b)':       540,
    'exp_minigrid (Tier 3)':              400,
}


def scan_tier(dirname: str, expected: int) -> dict:
    d = ROOT / dirname.split(' ')[0]
    if not d.exists():
        return {'count': 0, 'expected': expected, 'checkpoint': 0, 'last_mtime': None}

    json_files = [f for f in d.glob("*.json") if f.name != 'checkpoint.json']
    count = len(json_files)
    last_mtime = max((f.stat().st_mtime for f in json_files), default=None)

    ckpt_path = d / 'checkpoint.json'
    checkpoint_count = 0
    if ckpt_path.exists():
        try:
            checkpoint_count = len(json.loads(ckpt_path.read_text()))
        except Exception:
            pass

    return {
        'count': count,
        'expected': expected,
        'checkpoint': checkpoint_count,
        'last_mtime': last_mtime,
    }


def human_time(ts: float | None) -> str:
    if ts is None:
        return '—'
    elapsed = time.time() - ts
    if elapsed < 60: return f"{elapsed:.0f}s ago"
    if elapsed < 3600: return f"{elapsed/60:.0f}m ago"
    if elapsed < 86400: return f"{elapsed/3600:.1f}h ago"
    return f"{elapsed/86400:.1f}d ago"


def main() -> None:
    print(f"\n=== Engram experiment progress — {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    print(f"{'Tier':<35} {'Files':<12} {'Ckpt':<10} {'Last write':<15} Status")
    print("-" * 90)

    total_done = 0
    total_expected = 0
    for tier_label, expected in TIERS.items():
        info = scan_tier(tier_label, expected)
        done = info['count']
        mtime = info['last_mtime']
        total_done += done
        total_expected += expected

        if done == 0:
            status = 'not started'
        elif done < expected:
            elapsed_since = (time.time() - mtime) if mtime else 9999
            if elapsed_since < 120:
                status = 'RUNNING'
            else:
                status = 'IDLE/DEAD'
        else:
            status = 'COMPLETE'

        pct = 100 * done / expected if expected else 0
        print(f"{tier_label:<35} {done}/{expected:<7} {info['checkpoint']:<10} "
              f"{human_time(mtime):<15} {status} ({pct:.0f}%)")

    print("-" * 90)
    print(f"{'TOTAL':<35} {total_done}/{total_expected}")
    print()


if __name__ == '__main__':
    main()
