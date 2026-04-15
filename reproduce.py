"""reproduce.py — publication-grade reproducibility verifier.

Two modes:
  1. freeze: scan all raw_results/ subdirs, compute SHA-256 of each file, produce
     a manifest. Commit this alongside the paper.
  2. verify: load a pinned manifest, re-hash the files, check they match; then
     recompute headline summary statistics and compare against a pinned JSON.

This is the file reviewers will run. If it exits non-zero, the paper's claimed
numbers are NOT reproducible from the bundled code + data.

Usage:
  python reproduce.py freeze --out manifest.json
  python reproduce.py verify --manifest manifest.json --headline headline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RESULT_ROOTS = [
    Path("insurance_backup/exp_h200"),           # V1 baseline, 503 runs
    Path("raw_results/exp_h200"),                # V1 resume (if any)
    Path("raw_results/exp_oracle_random"),       # Tier 4a: BFS + random variants
    Path("raw_results/exp_reward_ablation_fast"),  # Tier 2 fast K4
    # Tier 2 slow orphaned data moved to attic/exp_reward_ablation_orphan (single-underscore naming, superseded by fast version)
    Path("raw_results/exp_memory_agents"),       # DRQN deterministic
    Path("raw_results/exp_v2_tabular"),          # Phase 3A V2 tabular
    Path("raw_results/exp_capacity_study"),      # Phase 3B capacity study
    Path("raw_results/exp_spiking_dqn"),         # Tier 1 (optional)
    Path("raw_results/exp_budget_matched_sb3"),  # Tier 2b (optional)
    Path("raw_results/exp_minigrid"),            # Tier 3 (optional)
]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def scan_and_hash(roots: list[Path]) -> dict:
    manifest = {'files': {}, 'n_files': 0, 'total_bytes': 0}
    for root in roots:
        if not root.exists():
            print(f"  skip (missing): {root}")
            continue
        for f in sorted(root.rglob("*.json")):
            # Exclude checkpoint.json (live-updated during experiments, not a
            # paper artifact). Only result JSONs and other permanent files.
            if f.name == 'checkpoint.json':
                continue
            rel = f.relative_to(Path.cwd()) if f.is_absolute() else f
            rel_str = str(rel).replace('\\', '/')
            h = sha256_of_file(f)
            size = f.stat().st_size
            manifest['files'][rel_str] = {'sha256': h, 'size': size}
            manifest['n_files'] += 1
            manifest['total_bytes'] += size
    return manifest


def compute_headline(roots: list[Path]) -> dict:
    """Compute the headline numbers the paper relies on: Random vs trained success rates."""
    from collections import defaultdict
    by_key: dict = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.json")):
            if f.name == 'checkpoint.json':
                continue
            try:
                data = json.load(open(f))
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                continue
            if isinstance(data, dict):
                data = [data]
            for r in data:
                if r.get('phase') != 'test':
                    continue
                key = (r.get('agent_name'), r.get('maze_size'), r.get('seed'))
                by_key[key].append(bool(r.get('solved', False)))

    # Per (agent, size) summary
    per_agent: dict = defaultdict(list)
    for (agent, size, seed), vals in by_key.items():
        if not vals:
            continue
        per_agent[(agent, size)].append(sum(vals) / len(vals))

    headline: dict = {}
    for (agent, size), seed_rates in per_agent.items():
        headline.setdefault(agent, {})[str(size)] = {
            'n_seeds': len(seed_rates),
            'mean_success': sum(seed_rates) / len(seed_rates),
            'min': min(seed_rates),
            'max': max(seed_rates),
        }
    return headline


def freeze(args) -> int:
    print("Scanning result directories...")
    manifest = scan_and_hash(RESULT_ROOTS)
    print(f"  hashed {manifest['n_files']} files ({manifest['total_bytes'] / 1e6:.1f} MB)")

    headline = compute_headline(RESULT_ROOTS)
    print(f"  computed headline for {len(headline)} agents")

    out = {
        'manifest': manifest,
        'headline': headline,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {args.out}")
    return 0


def verify(args) -> int:
    pinned = json.loads(Path(args.manifest).read_text())
    pinned_files = pinned['manifest']['files']
    pinned_headline = pinned['headline']

    print("Re-hashing result files...")
    current = scan_and_hash(RESULT_ROOTS)
    current_files = current['files']

    missing = [p for p in pinned_files if p not in current_files]
    extra = [p for p in current_files if p not in pinned_files]
    mismatched = [p for p in pinned_files
                  if p in current_files and current_files[p]['sha256'] != pinned_files[p]['sha256']]

    print(f"  expected {len(pinned_files)} files, found {len(current_files)}")
    print(f"  missing: {len(missing)}, extra: {len(extra)}, mismatched: {len(mismatched)}")
    if missing[:5]:
        print(f"  e.g. missing: {missing[:5]}")
    if mismatched[:5]:
        print(f"  e.g. mismatched: {mismatched[:5]}")

    print("\nRecomputing headline numbers...")
    current_headline = compute_headline(RESULT_ROOTS)

    drift_rows = []
    for agent, sizes in pinned_headline.items():
        for size, pinned_stats in sizes.items():
            cur = current_headline.get(agent, {}).get(size)
            if cur is None:
                drift_rows.append((agent, size, 'MISSING', pinned_stats.get('mean_success'), None))
                continue
            drift = abs(cur['mean_success'] - pinned_stats['mean_success'])
            if drift > 1e-4:
                drift_rows.append((agent, size, f'DRIFT {drift:.4f}',
                                   pinned_stats['mean_success'], cur['mean_success']))

    if drift_rows:
        print(f"\n[WARN] {len(drift_rows)} headline drift events:")
        for r in drift_rows[:20]:
            print(f"  {r[0]:20s} {str(r[1]):5s} {r[2]:12s} pinned={r[3]} current={r[4]}")
    else:
        print("\n[OK] Headline numbers match pinned manifest exactly.")

    n_problems = len(missing) + len(mismatched) + len(drift_rows)
    if n_problems:
        print(f"\nVERIFY FAILED ({n_problems} problems). Exit 1.")
        return 1
    print("\nVERIFY PASSED.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_freeze = sub.add_parser('freeze', help='Create manifest from current results')
    p_freeze.add_argument('--out', default='manifest.json')
    p_freeze.set_defaults(func=freeze)

    p_verify = sub.add_parser('verify', help='Verify current results match manifest')
    p_verify.add_argument('--manifest', required=True)
    p_verify.set_defaults(func=verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
