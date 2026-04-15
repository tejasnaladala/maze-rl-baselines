"""finalize.py — runs after all experiments complete.

Does the full finalization:
  1. Regenerates stats_pipeline output for the current data
  2. Regenerates all figures
  3. Updates SESSION_REPORT_tables.md
  4. Runs phase4 attack matrix
  5. Runs reward decomposition
  6. Runs cover time analysis
  7. Freezes reproduce.py manifest
  8. Prints a final summary

Usage: python finalize.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], name: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  Running: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED with exit code {result.returncode}")
        print(f"  STDERR: {result.stderr[-500:]}")
        return False
    # Print last 15 lines of stdout
    lines = result.stdout.strip().split('\n')
    for line in lines[-15:]:
        print(f"  {line}")
    print(f"  OK ({len(lines)} lines output)")
    return True


def main() -> int:
    steps = [
        (['python', 'final_analysis.py'], "final_analysis.py (tables 1-3)"),
        (['python', 'generate_figures.py'], "generate_figures.py (figs 1-5)"),
        (['python', 'update_session_report.py'], "update_session_report.py (auto tables)"),
        (['python', 'phase4_reviewer_attacks.py'], "phase4_reviewer_attacks.py (attack matrix)"),
        (['python', 'reward_decomposition.py'], "reward_decomposition.py (A8 diagnostic)"),
        (['python', 'cover_time_analysis.py'], "cover_time_analysis.py (A9 diagnostic)"),
        (['python', 'reproduce.py', 'freeze', '--out', 'manifest_final.json'],
            "reproduce.py freeze"),
    ]

    results = []
    for cmd, name in steps:
        ok = run(cmd, name)
        results.append((name, ok))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  FINALIZATION SUMMARY")
    print(f"{'=' * 70}")
    for name, ok in results:
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {name}")

    failed = sum(1 for _, ok in results if not ok)
    if failed:
        print(f"\n{failed}/{len(results)} steps FAILED")
        return 1

    # Print the top-level deliverables
    print(f"\n{'=' * 70}")
    print(f"  TOP-LEVEL DELIVERABLES")
    print(f"{'=' * 70}")
    for path in [
        'paper.md',
        'FINAL_STATUS.md',
        'PHASE4_REVIEWER_ATTACKS.md',
        'COMPREHENSIVE_AUDIT.md',
        'SESSION_REPORT.md',
        'SESSION_REPORT_tables.md',
        'analysis_output/final/table1_summary.csv',
        'analysis_output/final/table2_vs_random.csv',
        'analysis_output/final/headline.json',
        'analysis_output/phase4_attacks/attack_matrix.csv',
        'analysis_output/reward_decomposition/decomposition_9x9.csv',
        'analysis_output/cover_time/cover_time_9x9.csv',
        'manifest_final.json',
        'paper_figures/fig1_scale_curves.png',
        'paper_figures/fig2_paired_diffs.png',
        'paper_figures/fig3_k4_ablation.png',
        'paper_figures/fig4_pain_scatter.png',
        'paper_figures/fig5_capacity_study.png',
    ]:
        p = Path(path)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  [OK] {path} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
