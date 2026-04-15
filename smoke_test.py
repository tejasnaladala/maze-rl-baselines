"""smoke_test.py — fast CI sanity check for the whole experiment harness.

Runs every agent on every maze size with a single seed and a tiny train budget.
Detects import errors, runtime crashes, NaN/Inf, and silly regressions before
burning GPU time on the real sweeps.

Target runtime: ~2 minutes on CPU, <30 seconds on GPU.
Exit 0 = all green. Exit 1 = at least one agent crashed or produced NaN.

Usage:
    python smoke_test.py
    python smoke_test.py --sizes 9,13     # quick subset
    python smoke_test.py --gpu            # use GPU if available
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiment_lib_v2 import (
    RandomAgent, NoBacktrackRandomAgent, LevyRandomAgent, BFSOracleAgent,
    TabularQAgent, FeatureQAgent, MLPDQNAgent, SpikingDQNAgent, DoubleDQNAgent,
    run_experiment, set_all_seeds, code_hash,
)

ALL_AGENTS: list[tuple[str, callable]] = [
    ('Random',      lambda d: RandomAgent()),
    ('NoBackRand',  lambda d: NoBacktrackRandomAgent()),
    ('LevyRand',    lambda d: LevyRandomAgent(alpha=1.5)),
    ('BFSOracle',   lambda d: BFSOracleAgent(avoid_hazards=True)),
    ('TabularQ',    lambda d: TabularQAgent()),
    ('FeatureQ',    lambda d: FeatureQAgent()),
    ('MLP_DQN',     lambda d: MLPDQNAgent(device=d, eps_decay=500)),
    ('DoubleDQN',   lambda d: DoubleDQNAgent(device=d, eps_decay=500)),
    ('SpikingDQN',  lambda d: SpikingDQNAgent(device=d, eps_decay=500)),
]

# Minimum expected behavior — if violated, flag the test as suspect (not fail).
# BFS should always solve; Random should not be 100%; TabularQ should not exceed 50%
# at short training budgets.
SANITY_CHECKS: dict[str, tuple[float, float]] = {
    'BFSOracle':  (0.90, 1.01),   # 90-100% success
    'Random':     (0.0,  0.85),   # 0-85%
    'TabularQ':   (0.0,  0.60),
}


def check_finite(values) -> bool:
    for v in values:
        if not math.isfinite(v):
            return False
    return True


def smoke(agent_name: str, make_agent, size: int, device: str, num_train: int, num_test: int) -> dict:
    t0 = time.time()
    set_all_seeds(42, deterministic=False)
    agent = make_agent(device)
    try:
        results = run_experiment(agent, agent_name, size, num_train, num_test, 42)
    except Exception as e:
        return {
            'ok': False, 'agent': agent_name, 'size': size,
            'error': f"{type(e).__name__}: {e}",
            'traceback': traceback.format_exc(),
            'elapsed': time.time() - t0,
        }

    test = [r for r in results if r.phase == 'test']
    if not test:
        return {'ok': False, 'agent': agent_name, 'size': size, 'error': 'no test results'}

    success = sum(r.solved for r in test) / len(test)
    rewards = [r.reward for r in test]
    if not check_finite(rewards):
        return {'ok': False, 'agent': agent_name, 'size': size,
                'error': f'NaN/Inf in reward: {rewards[:3]}'}

    elapsed = time.time() - t0
    warn = None
    if agent_name in SANITY_CHECKS:
        lo, hi = SANITY_CHECKS[agent_name]
        if not (lo <= success <= hi):
            warn = f"success={success:.2f} outside expected [{lo:.2f}, {hi:.2f}]"

    return {
        'ok': True, 'agent': agent_name, 'size': size,
        'success': success, 'mean_reward': sum(rewards) / len(rewards),
        'elapsed': elapsed, 'warn': warn,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', default='9,13,17', help='Comma-separated maze sizes')
    parser.add_argument('--num-train', type=int, default=10)
    parser.add_argument('--num-test', type=int, default=20)
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]
    device = 'cuda' if args.gpu else 'cpu'
    try:
        import torch
        if args.gpu and not torch.cuda.is_available():
            print("WARNING: --gpu specified but CUDA not available, falling back to CPU")
            device = 'cpu'
    except ImportError:
        pass

    print(f"Engram smoke test  code_hash={code_hash()}")
    print(f"Device: {device}, sizes={sizes}, train_eps={args.num_train}, test_eps={args.num_test}")
    print("=" * 80)

    results = []
    failed = 0
    warned = 0
    total_start = time.time()

    for size in sizes:
        for agent_name, make_agent in ALL_AGENTS:
            r = smoke(agent_name, make_agent, size, device, args.num_train, args.num_test)
            results.append(r)
            status = ''
            if not r['ok']:
                failed += 1
                status = 'FAIL'
                print(f"  {agent_name:12s} size={size:<3} {status:6s} {r.get('error')}")
            else:
                if r.get('warn'):
                    warned += 1
                    status = 'WARN'
                else:
                    status = 'OK'
                print(f"  {agent_name:12s} size={size:<3} {status:6s} success={r['success']*100:5.1f}%  "
                      f"reward={r['mean_reward']:+6.2f}  t={r['elapsed']:5.2f}s"
                      + (f"  [{r['warn']}]" if r.get('warn') else ''))

    total = time.time() - total_start
    print("=" * 80)
    print(f"Total time: {total:.1f}s")
    print(f"Passed: {len(results) - failed - warned}, Warnings: {warned}, Failed: {failed}")

    if failed > 0:
        print("\nFAILED runs:")
        for r in results:
            if not r['ok']:
                print(f"\n--- {r['agent']} size={r['size']} ---")
                print(r.get('traceback', r.get('error', '')))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
