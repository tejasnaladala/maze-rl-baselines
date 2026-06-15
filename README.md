# Procedural Maze RL Baselines

A small, fully reproducible procedural-maze benchmark with one sharp finding: on the same audited test harness at 9x9 mazes,

- A 5-line egocentric wall-following heuristic solves **100%** of unseen instances.
- A behavior-cloned MLP (same architecture, observation, and optimizer as the neural DQN) reaches **97.4%**.
- The best of seven hyperparameter-tuned modern reward-driven baselines (SB3 PPO, DQN, and A2C, three learning rates each, 70 runs total) reaches **31.4%**, statistically tied with uniform Random (31.7%).
- A behavior-cloning warm-start (initialize the MLP_DQN at the 97% distilled weights, then fine-tune with standard DQN) collapses test success to **13.6%** across all 5 seeds.

The neural policy class can express the maze-solving policy: behavior cloning finds it. Standard reward-driven RL does not discover that policy from random initialization, and actively pushes the network out of the high-performing basin even when it starts inside it.

## Paper

- [`PAPER_SHORT.pdf`](PAPER_SHORT.pdf): 6-page paper with the headline tables and claims.
- [`PAPER_PREVIEW.pdf`](PAPER_PREVIEW.pdf): 11-page version with appendices.
- [`paper.md`](paper.md): full working draft (markdown).

## Headline result (9x9)

| Tier | Agent | Mean success (%) | sd | n |
|---|---|---|---|---|
| Oracle | BFSOracle | 100.0 | 0.0 | 20 |
| Heuristic (5-line, ego-only) | EgoWallFollowerLeft | 100.0 | 0.0 | 20 |
| Distillation (same arch as DQN) | DistilledMLP_from_BFSOracle | 97.4 | 2.5 | 20 |
| Random walk | NoBackRandom | 51.5 | 6.6 | 50 |
| Random walk | Random | 32.7 | 6.1 | 50 |
| Modern RL (SB3 DQN, default LR) | SB3_DQN_lr5e-4 | 31.4 | 7.2 | 10 |
| Neural RL (custom) | MLP_DQN | 19.3 | 6.7 | 40 |
| Modern RL (SB3 A2C) | A2C_default | 8.4 | 4.3 | 10 |
| Modern RL (SB3 PPO, best LR) | PPO_lr3e-4 | 6.0 | 6.9 | 10 |

## BC warm-start collapse

Each row starts the MLP_DQN from behavior-cloned weights, then fine-tunes for 200k steps of standard DQN.

| Seed | BC test (%) | Post-fine-tune (%) | Drop (pp) |
|---|---|---|---|
| 42 | 98.0 | 0.0 | -98.0 |
| 123 | 100.0 | 18.0 | -82.0 |
| 456 | 98.0 | 16.0 | -82.0 |
| 789 | 90.0 | 12.0 | -78.0 |
| 1024 | 100.0 | 22.0 | -78.0 |
| **Mean** | **97.2** | **13.6** | **-83.6** |

## Reproducing the headline

```bash
git clone https://github.com/tejasnaladala/maze-rl-baselines
cd maze-rl-baselines
pip install -r requirements-experiments.txt

# Re-hash every result file against the pinned manifest, then recompute
# every numerical claim and compare to the pinned headline.
python reproduce.py verify --manifest manifest_final.json

# Fast sanity check: every agent class on every maze size, tiny budget.
# ~2 min on CPU, under 30 s on GPU.
python smoke_test.py

# Verify the 5-line wall-follower hits 100%.
python verify_wall_follower.py
```

## Key experiments (re-runnable)

| Script | Runs | Wall time on RTX 5070 Ti |
|---|---|---|
| `launch_modern_baselines.py` | 70 (PPO/DQN/A2C, 3 LRs, 10 seeds) | ~7 hr |
| `launch_bc_warmstart.py` | 5 seeds (BC + DQN fine-tune) | ~50 min |
| `launch_policy_distillation.py` | distillation headline | ~30 min |
| `launch_ppo_shaped.py` | 10 seeds | ~85 min |
| `launch_loopy_pilot.py` | 25 (5 agents, 5 seeds) | ~4 min |

## Data and reproducibility

- **Result manifest**: [`manifest_final.json`](manifest_final.json) pins a SHA-256 hash of every result file (4,200+ JSON records) plus the pinned headline summary. `reproduce.py verify` re-hashes and recomputes.
- **Raw results**: per-experiment JSON under `raw_results/` (e.g. `exp_modern_baselines`, `exp_bc_warmstart`, `exp_wall_following`).
- **Code-hash pinned** per result. Current main-sweep hash: `ed681d75c27fe352`.
- **Statistics**: single-file pipeline in [`stats_pipeline.py`](stats_pipeline.py) (paired bootstrap, Mann-Whitney U, Cohen d, Holm-Bonferroni correction, BCa intervals).
- **20+ seeds per cell** on the headline table.
- **Harness-bug audit trail**: a real harness bug was caught and fixed during development; the before/after is documented in the paper (section 3.2.1).

## What this is and is not

**It is**: a small, well-audited benchmark with one narrow falsifiable claim, backed by ~3,500 runs across 20+ seeds per cell, a complete reproducibility apparatus, and a BC warm-start collapse that did not appear in prior work.

**It is not**: a method paper or a result on large-scale environments. The claim is documented at 9x9 mazes and replicated on 4 MiniGrid environments. Larger-scale generalization is open follow-up work.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citing

Single-author preprint. Citation format will be added when an arXiv id is assigned.

## Contact

Tejas Naladala: `tejas.naladala@gmail.com`. Independent reproduction is welcome.
