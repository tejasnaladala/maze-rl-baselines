# Procedural Maze RL Baselines

Research code and result artifacts for comparing heuristics, behavior cloning, custom DQN, and Stable-Baselines3 agents on procedurally generated mazes.

> [!WARNING]
> **Artifact reconciliation fails in the current checkout.** Running `python reproduce.py verify --manifest manifest_final.json` exits 1. The current scan contains 2,193 files against the 1,540-file pin: 653 later files are outside the manifest, and three pinned 9x9 headline cells recompute differently.
>
> | Cell | Pinned mean success | Current mean success |
> |---|---:|---:|
> | `DRQN` | 14.75% | 19.0% |
> | `MLP_DQN_h32` | 12.0% | 13.6% |
> | `vanilla__DoubleDQN` | 13.71% | 11.6% |
>
> No pinned file is missing or hash-mismatched. `manifest_final.json` is intentionally unchanged; the later files and the three drifting cells require reconciliation before a new pin is created. Tables and papers below record reported experiment outputs. The current artifact set does not pass reconciliation.

## Paper

- [`PAPER_SHORT.pdf`](PAPER_SHORT.pdf): 6-page paper with the headline tables and claims.
- [`PAPER_PREVIEW.pdf`](PAPER_PREVIEW.pdf): 11-page version with appendices.
- [`paper.md`](paper.md): full working draft (markdown).

## Artifact checks

```bash
git clone https://github.com/tejasnaladala/maze-rl-baselines
cd maze-rl-baselines
pip install -r requirements-experiments.txt

# Current status: exits 1 with 653 files outside the pin and 3 drifting cells.
python reproduce.py verify --manifest manifest_final.json

# Fast small-budget sweep across agent classes and maze sizes.
python smoke_test.py

# Independently exercise the 5-line wall follower.
python verify_wall_follower.py
```

## Reported results (9x9)

These values are retained as the repository's reported result table. They have not been reconciled with the current artifact set described above.

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

The behavior-cloning result tests whether the MLP architecture can represent the oracle policy under supervised training. The reward-driven rows use either the custom implementation or SB3. Only one DQN fine-tuning recipe was tested from the behavior-cloned initialization.

## Reported BC warm-start experiment

Initialize the MLP_DQN online and target networks from the BFS-distilled weights, then fine-tune for 200k steps of standard DQN (epsilon 0.20 to 0.05 over 50k steps). Across 20 seeds, the post-fine-tune mean is 18.4% (sd 11.5), well below the 97.2% BC starting accuracy but statistically tied with from-scratch MLP_DQN (19.3%).

| Statistic | Value |
|---|---|
| n | 20 seeds |
| BC mean (sample of n=5) | 97.2% |
| Post-fine-tune mean | 18.4% |
| Post-fine-tune median | 17.0% |
| Post-fine-tune sd | 11.5 |
| Range | 0 to 38% |
| Per-seed sorted | 0, 0, 4, 8, 8, 12, 14, 14, 16, 16, 18, 20, 22, 24, 26, 26, 28, 36, 38, 38 |

Across the recorded seeds, 6 of 20 finish between 0 and 8%; 4 of 20 finish between 26 and 38%. The reported 18.4% mean is close to the 19.3% from-scratch MLP_DQN result and 13 percentage points below the 31.4% SB3 DQN result. Lower learning rates, a frozen target network, zero exploration, and offline-RL variants such as CQL or IQL remain untested here.

## Key experiments (re-runnable)

| Script | Runs | Wall time on RTX 5070 Ti |
|---|---|---|
| `launch_modern_baselines.py` | 70 (PPO/DQN/A2C, 3 LRs, 10 seeds) | ~7 hr |
| `launch_bc_warmstart.py` | 20 seeds (BC + DQN fine-tune) | ~32 min (8-way parallel) |
| `launch_policy_distillation.py` | distillation headline | ~30 min |
| `launch_ppo_shaped.py` | 10 seeds | ~85 min |
| `launch_loopy_pilot.py` | 25 (5 agents, 5 seeds) | ~4 min |

## Data and reproducibility

- **Pinned artifact set**: [`manifest_final.json`](manifest_final.json) contains SHA-256 entries for 1,540 result files and a pinned headline summary. The current verifier scan finds 2,193 files; see the reconciliation warning above.
- **Raw results**: per-experiment JSON under `raw_results/` (e.g. `exp_modern_baselines`, `exp_bc_warmstart`, `exp_wall_following`).
- **Code-hash pinned** per result. Current main-sweep hash: `ed681d75c27fe352`.
- **Statistics**: single-file pipeline in [`stats_pipeline.py`](stats_pipeline.py) (paired bootstrap, Mann-Whitney U, Cohen d, Holm-Bonferroni correction, BCa intervals).
- **Harness-bug audit trail**: a real harness bug was caught and fixed during development; the before/after is documented in the paper (section 3.2.1).

## Scope

The primary study covers 9x9 procedural mazes, with additional experiments on four MiniGrid environments. Larger-scale environments, broader fine-tuning recipes, and reconciliation of the current artifacts remain open work.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citing

Single-author preprint. Citation format will be added when an arXiv id is assigned.

## Contact

Tejas Naladala: `tejas.naladala@gmail.com`. Independent reproduction is welcome.
