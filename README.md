# Procedural Maze RL Baselines

A small, fully reproducible procedural-maze benchmark with one sharp finding: on the same audited test harness at 9x9 mazes,

- A 5-line egocentric wall-following heuristic solves **100%** of unseen instances.
- A behavior-cloned MLP (same architecture, observation, and optimizer as the neural DQN) reaches **97.4%**.
- The best of seven hyperparameter-tuned modern reward-driven baselines (SB3 PPO, DQN, and A2C, three learning rates each, 70 runs total) reaches **31.4%**, statistically tied with uniform Random (32.7%).
- A behavior-cloning warm-start (initialize the MLP_DQN at the 97% distilled weights, then fine-tune with standard DQN, 20 seeds) drops test success to a post-fine-tune mean of **18.4%** (sd 11.5, median 17.0, range 0 to 38), an approximately 79 pp drop from the 97.2% BC starting accuracy.

The neural policy class can express the maze-solving policy: behavior cloning finds it. Standard reward-driven RL does not discover that policy from random initialization, and the BC initialization does not retain its high-performing basin under standard DQN fine-tuning. The post-fine-tune policy is statistically tied with from-scratch MLP_DQN (19.3%); we test one fine-tune recipe and report robustness to alternatives as open follow-up.

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

## BC warm-start: basin not retained under fine-tuning

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

The ~79 pp drop from BC starting accuracy is consistent across the n=20 sweep; variance is substantial (6 of 20 seeds collapse to 0 to 8%, 4 of 20 reach 26 to 38%). The post-fine-tune mean is statistically indistinguishable from from-scratch MLP_DQN (19.3%) and 13 pp below from-scratch SB3 DQN at default LR (31.4%). We test one fine-tune recipe; robustness to lower LR, target-network freeze, zero-exploration, and offline-RL variants (CQL, IQL) is open follow-up.

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
| `launch_bc_warmstart.py` | 20 seeds (BC + DQN fine-tune) | ~32 min (8-way parallel) |
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

**It is**: a small, well-audited benchmark with one narrow falsifiable claim, backed by ~3,500 runs across 20+ seeds per cell, a complete reproducibility apparatus, and a BC warm-start probe showing the distilled basin does not survive standard DQN fine-tuning.

**It is not**: a method paper or a result on large-scale environments. The claim is documented at 9x9 mazes and replicated on 4 MiniGrid environments. Larger-scale generalization is open follow-up work.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citing

Single-author preprint. Citation format will be added when an arXiv id is assigned.

## Contact

Tejas Naladala: `tejas.naladala@gmail.com`. Independent reproduction is welcome.
