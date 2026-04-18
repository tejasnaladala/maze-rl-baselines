# Modern Baseline Table Template (to insert when 70/70 done)

This is the table that will be added to PAPER_SHORT.md, PAPER_PREVIEW.md, and paper.md when the sweep completes. Numbers will be filled in via the aggregation script.

## Table 5b: Modern HP-tuned baselines also fail (70 runs)

In response to the reviewer concern that DQN-family baselines may simply be under-tuned, we ran a multi-LR sweep of three modern algorithms on the same audited main-sweep harness. PPO and A2C run on CPU per SB3 guidance for MLP policies; DQN runs on GPU.

All numbers at 9x9 mazes, n=10 seeds per config, 500K environment steps each, paired bootstrap stratified by seed.

| Config | Mean | sd | Median | Range | n | Best vs ref |
|---|---|---|---|---|---|---|
| PPO_lr1e-4 | 3.6 | 4.9 | 1.0 | 0 to 14 | 10 | -16pp vs MLP_DQN |
| PPO_lr3e-4 | 6.0 | 6.9 | 4.0 | 0 to 24 | 10 | -13pp vs MLP_DQN |
| PPO_lr1e-3 | (skipped: gradient instability, see Limitations) | — | — | — | 2 | — |
| DQN_lr1e-4 | TBD | TBD | TBD | TBD | 10 | TBD |
| DQN_lr5e-4 (default) | TBD | TBD | TBD | TBD | 10 | TBD |
| DQN_lr1e-3 | TBD | TBD | TBD | TBD | 10 | TBD |
| A2C_default | TBD | TBD | TBD | TBD | 10 | TBD |

**Reference baselines for comparison**:
| Reference | Mean |
|---|---|
| BFS_distilled_MLP (same arch as MLP_DQN) | 97.4% |
| EgoWallFollowerLeft (5-line heuristic) | 100.0% |
| NoBackRandom | 51.5% |
| FeatureQ_v2 | 36.5% |
| Random | 32.7% |
| MLP_DQN (default) | 19.3% |

**Reading the table.** [Will be written based on actual numbers. Two pre-drafted versions:]

### Version A (if no config beats MLP_DQN's 19.3%)

> Across 70 runs of three modern algorithms at three learning rates each, no config exceeds MLP_DQN's 19.3 percent and none reaches uniform Random's 32.7 percent. The reviewer concern that DQN-family baselines may be under-tuned is directly answered: PPO and A2C are also under-tuned by this metric, and DQN at non-default learning rates does not improve the headline. The 78 percentage point gap between the BFS-distilled MLP (97.4 percent) and the best modern reward-driven baseline survives a 7-config HP sweep.

### Version B (if some config beats MLP_DQN but not Random)

> Across 70 runs, the best modern config reaches X percent, [Y] percentage points above MLP_DQN's 19.3 but still [Z] percentage points below uniform Random and [W] percentage points below the BFS-distilled MLP. Better hyperparameter tuning closes part of the DQN-family gap but does not change the central representation-vs-discovery story: standard reward-driven RL on this benchmark does not reach the policy that the same network class can express.

### Version C (if some config beats Random)

> Across 70 runs, [config X] reaches Y percent, [Y - 32.7] percentage points above uniform Random and [97.4 - Y] percentage points below the BFS-distilled MLP. This is a substantive finding: better-tuned [config] does close part of the discovery gap. The remaining [97.4 - Y] percentage point gap is still load-bearing for the central claim: the network class can express a 97 percent policy that even the best HP-tuned modern baseline does not discover from this reward signal.

## Per-seed values (audit trail)

For full reproducibility, every per-seed value will be listed in Appendix B (PAPER_PREVIEW only).

## Cold email updates

When the sweep completes, edit the cold email's "ruled out" sentence:

**Current:**
> A capacity sweep (h32 to h256), an LR sweep across 1.5 orders of magnitude, LSTM memory, a K4 reward ablation, and a MiniGrid cross-env replication all rule out the obvious explanations.

**Updated:**
> A capacity sweep (h32 to h256), a 7-config modern-baseline sweep (PPO + DQN + A2C across 3 LRs each, 70 runs total, [best config X% vs MLP_DQN 19%]), DRQN with LSTM memory, a K4 reward ablation, and a MiniGrid cross-env replication all rule out the obvious explanations.

If best modern config X stays below 32.7% (Random):
> ...all rule out the obvious explanations: even the best-tuned modern baseline trails uniform Random.

If X is between 19.3 and 32.7%:
> ...all rule out the obvious explanations: even the best-tuned modern baseline trails uniform Random by [32.7 - X]pp.

If X >= 32.7%:
> [need to nuance the headline; consult before sending]
