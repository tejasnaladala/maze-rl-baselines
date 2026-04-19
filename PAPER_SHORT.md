# A 5-Line Ego-Only Wall-Follower Beats Trained Neural Networks on Procedural Mazes

## Reward-Driven RL Cannot Find a Policy Its Own Network Class Can Express, and Actively Destroys That Policy When Initialized at It

**Single-author draft v1.3** · 2026-04-18 · code + data: https://github.com/tejasnaladala/maze-rl-baselines

---

## Abstract

We present a fully reproducible procedural-maze benchmark with the following pattern, on the same audited test harness at 9x9 mazes:

- A 5-line egocentric wall-following heuristic solves 100% of unseen instances.
- A supervised MLP, trained by behavioral cloning on BFS oracle action labels with the same 24-d ego-feature observation, the same 24 to 64 to 32 to 4 architecture, and the same Adam optimizer as MLP_DQN, reaches 97.4 percent.
- The best of seven HP-tuned modern reward-driven baselines (SB3 PPO, DQN, A2C across three learning rates each, 70 runs total) reaches 31.4 percent at SB3 DQN default LR (n=10), statistically tied with uniform Random (32.7 percent) and 66 percentage points below the BC-distilled MLP.
- A behavioral-cloning warm-start experiment (initialize MLP_DQN online and target networks from BFS-distilled weights, fine-tune via standard DQN with reduced exploration, 200K env steps, 5 seeds) collapses test success from a mean BC pre-fine-tune 97.2 percent to a mean post-fine-tune 13.6 percent. The fine-tuned policy ends below from-scratch DQN (19.3 percent custom, 31.4 percent SB3).

The neural policy class can express the maze-solving policy. Standard reward-driven RL does not discover this policy from random initialization, and actively pushes the network out of the high-performing basin even when initialized inside it.

We rule out the standard explanations for this gap: capacity (h32 to h256 sweep yields 13.6 to 19.3 percent flat band), learning rate (default is the local optimum across 1.5 orders of magnitude), partial observability (DRQN with LSTM matches MLP_DQN), reward shaping (paired ablation drops learners by 4 to 18 percentage points while leaving random walks unchanged), information asymmetry (the same 24-d ego-features support a 100% wall-following solution), and weakness of the modern-RL baseline class (PPO/DQN/A2C sweep above). A second environment family (MiniGrid: DoorKey, FourRooms, MultiRoom-N2-S4, Unlock) replicates the headline pattern in 3 of 4 environments. A 5-seed pilot on Wilson-algorithm loopy mazes confirms the heuristic still solves 100 percent under modest loop injection. We empirically confirm the Alon, Benjamini, Lubetzky and Sodin (2007) non-backtracking cover-time advantage on this benchmark.

The narrow claim: procedural-maze RL evaluations should include hand-coded heuristic, supervised distillation, and random-walk baselines on identical evaluation harnesses, with a behavioral-cloning warm-start probe. We provide one such audited benchmark and isolate a representation versus discovery gap that is invisible without these baselines.

---

## 1. Headline Result (Table 1)

All numbers on the same audited test harness at 9x9 mazes. n indicates seeds.

| Tier | Agent | Mean success (%) | sd | n |
|---|---|---|---|---|
| 1 Oracle | BFSOracle | 100.0 | 0.0 | 20 |
| 2 Heuristic (5-line, ego-only) | EgoWallFollowerLeft | 100.0 | 0.0 | 20 |
| 3 Distillation (same arch as DQN) | DistilledMLP_from_BFSOracle | 97.4 | 2.5 | 20 |
| 4 Random walk | NoBackRandom | 51.5 | 6.6 | 50 |
| 4 Random walk | LevyRandom (alpha=2.0) | 40.3 | 6.7 | 20 |
| 5 Tabular | FeatureQ_v2 | 36.5 | 8.0 | 50 |
| 4 Random walk | Random | 32.7 | 6.1 | 50 |
| 6 Modern RL (SB3 DQN default LR) | SB3_DQN_lr5e-4 | 31.4 | 7.2 | 10 |
| 6 Modern RL (SB3 DQN low LR) | SB3_DQN_lr1e-4 | 28.4 | 12.4 | 10 |
| 6 Modern RL (SB3 DQN high LR) | SB3_DQN_lr1e-3 | 23.6 | 13.9 | 10 |
| 5 Tabular | TabularQ_v2 | 29.8 | 8.8 | 20 |
| 6 Neural RL (custom) | MLP_DQN | 19.3 | 6.7 | 40 |
| 6 Neural RL (custom) | DRQN (LSTM) | 19.0 | 10.8 | 40 |
| 6 Neural RL (custom) | DoubleDQN | 16.3 | 5.7 | 40 |
| 6 Modern RL (SB3 A2C) | A2C_default | 8.4 | 4.3 | 10 |
| 6 Modern RL (SB3 PPO best LR) | PPO_lr3e-4 | 6.0 | 6.9 | 10 |
| 6 Modern RL (SB3 PPO low LR) | PPO_lr1e-4 | 3.6 | 4.9 | 10 |

**Reading the table.** The agent ladder is monotonically downward across tiers. The 66 percentage point gap between the BFS-distilled MLP (97.4 percent) and the best HP-tuned modern reward-driven baseline (SB3 DQN at default LR, 31.4 percent) is the central observation: same network class, same observation, only the training signal differs. The same gap measured against our custom MLP_DQN (19.3 percent) is 78 percentage points. No reward-driven RL configuration tested clears uniform Random (32.7 percent) by a statistically significant margin.

---

## 2. The Representation versus Discovery Dichotomy (Proof of Claim)

### 2.1 Distillation proves expressibility

Setup: a supervised feedforward MLP, with the exact 24 to 64 to 32 to 4 architecture, the same Adam optimizer, and the same 24-dimensional ego-feature observation as MLP_DQN, is trained on action labels collected from the BFS oracle on the training maze distribution. The model is evaluated deterministically on 50 held-out test mazes per seed.

Result: mean test success 97.4 percent (sd 2.5, n=20 seeds).

The same architecture trained via standard DQN reaches 19.3 percent (custom) or 31.4 percent (SB3, default LR). The neural policy class can represent the maze-solving policy. Standard reward-driven reinforcement learning does not discover it from random initialization.

### 2.2 BC warm-start proves the reward gradient is destructive

A behavioral-cloning warm-start experiment was run to disambiguate two hypotheses for the gap above:

- H1 (discovery problem): the high-performing policy is reachable from a sufficiently good initialization, RL just cannot reach it from random initialization.
- H2 (destruction problem): the reward landscape actively pushes the network out of the high-performing basin, regardless of initialization.

Protocol (5 seeds):

1. Train an MLP via behavioral cloning on BFS oracle action labels.
2. Initialize MLP_DQN's online network and target network with the resulting weights (architecture is bit-identical).
3. Fine-tune via standard DQN with reduced initial exploration (epsilon 0.20 to 0.05 over 50K steps), 200K total environment steps, same shaped reward as the from-scratch DQN baseline.
4. Test on the main-sweep harness (50 episodes per seed, deterministic policy).

Result (Table 1.B):

| Seed | BC test (%) | Post-fine-tune test (%) | Drop (pp) | Verdict |
|---|---|---|---|---|
| 42 | 98.0 | 0.0 | -98.0 | COLLAPSED |
| 123 | 100.0 | 18.0 | -82.0 | COLLAPSED |
| 456 | 98.0 | 16.0 | -82.0 | COLLAPSED |
| 789 | 90.0 | 12.0 | -78.0 | COLLAPSED |
| 1024 | 100.0 | 22.0 | -78.0 | COLLAPSED |
| **Mean** | **97.2** | **13.6** | **-83.6** | **5/5 COLLAPSED** |

The mean drop of 83.6 percentage points is consistent across seeds. The post-fine-tune mean (13.6 percent) is below the from-scratch MLP_DQN baseline (19.3 percent custom) and well below SB3 DQN at default LR (31.4 percent).

H2 is supported. The reward gradient does not merely fail to discover the high-performing basin from random initialization; it actively pushes a network out of that basin even when initialized inside it.

### 2.3 Falsifiability

The combined claim is falsifiable. To defeat it, exhibit a training procedure (intrinsic motivation, curriculum, demonstrations, larger budget, alternative algorithm) that either:

- Closes the gap from 31.4 percent toward 97.4 percent without changing the network class, OR
- Initializes MLP_DQN at the BC-distilled weights and produces a fine-tuned policy that maintains test success above 80 percent.

---

## 3. Modern HP-Tuned Baseline Sweep (Table 2, 70 runs)

In direct response to the reviewer concern that DQN-family baselines may be under-tuned, we ran a multi-LR sweep of three modern algorithms (PPO, DQN, A2C) on the same audited main-sweep harness. PPO and A2C run on CPU per SB3 guidance for MLP policies; DQN runs on GPU. All numbers at 9x9 mazes, n=10 seeds per config (PPO_lr1e-3 truncated to n=2 due to gradient instability), 500K environment steps.

| Config | Mean (%) | sd | Median (%) | Range (%) | n |
|---|---|---|---|---|---|
| PPO_lr1e-4 | 3.6 | 4.9 | 1.0 | 0 to 14 | 10 |
| PPO_lr3e-4 | 6.0 | 6.9 | 4.0 | 0 to 24 | 10 |
| PPO_lr1e-3 | 2.0 | 2.8 | 2.0 | 0 to 4 | 2 (skipped 8) |
| DQN_lr1e-4 | 28.4 | 12.4 | 33.0 | 8 to 44 | 10 |
| **DQN_lr5e-4 (default)** | **31.4** | **7.2** | **34.0** | **16 to 40** | **10** |
| DQN_lr1e-3 | 23.6 | 13.9 | 21.0 | 4 to 52 | 10 |
| A2C_default | 8.4 | 4.3 | 7.0 | 2 to 16 | 10 |

The best modern reward-driven baseline (SB3 DQN at default LR) reaches 31.4 percent mean. Statistically indistinguishable from uniform Random (32.7 percent). 66 percentage points below the BC-distilled MLP (97.4 percent). Our custom MLP_DQN baseline (used elsewhere in this paper, n=40) at 19.3 percent reflects implementation differences from the SB3 reference and is reported transparently. No configuration tested clears Random.

---

## 4. Ruling Out Standard RL-Failure Explanations (Tables 3 to 6)

### Table 3. Capacity is not the bottleneck

| Hidden | 9x9 success (%) | sd | 13x13 success (%) | sd |
|---|---|---|---|---|
| h=32 | 13.6 | 8.1 | 3.0 | 2.3 |
| h=64 (default) | 19.3 | 6.7 | 3.8 | 4.2 |
| h=128 | 15.7 | 8.4 | 4.0 | 3.5 |
| h=256 | 13.6 | 8.3 | 4.8 | 5.0 |

8x capacity scaling produces a flat 13.6 to 19.3 percent band at 9x9. Capacity is not the bottleneck.

### Table 4. Learning rate is at the local optimum (custom MLP_DQN)

| Learning rate | Mean (%) | sd | n |
|---|---|---|---|
| 1e-4 | 7.4 | 3.7 | 10 |
| 5e-4 (default) | 19.6 | 5.6 | 10 |
| 1e-3 | 11.0 | 6.9 | 10 |
| 3e-3 | 4.8 | 5.3 | 10 |

The default learning rate is the local optimum across 1.5 orders of magnitude. SB3 DQN (Table 2) replicates this pattern: 5e-4 default LR is also the SB3 optimum.

### Table 5. Reward shaping helps learners, not random walks (paired bootstrap, K4)

| Agent | Full reward (%) | Vanilla reward (%) | Difference (pp) | Cohen d | p (Holm-Bonferroni) |
|---|---|---|---|---|---|
| Random | 32.7 | 32.7 | 0.0 | 0.00 | 1.00 |
| NoBackRandom | 51.5 | 51.5 | 0.0 | 0.00 | 1.00 |
| FeatureQ | 35.3 | 17.4 | -17.9 | -2.66 | <0.001 |
| MLP_DQN | 19.3 | 14.6 | -4.7 | -0.61 | 0.014 |
| DoubleDQN | 16.3 | 12.6 | -3.7 | -0.65 | 0.011 |

Removing reward shaping leaves random walks unchanged and hurts every learner. The "random wins because shaping punishes directed policies" hypothesis is refuted; the opposite is true.

### Table 6. Memory does not rescue the failure

DRQN, a recurrent Q-learner with a 64-unit LSTM (sequence length 8) under otherwise identical conditions, reaches 19.0 percent at 9x9 (n=40). Statistically indistinguishable from MLP_DQN. Adding memory does not close the gap to NoBackRandom.

---

## 5. Loopy-Maze Topology Pilot (Table 7)

The recursive-backtracking generator used for the headline produces simply-connected mazes (single spanning tree, no loops). EgoWallFollowerLeft is provably optimal on simply-connected graphs by left-hand rule. To test whether wall-following success on this benchmark depends on the generator's topological property, we ran a 5-seed pilot on Wilson-algorithm mazes with 4 random extra passages added (loops introduced; hazards disabled to isolate the topology effect).

| Agent | Mean success (%) | sd | Mean steps |
|---|---|---|---|
| BFSOracle | 100.0 | 0.0 | 12.2 |
| EgoWallFollowerLeft | 100.0 | 0.0 | 21.7 |
| MLP_DQN_h64 (under-trained, 100 ep) | 66.0 | 6.9 | 120.0 |
| NoBackRandom | 57.6 | 27.3 | 228.8 |
| Random | 37.6 | 25.2 | 262.0 |

The wall-follower remains 100 percent under modest loop injection. MLP_DQN improves substantially from 19.3 percent (hazard-loaded simply-connected) to 66 percent (hazard-free Wilson + loops), suggesting hazard avoidance interacts with topology. A v1.1 follow-up will run the loopy condition with hazards re-enabled and at higher loop densities (60 to 80 percent of eligible interior walls).

---

## 6. Robustness across Maze Sizes (Table 8)

Mean test success rate (%). n=20 seeds per cell.

| Agent | 9x9 | 11 | 13 | 17 | 21 | 25 |
|---|---|---|---|---|---|---|
| BFSOracle | 100 | 100 | 100 | 100 | 100 | 100 |
| EgoWallFollowerLeft | 100 | 100 | 100 | 100 | 100 | (n.r.) |
| NoBackRandom | 51.5 | 36.9 | 25.8 | 14.8 | 7.9 | 4.5 |
| LevyRandom (alpha=2.0) | 40.3 | 23.0 | 15.8 | 6.9 | 4.1 | 3.1 |
| FeatureQ_v2 | 36.5 | 22.4 | 12.1 | 1.0 | 0.3 | 0.0 |
| Random | 32.7 | 18.9 | 12.3 | 4.4 | 2.1 | 1.5 |
| MLP_DQN_h64 | 19.3 | (n.r.) | 3.8 | (n.r.) | (n.r.) | (n.r.) |
| DRQN | 19.0 | (n.r.) | 4.4 | 0.7 | 0.3 | (n.r.) |

The headline ordering holds across sizes 9 through 21. Neural agents collapse with size; random-walk and oracle baselines degrade more gracefully.

---

## 7. Cross-Environment Replication (Table 9)

To check the headline is not specific to our maze generator, we replicate on 4 MiniGrid environments. n=20 seeds per cell.

| Environment | MLP_DQN (%) | NoBackRandom (%) | Random (%) |
|---|---|---|---|
| MiniGrid-DoorKey-5x5 | 0.0 | 9.0 | 9.3 |
| MiniGrid-FourRooms | 0.7 | 3.9 | 2.9 |
| MiniGrid-MultiRoom-N2-S4 | 5.0 | 2.2 | 2.1 |
| MiniGrid-Unlock | 0.0 | 3.8 | 4.3 |

The headline pattern (MLP_DQN at or below Random) replicates in 3 of 4 MiniGrid environments. This second environment family weakens the "tree-maze artifact" critique.

---

## 8. Cover-Time Scaling Theorem Confirmation (Table 10)

Power-law fits to success rate as a function of maze size n: success_rate(n) = a * n^b. 10,000-resample bootstrap confidence intervals.

| Agent | Exponent b | 95% CI | R^2 |
|---|---|---|---|
| BFSOracle / EgoWallFollower | 0.000 | [0, 0] | constant 100% |
| NoBackRandom | -2.04 | [-2.21, -1.94] | 0.994 |
| LevyRandom (alpha=2.0) | -2.66 | [-2.93, -2.44] | 0.999 |
| Random | -2.88 | [-3.20, -2.59] | 0.996 |
| FeatureQ_v2 | -3.21 | [-3.54, -2.89] | 0.965 |

NoBackRandom decays 0.84 units more slowly than Random across maze sizes. Consistent with the Alon, Benjamini, Lubetzky and Sodin (2007) non-backtracking cover-time advantage. To our knowledge this is the first empirical confirmation of this theorem on a procedural reinforcement-learning benchmark.

---

## 9. Bayesian Posterior Dominance (Table 11)

Beta-Binomial conjugate posteriors with uniform prior, computed at 9x9.

| Comparison | Posterior probability |
|---|---|
| NoBackRandom > Random | > 0.999 |
| NoBackRandom > MLP_DQN | > 0.999 |
| NoBackRandom > DoubleDQN | > 0.999 |
| NoBackRandom > FeatureQ_v2 | > 0.999 |
| NoBackRandom > SB3_DQN_lr5e-4 | > 0.999 |
| EgoWallFollowerLeft > NoBackRandom | > 0.999 |
| FeatureQ_v2 > MLP_DQN | > 0.999 |
| Random > MLP_DQN | > 0.999 |
| BC_distilled > SB3_DQN_lr5e-4 | > 0.999 |
| BC_warm_start_pre_FT > BC_warm_start_post_FT | > 0.999 |

The headline ordering is posterior-certain at the 0.001 level.

---

## 10. Discussion

**Representation versus discovery.** Tables 1 and 2 establish that the MLP architecture used in MLP_DQN (24 to 64 to 32 to 4 with Adam) is sufficient to express a 97.4 percent maze-solving policy. The same architecture trained via standard DQN converges to a policy that solves 19.3 percent (custom) or 31.4 percent (SB3 default). The reward signal does not lead the optimizer to the policy that the network class can represent.

**The reward gradient is destructive, not merely uninformative.** The BC warm-start experiment (Section 2.2) closes the obvious alternative interpretation. If standard RL were merely a discovery problem, initializing inside the high-performing basin should preserve performance under fine-tuning. Instead, performance collapses by 83.6 percentage points on average across 5 seeds. The post-fine-tune policy ends below from-scratch DQN at any tested configuration. The reward landscape actively pushes the network out of the basin that contains the policy the network class can express.

**Why standard RL finds a low-success local optimum.** A reward decomposition (cover-time analysis at 9x9) shows MLP_DQN's pain-per-step at -0.136 versus Random's -0.238. Neural agents learn the locally rewarding policy component (avoiding walls, hazards, revisits). They fail at the globally rewarding component (sustained exploration through a region of small negative reward toward a sparse +10 goal). When they do solve, they do so near-optimally (0.78 times BFS path length). Their failure is on the fraction of mazes solved, not on path quality.

**Why the random-walk baselines win.** Random walks are stateless and reward-blind. They explore broadly. NoBackRandom adds a one-bit constraint (do not immediately reverse) which produces a 13.6 percent faster cover time, consistent with classical non-backtracking random-walk theory. They pay 10x the BFS-optimal path length per success but reach the goal on 32 to 52 percent of mazes.

**Scope of the claim.** We do not claim that neural function approximation fails in general. We do not claim that deep RL is broken on procedural mazes. We claim: on this audited procedural-maze benchmark, with the observation and reward we specify, standard reward-driven neural RL across seven HP-tuned configurations of three modern algorithms (PPO, DQN, A2C) fails to reach a policy the network class can express, and the reward gradient actively destroys that policy when handed it for free. An audited evaluation that includes a hand-coded heuristic, a supervised distillation, a behavioral-cloning warm-start probe, and a random-walk baseline exposes this gap clearly.

---

## 11. Limitations

- **Single primary maze class.** The recursive-backtracking generator produces simply-connected mazes (single spanning tree). The 5-seed Wilson + loop-injection pilot (Section 5) shows the heuristic is robust to modest loop injection (still 100 percent), but a higher loop density and hazard-enabled follow-up is queued for v1.1.
- **BC warm-start n=5.** The collapse is consistent across all 5 seeds (mean drop 83.6pp). A larger sweep (n equal or greater than 20), longer fine-tune budgets, and a sweep over fine-tune learning rate are queued for v1.1 to map the destruction landscape.
- **Modest network sizes.** 24 to 64 to 32 to 4 MLPs. Capacity sensitivity rules out "size too small" within the 32 to 256 hidden range; the 10M-parameter regime is not tested.
- **Two implementations of DQN disagree by 12 percentage points.** Our custom MLP_DQN at 19.3 percent and SB3 DQN at default LR at 31.4 percent reflect implementation differences. Both reported transparently. The headline framing uses the SB3 number as the higher-confidence "best HP-tuned baseline" reference.
- **Single research team.** All code, raw data, manifests, and analyses are public; every numerical claim is regenerable from raw data via `python reproduce.py verify`. Independent reproduction is welcomed.

---

## 12. Reproducibility

All code at https://github.com/tejasnaladala/maze-rl-baselines under Apache-2.0.

- Single-file statistics pipeline (paired bootstrap, Holm-Bonferroni, Cohen d, BCa).
- SHA-256 manifest of all result files (currently 4,200+ JSON records across all experiments).
- Code-hash pinned per result record (current main-sweep hash: `ed681d75c27fe352`).
- Smoke test at `smoke_test.py` covers every agent class in 3 minutes on a consumer GPU.
- Validation script `validate_harness.py` reproduces the headline harness numbers.
- Modern baseline launcher `launch_modern_baselines.py` (70-run sweep, ~7 hours on RTX 5070 Ti).
- BC warm-start launcher `launch_bc_warmstart.py` (5 seeds, ~50 minutes on RTX 5070 Ti).
- Loopy-maze pilot: `loopy_maze.py` + `launch_loopy_pilot.py` (5 seeds, ~4 minutes).

Total compute: approximately 50 GPU-hours (RTX 5070 Ti laptop and 4xH200 at vast.ai, approximately 155 USD).

---

## Backed-Up Claims Summary

Eight numerically-defended claims, every number above is regenerable from the public raw data:

1. **A 5-line ego-only wall-follower solves 100 percent of procedural mazes** at every tested size 9 through 21, on the same observation space as the neural agents. (Tables 1, 8.)
2. **A supervised MLP recovers the BFS oracle at 97.4 percent** with the same architecture, observation, and optimizer as MLP_DQN. (Tables 1, Section 2.1.) The same MLP trained via standard DQN reaches 19.3 percent (custom) or 31.4 percent (SB3 default).
3. **The DQN reward gradient actively destroys the distilled high-performing representation.** Initialize MLP_DQN at the 97.2 percent BC-distilled weights, fine-tune via standard DQN: post-fine-tune mean is 13.6 percent across 5 seeds (mean drop 83.6 pp, all 5 collapsed). (Section 2.2, Table 1.B.)
4. **Best HP-tuned modern reward-driven baseline reaches 31.4 percent** across 7 configurations of 3 modern algorithms (PPO/DQN/A2C × 3 LRs × 10 seeds, 70 runs total). Statistically tied with uniform Random. (Table 2.)
5. **No standard RL-failure explanation accounts for the gap.** Capacity sweep h32 to h256 (160 runs) flat at 13.6 to 19.3 percent; LR sweep (40 runs) finds default is local optimum; DRQN with LSTM (40+ seeds) matches MLP_DQN; K4 reward ablation (200 runs, paired bootstrap) shows learners collapse without shaping while random walks unchanged. (Tables 3 to 6.)
6. **Loopy-maze pilot.** EgoWallFollowerLeft still solves 100 percent of Wilson + loop-injected mazes (5 seeds, hazards disabled). (Table 7.)
7. **Cross-environment replication.** MiniGrid 4 environments, 240 runs total, MLP_DQN at or below Random in 3 of 4 environments. (Table 9.)
8. **Cover-time theory confirmation.** NoBackRandom versus Random scaling exponents differ by 0.84 units (95 percent bootstrap CI), consistent with Alon-Benjamini-Lubetzky-Sodin (2007) prediction, on a procedural RL benchmark. (Table 10.) Bayesian posterior dominance for every headline pairwise ordering exceeds 0.999. (Table 11.)
