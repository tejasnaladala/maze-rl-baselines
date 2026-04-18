# A 5-Line Ego-Only Wall-Follower Beats Trained Neural Networks by 80 Percentage Points on Procedural Mazes

## The Failure is Exploration, Not Function Approximation

**Single-author draft v1.2** · 2026-04-17 · code + data: https://github.com/tejasnaladala/engram

---

## Abstract

We show that in a hazard-maze benchmark, a five-line egocentric wall-following heuristic solves 100% of 9x9 instances and a BFS-distilled MLP reaches 97.4%, while DQN-family agents (MLP_DQN, DoubleDQN, DRQN) with the same observation class plateau near 19%, below uniform Random (32.7%) and well below a no-backtracking random walk (51.5%).

The heuristic and the distilled MLP both consume the same 24-dimensional ego-feature observation; the distilled MLP uses the same 24 to 64 to 32 to 4 architecture and Adam optimizer as MLP_DQN. This benchmark therefore exposes a sharp gap between policies the network class can represent and policies that standard reinforcement learning discovers from reward.

We rule out every standard explanation for this gap: capacity (a sweep h32 to h256 yields a flat 13.6 to 19.3 percent band), learning rate (the default is the local optimum across 1.5 orders of magnitude), partial observability (DRQN with LSTM matches MLP-DQN), reward shaping (a paired ablation drops reward-driven learners by 4 to 18 percentage points while leaving random walks unchanged), and information asymmetry (the same 24-dimensional ego-features support a 100% wall-following solution). A second environment family (MiniGrid: DoorKey, FourRooms, MultiRoom-N2-S4, Unlock) replicates the headline pattern in 3 of 4 environments. We empirically confirm a non-backtracking cover-time advantage consistent with Alon, Benjamini, Lubetzky and Sodin (2007).

The paper's central, narrow claim: procedural-maze RL evaluations should include hand-coded heuristic, distillation, and random-walk baselines on identical evaluation harnesses. We provide one such audited benchmark and isolate a representation versus discovery gap that is invisible without these baselines.

---

## 1. Headline Result (Table 1)

All numbers on the same audited test harness at 9x9 mazes. n indicates seeds.

| Tier | Agent | Mean success | sd | n |
|---|---|---|---|---|
| 1 Oracle | BFSOracle | 100.0 | 0.0 | 20 |
| 2 Heuristic (5-line, ego-only) | EgoWallFollowerLeft | 100.0 | 0.0 | 20 |
| 3 Distillation (same arch as DQN) | DistilledMLP_from_BFSOracle | 97.4 | 2.5 | 20 |
| 4 Random walk | NoBackRandom | 51.5 | 6.6 | 50 |
| 4 Random walk | LevyRandom (alpha=2.0) | 40.3 | 6.7 | 20 |
| 5 Tabular | FeatureQ_v2 | 36.5 | 8.0 | 50 |
| 4 Random walk | Random | 32.7 | 6.1 | 50 |
| 5 Tabular | TabularQ_v2 | 29.8 | 8.8 | 20 |
| 6 Neural RL | MLP_DQN | 19.3 | 6.7 | 40 |
| 6 Neural RL | DRQN (LSTM) | 19.0 | 10.8 | 40 |
| 6 Neural RL | DoubleDQN | 16.3 | 5.7 | 40 |

**Reading the table.** The agent ladder is monotonically downward across tiers. The 78 percentage point gap between the BFS-distilled MLP (97.4 percent) and MLP-DQN (19.3 percent) is the central observation: same network class, same observation, same optimizer; only the training signal differs.

---

## 2. The Representation versus Discovery Dichotomy (Proof of Claim)

**Setup.** A supervised feedforward MLP, with the exact 24 to 64 to 32 to 4 architecture, the same Adam optimizer, and the same 24-dimensional ego-feature observation as MLP-DQN, is trained on action labels collected from the BFS oracle on the training maze distribution. The model is evaluated deterministically on 50 held-out test mazes per seed.

**Result.** Mean test success 97.4 percent (sd 2.5, n=20 seeds).

**Comparison.** The same architecture trained via standard DQN (epsilon-greedy, target network, replay buffer, identical optimizer and learning rate) reaches 19.3 percent.

**Inference.** The neural policy class can represent the maze-solving policy. Standard reward-driven reinforcement learning does not discover it.

**Falsifiability.** This is a precise, falsifiable claim. To defeat it, exhibit a training procedure (curriculum, demonstrations, intrinsic motivation, larger budget, alternative architecture) that closes the gap from 19.3 percent toward 97.4 percent without changing the network class.

---

## 3. Ruling Out Standard Explanations (Tables 2 to 5)

### Table 2. Capacity is not the bottleneck

| Hidden | 9x9 success | sd | 13x13 success | sd |
|---|---|---|---|---|
| h=32 | 13.6 | 8.1 | 3.0 | 2.3 |
| h=64 (default) | 19.3 | 6.7 | 3.8 | 4.2 |
| h=128 | 15.7 | 8.4 | 4.0 | 3.5 |
| h=256 | 13.6 | 8.3 | 4.8 | 5.0 |

8x capacity scaling produces a flat 13.6 to 19.3 percent band at 9x9. Capacity is not the bottleneck.

### Table 3. Learning rate is at the local optimum

| Learning rate | Mean | sd | n |
|---|---|---|---|
| 1e-4 | 7.4 | 3.7 | 10 |
| 5e-4 (default) | 19.6 | 5.6 | 10 |
| 1e-3 | 11.0 | 6.9 | 10 |
| 3e-3 | 4.8 | 5.3 | 10 |

The default learning rate is the local optimum across 1.5 orders of magnitude. Even at the optimum, MLP-DQN trails NoBackRandom by 32 percentage points.

### Table 4. Reward shaping helps learners, not random walks (paired bootstrap, K4)

| Agent | Full reward | Vanilla reward | Difference | Cohen d | p (Holm-Bonferroni) |
|---|---|---|---|---|---|
| Random | 32.7 | 32.7 | 0.0 | 0.00 | 1.00 |
| NoBackRandom | 51.5 | 51.5 | 0.0 | 0.00 | 1.00 |
| FeatureQ | 35.3 | 17.4 | -17.9 | -2.66 | <0.001 |
| MLP_DQN | 19.3 | 14.6 | -4.7 | -0.61 | 0.014 |
| DoubleDQN | 16.3 | 12.6 | -3.7 | -0.65 | 0.011 |

Removing reward shaping leaves random walks unchanged and hurts every learner, refuting the naive hypothesis that random wins because shaping punishes directed policies. The opposite is true.

### Table 5. Memory does not rescue the failure

DRQN, a recurrent Q-learner with a 64-unit LSTM (sequence length 8) under otherwise identical conditions, reaches 19.0 percent at 9x9 (n=40). Statistically indistinguishable from MLP-DQN. Adding memory does not close the gap to NoBackRandom.

---

## 4. Robustness across Maze Sizes (Table 6)

Mean test success rate, percent. n=20 seeds per cell.

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

The headline ordering holds across sizes 9 to 21. Neural agents collapse with size; random-walk and oracle baselines degrade more gracefully.

---

## 5. Cross-Environment Replication (Table 7)

To check the headline is not specific to our maze generator, we replicate on 4 MiniGrid environments. n=20 seeds per cell.

| Environment | MLP_DQN | NoBackRandom | Random |
|---|---|---|---|
| MiniGrid-DoorKey-5x5 | 0.0 | 9.0 | 9.3 |
| MiniGrid-FourRooms | 0.7 | 3.9 | 2.9 |
| MiniGrid-MultiRoom-N2-S4 | 5.0 | 2.2 | 2.1 |
| MiniGrid-Unlock | 0.0 | 3.8 | 4.3 |

The headline pattern (MLP_DQN at or below Random) replicates in 3 of 4 MiniGrid environments. This second environment family weakens the "tree-maze artifact" critique.

---

## 6. Theory: Cover-Time Scaling (Table 8)

Power-law fits to success rate as a function of maze size n, success_rate(n) = a times n^b. 10,000-resample bootstrap confidence intervals.

| Agent | Exponent b | 95% CI | R^2 |
|---|---|---|---|
| BFSOracle / EgoWallFollower | 0.000 | [0, 0] | constant 100% |
| NoBackRandom | -2.04 | [-2.21, -1.94] | 0.994 |
| LevyRandom (alpha=2.0) | -2.66 | [-2.93, -2.44] | 0.999 |
| Random | -2.88 | [-3.20, -2.59] | 0.996 |
| FeatureQ_v2 | -3.21 | [-3.54, -2.89] | 0.965 |

NoBackRandom decays 0.84 units more slowly than Random across maze sizes, consistent with the Alon, Benjamini, Lubetzky and Sodin (2007) non-backtracking cover-time advantage. To our knowledge this is the first empirical confirmation of this theorem on a procedural reinforcement-learning benchmark.

---

## 7. Bayesian Posterior Dominance (Table 9)

Beta-Binomial conjugate posteriors with uniform prior, computed at 9x9.

| Comparison | Posterior probability |
|---|---|
| NoBackRandom > Random | > 0.999 |
| NoBackRandom > MLP_DQN | > 0.999 |
| NoBackRandom > DoubleDQN | > 0.999 |
| NoBackRandom > FeatureQ_v2 | > 0.999 |
| EgoWallFollowerLeft > NoBackRandom | > 0.999 |
| FeatureQ_v2 > MLP_DQN | > 0.999 |
| Random > MLP_DQN | > 0.999 |

The headline ordering is posterior-certain at the 0.001 level.

---

## 8. Discussion

**Representation versus discovery.** Table 1 row 3 establishes that the MLP architecture used in MLP_DQN (24 to 64 to 32 to 4 with Adam) is sufficient to express a 97.4 percent maze-solving policy. The same architecture trained via DQN converges to a policy that solves 19.3 percent. The reward signal does not lead the optimizer to the policy that the network class can represent.

**Why standard RL finds a low-success local optimum.** A reward decomposition (cover-time analysis at 9x9) shows MLP-DQN's pain-per-step at -0.136 versus Random's -0.238. Neural agents learn the locally rewarding policy component (avoiding walls, hazards, revisits). They fail at the globally rewarding component (sustained exploration through a region of small negative reward toward a sparse +10 goal). When they do solve, they do so near-optimally (0.78 times BFS path length). Their failure is on the fraction of mazes solved, not on path quality.

**Why the random-walk baselines win.** Random walks are stateless and reward-blind. They explore broadly. NoBackRandom adds a one-bit constraint (do not immediately reverse) which produces a 13.6 percent faster cover time consistent with classical non-backtracking random-walk theory. They pay 10 times the BFS-optimal path length per success but reach the goal on 32 to 52 percent of mazes.

**Scope of the claim.** We do not claim that neural function approximation fails in general. We do not claim that deep RL is broken on procedural mazes. We claim: on this audited procedural-maze benchmark, with the observation and reward we specify, standard reward-driven neural RL (DQN, DoubleDQN, DRQN) fails to discover a policy the network class can represent. An audited evaluation that includes a hand-coded heuristic, a supervised distillation, and a random-walk baseline exposes this gap clearly. Procedural-maze RL evaluations should include heuristic, distillation, and random-walk baselines on identical evaluation harnesses. Without them, neural RL results on this class of benchmarks risk being over-interpreted.

---

## 9. Limitations

- Single primary maze class (recursive backtracking with hazards). Results may not extend to other generators or to partially-observable variants beyond DRQN.
- Modest network sizes (24 to 64 to 32 to 4 MLPs). Capacity sensitivity rules out "size too small" within the 32 to 256 hidden range; we did not test the 10M-parameter regime.
- A modern PPO-with-shaped-reward baseline on the main benchmark, full n=10 seeds at 500K environment steps each (same shaped reward, observation, and maze training distribution as MLP_DQN), reaches mean 2.6 percent (sd 3.9, median 1.0, range 0 to 12, per-seed [0, 0, 0, 0, 0, 2, 2, 4, 6, 12]). PPO underperforms uniform Random by 30 percentage points and MLP_DQN by 17 percentage points on this harness. A modern multi-LR sweep across PPO, A2C, and DQN (3 LRs each, 10 seeds each, 70 runs total) is in progress and will land in v1.1.
- A single research team. All code, raw data, manifests, and analyses are public; every numerical claim is regenerable from raw data via `python reproduce.py verify`. Independent reproduction is welcomed.

---

## 10. Reproducibility

All code at https://github.com/tejasnaladala/engram under Apache-2.0.

- Single-file statistics pipeline (paired bootstrap, Holm-Bonferroni, Cohen d).
- SHA-256 manifest of all 4,131 result files.
- Code-hash pinned per result record.
- Smoke test at `smoke_test.py` covers every agent class in 3 minutes on a consumer GPU.
- Validation script `validate_harness.py` reproduces the headline harness numbers.

Total compute: approximately 40 GPU-hours (RTX 5070 Ti laptop and 4xH200 at vast.ai, approximately 155 USD).

---

## Backed-Up Claims Summary

Six numerically-defended claims, every number above is regenerable from the public raw data:

1. **A 5-line ego-only wall-follower solves 100% of procedural mazes** at every tested size 9 through 21, on the same observation space as the neural agents. (Table 1, Table 6.)
2. **A supervised MLP recovers the BFS oracle at 97.4%** with the same architecture, observation, and optimizer as MLP-DQN. (Table 1, §2.) The same MLP trained via DQN reaches 19.3%.
3. **No standard RL-failure explanation accounts for the gap.** Capacity sweep h32 to h256 (160 runs) flat at 13.6 to 19.3%; LR sweep (40 runs) finds default is local optimum; DRQN with LSTM (40+ seeds) matches MLP-DQN; K4 reward ablation (200 runs, paired bootstrap) shows learners collapse without shaping while random walks unchanged. (Tables 2 to 5.)
4. **Headline replicates in a second environment family.** MiniGrid 4 environments, 240 runs total, MLP-DQN at or below Random in 3 of 4 environments. (Table 7.)
5. **Cover-time theory confirmation.** NoBackRandom versus Random scaling exponents differ by 0.84 units (95% bootstrap CI), consistent with Alon-Benjamini-Lubetzky-Sodin 2007 prediction, on a procedural RL benchmark for the first time we know of. (Table 8.)
6. **Bayesian posterior dominance** for every headline pairwise ordering exceeds 0.999. (Table 9.)
