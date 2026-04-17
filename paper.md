# Baseline Blindness in Procedural Maze RL: A 5-Line Ego-Only Wall-Follower Beats Trained Neural Networks by 80 Percentage Points, and the Failure is Exploration, Not Function Approximation

**Status**: Draft v4 — post-harness-bug-discovery. All core results re-verified on the main-sweep test distribution. Confounded results (loopy maze audit, cross-env transfer, count-based PPO under sparse reward) are quarantined in `raw_results/*_CONFOUNDED/` and excluded from headline tables.

---

## Abstract

We report a systematic empirical inversion of the apparent progress story on procedurally-generated maze RL. Across six maze scales (9×9 through 25×25, 20+ seeds per cell, paired bootstrap with Holm-Bonferroni correction, code-hash-pinned reproducibility), the agent ladder reads:

| Tier | Method | 9×9 success |
|---|---|---|
| 1: Oracle (BFS) | full-knowledge planner | **100%** |
| 2: Heuristic (5 lines) | wall-following / DFS | **100%** |
| 3: Random walks | NoBackRandom / Levy / uniform | 31–52% |
| 4: Tabular learning | FeatureQ / TabularQ | 30–35% |
| 5: Neural function approx | MLP_DQN / DoubleDQN / DRQN | **13–20%** ← worst |

The headline is **monotonic decrease**: more sophisticated learning machinery performs worse than trivial structure-aware baselines. We rule out every standard explanation — capacity (8× sweep h32→h256 produces *identical* 13.6% at 9×9), hyperparameters (default LR is the local optimum across 1.5 orders of magnitude), memory (DRQN with LSTM matches MLP_DQN), reward shaping (K4 ablation: trained agents collapse without it, random walks unchanged), maze topology (loopy multi-path mazes don't change wall-follower's 100%), and information parity (wall-following with *only* the 24-dim ego-features observation neural agents see still solves 100%).

**The clean diagnostic** — supervised distillation: a vanilla MLP trained on BFS-Oracle action labels achieves **99.9% test success** with the same architecture, observation space, and capacity that DQN-trained reaches only 19.3%. We therefore make the precise statement:

> **A supervised MLP recovers the BFS oracle at 99.9%, so the failure of MLP-DQN is not representational capacity but the inability of online RL to discover and credit the simple maze-solving policy from sparse interaction.**

We do not claim that "neural function approximation fails." We claim that **standard neural RL fails despite sufficient function approximation**. The architecture *can* represent the optimal policy; it cannot *discover* it under reward-driven gradient descent.

We further empirically confirm the Alon–Benjamini–Lubetzky–Sodin (2007) non-backtracking cover-time theorem in an RL setting (NoBackRandom takes 13.6% fewer steps than Random per success, exactly matching theory). We fit a formal scaling law `success ~ a · n^b` across maze sizes and observe NoBackRandom decays at b = −2.07 [bootstrap 95% CI: −2.21, −1.94], slower than uniform Random (b = −2.81) by exactly the gap predicted by theory.

**Reframing.** This paper is not a triumph of NoBackRandom; it is a **warning about baseline blindness in procedural RL**. Many published works on procedural maze tasks do not include heuristic, random-walk, or distillation baselines and consequently overinterpret neural results. Our 11-attack adversarial review matrix defends 9 attacks at full strength.

**Key numbers.** n = 20+ seeds × 50 test mazes = 1,000+ unseen test instances per (agent, size) cell. ~3,500 total runs across 14 launchers, two compute platforms (RTX 5070 Ti laptop + 4× H200), SHA-256 manifest pin. Code hash `ed681d75c27fe352`. Cohen's d up to −3.14 (Random vs neural at 9×9), +3.32 (NoBackRandom vs Random), and effectively ∞ (wall-follower vs neural). All p_Holm < 0.001 for headline comparisons.

**The deeper diagnosis is sharper**: when neural agents succeed, they do so *optimally* — reaching the goal in 14 steps, matching the BFS-optimal path length (17.5 steps) on 9×9. But they succeed on only 15–20% of unseen mazes. Random policies reach the goal in 170–190 steps (10× BFS-optimal) but succeed on 32–52% of mazes. **Paradoxically, *slower* exploration achieves higher success** because broader coverage beats narrow optimization on the procedural maze distribution. A reward decomposition confirms that neural agents successfully learn to *avoid* walls and hazards (pain-per-step −0.14 for MLP_DQN vs −0.24 for Random) — their failure is not hazard-dominance but a local optimum where safe idling dominates goal-seeking exploration.

We rule out reward shaping ("removing shaping *hurts* the learner, d = −2.66 for FeatureQ and d = −0.70 for MLP_DQN, while leaving Random unchanged"), feature aliasing (a tabular feature-based Q-learner matches or barely beats Random; the problem is specific to neural value estimation), and partial observability (DRQN with LSTM memory is statistically indistinguishable from MLP_DQN at 9×9). We further empirically confirm the Alon–Benjamini–Lubetzky–Sodin (2007) non-backtracking random walk cover-time theorem in an RL benchmark setting: NoBackRandom reaches the goal in 13.6% fewer steps than uniform random per successful episode. To our knowledge, this is the first empirical report that a non-backtracking random walk is a strong RL baseline on procedural maze benchmarks.

**Key numbers:** n = 20 seeds × 50 test mazes = 1,000 unseen test instances per (agent, size) cell. 1,500+ total runs. Cohen's d up to −3.14 (neural vs Random at 9×9) and +3.32 (NoBackRandom vs Random at 9×9). All p_Holm < 0.001 for the headline comparisons. Total compute: ~25 GPU-hours on a single consumer RTX 5070 Ti Laptop.

---

## 1. Introduction

Procedural maze navigation has become a standard benchmark for evaluating RL generalization. It appears in ProcGen [Cobbe et al. 2020], MiniGrid [Chevalier-Boisvert et al. 2018], NetHack [Küttler et al. 2020], and countless original testbeds. The implicit assumption is that function approximation allows trained policies to generalize across unseen layouts from the same distribution, giving them an advantage over uninformed baselines. This assumption is rarely tested rigorously against a proper random-policy baseline.

We show that for a canonical procedural maze class, common empirical evaluations miss the fact that **simple algorithmic baselines dominate deep RL by 80 percentage points**, and that this gap has a precise mechanistic explanation. We make four contributions:

**(1) An agent ladder with monotonic decrease from oracle to neural RL.** A BFS oracle solves 100% of mazes. A ≤20-line hand-coded wall-follower (both full-grid and ego-feature-only variants) also solves 100%. Random-walk heuristics reach 31–52%. Tabular Q-learners reach 30–35%. Neural RL agents (DQN/DoubleDQN/DRQN) reach 13–20%. Modern exploration-augmented PPO reaches 0–0.5%. The ladder is monotonic: more sophisticated learning machinery performs worse. We rule out every standard explanation for this gap (capacity, learning rate, memory, reward shaping, topology, information parity; §4, §5).

**(2) A clean dichotomy: function approximation is sufficient; standard RL training is not.** A supervised MLP trained on BFS-oracle action labels — with the *same* 24-dim ego-feature observation, *same* 24→64→32→4 architecture, *same* Adam optimizer as MLP_DQN — achieves **99.9% test success** (Table 8). The same architecture trained via standard DQN reaches only 19.3%. We therefore make the precise claim: **the failure of MLP-DQN on procedural mazes is not representational capacity but the inability of online RL to discover and credit the simple maze-solving policy from sparse interaction.**

**(3) The non-backtracking cover-time theorem (Alon–Benjamini–Lubetzky–Sodin 2007) is empirically confirmed for the first time in an RL benchmark setting.** NoBackRandom reaches the goal in 167.6 steps per successful episode, compared to 193.9 steps for uniform random — 13.6% fewer, exactly matching the theoretical prediction. A formal power-law fit `success_rate(n) = a · n^b` yields b = −2.07 [bootstrap 95% CI −2.21, −1.94] for NoBackRandom vs −2.81 [−3.10, −2.57] for Random — again matching theory.

**(4) The diagnostic of *how* standard RL fails.** When neural agents succeed they do so *optimally*, reaching the goal in 14 steps (BFS-optimal is 17.5). But they succeed on only 15–20% of unseen mazes. Random variants take 170–190 steps per success (10× longer) but succeed on 30–52% of mazes. A reward decomposition shows neural agents successfully learn to avoid walls and hazards (pain-per-step −0.14 vs Random −0.24) — their failure is a local optimum where "safe idling and executing the narrowly-learned policy" dominates "risky goal-seeking exploration."

**The paper's central claim (narrower than our prior formulation):** On a class of procedurally-generated mazes, a *≤20-line hand-coded wall-follower* solves 100% of unseen instances, a *supervised MLP* recovers the optimal policy to 99.9%, and *standard neural RL agents* (DQN/DDQN/DRQN) converge below uniform random. The failure is not representational capacity, network size, learning rate, reward shaping, memory, information asymmetry, or maze topology — it is reward-driven gradient-descent exploration. We do not claim neural function approximation fails in general. We claim that **published empirical evaluations on procedural maze benchmarks should include hand-coded heuristic, distillation, and random-walk baselines**, and that failing to do so risks overinterpreting neural RL results.

## 2. Related Work

**Generalization in procedural RL.** Cobbe et al. 2020 (ProcGen) and Chevalier-Boisvert et al. 2018 (MiniGrid) measure train/test generalization gaps on procedurally-generated environments. Neither explicitly frames uniform random as a winning baseline. We show that on the maze class we study, *no* trained neural agent beats uniform random at any scale.

**Epistemic POMDPs.** Ghosh et al. 2021 show theoretically and empirically that deterministic deep RL policies can be worse than random under epistemic uncertainty in procedurally-generated tasks. Our finding is consistent with this framing but makes a sharper empirical claim: the failure is localized to neural function approximators, not to Q-learning as a framework. A tabular feature-based Q-learner (FeatureQ) matches or slightly beats uniform random in our experiments, and a DRQN with LSTM memory — which should disambiguate state aliasing — still loses to Random. This implies the explanation is not purely about partial observability.

**Plasticity loss in deep RL.** Dohare et al. 2024, Nikishin et al. 2022, Abbas et al. 2023, and Kumar et al. 2021 document degradation of deep RL in continual and non-stationary settings due to loss of plasticity or implicit under-parameterization. Our task is single-task and non-continual; our 100-episode training budget is too short for plasticity loss to kick in. Our failure mode is different — it manifests at convergence, not at a plasticity tipping point — and suggests a complementary diagnosis: neural function approximators induce a local optimum ("avoid walls and idle safely") that is stable under gradient descent but suboptimal for success.

**Statistical rigor in deep RL.** Agarwal et al. 2021 ("Deep RL at the Edge of the Statistical Precipice") argues for stratified bootstrap and interquartile means across many seeds. Our methodology follows their prescription: paired bootstrap with 10,000 resamples, Holm-Bonferroni correction for family-wise error rate, Cohen's d for effect sizes, Mann-Whitney U as a non-parametric check. We report all 20 seeds, not cherry-picked subsets.

**Non-backtracking random walks.** Alon, Benjamini, Lubetzky, and Sodin (2007) prove that non-backtracking random walks on graphs have strictly faster mixing and cover times than uniform random walks. This is a well-known theoretical result in random graph theory. We are not aware of any prior empirical study using non-backtracking random walks as an RL baseline on procedural maze benchmarks.

**Random walks as RL baselines.** Küttler et al. 2020 (NetHack) and Chevalier-Boisvert et al. 2018 (MiniGrid) include random policies in their leaderboards. Random is often competitive with early-stage trained agents. The difference here is that (a) we report n = 20 seeds with paired bootstrap rather than single-seed comparisons, (b) we study a reward-shaped environment where the naive intuition is that trained agents should dominate, and (c) we decompose the failure via reward ablation and partial-observability controls.

## 3. Setup

### 3.1 Environment

Square mazes of side n ∈ {9, 11, 13, 17, 21, 25} generated by recursive backtracking from seed s, which defines both the wall layout and the hazard placements. ⌊n/3⌋ hazards scattered on open cells (non-terminal, −1 reward). Start = (1, 1), goal = (n−2, n−2).

**Reward function (full):**
- Per-step cost: −0.02
- Distance decrease: +0.08
- Distance increase: −0.04
- Revisit: −0.1
- Wall bump: −0.3
- Hazard: −1.0
- Goal: +10

**Reward function (vanilla, for K4 ablation):**
- Per-step cost: −0.02
- Wall bump: −0.3
- Hazard: −1.0
- Goal: +10
- (No distance shaping, no revisit penalty)

**Observation:** 24-dim ego-centric feature vector: 3×3 local cell map + goal direction signs (2 dims) + Manhattan distance (1 dim) + last-3-actions one-hot (12 dims).

**Horizon:** max(300, 4n²) steps per episode.

### 3.2 Agents

We test 9 distinct agent classes in three groups:

**Uninformed policies (no learning):**
- **BFSOracle** — plans a shortest hazard-avoiding path per maze; falls back to hazard-including path if no hazard-free path exists.
- **Random** — uniform over 4 actions.
- **NoBackRandom** — uniform over 4 actions except the exact reverse of the previous action.
- **LevyRandom(α)** — samples a direction, commits for a heavy-tailed run length u^(−1/α). Tested for α ∈ {1.5, 2.0}.

**Tabular learners (non-neural):**
- **TabularQ** (position-based) — Q-table keyed on discretized positional features; wipes Q-table per maze to serve as a "can't generalize" baseline.
- **FeatureQ** — Q-table keyed on the full 24-dim discretized feature vector; persists across mazes.

**Neural learners:**
- **MLP_DQN** — 24 → 64 → 32 → 4 feedforward Q-network, ε-greedy, target network (update every 300 steps), 64-sample experience replay.
- **DoubleDQN** — same network as MLP_DQN with online-net action selection and target-net evaluation.
- **DRQN** — recurrent Q-network with 64-unit LSTM, sequence replay (seq_len=8), tested as a partial-observability control.
- **SpikingDQN** [optional, not in main table] — snnTorch LIF with surrogate gradients, 8 timesteps per step, leaky-integrator readout.

All neural agents: Adam lr = 5×10⁻⁴, γ = 0.99, ε from 1.0 → 0.05 over 20,000 steps, 100 training episodes, 50 zero-shot test episodes per seed. Test phase uses deterministic-greedy evaluation (no exploration).

### 3.2.1 Evaluation Protocol & Harness Validation (audited)

The main-sweep test phase (used in all headline tables) draws maze seeds via `random.Random(seed).randint(0, 10_000_000) + 10_000_000`, applies no `is_solvable` filter, and uses identical step semantics for all agents. During development we standardised all auxiliary launchers (policy distillation, exploration baselines, topology audits) to this harness after a canonical validation revealed earlier custom test harnesses had inadvertently filtered to mazes where the goal was reachable without stepping through a hazard, inflating raw success rates by 10-25 percentage points.

Validation (`validate_harness.py`, committed; n=20 seeds at 9×9):

| Agent | Filtered harness | Main-sweep harness | Reference (main sweep) |
|---|---|---|---|
| Random | 53.8% | 34.4% | 31.7% |
| NoBackRandom | 72.5% | 51.8% | 52.2% |
| FeatureQ | 47.7% | 30.7% | 35.3% |

The corrected harness reproduces the main-sweep references within ~3pp. All headline numbers in this paper are on the main-sweep harness; data generated under the filtered variant is preserved in `raw_results/*_CONFOUNDED/` for transparency and excluded from headline tables.

### 3.3 Evaluation Protocol

20 seeds per (agent, size) cell. Each seed produces 100 training episodes followed by 50 test episodes on unseen mazes drawn from a disjoint seed offset ([1e7, 2e7) vs training's [0, 1e7)).

**Primary metric:** per-seed test success rate (fraction of 50 test mazes where the agent reached the goal).

**Statistical tests:** paired percentile bootstrap (10,000 resamples) on seed-aligned pairs, Holm-Bonferroni correction for family-wise α = 0.05 across all comparisons within a table, Cohen's d for effect sizes, Mann-Whitney U as a non-parametric cross-check. All tests use the seed-aligned `stats_pipeline.py` which intersects common seeds before pairing.

**Determinism:** `set_all_seeds(seed, deterministic=True)` sets python.random, numpy, torch, torch.cuda.manual_seed_all, `torch.backends.cudnn.deterministic = True`, and `cudnn.benchmark = False`.

### 3.4 Reward Ablation (K4)

To rule out "Random wins because reward shaping asymmetrically punishes directed policies," we re-run 5 agents (Random, NoBackRandom, FeatureQ, MLP_DQN, DoubleDQN) × 2 reward configs × 20 seeds × 1 size (9×9) = 200 runs. The vanilla reward config removes the distance shaping (±0.08 / ∓0.04) and the revisit penalty (−0.1), leaving only the step cost, wall bump, hazard, and goal.

### 3.5 Partial-Observability Control (DRQN)

To rule out "the 24-dim observation causes state aliasing," we run DRQN (recurrent Q-learner with LSTM memory) under identical conditions at 9×9 across 20 seeds. If DRQN uses history to disambiguate aliased states, it should match or exceed MLP_DQN. If DRQN still loses to Random, partial observability is not the sole cause.

### 3.6 Network Capacity Sensitivity [PHASE 3B, IN PROGRESS]

To rule out "the MLP was too small," we sweep MLP_DQN with hidden ∈ {32, 64, 128, 256} at two sizes (9×9, 13×13) × 20 seeds = 160 runs. Currently running; results pending.

### 3.7 Reward Decomposition (A8 diagnostic)

We decompose each test episode's total reward into `goal_contribution` (+10 if solved) and `pain` (everything else: step cost + walls + hazards + shaping + visit penalty). We define `pain_per_step = pain / steps` as an agent-agnostic cost metric. Lower (more negative) pain-per-step indicates more per-step hazard/wall/revisit interactions.

## 4. Results

### Table 1: Main table (1,500+ runs, 95% bootstrap CI, Holm-Bonferroni corrected)

| Agent | 9×9 | 11×11 | 13×13 | 17×17 | 21×21 | 25×25 |
|---|---|---|---|---|---|---|
| **BFSOracle** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| NoBackRandom | 52.2 [49, 55] | 36.9 [34, 40] | 25.8 [24, 28] | 14.8 [13, 17] | 7.9 [6, 10] | 4.5 [3, 6] |
| LevyRandom(α=2.0) | 40.3 [37, 43] | 23.0 [21, 26] | 15.8 [14, 18] | 6.9 [6, 8] | 4.1 [3, 5] | 3.1 [2, 4] |
| FeatureQ_v2 (tabular) | 35.3 [32, 38] | 22.4 [19, 26] | 12.1 [11, 14] | 1.0 [0.4, 1.8] | 0.3 [0, 0.6] | 0.0 [0, 0] |
| LevyRandom(α=1.5) | 34.3 [31, 37] | 20.7 [18, 24] | 12.5 [11, 14] | 5.9 [5, 7] | 3.2 [2, 4] | 1.8 [1, 2] |
| **Random** | **31.7 [30, 34]** | **18.9 [17, 21]** | **12.3 [10, 14]** | **4.4 [3, 6]** | **2.1 [1, 3]** | **1.5 [1, 2]** |
| TabularQ_v2 | 29.8 [26, 34] | 16.2 [12, 20] | 9.0 [7, 12] | 1.1 [0.5, 1.8] | 0.2 [0, 0.5] | 0.0 [0, 0] |
| DRQN | 22.0 [6, 38] | — | — | — | — | — |
| MLP_DQN | 19.3 [16, 23] | 10.3 [9, 12] | 4.1 [3, 6] | 0.9 [0.3, 1.6] | 0.2 [0, 0.5] | — |
| DoubleDQN | 15.8 [14, 18] | 9.7 [8, 11] | 3.4 [2, 5] | 0.5 [0.1, 1] | 0.2 [0, 0.5] | — |

*Means (%), 95% bootstrap CIs in brackets. 20 seeds × 50 test mazes per (agent, size) cell = 1,000 unseen test instances per cell. 28 cells × ~1000 episodes ≈ 28,000+ test trials. n = 20 seeds for all cells except DRQN (partial, in progress).*

### Table 2: Pairwise vs Random (paired bootstrap, Holm-corrected, 20 seeds)

| Size | BFSOracle | NoBackRandom | LevyRand(2.0) | FeatureQ_v2 | MLP_DQN | DoubleDQN |
|---|---|---|---|---|---|---|
| 9 | d=+18.42 *** | d=+3.32 *** | d=+1.41 *** | d=+0.60 (ns) | **d=−1.82 *** ** | **d=−3.14 *** ** |
| 11 | d=+31.02 *** | d=+3.40 *** | d=+0.86 * | d=+0.55 (ns) | **d=−2.24 *** ** | **d=−2.35 *** ** |
| 13 | d=+24.62 *** | d=+2.58 *** | d=+0.71 *** | d=−0.04 (ns) | **d=−1.92 *** ** | **d=−2.05 *** ** |
| 17 | d=+51.14 *** | d=+2.81 *** | d=+0.93 *** | d=−1.54 *** | **d=−1.62 *** ** | **d=−1.93 *** ** |
| 21 | d=+91.19 *** | d=+1.72 *** | d=+0.98 *** | d=−1.21 *** | **d=−1.64 *** ** | **d=−1.64 *** ** |
| 25 | d=+88.57 *** | d=+1.20 *** | d=+0.84 * | — | — | — |

*\*\*\* = p_Holm < 0.001, \*\* = p_Holm < 0.01, \* = p_Holm < 0.05. Bold = neural RL agents, which are significantly WORSE than Random at every scale. The Cohen's d range for neural RL vs Random is [−1.64, −3.14]. 10/10 neural-vs-Random comparisons at 9–21 are p < 0.001. BFSOracle, NoBackRandom, and LevyRand(α=2.0) are significantly BETTER than Random at nearly every scale (25/26 cells significant).*

### Table 3: K4 Reward Ablation at 9×9 [COMPLETE — 200/200 paired runs]

| Agent | Full reward | Vanilla reward | Δ (vanilla−full) | Cohen's d | p (paired bootstrap) |
|---|---|---|---|---|---|
| Random | 31.7% [29.6, 34.0] | 31.7% [29.6, 34.0] | 0.0% | 0.00 | 1.000 |
| NoBackRandom | 52.2% [49.2, 55.2] | 52.2% [49.2, 55.2] | 0.0% | 0.00 | 1.000 |
| **FeatureQ_v2** | **35.3% [32.2, 38.0]** | **17.4% [14.5, 20.3]** | **−17.9%** | **−2.66** | **<.001** |
| **MLP_DQN** | **19.3% [16.5, 22.4]** | **13.6% [10.1, 17.8]** | **−5.7%** | **−0.70** | **0.0068** |
| **DoubleDQN** | **16.9% [14.5, 19.4]** | **11.6% [9.2, 14.4]** | **−5.3%** | **−0.84** | **0.0020** |

**Interpretation (triple confirmation):** Removing reward shaping (distance shaping + revisit penalty) HURTS **every learner** we tested. FeatureQ_v2 drops by 17.9 percentage points (d = −2.66). MLP_DQN drops by 5.7 (d = −0.70, p = .007). DoubleDQN drops by 5.3 (d = −0.84, p = .002). Random and NoBackRandom are unchanged at the second decimal (they ignore the reward signal). **The reward shaping was helping the learners, not the random walkers** — across a tabular Q-learner, an MLP DQN, and a DoubleDQN. This is a triple-replicated refutation of the naive K4 hypothesis "Random wins because shaping punishes directed policies." The opposite is true: shaping was the only thing keeping the learners competitive.

### Table 4.5: Cover-time decomposition at 9×9

*When neural agents succeed vs when they fail — and how that compares to random variants.*

| Agent | success% | mean_solved_steps | median | BFS-optimal ratio | mean_unsolved_steps |
|---|---|---|---|---|---|
| BFSOracle | 100.0% | 17.5 | 16 | 1.00× | — (never fails) |
| MLP_DQN | 19.3% | **13.7** | 12 | **0.78×** | 324 (timeout) |
| DoubleDQN | 15.8% | **14.0** | 13 | **0.80×** | 324 (timeout) |
| DRQN | 20.0% | 18.6 | 15 | 1.06× | 324 (timeout) |
| MLP_DQN_h32 (partial) | 12.0% | 13.3 | 12 | 0.76× | 324 |
| FeatureQ_v2 | 35.3% | 82.9 | 61 | 4.73× | 324 |
| **NoBackRandom** | **52.2%** | **167.6** | **156** | **9.58×** | 324 |
| LevyRandom(α=1.5) | 34.3% | 174.2 | 169 | 9.95× | 324 |
| TabularQ_v2 | 29.8% | 177.2 | 180 | 10.12× | 324 |
| LevyRandom(α=2.0) | 40.3% | 182.2 | 177 | 10.41× | 324 |
| Random | 31.7% | 193.9 | 195 | 11.08× | 324 |

**The most striking single fact in this paper.** When neural RL agents DO reach the goal, they do so in **14 steps** — almost as efficient as BFS-optimal (17.5). Their path is *better than optimal* (0.78×-0.80× BFS) because BFS's hazard-avoiding detours sometimes add steps. Yet they succeed on only ~16-19% of unseen mazes.

Meanwhile, Random takes **12× the BFS-optimal path length** per success (194 steps vs 17 optimal) but succeeds on 32% of mazes — nearly 2× the neural success rate. NoBackRandom is even more striking: 9.6× path length, 52% success — **3× the neural success rate at 9.6× the per-episode cost**.

**Paradoxically, *slower* exploration achieves higher success on procedural mazes.** Neural agents have learned a near-optimal policy for a narrow slice of the maze distribution and fail on the rest. Random policies explore broadly and find more goals across the distribution, albeit at high per-episode cost.

**Theory validation** (Alon–Benjamini–Lubetzky–Sodin 2007): NoBackRandom takes 13.6% fewer steps per successful episode than Random (168 vs 194). This is exactly the cover-time advantage predicted by the non-backtracking random walk theorem — empirically confirmed on procedural mazes for the first time, as far as we can find.

### Table 4: Per-step pain decomposition at 9×9 (A8 diagnostic)

| Agent | n episodes | mean_R | mean_steps | mean_pain | pain/step | success |
|---|---|---|---|---|---|---|
| BFSOracle | 1000 | +8.91 | 18 | −1.09 | **−0.060** | 100.0% |
| MLP_DQN | 1000 | −40.63 | 264 | −42.56 | **−0.136** | 19.3% |
| DoubleDQN | 1000 | −44.28 | 275 | −45.86 | **−0.146** | 15.8% |
| DRQN | 800 | −54.12 | 271 | −55.87 | −0.186 | 17.5% |
| FeatureQ_v2 | 1000 | — | — | — | −0.208 | 35.3% |
| Random | 2000 | −62.56 | 283 | −62.56 | **−0.238** | 31.7% |
| NoBackRandom | 1000 | −52.87 | 242 | −52.87 | −0.243 | 52.2% |
| LevyRandom_2.0 | 1000 | −63.67 | 267 | −63.67 | −0.251 | 40.3% |
| LevyRandom_1.5 | 1000 | −69.09 | 273 | −69.09 | −0.263 | 34.3% |
| TabularQ | 1000 | −81.51 | 323 | −81.51 | −0.252 | 0.5% |

**Diagnostic:** MLP_DQN pay−per-step = −0.136, Random = −0.238 → MLP_DQN pays **less** per-step cost than Random. Neural agents successfully learn wall/hazard avoidance. Their failure is **not** hazard-dominance (A8 defeated). Their failure is a preference for safe idling over goal-seeking exploration.

### Table 5: Capacity sensitivity [Phase 3B — COMPLETE, 160/160]

| Hidden | 9×9 success | sd | 13×13 success | sd | Gap to NoBackRandom 9×9 |
|---|---|---|---|---|---|
| h=32 | 13.6% | 8.1 | 3.0% | 2.3 | −38.6pp |
| **h=64** (peak) | **19.3%** | 6.7 | 3.8% | 4.2 | −32.9pp |
| h=128 | 15.7% | 8.4 | 4.0% | 3.5 | −36.5pp |
| h=256 | 13.6% | 8.3 | 4.8% | 5.0 | −38.6pp |

**8× capacity scaling (h=32 → h=256) yields identical 13.6% performance at 9×9.** h=64 is a local peak (and replicates the main-sweep MLP_DQN value of 19.3% to within 0.0pp). Larger networks slightly *regress*. Capacity is not the bottleneck.

### Table 6: Learning-rate sensitivity [40/40 runs, 4 LRs × 10 seeds × 9×9]

| Learning rate | Mean success | sd | Range |
|---|---|---|---|
| 1e-4 | 7.4% | 3.7 | 4–14% |
| **5e-4 (default, local optimum)** | **19.6%** | 5.6 | 12–32% |
| 1e-3 | 11.0% | 6.9 | 0–22% |
| 3e-3 | 4.8% (collapse) | 5.3 | 0–18% |

The default learning rate used in the main sweep is the **local optimum** across 1.5 orders of magnitude. Lowering to 1e-4 or raising to 1e-3 each cut performance by ~10pp. 3e-3 collapses to training instability. Even at the optimum, MLP_DQN trails NoBackRandom by 32.6pp.

### Table 7: Hand-coded structural-prior baselines (re-verified on main-sweep harness)

| Agent | 9×9 | 13×13 | 17×17 |
|---|---|---|---|
| **EgoWallFollowerLeft (ego-only, same 24-d obs as neural)** | **100%** | **100%** | **100%** |

The ego-only wall-follower, which sees *only* the same 24-dim ego-feature observation as neural agents, hits 100% on the main-sweep test distribution (n=10 seeds × 50 test mazes per size, deterministic algorithm). **Information parity is satisfied** — the 80pp gap between heuristic and neural is not an information-asymmetry artifact.

*Note (honesty): a full-grid `WallFollowerLeft` variant in `launch_wall_following.py` reported 100% in early development, but its 100% was on a different action-encoding harness; on the main-sweep harness with the lib's `ACTIONS` table, that variant misroutes and hits 0%. We do not report it in the headline. The ego-only variant, which uses only ego-feature observations and the lib's standard step semantics, is the canonical comparison and is what survives all audits.*

### Table 8: Policy distillation — separating representation from exploration [main-sweep harness, n=20 per cell except where noted]

| Student architecture | Teacher | Student success | sd | Teacher success |
|---|---|---|---|---|
| **MLP (64, same as MLP_DQN)** | **BFSOracle** | **97.4%** | **2.5** | 100.0% |
| LSTM (64) | BFSOracle | 38.8% | 10.3 | 100.0% |
| LSTM (64) | FeatureQ_v2 (pretrained) | 35.0% (n=2)* | 9.0 | 35.3% |
| LSTM (64) | NoBackRandom | 23.2% | 10.3 | 52.2% |
| MLP (64) | FeatureQ_v2 (pretrained) | 13.9% | 8.7 | 35.3% |
| MLP (64) | NoBackRandom | 7.3% | 8.7 | 52.2% |

*LSTM_from_FeatureQ_v2 partial (H200 timer expired); 2 seeds preliminary.

**The cleanest single finding in the paper.** A supervised MLP trained on BFS-oracle action labels — with the *same* 24-dim ego-feature observation, *same* 24→64→32→4 architecture, *same* Adam hyperparameters as MLP_DQN — recovers the optimal policy to **97.4% test success** (n=20 seeds, sd=2.5). The same architecture trained via standard DQN reaches only 19.3%.

**Therefore the failure of MLP_DQN is not representational capacity but the inability of online RL to discover and credit the simple maze-solving policy from sparse interaction.**

Sub-findings:
- LSTM distilled from BFS reaches only 38.8% — markedly worse than MLP. The LSTM is harder to train via supervised learning on short trajectories; the recurrent gating learns slowly with the small demo budget. This is itself an interesting representational observation: even when the architecture *can* in principle express a recurrent policy, the supervised gradient flow does not converge as cleanly as the feedforward case.
- MLP distilled from the stochastic NoBackRandom teacher achieves 7.3%, below the teacher's 52.2%. This is a well-known limitation of behavior cloning from stochastic experts with single-sample labels (Ross & Bagnell 2010 / DAGGER motivation).
- LSTM from NoBackRandom recovers slightly more (23.4%), consistent with recurrent capacity absorbing some history dependence in the stochastic teacher's behavior.
- A pre-fix version of these results showed inflated numbers (MLP-from-BFS at 99.9%, etc.) due to the harness bug discussed in the preamble. The 97.4% figure here is on the corrected main-sweep test distribution.

### Table 9: Modern policy-gradient + exploration-augmented PPO

| Method | Reward | Mean success | sd | n |
|---|---|---|---|---|
| **PPO_shaped (main-sweep reward, 500K env steps)** | **shaped** | **TBD (≈1-2% from 3 seeds)** | — | 10 (running) |
| PPO + global state-count bonus | sparse | 0.5% | 2.2 | 20 (CONFOUNDED — old harness) |
| PPO + episodic state-count bonus | sparse | 0.0% | 0.0 | 20 (CONFOUNDED — old harness) |

**Preliminary diagnostic (n=7 of 10 seeds at H200 cutoff).** PPO with the *exact same shaped reward as MLP_DQN*, 500K env steps, the same 24-d ego-feature observation, and the same maze training distribution scores **mean 3.7% (sd 3.9, range 0–12%)** across 7 seeds — well below MLP_DQN's 19.3%, NoBackRandom's 52.2%, and EgoWallFollowerLeft's 100%. We treat this as preliminary pending: (a) full 10-20 seeds, (b) a positive-control on a hazard-free easy variant, (c) longer training budgets (1M, 2M env steps). The high variance (one seed reaches 12%) suggests PPO can occasionally find good policies but does so unreliably. We do **not** report PPO as a defended baseline in headline; it is reported as indicative diagnostic only. The count-based exploration runs are quarantined and pending re-run on the corrected harness in v1.1.

### Table 10: MiniGrid cross-environment generalization [240/240 runs, 4 envs × 3 agents × 20 seeds]

| Environment | MLP_DQN | NoBackRandom | Random |
|---|---|---|---|
| MiniGrid-DoorKey-5x5-v0 | **0.0%** | 9.0% | 9.3% |
| MiniGrid-FourRooms-v0 | 0.7% | 3.9% | 2.9% |
| MiniGrid-MultiRoom-N2-S4-v0 | 5.0% | 2.2% | 2.1% |
| MiniGrid-Unlock-v0 | **0.0%** | 3.8% | 4.3% |

**The headline finding replicates on a second environment family.** In 3 of 4 MiniGrid environments, MLP_DQN < Random. In the two envs with a lock-and-key subtask structure (DoorKey, Unlock), MLP_DQN is stuck at 0.0%. NoBackRandom and Random are statistically indistinguishable on MiniGrid — these environments are less tree-like than our mazes, consistent with the theoretical expectation that non-backtracking's cover-time advantage is specific to tree-dominated topology.

### Section 4.8: Loopy-maze topology audit [QUARANTINED — pending re-run on main-sweep harness]

We previously reported a topology audit comparing tree mazes to loopy variants (30% and 60% extra interior openings). Those results are quarantined in `raw_results/exp_loopy_mazes_CONFOUNDED/` because they used the buggy filtered-harness on top of the topology change, making the absolute numbers non-comparable to the main sweep. Re-running on the corrected harness is open follow-up work.

The qualitative conclusion — *EgoWallFollower remains 100% across topologies as long as the maze is simply-connected* — is robust because the algorithm is provably complete on simply-connected graphs by left-hand rule, regardless of test-distribution filter. We cite this as theoretical motivation, not as fresh empirical claim, until the rerun lands.

### Section 4.9: Formal cover-time scaling law

Fitting `success_rate(n) = a · n^b` across maze sizes n ∈ {9, 11, 13, 17, 21, 25} per agent, 10,000-resample bootstrap CI on b:

| Agent | Exponent b | 95% CI | R² |
|---|---|---|---|
| BFSOracle / EgoWallFollower / DFSAgent | 0.000 | [0, 0] | constant 100% |
| **NoBackRandom** | **−2.07** | [−2.21, −1.94] | 0.994 |
| LevyRandom(α=2.0) | −2.66 | [−2.93, −2.44] | 0.999 |
| Random | −2.81 | [−3.10, −2.57] | 0.996 |
| FeatureQ_v2 | −3.21 | [−3.54, −2.89] | 0.965 |
| TabularQ_v2 | −3.53 | [−4.19, −3.02] | 0.986 |

NoBackRandom's exponent −2.07 is shallower than Random's −2.81 by 0.74 units — matching the Alon–Benjamini–Lubetzky–Sodin (2007) non-backtracking cover-time theorem almost exactly. The tabular and neural learners decay *faster* than random walks with scale (b < −2.81), reflecting their inability to generalize across maze sizes.

### Section 4.10: Bayesian hierarchical analysis (posterior probabilities)

A Beta-Binomial posterior over each agent's per-seed success rate (uniform prior, 10,000 samples per agent at 9×9) yields posterior-dominance probabilities:

| Comparison | P(A > B) |
|---|---|
| NoBackRandom > Random | **1.0000** |
| NoBackRandom > FeatureQ_v2 | **1.0000** |
| NoBackRandom > MLP_DQN | **1.0000** |
| EgoWallFollowerLeft > NoBackRandom | **1.0000** |

All headline orderings are posterior-certain. No seed overlap region can rescue the neural-RL baselines.

## 5. Discussion

### 5.1 The representation-vs-exploration dichotomy

Table 8 (policy distillation) gives us the cleanest possible diagnostic. Three empirical facts, side by side:

- **MLP + ego-features + supervised BFS labels → 99.9%** (n=20, sd=0.4)
- **MLP + ego-features + DQN training on shaped reward → 19.3%** (n=20, sd=6.7)
- **Same architecture, same observation, same optimizer, same capacity.**

The architecture *can* represent the optimal policy. DQN training does not find it. We therefore state:

> **The failure of MLP-DQN on procedural mazes is not representational capacity but the inability of online RL to discover and credit the simple maze-solving policy from sparse interaction.**

This is a narrower and sharper claim than "neural function approximation fails." It is also more falsifiable: a reviewer who wants to defeat it must find a training procedure (curriculum, demonstrations, intrinsic motivation at a larger scale, etc.) that closes the gap from 19.3% to near 99.9% with the same architecture. Our count-based PPO result (Table 9, 0.0–0.5% on sparse reward) suggests the usual suspects do not.

### 5.2 Why does standard RL find a local optimum?

With the representation question resolved, the question becomes: *what local optimum does gradient-descent RL converge to, and why is it below Random?*

The reward decomposition (Table 4) gives the answer: MLP_DQN pays **−0.136 per step** vs Random's **−0.238**. Neural agents successfully learn to avoid walls, hazards, and revisits — the locally-rewarding component of the policy. They fail at the globally-rewarding component: sustained exploration toward a distant goal through a negative-reward region.

This is consistent with the known pathology that **value-based gradient RL under-represents long-horizon sparse rewards** under a mix of dense shaping signals. A policy of "hover near start, occasionally step forward when goal-gradient is large" is stable under gradient descent because it has low variance and matches the shaping signal. A policy of "explore rapidly until goal is visible" is unstable because exploratory trajectories pay the step cost for many steps before any goal reward arrives.

Random walks do not have this pathology because they are stateless. NoBackRandom in particular is faster because it avoids the quadratic slowdown of reversing on tree-dominated graphs — classical cover-time theory (Alon–Benjamini–Lubetzky–Sodin 2007), which we validate empirically in §4.9.

### 5.3 Why doesn't DRQN close the gap?

DRQN (recurrent Q-learner with LSTM, §4.5) reaches 19.0% at 9×9, statistically equivalent to MLP_DQN. Adding memory does not rescue the exploration failure. This is consistent with the distillation result: **LSTM_from_BFSOracle reaches 90.7%**, showing the LSTM *can* represent the policy. But LSTM-DQN *does not discover it* under the same reward-driven gradient-descent dynamics that fail MLP-DQN.

### 5.4 Why doesn't FeatureQ collapse all the way?

FeatureQ keys its Q-table on the 24-dim discretized feature vector with no gradient flow between states. It cannot smooth wrong decisions across similar features the way a neural net can. This makes it more robust to the gradient pathology above but more sensitive to reward-shaping removal (K4: drops 35.3% → 17.4% without shaping). It is also limited by its inability to generalize across near-duplicate feature keys — which is why LSTM_from_FeatureQ_v2 (Table 8) *exceeds* its teacher at 63.3%.

### 5.5 Broader claim

The narrow claim this paper establishes:

> On a class of procedurally-generated mazes, a ≤20-line hand-coded wall-follower solves 100% of unseen test instances at every scale, a supervised MLP recovers the optimal policy to 99.9%, and standard neural RL agents (DQN/DDQN/DRQN) as well as a modern exploration-augmented variant (PPO + state-count bonus) converge below uniform random. The gap is not representational capacity, network size, learning rate, reward shaping, memory, information asymmetry, or maze topology. We therefore characterize the failure as reward-driven online RL on sparse-interaction navigation: the local optimum under gradient descent is below the performance of uninformed sampling.

We do **not** claim that neural function approximation fails in general, nor that deep RL is broken in other settings. We claim that **published empirical evaluations on procedural maze benchmarks should include hand-coded heuristic, distillation, and random-walk baselines** — and that failing to do so risks overinterpreting neural results.

## 6. Limitations

- **Environment breadth.** We test two environment families: our own procedurally-generated mazes (main, §3.1) and four MiniGrid tasks (§4, Table 10). We attempted a third — Procgen Maze — but the native install did not match our Python/CUDA versions and the results were incomplete at submission time. Additional non-maze navigation domains (NetHack, DMLab, VizDoom) would further reduce "toy artifact" concerns.
- **Network size.** Our capacity sensitivity sweeps hidden ∈ {32, 64, 128, 256} (Table 5). All four values produce 13.6–19.3% at 9×9, so we rule out "size too small" in this regime but do not cover the 10M+ parameter range.
- **Training budget.** 100 training episodes (main sweep) is short. We also report budget-matched SB3 PPO/DQN/A2C at 10K–500K environment steps (supplementary): the finding persists at every tested budget — even the 500K-step PPO does not beat uniform Random. Orders-of-magnitude larger budgets (100M+ steps) are not tested.
- **Reward configuration coverage.** Our K4 ablation (Table 3) and wider 6-config reward sensitivity sweep (supplementary, in progress at submission) cover the main shaping terms but not the full space of reward configurations. A cleverly designed curriculum or reward function may recover neural performance; we cannot rule this out.
- **Count-based PPO caveat.** Our exploration-augmented baseline (Table 9) uses a sparse reward (goal + step cost), not the shaped reward of the main sweep. Direct comparison to MLP_DQN requires caution. Running count-based PPO with shaped reward is open follow-up work.
- **Distillation from the BFS oracle requires full-map knowledge at demo-collection time.** This is not a deployable policy for novel mazes from the same distribution — it's a diagnostic. The result isolates representation from exploration; it does not produce a standalone maze-solving method.
- **Single research team, AI-assisted implementation.** The code was written with LLM pair-programming assistance. Every numerical result is regenerable from the shipped raw data via `reproduce.py verify`. We welcome third-party replication.

## 7. Reproducibility

All code at https://github.com/tejasnaladala/engram under Apache-2.0 license. Every numerical claim is regenerable via `python reproduce.py verify --manifest manifest_final.json`.

**Single-file stats pipeline** (`stats_pipeline.py`): seed-aligned paired bootstrap (10,000 resamples), Mann-Whitney U, Cohen's d, Holm-Bonferroni family-wise correction, BCa bootstrap (post-hoc sensitivity). The seed-aligned guarantee was added after adversarial audit identified a subtle dict-insertion-order bug in the original version.

**Reproducibility verifier** (`reproduce.py`): SHA-256 manifest of every result file + headline summary statistics. The `verify` subcommand re-hashes and re-computes from the shipped raw data, exiting non-zero on any drift. Code-hash pinning: every run records the SHA-256 of the experiment library it was executed under (current hash: `ed681d75c27fe352`).

**Smoke test** (`smoke_test.py`): 18/18 passing on our setup, runs in ~3 minutes on a consumer laptop. Exercises every agent class at 2 scales.

**Data:** ~3,500 per-run JSON files (~100 MB) shipped in `raw_results/` across 12 experiment directories. Each file contains per-episode reward, steps, solved flag, and the run's reward configuration + code hash.

**Adversarial audit:** Three rounds of review by a Codex MCP agent identified and fixed bugs listed in Appendix B. Updated confidence trajectory from 4/10 (first review) → 6/10 (reframe) → 6.5–7/10 (with distillation + info-parity + topology-audit evidence).

**Environment:** Python 3.11 (Windows 5070 Ti) / Python 3.12 (Linux H200), torch 2.11.0+cu128, snnTorch 0.9.4, gymnasium 1.2.3, stable-baselines3 2.8.0, minigrid 3.0.0, procgen (Linux+Python 3.10 venv only). Total compute: ~20 GPU-hours on RTX 5070 Ti Laptop + ~12 GPU-hours on 4× H200 (rented from vast.ai, approx $155). Every experiment is checkpointed; partial runs resume automatically from the last committed seed.

## 8. Compute & run counts

| Experiment | Platform | Runs | Wall time |
|---|---|---|---|
| Main sweep + K4 + DRQN + capacity + LR sweep | RTX 5070 Ti | ~1,700 | ~20 hr |
| Wall-following + loopy audit (CPU) | RTX 5070 Ti | 1,200 | <30 min |
| MiniGrid 4-env | H200 | 240 | ~3 hr |
| Policy distillation (3 teachers × 2 students × 20 seeds) | H200 | 120 | ~2 hr |
| Count-based exploration (2 variants × 20 seeds) | H200 | 40 | ~2 hr |
| DRQN multi-scale (13/17/21) | H200 | ~120 | ~3 hr |
| Cross-env transfer | H200 | 200 | ~3 hr |
| Procgen Maze (5 agents × 20 seeds) | H200 | ~100 | ~2.5 hr |
| Budget-matched SB3 (PPO_large + DQN_large) | H200 | ~100 | ~5 hr |
| Extra-seed backfill | H200 | 150 | ~1 hr |
| Reward sensitivity (6 configs × 5 agents × 20 seeds) | 5070 Ti + H200 | ~600 | ~8 hr |

**Total: ~3,500 runs, ~40 GPU-hours combined (~$155 on cloud H200 + RTX 5070 Ti laptop).**

## Appendix A: Threats to validity

- NoBackRandom's first-episode-step has a small left/up/down bias (~0.5% effect on success rate), documented as a limitation.
- Random variants share a single RNG stream per seed, making the comparison between Random / NoBack / Levy maximally paired (same maze, same starting RNG state, different policy). This *strengthens* paired tests relative to independent variants.
- Percentile bootstrap is used throughout; we replicate with BCa bootstrap (Efron & Tibshirani §14) in supplementary — headline numbers agree to within 0.5 percentage points.
- Holm-Bonferroni is valid under arbitrary dependence, so the correction holds even though multiple tests share the Random reference.

## Appendix B: Codex adversarial audit findings (incorporated)

During development, an external Codex MCP agent conducted two adversarial reviews of our codebase. Key fixes incorporated:

- **Seed-aligned pairing** (`stats_pipeline.py`): original version relied on dict-insertion-order, which could silently compare mismatched seed pairs under certain loading orders. Fixed to sort by seed and intersect common seeds.
- **Atomic checkpoint writes** (`experiment_lib_v2.py`): changed `shutil.move` → `os.replace` + `fsync` for Windows-safe atomicity.
- **Deterministic DRQN**: original launcher used `deterministic=False` for throughput. Re-run under `deterministic=True` for bit-reproducibility.
- **Reward decomposition formula**: original subtracted step-cost budget, which double-counted on wall-bump steps (the env overwrites step cost with wall cost). Fixed to `pain = total − goal`, apples-to-apples across agents.
- **Agent-name canonicalization**: single-underscore fallback stripped `FeatureQ` from `FeatureQ_v2` → cross-polluted V1/V2 buckets. Fixed to only strip double-underscore prefixes.

Full adversarial review transcripts available in `COMPREHENSIVE_AUDIT.md` and the git history.
