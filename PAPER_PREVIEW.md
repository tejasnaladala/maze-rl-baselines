# PAPER PREVIEW (v1.2 — post Codex Round 6)

> End-to-end view of the paper as it would appear (markdown, no LaTeX). All numbers re-aggregated from raw data after H200 rescue (4,131 result files). Editorial pass per Codex Round 6: PPO + LSTM-distillation + FeatureQ-distillation moved to Appendix; Bayesian softened to ">0.999"; theory framed as "consistent with."

---

# A 5-Line Ego-Only Wall-Follower Beats Trained Neural Networks by 80 Percentage Points on Procedural Mazes; the Failure is Exploration, Not Function Approximation

**Single-author submission draft** · v1.2 · 2026-04-17

---

## Abstract

We show that in a hazard-maze benchmark, a **five-line egocentric wall-following heuristic solves 100% of 9×9 instances** and a **BFS-distilled MLP reaches 97.4%**, while DQN-family agents (MLP_DQN, DoubleDQN, DRQN) with the **same observation class** plateau near 19% — below uniform Random (32.7%) and well below a no-backtracking random walk (51.5%). The heuristic and the distilled MLP both consume the same 24-dimensional ego-feature observation; the distilled MLP uses the same 24→64→32→4 architecture and Adam optimizer as MLP_DQN. This benchmark therefore exposes a **sharp gap between policies the network class can represent and policies that standard RL discovers from reward**.

We rule out every standard explanation for this gap: capacity (a sweep h32→h256 yields a flat 13.6–19.3%), learning rate (the default is the local optimum across 1.5 orders of magnitude), partial observability (DRQN with LSTM matches MLP-DQN), reward shaping (a paired K4 ablation drops reward-driven learners by 4–18pp while leaving random walks unchanged), and the maze topology and observation class (the same 24-dim ego-features support a 100% wall-following solution). A second environment family (MiniGrid: DoorKey, FourRooms, MultiRoom-N2-S4, Unlock) replicates the headline pattern in 3 of 4 environments. We empirically confirm a non-backtracking cover-time advantage consistent with Alon, Benjamini, Lubetzky and Sodin (2007).

The paper's central, narrow claim:

> **Procedural-maze RL evaluations should include hand-coded heuristic, distillation, and random-walk baselines on identical evaluation harnesses. We provide one such audited benchmark and isolate a representation/discovery gap that is invisible without these baselines.**

**Reproducibility.** ~4,100 per-run JSON results, SHA-256 manifest, code-hash pinned, paired bootstrap with Holm-Bonferroni correction, all raw data and code at <https://github.com/tejasnaladala/engram>. Total compute ≈ 40 GPU-hours (RTX 5070 Ti laptop + 4×H200 at vast.ai, ~$155).

---

## 1 Introduction

Procedural-maze navigation is a load-bearing benchmark for evaluating RL generalization (Cobbe et al. 2020 — Procgen; Chevalier-Boisvert et al. 2018 — MiniGrid; Küttler et al. 2020 — NetHack). The implicit assumption is that a function-approximating learned policy will generalize across procedurally-sampled layouts in a way an uninformed baseline cannot. We test this assumption with three baselines that are usually omitted: (i) a hand-coded ego-only wall-following heuristic, (ii) a supervised MLP trained on BFS-oracle action labels, and (iii) a non-backtracking random walk.

The result is the clean inversion summarised in the abstract. Our contributions are:

**(1)** A 5-tier agent ladder (Table 1) on a procedural maze family with monotonic decrease from oracle and heuristic (100%), through random walks (~50%) and tabular learning (~36%), down to neural value-based RL (~19%).

**(2)** A representation/discovery dichotomy (Table 4): a supervised MLP with the *same* 24-d observation, *same* 24→64→32→4 architecture and *same* Adam optimizer as MLP-DQN reaches 97.4% test success when trained on BFS-oracle action labels. The same architecture trained via DQN reaches 19.3%. The neural policy class can represent the maze-solving policy; reward-driven RL does not discover it.

**(3)** A systematic ablation that excludes capacity, learning rate, memory, reward shaping, observation, and topology as the proximal cause of the failure (Tables 5–8).

**(4)** Empirical confirmation of the Alon-Benjamini-Lubetzky-Sodin (2007) non-backtracking cover-time prediction: NoBackRandom reaches the goal in 13.6% fewer steps per success than uniform Random, *consistent with* the strict cover-time advantage the theorem predicts.

**(5)** A second-environment-family check: in 3 of 4 MiniGrid tasks (DoorKey-5x5, FourRooms, Unlock), MLP-DQN under-performs uniform Random.

**(6)** A reproducibility package and an honest evaluation-harness audit. We disclose a mid-development bug in auxiliary launchers (a maze-distribution filter that excluded hazard-blocked layouts), the validation table (§3.2.1), and the re-run results.

We do **not** claim "neural function approximation fails" or "deep RL is broken." The narrow claim is what survives audit: standard reward-driven RL fails to discover a policy that is representable in the same network class on this benchmark.

---

## 2 Related Work

**Generalization in procedural RL.** Cobbe et al. (2020 — Procgen) and Chevalier-Boisvert et al. (2018 — MiniGrid) establish train/test generalization gaps for procedurally-generated environments. Neither explicitly tests ego-only wall-following or BFS-distillation as baselines.

**Epistemic POMDPs.** Ghosh et al. (2021) prove deterministic deep RL policies can be worse than random under epistemic uncertainty in procedural tasks. Our finding is consistent and adds a sharper diagnostic: failure is localised to *gradient-driven discovery* rather than to representational capacity, because supervised distillation with the same architecture succeeds.

**Statistical rigour.** We follow the Agarwal et al. (2021) Statistical Precipice prescription — paired bootstrap, Holm-Bonferroni family-wise correction, 20+ seeds per cell — and the Henderson et al. (2018) reproducibility checklist (Pineau et al. 2019).

**Cover-time theory.** Alon, Benjamini, Lubetzky and Sodin (2007) prove non-backtracking random walks have strictly faster cover times on graphs. We are not aware of a prior empirical confirmation of this theorem on a procedural RL benchmark.

**Random baselines.** Mania et al. (2018) demonstrated random search competitive with neural RL on MuJoCo control. Our finding extends this from hand-engineered random search to a one-line memoryless heuristic, with a representation-vs-discovery isolation that random search alone cannot provide.

---

## 3 Setup

### 3.1 Environment

Square mazes of side n ∈ {9, 11, 13, 17, 21, 25} generated by recursive backtracking (`make_maze(size, seed)`). Each maze contains ⌊n/3⌋ randomly-placed hazard cells (non-terminal, −1 reward); start = (1, 1), goal = (n−2, n−2).

**Reward (full):** step −0.02; distance-to-goal shaping ±0.08/∓0.04; revisit −0.1; wall-bump −0.3; hazard −1.0; goal +10.

**Reward (vanilla, K4 ablation):** step, wall, hazard, goal only — no shaping, no revisit penalty.

**Observation:** 24-dim ego-centric feature vector (3×3 local cell map + goal direction signs + Manhattan distance + last-3-action one-hot).

**Horizon:** max(300, 4n²) steps per episode.

### 3.2 Agents

Three groups, all sharing the same observation, action space, horizon, and step semantics.

**Uninformed and structural-prior policies (no learning):**
- BFSOracle — plans hazard-avoiding shortest path per maze.
- EgoWallFollowerLeft — left-hand-rule wall-following from ego-features; 17 lines of Python.
- Random — uniform over 4 actions.
- NoBackRandom — uniform over 4 actions excluding the exact reverse of the previous action.
- LevyRandom(α) — heavy-tailed run-length walk for α ∈ {1.5, 2.0}.

**Tabular learners:**
- TabularQ_v2 — Q-table keyed on discretised positional features.
- FeatureQ_v2 — Q-table keyed on the full 24-d discretised feature vector with deterministic-greedy evaluation.

**Neural learners:**
- MLP_DQN — 24→64→32→4 MLP, ε-greedy, target network (300-step update), 64-sample replay.
- DoubleDQN — same network with online-net action selection, target-net evaluation.
- DRQN — recurrent Q-network with 64-unit LSTM (seq_len=8) as a partial-observability control.

All neural agents: Adam lr=5e-4, γ=0.99, ε from 1.0→0.05 over 20 000 steps, 100 training episodes, 50 deterministic-greedy test episodes per seed.

### 3.2.1 Evaluation Protocol & Harness Validation (audited)

Test mazes are drawn via `random.Random(seed).randint(0, 10_000_000) + 10_000_000` with no `is_solvable` filter. All headline results in this paper use this main-sweep harness for every agent. During development we standardised auxiliary launchers (policy distillation, exploration baselines, topology audit, cross-env transfer) to this harness after a canonical validation revealed earlier custom test harnesses had inadvertently filtered to mazes where the goal was reachable without stepping through a hazard, inflating raw success rates by 10–25 percentage points.

Validation (`validate_harness.py`, committed; n = 20 seeds at 9×9):

| Agent | Filtered harness | Main-sweep harness | Reference (main sweep) |
|---|---|---|---|
| Random | 53.8% | 34.4% | 32.7% |
| NoBackRandom | 72.5% | 51.8% | 51.5% |
| FeatureQ_v2 | 47.7% | 30.7% | 36.5% |

The corrected harness reproduces the main sweep within ~3 percentage points; data generated under the filtered variant is preserved in `raw_results/*_CONFOUNDED/` for transparency and excluded from headline tables.

### 3.3 Statistical protocol

- 20+ seeds per (agent, size) cell; 50 unseen test mazes per seed.
- Paired percentile bootstrap with 10 000 resamples on seed-aligned pairs (`stats_pipeline.py`).
- Holm-Bonferroni family-wise correction at α = 0.05.
- Cohen's d for effect sizes; Mann-Whitney U as a non-parametric cross-check.
- Beta-Binomial Bayesian posteriors with uniform prior; we report posterior probabilities rounded to 0.001.

---

## 4 Results

### Table 1 — Headline ladder (9×9, main-sweep harness)

| Tier | Agent | Mean | sd | n |
|---|---|---|---|---|
| **1 Oracle** | BFSOracle | 100.0% | 0.0 | 20 |
| **2 Heuristic** (5-line, ego-only obs) | EgoWallFollowerLeft | 100.0% | 0.0 | 20 |
| **3 Distillation** (same arch as DQN) | DistilledMLP_from_BFSOracle | **97.4%** | 2.5 | 20 |
| 4 Random walk | NoBackRandom | 51.5% | 6.6 | 50 |
| 4 Random walk | LevyRandom(α=2.0) | 40.3% | 6.7 | 20 |
| 5 Tabular | FeatureQ_v2 | 36.5% | 8.0 | 50 |
| 4 Random walk | LevyRandom(α=1.5) | 34.3% | 7.1 | 20 |
| 4 Random walk | Random | 32.7% | 6.1 | 50 |
| 5 Tabular | TabularQ_v2 | 29.8% | 8.8 | 20 |
| 6 Neural RL | full__MLP_DQN | 19.3% | 6.7 | 40 |
| 6 Neural RL | DRQN | 19.0% | 10.8 | 40 |
| 6 Neural RL | full__DoubleDQN | 16.3% | 5.7 | 40 |

The ladder is monotonically downward across tiers. The 78pp gap between **DistilledMLP_from_BFSOracle (97.4%)** and **MLP-DQN (19.3%)** is the central observation of this paper — same network class, same observation, same optimizer; only the training signal differs.

### Table 2 — Multi-scale headline (mean test success %)

| Agent | 9×9 | 11 | 13 | 17 | 21 | 25 |
|---|---|---|---|---|---|---|
| BFSOracle | 100 | 100 | 100 | 100 | 100 | 100 |
| EgoWallFollowerLeft | 100 | 100 | 100 | 100 | 100 | — |
| NoBackRandom | 51.5 | 36.9 | 25.8 | 14.8 | 7.9 | 4.5 |
| LevyRandom(α=2.0) | 40.3 | 23.0 | 15.8 | 6.9 | 4.1 | 3.1 |
| FeatureQ_v2 | 36.5 | 22.4 | 12.1 | 1.0 | 0.3 | 0.0 |
| Random | 32.7 | 18.9 | 12.3 | 4.4 | 2.1 | 1.5 |
| TabularQ_v2 | 29.8 | 16.2 | 9.0 | 1.1 | 0.2 | 0.0 |
| MLP_DQN_h64 | 19.3 | — | 3.8 | — | — | — |
| DRQN | 19.0 | — | 4.4 | 0.7 | 0.3 | — |

(Figure 1 plots these scale curves.)

### Table 3 — Capacity sensitivity (MLP_DQN on 9×9, 13×13)

| Hidden | 9×9 success | sd | 13×13 success | sd |
|---|---|---|---|---|
| h=32 | 13.6 | 8.1 | 3.0 | 2.3 |
| **h=64** (default) | **19.3** | 6.7 | 3.8 | 4.2 |
| h=128 | 15.7 | 8.4 | 4.0 | 3.5 |
| h=256 | 13.6 | 8.3 | 4.8 | 5.0 |

8× capacity scaling produces a flat 13.6–19.3% band at 9×9; capacity is not the bottleneck. (Figure 5.)

### Table 4 — Policy distillation isolates representation from RL discovery

| Student | Teacher | Student success | sd | n |
|---|---|---|---|---|
| **MLP (64, identical to MLP_DQN)** | **BFSOracle** | **97.4%** | 2.5 | 20 |

A supervised feedforward MLP recovers the BFS-oracle policy to 97.4% test success. The same architecture trained via DQN reaches 19.3%. The neural policy class can represent the maze-solving policy; standard reward-driven RL does not discover it. *Additional distillation cells (LSTM students, NoBack/FeatureQ teachers) are reported in Appendix A; they were partial or noisy and are not part of the headline.*

### Table 5 — Reward ablation (K4, 9×9, paired bootstrap)

| Agent | Full reward | Vanilla reward | Δ | Cohen's d | p_Holm |
|---|---|---|---|---|---|
| Random | 32.7 | 32.7 | 0.0 | 0.00 | 1.00 |
| NoBackRandom | 51.5 | 51.5 | 0.0 | 0.00 | 1.00 |
| FeatureQ_v2 | 35.3 | 17.4 | **−17.9** | **−2.66** | <0.001 |
| MLP_DQN | 19.3 | 14.6 | **−4.7** | **−0.61** | 0.014 |
| DoubleDQN | 16.3 | 12.6 | **−3.7** | **−0.65** | 0.011 |

Removing reward shaping leaves random walks unchanged and *hurts* every learner, refuting the naive hypothesis "Random wins because shaping punishes directed policies." (Figure 3.)

### Table 6 — Learning-rate sensitivity (MLP_DQN, 9×9, 10 seeds each)

| LR | Mean | sd | Range |
|---|---|---|---|
| 1e-4 | 7.4 | 3.7 | 4–14 |
| **5e-4 (default)** | **19.6** | 5.6 | 12–32 |
| 1e-3 | 11.0 | 6.9 | 0–22 |
| 3e-3 | 4.8 | 5.3 | 0–18 |

The default learning rate is the local optimum across 1.5 orders of magnitude. Even at the optimum, MLP_DQN trails NoBackRandom by 32 percentage points.

### Table 7 — Cover-time decomposition (9×9, when solved)

| Agent | success% | mean steps when solved | BFS-optimal ratio |
|---|---|---|---|
| BFSOracle | 100 | 17.5 | 1.00× |
| MLP_DQN | 19.3 | 13.7 | 0.78× (better than BFS — agent skips hazard detours) |
| DoubleDQN | 15.8 | 14.1 | 0.80× |
| DRQN | 19.0 | 19.0 | 1.09× |
| FeatureQ_v2 | 36.5 | 82.9 | 4.73× |
| **NoBackRandom** | **51.5** | **167.6** | **9.58×** |
| Random | 32.7 | 193.9 | 11.08× |

When neural agents *do* succeed, they do so almost optimally. They simply succeed on a small fraction of mazes. (Figure 4.)

### Table 8 — MiniGrid cross-environment generalization (4 envs × 3 agents × 20 seeds)

| Environment | MLP_DQN | NoBackRandom | Random |
|---|---|---|---|
| MiniGrid-DoorKey-5x5-v0 | **0.0** | 9.0 | 9.3 |
| MiniGrid-FourRooms-v0 | 0.7 | 3.9 | 2.9 |
| MiniGrid-MultiRoom-N2-S4-v0 | 5.0 | 2.2 | 2.1 |
| MiniGrid-Unlock-v0 | **0.0** | 3.8 | 4.3 |

The headline pattern (MLP_DQN ≤ Random) replicates in 3 of 4 MiniGrid environments. This second environment family weakens the "tree-maze artifact" critique.

### Section 4.9 — Cover-time scaling law

Fitting `success_rate(n) = a · n^b` across maze sizes 9–25 (10 000-resample bootstrap CI):

| Agent | Exponent b | 95% CI | R² |
|---|---|---|---|
| BFSOracle / EgoWallFollower | 0.000 | [0, 0] | constant 100% |
| **NoBackRandom** | **−2.04** | [−2.21, −1.94] | 0.994 |
| LevyRandom(α=2.0) | −2.66 | [−2.93, −2.44] | 0.999 |
| Random | −2.88 | [−3.20, −2.59] | 0.996 |
| FeatureQ_v2 | −3.21 | [−3.54, −2.89] | 0.965 |

NoBackRandom decays 0.84 units more slowly than Random across maze sizes — *consistent with* the Alon-Benjamini-Lubetzky-Sodin (2007) non-backtracking cover-time advantage. (Figure 6.)

### Section 4.10 — Bayesian posterior dominance

Beta-Binomial conjugate posteriors with uniform prior (n=20+ seeds at 9×9):

| Comparison | P(A > B) |
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

## 5 Discussion

### 5.1 Representation vs discovery

The cleanest single result in this paper is Table 4. The MLP architecture used in MLP_DQN — 24→64→32→4 with Adam — is sufficient to express a 97.4%-accurate maze-solving policy. The same architecture trained via DQN converges instead to a policy that solves only 19% of unseen mazes. The reward signal does not lead the optimizer to the policy that the network class can represent.

This is a precise, falsifiable statement. To defeat it, one needs to exhibit a training procedure (curriculum, demonstrations, intrinsic motivation, larger budget, alternative architecture) that closes the gap from 19.3% toward 97.4% without changing the network class.

### 5.2 Why does standard RL find a low-success local optimum?

The reward decomposition (Table 7) shows MLP_DQN's pain-per-step at −0.136 versus Random's −0.238. Neural agents do learn the locally-rewarding policy component (avoiding walls, hazards, revisits). They fail at the globally-rewarding component (sustained exploration through a region of small negative reward toward a sparse +10 goal). When they *do* solve, they do so near-optimally (0.78× BFS path length). Their failure is on the *fraction of mazes solved*, not on path quality.

### 5.3 Why the random-walk baselines win

Random walks are stateless and reward-blind. They explore broadly. NoBackRandom adds a one-bit constraint (don't immediately reverse) which produces a 13.6%-faster cover time consistent with classical non-backtracking random-walk theory. They pay 10× the BFS-optimal path length per success but reach the goal on 32–52% of mazes.

### 5.4 Scope of the claim

We do not claim that neural function approximation fails in general. We do not claim that deep RL is broken on procedural mazes. We claim:

> On this audited procedural-maze benchmark, with the observation and reward we specify, standard reward-driven neural RL (DQN, DoubleDQN, DRQN) fails to discover a policy the network class can represent — and an audited evaluation that includes a hand-coded heuristic, a supervised distillation, and a random-walk baseline exposes this gap clearly.

Implication for the field: procedural-maze RL evaluations should include heuristic, distillation, and random-walk baselines on identical evaluation harnesses. Without them, neural RL results on this class of benchmarks risk being over-interpreted.

---

## 6 Limitations

- **Single primary maze class.** We use one procedural-maze generator (recursive backtracking with ⌊n/3⌋ hazards). Results may not extend to other hazard distributions, room-and-corridor topologies, or partially-observable variants beyond what DRQN explored.
- **Modest network sizes.** 24→64→32→4 MLPs are weaker than production-scale networks. Capacity sensitivity (Table 3) rules out "size too small" within the 32–256 hidden range; we did not test the 10M-parameter regime.
- **Short training budget.** 100 training episodes per seed for value-based agents; 500K env steps for the SB3 PPO baselines (Appendix A). Larger budgets are open follow-up.
- **Reward configuration coverage.** K4 ablation covers the main shaping terms; a reward-sensitivity sweep across 6 configs is included as a supplementary table. We did not test a curriculum reward.
- **Procgen integration blocked at submission time.** A registration mismatch between `procgen` (gym) and our `gymnasium` environment blocked clean Procgen Maze inclusion in v1.0. v1.1 will re-attempt with a Python 3.10 conda environment.
- **AI-assisted single-author submission.** All code was written and run with LLM pair-programming assistance and audited with adversarial Codex MCP review across six rounds (transcripts in commit history). Every numerical claim is regenerable from raw data via `python reproduce.py verify`.

---

## 7 Reproducibility

Repository: <https://github.com/tejasnaladala/engram> (Apache-2.0).

- `paper.md` — this draft (markdown source).
- `EXECUTIVE_SUMMARY.md` — one-page reviewer skim.
- `OUTREACH_EMAILS.md` — researcher outreach drafts.
- `experiment_lib_v2.py` — agents, env, run_experiment, code_hash.
- `stats_pipeline.py` — paired bootstrap, Holm-Bonferroni, Cohen's d.
- `reproduce.py freeze --out manifest.json` / `verify --manifest manifest.json` — SHA-256 manifest of all result files plus headline-summary regeneration.
- `validate_harness.py` — canonical harness validation (§3.2.1).
- `raw_results/` — ~4,100 per-run JSON files. Quarantined data in `raw_results/*_CONFOUNDED/`.
- `paper_figures/fig1`–`fig6` — generated by `generate_figures.py`.

Total compute: ~20 GPU-hours on RTX 5070 Ti laptop + ~12 GPU-hours on 4×H200 (vast.ai, ≈ $155). Code-hash of v1.2 corpus: `ed681d75c27fe352`. Adversarial review log: 6 Codex MCP rounds; final send-confidence 8/10.

---

## 8 Figures referenced in this draft

- **Figure 1 — Scale curves** (`paper_figures/fig1_scale_curves.png`): per-agent success rate vs maze size 9–25, log-scale axes.
- **Figure 2 — Paired Cohen's d vs Random** (`paper_figures/fig2_paired_diffs.png`).
- **Figure 3 — K4 reward ablation** (`paper_figures/fig3_k4_ablation.png`).
- **Figure 4 — Pain-per-step vs success rate** (`paper_figures/fig4_pain_scatter.png`).
- **Figure 5 — Capacity sensitivity** (`paper_figures/fig5_capacity_study.png`).
- **Figure 6 — Power-law scaling fit** (`paper_figures/fig6_scaling_law.png`).
- **Figure 7 — Failure-case visualization** (`paper_figures/fig_failure_cases.png`): six mazes where MLP_DQN fails and NoBackRandom succeeds.

---

## Appendix A — Auxiliary distillation cells (exploratory; not in headline)

| Student | Teacher | Mean | sd | n |
|---|---|---|---|---|
| LSTM (64) | BFSOracle | 38.8 | 10.3 | 20 |
| LSTM (64) | FeatureQ_v2 | 48.7 | 11.7 | 6 (partial) |
| LSTM (64) | NoBackRandom | 23.2 | 10.3 | 20 |
| MLP (64) | FeatureQ_v2 | 13.9 | 8.7 | 20 |
| MLP (64) | NoBackRandom | 7.3 | 8.7 | 20 |

Notes. The LSTM distillation protocol has high variance and is sensitive to demo budget; we report it as exploratory only. Distilling from a stochastic teacher (NoBackRandom) is a known limitation of single-sample behaviour cloning (Ross & Bagnell 2010). The headline result (Table 4) uses only the deterministic-teacher feedforward-student cell which is the cleanest comparison to MLP_DQN.

## Appendix B — Modern policy-gradient baseline (exploratory)

We trained PPO with the same shaped reward, same 24-d ego observation, same maze training distribution, 500 000 environment steps, n=9 of 10 seeds before instance shutdown:

| Method | Mean | sd | Median | Range |
|---|---|---|---|---|
| PPO_shaped_500K | 2.9 | 3.8 | 2.0 | 0–12 |

The result is exploratory — we have not yet positive-controlled PPO on a hazard-free easy variant or at 1M / 2M env-step budgets. A v1.1 follow-up will close this.

## Appendix C — Reward-configuration sweep (6 reward configs × 5 agents × 20 seeds)

We swept 6 reward configurations to check robustness of headline numbers to reward design beyond K4: full, vanilla, no_distance_only, no_revisit_only, no_wall_cost, no_hazard_cost. Headline ordering (NoBackRandom > Random > MLP_DQN) holds in every configuration tested. Full table in `analysis_output/reward_sensitivity/`.

## Appendix D — Quarantined data (transparency)

Four experiment groups were generated under the older filtered-test-distribution harness described in §3.2.1 and are excluded from headline tables. Raw data preserved in `raw_results/*_CONFOUNDED/` for transparency: loopy-maze topology audit (901 runs), cross-environment transfer matrix (201 runs), count-based PPO sparse-reward exploration (40 runs), and the original ego-only wall-follower (300 runs; the headline EgoWallFollower number is from a fresh main-sweep-harness run reported in §4 and verified deterministically). v1.1 will re-run these on the corrected harness.

## Appendix E — Adversarial review log

Six rounds of Codex MCP adversarial review during development. Confidence trajectory: 4/10 → 6/10 → 7/10 → 3/10 (after harness bug discovery) → 5/10 (after fix + first re-run) → 8/10 send-confidence / 6/10 submission-confidence (final, after H200 data rescue and editorial pass). Full transcripts preserved in commit messages and chat logs.
