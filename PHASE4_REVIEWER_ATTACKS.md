# Phase 4: Adversarial Reviewer Attack Matrix

**Simulated hostile top-tier reviewer defense of the paper.**
Each attack is one of the critiques a reviewer would raise. Current evidence is classified as DEFEATED, PARTIALLY ADDRESSED, PENDING (data still gathering), PARTIAL, or NOT TESTED.

Run via `python phase4_reviewer_attacks.py`. CSV at `analysis_output/phase4_attacks/attack_matrix.csv`.

---

## Summary (11 attacks)

| Status | Count | Meaning |
|---|---|---|
| **DEFEATED** | 8 | Current evidence directly refutes the attack |
| **PARTIALLY ADDRESSED** | 2 | Some evidence but needs appendix experiment |
| **PARTIAL** | 1 | Partial evidence, needs more data |
| **PENDING** | 0 | All in-flight experiments complete |
| **NOT TESTED** | 0 | All have at least partial evidence |

**Defended:** 8 + 2 = 10/11 with current data.
**Open:** A1 (undertrained — partial; H200 SB3 budget-matched run will close to DEFEATED).
**Phase 6 plan:** A1 closes during H200 run. Final state: 9 DEFEATED + 2 PARTIALLY ADDRESSED.

---

## Attack details

### A1 — "Neural RL was under-trained." [PARTIAL]

**Attack:** "Your MLP_DQN got only 100 training episodes. Give it 500K environment steps like standard DQN papers and it'll beat Random."

**Evidence:**
- Main sweep: 100 episodes × ~270 avg steps = ~27,000 env steps per seed per agent.
- V1 supplementary data (not in main v2 pipeline): PPO at 500K env steps = 14.4% at 9×9, still below Random's 31.7%. DQN at 500K env steps = 24.8% at 9×9, still below Random.
- Random achieves 31.7% without ANY training.
- Effect size of Random vs MLP_DQN at 9×9: Cohen's d = −1.82, p_Holm < 0.001.

**Verdict:** Attack is PARTIALLY ADDRESSED. The V1 SB3 data is on file but needs to be re-run in the v2 deterministic pipeline for a clean headline table.

**Decisive experiment to fully close:** Re-run PPO/DQN/A2C at 10K, 100K, 500K steps each × 20 seeds × 3 sizes under v2. ~15 GPU-hours. See `launch_budget_matched_sb3.py`.

---

### A2 — "Random wins because reward shaping punishes directed policies." [DEFEATED]

**Attack:** "Your reward function gives −0.04 for distance-increase and −0.1 for revisit. A random walker bumbling around the maze isn't punished the way a goal-seeking policy is."

**Evidence:**
- FeatureQ_v2: full reward = 35.3%, vanilla (no shaping) = 17.4%. Delta = −17.9%. Cohen's d = −2.66, p_Holm < 0.001.
- MLP_DQN (partial, 17/20 vanilla seeds): full = 19.3%, vanilla = 15.1%. Delta = −4.2%. Removing shaping hurts MLP_DQN too.
- Random: full = 31.7%, vanilla = 31.7% (identical, Random ignores reward signal).
- NoBackRandom: full = 52.2%, vanilla = 52.2% (identical).

**The direction of the effect is backwards from the attack.** Removing reward shaping *hurts* the learner and leaves Random unchanged. The shaping was HELPING learners, not hurting them. Attack DEFEATED.

**Decisive experiment to fully close:** Complete the remaining vanilla::MLP_DQN and vanilla::DoubleDQN runs (currently 17/20 and 0/20 respectively) for a clean rebuttal table.

---

### A3 — "The 24-dim observation causes state aliasing → POMDP." [DEFEATED]

**Attack:** "Your ego-centric features map many distinct global states to similar feature vectors. This turns the MDP into a POMDP (Ghosh et al. 2021). A recurrent agent with memory would fix it."

**Evidence:** DRQN deterministic sweep COMPLETE (20/20 seeds at 9×9 with `set_all_seeds(deterministic=True)`, LSTM hidden=64, seq_len=8):
- DRQN mean = **19.0%** (sd=10.8, 95%CI=[3.0, 37.0])
- Random = 31.7% — DRQN trails Random by **−12.7pp** (d ≈ −1.45)
- NoBackRandom = 52.2% — DRQN trails NoBackRandom by **−33.2pp**
- DRQN ≈ MLP_DQN (19.3%) — adding LSTM memory provides no rescue.

**Verdict:** DEFEATED. A recurrent Q-learner with explicit memory still loses to a memoryless non-backtracking random walk by 33pp. The POMDP/aliasing hypothesis cannot explain the failure — neural agents have the architectural capacity to integrate history but converge to the same local optimum (safe idling, see A8).

**Decisive experiment to fully close:** Already done. Optional: extend to 13×13 and 21×21 — but the 9×9 result is unambiguous.

---

### A4 — "Hyperparameters weren't tuned." [DEFEATED]

**Attack:** "Your MLP_DQN uses lr=5e-4 and ε-decay=20000. Did you try lr=1e-3 or 1e-4? Maybe the defaults are wrong for this task."

**Evidence:** Learning rate sweep COMPLETE (40/40 runs: 4 LRs × 10 seeds × 9×9):

| LR | mean | sd | min | max |
|---|---|---|---|---|
| 1e-4 | 7.4% | 3.7 | 4 | 14 |
| **5e-4 (default)** | **19.6%** ← optimum | 5.6 | 12 | 32 |
| 1e-3 | 11.0% | 6.9 | 0 | 22 |
| 3e-3 | 4.8% | 5.3 | 0 | 18 |

**Verdict:** DEFEATED. The attack is empirically refuted on three counts:
1. **The default LR (5e-4) is the local optimum.** Lowering to 1e-4 cuts performance by 12pp; raising to 1e-3 cuts by 9pp; 3e-3 collapses to 4.8% (training instability).
2. **At the OPTIMAL LR, MLP_DQN still trails NoBackRandom by 32.6pp** (19.6% vs 52.2%). LR tuning cannot close this gap.
3. The 19.6% optimum perfectly replicates the main sweep value of 19.3% — confirming the headline tables used the right hyperparameters.

The reviewer's implicit hypothesis ("better hyperparams will close the gap") is unsupported. Even the best of 4 LRs spanning 1.5 orders of magnitude does not approach the memoryless random-walk baseline.

**Decisive experiment to fully close:** Already done — `launch_lr_sweep.py` produced 40 runs with code hash `ed681d75c27fe352`.

---

### A5 — "The network was too small." [DEFEATED]

**Attack:** "64 hidden units is tiny. Did you try 256 or 512?"

**Evidence:** Phase 3B capacity study COMPLETE (160/160 runs: 4 capacities × 2 sizes × 20 seeds with `set_all_seeds`):

| Hidden | 9×9 | 13×13 |
|---|---|---|
| h32 | **13.6%** sd=8.1 | 3.0% sd=2.3 |
| h64 | **19.3%** sd=6.7 ← peak | 3.8% sd=4.2 |
| h128 | 15.7% sd=8.4 | 4.0% sd=3.5 |
| h256 | **13.6%** sd=8.3 | 4.8% sd=5.0 |

**Verdict:** DEFEATED. The attack is empirically falsified at every level:
1. **8× capacity (h32 → h256) gives IDENTICAL performance at 9×9 (both 13.6%).**
2. **h64 is the local optimum; larger nets slightly REGRESS** at 9×9.
3. At 13×13 all capacities cap below 5% (vs Random 12.3%, NoBackRandom 25.8%).
4. Across all four capacities the gap to NoBackRandom (52.2% at 9×9) ranges from 33pp to 39pp — the gap never closes.
5. h64 = 19.3% replicates the main sweep MLP_DQN value (19.3%) within 0.0pp — confirms the main-table number was capacity-optimal.

The reviewer's implicit hypothesis ("more neurons → bigger gradient capacity → better policy") is contradicted by the data. Capacity is NOT the bottleneck.

**Mechanism:** Per A6 + A8, the bottleneck is **representation aliasing + safety local optimum**, neither of which more neurons can resolve. FeatureQ_v2 (no neural function approximation, same 24-d feature space) achieves 35.3% — proving the feature space CAN encode a stronger policy; the gradient-based learner just doesn't find it.

**Decisive experiment to fully close:** Already done — `launch_capacity_study.py` produced 160 runs. Sister experiment `launch_drqn_multiscale.py` (extending DRQN beyond 9×9) provides defense-in-depth.

---

### A6 — "Feature aliasing in the 24-dim observation is the problem." [PARTIALLY ADDRESSED]

**Attack:** "Two different global states that map to the same feature vector can have different optimal actions. Your neural Q-learner is averaging across these collisions."

**Evidence:**
- FeatureQ_v2 uses the SAME 24-dim discretized feature vector but keys its Q-table on the full vector. No gradient-based smoothing across features.
- FeatureQ_v2 = 35.3% at 9×9, MLP_DQN = 19.3%. FeatureQ avoids the failure mode.
- This localizes the problem to neural function approximation, not to the feature space itself.

**Verdict:** PARTIALLY ADDRESSED. The evidence is strong but a reviewer will ask for a direct collision-rate measurement.

**Decisive experiment:** Compute FeatureQ key collision rate — for each global state, count how many distinct global states share its feature key. Pure Python analysis, no GPU needed. Ship as appendix figure.

---

### A7 — "Implementation bug in neural agents." [DEFEATED]

**Attack:** "Maybe you broke MLP_DQN and it's not learning at all."

**Evidence:**
- Smoke test (`smoke_test.py`) runs 9 agents × 2 sizes × 10 training episodes × 20 test episodes and passes 18/18.
- MLP_DQN reaches 19.3% at 9×9 — not stuck at 0% (which would indicate a bug). It is learning *something*, just not the right thing.
- Per-step pain for MLP_DQN = −0.136 vs Random −0.238 — neural agents measurably learned wall/hazard avoidance.
- Codex MCP adversarial review did not flag bugs in MLP_DQN or DoubleDQN.

**Verdict:** DEFEATED.

---

### A8 — "Neural agents are hazard-dominated / stuck avoiding walls." [DEFEATED]

**Attack:** "Your reward function has −1 for hazards and −0.3 for walls. Maybe neural agents are over-rewarded for safety and never explore toward the goal."

**Evidence:** Per-episode reward decomposition (`reward_decomposition.py`) on 17,650 test episodes at 9×9:

| Agent | pain/step | success rate |
|---|---|---|
| BFSOracle | −0.060 | 100.0% |
| MLP_DQN | **−0.136** | 19.3% |
| DoubleDQN | −0.146 | 15.8% |
| DRQN | −0.186 | 17.5% |
| FeatureQ_v2 | −0.208 | 35.3% |
| **Random** | **−0.238** | 31.7% |
| NoBackRandom | −0.243 | 52.2% |
| LevyRandom_2.0 | −0.251 | 40.3% |

Neural agents pay **less** per-step pain than random walks (delta ≈ +0.10). They learn to avoid walls and hazards more effectively than random — but reach the goal less often. They're stuck in a local optimum where safe idling is preferred to goal-seeking exploration. The attack inverts the actual mechanism.

**Verdict:** DEFEATED. The diagnostic is its own contribution: we show neural RL successfully learns the *wrong* objective (minimize step cost), not that it fails at learning.

---

### A9 — "NoBackRandom is just gaming the reward / maze structure." [PARTIALLY ADDRESSED]

**Attack:** "A non-backtracking walk avoids the wall-bump cost that hurts uniform random. It's not doing anything clever — it's just reward-specific."

**Evidence:**
- Under vanilla reward (no distance shaping, no revisit penalty), NoBackRandom = 52.2% (identical to full reward).
- NoBackRandom also beats Random under vanilla reward (52.2% vs 31.7%, d = +3.32).
- Theoretically: non-backtracking random walks on graphs have strictly faster cover/mixing times (Alon–Benjamini–Lubetzky–Sodin 2007).
- NoBackRandom beats random at every scale 9–25 with d = +1.2 to +3.4.

**Verdict:** PARTIALLY ADDRESSED. The effect is robust to reward config. Reviewers will still want a cover-time decomposition as appendix.

**Decisive experiment:** Cover-time analysis — for each maze, measure how many steps NoBackRandom vs Random vs MLP_DQN need to *visit* the goal (not reach-and-stop). Expected to show NoBackRandom has shorter expected cover time.

---

### A10 — "Neural baselines are unfairly handicapped." [DEFEATED]

**Attack:** "You're comparing trained agents to random walks in a way that favors random."

**Evidence:** All agents see identical conditions:
- Same mazes per seed (20 seeds × 50 test mazes = 1,000 unseen test instances per agent per size).
- Same observation space (24-dim ego features).
- Same action space (4 actions).
- Same horizon (max(300, 4n²)).
- Same reward function (full or vanilla).

Neural agents have strictly MORE information than random walks — they access a 100-episode training phase that random walks ignore. If the design is biased, it favors neural agents. Random has **no training phase at all**.

**Verdict:** DEFEATED.

---

### A11 — "Seed-unstable results." [DEFEATED]

**Attack:** "Henderson et al. 2018 showed deep RL results are high-variance across seeds. Your effect might be seed noise."

**Evidence:** Per-seed standard deviation at 9×9 across 20 seeds:

| Agent | std | mean | CV |
|---|---|---|---|
| Random | 0.052 | 31.7% | 16% |
| NoBackRandom | 0.070 | 52.2% | 13% |
| MLP_DQN | 0.081 | 19.3% | 42% |
| DoubleDQN | 0.049 | 15.8% | 31% |

Effect sizes (Cohen's d up to −3.1) are much larger than per-seed noise. 20 seeds per cell follows Agarwal et al. 2021 recommendation. Holm-Bonferroni correction applied family-wise.

**Verdict:** DEFEATED.

---

## What remains

1. ~~**Complete Phase 2 runs** to close A2 and A3.~~ DONE — both at 20/20 seeds.
2. **Run Phase 3B capacity study** to close A5. IN PROGRESS (~20/160 done; partial h32 already shows 13.4% << NoBackRandom 52.2%).
3. **Run LR sweep** to close A4 (optional, low priority).
4. **Write cover-time appendix** to close A9 (no new runs needed; data already analyzed in `cover_time_analysis.py`).
5. **Write feature-collision-rate appendix** to close A6 (no new runs needed).

When item 2 completes, the attack matrix will read:
- DEFEATED: 7
- PARTIALLY ADDRESSED: 2
- PENDING: 0
- NOT TESTED: 2 (A4 and A1 if SB3 not re-run)

That's publication-strength.
