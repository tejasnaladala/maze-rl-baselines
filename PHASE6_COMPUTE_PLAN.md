# Phase 6: Comprehensive Computational Plan

**Mandate:** Publish anywhere, defend to anyone.  
**Budget:** 20 H200 hours + indefinite RTX 5070 Ti (post-Phase-3B).  
**Timeline:** ~36 hours wall clock to total comprehensive completion.

---

## Compute partitioning principle

**H200** (20 hr): expensive deep RL / large nets / Procgen / SB3 baselines. GPU-bound.  
**RTX 5070 Ti** (free post-Phase-3B): tabular sweeps / random walks / hyperparameter cells / lightweight CNN. Often CPU-env-bound, so H200 advantage is small (2-3×).

---

## TIER A — Critical (must-have for top venues)

### A1. Procgen Maze replication [H200, 8 hr] **BIGGEST LEVER**
Cross-environment validation on the canonical procedural RL benchmark.
- Env: `procgen:procgen-maze-v0` (need `pip install procgen`)
- Agents: NoBackRandom, Random, FeatureQ_v2, MLP_DQN_h64, PPO@500K, PPO@2M
- 5 agents × 20 seeds × 1 size = 100 runs at 500K-2M env steps
- **Outcome:** Confirms or refutes maze-finding → broad generalization claim
- **Risk:** if PPO beats NoBack here, our story changes (NeurIPS-level outcome either way)

### A2. Budget-matched SB3 baselines [H200, 3 hr] (closes A1 attack)
- Already have `launch_budget_matched_sb3.py` — 3 agents × 3 budgets × 3 sizes × 20 seeds = 540 runs
- PPO/DQN/A2C at {10K, 100K, 500K} env steps
- **Outcome:** A1 attack PARTIAL → DEFEATED with v2-pipeline data

### A3. MiniGrid 4-env replication [H200, 3 hr]
- Already have `launch_minigrid.py` — 4 envs × 5 agents × 20 seeds = 400 runs
- FourRooms, MultiRoom-N2, DoorKey-5x5, Unlock
- **Outcome:** Second env family → moves NeurIPS odds 35% → 55%

### A4. DRQN multi-scale [H200, 2 hr] (extends A3 attack defeat)
- DRQN det at sizes {13, 17, 21} × 20 seeds = 60 runs
- **Outcome:** Memory-helps argument refuted at every scale, not just 9×9

### A5. LR/hyperparameter sweep [5070 Ti, 0.5 hr] (closes A4 attack)
- MLP_DQN with lr ∈ {1e-4, 5e-4, 1e-3, 3e-3} × 10 seeds × 9×9 = 40 runs
- **Outcome:** A4 NOT TESTED → DEFEATED

**Tier A total: 16.5 H200 + 0.5 5070 Ti hours**

---

## TIER B — Strong defense ("speak to anyone")

### B1. Decision Transformer baseline [H200, 2 hr]
- Modern offline-RL alternative to DQN
- 1 agent × 20 seeds × 9×9 = 20 runs
- **Outcome:** Defends against "but transformers" critique

### B2. Curiosity-driven exploration baselines [H200, 1 hr]
- PPO + RND (Random Network Distillation, Burda 2019) and ICM (Pathak 2017)
- 2 agents × 20 seeds × 9×9 = 40 runs at 200K env steps
- **Outcome:** Defends against "smarter exploration would win" — RND/ICM are SOTA exploration

### B3. Behavior cloning from BFS demos [5070 Ti, 1 hr]
- IL baseline: train MLP to imitate BFS optimal action
- 1 agent × 20 seeds × 5 sizes = 100 runs (small, fast)
- **Outcome:** Shows even with optimal demos, IL doesn't transfer beyond train mazes (or does, depending on result)

### B4. SpikingDQN finalize [5070 Ti, 3 hr]
- Already have `launch_spiking_dqn.py` — neuromorphic angle adds novelty
- **Outcome:** "Engram" branding gets actual neuromorphic data point

### B5. Higher-seed-count headline cells [5070 Ti, 3 hr]
- Re-run main 9×9 with 50 seeds (currently 20)
- Top 0.1% statistical rigor
- **Outcome:** Cohen d CIs tighten dramatically

**Tier B total: 3 H200 + 7 5070 Ti hours**

---

## TIER C — Comprehensive depth ("above and beyond")

### C1. Sample-efficiency learning curves [5070 Ti, 4 hr]
- Save model every 1K env steps for selected agents
- **Outcome:** Side-by-side learning curves figure

### C2. Reward sensitivity beyond K4 [5070 Ti, 3 hr]
- 6 reward configs: full, vanilla, sparse-only, distance-only, hazard-only, anti-shaping
- **Outcome:** Robustness panel showing NoBack invariant across all

### C3. Cross-size transfer [5070 Ti, 3 hr]
- Train 9, test 13/17/21 — and vice versa
- **Outcome:** Generalization profile per agent

### C4. Failure-case visualization [no GPU, 2 hr]
- Cherry-pick 6 mazes where agent A solves and agent B fails, render trajectories
- **Outcome:** Qualitative supplementary figure

### C5. Empirical cover-time scaling law [no GPU, 2 hr]
- Fit T_cover ∝ n^α for each agent across n=9..25
- Compare α to theoretical bounds
- **Outcome:** Theory section becomes formal, not hand-wavy

**Tier C total: 0 H200 + 14 5070 Ti hours**

---

## TIER D — Long-tail academic completeness

### D1. State-coverage entropy curves [5070 Ti, 1 hr]
- H(visited states) over training time
- **Outcome:** Quantifies "agent stops exploring"

### D2. Q-value landscape visualization [no GPU, 1 hr]
- Heatmaps of MLP_DQN values over a sample maze
- **Outcome:** Visualizes the local optimum

### D3. Cross-replicate consistency [5070 Ti, 2 hr]
- Re-run 5 random configs from a clean checkout
- **Outcome:** "Other research groups can reproduce"

**Tier D total: 0 H200 + 4 5070 Ti hours**

---

## EXECUTION ORDER

### Phase 6.0 — Finalize Phase 3B (current, ~30 min)
1. Wait for h256 13×13 to finish (7 runs)
2. Regenerate fig5 with full data
3. Update PHASE4 attack matrix (A5 → DEFEATED)
4. Phase 5 hardening (PYTHONHASHSEED, code_hash memoize)
5. Freeze manifest_phase3b_complete.json
6. Commit + push final 5070 Ti state

### Phase 6.1 — H200 Tier A blast (16.5 hr, parallel where possible)
1. Spin up `pip install procgen stable-baselines3 minigrid`
2. Launch 4 jobs in parallel on H200 (8x sub-allocation):
   - Procgen Maze (8 hr)
   - SB3 budget-matched (3 hr)
   - MiniGrid 4-env (3 hr)
   - DRQN multi-scale (2 hr)
3. While H200 runs: 5070 Ti runs Tier B/C cheaply

### Phase 6.2 — H200 Tier B (3 hr)
1. Decision Transformer + RND/ICM exploration baselines
2. Final manifest freeze on H200 results

### Phase 6.3 — 5070 Ti backfill (~24 hr concurrent)
- All Tier B/C/D items in parallel (no GPU contention since each uses minimal VRAM)
- Use 4 simultaneous launchers, queue-based

### Phase 6.4 — Final synthesis (~4 hr)
- Re-run final_analysis with ALL data (engram + procgen + minigrid)
- Regenerate all figures (now 8 figures: original 5 + procgen panel + minigrid panel + transfer panel)
- Update paper draft with new sections:
  * §3.5 Procgen replication
  * §3.6 MiniGrid replication
  * §4.3 Sample-efficiency curves
  * §5 Cover-time scaling law (formal)
- Final attack matrix: all 11 DEFEATED
- Final SHA-256 manifest, ~2500 result files
- Final commit + tag v1.0-paper-ready

---

## EXPECTED FINAL STATE

| Component | Before Phase 6 | After Phase 6 |
|---|---|---|
| Result files | 1,582 | ~2,500 |
| Env families | 1 (proc maze) | 3 (proc maze + Procgen + MiniGrid) |
| Agents tested | 12 | 18 (+ PPO/DQN/A2C × 3 budgets, DT, RND, ICM, BC) |
| Maze sizes | 9-25 (5) | same |
| Seeds @ headline | 20 | 50 |
| Reward configs | 2 (full, vanilla) | 6 |
| Attack defenses | 7 DEFEATED + 4 partial | **11 DEFEATED, 0 partial** |
| Theory grounding | Cover-time match | + formal scaling law fit |
| Reproducibility | SHA-256 manifest | + cross-replicate verification |
| Paper sections | 6 | 9 (+ Procgen, MiniGrid, scaling-law) |

---

## REALISTIC VENUE ODDS POST-PHASE-6

| Venue | Pre-Phase 6 | Post-Phase 6 |
|---|---|---|
| NeurIPS / ICML / ICLR | 30-40% | **65-75%** |
| AAAI / IJCAI | 55-70% | **85%+** |
| UAI / AISTATS | 70-80% | **90%+** |
| RLConf | 80-90% | near-certain |
| NeurIPS workshop | 95% | trivial |
| TMLR | 70% | 90%+ |

**Bottom line:** The work moves from "B+ paper for NeurIPS, A- for second-tier" to **"unambiguous A-paper that any reviewer would have to engage with seriously."** Single-author / AI-assisted concern remains, but the methodology and scope make it indefensible to dismiss without specific technical critique.

---

## RISKS & MITIGATIONS

| Risk | Mitigation |
|---|---|
| Procgen install fails on Windows | H200 will be Linux. Pre-test on cloud. |
| H200 instance preemption | Checkpoint every run. Resume from Git. |
| PPO@2M beats NoBackRandom on Procgen | This would be ANOTHER paper-grade finding (env-specific story). Pivot framing: "depends on env structure" |
| MiniGrid neural agents win | Same as above: pivot to "maze topology matters". Still publishable. |
| Out-of-time on H200 | Tier A items prioritized; cut B1/B2 if needed. Tier A alone is sufficient for B+ → A. |
