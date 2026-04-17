# Graph Report - .  (2026-04-17)

## Corpus Check
- 184 files · ~270,234 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1598 nodes · 3155 edges · 64 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 1218 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Anomaly Stream Env|Anomaly Stream Env]]
- [[_COMMUNITY_Bayesian Analysis|Bayesian Analysis]]
- [[_COMMUNITY_ExperimentLib v1 Core|ExperimentLib v1 Core]]
- [[_COMMUNITY_DRQN + Continual Learning|DRQN + Continual Learning]]
- [[_COMMUNITY_Action Selector  Neurons|Action Selector / Neurons]]
- [[_COMMUNITY_Failure Case Viz|Failure Case Viz]]
- [[_COMMUNITY_Prior Art Citations|Prior Art Citations]]
- [[_COMMUNITY_Rust STDP Learning Rules|Rust STDP Learning Rules]]
- [[_COMMUNITY_Approach C NeurIPS Suite|Approach C NeurIPS Suite]]
- [[_COMMUNITY_Engram CLI|Engram CLI]]
- [[_COMMUNITY_Training Scripts|Training Scripts]]
- [[_COMMUNITY_Rust Runtime Protocols|Rust Runtime Protocols]]
- [[_COMMUNITY_SB3 Evaluation Wrapper|SB3 Evaluation Wrapper]]
- [[_COMMUNITY_WASM Bindings|WASM Bindings]]
- [[_COMMUNITY_SNN Literature|SNN Literature]]
- [[_COMMUNITY_Engram Runtime Crates|Engram Runtime Crates]]
- [[_COMMUNITY_Core Lib Utilities|Core Lib Utilities]]
- [[_COMMUNITY_Maze Gym Env|Maze Gym Env]]
- [[_COMMUNITY_ExperimentLib v2 Agents|ExperimentLib v2 Agents]]
- [[_COMMUNITY_Spike Mechanics (Rust)|Spike Mechanics (Rust)]]
- [[_COMMUNITY_C20 capacity_breakdown(), clear()|C20: capacity_breakdown(), clear()]]
- [[_COMMUNITY_C21 run_strong_baselines.py, evaluate_sb3()|C21: run_strong_baselines.py, evaluate_sb3()]]
- [[_COMMUNITY_C22 LiveAgent.tsx, createDual()|C22: LiveAgent.tsx, createDual()]]
- [[_COMMUNITY_C23 (n=5)|C23 (n=5)]]
- [[_COMMUNITY_C24 (n=4)|C24 (n=4)]]
- [[_COMMUNITY_C25 (n=3)|C25 (n=3)]]
- [[_COMMUNITY_C26 (n=2)|C26 (n=2)]]
- [[_COMMUNITY_C27 (n=2)|C27 (n=2)]]
- [[_COMMUNITY_C28 (n=2)|C28 (n=2)]]
- [[_COMMUNITY_C29 (n=2)|C29 (n=2)]]
- [[_COMMUNITY_C30 (n=2)|C30 (n=2)]]
- [[_COMMUNITY_C31 (n=2)|C31 (n=2)]]
- [[_COMMUNITY_C32 (n=2)|C32 (n=2)]]
- [[_COMMUNITY_C33 (n=1)|C33 (n=1)]]
- [[_COMMUNITY_C34 (n=1)|C34 (n=1)]]
- [[_COMMUNITY_C35 (n=1)|C35 (n=1)]]
- [[_COMMUNITY_C36 (n=1)|C36 (n=1)]]
- [[_COMMUNITY_C37 (n=1)|C37 (n=1)]]
- [[_COMMUNITY_C38 (n=1)|C38 (n=1)]]
- [[_COMMUNITY_C39 (n=1)|C39 (n=1)]]
- [[_COMMUNITY_C40 (n=1)|C40 (n=1)]]
- [[_COMMUNITY_C41 (n=1)|C41 (n=1)]]
- [[_COMMUNITY_C42 (n=1)|C42 (n=1)]]
- [[_COMMUNITY_C43 (n=1)|C43 (n=1)]]
- [[_COMMUNITY_C44 (n=1)|C44 (n=1)]]
- [[_COMMUNITY_C45 (n=1)|C45 (n=1)]]
- [[_COMMUNITY_C46 (n=1)|C46 (n=1)]]
- [[_COMMUNITY_C47 (n=1)|C47 (n=1)]]
- [[_COMMUNITY_C48 (n=1)|C48 (n=1)]]
- [[_COMMUNITY_C49 (n=1)|C49 (n=1)]]
- [[_COMMUNITY_C50 (n=1)|C50 (n=1)]]
- [[_COMMUNITY_C51 (n=1)|C51 (n=1)]]
- [[_COMMUNITY_C52 (n=1)|C52 (n=1)]]
- [[_COMMUNITY_C53 (n=1)|C53 (n=1)]]
- [[_COMMUNITY_C54 (n=1)|C54 (n=1)]]
- [[_COMMUNITY_C55 (n=1)|C55 (n=1)]]
- [[_COMMUNITY_C56 (n=1)|C56 (n=1)]]
- [[_COMMUNITY_C57 (n=1)|C57 (n=1)]]
- [[_COMMUNITY_C58 (n=1)|C58 (n=1)]]
- [[_COMMUNITY_C59 (n=1)|C59 (n=1)]]
- [[_COMMUNITY_C60 (n=1)|C60 (n=1)]]
- [[_COMMUNITY_C61 (n=1)|C61 (n=1)]]
- [[_COMMUNITY_C62 (n=1)|C62 (n=1)]]
- [[_COMMUNITY_C63 (n=1)|C63 (n=1)]]

## God Nodes (most connected - your core abstractions)
1. `ReplayBuffer` - 48 edges
2. `SpikingQNetwork` - 43 edges
3. `Runtime` - 34 edges
4. `NoBacktrackRandomAgent` - 31 edges
5. `load_checkpoint()` - 29 edges
6. `run_experiment()` - 29 edges
7. `default_rng()` - 28 edges
8. `code_hash()` - 27 edges
9. `save_checkpoint()` - 26 edges
10. `atomic_save()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `Three-Factor STDP Learning` --semantically_similar_to--> `stats_pipeline.py`  [AMBIGUOUS] [semantically similar]
  ARCHITECTURE.md → paper.md
- `Session TL;DR (Publishable Finding)` --semantically_similar_to--> `Steelmanned Thesis: Neural RL < Random on Procedural Mazes`  [INFERRED] [semantically similar]
  SESSION_REPORT.md → COMPREHENSIVE_AUDIT.md
- `Tier 4b: Memory-augmented agents (DRQN) to test partial-observability hypothesis` --uses--> `ExpResult`  [INFERRED]
  launch_memory_agents.py → experiment_lib_v2.py
- `Tier A.4 — DRQN multi-scale extension.  Already have DRQN at 9x9 (n=20, mean=19.` --uses--> `DRQNAgent`  [INFERRED]
  launch_drqn_multiscale.py → launch_memory_agents.py
- `Train an Engram agent on the Grid World environment.  Usage:     python examples` --uses--> `GridWorldEnv`  [INFERRED]
  examples\train_grid_world.py → python\engram\environments\grid_world.py

## Hyperedges (group relationships)
- **** — concept_bfs_oracle, concept_wall_follower, concept_nobackrandom, concept_mlp_dqn, paper_table1_main [EXTRACTED 1.00]
- **** — paper_distillation_diagnostic, concept_mlp_dqn, concept_bfs_oracle, paper_central_claim [EXTRACTED 1.00]
- **** — phase4_a2_reward_shaping, phase4_a3_state_aliasing, phase4_a5_network_size, phase4_a4_hyperparams, phase4_a8_hazard_dominance [EXTRACTED 1.00]
- **Codex reviews drive Approach C experiment design via risk register** — codex_review_1, codex_review_2, risk_register, approach_c_plan [INFERRED 0.85]
- **Final results inform paper framing, refining preliminary reframing** — preliminary_results, final_results, paper_outline, paper_abstract [INFERRED 0.90]
- **Spiking RL canonical citations shared across truth doc, paper, and related work** — paper_cite_dsqn, paper_cite_popsan, paper_cite_ilc_san, paper_cite_proxy_target, paper_cite_care_bn, paper_cite_adaptive_sg [EXTRACTED 1.00]

## Communities

### Community 0 - "Anomaly Stream Env"
Cohesion: 0.02
Nodes (72): AnomalyStreamEnv, Streaming anomaly detection environment.  Tests the system's ability to learn no, Streaming sensor data with rare anomalies.      The environment produces a strea, Generate a sensor reading (normal or anomalous)., avg_reward_last_20(), bench_continual_learning(), bench_grid_world(), bench_pattern_recognition() (+64 more)

### Community 1 - "Bayesian Analysis"
Cohesion: 0.03
Nodes (123): hdi(), load_test_outcomes(), main(), posterior_beta(), Bayesian hierarchical analysis (per Codex review).  For each agent at 9x9, fits, Returns: agent -> list of (n_success, n_total) tuples per seed., Conjugate update: Beta(alpha + s, beta + n - s) per seed, then     mixture over, Highest-density interval via interval-width minimization. (+115 more)

### Community 2 - "ExperimentLib v1 Core"
Cohesion: 0.03
Nodes (76): load_checkpoint(), make_dense_maze(), make_maze(), make_sparse_maze(), mulberry32(), Shared infrastructure for all Approach C experiments.  Contains: maze generation, Maze with walls removed (open space)., Generate a solvable maze using recursive backtracking. Size must be odd. (+68 more)

### Community 3 - "DRQN + Continual Learning"
Cohesion: 0.03
Nodes (65): main(), Task 4: Ablation Study -- which ego-centric features matter for generalization?, main(), Task 5: Continual Learning -- does training on new mazes cause forgetting?  Prot, Run one phase of continual learning., DRQNAgent, DRQNNetwork, main() (+57 more)

### Community 4 - "Action Selector / Neurons"
Cohesion: 0.03
Nodes (21): ActionSelector, genDemo(), AssociativeMemory, deserialize(), roundtrip_msgpack(), serialize(), Episode, EpisodeFrame (+13 more)

### Community 5 - "Failure Case Viz"
Cohesion: 0.04
Nodes (79): main(), Failure case visualization: cherry-pick mazes where neural agent fails but a sim, Return (trajectory list of (ax, ay), solved, steps)., Render one maze panel., render_one(), trace_agent(), run_phase(), is_solvable() (+71 more)

### Community 6 - "Prior Art Citations"
Cohesion: 0.02
Nodes (93): Agarwal et al. 2021 (Statistical Precipice), Alon-Benjamini-Lubetzky-Sodin 2007 (Non-backtracking Cover-time), Chevalier-Boisvert et al. (MiniGrid/BabyAI), Cobbe et al. ICML 2020 (ProcGen), Dohare 2024 (Plasticity Loss), Ghosh et al. 2021 (Epistemic POMDPs), Kuttler et al. NeurIPS 2020 (NetHack), Codex MCP Adversarial Review (+85 more)

### Community 7 - "Rust STDP Learning Rules"
Cohesion: 0.04
Nodes (27): RuntimeConfig, HebbianRule, LearningRule, Neuromodulators, neuromodulators_update_from_reward(), three_factor_eligibility_accumulates(), three_factor_reward_drives_weight_change(), ThreeFactorSTDP (+19 more)

### Community 8 - "Approach C NeurIPS Suite"
Cohesion: 0.03
Nodes (79): ANN Energy Estimate (4.6 pJ/MAC), 24-dim Ego-Centric Features, Loihi 2 Energy Estimate (0.9 pJ/SynOp), Approach C NeurIPS-Grade Experiment Suite, ProceduralMazeEnv (Gymnasium), Rationale: Ablate FeatureQ (the winner's features), Rationale: H200 needed for spiking sweep (120h compute), rliable IQM Methodology (+71 more)

### Community 9 - "Engram CLI"
Cohesion: 0.04
Nodes (35): dashboard(), main(), Engram CLI -- command-line interface for the cognitive runtime., Engram -- Brain-inspired adaptive intelligence runtime., Run an Engram agent in the grid world environment., Start the Engram server and open the dashboard., run(), ReplayBuffer (+27 more)

### Community 10 - "Training Scripts"
Cohesion: 0.05
Nodes (39): find_start_goal(), ImageMazeEnv, load_maze_from_image(), Load a maze from an image file and solve it with Engram.  Supports PNG/JPG maze, Environment that loads a maze from an image file.      The agent must navigate f, Render maze as RGB image array for video/display.          Returns numpy array o, Load a maze image and convert to a binary grid.      Args:         path: Path to, Find start and goal positions.      Tries to find green (start) and red (goal) p (+31 more)

### Community 11 - "Rust Runtime Protocols"
Cohesion: 0.06
Nodes (22): AppState, main(), ws_upgrade(), decode_client_message(), encode_server_message(), hello_message(), High-level Python wrapper for the Engram cognitive runtime., tick_count() (+14 more)

### Community 12 - "SB3 Evaluation Wrapper"
Cohesion: 0.07
Nodes (14): evaluate_agent(), evaluate_sb3(), GreedyToGoalAgent, main(), ProceduralMazeGym, Fixed strong baselines with proper training budgets.  Codex review finding: 20K-, Right-hand wall following., Evaluate any agent on unseen mazes (no learning). (+6 more)

### Community 13 - "WASM Bindings"
Cohesion: 0.11
Nodes (19): decodeText(), getArrayF32FromWasm0(), getArrayU8FromWasm0(), getDataViewMemory0(), getFloat32ArrayMemory0(), getStringFromWasm0(), getUint8ArrayMemory0(), initSync() (+11 more)

### Community 14 - "SNN Literature"
Cohesion: 0.06
Nodes (35): Citation: BindsNET creators on SNN OpenAI gym failures, Citation: CoLaNET, Citation: DSQN (Chen et al. 2022), Citation: NorthPole (Science), LITERATURE: CoLaNET continual learning (92% across 10 tasks), LITERATURE: DSQN 193.5% vs DQN 142.8% on Atari (Chen et al. 2022), LITERATURE: SNN energy efficiency (NorthPole, Loihi), PLAUSIBLE: Dual-phase training (surrogate + STDP) (+27 more)

### Community 15 - "Engram Runtime Crates"
Cohesion: 0.07
Nodes (35): CSR Sparse Synapses Rationale, Engram Cognitive Runtime, engram-core Crate, engram-modules Crate, engram-python (PyO3 bindings), engram-runtime Crate, engram-server (Axum WebSocket), engram-wasm (WebAssembly Target) (+27 more)

### Community 16 - "Core Lib Utilities"
Cohesion: 0.08
Nodes (13): bfs_path(), is_solvable(), LevyRandomAgent, make_dense_maze(), make_maze(), make_sparse_maze(), mulberry32(), Experiment library v2 — audit-fixed version for Tier 2+ experiments.  Changes fr (+5 more)

### Community 17 - "Maze Gym Env"
Cohesion: 0.14
Nodes (12): ExpResult, main(), MazeGymEnv, Tier 2: Budget-matched SB3 baselines (PPO, DQN, A2C).  The original exp1b gave S, Zero-shot rollouts of a trained model on unseen mazes., Gymnasium wrapper for SB3 agents., test_rollouts(), train_and_test() (+4 more)

### Community 18 - "ExperimentLib v2 Agents"
Cohesion: 0.12
Nodes (21): experiment_lib_v2.py (Audit-fixed library), W6 Bug: FeatureQ epsilon-floor at eval, BFSOracle Agent, DoubleDQN Agent, DRQN Agent (LSTM), FeatureQ Tabular Agent, LevyRandom Agent, MLP_DQN Agent (+13 more)

### Community 19 - "Spike Mechanics (Rust)"
Cohesion: 0.19
Nodes (6): make_spike(), spike_buffer_evicts_oldest(), spike_buffer_recent_window(), spike_train_firing_rate(), SpikeBuffer, SpikeTrain

### Community 20 - "C20: capacity_breakdown(), clear()"
Cohesion: 0.22
Nodes (14): capacity_breakdown(), clear(), count_results(), enable_vt(), gpu_status(), main(), newest_log_in(), python_procs() (+6 more)

### Community 21 - "C21: run_strong_baselines.py, evaluate_sb3()"
Cohesion: 0.24
Nodes (8): evaluate_sb3(), main(), mulberry32(), ProceduralMazeEnv, Strong modern baselines: PPO + DQN (stable-baselines3) on procedural mazes.  Tes, Evaluate a stable-baselines3 model on procedural mazes., Gymnasium-compatible procedural maze with ego-centric features., Result

### Community 22 - "C22: LiveAgent.tsx, createDual()"
Cohesion: 0.38
Nodes (10): createDual(), dualStep(), featureKey(), isSolvable(), loadQTable(), mulberry32(), newRunner(), randomMaze() (+2 more)

### Community 23 - "C23 (n=5)"
Cohesion: 0.6
Nodes (4): human_time(), main(), progress_monitor.py — one-shot status of all running experiments.  Scans raw_res, scan_tier()

### Community 24 - "C24 (n=4)"
Cohesion: 0.5
Nodes (4): Cohen's d Effect Size, Holm-Bonferroni Correction, Paired Percentile Bootstrap (10K resamples), Evaluation Protocol (Paired Bootstrap)

### Community 25 - "C25 (n=3)"
Cohesion: 0.67
Nodes (0): 

### Community 26 - "C26 (n=2)"
Cohesion: 1.0
Nodes (1): BrainModule

### Community 27 - "C27 (n=2)"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "C28 (n=2)"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "C29 (n=2)"
Cohesion: 1.0
Nodes (1): WasmRuntime

### Community 30 - "C30 (n=2)"
Cohesion: 1.0
Nodes (2): Competing Explanations (A1-A8), Falsification Criteria / Claims C1-C7

### Community 31 - "C31 (n=2)"
Cohesion: 1.0
Nodes (2): Experiment Data Inventory, Phase Completion Table

### Community 32 - "C32 (n=2)"
Cohesion: 1.0
Nodes (2): Cover-Time Theory (ABLS 2007), Section 4.9: Formal Cover-Time Scaling Law

### Community 33 - "C33 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "C34 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "C35 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "C36 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "C37 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "C38 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "C39 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "C40 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "C41 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "C42 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "C43 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "C44 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "C45 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "C46 (n=1)"
Cohesion: 1.0
Nodes (1): Current prediction error magnitude.

### Community 47 - "C47 (n=1)"
Cohesion: 1.0
Nodes (1): Current simulation tick.

### Community 48 - "C48 (n=1)"
Cohesion: 1.0
Nodes (1): Lifetime spike count.

### Community 49 - "C49 (n=1)"
Cohesion: 1.0
Nodes (1): Lifetime safety veto count.

### Community 50 - "C50 (n=1)"
Cohesion: 1.0
Nodes (1): Simulation time in milliseconds.

### Community 51 - "C51 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "C52 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "C53 (n=1)"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "C54 (n=1)"
Cohesion: 1.0
Nodes (1): Rust Core Codebase Audit (C1-C7 bugs)

### Community 55 - "C55 (n=1)"
Cohesion: 1.0
Nodes (1): Security Audit (CorsLayer Permissive, Unbounded MessagePack)

### Community 56 - "C56 (n=1)"
Cohesion: 1.0
Nodes (1): Dashboard 100% Fake Data Finding

### Community 57 - "C57 (n=1)"
Cohesion: 1.0
Nodes (1): Development Prerequisites (Rust/Python/Node)

### Community 58 - "C58 (n=1)"
Cohesion: 1.0
Nodes (1): Code Hash fe0b8142940e55de

### Community 59 - "C59 (n=1)"
Cohesion: 1.0
Nodes (1): arXiv-ready Verdict

### Community 60 - "C60 (n=1)"
Cohesion: 1.0
Nodes (1): Environment Setup (Recursive Backtracking Mazes)

### Community 61 - "C61 (n=1)"
Cohesion: 1.0
Nodes (1): Table 4.5: Cover-Time Decomposition

### Community 62 - "C62 (n=1)"
Cohesion: 1.0
Nodes (1): Table 10: MiniGrid Cross-Env Results

### Community 63 - "C63 (n=1)"
Cohesion: 1.0
Nodes (1): Compute & Run Counts Table

## Ambiguous Edges - Review These
- `Three-Factor STDP Learning` → `stats_pipeline.py`  [AMBIGUOUS]
  ARCHITECTURE.md · relation: semantically_similar_to

## Knowledge Gaps
- **322 isolated node(s):** `Bayesian hierarchical analysis (per Codex review).  For each agent at 9x9, fits`, `Returns: agent -> list of (n_success, n_total) tuples per seed.`, `Conjugate update: Beta(alpha + s, beta + n - s) per seed, then     mixture over`, `Highest-density interval via interval-width minimization.`, `Analyze MiniGrid 4-env results across agents.` (+317 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `C26 (n=2)`** (2 nodes): `module_trait.rs`, `BrainModule`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C27 (n=2)`** (2 nodes): `MemoryHeatMap.tsx`, `handleMouse()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C28 (n=2)`** (2 nodes): `MetricsBar.tsx`, `fmt()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C29 (n=2)`** (2 nodes): `WasmRuntime`, `engram_wasm.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C30 (n=2)`** (2 nodes): `Competing Explanations (A1-A8)`, `Falsification Criteria / Claims C1-C7`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C31 (n=2)`** (2 nodes): `Experiment Data Inventory`, `Phase Completion Table`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C32 (n=2)`** (2 nodes): `Cover-Time Theory (ABLS 2007)`, `Section 4.9: Formal Cover-Time Scaling Law`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C33 (n=1)`** (1 nodes): `lib.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C34 (n=1)`** (1 nodes): `lib.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C35 (n=1)`** (1 nodes): `lib.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C36 (n=1)`** (1 nodes): `lib.rs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C37 (n=1)`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C38 (n=1)`** (1 nodes): `CinematicDemo.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C39 (n=1)`** (1 nodes): `main.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C40 (n=1)`** (1 nodes): `BrainVisualization.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C41 (n=1)`** (1 nodes): `ModuleActivity.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C42 (n=1)`** (1 nodes): `PredictionError.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C43 (n=1)`** (1 nodes): `SafetyLog.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C44 (n=1)`** (1 nodes): `SpikeRaster.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C45 (n=1)`** (1 nodes): `theme.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C46 (n=1)`** (1 nodes): `Current prediction error magnitude.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C47 (n=1)`** (1 nodes): `Current simulation tick.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C48 (n=1)`** (1 nodes): `Lifetime spike count.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C49 (n=1)`** (1 nodes): `Lifetime safety veto count.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C50 (n=1)`** (1 nodes): `Simulation time in milliseconds.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C51 (n=1)`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C52 (n=1)`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C53 (n=1)`** (1 nodes): `engram_wasm_bg.wasm.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C54 (n=1)`** (1 nodes): `Rust Core Codebase Audit (C1-C7 bugs)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C55 (n=1)`** (1 nodes): `Security Audit (CorsLayer Permissive, Unbounded MessagePack)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C56 (n=1)`** (1 nodes): `Dashboard 100% Fake Data Finding`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C57 (n=1)`** (1 nodes): `Development Prerequisites (Rust/Python/Node)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C58 (n=1)`** (1 nodes): `Code Hash fe0b8142940e55de`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C59 (n=1)`** (1 nodes): `arXiv-ready Verdict`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C60 (n=1)`** (1 nodes): `Environment Setup (Recursive Backtracking Mazes)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C61 (n=1)`** (1 nodes): `Table 4.5: Cover-Time Decomposition`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C62 (n=1)`** (1 nodes): `Table 10: MiniGrid Cross-Env Results`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C63 (n=1)`** (1 nodes): `Compute & Run Counts Table`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Three-Factor STDP Learning` and `stats_pipeline.py`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `Runtime` connect `Anomaly Stream Env` to `Engram CLI`, `Training Scripts`, `Rust Runtime Protocols`, `Failure Case Viz`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `ReplayBuffer` connect `DRQN + Continual Learning` to `Bayesian Analysis`, `ExperimentLib v1 Core`, `Action Selector / Neurons`, `Failure Case Viz`, `Engram CLI`, `Training Scripts`, `SB3 Evaluation Wrapper`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `main()` connect `ExperimentLib v1 Core` to `Bayesian Analysis`, `DRQN + Continual Learning`, `Failure Case Viz`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 42 inferred relationships involving `ReplayBuffer` (e.g. with `ExpResult` and `Agent`) actually correct?**
  _`ReplayBuffer` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `SpikingQNetwork` (e.g. with `ExpResult` and `Agent`) actually correct?**
  _`SpikingQNetwork` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `Runtime` (e.g. with `Environment` and `EpisodeResult`) actually correct?**
  _`Runtime` has 25 INFERRED edges - model-reasoned connections that need verification._