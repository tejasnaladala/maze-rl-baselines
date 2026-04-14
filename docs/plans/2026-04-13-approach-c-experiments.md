# Approach C: NeurIPS-Grade Experiment Suite

**Goal:** Build and run a comprehensive experiment suite comparing spiking vs non-spiking RL agents on cross-environment generalization, with enough rigor for IEEE Trans Cybernetics or Frontiers in Neuroscience.

**Architecture:** Single Python script (`research/01_experiments/run_approach_c.py`) that runs all experiments, saves per-run JSON, and generates figures. Uses existing `engram.spiking_dqn.SpikingQNetwork` for spiking agents and `stable_baselines3` for strong baselines. All runs are resumable via checkpoint files.

**Tech Stack:** Python 3.11, PyTorch, snnTorch, stable-baselines3, matplotlib, scipy (for stats)

---

### Task 1: Shared Infrastructure -- Maze Environment + Feature Extraction + Agent Base Classes

**Files:**
- Create: `research/01_experiments/experiment_lib.py`

**What this file contains:**
- `make_maze(size, seed)` -- procedural maze generation (reuse from run_all.py)
- `ego_features(grid, ax, ay, gx, gy, size, action_hist)` -- 24-dim feature vector
- `ego_features_ablated(grid, ax, ay, gx, gy, size, action_hist, ablation)` -- feature vector with one component removed
- `ProceduralMazeEnv(gym.Env)` -- Gymnasium-compatible env for SB3 agents
- `run_experiment(agent, maze_size, num_train, num_test, seed, max_steps)` -- shared train+test loop
- `ExpResult` dataclass for structured output
- `save_results(results, path)` and `load_results(path)` for JSON I/O
- 10 agent classes (see Task 2)
- `compute_synops(agent)` -- SynOps counter for energy measurement

**Step 1:** Write the file with all shared infrastructure. Include type annotations on all functions.

**Step 2:** Verify import works: `python -c "from research.01_experiments.experiment_lib import make_maze, ego_features"`

**Step 3:** Commit: `git add research/01_experiments/experiment_lib.py && git commit -m "research: shared experiment infrastructure"`

---

### Task 2: All 10 Agent Implementations

**Files:**
- Modify: `research/01_experiments/experiment_lib.py`

**Agents to implement:**

1. `RandomAgent` -- uniform random baseline
2. `TabularQAgent` -- position-based tabular Q-learning (Q-table wiped each maze)
3. `FeatureQAgent` -- ego-centric feature Q-learning (Q-table persists across mazes)
4. `MLPDQNAgent` -- 2-layer MLP DQN with ego features, PyTorch, replay buffer, target network
5. `SpikingDQNAgent` -- spiking DQN using `SpikingQNetwork` from `engram.spiking_dqn`
6. `DoubleDQNAgent` -- Double DQN (separate action selection and evaluation networks)
7. `SB3_PPO_Features` -- stable-baselines3 PPO with ego feature observations
8. `SB3_DQN_Features` -- stable-baselines3 DQN with ego feature observations
9. `SB3_PPO_RawGrid` -- stable-baselines3 PPO with flattened raw grid observation
10. `SB3_A2C_Features` -- stable-baselines3 A2C with ego features

Each agent must implement:
- `act(obs, step) -> int`
- `learn(obs, action, reward, next_obs, done)`
- `reset_for_new_maze()`
- `get_synops() -> int`

**Step 1:** Implement all 10 agents with consistent interface.

**Step 2:** Smoke test each agent: run 2 episodes on a 9x9 maze and verify no crashes.

**Step 3:** Commit.

---

### Task 3: Primary Experiment -- Cross-Environment Generalization

**Files:**
- Create: `research/01_experiments/exp1_generalization.py`

**Configuration:**
```
AGENTS = [all 10]
MAZE_SIZES = [9, 11, 13, 17, 21, 25]
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]
NUM_TRAIN = 100 mazes per seed
NUM_TEST = 50 unseen mazes per seed
MAX_STEPS = 300
TRAIN_TIMESTEPS_SB3 = 30000 (for SB3 agents)
```

**Output:** `research/01_experiments/raw_results/exp1_generalization/` with one JSON per (agent, size, seed) combination.

**Checkpointing:** After each (agent, size, seed) completes, save result and mark as done in `checkpoint.json`. On restart, skip completed runs.

**Estimated compute:**
- TabularQ/FeatureQ/Random: ~1s per (seed, size) = 120 runs * 1s = 2 min
- MLP/Double DQN: ~30s per run = 120 runs * 30s = 60 min
- SpikingDQN: ~6min per run = 120 runs * 6min = 12 hours
- SB3 agents: ~20s per run = 360 runs * 20s = 2 hours
- **Total ~15 hours on RTX 5070 Ti, or ~2 hours on H200**

**Step 1:** Write the experiment runner with checkpointing.

**Step 2:** Run a smoke test: 1 seed, 1 size, all agents.

**Step 3:** Commit.

**Step 4:** Ping user for H200 access to run full suite.

---

### Task 4: Ablation Study

**Files:**
- Create: `research/01_experiments/exp2_ablation.py`

**Configuration:**
```
AGENT = FeatureQAgent (the winner -- ablate ITS features)
ABLATIONS = [
    'full',              # all features (baseline)
    'no_action_hist',    # remove last-3-action history from features
    'no_3x3_map',        # replace 3x3 map with 4-direction walls only
    'no_goal_dir',       # remove goal direction features
    'no_distance',       # remove distance-to-goal
    'no_visit_penalty',  # remove revisit penalty from reward
    'no_dist_shaping',   # remove distance-based reward shaping
    'walls_only',        # only 4-direction walls + goal direction (minimal)
]
MAZE_SIZE = 13 (mid-range)
SEEDS = 10 seeds
NUM_TRAIN = 100, NUM_TEST = 50
```

**Output:** `research/01_experiments/raw_results/exp2_ablation/`

**Estimated compute:** 8 ablations * 10 seeds * ~1s = under 2 min.

**Step 1:** Implement ablated feature functions.

**Step 2:** Run and save.

**Step 3:** Commit.

---

### Task 5: Continual Learning Experiment

**Files:**
- Create: `research/01_experiments/exp3_continual.py`

**Protocol:**
```
Phase A: Train on "dense" mazes (many walls, tight corridors) -- 50 mazes
Phase B: Train on "sparse" mazes (few walls, open space) -- 50 mazes
Phase C: Test recall on "dense" mazes (no further training) -- 30 mazes
```

**Agents:** FeatureQ, TabularQ, MLP DQN, SpikingDQN, SB3 PPO
**Maze size:** 13x13
**Seeds:** 10

**Metrics:**
- Phase A final success rate
- Phase B final success rate
- Phase C recall rate (THE forgetting metric)
- Forgetting = Phase A final - Phase C recall

**Dense maze generation:** `make_maze` with extra wall additions post-generation.
**Sparse maze generation:** `make_maze` with wall removals post-generation.

**Estimated compute:** 5 agents * 10 seeds * ~2min = ~100 min.

**Step 1:** Implement dense/sparse maze variants.

**Step 2:** Implement the 3-phase protocol.

**Step 3:** Run and save.

**Step 4:** Commit.

---

### Task 6: Hyperparameter Sensitivity Analysis

**Files:**
- Create: `research/01_experiments/exp4_hyperparam.py`

**Sweep for SpikingDQN:**
```
learning_rates = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
num_timesteps = [2, 4, 8, 16]
hidden_sizes = [32, 64, 128]
epsilon_decays = [2000, 5000, 10000, 20000]
```

**Sweep for FeatureQ:**
```
learning_rates = [0.05, 0.1, 0.2, 0.4, 0.8]
epsilon_starts = [0.1, 0.2, 0.3, 0.5]
optimistic_init = [0.0, 0.5, 1.0, 2.0]
```

**Maze size:** 13x13
**Seeds:** 5 per config
**Metric:** Test success rate

**Output:** `research/01_experiments/raw_results/exp4_hyperparam/`

**Estimated compute:** ~240 SpikingDQN configs * 5 seeds * 6min = 120 hours. **Needs H200.**

**Step 1:** Implement grid search with checkpointing.

**Step 2:** Run on H200.

**Step 3:** Commit.

---

### Task 7: Energy Efficiency Analysis

**Files:**
- Create: `research/01_experiments/exp5_energy.py`

**Method:**
- Count SynOps (synaptic operations) per inference step for spiking agents
- Count MACs (multiply-accumulate operations) per step for MLP/DQN agents
- Compute theoretical energy ratio using:
  - SNN: 0.9 pJ per SynOp (Loihi 2 estimate)
  - ANN: 4.6 pJ per MAC (45nm digital estimate)
- Also measure wall-clock time per step on CPU and GPU

**Agents:** MLP DQN, SpikingDQN, Double DQN
**Maze sizes:** 9, 13, 17, 21

**Step 1:** Instrument agents with operation counters.

**Step 2:** Run 1000 inference steps per config and average.

**Step 3:** Generate energy comparison table and figure.

**Step 4:** Commit.

---

### Task 8: Meta-Learning Experiment (Stretch)

**Files:**
- Create: `research/01_experiments/exp6_metalearning.py`

**Question:** Does the agent learn to learn new mazes faster over time?

**Protocol:**
- Train FeatureQ agent on 200 sequential mazes
- Measure steps-to-solve for each maze
- Plot learning curve: does the Nth maze get solved faster than the 1st?
- Compare against TabularQ (which resets each maze)

**Agents:** FeatureQ, TabularQ
**Maze size:** 9x9, 13x13
**Seeds:** 10

**Step 1:** Implement sequential training with per-maze solve metrics.

**Step 2:** Run and plot.

**Step 3:** Commit.

---

### Task 9: Figure Generation

**Files:**
- Modify: `research/04_paper/generate_figures.py`

**Figures to generate:**
1. Zero-shot test success by agent and maze size (bar chart, 6 sizes)
2. Generalization gap (test - train) by agent (bar chart)
3. Training vs test scatter plot (all agents, all sizes)
4. Box/violin plot across 20 seeds (9x9 and 13x13)
5. Scaling curve: success rate vs maze size (line chart)
6. Ablation study results (horizontal bar chart)
7. Continual learning: Phase A/B/C performance (grouped bar)
8. Forgetting metric comparison (bar chart)
9. Energy efficiency: SynOps/MACs comparison (bar chart)
10. Meta-learning curve: steps-to-solve vs maze number (line chart)
11. Hyperparameter sensitivity heatmaps (2x: SpikingDQN LR vs timesteps, FeatureQ LR vs init)

**All figures:** PDF + PNG, 300 DPI, publication style, consistent colors.

**Step 1:** Write figure generation script loading all experiment JSONs.

**Step 2:** Generate all 11 figures.

**Step 3:** Commit.

---

### Task 10: Statistical Analysis

**Files:**
- Create: `research/02_analysis/statistical_tests.py`

**Tests:**
- Mann-Whitney U test: FeatureQ vs each other agent (per maze size)
- Bootstrap confidence intervals (95%) for all success rates
- Effect size (Cohen's d) for key comparisons
- Bonferroni correction for multiple comparisons
- IQM (interquartile mean) following rliable methodology

**Output:** `research/02_analysis/stats_reports/stats_summary.json`

**Step 1:** Implement all statistical tests.

**Step 2:** Generate summary table.

**Step 3:** Commit.

---

### Task 11: Reproducibility Package

**Files:**
- Create: `research/01_experiments/requirements_experiments.txt`
- Create: `research/01_experiments/README_experiments.md`
- Create: `research/01_experiments/configs/` directory with all experiment configs as YAML

**Contents:**
- Exact Python/PyTorch/snnTorch versions
- Seed list
- All hyperparameters
- Instructions to reproduce each experiment
- Expected runtime per experiment
- Verification checksums for result files

**Step 1:** Create all reproducibility files.

**Step 2:** Verify a single experiment can be reproduced from the docs.

**Step 3:** Commit.

---

## Execution Summary

| Task | Compute | Needs H200? |
|------|---------|-------------|
| 1-2: Infrastructure + Agents | 0 | No |
| 3: Primary generalization | ~15 hours | Yes (2h on H200) |
| 4: Ablation | 2 min | No |
| 5: Continual learning | 100 min | No |
| 6: Hyperparameter sweep | 120 hours | Yes (10h on H200) |
| 7: Energy analysis | 10 min | No |
| 8: Meta-learning | 30 min | No |
| 9: Figures | 1 min | No |
| 10: Statistics | 1 min | No |
| 11: Reproducibility | 0 | No |

**Total H200 time needed: ~12 hours**
**Total local (RTX 5070 Ti) time: ~140 hours if no H200**

## Execution Order

1. Tasks 1-2 first (infrastructure, no compute)
2. Tasks 4, 5, 7, 8 next (run locally, under 3 hours total)
3. Ping user for H200 access
4. Tasks 3, 6 on H200 (12 hours)
5. Tasks 9, 10 after all data is in (minutes)
6. Task 11 last (packaging)
