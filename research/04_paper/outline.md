# Paper Outline: ICONS 2026

## Title Options
1. "Do Spiking Neural Networks Generalize Better? Cross-Environment Transfer in Procedural Reinforcement Learning"
2. "Spiking DQN for Zero-Shot Maze Navigation: Benchmarking Generalization Across Procedural Environments"
3. "Beyond Single-Environment Evaluation: Spiking Neural Networks for Cross-Environment Reinforcement Learning"

## Abstract (~150 words)
- Gap: Every spiking RL paper evaluates on fixed environments; no cross-environment generalization study exists
- Method: Spiking DQN with ego-centric features, trained on procedural mazes, tested zero-shot on unseen layouts
- Comparison: Spiking DQN vs MLP DQN vs Tabular Q-Learning, 5 seeds, 2 maze sizes
- Key finding: [AWAITING RESULTS]
- Energy: SynOps comparison showing theoretical efficiency advantage

## 1. Introduction (1 page)
- Spiking RL has matched ANNs on fixed environments (DSQN, PopSAN, ILC-SAN)
- But generalization across environments is untested
- This matters for robotics/edge deployment where environments change
- We ask: do SNNs generalize differently than ANNs?
- Contributions: (1) first cross-environment SNN RL benchmark, (2) ego-centric feature design, (3) comparison with equivalent ANN

## 2. Related Work (1 page)
- Spiking RL: DSQN, PopSAN, ILC-SAN, Proxy Target, CaRe-BN, Adaptive SG
- RL Generalization: Procgen, MiniGrid, zero-shot transfer survey
- SNN + Generalization: MTSpark (differentiate: multi-task vs cross-environment)
- State Abstraction: Li et al. 2006, ego-centric representations

## 3. Method (1.5 pages)
- Spiking DQN architecture (LIF + LI output, surrogate gradients)
- Ego-centric feature vector (3x3 map + goal direction + distance + action history)
- Procedural maze generation (recursive backtracking, variable size)
- Training protocol (DQN with replay, target network)
- Energy measurement (SynOps counting)

## 4. Experimental Setup (1 page)
- Maze sizes: 9x9, 13x13
- Train: 80 mazes, Test: 40 unseen mazes
- Seeds: 5
- Baselines: Tabular Q-Learning, Feature Q-Learning, MLP DQN
- Metrics: success rate, avg return, SynOps/step
- Statistical reporting: median + IQR, bootstrap CIs

## 5. Results (1.5 pages)
- Table 1: Zero-shot test success by agent and maze size
- Figure 1: Learning curves (train) and generalization gap
- Figure 2: Success rate vs maze size (scaling)
- Table 2: SynOps comparison
- Ablation: which features matter

## 6. Discussion (0.5 pages)
- What the results mean
- Limitations: small mazes, grid world only, no hardware measurements
- Why/why not spiking helps generalization

## 7. Conclusion (0.5 pages)
- Summary of findings
- Future: larger environments, continuous control, hardware deployment

## References (~30 citations)
