# Experiment Plan for ICONS 2026 Paper
> "Do Spiking Neural Networks Generalize Better Across Procedural Environments?"

## Critical Question
Does a spiking DQN with ego-centric features outperform an equivalent MLP DQN
on zero-shot transfer to unseen procedurally generated mazes?

## Experiments (Priority Order)

### EXP1: Spiking DQN vs MLP DQN vs Tabular Q (THE critical experiment)
- Train on 100 random mazes, test on 100 held-out mazes
- 5 seeds each
- Maze sizes: 9x9, 13x13, 17x17
- Metrics: success rate, avg return, sample efficiency, SynOps

### EXP2: Ablation -- which components matter?
- Full model vs no-action-history vs no-visit-penalty vs no-distance-shaping
- 5 seeds, 9x9 mazes

### EXP3: Scaling -- does the advantage grow with maze complexity?
- Compare generalization gap across 9x9, 13x13, 17x17, 21x21
- 3 seeds each

### EXP4: Energy efficiency
- Count SynOps per decision for spiking vs MLP
- Report theoretical energy ratio

### EXP5: Continual learning
- Train on distribution A (dense mazes), then B (sparse mazes)
- Test recall on A after B training
- 5 seeds
