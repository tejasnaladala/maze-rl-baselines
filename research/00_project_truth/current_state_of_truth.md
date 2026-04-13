# Current State of Truth
> Generated: 2026-04-12 | Honest assessment of what we know

## VERIFIED (by our experiments)

1. **Spiking DQN with surrogate gradients learns on maze navigation.**
   - 55.3% success vs Q-Learning 41.3% on fixed 4x4 maze (proof.py, seed=42, 150 episodes)
   - Caveat: single seed, single maze size, small scale

2. **Three-factor STDP with eligibility traces compiles and passes unit tests.**
   - 17/17 Rust tests passing
   - Eligibility traces accumulate from spike pairs (test verified)
   - Reward drives weight changes via eligibility (test verified)
   - Neuromodulator signals update from reward + prediction error (test verified)

3. **Pure STDP (without surrogate gradients) fails to learn on RL tasks.**
   - 0% success on grid world (benchmark.py honest results)
   - Below random baseline on pattern classification
   - Confirmed by BindsNET creators: "We haven't managed to get SNNs to learn anything interesting in OpenAI gym environments"

4. **Feature-based Q-learning generalizes across random mazes better than position-based.**
   - Observed in dashboard (LiveAgent.tsx) over multiple runs
   - Caveat: not rigorously benchmarked with multiple seeds/statistical tests

5. **The Rust core runtime works end-to-end.**
   - 672 neurons, 6 modules, 10-step cognitive loop
   - WebSocket streaming at 30fps
   - WASM compilation to 87KB binary
   - Python bindings functional

## SUPPORTED BY LITERATURE (but not verified by us)

1. **Spiking DQN can beat standard DQN on Atari games.**
   - DSQN: 193.5% vs 142.8% median human-normalized across 17 games (Chen et al. 2022)
   - We have not reproduced this

2. **SNNs achieve 95-100x energy efficiency on neuromorphic hardware.**
   - NorthPole: 46.9x faster, 72.7x more efficient (published in Science)
   - Loihi: 95% energy savings vs Jetson TX2
   - We have no hardware measurements

3. **Continual learning with local plasticity rules resists catastrophic forgetting.**
   - CoLaNET: 92% accuracy across 10 sequential tasks
   - We have not demonstrated this with our system

4. **Non-spiking output neurons (membrane voltage as Q-value) are critical for spiking RL.**
   - Used by every competitive spiking RL system (DSQN, PopSAN, ILC-SAN, Proxy Target)
   - Our spiking_dqn.py implements this correctly

## PLAUSIBLE BUT UNPROVEN

1. **Spiking networks have an inherent generalization advantage over ANNs.**
   - No paper has tested this directly
   - Plausible due to temporal processing and adversarial robustness
   - This is our main paper claim -- MUST be tested rigorously

2. **Feature-based spiking RL enables zero-shot transfer across environment distributions.**
   - No paper combines feature-based representations with spiking DQN for transfer
   - Our dashboard demo suggests it works but lacks statistical rigor

3. **Dual-phase training (surrogate gradients + local STDP) provides best of both worlds.**
   - Phase 1 builds competent policy, Phase 2 adapts without catastrophic forgetting
   - Our proof.py Phase 2 showed only 5% success -- needs work

4. **The brain-inspired modular architecture produces useful emergent behaviors.**
   - Speculative. The 6-module system generates spikes but we can't prove the modularity helps vs a monolithic network

## LIKELY WRONG OR OVERSTATED

1. **"Engram beats Q-Learning" -- claimed in commit messages and LinkedIn draft.**
   - True only on fixed 4x4 maze with specific hyperparameters
   - On random mazes, tabular Q-learning currently beats us
   - Must be scoped carefully in any publication

2. **"No retraining needed" -- implied throughout README and strategy docs.**
   - Our system DOES retrain (surrogate gradients = backpropagation through spiking network)
   - The STDP-only mode doesn't learn effectively
   - Honest framing: "Online adaptation via local plasticity after initial training"

3. **"Brain-inspired" architecture is functionally meaningful.**
   - The 6 modules (sensory, associative, predictive, episodic, action, safety) have brain-inspired names but the actual computation is standard ML (Q-learning, replay buffers, etc.)
   - The biological analogy is marketing, not science
   - Paper should be careful about biological claims

4. **Energy efficiency claims without hardware measurements.**
   - We cite literature numbers (100x efficiency) but have zero measurements
   - Can only claim theoretical spike count advantage

## WHAT STILL LACKS EVIDENCE

1. Cross-environment generalization advantage of spiking vs non-spiking (THE paper claim)
2. Statistical significance of any benchmark result (need 5+ seeds, confidence intervals)
3. Energy efficiency on our system (need SynOps counting at minimum)
4. Scalability beyond 9x9 mazes
5. Phase 2 local adaptation actually working (current: 5% success)
6. Comparison against modern baselines (Double DQN, PPO, SAC)
7. Ablation studies (which components matter?)
8. Robustness to hyperparameter choices
