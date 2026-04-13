# Research Risk Register
> Generated: 2026-04-12

## RISK 1: NOVELTY
**Severity: HIGH**
**Status: PARTIALLY MITIGATED**

Feature-based Q-learning that transfers across environments is textbook (Li et al. 2006, state abstraction). The egocentric vs allocentric distinction is well-established. A reviewer will say: "This is a straightforward application of state abstraction."

**What IS novel:** No paper combines spiking Q-learning with procedurally generated environments for generalization testing. The SNN + cross-environment transfer intersection is genuinely unexplored.

**Mitigation:** Frame the paper as "Do spiking networks generalize better than ANNs across procedural environments?" -- a question nobody has answered. The feature engineering is the method, not the contribution.

---

## RISK 2: BENCHMARK RIGOR
**Severity: CRITICAL**
**Status: UNMITIGATED**

Current results: single seed, single maze size, no confidence intervals, no statistical tests, limited baselines. No reviewer will accept this.

**Required:** 5+ seeds, confidence intervals, significance tests (Mann-Whitney U or bootstrap), multiple maze sizes (5x5, 9x9, 15x15, 21x21), comparison against DQN + Double DQN + PPO baselines on same environments.

**Mitigation:** Design and run the full experiment suite before submission.

---

## RISK 3: BASELINE STRENGTH
**Severity: HIGH**
**Status: PARTIALLY MITIGATED**

Current baselines: Random + Tabular Q-Learning. These are trivial. A reviewer will ask: "Why not compare against DQN, Double DQN, PPO? Why not compare against a small MLP with the same features?"

**Required baselines:**
- Random (trivial)
- Tabular Q-Learning (position-based)
- Feature Q-Learning (same features, no spiking)
- Small MLP DQN (2-layer, same features)
- Spiking DQN (our method)
- Ablated versions of our method

**Mitigation:** The critical comparison is Spiking DQN vs MLP DQN on the SAME features and SAME environments. If spiking wins, the contribution is clear. If MLP wins, we need to pivot to energy efficiency as the argument.

---

## RISK 4: OVERCLAIM
**Severity: HIGH**
**Status: PARTIALLY MITIGATED**

Multiple places in README, strategy docs, and LinkedIn post make claims stronger than evidence supports:
- "beats Q-Learning" (true only on specific setup)
- "no retraining needed" (uses surrogate gradient training = backprop)
- "brain-inspired" (marketing, not science)

**Mitigation:** Every claim in the paper must be tagged with evidence level. Downgrade any claim lacking statistical support.

---

## RISK 5: STATISTICAL POWER
**Severity: HIGH**
**Status: UNMITIGATED**

With 1 seed and ~20 episodes per evaluation window, our results have enormous variance. The 55.3% vs 41.3% difference could easily be noise.

**Required:** Minimum 5 seeds, preferably 10. Use rliable library for interval estimates. Report IQM (interquartile mean) not just mean. Apply correction for multiple comparisons.

**Mitigation:** Run all experiments with 10 seeds. Report median + IQR. Use bootstrap confidence intervals.

---

## RISK 6: DATASET / ENVIRONMENT LEAKAGE
**Severity: LOW**
**Status: MITIGATED**

Procedurally generated mazes with different seeds per episode. No pre-defined test set that could leak into training. Risk is low because environments are generated on-the-fly.

**Remaining risk:** If the feature space is too small, the agent might "memorize" all possible feature combinations rather than truly generalizing.

**Mitigation:** Use larger mazes (15x15, 21x21) where feature space is too large for memorization.

---

## RISK 7: IMPLEMENTATION BUGS
**Severity: MEDIUM**
**Status: PARTIALLY MITIGATED**

Previous bugs found:
- Unreachable goal in even-sized mazes (GRID=8, fixed)
- Action selector always returning action 0 (fixed with epsilon-greedy)
- Fake metrics from unsolvable mazes (fixed with BFS solvability check)

**Remaining risk:** Subtle bugs in surrogate gradient computation, STDP update, or Q-value calculation could inflate or deflate results.

**Mitigation:** Verify against known implementations (snnTorch reference, DQN reference). Run sanity checks. Compare spiking DQN against pure DQN on the same task -- if spiking DQN is dramatically better, something is wrong.

---

## RISK 8: METRIC MISMATCH
**Severity: MEDIUM**
**Status: UNMITIGATED**

Success rate (% episodes reaching goal) doesn't capture efficiency, path quality, or learning speed. A paper needs:
- Success rate
- Average return (reward)
- Sample efficiency (episodes to reach X% success)
- Path efficiency (steps vs optimal path length)
- Energy efficiency (SynOps or spike count per decision)
- Forgetting metric (performance on old task after learning new task)

**Mitigation:** Define comprehensive metric suite before running experiments.

---

## RISK 9: REPRODUCIBILITY
**Severity: MEDIUM**
**Status: PARTIALLY MITIGATED**

Fixed seeds in proof.py. But: snnTorch uses PyTorch which has non-deterministic operations on GPU. Different PyTorch versions may give different results.

**Mitigation:** Lock PyTorch version in requirements. Use CPU-only for reproducibility. Document exact environment. Provide Docker container or requirements.txt.

---

## RISK 10: REVIEWER ATTACK SURFACE
**Severity: HIGH (aggregate)**

Likely reviewer objections:
1. "Just state abstraction, nothing new" -- counter with SNN angle
2. "Small mazes, doesn't scale" -- need 15x15+ results
3. "No comparison to DQN/PPO" -- must add
4. "Feature engineering doesn't generalize to new task types" -- acknowledge as limitation
5. "No hardware results, energy claims are theoretical" -- use SynOps
6. "Phase 2 adaptation doesn't work (5% success)" -- either fix or remove from claims
7. "Single seed results" -- must fix with 10 seeds
8. "Why not just use a small MLP?" -- the critical question, must answer with data
