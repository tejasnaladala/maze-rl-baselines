# Adversarial Review R7 (NeurIPS D&B reviewer simulation)

Conducted 2026-04-17 19:55 PDT against PAPER_SHORT.md v1.2 with the
modern-baseline sweep ~25% complete.

**Score: 5.5 / 10** for NeurIPS D&B Track.

**Verdict: SHIP AFTER MODERN BASELINES.**

## Three Strongest Attacks

### Attack 1: DQN-family selection is a strawman
The paper sweeps DQN, DoubleDQN, DRQN. All variants of the same off-policy
replay-buffer epsilon-greedy family. The selection conveniently excludes
PPO, A2C, SAC, and intrinsic-motivation variants. PPO preliminary number
(2.6%, line 183) is buried in Limitations. A reviewer will read this as
"authors found a worse PPO result and disclosed it minimally."

**Defeats it:** Show PPO with default HPs, not just one LR, also fails. The
70-run sweep currently in progress is the right experiment.

### Attack 2: Benchmark is trivially solvable by construction
A wall-follower achieves 100% only if the generator produces simply-
connected mazes (recursive backtracking does exactly this). The paper
never states this property explicitly. A reviewer can argue the 100%
result is evidence of a maze-friendly structure, not exploration hardness.

**Defeats it:** Replicate on loopy-maze generators (Aldous-Broder,
Wilson). If wall-follower fails there and RL still fails, the claim is
generator-independent.

### Attack 3: Distillation does not prove "the network class can represent the policy"
Supervised distillation provides dense correct gradient at every state.
This is not the same as the policy being reachable via reward-driven RL.
A network can represent a function as a global minimum of supervised loss
while that function sits in a basin unreachable from random initialization
under reward gradient.

**Defeats it:** Behavioral-cloning warm-start. Initialize RL from distilled
weights, fine-tune with DQN. If perf holds at ~97%, representation was
reachable. If it collapses to ~19%, reward landscape actively destroys the
representation (which is a stronger claim).

## Three Strongest Defenses

1. **Falsifiability is operationalized** (§2 line 53). Specific quantitative
   condition for refutation. Unusual for empirical ML.
2. **Reward shaping ablation closes the obvious confound** (Table 4, K4
   paired bootstrap, 200 runs, Holm-Bonferroni).
3. **Information asymmetry closed** (Abstract line 16-17). Same 24-d obs is
   sufficient for the heuristic to achieve 100%, so partial observability
   is ruled out.

## Top 2 Missing Things

1. **Complete modern baseline sweep (blocks submission).** PPO/DQN/A2C × 3
   LRs × 10 seeds, in progress. Wait for it.
2. **Simply-connected maze disclosure and loopy-maze experiment.** Add one
   sentence in §9 acknowledging the topological constraint. Add one
   supplemental experiment (n=5+ seeds) on a loopy generator.

## Status of fixes (autonomous run)

- [in progress] Modern baseline sweep, ~25% done at review time, ETA 4-5 hr
- [pending overnight] Add simply-connected maze disclosure to PAPER_SHORT,
  PAPER_PREVIEW, paper.md
- [pending overnight, contingent] If time permits and modern baselines
  finish early: run 5-seed loopy-maze pilot
- [pending overnight] BC warm-start experiment: too expensive for tonight,
  flag as "open follow-up for v1.1"
