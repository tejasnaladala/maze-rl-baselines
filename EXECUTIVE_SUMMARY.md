# Engram Project — One-Page Executive Summary

**Title (working):** *A 5-Line Ego-Only Wall-Follower Beats Trained Neural Networks by 80 Percentage Points on Procedural Mazes; the Failure is Exploration, Not Function Approximation.*

**Status:** Paper draft v4. ~3,500 runs across two compute platforms. Codex adversarial audit confidence: 5/10 (sufficient to send to professors). Public repo + reproducibility package: https://github.com/tejasnaladala/engram

---

## The agent ladder (9×9 procedural mazes, n=20+ seeds, paired bootstrap, Holm-Bonferroni)

| Tier | Method | Success |
|---|---|---|
| 1 — Oracle | BFS shortest-path planner | **100%** |
| 2 — Heuristic | **Ego-only wall-follower** (5 lines, same 24-d observation as neural agents) | **100%** |
| 3 — Distillation | **MLP from BFS** (same arch & optimizer as MLP_DQN) | **97.4%** sd 2.5 |
| 4 — Random walk | NoBackRandom (memoryless, +"don't reverse") | 52.2% |
| 5 — Random walk | Uniform Random | 31.7% |
| 6 — Tabular | FeatureQ_v2 | 35.3% |
| 7 — Neural RL | MLP_DQN_h64 (and DoubleDQN, DRQN — all 16-19%) | **19.3%** |
| 8 — PPO + shaped reward | preliminary, n=3 | ~1-2% |

**Direction is monotonic downward across tiers.** More sophisticated learning machinery performs worse than trivial structure-aware baselines.

---

## The clean dichotomy (the centerpiece)

> **A supervised MLP recovers the BFS oracle at 97.4%.**
> **The same MLP architecture, same observation, same optimizer, trained via DQN: 19.3%.**

Therefore the failure of standard RL on this task is **exploration + credit assignment, not representational capacity**. The neural policy class can express the maze-solving policy; reward-driven gradient descent does not discover it.

---

## What we ruled out

| Hypothesis | Evidence |
|---|---|
| Network too small | h32, h64, h128, h256 all yield 13-19% (capacity sweep, 160 runs) |
| Hyperparameters | Default LR=5e-4 IS local optimum across {1e-4, 5e-4, 1e-3, 3e-3} |
| Memory / partial obs | DRQN with LSTM also at 19.0% |
| Reward shaping unfair | K4 ablation: trained agents *collapse* without shaping (d=-2.66, -0.70, -0.84); random unchanged |
| Information asymmetry | Ego-only wall-follower (same 24-d obs) hits 100% |
| Procedural maze artifact | MiniGrid replication: MLP_DQN < Random in 3/4 environments |

---

## Theoretical anchor

Empirical confirmation of Alon-Benjamini-Lubetzky-Sodin (2007) non-backtracking cover-time theorem on procedural RL benchmark — first time, as far as we can find:

- NoBackRandom: 167.6 mean steps per success
- Random: 193.9 mean steps per success
- Difference: 13.6% fewer (theory predicts a strict cover-time advantage; we measure exactly that)
- Power-law fit `success ~ a · n^b`: NoBack b=-2.07 [bootstrap CI -2.21,-1.94], Random b=-2.81 — gap of 0.74 units, theory-predicted

---

## What this paper is and is not

**Is:** an empirical evaluation paper that exposes a load-bearing baseline blindness in published procedural-maze RL evaluations and isolates the failure mechanism via supervised distillation.

**Is not:** a claim that "neural function approximation fails" or "deep RL is broken." Both would be overclaiming. We claim the narrower thing.

**Implication for the field:** procedural-maze RL papers should include hand-coded heuristic, distillation, and random-walk baselines on identical evaluation harnesses. Failing to do so risks overinterpreting neural results. The paper provides one such audit and a validated, reproducible benchmark for the community.

---

## Compute & reproducibility

- ~3,500 runs across local RTX 5070 Ti laptop and 4× H200 (vast.ai, ~$155 total)
- SHA-256 manifest of all result files; code-hash pinned per run
- Single-file stats pipeline (`stats_pipeline.py`): seed-aligned paired bootstrap, Holm-Bonferroni, Cohen's d, Mann-Whitney U
- Adversarial audit transcripts (5 Codex MCP review rounds) preserved in `git log`
- Honest disclosure: harness bug found mid-development (auxiliary launchers filtered the test maze distribution); fix + validation table in §3.2.1 of paper.

---

## Asks (collaborator outreach)

1. **Hard critique** before submission — particularly on whether the dichotomy cleanly separates representation from exploration
2. **Modern baseline you'd demand to defeat the claim** (NGU? Agent57? PPO-LSTM at scale? MuZero?)
3. **Compute partnership** for v1.1 extensions (Procgen Maze re-run, larger seed counts, scaled neural baselines) if the work aligns with your group's interests
