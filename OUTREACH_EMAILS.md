# Outreach Emails — Tier 1 Researchers

Cold email drafts. **Lead with the scientific tension** (per Codex Round 5).
Bury the harness audit story unless they open the PDF.

Each email: ≤250 words, plain text, personalised first paragraph, headline
ladder, link to GitHub, clear ask.

---

## Common hook (use across emails, lightly varied)

```
We found a procedural maze benchmark where a five-line egocentric
wall-following heuristic solves 100% of unseen test instances and a
BFS-distilled MLP reaches 97.4%, yet standard DQN/DoubleDQN/DRQN with
the same observation class and architecture stay around 17-19% — below
even uniform random (32%) and a no-backtracking random walk (52%).

Same architecture, same observation, same optimizer. The neural policy
class can clearly *express* the maze-solving policy (we prove this via
supervised distillation); standard reward-driven RL just doesn't
*discover* it. We isolate this with a 5-tier agent ladder, capacity
sweep (h32-h256, all flat at ~13-19%), LR sweep (default is local
optimum), reward ablation (K4: trained agents collapse without
shaping; random unchanged), and a harness audit after we discovered a
filtered-evaluation bug in early auxiliary launchers.

~3,500 runs, 20+ seeds per cell, paired bootstrap with
Holm-Bonferroni, SHA-256 manifest, code-hash-pinned reproducibility.

Paper draft: <ARXIV_LINK_TBD>
Code + data: https://github.com/tejasnaladala/engram

I'd value a hard critique before submission — particularly on whether
this cleanly separates representation from exploration, and what
modern baseline (NGU-style episodic novelty? PPO-LSTM at scale?) you'd
demand to defeat the claim.

Thanks for your time.
[Your name]
```

---

## Per-researcher personalization

### 1. Peter Henderson (Princeton)
*Hook:* "Your 2018 'Deep RL That Matters' is the methodological model we built around."

Subject: `MLP at 97% via BFS distillation, 19% via DQN — same architecture; would value your reproducibility lens`

Personal opening:
> Dear Prof. Henderson,
> 
> Your *Deep RL That Matters* (AAAI 2018) is the methodological model
> we built this work around — 20 seeds per cell, paired bootstrap,
> Holm-Bonferroni, Cohen's d, code-hash-pinned reproducibility,
> SHA-256 manifest of all raw results. After an adversarial review caught
> an evaluation harness bug we'd missed, we wrote the validation table
> and audit trail directly into the methods section. I'd be especially
> grateful for your reproducibility lens before we submit.
> 
> [headline ladder block]
> 
> Best,

---

### 2. Rishabh Agarwal (GDM / Mila)
*Hook:* "Direct application of your Statistical Precipice prescription."

Subject: `Procedural maze evaluation result — your Statistical Precipice protocol applied`

Personal opening:
> Dear Rishabh,
> 
> Your *Deep RL at the Edge of the Statistical Precipice* (NeurIPS 2021)
> shaped our analysis pipeline: stratified bootstrap, IQM, family-wise
> error correction, performance profiles. The headline result we
> arrived at would not be reportable under a less rigorous protocol,
> which is part of why I think it matters.
> 
> [headline ladder block]
> 
> The 78pp gap between MLP-distilled-from-BFS (97.4%) and MLP-DQN
> (19.3%) — same architecture, same observation, same optimizer — is
> the cleanest representation-vs-exploration dichotomy I've been able
> to construct. Whether the paper holds up under your statistical
> standards is the question I most want answered before submission.
> 
> Best,

---

### 3. Marc Bellemare (Mila / McGill / DeepMind)
*Hook:* "Random baselines as you championed in ALE; cover-time theorem confirmed empirically."

Subject: `Non-backtracking random walk as RL baseline — Alon-Benjamini-Lubetzky-Sodin 2007 confirmed empirically`

Personal opening:
> Dear Prof. Bellemare,
> 
> The Arcade Learning Environment paper convinced the field that the
> random-policy baseline is a load-bearing reference; this work tries to
> sharpen that for procedural mazes. We empirically confirm the
> Alon-Benjamini-Lubetzky-Sodin (2007) non-backtracking cover-time
> theorem on a procedural RL benchmark — NoBackRandom takes 13.6%
> fewer steps per success than uniform random, exactly the predicted
> advantage. A formal power-law fit (success ~ a · n^b) gives
> NoBackRandom b = -2.07 [-2.21, -1.94] vs Random b = -2.81, the same
> 0.74-unit gap.
> 
> [headline ladder block]
> 
> Would value your view on whether the cover-time framing is the right
> theoretical anchor for the empirical result.
> 
> Best,

---

### 4. Joelle Pineau (Meta / McGill)
*Hook:* "Reproducibility checklist; SHA-256 manifest of all runs."

Subject: `Reproducibility-first procedural RL evaluation — SHA-256 pinned, harness audit included`

Personal opening:
> Dear Prof. Pineau,
> 
> Your reproducibility work changed how I thought about RL evaluation,
> so I tried to build this with that lens from day one. Every run is
> SHA-256-hashed in a manifest, every code change pins a code_hash on
> result records, and we ran a third-party adversarial audit (Codex
> MCP) across 5 review rounds during development. The audit caught a
> harness filtering bug we'd missed; the corrected validation table is
> in §3.2.1.
> 
> [headline ladder block]
> 
> The reproducibility package, audit transcripts, and complete raw data
> are at https://github.com/tejasnaladala/engram.
> 
> Best,

---

### 5. Pablo Castro (Google DeepMind)
*Hook:* "Dopamine baseline rigour; random baseline as load-bearing."

Subject: `Compact procedural maze benchmark with surprising RL failure mode + 5-tier baseline ladder`

Personal opening:
> Dear Pablo,
> 
> The Dopamine philosophy — careful, comparable baselines first — is the
> spirit we tried to extend in this benchmark. We construct a 5-tier
> agent ladder on a procedural maze family and find a clean inversion of
> the assumed RL progress story:
> 
> [headline ladder block]
> 
> The most surprising row is policy distillation: a vanilla MLP trained
> on BFS-oracle action labels with the *same 24-d ego-feature
> observation as MLP-DQN* recovers the policy at 97.4% test success
> (n=20). The same architecture trained via DQN reaches 19.3%. This
> isolates the failure to RL discovery, not function approximation.
> 
> Would value Dopamine-team feedback on whether this benchmark is worth
> being added to a standard suite for sample-efficient agent eval.
> 
> Best,
