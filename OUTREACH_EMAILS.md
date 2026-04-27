# Outreach Dossier (post Codex R8 audit)

**Cut from 25 to 13 researchers** per Codex MCP audit recommendation:
spray-and-pray to 25 senior PIs has uneven topic fit and visible templating.
Tighter list with sharper per-recipient hooks.

**Tone fixes applied throughout** (per audit):
- "rules out the standard explanations" → "is not explained by the obvious candidates we tested"
- "reward gradient actively destroys" → "the BC initialization does not survive standard DQN fine-tuning in our setup"
- "first empirical confirmation of ABLS 2007" → "consistent with the ABLS 2007 prediction"
- "wall-follower is provably perfect" → "provably complete (terminates at goal on every tree maze, though not at shortest-path-optimal step count)"

**Strategic shifts**:
- Switched the cold ask from "15 minutes of your time" to a low-cost async ask: "if one claim looks wrong on a 2-minute skim, I would value a one-line reply saying which one"
- 15-minute synchronous ask reserved for UW-local researchers only (cheaper for them since no travel)
- Send order tighter: Tier 1 first to maximize signal in Tier 2/3 references

**Attach**: `SEED_PAPER.pdf` (v0.4, 7 pages, plain Cambria) for every send
**Repo**: `https://github.com/tejasnaladala/maze-rl-baselines`

---

## Send order (10 days, 13 researchers)

| Day | Tier | Researchers | Ask type |
|---|---|---|---|
| 1 | A | Pulkit Agrawal, Aviral Kumar, Roberta Raileanu | Async one-line skim reply |
| 3 | B | Aravind Rajeswaran, Pablo Castro, Peter Henderson | Async one-line skim reply |
| 5 | C | Rishabh Agarwal, Marc Bellemare, Tim Rocktäschel | Async one-line skim reply |
| 7 | D (UW local) | Natasha Jaques, Abhishek Gupta | 30 min in person at Allen |
| 10 | E (stretch) | Sergey Levine, Sham Kakade | Async one-line skim reply |

---

## TIER A: Direct technical fit, send Day 1

### 1. Pulkit Agrawal (MIT Improbable AI): `pulkitag@mit.edu`

```
Subject: Possible inverse of "RL's Razor": online RL erases a BC pretrain on mazes

Dear Prof. Agrawal,

Your "RL's Razor: Why Online RL Forgets Less" (ICLR 2025) is the framing I
most want to test against a finding that looks like the inverse phenomenon
on a small benchmark. On 9x9 procedural mazes (DFS-generated tree graphs,
where a wall-following heuristic is provably complete; we are honest that
this is the most important scope limit), behavior cloning from a BFS
oracle reaches 97.4 percent test success with a 24-64-32-4 MLP.
Initializing DQN online and target networks from those weights and
fine-tuning with the same shaped reward used by the from-scratch DQN run
collapses test success from 97.2 to 13.6 percent across all 5 seeds
(per-seed: 0, 12, 16, 18, 22). The fine-tuned policy ends below
from-scratch DQN at any tested LR.

Best HP-tuned modern reward-driven baseline (SB3 PPO/DQN/A2C across 3 LRs
each, 70 runs total) ties uniform Random at 31 percent. Whether this
generalizes beyond DFS tree mazes and beyond one fine-tune recipe is the
open question.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim in the
seed paper looks wrong on a 2-minute skim, I would value a one-line reply
saying which one.

Seed paper attached. Code, configurations, seeds:
github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. Improbable AI's curiosity-driven exploration line was on my reading
list before I knew the lab name; you have shaped how I think about
discovery.
```

### 2. Aviral Kumar (CMU AIRe): `aviralku@andrew.cmu.edu`

```
Subject: BC -> DQN warm-start collapses 97 to 14 on tree mazes; CQL/IQL diagnostic?

Dear Prof. Kumar,

Your CQL paper is the diagnostic I most want to run on a finding I cannot
explain. On 9x9 DFS procedural mazes (tree topology, where a 5-line
wall-follower is provably complete; this is the main scope caveat),
initializing DQN online and target networks from a 97.2 percent
behavior-cloning policy and fine-tuning with the same shaped reward as the
from-scratch DQN run collapses test success to 13.6 percent across all 5
seeds (per-seed: 0, 12, 16, 18, 22). Post-fine-tune ends below from-scratch
DQN at any tested LR. Best HP-tuned modern reward-driven baseline ties
uniform Random at 31 percent on the same harness.

The clean diagnostic is whether CQL or IQL fine-tune preserves the BC
basin. If yes, bootstrap instability is the mechanism. If no, the
mechanism is something else (distribution shift in replay, reward shape,
or something I am not seeing). I have the harness, the BC weights, all 5
seeds, and the trajectories ready to share.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which
one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The CQL paper's "value-overestimation under bootstrap" framing is
the lens I keep coming back to.
```

### 3. Roberta Raileanu (Meta AI): `rraileanu@meta.com`

```
Subject: BC -> DQN warm-start collapse on procedural mazes, MiniGrid-adjacent

Dear Roberta,

Your work on procedurally generated environments and exploration in
MiniGrid is the closest precedent for the result I am sitting on. On 9x9
DFS procedural mazes (tree topology, scope caveat: a wall-following
heuristic is provably complete here), the controlled finding is that
behavior cloning from a BFS oracle reaches 97.4 percent with a 24-64-32-4
MLP, best HP-tuned modern reward-driven baseline ties uniform Random at
31 percent, and BC -> DQN warm-start collapses to 13.6 percent across all
5 seeds. MiniGrid replication on 4 envs (DoorKey, FourRooms, MultiRoom,
Unlock) shows MLP-DQN at or below Random in 3 of 4, consistent with the
procgen pattern your group has documented at much larger scale.

This is a small benchmark (about 50 GPU-hours total) but the BC -> DQN
collapse is a controlled diagnostic for what may be a broader pattern.
The natural follow-up is to extend the BC -> DQN protocol to NetHack-scale
procedural environments, which I do not have the compute for.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. Your "Decoupling Exploration and Exploitation" paper is the cleanest
treatment of the distinction I have read.
```

---

## TIER B: Strategic technical + reproducibility fit, send Day 3

### 4. Aravind Rajeswaran (Microsoft AI Frontiers): `aravraj@microsoft.com`

```
Subject: UW alum: BC -> DQN warm-start collapses 97 to 14 (DAPG-shaped result)

Dear Aravind,

I am a UW undergraduate (ECE + Applied Math, '28) and your PhD work with
Sham and Emo, DAPG in particular, is the closest precedent for what I am
sitting on. On 9x9 DFS procedural mazes (tree topology, scope caveat:
wall-follower is provably complete here), behavior cloning from a BFS
oracle reaches 97.4 percent with a 24-64-32-4 MLP. Initializing MLP_DQN
online and target networks from those weights and fine-tuning with the
same shaped reward collapses test success to 13.6 percent across all 5
seeds (per-seed: 0, 12, 16, 18, 22). Post-fine-tune ends below
from-scratch DQN at any tested LR (best HP-tuned modern baseline ties
uniform Random at 31 percent).

DAPG showed demonstrations resolve exploration-hard tasks where pure RL
fails; this benchmark is a tiny inverse, where the fine-tune procedure
appears to undo the demonstration prior. Whether this generalizes beyond
tree mazes and beyond one fine-tune recipe is open.

If one claim looks wrong on a 2-minute skim, I would value a one-line
reply saying which one. Equally, if your group has a 2026 summer or fall
opening for someone with empirical execution capacity who can ship clean
ablations end-to-end, I would be glad to talk.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. Sham was on my reading list before I knew he was at UW.
```

### 5. Pablo Castro (Google DeepMind): `psc@google.com`

```
Subject: Compact procedural maze benchmark with audited 5-tier baseline ladder

Dear Pablo,

The Dopamine philosophy of careful, comparable baselines first is the
spirit I tried to extend with this 5-tier ladder. The most surprising row
is policy distillation: a vanilla MLP trained on BFS oracle action labels
with the same 24-d ego-feature observation as MLP_DQN reaches 97.4
percent test success (n=20). The same architecture trained via DQN
reaches 19.3 percent (custom) or 31.4 percent (SB3 default LR, the latter
ties uniform Random at 32.7 percent on the same harness).

The sharper finding: initializing DQN at the BC weights and fine-tuning
with the same shaped reward collapses test success to 13.6 percent across
all 5 seeds. Scope caveat: DFS mazes are tree graphs, where a wall-
follower is provably complete; whether the BC -> DQN collapse holds on
loopy generators and beyond one fine-tune recipe is open.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which
one. I would also value Dopamine-team feedback on whether this benchmark
is worth being added to a standard suite for sample-efficient agent eval.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The Dopamine README is still the cleanest baseline reference I know.
```

### 6. Peter Henderson (Princeton): `peter.henderson@princeton.edu`

```
Subject: Reproducibility-first procedural RL benchmark, harness audit in §3.2.1

Dear Prof. Henderson,

Your "Deep RL That Matters" (AAAI 2018) is the methodological model I
built this work around: 20+ seeds per cell, paired bootstrap with Holm-
Bonferroni, Cohen d, code-hash-pinned reproducibility, SHA-256 manifest
of all 4,200+ archived result records. After an adversarial review caught
a harness-filtering bug I had missed (an `is_solvable(avoid_hazards=True)`
filter that quietly inflated baselines by 12 to 22 pp), I wrote the
validation table directly into §3.2.1 of the paper and re-ran the
affected experiments on the corrected harness. The headline distillation
and DQN numbers survived the fix. Confounded experiment directories are
quarantined under `_CONFOUNDED` suffixes for audit purposes.

Headline finding: behavior-cloning MLP reaches 97.4 percent, best HP-tuned
modern reward-driven baseline ties uniform Random at 31 percent, BC -> DQN
warm-start collapses to 13.6 percent across all 5 seeds. Scope caveat:
DFS tree mazes only; broader generators are queued.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which one.

Seed paper attached. Code, raw data, manifest:
github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. "Deep RL That Matters" is the paper I most often cite when explaining
to people why I run 20 seeds.
```

---

## TIER C: Theory + procedural-RL fit, send Day 5

### 7. Rishabh Agarwal (GDM / Mila): `rishabhagarwal@google.com`

```
Subject: Statistical Precipice protocol applied to a small RL failure mode

Dear Rishabh,

Your "Deep RL at the Edge of the Statistical Precipice" (NeurIPS 2021)
shaped the analysis pipeline here: paired bootstrap stratified by seed,
IQM where applicable, family-wise error correction, performance profiles.
On 9x9 DFS procedural mazes (tree topology, scope caveat: wall-follower
provably complete here), behavior-cloning MLP reaches 97.4 percent, best
of 7 HP-tuned modern reward-driven configurations (SB3 PPO/DQN/A2C across
3 LRs each, 70 runs total) ties uniform Random at 31 percent, BC -> DQN
warm-start collapses to 13.6 percent across all 5 seeds.

The Bayesian dominance probabilities for every headline pairwise
comparison exceed 0.999 under Beta-Binomial conjugate priors with uniform
prior. The 5-seed BC -> DQN result is the only number flagged as
undersized; an n equal or greater than 20 extension is queued.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The "rliable" library has been in my dependencies since the second
week of this project.
```

### 8. Marc Bellemare (Mila / DeepMind): `bellemare@google.com`

```
Subject: ABLS 2007 cover-time consistent on procedural mazes + BC warm-start collapse

Dear Prof. Bellemare,

The ALE paper convinced the field that the random-policy baseline is
load-bearing; this work tries to sharpen that for procedural mazes. The
empirical NoBackRandom and uniform Random scaling exponents differ by
0.84 units (95 percent bootstrap CI), consistent with the Alon, Benjamini,
Lubetzky and Sodin (2007) non-backtracking cover-time prediction on this
benchmark.

Sharper finding: behavior-cloning MLP reaches 97.4 percent on 9x9 DFS
mazes, best HP-tuned modern reward-driven baseline ties uniform Random at
31 percent, BC -> DQN warm-start collapses to 13.6 percent across all 5
seeds. Scope caveat: DFS mazes are tree graphs, where a wall-follower is
provably complete; whether the cover-time framing is the right anchor for
the BC -> DQN collapse on loopy generators is open.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The "distributional RL" line is on my queue if the cover-time
framing turns out to be insufficient.
```

### 9. Tim Rocktäschel (Meta AI / UCL): `rockt@meta.com`

```
Subject: Procedural maze BC -> DQN collapse, MiniHack-adjacent at much smaller scale

Dear Prof. Rocktäschel,

Your NLE and MiniHack work made procedural exploration a first-class
topic for me. This benchmark is much smaller (9x9 hazard mazes, about
50 GPU-hours total compute, full reproducibility on a laptop GPU), but it
isolates a specific failure mode at a controllable scale. Behavior-cloning
MLP reaches 97.4 percent, best HP-tuned modern reward-driven baseline
ties uniform Random at 31 percent, BC -> DQN warm-start collapses to
13.6 percent across all 5 seeds. MiniGrid replication: 3 of 4 envs show
MLP-DQN at or below Random.

Scope caveat: DFS tree mazes, where a wall-follower is provably complete;
the BC -> DQN collapse may or may not survive on more complex topology.
Scaling the BC -> DQN protocol to MiniHack is the kind of test I do not
have the compute for.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The MiniHack design choices around procedural difficulty calibration
are the toolkit I would want for a v1.1 cyclic-maze extension.
```

---

## TIER D: UW local, send Day 7 (in-person ask is cheap for them)

### 10. Natasha Jaques (UW Allen, Social RL Lab): `nj@cs.washington.edu`

```
Subject: UW Social RL Lab: BC -> DQN warm-start collapse on procedural mazes

Dear Prof. Jaques,

Your Social RL work is part of why I started taking exploration as a
first-class object. On 9x9 DFS procedural mazes (tree topology, scope
caveat: wall-follower provably complete here), behavior-cloning MLP
reaches 97.4 percent, best HP-tuned modern reward-driven baseline ties
uniform Random at 31 percent, BC -> DQN warm-start collapses to 13.6
percent across all 5 seeds. MiniGrid replication: 3 of 4 envs show
MLP-DQN at or below Random.

I am a UW undergraduate (ECE + Applied Math, '28) and your group is the
most natural intellectual home for this kind of question on this campus.
I would value 30 minutes in person at Allen, whichever week is convenient
in May or June. I have the harness, the BC weights, all 5 fine-tune
seeds, and the trajectories ready to walk through. Specific question I
most want your view on: is the BC -> DQN collapse a controlled instance
of what the Social RL framing would predict for misaligned reward, or is
it a different mechanism.

If your group has a 2026 summer or fall opening for an undergraduate
research assistant who can ship empirical work end-to-end, I would also
be glad to talk about that.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. Caught the Social RL Lab launch announcement earlier this year,
congrats on the new group.
```

### 11. Abhishek Gupta (UW Allen, WEIRD Lab): `abhgupta@cs.washington.edu`

```
Subject: WEIRD Lab: BC -> DQN warm-start collapse 97 to 14 (UW-local, in person)

Dear Prof. Gupta,

Congratulations on the RAS Early Academic Career Award, well-earned. I am
a UW undergraduate (ECE + Applied Math, '28) and your robotics RL work is
one of the closest precedents on this campus for the result I am sitting
on. On 9x9 DFS procedural mazes (tree topology, scope caveat: wall-
follower is provably complete here), behavior cloning from a BFS oracle
reaches 97.4 percent with a 24-64-32-4 MLP. Initializing MLP_DQN online
and target networks from those weights and fine-tuning with the same
shaped reward as the from-scratch baseline collapses test success to
13.6 percent across all 5 seeds. Best HP-tuned modern reward-driven
baseline ties uniform Random at 31 percent.

The structural pattern (distillation works, reward-driven RL does not,
and the standard fine-tune recipe undoes the distillation prior) is the
analogue to manipulation tasks where demonstration-augmented learning
works but standard RL fine-tune erases the demo.

I would value 30 minutes in person at WEIRD Lab, whichever week works in
May or June. Equally, if your group has a 2026 summer or fall undergraduate
research opening, I would be glad to talk.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The "WEIRD" acronym alone deserved an award.
```

---

## TIER E: Stretch, send Day 10 only after package response signal

### 12. Sergey Levine (Berkeley RAIL): `svlevine@eecs.berkeley.edu`

```
Subject: Procedural maze: BC at 97% on same arch DQN solves at 19%, BC->DQN ends at 14%

Dear Prof. Levine,

I am a UW undergraduate (ECE + Applied Math, '28) reaching out with a
small empirical result. On 9x9 DFS procedural mazes (tree topology,
where a wall-following heuristic is provably complete; this is the most
important scope caveat), initializing DQN online and target networks from
a 97.2 percent behavior-cloning MLP and fine-tuning with the same shaped
reward as a from-scratch DQN run collapses test success to 13.6 percent
across all 5 seeds (per-seed: 0, 12, 16, 18, 22). Post-fine-tune ends
below from-scratch DQN at any tested LR. Best of seven HP-tuned modern
reward-driven configurations ties uniform Random at 31 percent.

RAIL has built most of the methods I would reach for to defeat this:
offline-RL bootstrap control (CQL, IQL), demonstration-augmented
exploration, distribution-aware fine-tuning. I am being careful not to
overclaim from n=5 and one fine-tune recipe; whether the basin
instability is robust under these alternatives is the most informative
follow-up.

If one claim looks wrong on a 2-minute skim, I would value a one-line
reply saying which one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. DAPG was on my reading list when I built the BC pipeline. The
pattern here looks like its inverse.
```

### 13. Sham Kakade (Harvard Kempner): `sham@seas.harvard.edu`

```
Subject: UW-adjacent: small RL benchmark, basin-instability under DQN fine-tune

Dear Prof. Kakade,

Your Statistical RL course materials at UW and the rltheorybook are part
of why I framed this result as a representation-versus-discovery question.
On 9x9 DFS procedural mazes (tree topology, scope caveat: wall-follower
is provably complete here), a behavior-cloning MLP reaches 97.4 percent
with a 24-64-32-4 architecture. Initializing DQN online and target
networks from those weights and fine-tuning with the same shaped reward
collapses test success to 13.6 percent across all 5 seeds. Post-fine-tune
ends below from-scratch DQN at any tested LR. Best HP-tuned modern
reward-driven baseline ties uniform Random at 31 percent.

The controlled framing: the high-performing minimum is in the function-
class reachable set (BC reaches it via supervised gradient) and is
approximately a fixed point of the policy itself at fine-tune step 0
(97.2 percent measured). The DQN bootstrap operator with epsilon-greedy
and shaped reward moves the iterate away from that fixed point. The
empirical setup is small enough to support fixed-point analysis of the
bootstrap operator near the BC minimum, but I do not have the theoretical
machinery to do that analysis cleanly.

I am a UW undergraduate (ECE + Applied Math, '28). If one claim looks
wrong on a 2-minute skim, I would value a one-line reply saying which
one.

Seed paper attached. Code: github.com/tejasnaladala/maze-rl-baselines.

Best,
Tejas Naladala
tejas.naladala@gmail.com

P.S. The "Transcendence" paper from Kempner is the framing I keep coming
back to here.
```

---

## SEND WORKFLOW

1. Verify each address from the lab page first (patterns above are best-known guesses)
2. Send ONE email per researcher per day; never CC, never BCC
3. Attach SEED_PAPER.pdf + paste Drive link in body (some filters drop attachments)
4. Track in spreadsheet: sent_date, replied (Y/N), reply_summary, follow_up_date
5. One follow-up after 5 business days if no reply, then drop
6. If Tier A lands a Yes: slow Tier B-E cadence to focus on the engaged conversation
7. If a Tier A reply offers strong critique that invalidates a claim: pause Tier B-E entirely, fix the claim, then resume

## ASKS LADDER (per researcher type)

| Recipient type | Primary ask |
|---|---|
| External senior PI (Tier A, B, C, E) | "If one claim looks wrong on a 2-minute skim, I would value a one-line reply saying which one" |
| UW local PI (Tier D) | 30 minutes in person at Allen + possible URA opening |
| Anyone who responds positively | Move to follow-up template (longer, propose specific collaboration mode) |

## REALISTIC EXPECTATIONS

- Reply rate from senior PIs to cold email at this caliber: 5 to 15 percent base rate
- BC -> DQN collapse is a strong enough hook to push toward the upper end of that range
- Realistic outcome: 2 to 4 of 13 reply with something substantive, 1 to 2 of those become a real conversation
- Best case: 1 collaborator engagement that converts to either coauthorship on a sharpened v1.0 or an internship/research-position offer
- Worst case: the seed paper gets zero substantive replies but the SHIP_AS_IS cleanup means it remains a defensible portfolio artifact

## CYCLIC-MAZE EXPERIMENTS: PAUSED. COUNT-BASED PPO: COMPLETE.

Per user instruction. The §2.1 cyclic-maze hypothesis remains in the seed paper as proposed work for v1.1, but no compute will be spent on cyclic-maze runs until the pause is lifted. The count-based intrinsic-motivation PPO sweep (n=20 seeds, the Codex audit's top experiment recommendation) is complete: mean 9.4 percent test success (sd 18.2, median 3.0, range 0 to 68; 10/20 seeds at 0-2 percent, only 2/20 above Random). Result is integrated into SEED_PAPER v0.4 abstract and Table 1; the under-baselined critique is materially defeated.
