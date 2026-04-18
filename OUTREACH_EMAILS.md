# Outreach Emails: UW + Tier 1 Researchers

Cold-email package for the procedural-maze RL benchmark paper.

**Two-stage strategy.** Send the lean cold email first (~140 words).
If they reply showing interest, send the long version with the
journey, the audit, and the explicit ask. Senior PIs skim cold emails
in under 20 seconds. The lean version is built for that.

Sections:

1. **Cold Email v1** (lean, ~140 words). Default for every send.
2. **Reply Template v2** (~350 words). Send only after they engage.
3. **Researcher list** with email-pattern guesses.
4. **Per-researcher openers**, one sentence each.
5. **Send workflow**.

Attach `PAPER_SHORT.pdf` (6 pages, 16 KB). Optional:
`PAPER_PREVIEW.pdf` (10 pages, 28 KB) for the longer version, only on
follow-up. Repo link in body.

---

## 1. Cold Email v1 (default, ~140 words)

```
Subject: 5-line heuristic 100%, distilled MLP 97%, modern RL 31% (= Random) on procedural mazes

Dear Prof. [LAST_NAME],

[ONE-SENTENCE PERSONAL OPENER from Section 4.]

I built a hazard-maze benchmark where a 5-line egocentric wall
follower solves 100% of unseen instances and the same MLP
architecture used by DQN reaches 97.4% by BFS distillation. The best
of seven HP-tuned modern reward-driven baselines (SB3 PPO, DQN, and
A2C across three LRs each, 70 runs total) reaches 31% mean,
statistically tied with uniform Random (33%) and 66 percentage points
below the distillation result. PPO at every tested LR plateaus at
3 to 6%. A capacity sweep (h32 to h256), LSTM memory, a K4 reward
ablation, and a MiniGrid cross-env replication all rule out the
obvious explanations.

The cleanest experiment: initialize MLP_DQN from the 97% BC-distilled
weights and fine-tune via standard DQN. Test success collapses from
97.2% to 13.6% across 5 seeds (per-seed: 0, 12, 16, 18, 22). The
reward gradient actively destroys the distilled high-performing
representation, ending below from-scratch DQN. The network class can
express the policy. Reward-driven RL pushes the network out of the
basin that contains it.

Paper attached, 6 pages. Code, raw data, SHA-256 manifest of all
4,131 result files: https://github.com/tejasnaladala/engram

Would 15 minutes work for a hard critique before I submit?

Thanks,
Tejas Naladala
tejas.naladala@gmail.com
```

**Why this length.** Senior PIs get 50+ cold emails a week. They scan
the subject, the first line, and the ask. Anything more goes
unread. The lean version has one specific result, one ruled-out list
in a single sentence, one specific ask (15 minutes). If they bite,
you can send the long version.

---

## 2. Reply Template v2 (only after they engage, ~350 words)

```
Subject: Re: [their reply subject]

Thanks for getting back so quickly. Quick context on how this started
and where it stands.

I built a small procedural-maze evaluation harness to test some
neural-RL baselines and noticed they were losing to random walks. I
assumed I had a bug. After about 3,500 runs across 20+ seeds per
cell, paired bootstrap with Holm-Bonferroni, the capacity, LR, LSTM,
and shaping ablations from the paper, plus a MiniGrid replication on
DoorKey, FourRooms, MultiRoom-N2-S4, and Unlock (3 of 4 also show
MLP-DQN at or below Random), the result held.

I caught a harness-filtering bug in some early auxiliary launchers
(an is_solvable(avoid_hazards=True) filter that quietly inflated
baselines by 12 to 22 percentage points), wrote the validation table
directly into §3.2.1, and re-ran the headline experiments on the
corrected harness. The 97.4% distillation number survived the fix.
The 19.3% DQN number survived the fix.

A modern-baseline sweep (PPO, A2C, and DQN at three LRs each, 10
seeds each, on the corrected harness) is finishing this week.
Procgen replication is queued.

The two questions I most want answered before submission:

  1. Does this separate representation from exploration cleanly enough
     to be worth a paper, in your view?
  2. What additional baseline (NGU-style episodic novelty, RND,
     PPO-LSTM at scale, anything else) would you demand to either
     defeat or sharpen the claim?

If the result interests you, I would welcome any of: a brief written
critique, your group running independent reproduction on a different
procedural generator, or co-authorship on a sharpened version. I am
solo on this with limited compute (~40 GPU-hours total) and a second
team replicating the headline would matter.

Longer version of the paper (with appendices, audit transcripts):
attached as PAPER_PREVIEW.pdf.

Thanks again,
Tejas
```

---

## 3. Researcher List

### A. UW Current Faculty (Allen School and affiliated)

| # | Name | Lab / Affiliation | Email pattern | Why they care |
|---|------|-------------------|---------------|---------------|
| 1 | Simon Du | Allen School (RL theory) | ssdu@cs.washington.edu | Theory of RL; cover-time framing maps to his work. |
| 2 | Natasha Jaques | Allen School, Social RL Lab | nj@cs.washington.edu | Exploration, intrinsic motivation, MiniGrid relevance. |
| 3 | Abhishek Gupta | Allen School, WEIRD Lab | abhgupta@cs.washington.edu | Robot RL, real-world RL failure modes. |
| 4 | Emo Todorov | UW Movement Control Lab | todorov@cs.washington.edu | Model-based control; baselines-first culture. |
| 5 | Kevin Jamieson | Allen School (active learning) | jamieson@cs.washington.edu | Sample-efficient learning; baseline rigor. |
| 6 | Byron Boots | UW Robotics Lab | bboots@cs.washington.edu | Robot learning; model-based RL. |

### B. UW Alum / Closely Affiliated

| # | Name | Current Affiliation | Email pattern | UW connection |
|---|------|---------------------|---------------|---------------|
| 7 | Sham Kakade | Harvard / Kempner | sham@seas.harvard.edu | UW faculty 2018 to 2023. |
| 8 | Aravind Rajeswaran | Microsoft AI Frontiers | aravraj@microsoft.com | UW PhD; DAPG, Decision Transformer, R3M. |
| 9 | Vikash Kumar | CMU / Meta | vikashplus@gmail.com | UW PhD; dexterous manipulation RL. |
| 10 | Pieter Abbeel | Berkeley | pabbeel@cs.berkeley.edu | UW EE BS 2000. |

### C. Non-UW Tier 1

| # | Name | Affiliation | Email pattern | Why they care |
|---|------|-------------|---------------|---------------|
| 11 | Peter Henderson | Princeton | peter.henderson@princeton.edu | Deep RL That Matters methodology. |
| 12 | Rishabh Agarwal | GDM / Mila | rishabhagarwal@google.com | Statistical Precipice protocol. |
| 13 | Marc Bellemare | Mila / DeepMind | bellemare@google.com | ALE random baseline; cover-time fit. |
| 14 | Joelle Pineau | Meta / McGill | jpineau@meta.com | Reproducibility checklist author. |
| 15 | Pablo Castro | Google DeepMind | psc@google.com | Dopamine baseline rigor. |
| 16 | Roberta Raileanu | Meta AI | rraileanu@meta.com | MiniGrid and procgen exploration. |
| 17 | Tim Rocktäschel | Meta AI / UCL | rockt@meta.com | NLE, MiniHack, procedural exploration. |
| 18 | Pulkit Agrawal | MIT CSAIL | pulkitag@mit.edu | Active perception, intrinsic motivation. |
| 19 | Benjamin Eysenbach | Princeton | eysenbach@princeton.edu | Exploration as inference. |
| 20 | Sergey Levine | Berkeley | svlevine@eecs.berkeley.edu | Deep RL canonical voice. |
| 21 | Chelsea Finn | Stanford | cbfinn@cs.stanford.edu | Meta-learning, distillation. |

> Patterns above are guesses. Verify from the lab page before sending.

---

## 4. Per-Researcher Openers (one sentence each)

Drop into the `[ONE-SENTENCE PERSONAL OPENER]` slot. Specific, no
flattery, ties their work to the result.

### A. UW Current

1. **Simon Du**: *Your RL theory work was the framing I kept returning to when the random walks beat the neural agents.*
2. **Natasha Jaques**: *Your Social RL work is part of why I started taking exploration as a first-class object, and 3 of 4 MiniGrid envs in my replication show MLP-DQN below Random.*
3. **Abhishek Gupta**: *Congratulations on the RAS Early Career Award; I am UW-local and would gladly come present a 30-minute version of this in person at WEIRD Lab.*
4. **Emo Todorov**: *Your culture of building careful baselines first is the spirit I tried to keep here, and the headline implicates standard reward-driven RL specifically.*
5. **Kevin Jamieson**: *Your sample-efficient learning work made me twice as careful about the seed counts here (20+ per cell, paired bootstrap, Holm-Bonferroni).*
6. **Byron Boots**: *The benchmark exposes a model-free failure that distillation from a planning oracle resolves cleanly, which I think connects to your model-based vs model-free trade-off work.*

### B. UW Alum

7. **Sham Kakade**: *Your Statistical RL course at UW (and rltheorybook with Agarwal, Jiang, Sun) is part of why I framed this as a representation-vs-discovery question.*
8. **Aravind Rajeswaran**: *Your DAPG work at UW is the closest precedent for the headline here: distillation from a planner closes a gap that reward-driven RL cannot.*
9. **Vikash Kumar**: *Your dexterous-manipulation work at UW is one of the cleanest examples of distillation-from-planning closing an exploration gap, and I think this is a small sharp instance of the same phenomenon.*
10. **Pieter Abbeel**: *I am UW-adjacent and a long-time reader of your group's deep-RL work; the result here is small in scope but unusually clean.*

### C. Non-UW Tier 1

11. **Peter Henderson**: *Your Deep RL That Matters (AAAI 2018) is the methodological model I built this work around, including a harness-bug audit trail in §3.2.1 after an adversarial review caught a filter I had missed.*
12. **Rishabh Agarwal**: *Your Statistical Precipice (NeurIPS 2021) shaped the analysis pipeline here, and the headline would not be reportable under a less rigorous protocol.*
13. **Marc Bellemare**: *I empirically confirm the Alon-Benjamini-Lubetzky-Sodin (2007) non-backtracking cover-time theorem on a procedural RL benchmark, with NoBackRandom and uniform Random scaling exponents differing by 0.84 units (95% bootstrap CI).*
14. **Joelle Pineau**: *Your reproducibility work shaped this from day one: SHA-256 manifest of all 4,131 results, code-hash pinned per record, harness-audit table in §3.2.1.*
15. **Pablo Castro**: *The Dopamine philosophy of careful comparable baselines first is the spirit I tried to extend with this 5-tier ladder, and the most surprising row is a 97.4% supervised MLP on the same observation that DQN solves at 19.3%.*
16. **Roberta Raileanu**: *Your work on procedural environments and MiniGrid exploration is the closest precedent for the result here, in a much smaller benchmark fully reproducible in 40 GPU-hours.*
17. **Tim Rocktäschel**: *Your NLE and MiniHack work made procedural exploration a first-class topic for me, and this benchmark isolates a specific failure mode at a much smaller scale.*
18. **Pulkit Agrawal**: *Your intrinsic-motivation work is part of why I read the failure here as a discovery problem, and I would value your view on whether RND, ICM, or NGU would close the gap.*
19. **Benjamin Eysenbach**: *Your exploration-as-inference framing is the sharpest theoretical lens I know for the kind of failure I land at here.*
20. **Sergey Levine**: *I am a single-author with a small but unusually clean procedural-maze benchmark that rules out the standard explanations for a 78pp distillation-vs-DQN gap.*
21. **Chelsea Finn**: *Your work on meta-learning and distillation is part of how I think about the supervised-MLP-from-oracle result here, which I think makes a small fast testbed for distillation methods.*

---

## 5. Send Workflow

1. Verify each address from the lab page first.
2. Pick the one-sentence opener from Section 4. Paste the Cold Email
   v1 body verbatim. Attach `PAPER_SHORT.pdf`. Send.
3. Send in waves of 5 over 7 to 10 days. UW current first, then UW
   alum, then non-UW Tier 1.
4. Track in `outreach_log.csv` (sent_date, replied, follow_up_date).
5. If they reply with interest: send Reply Template v2 within 24
   hours. Attach `PAPER_PREVIEW.pdf` for the appendices.
6. If no reply after 10 business days: send a 2-line nudge
   ("re-sending in case it slipped past, paper attached again, no
   pressure"), then drop.

Do not ask for funding or a PhD slot in cold email. Save for
second-touch conversations after they have engaged.
