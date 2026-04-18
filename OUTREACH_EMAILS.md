# Outreach Emails: UW + Tier 1 Researchers

Cold email package for the procedural-maze RL benchmark paper. The
master template explains the journey, the finding, what we are doing
now, and the collaboration ask. Reuse the body verbatim and rotate the
personal opener and subject line per researcher.

Three sections:

1. **Master template** with the full body.
2. **Researcher list**: UW current, UW alum, then non-UW Tier 1.
3. **Per-researcher openers**, one short paragraph each.

Attach `PAPER_SHORT.pdf` (6 pages, ~16 KB) to every send. Optional
longer version: `PAPER_PREVIEW.pdf` (10 pages, ~28 KB) for those who
want appendices. Link to `https://github.com/tejasnaladala/engram` in
body.

---

## 1. Master Template (use for every send)

```
Subject: [PERSONAL_SUBJECT_LINE]

Dear Prof. [LAST_NAME],

[PERSONAL_OPENER, 2 to 3 sentences. See Section 3 below.]

I built a fully reproducible hazard-maze benchmark where a five-line
egocentric wall follower solves 100% of unseen instances and the same
MLP architecture used by DQN reaches 97.4% by BFS distillation. Yet
DQN, DoubleDQN, and DRQN all stay around 15 to 19%. This looks like a
clean representability-versus-discovery failure rather than a capacity
failure, and I have been unable to make any of the standard
explanations stick (capacity, learning rate, partial observability,
reward shaping, observation richness).

Headline ladder, all on the same audited test harness at 9x9 mazes:

  Tier 1 Oracle              BFSOracle                 100.0%
  Tier 2 Heuristic (5 line)  EgoWallFollowerLeft       100.0%
  Tier 3 Distillation        DistilledMLP_from_BFS      97.4%
  Tier 4 Random walk         NoBackRandom               51.5%
  Tier 4 Random walk         Levy alpha=2.0             40.3%
  Tier 5 Tabular             FeatureQ_v2                36.5%
  Tier 4 Random walk         Random                     32.7%
  Tier 6 Neural RL           MLP_DQN                    19.3%
  Tier 6 Neural RL           DRQN (LSTM)                19.0%
  Tier 6 Neural RL           DoubleDQN                  16.3%

The 78 percentage point gap between Tier 3 and Tier 6 is the central
observation. Same 24-d ego-feature observation, same 24-64-32-4
architecture, same Adam optimizer. Only the training signal differs.
The neural class can express the policy. Standard reward-driven RL
just does not discover it.

How this started. I built a small procedural-maze evaluation harness
to test some neural-RL baselines and noticed they were losing to
random walks. I assumed I had a bug. After about 3,500 runs across 20
or more seeds per cell, paired bootstrap with Holm-Bonferroni, a
capacity sweep (h32 to h256, all flat at 13 to 19%), an LR sweep
(default is the local optimum across 1.5 orders of magnitude), DRQN
with LSTM matching MLP-DQN, a K4 reward ablation (learners collapse
without shaping; random walks unchanged), and a cross-environment
replication on MiniGrid (DoorKey, FourRooms, MultiRoom-N2-S4, Unlock,
where 3 of 4 also show MLP-DQN at or below Random), the result held.

I also caught a harness-filtering bug in early auxiliary launchers (an
`is_solvable(avoid_hazards=True)` filter that quietly inflated
baselines by 12 to 22 percentage points), wrote the validation table
directly into the methods section, and re-ran the headline experiments
on the corrected harness. The 97.4% distillation number survived the
fix. The 19.3% DQN number survived the fix. The gap is not an artifact
of the bug.

Where I am now. A modern-baseline sweep (PPO, A2C, and DQN at three
learning rates each, 10 seeds each, on the corrected main-sweep
harness) is finishing this week. Cross-environment replication on
Procgen is queued. The paper, the full reproducibility package
(SHA-256 manifest of all 4,131 result files, code-hash pinned per
record, single-file statistics pipeline, smoke test that hits every
agent class in 3 minutes), and the raw data are public.

What I am hoping for from you.

  1. A hard critique before submission. Specifically: does this cleanly
     separate representation from exploration in a way you find
     convincing, and what additional baseline (NGU-style episodic
     novelty, RND, IMPALA, PPO-LSTM at scale, decision transformer
     with offline data, anything else) would you demand to defeat or
     sharpen the claim?

  2. If the result interests you, I would welcome a collaboration. That
     could be co-authorship on a sharpened version, your group running
     independent reproduction on different procedural generators, or
     feedback that reframes the result toward a venue you think is the
     right home for it. I am a single author with limited compute
     (about 40 GPU-hours total to date), and the work would benefit
     from a second team replicating the headline.

  3. If you know a junior researcher or PhD student in your group who
     works on RL exploration, intrinsic motivation, or neural-RL
     failure modes, I would be glad of an introduction.

Paper (6 pages, claims plus proofs): attached as PAPER_SHORT.pdf
Longer version with appendices:      attached as PAPER_PREVIEW.pdf
Code, raw data, manifest:            https://github.com/tejasnaladala/engram

Happy to share the full draft, the audit transcripts, or the harness
validation script if useful. I am also glad to do a 30-minute call at
your convenience.

Thanks for your time.

Best,
Tejas Naladala
tejas.naladala@gmail.com
```

---

## 2. Researcher List

### A. UW Current Faculty (Allen School and affiliated)

| # | Name | Lab / Affiliation | Email pattern | Why they care |
|---|------|-------------------|---------------|---------------|
| 1 | **Simon Du** | Allen School (RL theory) | ssdu@cs.washington.edu | Theory of RL; cover-time framing maps to his work. |
| 2 | **Natasha Jaques** | Allen School, Social RL Lab | nj@cs.washington.edu | Social RL, exploration, intrinsic motivation; MiniGrid relevance. |
| 3 | **Abhishek Gupta** | Allen School, WEIRD Lab | abhgupta@cs.washington.edu | Robot RL, real-world RL failure modes; recent RAS Early Career winner. |
| 4 | **Emo Todorov** | UW (Movement Control Lab, MuJoCo) | todorov@cs.washington.edu | Model-based control, exploration, baselines-first culture. |
| 5 | **Kevin Jamieson** | Allen School (active learning, bandits) | jamieson@cs.washington.edu | Sample-efficient learning, baseline rigor. |
| 6 | **Byron Boots** | UW Robotics Lab (RL + control) | bboots@cs.washington.edu | Robot learning, model-based RL. |

### B. UW Alum / Closely Affiliated (high signal, low cost; they know the building)

| # | Name | Current Affiliation | Email pattern | UW connection |
|---|------|---------------------|---------------|---------------|
| 7 | **Sham Kakade** | Harvard / Kempner Institute | sham@seas.harvard.edu | UW faculty 2018 to 2023; built the Statistical RL course there. |
| 8 | **Aravind Rajeswaran** | Microsoft AI Frontiers (was Meta FAIR) | aravraj@microsoft.com | UW PhD with Sham and Emo; DAPG, Decision Transformer, R3M. |
| 9 | **Vikash Kumar** | CMU / Meta | vikashplus@gmail.com | UW PhD with Emo; dexterous manipulation RL. |
| 10 | **Pieter Abbeel** | Berkeley (BAIR) | pabbeel@cs.berkeley.edu | UW EE BS 2000; deep RL pioneer. |

### C. Non-UW Tier 1 (selected for fit with the failure-mode story)

| # | Name | Affiliation | Email pattern | Why they care |
|---|------|-------------|---------------|---------------|
| 11 | **Peter Henderson** | Princeton | peter.henderson@princeton.edu | "Deep RL That Matters" methodology lens. |
| 12 | **Rishabh Agarwal** | GDM / Mila | rishabhagarwal@google.com | Statistical Precipice protocol fit. |
| 13 | **Marc Bellemare** | Mila / DeepMind | bellemare@google.com | ALE random-baseline tradition; cover-time theorem fit. |
| 14 | **Joelle Pineau** | Meta / McGill | jpineau@meta.com | Reproducibility checklist author. |
| 15 | **Pablo Castro** | Google DeepMind | psc@google.com | Dopamine baseline rigor. |
| 16 | **Roberta Raileanu** | Meta AI | rraileanu@meta.com | MiniGrid + procedural-environment exploration; direct topical match. |
| 17 | **Tim Rocktäschel** | Meta AI / UCL | rockt@meta.com | NLE, MiniHack, exploration in procedural envs. |
| 18 | **Pulkit Agrawal** | MIT CSAIL | pulkitag@mit.edu | Active perception, intrinsic motivation. |
| 19 | **Benjamin Eysenbach** | Princeton | eysenbach@princeton.edu | Exploration as inference, RL failure modes. |
| 20 | **Sergey Levine** | Berkeley | svlevine@eecs.berkeley.edu | Deep RL canonical voice. |
| 21 | **Chelsea Finn** | Stanford | cbfinn@cs.stanford.edu | Meta-learning, robot RL, distillation lens. |

> Email addresses above are best-known patterns. Verify each one from
> the lab page before sending. If `@google.com` bounces, try
> `@deepmind.com` or the lab page contact form.

---

## 3. Per-Researcher Personal Openers

Drop these into the `[PERSONAL_OPENER]` slot. Two to three sentences
each. Specific, no flattery.

### A. UW Current

**1. Simon Du**, `ssdu@cs.washington.edu`
- Subject: *Procedural maze benchmark, empirical confirmation of Alon-Benjamini-Lubetzky-Sodin (2007) cover-time theorem on RL*
- Opener: *Your work on the theory of RL was the framing I kept coming back to when the random walks beat the neural agents. The empirical result I land at, with NoBackRandom decaying with maze size at scaling exponent -2.04 [-2.21, -1.94] versus uniform Random at -2.88 [-3.20, -2.59] (the predicted ABLS 2007 0.84-unit gap), felt worth your eye specifically.*

**2. Natasha Jaques**, `nj@cs.washington.edu`
- Subject: *Social RL Lab: neural RL that loses to random in MiniGrid*
- Opener: *Your Social RL work is part of why I started caring about exploration as a first-class object. Three of four MiniGrid environments in my cross-env replication (DoorKey, FourRooms, Unlock) show MLP-DQN below Random, a result I think your group is unusually well placed to interpret, and one I would value being challenged on.*

**3. Abhishek Gupta**, `abhgupta@cs.washington.edu`
- Subject: *WEIRD Lab: neural RL failure mode on procedural mazes; UW-local, would value 30 min*
- Opener: *Congratulations on the RAS Early Academic Career Award, well earned. I am a single-author working on a small procedural-maze RL benchmark, and I think the result speaks to the broader discovery-versus-representation question your robotics RL work has been sharpening. I am UW-local and would be glad to come present in person if useful.*

**4. Emo Todorov**, `todorov@cs.washington.edu`
- Subject: *Procedural maze benchmark; cover-time framing for an RL failure*
- Opener: *Your culture of building careful baselines first, and then questioning what they say, is the spirit I tried to keep here. The headline implicates standard reward-driven RL specifically, not function approximation, and I would value your view on whether the cover-time framing is the right theoretical anchor.*

**5. Kevin Jamieson**, `jamieson@cs.washington.edu`
- Subject: *Sample-efficient evaluation of RL baselines: 20+ seeds, paired bootstrap*
- Opener: *Your work on sample-efficient learning made me twice as careful about the seed counts here (20+ seeds per cell, paired bootstrap, Holm-Bonferroni). I would value your read on whether the headline ordering, posterior-certain at the 0.001 level by Beta-Binomial dominance, is reported with the rigor your active-learning standards would demand.*

**6. Byron Boots**, `bboots@cs.washington.edu`
- Subject: *Distillation closes a procedural-RL exploration gap*
- Opener: *Your robotics RL work is part of how I learned to think about model-based versus model-free trade-offs. The benchmark exposes a model-free failure that distillation from a planning oracle resolves cleanly; I would value your view on whether the result is interesting outside the procedural-maze niche.*

### B. UW Alum

**7. Sham Kakade**, `sham@seas.harvard.edu`
- Subject: *Representation versus discovery in procedural RL; UW alum reaching out*
- Opener: *Your Statistical RL course at UW (and the rltheorybook with Agarwal, Jiang, and Sun) is part of why I framed this as a representation-versus-discovery question rather than a capacity question. The 78pp gap between supervised distillation and DQN on identical architecture is the cleanest separation I have been able to construct, and I would value your view on whether the framing is right.*

**8. Aravind Rajeswaran**, `aravraj@microsoft.com`
- Subject: *DAPG-shaped result on a procedural maze benchmark*
- Opener: *Your PhD work at UW with Sham and Emo (DAPG in particular, where demonstrations resolve an exploration-hard task) is the closest precedent I have for the headline here: a supervised MLP from a BFS oracle reaches 97.4% on the same observation that DQN solves at 19.3%. I would value your view on whether this generalizes beyond procedural mazes the way DAPG did beyond dexterous manipulation.*

**9. Vikash Kumar**, `vikashplus@gmail.com`
- Subject: *Distillation-from-planning on a procedural maze benchmark*
- Opener: *Your dexterous manipulation work at UW is one of the cleanest examples of distillation-from-planning closing an exploration gap. I think this benchmark is a small, sharp instance of the same phenomenon, and I would value your view on whether it is worth a co-authored sharpening pass.*

**10. Pieter Abbeel**, `pabbeel@cs.berkeley.edu`
- Subject: *UW alum: small procedural maze where standard DQN loses to random*
- Opener: *I am a UW-adjacent single-author and a long-time reader of your group's deep-RL work. The result here is small in scope but unusually clean: a 5-line heuristic and a supervised MLP both solve a procedural-maze family that DQN, DoubleDQN, and DRQN with the same observation cannot, ruling out capacity, LR, memory, and shaping as explanations. I would value your view on whether it is a benchmark worth being more widely aware of.*

### C. Non-UW Tier 1

**11. Peter Henderson**, `peter.henderson@princeton.edu`
- Subject: *MLP at 97% via BFS distillation, 19% via DQN, same architecture; would value your reproducibility lens*
- Opener: *Your "Deep RL That Matters" (AAAI 2018) is the methodological model I built this work around: 20+ seeds per cell, paired bootstrap, Holm-Bonferroni, Cohen d, code-hash-pinned reproducibility, SHA-256 manifest of all raw results. After an adversarial review caught an evaluation-harness bug I had missed, I wrote the validation table and the audit trail directly into §3.2.1 of the paper. I would be especially grateful for your reproducibility lens before submission.*

**12. Rishabh Agarwal**, `rishabhagarwal@google.com`
- Subject: *Procedural maze evaluation result, your Statistical Precipice protocol applied*
- Opener: *Your "Deep RL at the Edge of the Statistical Precipice" (NeurIPS 2021) shaped the analysis pipeline here: stratified bootstrap, IQM, family-wise error correction, performance profiles. The headline would not be reportable under a less rigorous protocol, which is part of why I think it matters. Whether the paper holds up under your statistical standards is the question I most want answered before submission.*

**13. Marc Bellemare**, `bellemare@google.com`
- Subject: *Non-backtracking random walk as RL baseline, ABLS (2007) confirmed empirically*
- Opener: *The Arcade Learning Environment paper convinced the field that the random-policy baseline is load-bearing; this work tries to sharpen that for procedural mazes. I empirically confirm the Alon-Benjamini-Lubetzky-Sodin (2007) non-backtracking cover-time theorem on a procedural RL benchmark: NoBackRandom and uniform Random scaling exponents differ by 0.84 units (95% bootstrap CI), exactly the predicted advantage. I would value your view on whether the cover-time framing is the right theoretical anchor.*

**14. Joelle Pineau**, `jpineau@meta.com`
- Subject: *Reproducibility-first procedural RL evaluation, SHA-256 pinned, harness audit included*
- Opener: *Your reproducibility work changed how I thought about RL evaluation, so I tried to build this with that lens from day one. Every run is SHA-256-hashed in a manifest, every code change pins a code_hash on result records, and I ran an adversarial audit across five rounds during development. The audit caught a harness-filtering bug I had missed; the corrected validation table is in §3.2.1.*

**15. Pablo Castro**, `psc@google.com`
- Subject: *Compact procedural maze benchmark with surprising RL failure mode and 5-tier baseline ladder*
- Opener: *The Dopamine philosophy of careful, comparable baselines first is the spirit I tried to extend in this benchmark. The most surprising row is policy distillation: a vanilla MLP trained on BFS-oracle action labels with the same 24-d ego-feature observation as MLP-DQN recovers the policy at 97.4% test success (n=20). The same architecture trained via DQN reaches 19.3%. I would value Dopamine-team feedback on whether this benchmark is worth being added to a standard suite for sample-efficient agent eval.*

**16. Roberta Raileanu**, `rraileanu@meta.com`
- Subject: *MiniGrid replication of a procedural-RL failure mode; 3 of 4 envs show MLP-DQN below Random*
- Opener: *Your work on procedurally generated environments and exploration in MiniGrid is the closest precedent for the result here. Three of four MiniGrid environments (DoorKey, FourRooms, Unlock) in my cross-env replication show MLP-DQN at or below Random; this is consistent with the pattern your group has documented in larger procgen settings, but in a much smaller benchmark that is fully reproducible in 40 GPU-hours. I would value your view on whether the result is worth a sharpening pass.*

**17. Tim Rocktäschel**, `rockt@meta.com`
- Subject: *Procedural maze exploration failure; MiniHack-adjacent, much smaller scale*
- Opener: *Your work on NLE and MiniHack made procedural exploration a first-class topic for me. This benchmark is much smaller (9x9 hazard mazes, full reproducibility on a laptop GPU), but it isolates a specific failure mode: a 5-line heuristic and a supervised MLP both solve the family that DQN, DoubleDQN, and DRQN cannot, with the same observation. I would value your view on what NetHack-scale exploration intuition predicts here.*

**18. Pulkit Agrawal**, `pulkitag@mit.edu`
- Subject: *Procedural maze with a 78pp distillation-versus-DQN gap; intrinsic motivation question*
- Opener: *Your active-perception and intrinsic-motivation work is part of why I took the failure here as evidence about discovery rather than representation. I would value your view on whether an intrinsic-motivation augmentation (RND, ICM, NGU-style episodic novelty) would close the 78pp gap, or whether the right diagnosis is somewhere else entirely.*

**19. Benjamin Eysenbach**, `eysenbach@princeton.edu`
- Subject: *Exploration-as-inference framing on a small RL failure mode*
- Opener: *Your "exploration as inference" framing is the sharpest theoretical lens I know for the kind of failure I land at here. Whether this benchmark is a clean instance of the failure your theory predicts, or whether it requires a different framing, is a question I would value your view on.*

**20. Sergey Levine**, `svlevine@eecs.berkeley.edu`
- Subject: *Small procedural maze benchmark; 78pp distillation-versus-DQN gap on identical architecture*
- Opener: *I am a single-author working on a small but unusually clean procedural-maze benchmark. The headline rules out capacity, learning rate, partial observability (DRQN with LSTM), and reward shaping as explanations for a 78pp gap between supervised distillation and DQN on identical architecture. I would value your view on whether this is worth bringing to your group's exploration-research attention.*

**21. Chelsea Finn**, `cbfinn@cs.stanford.edu`
- Subject: *Small RL benchmark for distillation and demonstration-augmented methods*
- Opener: *Your work on meta-learning and distillation is part of how I think about the supervised-MLP-from-oracle result here (97.4% test success on the same observation that DQN solves at 19.3%). I would value your view on whether this benchmark is useful as a small, fast testbed for distillation and demonstration-augmented RL methods.*

---

## 4. Send Workflow

1. **Verify each address** from the lab page (patterns above are the
   most likely match but not guaranteed).
2. **Personalize the opener** from Section 3, paste the master template
   body verbatim from Section 1.
3. **Attach `PAPER_SHORT.pdf`** (and optionally `PAPER_PREVIEW.pdf` for
   reviewers who want appendices). Link to the GitHub repo in the body.
4. **Send in waves of 5 over 7 to 10 days**, not all at once. Batch by
   tier: UW current first (tightest signal, shortest distance), then
   UW alum, then non-UW Tier 1.
5. **Track responses** in `outreach_log.csv` (sent_date, replied,
   reply_summary, follow_up_date).
6. **Follow up** once after 10 business days if no reply, then drop.

---

## 5. Asks Tier (for context; the master ask block stays as is)

Top-priority asks in descending order:
- **Hard critique before submission** (everyone)
- **Independent reproduction** by their group on different generators (UW current, especially WEIRD Lab; Raileanu group)
- **Co-authorship on a sharpened version** (Kakade, Rajeswaran, Henderson, Agarwal, Castro)
- **Junior-researcher introduction** (everyone, especially helpful from senior PIs at UW alum nodes)

Lowest-priority ask: do not ask for funding or a PhD slot in a cold email.
Save those for replies and second-touch conversations.
