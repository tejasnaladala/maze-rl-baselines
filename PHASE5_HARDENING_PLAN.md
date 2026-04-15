# Phase 5: Code/Pipeline Hardening Plan

Run after all experiments complete. Safe to edit `experiment_lib_v2.py` once no launchers reference it.

## Applied already (during Phase 5 partial)

- ✅ `stats_pipeline.py` seed-aligned paired bootstrap + alias normalization (Codex S2)
- ✅ `reward_decomposition.py` corrected formula + logging + canonical_agent strict `__` separator
- ✅ `launch_memory_agents.py` deterministic=True + full config writeback + code_hash
- ✅ `reproduce.py` excludes `checkpoint.json` from manifest + added all result dirs
- ✅ `final_analysis.py` includes V2 tabular + capacity study dirs
- ✅ `phase4_reviewer_attacks.py` A9 upgraded to DEFEATED based on cover-time analysis
- ✅ Orphaned Tier 2 slow runs moved to `attic/exp_reward_ablation_orphan/`
- ✅ Schema validation: 1,512 result files, 0 integrity issues
- ✅ Code hash audit: single hash `fe0b8142940e55de` across all 1,009+ runs (no drift)

## Pending (after experiments complete)

### 1. `experiment_lib_v2.py::set_all_seeds` — remove PYTHONHASHSEED no-op

**Issue:** Line 740 sets `os.environ['PYTHONHASHSEED'] = str(seed)` at runtime. Python reads `PYTHONHASHSEED` only at interpreter startup, so a runtime assignment is a no-op. Misleads readers.

**Fix:** Delete the line. Document in the function docstring that deterministic hashing requires launching Python with `PYTHONHASHSEED=<seed>` or `PYTHONHASHSEED=0` at shell level.

### 2. `experiment_lib_v2.py::code_hash()` — memoize at module load

**Issue:** `code_hash()` reads `__file__` bytes on every call. If the file is edited mid-run, subsequent calls produce a different hash than earlier calls, and result records written within the same process diverge.

**Fix:** Memoize the value at module import time:
```python
_CODE_HASH_AT_IMPORT = hashlib.sha256(
    open(__file__, 'rb').read()
).hexdigest()[:16] if Path(__file__).exists() else 'unknown'

def code_hash() -> str:
    return _CODE_HASH_AT_IMPORT
```

### 3. `experiment_lib_v2.py::run_experiment` — seed diversification for Random variants

**Issue:** Codex C1: Random variants seed their private RNG with the *same* seed value via `agent.seed(seed)`. This creates correlated action streams across Random, NoBackRandom, LevyRandom(1.5), LevyRandom(2.0). The effect is statistically benign (it strengthens paired tests because the policies see identical initial randomness on identical mazes), but it should be documented.

**Fix:** Either (a) document the design choice in the paper and method notes as "deliberately paired RNG", or (b) diversify by hashing agent name into the seed: `agent.seed(seed + (hash(agent_name) & 0xFFFFFFFF))`. Option (a) is simpler and consistent with paired-bootstrap logic. Go with (a) — no code change.

### 4. `experiment_lib_v2.py::NoBacktrackRandomAgent._reverse(-1)` — first-step bias

**Issue:** On the initial step (before `_last_action` is set), `_reverse(-1) = 1` (the "right" action) is excluded. Small bias toward {up, down, left} on step 0 of each episode.

**Fix:**
```python
def act(self, obs, step):
    if self._last_action < 0:
        choices = list(range(NUM_ACTIONS))
    else:
        choices = [a for a in range(NUM_ACTIONS)
                   if a != self._reverse(self._last_action)]
    a = self._rng.choice(choices)
    self._last_action = a
    return a
```

**Effect:** Documented as <1% change to success rate. Re-run NoBackRandom after the fix as a sanity check (20 seeds × 6 sizes = 120 runs, ~30 min).

### 5. `experiment_lib_v2.py::SpikingQNetwork::act` — per-layer synops (if running SpikingDQN)

**Issue:** Codex S8: synops = `dense_ops * (fr1 + fr2) / 2` averages layer-1 and layer-2 firing rates. Correct formula scales each layer by *input* firing rate:
```python
ops_l1 = num_steps * OBS_DIM * hidden  # rate=1 (dense input)
ops_l2 = num_steps * hidden * (hidden//2) * fr1
ops_out = num_steps * (hidden//2) * NUM_ACTIONS * fr2
```

**Fix:** Only needed if we report SpikingDQN synops in the paper. Currently we're not running SpikingDQN. Skip unless Tier 1 runs.

## Regression test

After all Phase 5 edits to `experiment_lib_v2.py`:
1. Run `python smoke_test.py --sizes 9,13 --num-train 10 --num-test 20` — expect 18/18 passing
2. Run `python reproduce.py freeze --out manifest_post_hardening.json` and confirm `manifest_post_hardening.json` matches `manifest_final.json` on the `headline` field for every existing agent. If any divergence, that means the edits changed behavior and we need to decide whether to re-run the affected experiments.

## Sanity check: bit-reproducibility audit

Codex recommended this — do it once after experiments finish:
1. `cp raw_results/exp_reward_ablation_fast/full__MLP_DQN_9_1024.json /tmp/ref_mlp_1024.json`
2. Remove the entry from `checkpoint.json` and delete the result file
3. Re-run `launch_reward_ablation_fast.py` — it will redo only that one seed
4. Compare the new file's `reward`/`steps`/`solved` fields against `/tmp/ref_mlp_1024.json`
5. If they match exactly, determinism is confirmed. If not, investigate.

## Shipping checklist

- [ ] All experiments complete per FINAL_STATUS.md
- [ ] `python finalize.py` runs clean (7/7 OK)
- [ ] `python reproduce.py freeze --out manifest_final.json` produces stable manifest
- [ ] `python reproduce.py verify --manifest manifest_final.json` exits 0
- [ ] Phase 5 edits applied to `experiment_lib_v2.py`
- [ ] `python smoke_test.py` passes 18/18
- [ ] `paper.md` has all tables populated with real numbers
- [ ] `paper_figures/` has fig1-5 in PNG and PDF
- [ ] `PHASE4_REVIEWER_ATTACKS.md` verdict updated (target: 8+ DEFEATED out of 11)
- [ ] `FINAL_STATUS.md` "Paper readiness verdict" updated
- [ ] Git commit + tag as `arxiv-v1`
