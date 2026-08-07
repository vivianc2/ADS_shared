# RPG v7 — where we are, and what to confirm (step by step)

**Date:** 2026-08-07
**Purpose:** single place to see current state + the checklist you confirm before we
restructure into core/RL. Nothing here is destructive; the core/RL split has **not**
happened yet (deliberately held until you confirm).

---

## 1. Where everything lives

Working dir (the "prototype folder"):
```
ADS_collab_clean/dataset_generation_code/rpg_v7_prototype/
```

| File | Role |
|---|---|
| `engine.py` | SCM engine (mechanisms, selection/conditioning, per-unit policy actuator) |
| `oracle_v6.py` | gold search, counterfactual battery, grader (`grade`, `optimal_gold`, `_score_battery`) |
| `sampler.py` | structural sampler — 3 archetypes (`confounded_chain`, `collider_selection`, `hidden_subtype`) |
| `skins.py` | 10 domain skins (bioprocess, datacenter, watertreatment, agronomy, clinical, semiconductor, aquaculture, battery, catalysis, fermentation) |
| `resolver.py` | free-text → canonical id resolver; **new** `resolve_answer_term()` (the mixed9 fix) |
| `generate_v7.py` | generate + audit-gate a batch of worlds (`--archetype`, `--skins`, `--require-feature`) |
| `run_agent_v6.py` | the agent rollout loop + answer translation + grading hookup |
| `run_batch_v6.py` | run a batch; **has `--resume`** |
| `regrade_v7.py` | **new** — offline re-grade of a finished run (no API needed unless `--resolver-llm`) |
| `test_reward_integrity.py` | **new** — master-key + articulate-correct reward guardrail |
| `sim_v6.py`, `sandbox.py`, `bedrock_llm.py`, `openai_llm.py`, `analyze_results.py` | runtime sim, code tool, LLM backends, aggregation |

Generated worlds:
```
out_v7/chain/      (24)   out_v7/collider/   (24)   out_v7/subtype/  (24)
out_v7/mixed9/     (9 = 3+3+3, the live-run set)
```

Live run + results:
```
results_v7/mixed9_opus/   result_<wid>.json (9) + summary.json  (Opus 4.8, ~41 min)
```

Docs (all under `ADS_collab_clean/docs/rpg/`):
| Doc | What it is |
|---|---|
| `rpg_v7_rl_research.md` | field survey + RL plan + compute (you expanded with the lit sweep) |
| `rpg_v7_mixed9_opus_analysis.md` | the 0/9 analysis + the fixes-applied section (§8) |
| `rpg_v7_STATUS_and_confirm.md` | **this file** |

---

## 2. The core/RL split — WHERE IT WILL GO (not done yet)

**Status: NOT created. Held on purpose until you confirm §4.** Nothing has moved.

Proposed target layout (sibling packages, so the RL work imports the science as a
library instead of forking it):
```
ADS_collab_clean/dataset_generation_code/
├── rpg_core/          # promoted from rpg_v7_prototype — the trustworthy science
│   ├── engine.py  oracle.py  sampler.py  skins.py  resolver.py
│   ├── generate.py            # was generate_v7.py
│   └── (tests)
├── rpg_eval/          # running an agent + grading a batch (eval, not training)
│   ├── run_agent.py  run_batch.py  regrade.py  sim.py  sandbox.py  *_llm.py
│   └── test_reward_integrity.py
└── rpg_rl/            # NEW — the RL training space (empty until we build it)
    ├── env.py                 # RPGEnv: reset()/step() around the rollout loop
    ├── reward.py              # w_A·benefit + w_B·battery − penalties (translate→grade)
    └── (Tinker glue, world-stream, curriculum, held-out split)
```
Renames drop the `_v6`/`_v7` suffixes (they're just version scars now). This is a
mechanical move + import-path update; no logic change. **I will not touch it until you
say go**, because moving files invalidates the paths inside the existing `results_v7/`
records and the docs.

---

## 3. What is actually true right now (verified)

- **Worlds are sound.** Generate 60 (10 skins × 3 archetypes) → 60/60, audit filter
  working (85% accept). Mock solvability (mock plays gold) → **60/60 accepted, partA
  60/60, 0 artifacts.**
- **Grader/resolver fixes are in and verified** (no API used):
  - verbose proxy strings resolve (`"LDH (lactate dehydrogenase…)"` → `LDHRelease`, was
    misfiring to a distractor);
  - verbose action/policy strings route through the same resolver;
  - part-B credits a valid *alternative* fix, not only the gold's lever;
  - `"do the measurement"` no longer false-matches dissolved-oxygen.
- **Reward-integrity test passes 6/6** across archetypes (degenerate → ~0; articulate-
  correct → high).
- **mixed9 lexical re-grade** lifted some scores (bioprocess_chain part B 0.50→0.75) but
  flipped no accepts, because the remaining proxy misses are *semantic* and need the LLM
  resolver.

---

## 4. WHAT YOU ARE CONFIRMING (the checklist)

Do these in order. Each is a yes/no gate; if any looks wrong, stop and tell me.

### ☐ Step A — reward integrity (you can run now, no API)
```bash
cd ADS_collab_clean/dataset_generation_code/rpg_v7_prototype
conda activate ADS-rpg
python test_reward_integrity.py
```
**Confirm:** prints `ALL REWARD-INTEGRITY CHECKS PASSED`.
*Meaning:* the grader can't be gamed by trivial answers AND doesn't punish correct-but-
verbose answers — the precondition for using it as an RL reward.

### ☐ Step B — worlds still generate & are solvable (you can run now, no API)
```bash
python generate_v7.py --outdir /tmp/confirm --n 30 --seed 900001
python run_batch_v6.py --worlds-dir /tmp/confirm --backend mock --outdir /tmp/confirm_res
```
**Confirm:** generation ~80–100% acceptance across all 3 archetypes; mock summary shows
`accepted N/N, partA N/N, artifacts 0`.
*Meaning:* every world's gold is recoverable and self-consistent (no world-gen bug).

### ☑ Step C — the real read on mixed9 (DONE 2026-08-07, with Bedrock LLM resolver)
`python regrade_v7.py --results-dir results_v7/mixed9_opus --resolver-llm --write` — ran.
**Result: 0/9 -> 0/9, no flips.** My ~4–5/9 prediction was WRONG. Inspecting all 5 part-A
passers showed the remaining part-B misses are **mostly genuine mechanism errors, not
resolution artifacts**: proxy/decoy inversions, naming an actuator or a from-priors marker
that isn't the world's sampled proxy, and (collider) not detecting the selection decoy.
The grader fixes were still correct + necessary (bioprocess_chain rose B 0.50→0.75, proxy
+ decoy now credited) — they just revealed the answers are wrong more often than the raw
artifact flag implied. **Revised headline: part A solved 5/9 (real), part B not met 8/9
(mostly real) = "acts right, explains wrong"** — the expert-gap, now clean on a trustworthy
grader. Full detail: `rpg_v7_mixed9_opus_analysis.md` §9.

### ☐ Step D — sign off on the on-thesis finding
After Step C, read `docs/rpg/rpg_v7_mixed9_opus_analysis.md` and confirm you agree with
the headline: **chain solved, collider mixed, hidden_subtype 0/3 because the agent never
produced a conditional policy** (the designed expert-gap).
*Meaning:* this is the result the RL work is meant to move.

### ☐ Step E — approve the restructure (§2)
Only after A–D look right, tell me to promote `rpg_v7_prototype` → `rpg_core` /
`rpg_eval` and create the empty `rpg_rl/` space. I'll update all import paths and the
doc/results references in one pass and re-run Steps A–B to prove nothing broke.

---

## 5. After confirmation — what I build next (for reference, not now)

1. `rpg_rl/reward.py` — the translate→grade reward (the mixed9 fix means the reward path
   must include resolution, or use a canonical-id answer contract).
2. `rpg_rl/env.py` — backend-agnostic `reset()/step()` around the rollout loop; targets
   Tinker's `Env` per the RL research doc (Managed-API lane you chose).
3. World-stream + held-out split (reserve ≥2 skins + ≥1 archetype); pass@k eval.
4. Then the Tinker debug loop on a small model.

---

## 6. One-line status

*Prototype + grader are trustworthy and fully verified (Steps A–C done). mixed9 is a REAL
result — part A solved 5/9, part B mostly genuinely missed (8/9) = "acts right, explains
wrong", not a grading artifact. Open items: (D) you sign off on that headline; (E) approve
the core/RL split. One reward-design decision to make before RL: strict named-variable
proxy check vs. accept-any-causally-valid-downstream-observable (analysis §9).*
