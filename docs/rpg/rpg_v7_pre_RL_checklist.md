# RPG v7 → RL: what to verify before wiring up training

**Date:** 2026-08-07
**Canonical location:** all work continues in `ADS_shared/` (this repo). The v7 prototype
here is now byte-identical to the last `ADS_collab_clean/` copy + the fresh mixed9 re-grade
outputs (synced 2026-08-07). Treat `ADS_shared/dataset_generation_code/rpg_v7_prototype/`
as the source of truth.

---

## 0. Is the code fully correct now? — YES, with two known, documented caveats

- **Code parity:** `ADS_shared` v7 == the fully-fixed prototype (recursive diff clean).
- **Verified in `ADS_shared`:** `test_reward_integrity.py` passes 6/6; generate + mock is
  15/15 accepted, partA 15/15, 0 artifacts.
- **Grader/resolver fixes all in:** verbose-proxy resolution, verbose-action resolution,
  alt-fix part-B fairness, short-English guard. See `rpg_v7_mixed9_opus_analysis.md` §8.
- **mixed9 LLM re-grade done:** 0/9 is a *real* result (part A solved 5/9, part B mostly
  genuinely missed) — "acts right, explains wrong", not a grading artifact (§9).

**Two caveats that are correctness-adjacent (not bugs, but decisions):**
1. **Proxy contract can be muddy in collider/subtype worlds** — an intended inert
   distractor can end up wired onto the causal chain, so the battery lists it as a "valid
   proxy" (battery_collider: `CrimpPressure`). Tighten before large-scale RL (item V4).
2. **Part B requires naming the *specific sampled variable*** — a strong model naturally
   names the *mechanism concept*. This is a reward-design choice, not a bug (item V5).

---

## 1. Verification checklist BEFORE setting up RL

Grouped by risk. Each item: what to check, why it matters for RL, how to test. Items
marked **[done]** are already verified; **[todo]** need doing before/at RL bring-up.

### A. Reward correctness (the reward IS the training signal — highest priority)

- **V1 [done] Master-key: degenerate answers score ~0.** empty / generic / all-decoy /
  single-knob-on-conjunction all reject. `test_reward_integrity.py`. *Why:* an RL policy
  finds and exploits any trivial high-reward path.
- **V2 [done] Articulate-correct: verbose-but-right answers score high.** The mixed9 bug in
  the other direction. Same test. *Why:* otherwise RL learns terseness, not reasoning.
- **V3 [todo] Reward is monotonic + smooth enough to learn from.** Check that partial
  progress earns partial reward (benefit_recovered is continuous ∈[0,1]; battery_fraction
  steps in 0.2s). *Why:* GRPO needs within-group reward *spread*; a near-binary reward
  gives zero-variance groups (no gradient). *Test:* sample G=8 varied answers per world,
  confirm reward std > 0 on most worlds. **Add this to the test suite.**
- **V4 [todo] Proxy/decoy contract is clean.** Fix the collider/subtype augmentation so an
  added observable never sits on the causal chain (else a "decoy" is a valid proxy).
  *Test:* assert `valid_mechanism_proxies ∩ intended_inert_distractors == ∅` in the audit.
- **V5 [todo — DECIDE] Part-B proxy-credit policy.** Strict (name the exact variable) vs
  lenient (any causally-valid downstream observable counts). *Why:* materially changes what
  the model optimizes; pick before spending compute. Recommend: **lenient for RL reward**
  (rewards "understood the mechanism"), keep strict for the eval/benchmark.

### B. Reward path plumbing (grade() alone is not the reward)

- **V6 [known] The reward path is translate→grade, NOT grade().** Free-text resolution
  lives in the runner (`_translate_structured`, `_resolve_answer_*`); `grade()` expects
  canonical ids. *Action for RL:* either (a) the reward fn runs resolve→translate→grade, or
  (b) — recommended — give the RL env a **structured/canonical answer contract** (the model
  emits variable ids / a tool-call schema) so the reward never depends on free-text
  resolution. Decide this at env-design time.
- **V7 [todo] Resolver cost in-loop is bounded.** Lexical resolver only during training
  (LLM fallback = a second model call per action → kills throughput). *Test:* time a full
  rollout with lexical-only resolver; confirm per-turn overhead ≪ generation time.

### C. Environment mechanics (fast, deterministic, process-safe)

- **V8 [done] Determinism: same seed → identical world.** Verified. *Why:* reproducible
  rollouts, cacheable worlds, resumable runs.
- **V9 [done] Env construction is ~free from a stored world (0 ms w/ precomputed gold;
  284 ms if it recomputes).** *Action for RL:* **always load worlds with their stored
  oracle** (`load_world_file` / precomputed); never let the env recompute gold per episode.
- **V10 [todo] Rollout loop is process/thread-safe for parallel rollouts.** The trainer
  runs N rollouts concurrently. *Test:* run 16 `run_world` in parallel processes over the
  same world dir; confirm no shared-state corruption, identical grades to serial.
- **V11 [todo] Turn/step budget + termination are well-defined.** Confirm every episode
  ends (answer / give_up / turn cap) and emits a terminal reward; no infinite loops.
  Already largely handled (forced-answer at cap) — just re-verify under the RL harness.

### D. World supply + anti-memorization (what makes it *reasoning*, not lookup)

- **V12 [todo] On-demand world stream behind the audit gate.** RL wants fresh seeded worlds
  each step, not a fixed file set. *Test:* a generator that yields audited worlds
  indefinitely; confirm ~80–90% acceptance sustained and no seed collisions.
- **V13 [todo] Held-out split has zero leakage.** Reserve ≥2 skins + ≥1 archetype entirely
  for eval. *Test:* assert train/heldout share no (skin, archetype) cell and no world_id.
  *Why:* the held-out transfer number is the headline result — leakage invalidates it.
- **V14 [done] Structural diversity is real.** 10 skins × 3 archetypes × features × seeds,
  non-leaking names. Already the design; just don't regress it.

### E. Scale sanity (before committing compute)

- **V15 [todo] Measure real trace length + rollout wall-time on the target model.** The RL
  research doc's whole compute estimate hinges on ~12k tokens/trace — measure it first
  (Tinker/vLLM). *Why:* biggest single lever on the budget.
- **V16 [todo] Cost/curriculum smoke.** Run ~50 worlds × G=8 rollouts on the debug model
  (8B), confirm reward moves, entropy bounded, groups non-zero variance.

---

## 2. Recommended order of operations

1. **V3, V4, V5** — lock the reward (correctness + the two contract decisions). Cheap,
   local, highest leverage. *Do these next.*
2. **V6** — decide the answer contract (structured ids vs free-text-resolve). Shapes the
   env API, so decide before writing `env.py`.
3. **V10, V11, V7** — env mechanics under parallelism.
4. **V12, V13** — world stream + held-out split.
5. **V15, V16** — measure on the real model, then scale.

Items 1–4 are all doable locally with the mock backend (no GPUs, no API). Only 5 needs the
training stack.

---

## 3. The core/RL restructure (still pending your go)

Per `rpg_v7_STATUS_and_confirm.md` §2 + Step E: promote `rpg_v7_prototype` →
`rpg_core/` (engine, oracle, sampler, skins, resolver, generate) + `rpg_eval/` (run/grade/
regrade/sim) + a new empty `rpg_rl/` (env, reward, curriculum, Tinker glue). Do this
**inside `ADS_shared`** now that it's canonical. I'll update all import paths and re-run
V1/V8/V9 to prove nothing broke. Waiting on your approval before moving files.

---

## 4. One-line status

*`ADS_shared` is now the correct, up-to-date, verified copy. Code is correct; the open
work before RL is reward-contract decisions (V3–V6) + env-under-parallelism (V10–V11) +
world-stream/held-out (V12–V13), all local/no-GPU. Then measure trace length on the target
model (V15) before committing compute.*
