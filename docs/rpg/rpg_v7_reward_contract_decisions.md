# RPG v7 reward-contract decisions (V3–V6)

**Date:** 2026-08-07 · **Location:** ADS_shared (canonical)
**Scope:** the four reward-contract items that must be settled before wiring RL, decided
with measurements rather than from first principles. Each has: finding → decision →
action.

---

## V3 — Does the reward give a learnable signal (within-group variance)?

### Finding (measured)
On a real world, the reward `r = 0.5·benefit_recovered + 0.5·battery_fraction` spreads
cleanly across an answer-quality ladder:

| answer | reward | benefit | battery |
|---|---|---|---|
| empty / wrong knob | 0.01 | 0.01 | 0.00 |
| right fix, no mechanism | 0.50 | 1.00 | 0.00 |
| + proxy | 0.67 | 1.00 | 0.33 |
| + decoys | 0.83 | 1.00 | 0.67 |
| full (+signs) | 1.00 | 1.00 | 1.00 |

- **benefit is smooth across dose** (chelator 0→66: 0.74, 0.81, 0.87, 0.93, 0.98, 1.00;
  then over-treatment falls 0.36, −0.54). Good partial-credit gradient, including the
  interior-optimum penalty — exactly what we want.
- **A realistic quality-spread group has std ≈ 0.284** — strong learning signal.
- **BUT the reward saturates:** an all-`q=1` group (all "right fix, no mechanism") → all
  0.50 → **std = 0 → no gradient**; an all-solved group → all 1.0 → std = 0. This is the
  classic GRPO zero-variance-group failure, and it *will* happen: once the policy reliably
  finds the fix but not the mechanism, most rollouts land at 0.50.

### Decision
- **Keep the dense continuous reward** (`0.5·benefit + 0.5·battery`); do NOT collapse to
  binary accept/reject — the continuity is what gives the gradient.
- **Require DAPO dynamic sampling** in the trainer (drop groups whose rollouts are all-
  equal reward) — this is a trainer setting, not a reward change, and it directly kills the
  zero-variance case.
- **Curriculum keeps groups in the informative band:** don't train long on worlds the
  policy has saturated (all-1.0) or can't touch (all-0). Adaptive difficulty (raise when
  batch pass-rate is high) keeps groups spread.
- Add a **V3 regression test**: reward std > 0 on a realistic mixed-quality group.

### Status
Reward shape is correct; no reward-formula change needed. Action = trainer config (DAPO) +
curriculum + a test.

---

## V4 — Proxy/decoy contract is muddy in collider/subtype worlds (BUG — fix)

### Finding (measured)
**58/149 audited worlds (39%)** list an intended *inert distractor* as a
`valid_mechanism_proxy`. Concentrated in `collider_selection` (nearly every one). Root
cause, traced exactly:

- The collider archetype draws the **collider node** and **selection-decoy driver** from the
  inert-var pool, then wires the collider node as a child of `last_mediator` (the outcome-
  driving chain signal) + the driver.
- So the **targeted actuator moves the collider node** (it's causally downstream of the
  mediator by construction) → the battery's empirical proxy check (`targeted actuator moves
  it > 0.5 SD`) flags the collider node as a "valid proxy."
- The collider node therefore appears in `valid_mechanism_proxies` **even though it has no
  ground-truth role** (it's not the labeled proxy and not a labeled decoy). Example
  (seed 600000): `valid_mechanism_proxies = ['GillHistologyScore', 'UVSterilizer']` where
  `UVSterilizer` is the collider node.

Why it matters for RL: the battery is the part-B reward. If a distractor node counts as a
"valid proxy," then (a) an agent naming that distractor gets part-B credit it shouldn't,
and (b) the proxy/decoy contract the world is supposed to teach is blurred — the exact kind
of oracle gap an RL policy exploits.

### Decision
**Exclude selection/collider machinery nodes from the empirical proxy scan**, and give the
collider node an explicit role so it is never an unlabeled free-floater. Specifically:
1. Sampler records the collider node + selection driver names in `ground_truth`
   (`_selection_nodes`), and the collider node is added to `confounded_decoys` (it IS a
   thing that correlates with the outcome but is not the mechanism — that's a decoy by
   definition here).
2. `counterfactual_battery` excludes `_selection_nodes` from the `valid_mechanism_proxies`
   scan (they are structurally downstream but are not mechanism proxies; they're the
   selection apparatus).
3. Audit gains a check: `valid_mechanism_proxies ∩ (inert distractors ∪ selection nodes) = ∅`.

This keeps the true proxy correct, removes the leak, and makes the collider node a proper
decoy the agent is *supposed* to reject.

### Status
Implementing now (sampler + oracle + audit). Re-verify: 0 leaks across the same scan;
mock still 60/60; reward-integrity still passes.

---

## V5 — Part-B proxy credit: strict (exact variable) vs lenient (any valid downstream)

### The tension (from the mixed9 LLM re-grade)
Strong models name the **mechanism concept** ("coulombic efficiency", "interveinal
chlorosis"), often a real quantity that just isn't the *specific sampled proxy variable* in
this world. Under a strict check that scores 0; the model understood the mechanism but is
docked for not naming our exact node. Conversely, a fully lenient check ("any observable the
true lever moves") would credit a distractor that happens to sit downstream (the V4 leak) —
so lenient is only safe *after* V4 is fixed.

### Decision — CORRECTED after measurement: keep STRICT; leniency doesn't apply
My first draft proposed a lenient reward that credits "any observable downstream of the
true root that isn't a decoy." **Measurement killed this idea:** in **0/96** sampled worlds
does the lenient set differ from the strict set. Reason — by construction of the sampler,
the mediators are **latent** and the proxy is the **single observable attached to the
mechanism chain**, so "any downstream observable" == "the proxy". Lenient-by-graph-position
is a no-op here.

More importantly, inspecting the mixed9 failure showed the premise was wrong. The agent did
**not** name a *different world variable*; it named **"coulombic efficiency"**, which is
**not a variable in that world at all** (the world's proxy is `ImpedanceSpectrum`, one of 15
observables the agent could have identified from the data). That is a **genuine reasoning
miss** — naming a marker from domain priors instead of identifying the world's actual
measurable — which is exactly the gap we want to train *against*, not paper over.

**So: keep the STRICT check for both reward and eval.** The legitimate leniency we DO want —
crediting the agent for naming the true proxy by an alias or a verbose description — is the
**resolver's** job (`resolve_answer_term`, already fixed in V4-era work: "LDH (…)" →
`LDHRelease`), NOT the proxy-set's. Strict proxy-set + good resolver is the right
combination: precise about *which* signal, forgiving about *how it's phrased*.

### Action (done)
- `_score_battery` / `grade` gained a `strict` flag (default **True**) and the battery now
  also stores `lenient_mechanism_proxies` — kept as **plumbing for the future** (e.g. if a
  later sampler adds multiple observables per chain, or we want an ablation), but the reward
  will use `strict=True` like eval. No behavioral difference today (lenient==strict).
- The real lever for "don't punish articulate-correct" stays the resolver + the V6
  structured-answer contract (canonical ids), not a lenient proxy set.

### Status
Decided (corrected). Flag implemented and harmless; default and recommendation = strict.

---

## V6 — Answer contract: structured canonical ids vs free-text→resolve

### The problem (established)
`grade()` expects **canonical ids**; free-text resolution lives in the runner
(`_translate_structured`, `_resolve_answer_*`). The mixed9 run showed free-text resolution
is lossy (verbose strings misfire) and, for RL, the LLM-resolver fallback is a second model
call per action → throughput killer. So the reward must not depend on fragile free-text
resolution.

### Decision — structured/canonical answer contract for RL, free-text kept for eval
For the RL environment, the agent emits its **final answer as a structured object with
canonical ids** chosen from the world's presented catalog — the env shows the agent the
list of measurable/actuator ids (already neutral, non-leaking names), and the answer
references them directly:
```json
{"recommended_actions": [{"actuator": "<id>", "value": <num>}],
 "policy": {"treatment": "<id>", "stratifier": "<id>", "threshold": .., "dose_if_ge": .., "dose_if_lt": ..},
 "true_mechanism_proxy": "<measurable id>",
 "confounded_decoys": ["<id>", ...],
 "actuator_sign_predictions": {"<id>": "+|-|0"}}
```
- **Reward path = translate-free:** ids go straight into `grade()`. No resolver in the
  reward loop → fast, deterministic, unhackable-via-phrasing.
- The agent still reasons in free text *during* the rollout (measure/intervene actions can
  stay free-text + lexical resolver in-loop); only the **terminal answer** is structured.
- **Eval/benchmark keeps the free-text answer + resolver** (that's how a general model
  would answer; it's the honest external measure).

Rationale: this removes the single largest source of reward noise and cost (free-text
resolution) from the training signal, while keeping the benchmark faithful to how an
un-structured model answers. It also makes the reward path a pure function of ids → no LLM,
fully reproducible — important for RLVR integrity.

### Trade-off accepted
The structured contract makes the answer *slightly* easier (no naming ambiguity), so the RL
task is marginally easier than the benchmark. That's fine and intended: we want the reward
to measure *reasoning*, not *naming*, and we measure the harder end-to-end version at eval.

### Action
- Define the answer schema + a `parse_structured_answer()` that validates ids against the
  world catalog (unknown id → that field scores 0, no resolver).
- `reward.py` (in the future `rpg_rl/`) calls grade() directly on the parsed ids.
- Keep the runner's free-text path for `run_batch_v6` eval unchanged.

### Status
Decided. Implement as part of the `rpg_rl/` env (after the core/RL split), since it defines
the env's answer API. V5's `strict=False` reward and V6's structured contract are the two
things `reward.py` will encode.

---

## Summary of decisions

| Item | Decision | Change type |
|---|---|---|
| **V3** | keep dense continuous reward; rely on DAPO dynamic sampling + curriculum for group variance | trainer config + test |
| **V4** | exclude selection/collider nodes from valid-proxy scan; make collider node a labeled decoy | **code fix now** (sampler+oracle+audit) |
| **V5** | KEEP STRICT for reward + eval (lenient is a no-op: 0/96 worlds differ; mixed9 miss was naming a non-existent variable = real error). Leniency-of-phrasing = resolver's job, not proxy-set's | `strict` flag added (default True), harmless plumbing |
| **V6** | RL answer = structured canonical ids (translate-free reward); eval keeps free-text+resolver | env API, at `rpg_rl/` build |

**Order:** implement V4 now (unblocks V5); add the V3 test now; V5 flag now (small); V6 lands
with the RL env. V4 + the V3 test + the V5 flag are all local/no-GPU and I can do them in
this pass.

---

## Implementation status (2026-08-07) — V3/V4/V5 done, V6 deferred to rpg_rl

- **V4 [done]** sampler records `_selection_nodes` (collider node + selection driver);
  `counterfactual_battery` excludes them from the proxy scan. **Verified: 0 valid-proxy
  leaks across all 94 stored worlds** (was 58/149 pre-fix). Collider audit pass-rate
  unchanged (~50-67%; the weak-correlation rejects are by design, not this change).
- **V3 [done]** `test_reward_integrity.py` now asserts a realistic mixed-quality group has
  reward std > 0.05 (measured 0.31). Passes 6/6. Reward formula unchanged; DAPO dynamic
  sampling remains a trainer requirement for the saturated-group case.
- **V5 [done, corrected]** `strict` flag added to `_score_battery`/`grade` (default True);
  battery stores `lenient_mechanism_proxies`. Measurement showed lenient == strict in 0/96
  worlds (single observable per chain), so **reward uses strict too**; flag kept as future
  plumbing.
- **V6 [deferred]** structured canonical-id answer contract — lands with the `rpg_rl/` env
  (it defines the env's answer API). Decision recorded above.
- **All regressions green:** generate 45 → mock 45/45, 0 artifacts; mixed9 (clean 9) mock
  9/9; reward-integrity 6/6; determinism holds.

### ⚠️ Consequence: the old live mixed9 results are now STALE
Recording `_selection_nodes` changed how names are consumed from the pools, so the SAME
seeds now draw slightly different worlds → **world_ids shifted**. The regenerated
`out_v7/{chain,collider,subtype,mixed9}` are the correct current sets, but the old
`results_v7/mixed9_opus/` live-Opus run (and its `regrade_*.json`) reference pre-regeneration
worlds — **4/9 of those world files no longer exist on disk**, and the grader semantics
changed anyway. Those results are kept for the record but **must not be compared to the new
worlds**. To get comparable numbers under the fixed grader, the mixed9 live Opus run must be
**re-run** against the regenerated `out_v7/mixed9/` (needs Bedrock creds). This is the one
cost of regenerating; flag it before quoting any mixed9 number.
