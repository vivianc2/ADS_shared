# RPG v7 mixed9 — Opus 4.8 live run analysis

**Date:** 2026-08-06
**Batch:** `results_v7/mixed9_opus/` — 9 worlds (3 confounded_chain, 3 collider_selection,
3 hidden_subtype), 10-skin sampler, Opus 4.8, 32-turn / 15-query budget, ~41 min wall.
**Headline number:** accepted **0/9**. **Do not read this as difficulty** — it is
dominated by a grading/resolver artifact, exactly the "false 0/N" pattern this project
has hit at every prior scale-up (v4 0/8, v6 batch1 0/10). Inspected all 9; verdict below.

---

## 1. TL;DR

- **0/9 accepted, but 5/9 passed part A with benefit ≈ 0.92–1.00** — i.e. on 5 worlds
  the agent found a fix that recovers ~all achievable utility. The 0/9 is a **part B
  (mechanism battery) artifact**, not a reasoning wall.
- **Root cause of the artifact: the resolver cannot map the verbose free-text proxy
  strings the agent writes.** 8/9 worlds are artifact-flagged; the flag is correct.
  Example: on `bioprocess_chain` the agent named the proxy **"LDH (lactate dehydrogenase
  release from cell lysis)"** — that IS the true proxy `LDHRelease` — but the string
  didn't resolve, so part B scored it wrong. The science was right; the grader missed it.
- **This is a known, documented residual** (memory + v6 batch20 audit: "resolver
  under-maps very long free-text proxy strings … verdict-neutral, inflates flag count").
  At n=9 with a strict accept gate it stops being verdict-neutral and **sinks the whole
  batch**. It must be fixed before any number here is quoted or used as an RL reward.
- **Second, smaller issue: recommendation-action strings also fail to resolve on the
  hardest worlds** (2 subtype worlds: the agent's one action was rejected → gap 68).
  Part real reasoning failure (never found the conjunction / never stratified), part
  resolver miss (the action it did propose didn't map).
- **The worlds themselves are sound.** Spot-checked golds recompute correctly, including
  the subtype world whose gold is `two_cause pair + conditional policy` (base 27 →
  gold 95). No world-generation bug found.
- **Genuine signal that survives the artifact:** collider worlds split 1 solved / 2 not;
  subtype worlds 0/3 solved (the agent never produced a stratified policy on any of
  them) — a real, on-thesis reasoning gap. See §4.

---

## 2. The batch, per world

| world | archetype | part A | benefit | part B | rec∩gold | verdict |
|---|---|---|---|---|---|---|
| agronomy_chain | chain | **Y** | 0.92 | 0.20 | yes | **solved-A, partB artifact** |
| bioprocess_chain | chain | **Y** | 0.99 | 0.50 | no (alt fix) | **solved-A, partB artifact + alt-fix** |
| catalysis_chain | chain | **Y** | 0.92 | 0.20 | yes | **solved-A, partB artifact** |
| battery_collider | collider | **Y** | 1.00 | 0.60 | yes | **solved-A, partB artifact** |
| bioprocess_collider | collider | **Y** | 0.99 | 0.40 | yes | **solved-A, partB artifact** |
| aquaculture_collider | collider | . | 0.00 | 0.29 | no | **genuine fail** (wrong levers) |
| agronomy_subtype | subtype | . | 0.00 | 0.33 | no | **genuine fail** + action didn't resolve |
| battery_subtype | subtype | . | 0.20 | 0.17 | partial | **genuine fail** (no policy) |
| catalysis_subtype | subtype | . | 0.00 | 0.00 | no | **genuine fail** + action didn't resolve |

**Reading:** 5 "solved-A / partB-artifact" + 3 genuine subtype fails + 1 genuine collider
fail. A fair grader would likely accept **~4–5 of the 5 part-A passers** (need to confirm
part B once proxy strings resolve). So the *true* accept rate is plausibly **~4–5/9**, not
0/9 — pending the resolver fix and a re-grade.

---

## 3. The two artifact classes (verified)

### 3.1 Verbose proxy strings don't resolve (dominant — hits 7/9)

Part B's `true_mechanism_proxy` check resolves the agent's named proxy string to a world
variable, then checks membership in `valid_mechanism_proxies`. The agents write **rich,
explanatory** proxy descriptions; the lexical resolver (tuned for short aliases) drops them.

Confirmed examples (all scored `proxy_ok=False` despite correct science):
- `bioprocess_chain`: **"LDH (lactate dehydrogenase release from cell lysis)"** → true proxy
  is `LDHRelease`. **Correct.** Not resolved.
- `battery_collider`: "early-life coulombic efficiency (poor SEI formation from insufficient
  electrolyte additive)" → `CoulombicEfficiency`. Correct core. Not resolved.
- `agronomy_chain`: "interveinal chlorosis / Fe availability response to soil acidification"
  → `LeafGreenness`. Correct concept. Not resolved.
- The **one** that resolved (`aquaculture_collider`: bare `GillHistologyScore`) did so
  precisely because it was a single clean token.

The pattern is unambiguous: **resolution success is inversely correlated with how much the
agent explains itself.** We are penalizing the model for being articulate.

### 3.2 Recommendation-action strings don't resolve (hits the 2 hardest worlds)

On `agronomy_subtype` and `catalysis_subtype` the agent's recommended action was a full
sentence — "increase fertigation micronutrient/amendment dosing to overcome iron lockout",
"run feed purification/polishing to remove trace contaminants" — which the resolver
rejected, so `recommended_intervention` was empty → benefit 0 → gap ~68. Here the resolver
miss **compounds a genuine reasoning failure** (see §4), but it still means the reported
gap overstates the true miss.

### 3.3 Not artifacts (confirmed sound)

- **Alt-fix on `bioprocess_chain`**: agent used `FeedWaterFlowRate=0` (benefit 0.99), gold
  used `AntioxidantDosing=100`. Both are legitimate — reducing the source knob also reduces
  the root cause. Part A correctly credits it. Part B's actuator-sign check scores the
  *gold's* fix actuator, which the agent didn't touch → looks like a miss. This is a
  **grader rigidity**, arguably a third artifact class: part B should score the sign of
  *whatever* causally-valid lever the agent used, not only the gold's.
- **Subtype golds are correct**: `agronomy_subtype` gold = two_cause pair
  (CO2Enrichment+IrrigationRate, both=72 vs either≈28-41) + PruningIntensity policy →
  95. Recomputed and sound.

---

## 4. Genuine reasoning signal (survives the artifact)

Stripping the grading noise, the attributable findings:

- **Confounded_chain (3/3 solved on part A, benefit 0.92–0.99).** Opus reliably breaks the
  confound and finds a utility-optimal fix on the backbone. Consistent with all v6 batches
  (bioreactor-class worlds are the ones it solves).
- **Collider_selection: 2 solved-A / 1 genuine fail.**
  - `battery_collider` (benefit 1.00) and `bioprocess_collider` (0.99): **not trapped** by
    the selection decoy — recovered the true fix. Encouraging: the selection-bias trap did
    not fool it here.
  - `aquaculture_collider` (benefit 0.002): **genuine failure** — pushed `WaterpH` +
    `BiofilterLoad`, gold was `StockingDensity` + `Aeration`. Wrong causal levers entirely.
  - *Caveat:* none of the three explicitly flagged the selection decoy as a decoy, so we
    can't yet say they *understood* the selection structure vs. got the fix another way.
    Need the reasoning traces read closely (next pass).
- **Hidden_subtype: 0/3 solved, and the key finding — the agent NEVER produced a
  conditional policy.** On all three, `recommended_policy_text` was `None`; it treated the
  world as a single-lever problem. This is exactly the designed expert-gap: the model does
  not spontaneously hypothesize a hidden subtype and stratify. `battery_subtype` got
  partial benefit (0.20) by pushing the treatment uniformly; the other two failed outright.
  **This is the headline on-thesis result** — but it's entangled with the resolver miss on
  two of them, so re-run after the fix to get it clean.

---

## 5. What to fix before any re-grade or RL use

Priority order (all are grader/harness, not world-gen):

1. **Resolve verbose proxy/decoy strings (blocking).** Route the part-B structured-answer
   strings (`true_mechanism_proxy`, `confounded_decoys`) through the **LLM resolver
   fallback** when the lexical resolver fails, and/or strip parentheticals + post-dash
   clauses and retry progressively (the v6 note started this in `_translate_structured`;
   it is clearly insufficient for these strings). Target: LDH-class answers resolve.
2. **Resolve verbose recommendation-action strings.** Same treatment for
   `recommended_intervention_text` (and `recommended_policy_text.treatment/stratifier`).
   The lexical resolver was hardened for short names; long explanatory requests need the
   LLM fallback in the answer path too.
3. **Part-B actuator-sign fairness (alt-fix).** Score the sign of the causally-valid lever
   the agent actually used, not only the gold's fix actuator — otherwise a legitimate
   alternative fix (bioprocess_chain) is docked. Consider crediting part B if the agent's
   used actuator has the correct sign AND it named a valid proxy.
4. **Re-grade offline** (no new API spend) with `--fresh-oracle`-style re-resolution over
   the stored `answer_raw`, then re-read the accept rate. Expect ~4–5/9.
5. **Only then** quote any accept number or wire part B into an RL reward — a reward built
   on today's part B would **train the model to emit terse variable names instead of
   reasoning**, the opposite of the goal.

---

## 6. Implications for the RL plan

- **The resolver is now on the critical path for RL, not just eval.** Per the RL research
  doc, we planned lexical-resolver-in-loop for throughput. This run shows the **answer-time
  proxy/action resolution must use the LLM fallback** or part B is unusable as a reward
  signal. Options: (a) LLM-resolve only the terminal answer (cheap — one call at episode
  end), or (b) constrain the answer format so the model emits a canonical variable id
  (structured answer / tool-call schema) — the cleaner long-term fix, and it removes the
  resolver from the reward path entirely.
- **Reward-hacking angle (from the research doc):** today's part B is a *false-negative*
  machine (penalizes correct-but-verbose). That's the benign direction, but it still
  corrupts the gradient. The "master-key test" must include the inverse: confirm a
  correct-and-verbose answer scores high, not just that a degenerate answer scores low.
- **On-thesis signal is present and gradeable-in-principle:** chain solved, subtype not
  (no stratification), collider mixed. Once the grader is fair, this batch is a good
  smoke-test target for the reward function — several clear positives (chain), several
  clear negatives (subtype), which is exactly the reward spread RL needs.

---

## 7. Concrete next actions

1. Implement §5.1–§5.3 grader/resolver fixes (local, no API).
2. Add an offline re-grade script that re-resolves `answer_raw` from the stored results and
   recomputes part B — re-read accept rate on this exact batch (should rise from 0).
3. Add the inverse master-key test (correct-verbose answer must score high).
4. Read the collider reasoning traces to confirm whether the 2 "solved" ones actually
   understood the selection structure.
5. Fold these into the RL environment's answer contract (structured/canonical answer id) so
   the reward never depends on free-text resolution.

---

## 8. Fixes applied (2026-08-06) — status

All grader/resolver fixes from §5 are implemented and verified locally (no API spend):

- **§5.1 verbose proxy/decoy resolution** — new `Resolver.resolve_answer_term()`: tries
  progressively-shortened variants **head-form first** (so a parenthetical gloss can't
  hijack the match), plus an **exact rare-token rule** (a canonical acronym like "ldh"
  that is a unique alias token resolves even below the lexical threshold), plus LLM
  fallback on the head form. Verified: `"LDH (lactate dehydrogenase release from cell
  lysis)"` now → `LDHRelease` (was misfiring to the `LactateConc` distractor).
- **§5.2 verbose action resolution** — `_resolve_answer_intervention` /
  `_resolve_answer_policy` / sign-key translation all route through `resolve_answer_term`.
- **§5.3 alt-fix part-B fairness** — `_score_battery` now credits the sign of the
  causally-valid lever the agent actually used, not only the gold's fix actuator, when the
  agent recommended a different-but-valid fix and didn't opine on the gold lever.
- **Offline re-grade** — `regrade_v7.py` re-resolves stored `answer_raw` and re-grades with
  current code, no new API. Lexical-only re-grade of mixed9: several part-B scores rise
  (e.g. bioprocess_chain 0.50 → 0.75); the remaining misses are **semantic** proxy matches
  (coulombic efficiency, interveinal chlorosis) that need the **LLM resolver**.
- **Reward-integrity tests** — `test_reward_integrity.py`: master-key (empty / generic /
  all-decoy / single-knob-on-conjunction score ~0) **and** articulate-correct (verbose-but-
  right answers score high). Passes 6/6 worlds across all archetypes.

**Regression:** generate 60 (10 skins × 3 archetypes) → mock **60/60 accepted, partA 60/60,
0 artifacts**; the "do the measurement" → dissolved-oxygen false positive found and fixed.

**Confirmed genuine (not artifact) after fixes:** on `catalysis_chain` the agent named an
*actuator* (`do_RegenerationCycle`) as the proxy and mislabeled the true proxy
(`EffluentByproduct`) as a decoy — a real proxy/decoy inversion, correctly scored low.

**Still to do (needs the user's Bedrock creds):** re-run
`python regrade_v7.py --results-dir results_v7/mixed9_opus --resolver-llm --write`
to resolve the semantic proxy strings and read the true, fair accept rate on mixed9
(expected to rise from 0 toward the ~4–5/9 the part-A passers suggest).

---

## 9. LLM re-grade result (2026-08-07) — I was wrong about ~4–5/9; the 0/9 is largely REAL

Ran the LLM-resolver re-grade (`--resolver-llm`, Bedrock Opus). Result: **still 0/9, no
flips.** My "~4–5/9" prediction did **not** hold. Inspected all five part-A passers to see
why part B stays < 0.8 — and the honest finding is that **most of the remaining part-B
misses are genuine, not resolution artifacts.** The grader fixes were still correct and
necessary (bioprocess_chain rose to B=0.75, proxy+decoy now credited), but they revealed
that the underlying answers are mechanistically wrong more often than the raw flag implied.

**Per part-A passer, why B fails (after LLM resolution):**
| world | B | why (verified) |
|---|---|---|
| bioprocess_chain | 0.75 | **Closest to a pass.** Proxy ✓, decoys ✓, alt-fix sign ✓; the ONE miss is genuine — agent said the antioxidant (the gold lever) has sign "0" when it's "+". Real error. |
| battery_collider | 0.60 | Agent named proxy **"coulombic efficiency"** — which **is not a variable in this world** (world's proxy is `ImpedanceSpectrum`). It guessed a plausible battery marker from priors, not the world's actual proxy. Also failed to flag the selection decoy `CalendarAge`. Genuine. |
| bioprocess_collider | 0.40 | Named 2/3 true decoys but missed `ViableCellDensity`; proxy string didn't match. Partly genuine (incomplete decoy set), partly a naming gap. |
| agronomy_chain | 0.20 | **Proxy/decoy inversion:** called the true proxy `TissueNutrientAssay` a decoy; named the proxy as a symptom phrase ("interveinal chlorosis…"). Genuine inversion. |
| catalysis_chain | 0.20 | Named an **actuator** (`do_RegenerationCycle`) as the proxy; called the true proxy `EffluentByproduct` a decoy. Genuine inversion. |

**Revised verdict on mixed9:** part A (find a utility-optimal fix) is genuinely solved on
5/9 — that part is real and strong. But part B (correctly name the mechanism proxy, reject
the decoy, get signs right) is genuinely **not** met on 8/9, and after the resolver fixes
those are mostly real mechanistic errors, not miscredits: proxy/decoy inversions, naming an
actuator or a from-priors marker that isn't the world's proxy, and (collider) not detecting
the selection decoy. **So the headline is "acts right, explains wrong"** — the exact
expert-gap seen across every v6 batch, now confirmed clean on v7 with a trustworthy grader.

**Two real issues this surfaced (for follow-up, not blocking):**
1. **Battery/collider proxy contract muddiness.** In `battery_collider`, an intended inert
   distractor (`CrimpPressure`) is wired as a child of a chain mediator (`ActiveMaterialLoss`),
   so the targeted actuator moves it > 0.5 SD and the battery lists it as a "valid proxy."
   Not wrong (it IS causally downstream), but it means a distractor-named node counts as a
   proxy, blurring the proxy/decoy line. Worth tightening the collider/subtype augmentation
   so added observables don't accidentally sit on the causal chain.
2. **Part B is hard to pass by construction on these skins** because the agent must name the
   *specific world variable*, while a strong model naturally names the *mechanism concept*
   (often a real quantity that just isn't the sampled proxy). This is a **reward-design
   choice for RL**, not a bug: either (a) accept any causally-valid downstream observable as
   the proxy (more lenient, closer to "understood the mechanism"), or (b) keep the strict
   named-variable check (harder, more precise). Decide this before wiring part B into a
   reward — it materially changes the signal.

**Net:** the prototype and grader are now trustworthy — the 0/9 is a *real* result
(part-A-solved / part-B-not), not an artifact. My earlier ~4–5/9 estimate assumed the
misses were resolution failures; the LLM re-grade proved they're mostly genuine
mechanism errors. That is a cleaner, more honest headline for the RL work.
