# RPG v5 — Progress Log + Worked Example

Companion to `worldgen_rpg_plan_v5_scm_chain.md` (the design) and the runnable
code in `dataset_generation_code/rpg_v5_prototype/`. Snapshot: **2026-08-04**.

## Part A — Progress

### What is built and working

A self-contained v5 pipeline (numpy-only for gen; Bedrock for the agent),
deliberately kept out of the fragile shared runner so trial runs don't trip over
the old star-graph schema check or the old string-match grader.

| File | Role | Status |
|---|---|---|
| `scm.py` | generic SCM evaluator + closed mechanism library; JSON (de)serialization | ✅ |
| `worlds.py` | 2 domains (bioreactor, water) on one structural family; serializable | ✅ |
| `oracle.py` | golden intervention (MC + golden-section), counterfactual battery, auto-calibration, faithfulness audits, solvability certificate, record-based grader | ✅ |
| `generate.py` | jitter params + **shuffle neutral labels** + calibrate + audit; emit only worlds that pass every audit | ✅ 6/6 |
| `sim_v5.py` | runtime simulator: observational / interventional / **sweep** / **clamp**; returns compact stats + correlations | ✅ |
| `run_agent_v5.py` | LLM scientist loop (Opus 4.8 default via Bedrock), full per-turn trace logging, computed grading, `mock` backend for smoke tests | ✅ |
| `demo_solve.py` | scripted expert trajectory + correct-vs-surface grading demo | ✅ |

### Sanity checks passed (under conda env `ADS-rpg`)

- Generation: 6/6 worlds pass name-leakage, decoy-inertness, proxy-signal-band,
  and solvability audits. Neutral labels shuffle so the targeted knob is not in
  a fixed slot (observed: RegimenC / RegimenD / RegimenF across the batch).
- Mock backend: drives all four query modes, grader wiring correct, 6/6 "accept"
  when replaying gold (this only checks plumbing, not capability).
- **Negative control:** the grader rejects a null/wrong answer (accepted=False,
  utility gap ≈48, battery 0.25). This is the property the v4 string-matcher
  lacked.
- Bedrock layer imports under `ADS-rpg`; temperature-rejection guard extended to
  the Opus 4.x line so a live Opus-4.8 run won't fail on an unsupported param.

### The key correctness change vs v4

v4 accepted a latent-cause answer by **keyword/alias string-matching**
(`simulator_rpg.py::_score_latent_cause_answer`). v5 accepts iff **both**:

- **(A) utility-optimal** — the recommended intervention's expected utility
  (recomputed from the SCM by Monte Carlo) is within tolerance of the golden
  intervention's; and
- **(B) counterfactual battery ≥ 80%** — the agent's structured predictions
  (which observable is the true mechanism proxy, which are confounded decoys,
  the sign of each knob's true effect on the outcome) match ground truth derived
  from the SCM.

Because both are computed from the world we own, a correct-but-differently-worded
answer passes and a plausible-sounding surface-proxy answer fails — which is
exactly what we need to tell "hard" apart from "broken."

### How the trace lets us see the human/agent gap *meaningfully*

Every run writes, per turn: the agent's `reasoning`, the exact query, the
returned stats, the accumulated `scientist_memory`, and (on the answer turn) the
full grade breakdown — `part_a_utility_ok`, `part_b_battery_ok`,
`battery_items` (which specific predictions were right/wrong), `utility_gap`,
and `queries_used`. So a failure is attributable to a *reasoning* error
(e.g. "never ran a clamp, so mistook the confound for the cause"; "stopped at 3
queries with an untested hypothesis"; "recommended the max dose, missing the
interior optimum") rather than to a parse/exec artifact. Answer-parse and
query-error turns are logged separately and do **not** count as capability
failures.

### First live Opus-4.8 result (1-world smoke) + a bug it surfaced

The one-world live smoke (bioreactor) produced a genuinely useful outcome:
**accepted=False, but for the right reason.** Opus passed part B (battery 0.88 →
correct true proxy, correct signs, ran the confound-breaking clamp, found the
sign flip, ruled out oxygen) yet **failed part A on dose**. Two real,
trace-visible reasoning errors:

1. It swept the chelator on the coarse default grid `{0,25,50,75,100}`, read the
   peak at 75, and never refined → missed the true interior optimum at 66.
2. It also set `FeedWaterFlow=0` ("avoid pulling in contaminant") — but flow's
   harm runs *through* copper, which chelation already removed, so zeroing flow
   was an unnecessary, slightly costly move. It did not re-reason the graph
   *after* its own intervention.

Investigating the utility gap surfaced a **generator bug**, now fixed: the
bioreactor SCM had a small direct `FeedWaterFlow → CarbonFlux` "nutrient" edge.
Once chelation nulls the copper path, that edge made "chelate + high flow"
(util ~83) beat the single-knob gold (~72). Since the oracle only searches
single-knob interventions, it mislabeled the gold — and the grader would have
*accepted* an answer the worked example calls wrong ("crank the flow"). Fix:
flow now acts only through copper (the counterintuitive "more feed = worse"
survives; the sweep is monotone-down). Added a **`gold_optimality` audit** that
rejects any world where a targeted-knob × other-knob pair beats single-knob gold
by more than the tolerance, so this bug class can never ship silently. All 6
trial worlds pass it; the interior optimum (66, util 65 → declining to 55 at max)
is intact.

Takeaway for the batch: even a frontier model that *reconstructs the mechanism
correctly* can miss the **quantitative** part (interior dose, post-intervention
re-reasoning). That is precisely the expert-vs-agent gap we set out to expose,
and it is legible per-turn in the trace (coarse-grid sweep, no refinement) — not
an execution artifact.

### Not yet done

- Port `scm.py`/`oracle.py` into the shared engine as archetype
  `scm_mechanism_chain` (additive; see design §9). The prototype is intentionally
  standalone until the trial confirms the difficulty is right.
- A third+ domain, and a hidden-subtype variant (dose depends on a latent
  regime → conditional-policy answer).
- Belief-convergence probe (compare the agent's evidence trajectory to an ideal
  updater).

---

## Part B — Worked example (the bioreactor world)

This is the flagship world the agent will see. I give the full ground truth
first (which the agent never sees), then exactly what the agent sees, why it is
hard, how a human expert solves it, and the specific ways an agent goes wrong.

### B.1 The true structure (hidden from the agent)

A 3-hop hidden chain from a knob to the outcome, plus a confounder that
manufactures a misleading correlation:

```
FeedWaterFlow ─┐                                     BatchSeedAge (hidden confounder)
               │ (interaction with corrosion)         │        │
FittingCorrosionSeverity (hidden, per-unit) ──────────┘        │ (small)
               ▼                                               ▼
        DissolvedCopper ─► ROS ─► CarbonFlux ─────────────► ProductYield  (OUTCOME)
         (hidden)      (hidden) (hidden)   ▲                    ▲
                          │                │ (+ direct feed nutrients)     BatchSeedAge also drives ↓
                          ▼                                          DissolvedOxygenReading (observable DECOY)
                 BrothProteinTurbidity (observable TRUE proxy, 2 hops from root)

Knobs (neutral names; the labels are shuffled per world so these are illustrative):
  RegimenC = chelator : scales DissolvedCopper down by (1 − sat(dose)); over-dose strips a nutrient ⇒ yield falls  → INTERIOR OPTIMUM
  RegimenD = nutrient bolus : biases the MEASURED ProductYield up without changing the true state → SYMPTOM TRAP
  FeedWaterFlow : helps a little (nutrients) but drives copper leaching once the fitting is corroded → SIGN FLIP
  TemperatureSetpoint, pHSetpoint : near-inert decoys
  AssayProbe : diagnostic
```

Ground-truth answer (computed, from `oracle`): gold intervention =
**chelator ≈ 66/100** (the interior optimum; util rises ~24 → ~72). The true
mechanism proxy is **BrothProteinTurbidity**; the confounded decoy is
**DissolvedOxygenReading**; knob signs toward "better": chelator `+`, feed-flow
`-`, temperature/pH `0`, nutrient bolus `0` (trap — moves the reading only).

### B.2 What the agent sees (partial projection)

- **Story:** yields fell after a shutdown in which a feed-water fitting was
  replaced; operators suspect dissolved-oxygen drift or a temperature excursion;
  broth is cloudier; feed/antifoam unchanged.
- **Observables (neutral, noisy):** `ProductYield`, `DissolvedOxygenReading`,
  `TemperatureReading`, `FoamHeight`, `BrothProteinTurbidity`. The hidden chain
  (`DissolvedCopper`, `ROS`, `CarbonFlux`, `BatchSeedAge`, corrosion) is **not**
  listed anywhere.
- **Knobs (neutral):** the Regimen* set + FeedWaterFlow + Temperature/pH
  setpoints + AssayProbe. **No knob is named "chelator" or "copper."**
- **Budget:** 12 queries; can observe, intervene, sweep a dose curve, or clamp a
  clampable reading (here `DissolvedOxygenReading`).
- **Question:** name the hidden cause, cite evidence, rule out alternatives,
  give the decisive test, recommend the intervention *and dose*, and fill the
  structured prediction block.

### B.3 Why it is hard

1. **The obvious signal is a confound.** `BatchSeedAge` drives both
   `DissolvedOxygenReading` and (a little) yield, so observationally DO tracks
   yield with correlation ~0.47 — but forcing DO does nothing. The agent must
   *distrust a strong correlation* and test it.
2. **The true cause is several hops up and unobserved.** The only honest clue is
   `BrothProteinTurbidity`, two hops downstream of the root and only ~0.49
   correlated with yield (calibrated to sit in-band: visible but not decisive).
   Nothing is labeled copper/chelator; the agent must infer an unobserved
   toxin from the pattern of interventional responses.
3. **A counterintuitive sign flip.** "More feed = more product" is false here:
   raising `FeedWaterFlow` *lowers* yield once the corroded fitting leaches
   copper. An agent anchored on nutrient-limitation reasoning gets the sign
   backwards.
4. **An interior optimum.** The fix is a *dose*, not a switch: chelator helps up
   to ~66/100, then over-stripping a nutrient hurts. "Max the good knob" is
   wrong.
5. **A symptom trap.** The nutrient-bolus knob raises the *measured* yield while
   the true state and the mechanism proxy are unchanged. An agent that watches
   only the outcome reading — not the proxy — will "fix" the problem cosmetically
   and be graded wrong (its recommended intervention has low true utility, and
   its knob-sign prediction for the trap is wrong).

### B.4 How a human expert solves it (ideal trajectory, ~4–6 queries)

1. **Observe.** Sample `ProductYield`, `DissolvedOxygenReading`,
   `BrothProteinTurbidity`, one more. Notice DO tracks yield (the tempting
   story) *but* turbidity is elevated and also tracks low yield — inconsistent
   with a pure oxygen-starvation account.
2. **Break the confound with a clamp.** `clamp DissolvedOxygenReading` at 20 vs
   80, watch `ProductYield`. It barely moves ⇒ DO is a confounded decoy, not the
   cause. (This is the single most diagnostic move, and the one weak agents
   skip.)
3. **Rule out the named suspects.** Brief interventional checks on Temperature /
   pH ⇒ near-zero effect.
4. **Follow the mechanism clue + story.** Turbidity ⇒ cell stress/lysis, not O₂
   limitation; the story flags the replaced feed-water fitting. Hypothesize a
   feed-borne toxin.
5. **Discriminating sweep.** `sweep FeedWaterFlow` ⇒ yield *falls* as flow rises
   (sign flip) — consistent with feed-borne toxicity, not nutrient limitation.
6. **Decisive test + dose.** `sweep` the chelator regimen ⇒ yield recovers and
   turbidity drops together (pins a metal toxin bound by chelation), and the
   curve **peaks at an interior dose (~66) then declines** ⇒ report that dose.
7. **Reject the trap.** `sweep` the nutrient-bolus regimen ⇒ measured yield rises
   but turbidity does not improve ⇒ symptom masking, not a fix.
8. **Answer:** hidden cause = feed-borne metal contaminant from the new fitting
   driving oxidative cell stress; DO correlation is confounded by batch/seed
   age; decisive test = chelation restores yield and lowers turbidity;
   recommend chelator ≈ standard dose; avoid high feed flow and the bolus.

Every step maps to a battery item, so an expert-quality trajectory scores
part A (right dose) and part B (right proxy, right decoys, right signs) → accept.

### B.5 Where the agent is expected to go wrong (the gap we want to surface)

These are *reasoning* failures, distinct from parse/exec noise, and each is
visible in the trace + grade:

- **Confound not broken.** Agent trusts the DO↔yield correlation, never clamps,
  concludes "oxygen control." → part B `true_mechanism_proxy`/`confounded_decoys`
  wrong; `clamp` never appears in the trace.
- **Stops too early.** Answers after 2–3 observational queries with an untested
  hypothesis (the v4 "avg 3.6 turns" smell). → `queries_used` low, no
  interventional/clamp turn, part A often wrong.
- **Sign-flip missed.** Recommends *raising* feed flow (nutrient intuition). →
  `knob_sign_predictions[FeedWaterFlow]` wrong and low-utility recommendation.
- **Max-dose error.** Finds chelator helps, recommends the maximum. → part A
  fails by the over-strip penalty (utility below gold-minus-tolerance).
- **Symptom trap.** Optimizes the measured outcome via the bolus knob without
  checking the proxy. → recommendation has low true utility; trap knob sign
  wrong.
- **Right story, wrong structured block.** Free-text names the cause but the
  structured predictions are sloppy. → part B < 80%. (This is the case v4 could
  not distinguish; v5 does, and the trace shows which specific predictions
  failed.)

The aim of the trial is precisely to see these buckets populate. If Opus solves
all six cleanly, we raise difficulty (deeper chain, stronger confound, more
decoys). If it fails on *execution* rather than *reasoning*, that's a harness
bug to fix — the trace tells us which.
