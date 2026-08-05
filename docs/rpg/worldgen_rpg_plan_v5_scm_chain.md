# RPG v5 — Declarative SCM Worlds, Multi-Hop Latent Discovery, Computed Golden Answers

> Supersedes the difficulty and grading direction of
> `worldgen_rpg_plan_v4_complex_neutral_dose.md`. It does **not** delete the v3
> `story_hidden_cause_discovery` archetype or its engine — v5 is **additive**: a
> new archetype (`scm_mechanism_chain`) backed by a generic SCM evaluator, plus
> new query modes and a computed (non-string-matched) grader. The v3/v4 role
> contract, schema shape, and live engine in `world_gen_rpg_old.py` stay intact,
> per `CLAUDE.md`.

## 0. Why v5 (what v3/v4 could not do)

Reading the live engine (`world_gen_rpg_old.py::_static_hidden_cause_apply`,
`_static_hidden_cause_observe`) and the grader
(`framework_code/simulator_rpg.py::_score_latent_cause_answer`, lines ~490–548)
exposes four structural limits. v5 targets exactly these:

1. **The graph is a star, not a chain.** Today there is one hidden driver
   (`LatentBurden`) → outcome, plus decoys that are *observationally correlated
   but have zero do-effect*. There are no intermediate mechanism nodes the agent
   must reconstruct — no "several hops." The task is single-hop confounding
   dressed up as discovery.
2. **Experiments are not mechanistically meaningful.** The mechanism is
   `if knob=="on": target += const`, generalized in v4 to a dose fraction
   `d = index/(len-1)` with a saturation/overtreatment kink. There is no
   dynamics, no genuine dose-response *curve* to trace, no ability to **clamp an
   intermediate** to break a confound. The agent samples and reads
   `latent + Gaussian` proxies.
3. **The identification grade is string-matching, not math.** Acceptance for
   `latent_cause_hypothesis` is keyword/alias family matching (`"leaf"`,
   `"downspout"`, `"gutter"`…). This is the most likely cause of the v4
   "0/8, surface-proxy" mystery in `PROJECT_STATUS.md §2`: a correct-but-
   differently-worded answer can fail, and we cannot tell "hard" from "broken."
4. **The observation model is trivial:** every observable is `latent + noise`.
   No colliders, no selection, no measurement that requires *combining* signals.

The redesign has four jobs, and every section below serves one: **(a)** a
multi-hop, partly counterintuitive SCM; **(b)** an experiment interface where
interventions do something rich and continuous; **(c)** a golden answer computed
from the SCM, not matched by string; **(d)** faithful partial observation.

## 1. Target scenario (the north star)

A **bioreactor yield-collapse** world (one of several domains in §6; kept as the
running example because it is continuous by nature and naturally counterintuitive).

### 1.1 The true world (defined; agent never sees it)

```
              (hidden root chain)                              (hidden)
FeedWaterFlow ─► DissolvedCopper ─► ReactiveOxygenSpecies ─► CarbonFluxToProduct⁻ ─► ProductYield  (OUTCOME)
   (knob)            │  (Hill, saturating)                                 ▲
                     │                                                     │ (small direct confound path)
BatchSeedAge ────────┼──────────────────────────────► DissolvedOxygenReading (observable DECOY)
 (hidden context)    └──────────────────────────────► ProductYield (small)
                     │
                     └─► ReactiveOxygenSpecies ─► CellLysis ─► BrothProteinTurbidity (observable TRUE proxy, ≥2 hops)

ChelatorDose (knob) : multiplies DissolvedCopper by (1 − sat(d)); at high d strips a TraceNutrient ⇒ ProductYield⁻   (interior optimum)
NutrientBolus (knob): transient ProductYield boost that decays; does NOT touch copper                                 (SYMPTOM trap)
Temperature, pH (knobs): tiny real effect                                                                             (near-inert decoys)
AssayProbe (binary knob): reveals FreeMetalAssaySignal, informative only when run                                     (diagnostic)
```

Key structural features (each is a *difficulty lever*, §5):
- **Multi-hop:** root `DissolvedCopper` is 3 hops from `ProductYield`; the only
  *true* observable clue (`BrothProteinTurbidity`) sits 2 hops downstream of the
  root — identifying the root requires reasoning *through* the chain.
- **Counterintuitive sign flip:** `FeedWaterFlow` helps at low levels (nutrients)
  but harms once the corroded fitting leaches copper — "more feed is better"
  fails.
- **Confound:** `BatchSeedAge` independently lowers both `DissolvedOxygen` and
  yield ⇒ a strong observational DO↔yield correlation with **zero** causal effect
  of DO. This is the trap the operators (and naive agents) fall into.
- **Interior optimum + symptom trap:** chelator has a best *dose*, not "max";
  nutrient bolus lowers the visible outcome without touching the mechanism.

### 1.2 What the agent sees (partial projection)

- **Story** (neutral; plants clues, never names copper/chelation): yields fell
  after a shutdown; a feed-water fitting was replaced; broth is cloudier;
  operators suspect DO drift or temperature.
- **Observables** (noisy, neutral): `ProductYield`, `DissolvedOxygenReading`,
  `TemperatureReading`, `pHReading`, `BrothProteinTurbidity` (true proxy),
  `AgitationPower`, `FoamHeight` (decoys). `DissolvedCopper`, `ROS`,
  `CarbonFlux`, `BatchSeedAge` are **not observable**.
- **Knobs** (neutral, continuous where sensible): `FeedWaterFlow` (cont.),
  `TemperatureSetpoint` (cont.), `pHSetpoint` (cont.), `RegimenC`=chelator
  (cont.), `RegimenD`=nutrient bolus (cont., trap), `AssayProbe` (binary).
- **Question:** explain the hidden cause in ordinary language, cite queried
  evidence, rule out alternatives, name a decisive test, recommend the
  intervention **and its dose**. Plus a new **structured prediction block** (§4B).

### 1.3 The ideal expert trajectory

1. Observational baseline → sees DO↔yield correlation (the obvious hypothesis)
   but also elevated `BrothProteinTurbidity` that does not fit an O₂-starvation
   story.
2. Rule out the decoys: interventional sweeps of Temperature/pH move yield
   negligibly → believed causes are near-inert.
3. Follow the mechanism clue: turbidity ⇒ cell stress/lysis, not O₂ limitation;
   story points at the feed-water fitting.
4. Discriminating experiment: sweep `FeedWaterFlow` → yield **falls** as flow
   rises (sign flip) ⇒ feed-borne toxicity, not nutrient limitation.
5. Decisive test: sweep `RegimenC` (chelator) → yield recovers, turbidity drops
   ⇒ a metal toxin bound by chelation. Optionally run `AssayProbe`.
6. Dose optimization: yield peaks at an **interior** chelator dose (over-strip at
   high) ⇒ report the right dose, not "max."
7. Reject the trap: `RegimenD` (bolus) → yield jumps then decays, turbidity
   stays high ⇒ symptom masking.
8. Answer: hidden cause = feed-borne metal contaminant driving oxidative stress;
   DO correlation is confounded by batch/seed age; decisive test = chelation;
   recommend `RegimenC ≈ standard`, avoid high `FeedWaterFlow` and `RegimenD`.

Every step is an experiment the current engine cannot support. That is the gap.

## 2. World representation — a typed, declarative SCM

Replace per-archetype hardcoded mechanisms with an **SCM spec the engine
interprets**. A world is a DAG of typed nodes; the engine has one generic
topological evaluator. This single change makes faithfulness, the computed
golden answer, and new topologies fall out for free.

```jsonc
{
  "nodes": {
    "FeedWaterFlow":   {"kind":"knob","dtype":"continuous","range":[0,100],"default":40},
    "AssayProbe":      {"kind":"knob","dtype":"binary","values":["off","on"],"default":"off"},
    "BatchSeedAge":    {"kind":"latent","dist":{"normal":[50,15]}},
    "DissolvedCopper": {"kind":"latent","parents":["FeedWaterFlow"],
                        "mech":{"form":"sign_flip","of":"FeedWaterFlow","lo_gain":-0.3,"hi_gain":0.9,"knee":45},
                        "noise":{"normal":[0,2]}},
    "ROS":             {"kind":"latent","parents":["DissolvedCopper"],
                        "mech":{"form":"hill","of":"DissolvedCopper","vmax":60,"k":25,"n":2}},
    "CarbonFlux":      {"kind":"latent","parents":["ROS"],
                        "mech":{"form":"linear","weights":{"ROS":-0.9},"intercept":80}},
    "ProductYield":    {"kind":"outcome","parents":["CarbonFlux","BatchSeedAge"],
                        "mech":{"form":"linear","weights":{"CarbonFlux":1.0,"BatchSeedAge":-0.1},"intercept":10},
                        "obs_noise":{"normal":[0,3]}},
    "BrothProteinTurbidity":{"kind":"observable","parents":["ROS"],
                        "mech":{"form":"linear","weights":{"ROS":0.8},"intercept":5},
                        "obs_noise":{"normal":[0,5]}},
    "DissolvedOxygenReading":{"kind":"observable","parents":["BatchSeedAge"],
                        "mech":{"form":"linear","weights":{"BatchSeedAge":-0.7},"intercept":90},
                        "obs_noise":{"normal":[0,4]}}
  },
  "knob_effects": {
    "ChelatorDose": {"target":"DissolvedCopper","op":"scale","by":"1-sat(d;k=0.66)",
                     "side_effect":{"target":"ProductYield","op":"add","expr":"-overstrip(d;thr=0.66,gain=25)"}},
    "NutrientBolus":{"target":"ProductYield","op":"add","expr":"transient_boost(d)"}
  },
  "outcome":"ProductYield",
  "higher_is_better": true
}
```

- **Node kinds:** `knob` (agent-settable), `latent` (hidden), `observable`
  (measurable = mechanism value + `obs_noise`), `outcome`.
- **Mechanism library** (small, closed, pure, vectorized — auditable and
  analytically tractable): `linear`, `saturating`, `hill`, `soft_threshold`,
  `interaction` (product of two parents), `sign_flip` (monotone-then-reversing).
  New worlds are authored by picking forms + params, not by writing Python.
- **`knob_effects`** are structural interventions expressed as ops on target
  nodes: `scale`/`add`/`set`, with dose `d ∈ [0,1]` computed exactly as v4
  (`d = index/(len-1)` for categorical, `clip((v-min)/(max-min))` for continuous).
  This keeps binary/dose backward-compatible and makes continuous first-class.

The engine's `apply`/`observe` collapse to one generic function
(`_scm_apply`, `_scm_observe`, `_scm_sample`) that (a) samples exogenous latents,
(b) applies `knob_effects` from the intervention, (c) evaluates nodes in
topological order, (d) adds `obs_noise` for requested observables. Because the
mechanism *is* data, the evaluator is trivially faithful to the definition.

## 3. Meaningful experiments — extend the query interface

Current modes: `observational_sample`, `interventional_sample`, `inspect_unit`.
Add two:

1. **`clamp`** — a `do(.)` on an *intermediate observable* (where the story
   permits): hold DO fixed while varying temperature, i.e. the operation that
   *breaks a confound*. Each node carries `clampable: true/false`; the story and
   physics decide which are clampable. Internally this is `_scm_apply` with the
   clamped node forced to a constant instead of computed from parents.
2. **`sweep`** — a knob + grid → per-level outcome/proxy means with standard
   errors in one query, so tracing a dose-response curve is one legible query,
   not five. Internally batched `interventional_sample`; exposing it makes the
   experiment cheap and lets us log "did the agent trace the curve."

Budget stays cell-based (already implemented in `scientist_agent_rpg.py` /
`experiment_budget`), so richer queries cost more. `simulator_rpg.py::validate_*`
gains the two modes; `world_model_rpg.py` passes values through verbatim (no
change).

## 4. The golden answer — computed, not matched

Two independently-checkable parts, both derived from the SCM. **Acceptance =
(A) AND (B).**

### 4A. Optimal intervention + dose (exact)

Enumerate candidate knob-settings over grids; score expected utility by
common-random-number Monte Carlo (existing `_static_oracle_score` pattern). For
continuous knobs, add a **refinement pass**: coarse grid → golden-section search
around the peak, so the interior optimum is located precisely. Grade the agent's
dose by `expected_utility ≥ gold − tolerance` (tolerance in outcome units). This
path already exists (`_score_intervention_answer`) and stays; we only add the
continuous refinement.

### 4B. Latent-cause identification — a counterfactual battery (replaces string match)

Because we own the SCM, "understanding the cause" = "can predict the world's
response to interventions the agent has not run." At generation time precompute a
**held-out counterfactual battery** with ground-truth answers from the evaluator,
e.g.:

- sign (and coarse magnitude bucket) of `dProductYield/dFeedWaterFlow` → negative
  (the sign flip);
- `do(ChelatorDose=standard)` vs `do(NutrientBolus=high)` → recover vs mask;
- effect of `clamp(DissolvedOxygen)` on yield → ≈ 0 (confound, not cause);
- which observable is the true mechanism proxy vs. a confounded decoy.

Extend the answer JSON with a **structured prediction block**:

```jsonc
"structured": {
  "true_mechanism_proxy": "BrothProteinTurbidity",
  "confounded_decoys": ["DissolvedOxygenReading"],
  "intervention_sign_predictions": {
    "FeedWaterFlow": "-", "ChelatorDose": "+", "NutrientBolus": "0_or_transient"
  }
}
```

Grade by exact/interval match against the battery (a threshold, e.g. ≥ 80% of
items correct). Keep the free-text explanation for human inspection and an
**optional** LLM-judge, but acceptance is driven by 4A + battery, not keywords.
This resolves the v4 ambiguity: a surface-proxy answer gets the interventional
predictions wrong and fails; a correctly-reasoned but differently-worded answer
passes.

**Optional causal-structure score:** ask the agent to submit believed edges among
named observables + "unobserved cause" slots; grade precision/recall against the
true DAG projection. Directly measures multi-hop reasoning.

## 5. Difficulty knobs (dial "hard and interesting", then validate)

Generator parameterizes and each world is validated to land where intended:

- **chain depth** (root → outcome hops);
- **confound strength** (how convincing the decoy correlation is);
- **sign-flip / interaction order** present or not;
- **number of plausible-but-wrong hypotheses** (decoy count);
- **discriminability gap** — the minimum number of well-chosen interventions to
  separate the true cause from the best decoy, computed as a **solvability
  certificate** (§7.4). Worlds unsolvable within budget, or solvable in one
  query, are rejected.

## 6. Domain library (multi-domain by default)

The generator is domain-agnostic (the SCM is neutral); domains are *skins*
providing story + neutral names. Recommended starter set — deliberately spanning
different intuitions so no single domain prior helps the agent:

1. **Bioreactor yield collapse** (industrial/process) — the running example;
   feed-borne metal toxin masquerading as DO drift.
2. **Research apiary night mass-loss** (ecology) — reuse of the existing world's
   flavor but as a true chain: neighbor-colony robbing pressure (hidden) →
   guard-engagement/lysis proxies → mass loss, with predator/mite decoys.
3. **Municipal water discoloration** (hydrology/infra) — pipe-scale release
   (hidden) → turbidity/metal proxies → complaints; rainfall/temperature decoys;
   flushing dose has an interior optimum.
4. **Semiconductor wafer-yield drop** (materials) — trace-moisture in a gas line
   (hidden) → particle-count proxy → yield; chamber-temperature decoy; purge-flow
   sign flip.
5. **Greenhouse crop stunting** (agronomy) — root-zone salinity accumulation
   (hidden) from fertigation → osmotic-stress proxy → growth; light/CO₂ decoys;
   leaching dose has interior optimum + over-leach nutrient loss.

Each domain instantiates the *same* topology family with different names and
mechanism params, so we can measure whether the agent reasons structurally or
pattern-matches a domain. **We build 2–3 first (bioreactor + one other), inspect,
then decide the final set.**

## 7. Generation pipeline (end to end)

1. **Sample a template** = topology + mechanism-family choices + difficulty
   params. Topologies come from a small hand-authored library of counterintuitive
   DAGs; LLM may propose *topologies* (validated against the schema), not prose.
2. **Instantiate parameters**; sweep `obs_noise` SDs so the observational
   correlation between the true proxy and outcome lands in a target band (0.4–0.6)
   — visible enough to notice, weak enough to force intervention. (Reuses the
   `latent_driver_proxy_sd` calibration idea already in the engine.)
3. **Compute golden answer:** optimal intervention via MC + golden-section (4A);
   counterfactual battery (4B); true DAG projection.
4. **Solvability certificate:** simulate an oracle querying agent (knows the SCM)
   to confirm the true cause is separable from every decoy within budget, and
   record the **minimal discriminating query set** (the efficiency yardstick).
   Reject unsolvable / one-query-trivial worlds.
5. **Anti-leakage + faithfulness audit:** name audit (regex over
   `clear|fix|stop|chelat|copper|…`), depth audit (true proxy ≥2 hops from root),
   correlation-band audit, and a **decoy audit** — each decoy must be
   observationally convincing yet causally inert (compare observational
   correlation vs. interventional effect; require |do-effect| ≈ 0).
6. **Emit** the world JSON in the existing `visible/hidden/oracle/validators`
   shape so downstream modules need only the generic-evaluator swap.

## 8. Logging & trace observability

Append-only JSONL per world-run, one record per turn:

- agent `reasoning`, chosen query, `scientist_memory` snapshot (accumulated
  knowledge — already emitted, just persist every turn);
- exact query dict, cells spent, running budget;
- **belief probe** (optional): after each query, evaluator records which
  enumerated candidate cause is currently best-supported by the data the agent
  has seen → plot evidence convergence vs. an ideal Bayesian updater;
- **efficiency metrics:** queries-to-first-correct-hypothesis, whether the
  decisive test was run, cells vs. the oracle minimal discriminating set (§7.4),
  count of redundant/uninformative queries.

This gives the "accumulates knowledge" and "solves efficiently like an expert"
views directly, and turns a v3-vs-v4 post-mortem into a query over logs.

## 9. Code changes (scoped; respects CLAUDE.md contracts)

Additive throughout — the v3 archetype and the live engine keep working.

- **`world_gen_rpg_old.py`**: add `_scm_sample`, `_scm_apply`, `_scm_observe`,
  the mechanism library, and register archetype `scm_mechanism_chain` in the
  dispatch tables (`_static_sample_hidden`/`_static_apply`/`_static_observe`/
  `_static_candidate_interventions`/`_static_utility_from_outcomes`). Do **not**
  delete existing functions (imported at module load).
- **`world_gen_rpg.py`**: register the new archetype + difficulty/domain flags;
  add the generation-time golden-answer, battery, solvability, and audit steps.
- **`simulator_rpg.py`**: add `sweep` + `clamp` modes to
  `validate_query`/`run_query`; add the battery grader for the new archetype
  (`_score_scm_latent_answer`) alongside the existing scorers; keep the utility
  path.
- **`schemas_rpg.py`**: expose continuous ranges + new modes to the agent; extend
  the answer schema with the `structured` block.
- **`evaluate_rpg.py`**: acceptance = 4A AND 4B; add efficiency/convergence
  metrics from the log.
- **`scientist_agent_rpg.py`**: teach the new modes + structured answer block;
  loop otherwise unchanged.

## 10. Vertical-slice prototype (built alongside this doc)

A standalone, dependency-light prototype under
`dataset_generation_code/rpg_v5_prototype/` that stands the whole idea up before
touching the shared engine:

- `scm.py` — the generic SCM evaluator + mechanism library.
- `worlds.py` — the bioreactor world (and 1–2 more domains) as SCM specs.
- `oracle.py` — golden intervention (MC + golden-section), counterfactual
  battery, solvability certificate, faithfulness audits.
- `demo_solve.py` — a scripted "expert" trajectory that hand-solves the
  bioreactor world through the query interface, printing each experiment and its
  result, ending with a graded answer.

Purpose: prove (a) the SCM is faithful, (b) the world is solvable by meaningful
experiments, (c) the golden answer grades a correct answer as correct and a
surface-proxy answer as wrong — the three things v4 could not demonstrate. Once
validated, port `scm.py`/`oracle.py` into the shared engine per §9.

## 11. Open questions / follow-ups

- Contextual (conditional) optimal dose when a hidden subtype needs different
  doses — bridges to `latent_regime_policy`.
- Sharding oracle/battery cost for larger neutral catalogs.
- Whether to expose `clamp` on latents (physically implausible) or only on
  observables (recommended: observables only, to keep interventions realistic).
