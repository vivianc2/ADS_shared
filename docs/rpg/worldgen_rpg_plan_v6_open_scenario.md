# RPG v6 — Large Open-Scenario Worlds, Actuators-Only, No Menu

> Supersedes the *interaction model* of v5 (`worldgen_rpg_plan_v5_scm_chain.md`).
> Keeps v5's declarative SCM math, computed golden answer, and counterfactual
> battery **verbatim** — those already derive from the equations. v6 changes
> what the agent is handed and how it acts.

## 0. Why v6 (the leakage lesson, restated)

The v5 batch (Opus 4.8, 6 worlds) showed a clean result: Opus reconstructed
every causal structure (part B = 6/6) but grid-searched the fix knob once and
never refined the dose (part A = 3/6, split only by tolerance). The deeper
finding: **the world was too small.** With a 12-query budget and only 6 knobs,
*brute force is the rational strategy* — sweep every knob, keep the winner. The
agent never had to reason about **where to look**, so it never showed scientific
vision. Meaningful names would have made this worse (read the label, skip even
the sweep). That is the v3→v4 leakage problem in a new guise.

**The fix is scale, not hiding.** In a world with ~20–30 intervenable variables
and a budget of ~15, brute force is impossible. The agent *must* use world
knowledge to prioritize: "cloudy broth + a replaced fitting + the oxygen theory
already failing → this smells like a feed-borne contaminant → probe the feed side
first." The **choice of what to probe becomes the measurement of scientific
reasoning** — the thing small neutral worlds amputated.

> Meaningful names stop being a leak once the world is large enough that knowing
> *what a variable is* does not tell you *whether it matters*. Reasoning is
> required to connect world-knowledge → hypothesis → the decisive experiment.

## 1. The two design changes

### 1.1 No menu — a detailed neutral scenario; propose-and-ask

The agent is **not** handed `["RegimenA"…"RegimenF"]`. It gets a long, concrete,
neutral prose scenario — meaningful domain terms (feed-water flow, broth
turbidity, dissolved oxygen, the replaced fitting, the neighbor line, ambient
conditions), but **no mechanism and no action list**. Variables are simply
*mentioned in the narrative*, as a real scientist encounters them.

The agent then, in its own words:
- **asks to measure** something ("what's the dissolved-metal content of the feed
  water?"), or
- **proposes an intervention** ("add a metal-chelating agent to the feed line at
  a moderate dose"), possibly **combining several** at once.

The server **resolves** each request against the hidden world:
- request maps to a measurable variable → return a noisy assay reading;
- request maps to an available actuator → apply it, return post-intervention
  readings;
- variable exists but has no assay/actuator → reject with a plausible reason
  ("no direct actuator for intracellular oxidative state on this rig");
- variable not in the world at all → reject plausibly ("that equipment is not
  present on this line").

The yes/no of "can I do this?" is itself information the agent must *reason* to
earn, and **what it chooses to ask about is a direct, logged readout of its
hypotheses** — far more diagnostic than watching it sweep a fixed list.

### 1.2 Actuators only (no do() by fiat)

The agent can act **only through available real-world actuators**, never set a
causal variable directly. Deep causes are reachable only *indirectly*, if an
actuator exists:
- `DissolvedCopper` (hidden) cannot be set — but a **chelating-agent actuator**
  scales it down. Finding that lever is the scientific act.
- `ReactiveOxygenSpecies`, `CarbonFlux`, `FittingCorrosion`, `BatchSeedAge` have
  **no** actuator — genuinely uncontrollable, like reality.
- `FeedWaterFlow`, `Temperature`, `pH`, `DissolvedOxygen` have **set-type**
  controllers (a flow controller, a DO controller → this is how "clamp" is
  realized: you must discover there is an O₂ controller).

This dissolves v5's three-way `knob/observable/latent` split into two realistic
per-variable questions: *is there an assay to measure it?* and *is there an
actuator to intervene on it?* Hidden variables are those with neither.

### 1.3 Combinations → combinatorial action space

The agent may apply **several actuators jointly** (e.g. reduce feed flow **and**
add chelator). With ~15 actuators and joint interventions up to size 3, the
effective action space is enormous, and — by design — the **best answer is often
a combination** (remove the source *and* clear the existing contaminant). This is
what the user asked for and what makes single-knob brute force hopeless.

## 2. World scale (starting point; tune later)

Flagship world target (we will experiment and adjust):
- **~22 variables total:** 1 outcome; ~6 in the true causal chain (hidden +
  observable); ~2 confounders; ~13 **in-world, plausible, mostly measurable,
  some intervenable, but causally inert distractors** (agitation, antifoam, CO₂,
  glucose feed, osmolality, viable-cell-density, harvest volume, …).
- **~14 actuators**, of which only a few touch the true chain; the rest are inert
  controls that a naive agent will waste budget on.
- **Budget ~15 experiments**, joint interventions up to 3 actuators.

So variables ≫ budget, and actuators ≫ budget: prioritization by reasoning is
forced.

## 2b. Final action design (five orthogonal verbs + code)

After building the open-scenario loop, the settled action set is deliberately
small and orthogonal — not a big menu (which leaks) and not too few (which forces
awkward channels). The agent has exactly:

1. **`measure`** — free-text quantity requests → assays; returns raw per-unit
   rows (CSV) + a quick summary. Costs 1 experiment.
2. **`intervene`** — free-text actuator requests + doses, applied jointly →
   raw rows + readings. Costs 1 experiment.
3. **`code`** — sandboxed pandas/numpy/scipy over the accumulated experiment
   CSVs. Does **not** cost the experiment budget (it is analysis, not data
   collection). This is what closes the quantitative gap seen in v5 (Opus
   eyeballed means and missed the interior dose). Matches ACED's headline
   `coder_new` condition.
4. **`answer`** — structured conclusion, computed-graded (part A utility ∧
   part B battery).
5. **`give_up`**.

Things deliberately NOT added as separate verbs, and why:
- *sweep a dose curve* → just `intervene` at several levels + `code` to fit the
  curve. More realistic than a bespoke primitive.
- *clamp a variable* → a `set`-type actuator already IS a clamp.
- *ask about the apparatus* → invites the resolver to leak mechanism; the
  scenario prose is the only briefing.

Raw data grain: each `measure`/`intervene` writes `experiment_<id>.csv` of
per-unit rows (the requested columns + `do_<actuator>` columns), exposed to the
code tool as a variable `experiment_<id>_csv` holding the path. The sandbox is a
spawned subprocess (clean interpreter, single-thread BLAS, hard timeout, stdout
captured, small vars carried forward) — the compact analog of
`framework_code/scientist_coder_agent_new.py`. The agent never sees the SCM,
gold, or ground truth — only the CSVs of what it chose to measure.

The loop the agent is nudged toward: **design experiment → collect rows →
analyze with code → decide next experiment** — the real scientific cycle.

## 3. Architecture (what changes vs v5)

The SCM math (`engine.py`, mechanism library), the computed golden answer, and
the counterfactual battery are **carried over** from v5. New/changed pieces:

1. **`engine.py`** — variables carry `aliases`, `measurable` + `assay_noise`;
   **actuators are first-class objects** `{id, aliases, target, op∈{set,scale,add},
   dose spec, expr, description}`. `set` unifies "knob" and "clamp." Evaluation is
   the v5 topological pass plus actuator application + descendant re-propagation.
2. **`worlds_v6.py`** — the large bioreactor world (and later others) as a
   variables+actuators spec.
3. **`oracle_v6.py`** — golden search now over **actuator combinations**. To stay
   tractable it (a) screens each actuator's marginal effect, (b) keeps the
   "active" set (|effect| > ε), (c) searches combinations *only* among active
   actuators with golden-section refinement on continuous doses. Pruning inert
   distractors is provably safe: a distractor outside the outcome's ancestry with
   no actuator-effect on that ancestry cannot change the outcome, alone or in
   combination. An audit verifies this.
4. **`resolver.py`** — maps agent free-text → a canonical assay or actuator.
   Server-side (sees the full hidden catalog; the agent never does). Alias/keyword
   match first; LLM disambiguation fallback given the catalog. **Echoes its
   interpretation** ("I understood this as: measure dissolved metals in feed
   water — proceeding.") so a resolution miss is visible and correctable, never a
   silent wrong-mapping. Every resolution is logged.
5. **`sim_v6.py`** — runtime: `measure` / `intervene` / `answer`, driven through
   the resolver.
6. **`run_agent_v6.py`** — LLM loop with the free-text propose-and-ask protocol.

## 4. Grading (unchanged in spirit; extended for combos)

Acceptance = (A) utility-optimal AND (B) counterfactual battery ≥ 80%, both
computed from the SCM.
- **(A)** now compares the agent's *joint* intervention's expected utility to the
  best combination the oracle found. Continuous doses refined by golden-section.
- **(B)** unchanged: which measurable variable is the true mechanism proxy, which
  measured signals are confounded decoys, and the sign of each *actuator's* real
  effect on the outcome. The decoy item credits any causally-inert signal the
  agent flags (fixed in v5 grader), requiring only that it flags the true
  confounder(s) and does not mislabel the true proxy.

## 5. The resolver risk (meaningful-failure discipline)

The resolver is where "reasoning failure" must not be confused with "silly
execution failure." Mitigations, all carried from ACED's proven parser design:
- rich per-node/actuator alias tables;
- LLM resolver that **echoes its interpretation** and asks the agent to confirm
  on ambiguity;
- full logging of (request → interpretation → resolved id | rejection reason),
  so post-hoc we always separate a *reasoning* miss (probed the wrong thing) from
  a *resolution* miss (we mapped it wrong).

## 5b. Counterintuitiveness (the obvious first move must fail)

A world is only interesting if the *surface* reasoning is wrong. Each world's
ground truth carries ``naive_interventions``: the actuator settings an operator
would try FIRST from the story's stated theories (e.g. "the room reads hot →
increase cooling"). A **counterintuitiveness audit** requires that every naive
move does NOT meaningfully help — ideally it *hurts*. If any naive move recovers
most of the gold's benefit, the obvious reasoning basically works and the world
is rejected.

Two flavors, both used:
- **Inert-naive** (bioreactor): the obvious suspects (oxygen, temperature) have
  ~0 effect; the agent must look elsewhere. Naive gain ≈ 0.
- **Backwards-naive** (datacenter): the obvious move is the *opposite* of
  correct. A rack sensor reads hot, so everyone wants MORE cooling — but an
  over-aggressive cooling setpoint drove the coil below the dew point, so more
  cooling makes throughput WORSE (measured naive gain ≈ −6). The fix is to warm
  the supply air / dehumidify. This is the strongest form of counterintuitive.

The audit is a first-class acceptance gate alongside the faithfulness checks.

## 5c. Template diversity — topologies first, seeds second

A batch generated as *one template × many seeds* varies mechanism parameters,
noise, and labels — but holds the **causal graph topology constant**. Since the
hard part of an RPG world is the *structure* (how many hops, which role is the
confound vs. the proxy vs. the trap, whether there is a sign-flip), jittered
clones of a template test essentially the **same insight** repeatedly. That
measures *reliability*, not *breadth of scientific reasoning*.

Therefore the axis of scale is **distinct topologies first, seed-jitters
second**. Each topology is a different reasoning challenge:

| Topology | Reasoning it demands | Status |
|---|---|---|
| Mediation chain (bioreactor) | confound vs. multi-hop cause; interior dose | ✅ built |
| Backwards sign-flip (datacenter) | the obvious move is the wrong *direction* | ✅ built |
| Two interacting causes (greenhouse) | neither cause sufficient alone; fix needs BOTH (iron+pH AND-gate) | ✅ built |
| Hidden subtype / heterogeneity (clinic) | a lever helps one latent subgroup, harms another; avg effect ~0; interior dose | ✅ built |
| Collider / selection | conditioning on a common effect creates spurious links | deferred (needs selection/conditioning in engine) |
| Feedback / masking loop | a control loop hides the driver until you break it | planned |

Engine additions for these: `gated_and` (product of two sigmoids — true "both
required" AND-gate) and `abs` (distance-from-optimum). Oracle: `screen_actuators`
now has a **synergy pass** that catches pairs where neither actuator helps alone
but together they do (without it, marginal-only screening prunes both and never
finds the gold combo). `proxy_signal_audit` now accepts a proxy that is
informative *interventionally* (dormant-at-baseline mechanisms like the AND-gate
have ~0 observational correlation — the agent must intervene to see the signal).

Note: hidden-subtype's *ideal* answer is a conditional policy (treat one subgroup,
spare the other); the current single-intervention grader can't express that, so
the clinic world is tuned so the population-optimal is an interior dose. A future
conditional-policy answer schema would let it score the stronger stratified fix.

The v6 pilot batch (bioreactor + datacenter × seeds) validates the pipeline
end-to-end and already gives two genuinely different structures. It is a
**pilot, not the benchmark** — the benchmark target is ~5–6 structurally distinct
topologies × a few seeds each (seeds for reliability, topologies for breadth).

## 6. Faithfulness / difficulty audits (extend v5)

- name-leakage (now on the *scenario prose* + variable names, not a menu);
- decoy inertness (confounders correlate but have zero do-effect);
- proxy-signal band (true proxy correlation in-band);
- **distractor inertness at scale** — every distractor actuator has ≈0 marginal
  and no interaction with the outcome (justifies oracle pruning);
- **gold self-consistency** — the reported gold is the argmax over the searched
  actuator-combination space;
- **solvability + minimal decisive set** — an oracle-informed strategy identifies
  the cause and the right intervention within budget; record the minimal set as
  the efficiency yardstick (now including "did the agent avoid wasting budget on
  inert distractors?").

## 7. Build order

1. `engine.py` + large `worlds_v6.py` + `oracle_v6.py` + a **scripted expert
   demo** (no LLM) proving: faithful SCM, actuators-only, combinations matter,
   solvable within budget despite ~22 variables. ← this milestone first
2. `resolver.py` with an offline alias core + tests.
3. `sim_v6.py` + `run_agent_v6.py` (free-text protocol), mock-backed smoke, then
   live Opus 4.8.

## 8. What we expect to see (the hypothesis this tests)

If the world is genuinely large and menu-free, the accept/reject will no longer
be a tolerance artifact. We expect to see the agent's **prioritization** vary in
quality: strong runs go straight to the feed side from the story; weak runs
squander budget probing inert distractors (agitation, antifoam) or never find the
chelation lever at all. That distribution — *where the agent spends its limited
experiments* — is the expert-vs-agent gap we could not see in v5.
