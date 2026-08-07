# RPG v6 — Design and Current Results
### Advisor update. Snapshot: 2026-08-05.

> One `## Slide N` per slide. Bullets = slide content. "Detail:" = specific
> technical points you may include or drop. "Notes:" = what to say. All numbers
> are from actual runs.

---

## Slide 1 — Title / framing

- **RPG: a benchmark for causal scientific reasoning under partial observability.**
- Task: an LLM agent is given a partially-observed system with a known
  ground-truth structural causal model (SCM). It must identify the hidden cause
  of an outcome and prescribe an intervention, using a bounded number of
  experiments. Scored against the SCM.
- This update: the v6 design, the four world families, the grading/audit
  machinery, and first Opus-4.8 results.

Notes: "Since v3 the whole apparatus changed. I'll start from the design
decisions and setup, then results."

---

## Slide 2 — Design goal and the three failure modes we designed against

- Goal: measure whether an agent can (i) form hypotheses about unobserved
  structure, (ii) design discriminating experiments, (iii) recover the causal
  mechanism and the utility-optimal action, under a query budget.
- Three ways such a benchmark silently fails, and our countermeasure:
  - **Leakage** — the agent reads the answer off variable/action names or the
    question. → neutral-but-meaningful naming + large action space (below).
  - **Ungradable answers** — fuzzy text grading can't separate correct from
    wrong. → answer graded against SCM-computed quantities (Slide 8).
  - **Unfairness** — task needs outside knowledge or is unsolvable. → per-world
    automated audits, including a solvability/reachability check (Slide 9).

Notes: "Most of the engineering is in removing these three confounds. I'll flag
each."

---

## Slide 3 — World representation: a typed SCM

- A world is a DAG of typed nodes plus a set of **actuators**.
- Node kinds: `latent` (no assay), `observable` (has assay + noise), `outcome`.
- Each non-root node computes from its parents via one **mechanism** from a fixed
  library (Slide 3b). Exogenous (root) nodes carry a distribution (normal/uniform)
  instead.
- Two distinct levels — do not conflate:
  - **Mechanism** = the functional form on a single node (the 8 functions below).
  - **World family / archetype** = the whole DAG topology + role assignment
    (which node is the confound / proxy / trap): bioreactor, datacenter,
    greenhouse, clinic (Slide 10). Each family is assembled from many
    mechanism-nodes.
- Detail: nodes carry `aliases` (for the resolver), `measurable`, `assay_noise`.
  Sampling is a topological pass; `measure()` adds assay noise to the requested
  observables only.

Notes: "Mechanisms are the building blocks; archetypes are the assembled worlds.
Everything the oracle and grader need is derived from these equations — there is
no second, hand-written answer key."

---

## Slide 3b — Mechanism library (per-node functional forms)

Every computed node declares `{"form": …, params}`; `x`/`a`/`b` are parent values,
elementwise over sampled units.

| Form | Formula | Shape / role | Used for (example) |
|---|---|---|---|
| `linear` | `intercept + Σ wᵢ·parentᵢ` | weighted sum; many parents | combine signals; outcomes |
| `saturating` | `gain·x/(x+k)` | diminishing returns, no peak | "more helps, tapering" |
| `hill` | `vmax·xⁿ/(kⁿ+xⁿ)` | sigmoid; `k`=half-max, `n`=steepness | soft threshold (copper→ROS) |
| `soft_threshold` | `gain·σ((x−θ)/w)` | logistic on/off at `θ`, width `w` | smooth switch |
| `interaction` | `intercept + gain·(a/s)·(b/s)` | bilinear; effect of a depends on b | flow × corrosion → copper |
| `sign_flip` | `intercept + lo·min(x,knee) + hi·max(x−knee,0)` | slope changes at `knee` (effect reverses) | more helps then hurts |
| `gated_and` | `intercept + vmax·σ((a−tₐ)/wₐ)·σ((b−t_b)/w_b)` | AND-gate: high only if BOTH inputs pass | two-required-causes (iron ∧ pH) |
| `abs` | `intercept + gain·abs(x − center)` | distance from an optimum (V-shape) | interior optimum / subgroup flip |

- σ = logistic. Exogenous nodes: `dist ∈ {normal[μ,σ], uniform[lo,hi]}`.
- Design point: each hard property in a world traces to a specific form —
  `hill`/`saturating` → thresholds/diminishing returns; `interaction`/`gated_and`
  → synergy; `sign_flip`/`abs` → reversals and interior optima.

Notes: "Small closed library → every world is exactly specified → the golden
answer and the audits are computed, not authored."

---

## Slide 4 — Interventions are actuators, not free do-operators

- **Definition:** an actuator is the only handle through which the agent can
  intervene — a named record applying one **typed operation** to a single SCM
  variable (`target`). The agent cannot set variables directly (no free
  `do(X:=x)`); effects propagate through the DAG.
- Record: `{id, aliases, target, op, dtype, range/values, default, expr,
  side_effect, description}`.
- The four operations:

| op | Effect on `target` | Touches true state? | Models |
|---|---|---|---|
| `set` | forces `target` to the submitted value (overrides its mechanism) | yes | a controller / clamp (hold DO at 80; setpoint=75) |
| `scale` | multiplies `target` by `expr(d)` | yes | a dosed treatment (chelator scales copper by 1−sat(d)) |
| `add` | adds `expr(d)` to `target` | yes | a dosed additive |
| `mask` | biases only the *measured reading* of `target` | **no** | symptom-masking control (improves the readout, not reality) |

- `set`/`scale`/`add` edit the structural value, then all descendants are
  re-evaluated. `mask` is applied only in `measure()`.
- Deep causes are reachable only indirectly: dissolved copper has no actuator; a
  chelating-additive `scale` actuator reduces it.

Detail: dose fraction `d∈[0,1]` = `(value−min)/(max−min)` for continuous,
`index/(len−1)` for discrete; `scale`/`add` effects and `side_effect` are
functions of `d`. `mask` is the mechanism behind the symptom-trap in each world.

Notes: "Typing (latent/observable/outcome) says whether the agent can SEE a
variable; the actuator set says whether it can CHANGE it, and how. Independent
axes — a hidden cause with no actuator must be reached via a downstream lever."

---

## Slide 4b — Each op in the agent's own logged requests

Verbatim agent requests from the runs (147 `set`, 41 `scale`, 1 `mask` across the
batch; **`add` is supported by the engine but no current world uses it**).

| op | Actual agent request (logged) | Resolved to | Why the op is used / what it exposes |
|---|---|---|---|
| `set` | "reduce the feed-water flow rate" → 0 | `feed_flow_controller` | clamp a controllable process variable; here it tests the sign-flip lever |
| `set` | "set the cooling plant setpoint to maximum cooling" → 75 | `cooling_setpoint_controller` | forces a setpoint; used to test the (backwards) cooling hypothesis |
| `scale` | "add a metal-chelating additive to bind leachable metals" → 60 | `chelator_dosing` | dosed reduction of a hidden cause (copper); the interior-optimum dose test |
| `mask` | "add the protein stabilizer additive" → 50 | `stabilizer_additive` | the symptom trap — see below |
| `add` | (none observed) | — | reserved; e.g. a supplement that adds to a target. Not instantiated in the 4 worlds. |

- **The `mask` case, verified:** after the stabilizer at 50, the *measured* titer
  rose 21.3 → 25.3 while the *true* titer (utility) stayed 21.3 and turbidity
  stayed high (43). It improves the readout, not the system — exactly the trap the
  agent must avoid recommending. (This agent tried it once as a test, then did not
  recommend it.)

Notes: "So three of the four ops appear in real traces. `set` and `scale` are the
workhorses; `mask` is deliberately rare — it's the trap, and I can show a run
where the reading moved but the truth didn't. `add` exists for completeness but
none of the four current worlds needed it — worth saying plainly rather than
implying all four are exercised."

---

## Slide 5 — Partial observability and the "no menu" protocol

- The agent receives only: a neutral prose **scenario**, the outcome name and its
  direction (higher/lower better), and its budget. **No variable list, no action
  list.**
- It issues free-text requests; a **resolver** maps each to a canonical
  observable or actuator, or returns a typed rejection (`no_assay`,
  `no_actuator`, `not_in_world`).
- Resolver = lexical scoring (stem match, distinctiveness weighting, request-
  coverage gate) with an **LLM disambiguation fallback** (default on) for the
  uncertain band. Every mapping is echoed to the agent and logged.

Detail: this replaces v3/v4's descriptive action menu, which leaked the answer.
Names are meaningful (so world knowledge is usable) but the mapping from concept
to which variable *matters* must be learned from data.

Notes: "The resolver is the one component that could turn a reasoning success
into a scored failure, so it's heavily guarded — I'll return to that."

---

## Slide 6 — Agent action space and budget

- Four actions, one per turn:
  - `measure(requests)` — returns per-unit rows (a CSV) + summary stats +
    pairwise correlations. Costs 1 query.
  - `intervene(actions, measure)` — applies actuators jointly (up to 3), returns
    post-intervention readings + rows. Costs 1 query.
  - `code(...)` — runs Python (pandas/numpy/scipy) over the collected CSVs in a
    sandboxed subprocess. **Does not cost budget.**
  - `answer(...)` — structured final answer; ends the run.
- Defaults: **budget = 15 queries, ≤ 32 turns, 400 units/query.**

Detail: the loop injects directives near budget/turn exhaustion (force an
answer), when the agent has run 0 interventions after several queries, and when a
request family is rejected ≥3×.

Notes: "The code tool matters — the agent can fit a dose-response or regress out
a confound, not just eyeball means."

---

## Slide 7 — Why the world is large (anti-brute-force)

- Sizes (variables / actuators): bioreactor **25 / 14**, datacenter 23 / 11,
  greenhouse 19 / 9, clinic 20 / 7. Most variables and actuators are causally
  inert.
- With budget 15, exhaustive search over actuators (and joint combinations) is
  infeasible, so the agent must prioritize from the scenario.
- Automated **distractor-inertness audit** verifies every non-active actuator has
  ≈0 marginal effect and ≈0 interaction with the outcome — this both justifies
  pruning them in the oracle and guarantees they are genuine distractors.

Notes: "In v5 the worlds were small enough to brute-force; scaling up is what
forces hypothesis-driven search."

---

## Slide 8 — Golden answer and grading (both computed from the SCM)

- **Gold intervention:** screen each actuator's marginal effect (+ a synergy pass
  for pairs that only help jointly), search combinations over the active set (up
  to 3), refine continuous doses by golden-section. Monte-Carlo utility, common
  random numbers.
- **Grade = A ∧ B:**
  - **A (utility):** re-simulate the agent's recommended intervention; require
    `E[utility] ≥ gold − tolerance` (tolerance = 2.0 outcome units, n=30k).
  - **B (structure):** the agent names the true mechanism proxy, the confounded
    decoys, and each actuator's sign; require **≥ 0.8** of items correct.
- Detail: proxy is credited against the *set* of valid proxies (any measurable
  the true lever moves ≥0.5 SD), not one string; decoy item requires flagging the
  true confounder and not mislabeling a real proxy.

Notes: "The score is arithmetic on the SCM. A right dose with a wrong mechanism
fails B; a right mechanism with a bad dose fails A."

---

## Slide 9 — Per-world audits (a world ships only if all pass)

- **decoy inertness:** confounder correlates with outcome (|r|≥0.3) but forcing
  it moves the outcome ≤0.6 units.
- **proxy signal:** true proxy is informative — observational |r| in [0.35, 0.75]
  OR (for dormant-at-baseline mechanisms) shifts >1 SD under the true
  intervention.
- **distractor inertness:** all non-active actuators ≈0 effect (Slide 7).
- **gold self-consistency:** no single-actuator perturbation of gold beats gold.
- **counterintuitiveness:** each declared naive intervention (the operators'
  obvious first move) has utility gain < 3.0 over baseline — the obvious move must
  not work.
- Calibration auto-tunes proxy assay-noise and confounder loading to hit the
  bands before auditing.

Notes: "These make difficulty and fairness properties of the world, checked
automatically, not asserted by us."

---

## Slide 10 — The four world families (four causal structures)

| World | Structure | Required inference |
|---|---|---|
| Bioreactor | mediation chain + confound | trace outcome→…→hidden copper past an oxygen confound; interior dose |
| Datacenter | sign-reversal | the operators' fix direction is wrong (more cooling → worse) |
| Greenhouse | two interacting causes (AND-gate) | neither lever works alone; fix requires iron **and** pH |
| Clinic | latent-subtype effect heterogeneity | drug effect ≈0 on average, opposite-signed by subgroup |

- Detail: the four are distinct topologies, not reskins; each stresses a
  different failure mode. Two more (collider/selection, feedback/masking) are
  planned; collider needs selection support in the engine.

Notes: "Scaling axis is topologies first, seeds second — the hard part is
structure, not parameters."

---

## Slide 11 — Worked example: bioreactor (ground truth)

- Chain: `FeedWaterFlow × FittingCorrosion → DissolvedCopper → ROS → CarbonFlux →
  ProductTiter`. Confounder `BatchSeedAge → DissolvedOxygen` and → titer.
- Observable true proxy: `BrothTurbidity` (2 hops downstream). Confounded decoy:
  `DissolvedOxygen`. Targeted actuator: chelating additive (interior optimum;
  over-strip penalty at high dose). Symptom trap: a `mask` actuator that lifts the
  titer reading only.
- Quantities the agent must recover from data:
  - obs corr(DO, titer) = **0.42**; obs corr(turbidity, titer) = **−0.48**;
  - forcing DO high vs low → titer **21.4 vs 21.4** (DO is not causal);
  - chelator dose-response: 0→21, 40→46, **66→65 (peak)**, 100→55.

Notes: "Everything here is hidden from the agent; it must reconstruct it by
querying and intervening."

---

## Slide 11b — Bioreactor: actual Opus-4.8 trace (accepted run)

This is a verbatim trace from `opus48_bioreactor.json`. The agent **passed**
(A ∧ B): utility 65.1 vs gold 64.9 (gap −0.14); battery 5/6 = 0.83. 11 queries
used of 15 budget. 12 turns total.

- **Turn 0 (measure):** measures titer, DO, temperature, turbidity, viable cell
  density. Observes: turbidity~titer = **−0.545** (strong), DO~titer = 0.33
  (weak), temperature~titer ≈ 0 (rules out temp immediately).
  > "The timing points to the maintenance shutdown… possibly a metal
  > contamination from the new feed-water fitting (leaching)."
- **Turn 2 (intervene: chelator @ 60):** first interventional test of the metal
  hypothesis. Titer jumps from baseline **~21 → 63.5**, turbidity drops from
  **~44 → 6.5**. Massive signal.
- **Turn 3 (measure):** re-measures baseline (no intervention) to confirm
  contrast: titer **20.6**, turbidity **43.4**. The chelator effect is real.
- **Turn 5 (intervene: feed_flow = 0):** tests whether the source is the feed
  water. Titer = **64.7**, turbidity = **6.5**. Cutting the contaminant source
  at the inlet gives the same recovery as chelating it in the broth.
  > "stopping feed-water inflow removes the leaching metal source, restoring
  > titer."
- **Turn 6 (intervene: DO controller @ 50):** the decisive confound-breaking
  test. Forces DO to the "normal" setpoint. Titer = **20.9** (stays low),
  turbidity = **44.9** (stays high).
  > "fixing DO does NOT recover titer — DO is a confounded decoy."
- **Turns 8–9 (dose optimization):** tests chelator @ 40 → titer **45.7**;
  chelator @ 80 → titer **60.5**. Confirms the peak is near 60 and that
  over-dosing hurts.
- **Turn 10 (combination):** chelator 60 + feed_flow 0 → titer **65.1**,
  turbidity **5.3**. The best observed outcome.
- **Turn 11 (answer):** recommends `{chelator_dosing: 60, feed_flow_controller: 0}`.
  Names the hidden cause as "metal leaching from the replaced feed-water
  fitting." Names turbidity as the true proxy. Calls DO a confounded decoy.

**Why this is the slide to show:**
1. The agent followed the textbook path: observe → hypothesize → intervene to
   confirm → break the confound with a control experiment → dose-optimize → answer.
2. It did it in 11 queries (out of 15), efficiently — no budget wasted on inert
   distractors (never tried agitation, antifoam, glucose, etc.).
3. It explicitly stated and tested the confound: "DO correlates, but forcing it
   doesn't help → it's a decoy." That is the scientific reasoning step.
4. It found the interior optimum: 60 beats 80, chelator + flow cut beats either
   alone.
5. The one battery miss: it incorrectly reported the sign of `feed_flow_controller`
   (minor; it correctly recommended feed_flow = 0 in its intervention).

Notes: "This is the clean success case — what correct scientific reasoning looks
like in the benchmark. Compare with the datacenter trace (Slide 16), where the
same model stops one hop short."

---

## Slide 12 — Worked example: datacenter (counterintuitive)

- Chain: aggressive `CoolingSetpoint` → `CoilCondensation` → `InterfaceErrors` →
  `Throughput`. Decoy: `RackInletTemp` (driven by a maintenance confounder).
- Utilities (baseline vs interventions):
  - baseline **20.8**; more cooling (setpoint 40) **14.6**; less cooling (75)
    **34.1**; dehumidify (gold) **65.0**.
- The operators' obvious move (more cooling) is strictly worse. A "dew-point /
  coil-moisture" observable was added so the true mechanism is **discoverable
  from data**, not outside knowledge.

Notes: "This is the counterintuitive case: the naive intervention is negative,
and the audit enforces that."

---

## Slide 13 — Experiment: Opus-4.8, 20 worlds (4 topologies × 5 seeds)

- Setup: `claude-opus-4-8`, budget 15, ≤32 turns, LLM resolver on. Accept = A ∧ B.
- Part A = recovered ≥90% of achievable benefit `(rec−baseline)/(gold−baseline)`
  (a fraction, so "found the fix" is comparable across worlds of different
  utility ranges — an earlier absolute-tolerance version was too strict on
  wide-range worlds and too lenient on the narrow-range clinic).
- Overall: **4/20 accepted (0.20, 95% CI [0.08, 0.42])**; part-A rate 0.50;
  mean part-B (structure) 0.63.
- By topology (accept, part-A rate, mean part-B):

| Topology | Accept | Part-A | Part-B | What happens |
|---|---|---|---|---|
| Bioreactor (mediation chain) | **4/5** | 1.00 | 0.92 | solved; the one miss got the fix right (99% benefit) but mislabeled structure (B only) |
| Datacenter (sign reversal) | 0/5 | 0.00 | 0.65 | rejects "add cooling" but stops at "cool less"; never reaches the true fix |
| Greenhouse (two causes) | 0/5 | 0.40 | 0.66 | inconsistently tests iron+pH jointly; when it does, under-doses |
| Clinic (hidden subtype) | 0/5 | 0.60 | 0.37 | finds a dose that helps on average; never identifies the subgroup structure |

- Fairness note: every world is verified **solvable** (playing the ground-truth
  answer scores A∧B); the gap is difficulty-by-construction + genuine reasoning
  gaps, not broken worlds. **Caveat:** clinic's *ideal* answer is a conditional
  policy (treat the responsive subgroup); the grader scores only a single dose,
  so clinic's 0/5 partly reflects that skill/grader mismatch — do not read it as
  equivalent to datacenter's 0/5.

- Decomposition matters: several datacenter/greenhouse/clinic worlds get part-A
  **right** (recovered 95–99% of benefit) yet fail part-B — the model **acts
  correctly but names the mechanism wrong**. Others fail part-A (never find the
  fix). Both are genuine, not grading artifacts (audited world-by-world).

Notes: "The 0.15 aggregate is not the headline; the per-structure decomposition
is. Bioreactor is near-solved; the other three fail in structure-specific ways."

---

## Slide 13b — Second model: Qwen3.6-27B (open-weight, partial run)

- Same worlds, run via Nautilus. **Partial: 7 of 20 worlds completed** (5
  bioreactor + 2 clinic) — a Qwen batch was interrupted; comparison is only valid
  on the **matched topologies**, not pooled.
- Matched comparison, bioreactor (n=5 each):

| Model | Accept | Part-A | Part-B |
|---|---|---|---|
| Opus-4.8 | 4/5 | 1.00 | 0.90 |
| Qwen3.6-27B | 1/5 | 0.60 | 0.85 |

- Clinic (n=2, Qwen): 0/2 accepted; part-A 2/2 (found a reasonable dose), part-B
  0.50 — same "acts without naming the mechanism" pattern as Opus.
- **Read cautiously:** n=5 per cell (95% CI on Opus 4/5 is [0.38, 0.96] vs Qwen
  1/5 [0.04, 0.62] — overlapping). The signal is *directional* (Opus ≥ Qwen on
  bioreactor), not yet a significant gap. Datacenter/greenhouse Qwen runs are
  pending.

Notes: "Be upfront: Qwen is a partial run, so I only compare on the worlds both
models saw. Even there, n=5 means the CIs overlap — I'd frame it as 'consistent
with Opus ≥ Qwen, not yet significant', and note the run needs finishing."

---

## Slide 14 — Cross-structure pattern (main finding)

- In every family, the agent performs the hard *qualitative* step and then stops
  one inferential hop short of the complete answer:
  - Datacenter: recovers "cooling too aggressive" (correct, hard); does not reach
    "→ condensation → dehumidify". Recommends setpoint 75 (util ~34 vs gold 65).
  - Greenhouse: one run discovers the joint iron+pH dependence (correct proxy and
    signs) but under-doses (gap 2.95); the other never tests the pair.
  - Clinic: given the explicit "no average effect" cue, neither run stratifies by
    the subtype screen; one verbally proposes targeting by the wrong modifier.
- **Finding: the model reliably overturns the obvious-wrong hypothesis and breaks
  confounds, but does not reliably trace the mechanism to its root or tune the
  utility-optimal dose.** Same shape across four distinct structures.

Notes: "This is the claim I'd defend: a specific, repeatable boundary, with
per-turn traces to localize it."

---

## Slide 15 — Verification: safeguards against measurement artifacts

- The 20-world Opus batch auto-flagged **8/20** as possible harness artifacts —
  two patterns: (a) part-A passed but part-B failed, (b) the agent's named
  mechanism proxy was a long phrase the resolver could not map.
- Manual inspection of the flagged worlds: the "unresolved proxy" cases were
  mostly **genuine conceptual errors** — the agent named an *actuator or an
  experiment* ("root-zone pH", "diuretic dose level", "response to titration")
  where a *measurable downstream marker* was required — not the resolver dropping
  a correct answer. **No verdicts changed.**
- Prior history motivating this: two earlier batches showed 0/N that were entirely
  grader/resolver bugs. Standing rule: **a flagged failure is not read as
  difficulty until inspected.**
- Known residual: the resolver still under-maps very long free-text proxy
  descriptions; routing those through the LLM resolver is the pending fix. It is
  verdict-neutral here but inflates the flag count.

Notes: "8 of 20 sounds alarming, so I checked them: they're mostly the model
describing an experiment instead of naming a measurable proxy — a real error the
flag surfaced, not the grader cheating the model. I'll state the one residual
resolver gap plainly."

---

## Slide 16 — Trace evidence (datacenter, verbatim)

- The log records reasoning per turn. Example (datacenter):
  - > "raising the cooling setpoint INCREASED throughput. This implies the room
    isn't actually too hot; over-cooling is hurting."  — correct, non-obvious.
  - Final answer recommends reducing cooling; the run never measures coil
    moisture and never invokes the dehumidifier.
- The exact turn where the inference chain terminates is identifiable in every
  run.

Notes: "We localize the failure, not just score it."

---

## Slide 17 — Status and limitations

- Built and verified: SCM engine + 4 topologies; computed oracle (with synergy
  search) and A∧B grader; 5 per-world audits; free-text protocol with guarded
  resolver + sandboxed code; batch runner with artifact flagging + resume.
- Limitations to state plainly:
  - **n = 5 per world (Opus, 20 worlds); Qwen partial (7/20).** Per-topology CIs
    are still wide (5-trial 95% CI ≈ ±0.35); the per-structure *failure modes* are
    the reliable finding, not exact accuracy differences. The Qwen batch needs
    finishing before any cross-model claim.
  - Clinic's optimal answer is a **conditional policy** (treat one subgroup); the
    current single-intervention grader only scores an interior dose, so it
    under-tests the intended skill.
  - Resolver still misses very long free-text proxy descriptions (verdict-
    neutral so far; fix: route those through the LLM resolver too).
  - Within a topology, worlds differ only by parameter jitter (same graph). A
    structural sampler (v7) that varies topology per seed exists in prototype;
    the results here predate it.

---

## Slide 18 — Next steps

- Increase seeds (≥5/world) for stable accept-rates; add 1–2 more topologies
  (collider/selection needs engine support for conditioning).
- Add a conditional-policy answer schema to properly score the clinic family.
- Run additional models for comparison.
- Deliverable: a per-structure characterization of where LLM causal reasoning
  terminates early, with traces.

---

## Backup — key parameters (one place)

- Budget 15 queries; ≤32 turns; 400 units/query; joint interventions ≤3.
- Grade A: `E[util] ≥ gold − 2.0`, n=30k, CRN. Grade B: ≥0.8 of {proxy, decoys,
  per-actuator signs}. Accept = A ∧ B.
- Audits: decoy |r|≥0.3 & |do-effect|≤0.6; proxy |r|∈[0.35,0.75] or >1 SD
  interventional shift; naive gain < 3.0; distractor ≈0 marginal+interaction;
  gold self-consistency.
- Mechanism library (per-node forms): see Slide 3b table.
- Actuator operations (`set`/`scale`/`add`/`mask`): see Slide 4 table.

---

## Backup — lineage

- **ACED-Bench:** question + variable list given; queries a Bayesian network;
  outcome-graded. (Complete.)
- **RPG v3/v4:** cause hidden in a story, but small worlds + descriptive action
  names leaked the answer; keyword/string grading.
- **RPG v5:** declarative SCM + computed grader; worlds still small enough to
  brute-force.
- **RPG v6:** large, no menu, actuators-only, meaningful names, sandboxed code,
  counterintuitive-by-audit; leakage addressed by scale rather than concealment.

---

# FINAL RESULTS TABLE — Opus-4.8 vs Qwen3.6-27B (batch20, 4 topologies × 5 seeds)

Snapshot 2026-08-05. Both models run on the **same 20 audited worlds**, identical
protocol (budget 15, LLM resolver on, per-turn output = model max, unproductive
turns not charged). Grading identical (Part A = recovered ≥90% of achievable
benefit; Part B = ≥0.8 of {true proxy, decoys, monotone actuator signs}; accept =
A ∧ B). Qwen numbers are from the **fixed re-run** (`batch20_qwen_2`), after the
truncation and turn-cap fairness fixes; the earlier Qwen run is superseded.

## Overall

| Model | n | Accepted | Accuracy [95% CI] | Part-A rate | Mean Part-B |
|---|---|---|---|---|---|
| Opus-4.8 | 20 | 4 | 0.20 [0.08, 0.42] | 0.50 | 0.65 |
| Qwen3.6-27B (`qwen3-small`) | 20 | 2 | 0.10 [0.03, 0.30] | 0.40 | 0.61 |

## By topology (accept rate; part-A rate / mean part-B)

| Topology | Opus accept | Opus A / B | Qwen accept | Qwen A / B |
|---|---|---|---|---|
| Bioreactor (mediation chain) | 4/5 | 1.00 / 0.92 | 2/5 | 0.60 / 0.87 |
| Datacenter (sign reversal) | 0/5 | 0.00 / 0.65 | 0/5 | 0.00 / 0.67 |
| Greenhouse (two causes) | 0/5 | 0.40 / 0.66 | 0/5 | 0.20 / 0.55 |
| Clinic (hidden subtype) | 0/5 | 0.60 / 0.37 | 0/5 | 0.80 / 0.35 |

## How to read this (caveats, so the table is not over-claimed)

- **n = 5 per cell → wide CIs that overlap.** Opus ≥ Qwen overall and on
  bioreactor is *directional*, not statistically significant at this n. The
  reliable signal is the **per-structure failure pattern**, not the exact rates.
- **The comparison is fair.** The Qwen re-run fixed the two harness problems from
  the first attempt: output truncation (per-turn tokens now = model max, 32768)
  and turn-cap starvation (unproductive/empty turns are retried free, not
  charged). Verified on the re-run: 0 truncation-starved worlds, 0 turn-cap
  no-answers, parse errors down 47→2 of 514 turns. Near-cap answering is
  symmetric (Opus 14/20, Qwen 13/20 answer at turn ≥30), so the cap presses both
  models equally.
- **Every world is solvable** (playing ground-truth scores A∧B on all 20) and the
  failures were audited world-by-world as **genuine reasoning errors**, not
  grading/resolution artifacts:
  - datacenter: both models recommend the plausible-but-wrong "reseat the network
    card" and never reach the true mechanism (condensation → dehumidify);
  - greenhouse: inconsistent discovery of the iron∧pH conjunction;
  - clinic: both often name a *cause/knob* as the mechanism proxy instead of the
    downstream marker (part-B), even when they find a reasonable dose (part-A).
- **Clinic construction caveat:** its ideal answer is a conditional policy (treat
  the responsive subgroup); the grader scores only a single dose. Clinic's 0/5
  for both models partly reflects that skill/grader mismatch — do not read it as
  equivalent evidence to datacenter's 0/5.

## One-line takeaway

Both models do the hard qualitative steps (break confounds, reject the obvious
wrong fix) but repeatedly stop one inferential hop short of the full mechanism or
the exact fix; Opus does so less often than Qwen (directional, n=5). Bioreactor
(the single-lever mediation chain) is the only family either model solves
reliably.
