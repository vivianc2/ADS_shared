# RPG v6 — Pipeline & Structure Overview (for presentation)

A single map of the whole current system: what each piece is, how data flows,
and where the design choices live. Companion to the run analysis
(`rpg_v6_datacenter_run_analysis.md`) and the design doc
(`worldgen_rpg_plan_v6_open_scenario.md`). Snapshot: 2026-08-04.

## 1. The idea in one paragraph

We build a fully-specified generative causal world (an SCM: equations + known
noise) that we understand completely. A scientist *agent* sees only a plain-
language scenario — no variable list, no action menu — and must, like a real
scientist, decide what to measure, what to intervene on (in free text), analyze
raw data with code, and conclude what the hidden cause is and what to do. Because
we own the equations, we compute the mathematically correct answer and grade the
agent against it. Worlds are made **counterintuitive** (the obvious first move
fails) and **large** (variables ≫ budget) so success requires genuine reasoning,
not brute force or label-reading.

## 2. Pipeline (data flow)

```
                    ┌──────────────────────── GENERATION (offline, no API) ───────────────────────┐
  topology template │  worlds_v6.py  ──►  jitter params (seed)  ──►  calibrate difficulty bands    │
  (SCM + roles +    │       │                                          (oracle_v6.calibrate)        │
   scenario prose)  │       ▼                                                 │                     │
                    │  compute GOLD (oracle_v6.optimal_intervention: screen → combos →              │
                    │  golden-section refine)  +  COUNTERFACTUAL BATTERY  +  full AUDIT suite        │
                    │       │                                                                        │
                    │       ▼  emit only worlds that pass EVERY audit  ──►  world_<id>.json          │
                    └────────────────────────────────────────────────────────────────────────────┘
                                                     │
                    ┌──────────────────────────── RUN (per world, live LLM) ──────────────────────┐
                    │  run_batch_v6.py → run_agent_v6.run_world  ──loads──►  SimV6 (holds SCM,      │
                    │                                                        stored gold/battery)   │
                    │   each turn:  scenario + budget + last result + memory + DIRECTIVES           │
                    │        │                                                                      │
                    │        ▼   Opus 4.8 (bedrock_llm) → <reasoning><action><memory>               │
                    │        │                                                                      │
                    │    ┌───┴────────────── action ──────────────┐                                │
                    │  measure   intervene      code            answer                             │
                    │    │          │            │                │                                 │
                    │  resolver  resolver     sandbox         resolver + GRADE                      │
                    │  (free-text → var/actuator, echoed & logged)  (part A utility ∧ part B battery)│
                    │    │          │            │                                                  │
                    │    ▼          ▼            ▼                                                  │
                    │  raw CSV    raw CSV   pandas/np/scipy over the CSVs (no budget cost)           │
                    └────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                              per-world result JSON + batch summary.json (aggregate + table)
```

## 3. Files (what each one is)

Generation & world model:
- **`engine.py`** (SCM interpreter) — one generic topological evaluator over a
  typed DAG. Variables carry `aliases`, `measurable`+`assay_noise`, and a
  mechanism (`linear/saturating/hill/soft_threshold/interaction/sign_flip`).
  Actuators are first-class (`set/scale/add/mask`); the agent acts ONLY through
  them (no `do()` by fiat). `sample()` = true structural state; `measure()` =
  noisy assay + symptom-mask bias.
- **`worlds_v6.py`** — the topology templates as SCM specs + scenario prose +
  hidden ground-truth roles (`true_root`, `true_mechanism_proxy`,
  `confounded_decoys`, `targeted_actuator`, `symptom_trap_actuator`,
  `naive_interventions`). Two so far: `bioreactor_titer_loss_v6`,
  `datacenter_throughput_v6`.
- **`oracle_v6.py`** — computed gold (screen actuators → search combinations →
  golden-section refine continuous doses), counterfactual battery, calibration,
  and the audit suite.
- **`generate_v6.py`** — instantiate template → jitter → calibrate → gold+battery
  → audits → emit only passing worlds as `world_<id>.json`.

Runtime:
- **`sim_v6.py`** — loads a world, serves `measure`/`intervene` (writing raw
  per-unit CSVs), and grades answers. Holds the precomputed gold/battery so the
  live run matches exactly what was audited.
- **`resolver.py`** — maps agent free text → a canonical variable/actuator, or a
  plausible rejection. Hardened against artifacts from BOTH directions:
  stem-matching (chelating/chelation cluster), distinctiveness weighting (a rare
  concept word beats a generic one; generic tokens don't license a match), and a
  **request-coverage gate** (a confident match must explain ≥50% of the request's
  distinctive tokens — stops "network link speed"→"fan speed"). Anything in the
  uncertain band routes to an **LLM fallback** that sees the whole catalog
  (measurables + actuators + hidden vars) and returns the correct *outcome type*
  (measure / intervene / no_assay / no_actuator / not_in_world), so it yields a
  faithful rejection, not a blind id-or-none. Every resolution is echoed + logged.

### Artifact hardening (never read a false 0/N as difficulty)

Three times a scary batch number was our plumbing, not the science (v4 string
grader → 0/8; strict resolver → 0/10). Structural guards now in place:
- **Computed grading, not one blessed answer.** The battery credits ANY genuine
  mechanism proxy (measurables the true lever moves), computed from the SCM — a
  correct-but-alternative answer is not marked wrong.
- **`_artifact_check`** (run_agent_v6) flags a run as *suspect* when a failure
  smells like resolution/grading rather than reasoning: nothing the agent
  recommended resolved to an actuator; all recommended actions were rejected; the
  named proxy didn't resolve; or the run failed with **part A passed but part B
  failed** (the classic correct-but-miscredited pattern).
- **`run_batch_v6`** surfaces an `art` column + a loud warning listing suspects,
  and an `artifact_suspects` count in `summary.json`. Rule of practice: **inspect
  every suspect (and any 0/N) before interpreting it as difficulty.**
- **`sandbox.py`** — spawned-subprocess Python (pd/np/scipy, hard timeout, stdout
  captured, vars carried forward) over the experiment CSVs. Compact analog of
  ACED's `scientist_coder_agent_new` sandbox.
- **`run_agent_v6.py`** — the agent loop: prompt build, action parse/dispatch,
  budget/turn accounting, directives, forced answer at exhaustion, grading.
- **`run_batch_v6.py`** — runs a directory of worlds, writes per-world results +
  aggregate `summary.json` and a legible per-world table.
- **`demo_v6.py`** — no-LLM scripted expert walkthrough + audit dump.

## 4. The five agent actions (final design)

| Action | Effect | Costs budget? |
|---|---|---|
| `measure` | free-text quantity requests → assays; returns raw rows (CSV) + summary | yes |
| `intervene` | free-text actuator requests + doses, applied jointly; raw rows | yes |
| `code` | pandas/numpy/scipy over accumulated CSVs | **no** (analysis) |
| `answer` | structured conclusion → computed grade | — |
| `give_up` | bail | — |

Deliberately minimal & orthogonal. Not added (and why): *sweep* = intervene at
levels + code; *clamp* = a `set` actuator already is one; *ask about apparatus* =
leak risk.

## 5. How the golden answer is computed & graded

- **Gold intervention** (part A): screen each actuator's marginal effect, keep
  the active set, search combinations among them, refine continuous doses by
  golden-section. Grade = recommended intervention's re-simulated utility within
  tolerance of gold.
- **Counterfactual battery** (part B): SCM-derived ground truth for — which
  measurable signal is the true mechanism proxy, which are confounded decoys, and
  the sign of each actuator's true effect. Grade = ≥ 80% match.
- **Accept = A ∧ B.** Both are computed from the equations, so a correct-but-
  differently-worded answer passes and a plausible surface answer fails.

## 6. Audit gates (a world ships only if ALL pass)

- **decoy inertness** — confounders correlate with the outcome but have ~0
  do-effect.
- **proxy signal band** — the true proxy's observational correlation sits in a
  visible-but-not-decisive band (~0.4–0.6).
- **distractor inertness** — the many in-world distractor actuators have ~0
  marginal and ~0 interaction (justifies oracle pruning; makes brute force
  wasteful).
- **gold self-consistency** — no simple perturbation beats the reported gold.
- **counterintuitiveness** — the `naive_interventions` (obvious first move) must
  NOT meaningfully help; ideally they hurt.

## 7. Scale (where we are vs. where we're going)

- **Now (pilot):** 2 topologies × seed-jitter. Validates the full pipeline on
  real API; gives two genuinely different structures.
- **Design choice:** scale by **distinct topologies first, seeds second** — the
  hard part is *structure*, so jittered clones of one template test the same
  insight repeatedly (measures reliability, not breadth).
- **Benchmark target:** ~5–6 structurally distinct topologies (mediation chain ✓,
  backwards sign-flip ✓, collider/selection, two interacting causes, hidden
  subtype, feedback/masking) × a few seeds each.

## 8. Lineage (how we got here)

- **ACED-Bench** — agent given a question + variable list; queries a Bayesian
  net; string/`llm`-graded. (Finished; the code-agent condition inspired v6's
  `code` tool.)
- **RPG v3/v4** — hid the cause in a story, but small worlds + descriptive action
  names leaked the answer; grader was keyword string-matching.
- **RPG v5** — declarative SCM + computed grader (battery). Fixed grading, but
  worlds were small → brute-forceable.
- **RPG v6** — large, menu-free, actuators-only, meaningful names, code tool,
  counterintuitive-by-audit. Leakage solved by *scale*, not hiding.
