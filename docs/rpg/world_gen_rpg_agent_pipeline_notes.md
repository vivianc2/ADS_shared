# RPG Agent Pipeline Adaptation Notes

These notes track changes needed after `world_gen_rpg.py` starts producing
simulator-based worlds. They are intentionally separate from the generator so
dataset design can move first.

> **Two RPG generations now coexist in `world_gen_rpg.py`.**
> The original section is v1 — dynamic, time-based, fixed policy menus.
> The lower section (after the `STATIC RPG v2` banner) is v2 —
> static, partially-observed, no policy menu, agent submits a `do(.)` dict.
> The v2-specific updates to this notes file are in §A below; everything
> after it is the legacy v1 plan and still applies to the dynamic worlds in
> `all_out_rpg/out_rpg_v1/`.

---

## A. v2 (static) updates

### What changed vs v1

| | v1 (dynamic) | v2 (static) |
|---|---|---|
| Schema version | `rpg_v1` | `rpg_static_v2` |
| Time axis | `horizon` periods, trajectory rollouts | none; one-shot potential outcomes under `do(.)` |
| Agent action surface | pick from a fixed `allowed_policies` menu | construct a `do(.)` dict over `intervenable_variables` |
| Query modes | `observational_trajectory`, `policy_rollout`, `policy_comparison` | `observational_sample`, `interventional_sample`, `inspect_unit` |
| Answer schema | one policy id | `intervention_with_hypothesis`, future: `conditional_policy`, `anomaly_identification` |
| Budget unit | cells of `(n_units × horizon × n_policies)` | cells of `(n_units × n_columns)` per query |
| Archetypes shipped | 5 dynamic | 2 static (`hidden_cause`, `confounded_action`); 4 more planned |
| Optional LLM polish | none | `--llm-polish` rewrites narrative; `--llm-extra-templates N` adds domains |

The v1 simulator (`RPGSimulator`, schema `rpg_v1`) is unchanged. The agent
runtime needs a **parallel** `StaticRPGSimulator` selected by
`schema_version`.

### What the agent sees (v2)

Only the `visible` block:

- `story` — neutral 2-3 sentence framing, often names the prevailing
  (wrong) theory.
- `observed_variables[*]` — each has `name`, `description`, and `scale`
  (continuous 0-100 or categorical).
- `intervenable_variables[*]` — each has `name`, `values`, `default`,
  `description`. There is **no policy id**; the agent constructs a
  `do(.)` dict using these knob names + values.
- `allowed_query_modes` — `observational_sample`, `interventional_sample`,
  `inspect_unit`.
- `experiment_budget` — cells / max-units / max-measurements-per-query
  caps.
- `question` and `answer_schema` (currently `intervention_with_hypothesis`).
- `max_intervention_knobs` — hard cap on how many knobs the agent's
  answer may set.

What is `hidden` and must be stripped from any agent prompt:

- `hidden.simulator_config` (parameters, role mappings, template).
- `hidden.diagnostics` (recoverability band, observational correlations).
- `oracle` (gold answer, action scores, runner-up, margin).
- `validators` (signature checks, accepted flag).
- `questions[*].answer` and `questions[*].metadata.gold`.

### Required v2 runtime components

1. **`StaticRPGSimulator` in `framework_code/simulator_rpg.py`.**
   Selected by `schema_version == "rpg_static_v2"`. Mirrors the existing
   `RPGSimulator` class but exposes:
   ```python
   sim.run_query(StaticRPGQuery)  -> StaticRPGQueryResult
   sim.public_world()             -> dict (the visible block only)
   sim.score_answer(answer_dict)  -> dict (uses oracle for scoring)
   ```
   Query handlers must:
   - validate `intervention` dict against the knob name + value
     specs, rejecting unknown knob names with a clear error message
     (this is how the "invalid action" ability is tested implicitly);
   - enforce `max_units_per_query` and
     `max_measurements_per_query` per query;
   - decrement the global cell budget and reject queries that exceed it;
   - for `inspect_unit`, return all `allowed_measurements` for a single
     freshly-sampled unit; remember the unit ids so the agent can later
     reference them in `anomaly_identification` answers.

2. **`StaticRPGWorldModel` (or extend `world_model_rpg.py`).**
   Parses NL queries from the scientist agent into typed
   `StaticRPGQuery` objects. Must reject queries that mention hidden
   variable names.

3. **`--static` audit mode in `audit_rpg.py`.**
   - Schema check (`schema_version == "rpg_static_v2"`).
   - Visibility check: hidden variable names absent from
     `visible.story` / `visible.observed_variables[*].description` /
     `visible.question`.
   - Oracle recheck with fresh seed; gold answer must be stable.
   - Recoverability-band recheck.
   - Per-archetype mechanism signature recheck.

4. **Answer scoring** in `framework_code/evaluate_*.py`.
   - `intervention_with_hypothesis`: re-run the simulator under
     `do(agent_intervention)`; accept if utility is within
     `oracle.oracle_tolerance` of `oracle.gold_answer.expected_utility`.
     Hypothesis is *recorded but not scored* in v2.
   - Future schemas (`conditional_policy`, `anomaly_identification`)
     get added as v1.5/v2 archetypes ship.

5. **`run_agent_batch_rpg_static.py`** (or branch inside the existing
   batch runner). Loads `rpg_static_v2` worlds, instantiates
   `StaticRPGSimulator`, runs the scientist agent under the visible
   budget, and writes a trajectory log. Bedrock Opus 4.8 is still the
   default scientist (`--scientist-model us.anthropic.claude-opus-4-7`).

### Where v2 worlds and artifacts live

- Generator (both v1 dynamic and v2 static): `dataset_generation_code/world_gen_rpg.py`.
- 12-world v2 pilot: `dataset_generation_code/all_out_rpg/out_rpg_static_v2_pilot_12/`.
  Manifest: `manifest_rpg_static_v2.json`.
- Regenerate fresh with:
  ```bash
  cd dataset_generation_code
  python world_gen_rpg.py --static \
      --outdir all_out_rpg/out_rpg_static_v2_pilot_12 \
      --oracle-n 15000
  ```
  Add `--llm-polish` to rewrite narratives, `--llm-extra-templates 2` to
  let Opus invent fresh subdomains.

### Open v2 work (in order)

1. ~~`StaticRPGSimulator` + `audit_rpg.py --static`.~~ **DONE (2026-05-29).**
   - `framework_code/simulator_rpg.py` — `StaticRPGSimulator`,
     selected by `schema_version == "rpg_static_v2"`.
   - `dataset_generation_code/audit_rpg.py --static` — schema check,
     leakage check, fresh oracle recheck, proxy-calibration check,
     position-balance report.
2. ~~Answer scoring for `intervention_with_hypothesis`.~~ **DONE.**
   - `StaticRPGSimulator.score_answer(answer_dict)` returns
     `{accepted, expected_utility, gold_expected_utility,
     utility_gap_from_gold, ...}`. Scoring is exact-match for known
     intervention keys; otherwise fresh Monte Carlo against the same
     mechanism. Non-intervenable knob names rejected verbosely.
3. v1.5 archetypes (`hidden_subtype`, `anomaly_discovery`) — these
   introduce two new answer schemas with their own scoring code.
4. v2 archetypes (`mechanism_chain`, `negative_control`) — reuse the
   `intervention_with_hypothesis` schema, including `{}` as a valid
   answer.
5. 60-world full pilot with `--llm-polish --llm-extra-templates 2`.

### Reference docs for v2

- Plan: [worldgen_rpg_plan_static_partial_observation.md](worldgen_rpg_plan_static_partial_observation.md).
- Lessons + watchpoints from generation: [rpg_v2_slide_examples.md](rpg_v2_slide_examples.md).
- This file: agent-pipeline-side adaptations.

---

## A2. v3 direction: latent-regime discovery

The next scientific direction is not another small best-arm archetype. The
benchmark should test latent discovery: whether an agent can infer hidden
regimes, clusters, switches, or bifurcations from queried data and then
choose experiments/actions based on that representation.

### First implementation target

Add a new static archetype:

```text
latent_regime_discovery
```

Scientific shape:

- Population is a mixture of two hidden response regimes.
- The public prompt does not name the regimes.
- Several observed variables are plausible; only some reveal the split.
- Different actions help different regimes.
- The final answer is a latent-structure claim plus a regime-aware policy.

Answer schema:

```json
{
  "latent_structure": {
    "n_regimes": 2,
    "evidence": "..."
  },
  "policy": {
    "branch_variable": "ObservedProxy",
    "branch_threshold": 50,
    "if_above": {"ActionForHighRegime": "on"},
    "if_below": {"ActionForLowRegime": "on"}
  },
  "hypothesis": "..."
}
```

### Files to change

#### `dataset_generation_code/world_gen_rpg.py`

Add:

- `latent_regime_discovery` to `STATIC_ARCHETYPES`.
- A default distribution count of `0` initially, so existing all-six
  generation commands do not change accidentally.
- A new `STATIC_TEMPLATES["latent_regime_discovery"]` entry, starting with
  one medical-style domain.
- Mechanism functions:
  - `_static_latent_regime_default_params`
  - `_static_latent_regime_sample_hidden`
  - `_static_latent_regime_apply`
  - `_static_latent_regime_observe`
  - `_static_latent_regime_candidate_interventions`
  - `_static_latent_regime_assignment`
  - `_static_latent_regime_score_policy`
- Dispatch updates:
  - `_static_default_params`
  - `_static_sample_hidden`
  - `_static_apply`
  - `_static_observe`
  - `_static_candidate_interventions`
  - `_static_assignment`
  - `_static_primary_target_higher_is_better`
  - `_static_utility_from_outcomes`
  - `_static_primary_obs_name`
  - `_static_oracle_observational_correlations`
  - `_static_n_intervenable_knobs`
  - `_static_gold_anchor_role_key`
  - `STATIC_BUILDERS`
- A visible-block builder:
  - `_static_visible_block_latent_regime`
- A world builder:
  - `_static_build_latent_regime_discovery`

The first version will still expose actions as `intervenable_variables`
because the current runtime expects that field. The next redesign should
rename the public surface to `available_actions` and include diagnostic,
therapeutic, and delayed actions in one story-level catalog.

#### `framework_code/simulator_rpg.py`

Add scoring for:

```text
latent_regime_policy
```

The scorer should:

- parse `latent_structure.n_regimes`;
- require `n_regimes == 2`;
- parse the `policy` object using the same conditional-policy machinery;
- validate that the branch variable is public;
- estimate expected utility under the submitted branch rule;
- accept if utility is within `oracle_tolerance` of the stored gold policy;
- record whether a free-text hypothesis/evidence field was present.

#### `framework_code/scientist_agent_rpg.py`

Prompt changes:

- Add the new answer schema example.
- Add strategy guidance for latent discovery:
  - inspect distributions, not only means;
  - look for bimodality, clusters, thresholds, and response heterogeneity;
  - choose follow-up experiments that distinguish one-regime vs two-regime
    explanations.

Data-summary changes:

- Add simple distribution-shape cues to numeric summaries:
  - p25/p75;
  - IQR;
  - rough tail imbalance;
  - optional "possible bimodality" hint from large median-vs-mean or wide
    quantile gaps.

This is not a substitute for the coder agent, but it gives the plain
scientist agent more signal.

#### `dataset_generation_code/audit_rpg.py`

Add:

- hidden-name leakage terms for new latent variables;
- skip single-intervention recoverability thresholds for
  `latent_regime_discovery`;
- optional static checks for:
  - gold answer has `answer_schema == "latent_regime_policy"`;
  - gold `latent_structure.n_regimes == 2`;
  - conditional policy beats best static intervention.

#### `framework_code/evaluate_rpg.py`

No immediate structural change should be required if the run JSON stores
`score.accepted`, because the evaluator delegates rescoring to
`StaticRPGSimulator`. It should automatically group the new
`answer_schema`.

### First smoke commands

Generate one latent-regime world:

```bash
source /home/vivianchen/miniconda3/etc/profile.d/conda.sh
conda activate ADS

python3 dataset_generation_code/world_gen_rpg.py --static \
  --outdir /tmp/rpg_latent_regime_smoke \
  --distribution '{"latent_regime_discovery":1}' \
  --oracle-n 20000 \
  --max-attempts-per-world 5
```

Audit:

```bash
python3 dataset_generation_code/audit_rpg.py \
  --outdir /tmp/rpg_latent_regime_smoke \
  --static \
  --summary-only \
  --recheck-oracle-n 0
```

For the all-archetype static-plus-latent dataset, both of these are accepted:

```bash
python3 dataset_generation_code/audit_rpg.py \
  --outdir dataset_generation_code/all_out_rpg/out_rpg_static_plus_latent_v1 \
  --summary-only

python3 dataset_generation_code/audit_rpg.py \
  dataset_generation_code/all_out_rpg/out_rpg_static_plus_latent_v1 \
  --summary-only
```

Mock pipeline smoke:

```bash
python3 framework_code/run_agent_batch_rpg.py \
  --worlds-dir /tmp/rpg_latent_regime_smoke \
  --scientist-backend mock \
  --run-name rpg_latent_regime_mock_smoke \
  --output-dir /tmp/rpg_latent_regime_mock_smoke \
  --max-turns 3
```

Evaluate a completed all-archetype static-plus-latent run:

```bash
python3 framework_code/evaluate_rpg.py \
  framework_code/results/rpg_static_plus_latent_opus/rpg_static_plus_latent_opus.json \
  --rescore \
  --rescore-n 20000 \
  --details \
  -o framework_code/evaluations/rpg/eval_rpg_static_plus_latent_opus.json
```

For compact qualitative trajectory review, include per-query traces:

```bash
python3 framework_code/evaluate_rpg.py \
  framework_code/results/rpg_static_plus_latent_opus/rpg_static_plus_latent_opus.json \
  --rescore \
  --rescore-n 20000 \
  --include-traces \
  -o framework_code/evaluations/rpg/eval_rpg_static_plus_latent_opus_traces.json
```

The trace version stores query modes, measurements, interventions, sample sizes,
and query errors, but not full raw LLM text. Full trajectories remain in the
run directory's `rpg_agent_logs/`.

### What remains after first implementation

The first implementation will prove schema/runtime compatibility. It will
not yet fully solve the professor's "bigger RPG action space" point.

Next needed changes:

- replace public `intervenable_variables` with a broader
  `available_actions` catalog;
- support diagnostic actions and delayed follow-up observations;
- make action combinations large enough that enumeration is impossible;
- add validators where fixed scripts and mean-only baselines fail;
- build a coder-agent path that can inspect distributions and fit clusters.

---

## B. Legacy v1 (dynamic) notes

## Implementation philosophy

RPG worlds should get a parallel `_rpg.py` pipeline instead of being squeezed
into the Bayesian-network files. The BN path is built around static variables,
CPDs, and `do(X=x)` samples. RPG worlds are longitudinal simulators where the
primitive operation is a policy rollout over time. Keeping file boundaries
separate reduces accidental leakage, avoids fake BN semantics, and lets us
smoke-test the world side before changing scientist prompts.

The first milestone is world-side only:

1. Generate and audit RPG worlds.
2. Load one RPG world into a runtime simulator.
3. Execute typed trajectory experiments.
4. Return public CSV-style data only.
5. Demonstrate the interaction with a fake/manual scientist in a notebook.

Only after that should we wire the full agent loop.

## LLM target

The current RPG world generation, audit, simulator, and notebook path does not
call an external LLM. It is deterministic/local on purpose so that world
validity is not entangled with prompt quality.

For the future scientist-agent pipeline, use Bedrock Opus 4.8 as the default
scientist model:

```bash
--scientist-backend bedrock \
--scientist-model us.anthropic.claude-opus-4-7
```

The existing BN runner already has Bedrock support and defaults the BN
world-model parser to `us.anthropic.claude-opus-4-7`, but it assumes BN
JSONs. The RPG path should branch before BN conversion and instantiate
`RPGSimulator` plus `WorldModelRPG`.

## Required future components

- Add an `RPGSimulator` that loads `schema_version: rpg_v1` worlds and executes
  trajectory rollouts from `hidden.simulator_config`.
  - Initial implementation: `framework_code/simulator_rpg.py`.
  - It supports typed `observational_trajectory`, `policy_rollout`, and
    `policy_comparison` queries.
- Add a world-model/parser path for typed trajectory queries:
  `observational_trajectory`, `policy_rollout`, `policy_comparison`, and
  eventually `adaptive_policy_rollout`.
- Return CSVs with one row per `(unit_id, time)` and explicit columns for
  policy id, action values, requested measurements, and optionally baseline
  observed covariates. `world_gen_rpg.py` rollouts now include the public action
  columns in returned rows.
- Hide `hidden`, `oracle`, and `validators` from scientist agents. Expose only
  `story`, visible variable metadata, action variables, allowed policy ids, and
  question text.
- Treat top-level `edges` as intentionally empty for RPG worlds. Dynamic causal
  edges live under `hidden.causal_edges_unrolled_template` because they name
  latent simulator state.
- Decide budget accounting for longitudinal data. The cleanest default is
  returned dataframe cells so measurement choice matters. Current RPG worlds
  expose a visible budget: 24,000 cells total, 8,000 cells per query, default
  40 units, max 400 units, and 8 successful queries.
- Add evaluator/extractor prompts for RPG answer types:
  policy id, dose level, mediator/proxy name, mechanism label, and structured
  rejection.
- Add replay support: every successful query should log the world seed,
  query seed, policy id, measurements, horizon, and row count.

## Current RPG-specific files

- `dataset_generation_code/world_gen_rpg.py`: generator and mechanism source of
  truth for RPG v1 worlds.
- `dataset_generation_code/audit_rpg.py`: stricter dataset audit covering
  representation, difficulty signatures, visibility contracts, sample smoke
  tests, and optional fresh oracle rechecks.
- `framework_code/simulator_rpg.py`: standalone runtime simulator for typed RPG
  trajectory experiments.
- `framework_code/schemas_rpg.py`: typed query/result/world-info dataclasses
  for RPG worlds.
- `framework_code/world_model_rpg.py`: natural-language/JSON parser and
  agent-facing world interface for RPG trajectory experiments.
- `framework_code/notebooks/rpg_world_demo.ipynb`: executed inspection notebook
  showing public context, rollouts, parser behavior, hidden measurement
  rejection, and slide examples.

## Current discovery protocol

RPG v1 should be treated as a budgeted scientific-discovery environment, not an
unlimited policy-ranking oracle.

Agent-visible budget:

```json
{
  "sample_accounting": "cells",
  "max_total_samples": 24000,
  "max_samples_per_query": 8000,
  "default_units": 40,
  "max_units": 400,
  "max_queries": 8
}
```

The budget counts returned dataframe cells, so a query gets more expensive when
it asks for more policies, more periods, or more measurements. The intended
agent behavior is iterative: run a small exploratory experiment, analyze the
CSV, form or revise a hypothesis, then spend budget on a focused full-horizon
follow-up.

## File-by-file RPG plan

### `dataset_generation_code/world_gen_rpg.py`

Status: implemented.

Purpose:

- Generate self-contained `schema_version: rpg_v1` world JSON files.
- Keep visible variable/action/policy metadata separate from hidden simulator
  state, oracle scores, validators, and gold answers.
- Store `hidden.simulator_config`, which is the source of truth for runtime
  rollouts.

Important contracts:

- Top-level `variables` equals the agent-visible observed/action variables.
- Top-level `edges` is empty for RPG worlds; dynamic hidden-state edges live in
  `hidden.causal_edges_unrolled_template`.
- `visible.allowed_measurements` is exactly the public measurement catalog.
- `visible.allowed_policies` contains neutral policy ids (`policy_A`, etc.).
- `visible.default_observational_policy_id` defines the reference policy for
  passive/default trajectory queries.
- Rollout rows include `unit_id`, `time`, `policy_id`, public action columns,
  and requested visible measurements.

### `dataset_generation_code/audit_rpg.py`

Status: implemented.

Purpose:

- Provide a repeatable quality gate for every generated RPG dataset.
- Check representation, subtype balance, template balance, per-world visibility
  contracts, required difficulty signatures, sample-generation sanity, and
  optional fresh-oracle answer stability.

Expected use:

```bash
python3 dataset_generation_code/audit_rpg.py \
  --outdir dataset_generation_code/all_out_rpg/out_rpg_v1 \
  --sample-units 7 \
  --recheck-rollouts 100000 \
  --summary-only
```

### `framework_code/schemas_rpg.py`

Status: initial implementation exists.

Purpose:

- Define typed RPG query/result/world-info dataclasses that do not pretend RPG
  queries are BN observations/interventions.
- Provide XML/text formatting compatible with the style expected by scientist
  agents and orchestrator logs.

Planned objects:

- `RPGQueryMode`: `observational_trajectory`, `policy_rollout`,
  `policy_comparison`.
- `RPGParsedQuery`: mode, `n_units`, `policy_ids`, `measurements`, `horizon`,
  raw query, optional seed.
- `RPGQueryResult`: success/failure, data file, preview, columns, sample usage.
- `RPGWorldInfo`: story, observed variables, action variables, allowed
  policies, allowed query modes, default horizon, default observational policy.

### `framework_code/simulator_rpg.py`

Status: initial implementation exists.

Purpose:

- Load one RPG world JSON.
- Validate typed queries against public measurements/policies.
- Execute rollouts by calling the generator's `rollout(...)` function, so there
  is one mechanism source of truth.
- Return a pandas dataframe with longitudinal rows.

Supported modes:

- `observational_trajectory`: use `visible.default_observational_policy_id`.
- `policy_rollout`: exactly one policy id.
- `policy_comparison`: two or more policy ids.

Time contract:

- `time` runs from `1..horizon` in returned rows.
- A shorter query horizon is allowed for exploratory/myopic analyses.
- Horizons above the validated world horizon are rejected.
- Sample accounting is `n_units * horizon * number_of_policies`.

### `framework_code/world_model_rpg.py`

Status: initial implementation exists.

Purpose:

- Be the RPG equivalent of `world_model_causal.py`, but for typed trajectory
  experiments.
- Convert natural-language requests into `RPGParsedQuery`.
- Validate and execute those queries through `RPGSimulator`.
- Write CSV files and return result XML/previews.

Initial parser plan:

- Accept explicit JSON wrapped in `<json>...</json>` for deterministic testing.
- Also support simple natural-language patterns such as:
  - "run policy_A for 100 units over 8 periods measuring X, Y"
  - "compare policy_A and policy_B with 200 units measuring X"
  - "give observational trajectories for 50 units measuring all"
- If an LLM object is supplied, fall back to an LLM parser equivalent to the BN
  world model parser.

Current natural-language parsing contract:

- `observational`, `default`, `current practice`, or `passive` maps to
  `observational_trajectory` and uses `visible.default_observational_policy_id`.
- One policy id maps to `policy_rollout`.
- Two or more policy ids, or the word `compare`, maps to `policy_comparison`.
- Phrases such as `compare all policies` expand to all candidate policies in
  `visible.allowed_policies`.
- Unit counts are parsed from words like `units`, `patients`, `clients`,
  `participants`, `users`, `students`, or `trajectories`.
- Horizons are parsed from phrases like `over 8 periods`, `for 8 weeks`, or
  `for 8 months`.
- Measurements are matched against visible measurement names. Hidden latents
  are rejected during validation, not silently ignored.

### `framework_code/run_agent_batch_rpg.py`

Status: planned after world-side smoke tests.

Purpose:

- RPG-specific batch entrypoint, or a staging wrapper before modifying
  `run_agent_batch.py`.
- Load `rpg_v1` world JSON directly.
- Instantiate `RPGSimulator` and `WorldModelRPG`.
- Build `Question` objects from RPG world files while keeping answers/metadata
  evaluator-only.

Reason to stage separately:

- We can test RPG behavior end to end without risking the working BN pipeline.
- Once stable, this logic can become a schema branch inside `run_agent_batch.py`.

### `framework_code/evaluate_rpg.py`

Status: planned.

Purpose:

- Score `rpg_*` answers consistently.
- Support answer types: policy id, variable name, dose-level/policy id, and
  eventually structured rejection/custom policy.
- Reuse gold metadata from `questions[*].metadata.gold` but never expose it to
  scientist agents.

Initial scoring:

- Normalize text, extract exact `policy_[A-Z]` ids and visible variable names.
- Mark correct if extracted answer matches oracle gold.
- Keep richer metadata for later partial-credit analysis.

### `framework_code/notebooks/rpg_world_demo.ipynb`

Status: initial implementation exists.

Purpose:

- Human-inspectable notebook showing the RPG world side step by step.
- Acts as a smoke test and a slide-friendly artifact.

Notebook sections:

1. Load one world JSON.
2. Display public world info only.
3. Show hidden fields are not in public info.
4. Run an observational trajectory.
5. Run one policy rollout.
6. Compare policies over the validated horizon.
7. Show shorter-horizon behavior for myopic reasoning.
8. Demonstrate validation rejection for hidden measurements.
9. Plot or tabulate final-period policy means.
10. Compare the manual result with the saved oracle answer for inspection only.

## RPG experiment semantics

An RPG experiment is a typed trajectory rollout, not a static BN intervention:

```json
{
  "mode": "policy_comparison",
  "policy_ids": ["policy_A", "policy_B"],
  "n_units": 200,
  "horizon": 8,
  "measurements": ["OutcomeProxy", "HarmProxy"],
  "seed": 12345
}
```

The simulator samples `n_units` independent units, applies each selected policy
for `horizon` discrete periods, updates hidden simulator state each period, and
returns only requested visible measurements plus public action columns. Budget
accounting should use `unit_period_rows` by default:

```text
sample_rows = n_units * horizon * number_of_policies
sample_cells = sample_rows * returned_columns
```

Returned columns are `condition_id`, `unit_id`, `time`, `policy_id`, public
action columns, and requested measurements.

For RPG v1, query horizons should be positive and no larger than the validated
world horizon. Shorter horizons are useful for diagnosing myopic reasoning;
longer horizons should require a separate validation regime.

## Ideal fake-scientist interaction

For the world-side milestone, the scientist can be faked by hand or notebook
cells. A good interaction should look like this:

1. The world model shows only public context:

```text
Story: A subscription platform analytics team studies longitudinal policy
experiments over 8 campaign weeks...

Observed measurements: ChurnRiskScore, ComplaintSignal, RetentionIndex, ...
Actions: MessageIntensity, QuietModeSupport
Allowed policies: policy_A, policy_B, policy_C, ...
Question: Which candidate policy should be deployed...
```

2. The fake scientist asks a broad observational/default query:

```text
Give me 50 observational trajectories over 4 periods measuring
RetentionIndex, ComplaintSignal, MessageIntensity, QuietModeSupport.
```

3. The world model parses this to:

```json
{
  "mode": "observational_trajectory",
  "n_units": 50,
  "horizon": 4,
  "measurements": ["RetentionIndex", "ComplaintSignal"]
}
```

4. The fake scientist compares candidate policies:

```text
Compare policy_A, policy_B, and policy_C for 300 units over 8 periods
measuring RetentionIndex and ComplaintSignal.
```

5. The returned CSV has rows like:

```text
condition_id,unit_id,time,policy_id,MessageIntensity,QuietModeSupport,RetentionIndex,ComplaintSignal
0,0,1,policy_A,High,Off,...
0,0,2,policy_A,High,Off,...
...
```

6. The fake scientist groups by `policy_id` and final `time`, estimates target
and harm/safety, then asks one focused follow-up if needed.

7. The final answer is a policy id or variable name, matching the question's
declared answer format.

This interaction tests the intended agent skills: choosing useful policy
comparisons, remembering that rows are longitudinal, avoiding hidden variables,
and not over-trusting short-horizon/proxy metrics.

## Schema compatibility notes

- Current `json_converter.py` and `BNSimulator` are not appropriate for RPG
  worlds because there are no static CPDs.
- `run_agent_batch.py` currently assumes a BN conversion step before each
  world. It should branch on `schema_version`.
- Existing advanced evaluators rely on `question.metadata.gold`; RPG worlds
  should keep that convention, even though the gold block now comes from oracle
  rollouts instead of exact BN inference.
- Keep top-level `variables`, `story`, and `questions` fields so old dataset
  inspection scripts can still summarize files.
- Do not expose `questions[*].answer` or `questions[*].metadata` to scientist
  agents. Those fields are dataset/evaluator metadata only.
- `run_agent_batch.py::setup_world_model` currently always converts JSON to
  BIF. It needs a `schema_version == "rpg_v1"` branch that instantiates
  `RPGSimulator` plus an RPG-specific world model/parser instead.
- The existing `schemas.ParsedQuery` only has observational/interventional BN
  query types. RPG should either add parallel RPG query dataclasses or create a
  small adapter so the old `QueryResult` XML contract can be reused without
  pretending policy rollouts are static `do()` operations.

## Open design choices

- Whether adaptive candidate policies may condition on noisy observed proxies
  only, or whether some benchmark questions may allow oracle/clinical latent
  state access. The safer default is observed-proxy policies only.
- Whether agents may propose new policies, or must choose from finite candidate
  policies. RPG v1 should use finite candidates for clean scoring.
- Whether to expose action schedules directly or only policy descriptions. For
  early datasets, expose policy ids and action schedules to avoid parser
  ambiguity.
- Whether trajectory queries should support custom horizons shorter than the
  world horizon. This would help diagnose myopic failures but adds parser
  complexity.

## Next implementation steps

1. Add an RPG branch in `run_agent_batch.py` world setup, or create a staged
   `run_agent_batch_rpg.py` that uses Bedrock Opus 4.8 for the scientist side.
2. Update scientist prompts so agents ask for policy rollouts/comparisons over
   time rather than BN-style `do(X=x)` samples.
3. Update evaluators/extractors for `rpg_*` question types.
4. Add end-to-end smoke tests using 1-2 RPG worlds before running a full batch.
5. Log every query seed, policy set, horizon, measurements, row count, and CSV
   path so agent rollouts can be replayed exactly.

## 2026-06-03 v3 story-hidden-cause pipeline update

The v3 direction is now different from the earlier dynamic-policy RPG worlds.
The first working archetype is:

```text
story_hidden_cause_discovery
```

This is one archetype, not one hardcoded question. The concrete setting should
come from a generated **story template**. The template supplies:

- topic and subdomain;
- rich story with the hidden object/process/state mentioned casually;
- hidden continuous cause name and accepted natural-language aliases;
- observed continuous variables with required semantic roles;
- binary public actions with required semantic roles;
- scoring terms for semantic answer checking;
- gold explanation bullets for evidence, alternatives, decisive test, and
  targeted action.

`dataset_generation_code/world_gen_rpg.py` now supports:

```bash
python3 dataset_generation_code/world_gen_rpg.py \
  --outdir dataset_generation_code/all_out_rpg/out_rpg_v3_story_hidden_llm \
  --distribution '{"story_hidden_cause_discovery":6}' \
  --use-llm-templates \
  --llm-model us.anthropic.claude-opus-4-7 \
  --llm-template-count 6 \
  --oracle-n 50000
```

For reproducible runs after inspecting a generated template:

```bash
python3 dataset_generation_code/world_gen_rpg.py \
  --outdir dataset_generation_code/all_out_rpg/out_rpg_v3_story_hidden_saved \
  --distribution '{"story_hidden_cause_discovery":6}' \
  --template-json dataset_generation_code/all_out_rpg/out_rpg_v3_story_hidden_llm/rpg_story_hidden_templates.json \
  --oracle-n 50000
```

The generated worlds save `rpg_story_hidden_templates.json` beside the world
files. This file is important for slide/debug review because it contains both
the LLM prompt and the concrete structured templates used to make the worlds.

### Required role contract

Observed-variable roles:

- `context_intensity`
- `visible_trigger`
- `exposure_modifier`
- `maintenance_gap`
- `mechanism_proxy_primary`
- `mechanism_proxy_secondary`
- `location_effect`
- `secondary_outcome`
- `outcome`
- `alternative_proxy_primary`
- `alternative_proxy_secondary`
- `alternative_proxy_tertiary`
- `diagnostic_test_signal`

Action roles:

- `targeted_fix_primary`
- `targeted_fix_secondary`
- `diagnostic_test`
- `alternative_fix_primary`
- `symptom_mitigation`
- `partial_reroute`
- `alternative_fix_secondary`
- `distractor_check`
- `weak_buffer`
- `cosmetic_action`

The simulator uses these roles, not hardcoded variable names, to build the
continuous causal relations. This means a clinic world, factory world, or farm
world can share the same archetype while using different public names and
different stories.

### Action space

Each v3 story-hidden world currently exposes 10 binary public actions and
allows up to 3 simultaneous actions per interventional sample. The oracle now
scores all combinations up to that cap:

```text
1 + C(10,1) + C(10,2) + C(10,3) = 176 candidates
```

The agent prompt says these are available actions, not known causal variables.
The scientist must decide which actions are diagnostic tests, targeted fixes,
decoys, or alternative-cause probes based on semantics and queried evidence.

### Pipeline compatibility

Files touched for v3 compatibility:

- `dataset_generation_code/world_gen_rpg.py`: role-based template generator,
  optional LLM-template generation, combinatorial oracle action scoring.
- `framework_code/scientist_agent_rpg.py`: prompt tells the scientist to infer
  hidden story-implied causes and test them with mechanism-targeted queries.
- `framework_code/simulator_rpg.py`: `latent_cause_hypothesis` scorer uses
  template-specific aliases/scoring terms and enforces the intervention cap.
- `framework_code/world_model_rpg.py`: final answer is accepted only if the
  run actually contains a mechanism-verifying interventional query.
- `framework_code/evaluate_rpg.py`: rescoring uses the same template-specific
  trajectory requirements.

The real agent should not be told the hidden role labels. It sees only the
story, observed measurement catalog, public action catalog, budget, and answer
schema. The role labels live in hidden simulator config and oracle metadata.
