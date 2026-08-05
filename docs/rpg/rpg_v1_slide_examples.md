# RPG v1 Slide Examples

This note picks three representative `rpg_v1` worlds from
`dataset_generation_code/all_out_rpg/out_rpg_v1`. They are meant to illustrate
the scientific discovery process we want agents to learn: ask time-aware
experiments, compare the right policies, avoid hidden variables, and separate
visible proxy movement from the actual answer.

## QA Snapshot

- Dataset size: 60 worlds.
- Archetype balance: 12 worlds each for `delayed_harm`, `dose_response`,
  `proxy_metric_hacking`, `latent_mediator`, and `heterogeneous_response`.
- Subtype balance: `dose_response` is 4/4/4 across `inverted_u`,
  `minimum_effective`, and `saturation`; `latent_mediator` is 6/6 across
  `mediated_only` and `direct_and_mediated`.
- Visible measurement richness: 9-10 observed measurements per world, plus
  explicit action variables and hidden latent-state variables.
- Visibility split: top-level `variables` and `visible` contain only
  agent-facing observed/action variables; latent simulator state is under
  `hidden.latent_variables` and is rejected by the runtime world model.
- Validator status: 60/60 accepted, 0 failed signature checks.
- Independent oracle recheck: 0/60 gold-answer flips with 100,000 fresh
  rollouts per world.
- Policy ids are neutralized per world (`policy_A`, `policy_B`, etc.) so the
  answer cannot be learned from a fixed archetype-specific id.
- Discovery protocol: worlds now expose a visible budget of 24,000 returned
  cells total, 8,000 returned cells per query, default 40 units, max 400 units,
  and at most 8 successful queries.

## LLM And Agent Target

No external LLM was used to generate or audit the current RPG dataset. The
RPG world-side path is deterministic/local: `world_gen_rpg.py`,
`audit_rpg.py`, `simulator_rpg.py`, and `world_model_rpg.py`.

For the future scientist-agent pipeline, the target model should be Bedrock
Opus 4.8:

```bash
--scientist-backend bedrock \
--scientist-model us.anthropic.claude-opus-4-7
```

The current BN batch runner already has Bedrock hooks, but it still assumes
BN worlds. The RPG runner needs a schema branch that instantiates
`RPGSimulator` and `WorldModelRPG` instead of the BN simulator/world model.

## Agent-Side Pipeline Ideas

- Add `run_agent_batch_rpg.py` as a staged runner before touching the stable
  BN batch path.
- Use Bedrock Opus 4.8 for the scientist agent by default.
- Instantiate `RPGSimulator` and `WorldModelRPG` when `schema_version` is
  `rpg_v1`.
- Show agents only `RPGWorldInfo`: story, visible measurements, action
  variables, candidate policies, query modes, horizon, and question.
- Never pass `hidden`, `oracle`, `validators`, `questions[*].answer`, or
  `questions[*].metadata` into the scientist prompt.
- Replace BN/do-calculus language in prompts with longitudinal experiment
  language: trajectories, policies, horizons, time periods, final-period
  outcomes, safety rates, and sample budgets.
- Teach the agent the three query modes: `observational_trajectory`,
  `policy_rollout`, and `policy_comparison`.
- Encourage a small broad first comparison across candidate policies unless
  the question clearly asks for a focused mechanism test.
- Encourage full-horizon experiments before final answers; short-horizon
  rollouts are diagnostic only.
- Require agents to compute final-period summaries from CSVs themselves rather
  than asking the world model for conclusions.
- Add prompt examples for common failure modes: proxy metric hacking, delayed
  harm, nonlinear dose response, latent mediator decoys, and heterogeneous
  treatment response.
- Log every query text, parsed query, seed, horizon, policy set, measurements,
  row count, CSV path, and sample budget state.
- Add `evaluate_rpg.py` with answer extraction for policy ids and visible
  variable names.
- Score answers using `questions[*].metadata.gold` only in evaluator code, not
  in the agent-visible world context.
- Add replay tools that can rerun an agent trajectory exactly from logged query
  seeds and CSV paths.
- Use the visible cell-count budget so measuring all variables and comparing
  every policy is costly.
- Add a recovery path where invalid hidden-variable requests produce a clear
  error and the scientist agent is prompted to retry with visible measurements.
- Add notebook-style smoke tests for one world per RPG archetype before any
  full batch run.
- Keep old BN result formats separate from RPG result formats until the RPG
  evaluator is stable.
- Add failure-analysis labels for RPG-specific mistakes: myopic horizon,
  proxy-overtrust, missing dose level, insufficient policy coverage,
  hidden-variable request, and final-time aggregation error.

## Budgeted Discovery Protocol

The current world side no longer behaves like an unlimited policy-evaluation
oracle. Each world exposes:

| Budget field | Value |
|---|---:|
| `sample_accounting` | `cells` |
| `max_total_samples` | 24000 |
| `max_samples_per_query` | 8000 |
| `default_units` | 40 |
| `max_units` | 400 |
| `max_queries` | 8 |

One returned cell is one dataframe entry. A query that asks for more policies,
more time periods, or more measurements costs more. This makes measurement
choice and experiment design matter. A good agent should run small exploratory
experiments, inspect results, then spend the remaining budget on focused
full-horizon follow-ups.

## Natural-Language Query Contract

`WorldModelRPG` currently accepts explicit JSON and a small natural-language
grammar. If the simple parser fails and an LLM object is provided, it can fall
back to an LLM parser, but the current notebook/examples do not need that.

Main supported utterances:

| Scientist says | Parsed mode | Important fields |
|---|---|---|
| `give observational trajectories for 50 units over 4 periods measuring X and Y` | `observational_trajectory` | Uses `visible.default_observational_policy_id`; no explicit policy ids. |
| `run policy_A for 100 patients over 8 periods measuring X, Y` | `policy_rollout` | One policy id, unit count, horizon, visible measurements. |
| `compare policy_A, policy_B, and policy_C for 30 users over 8 periods measuring X and Y` | `policy_comparison` | Two or more explicit policies; focused measurements keep the query cheap. |
| `compare all policies for 10 users over 8 periods measuring all` | `policy_comparison` | Expands to every visible candidate policy; all-measurement sweeps are expensive and must stay small. |
| `<json>{"mode":"policy_comparison", ...}</json>` | exact typed query | Deterministic path for tests and agent recovery. |
| `run policy_A measuring TrueLearning` | validation error | Hidden/unknown measurements are rejected before execution. |

Parsing rules:

- Unit count is read from phrases like `300 units`, `300 patients`, `300
  clients`, or `300 participants`.
- Horizon is read from `over 8 periods`, `for 8 weeks`, `for 8 months`, etc.
- Policy ids are normalized to `policy_A`, `policy_B`, ...
- Measurements must match visible measurement names. Hidden latents such as
  `HomeRecoveryStatus`, `JobSkillGrowth`, or `HousingStability` are not valid.
- The returned CSV has one row per `(condition_id, unit_id, time)` plus
  `policy_id`, public action columns, and requested visible measurements.
- The world model enforces the visible budget by default. Large brute-force
  requests fail before simulation.

## New RPG File Slide Explanations

| File | Slide role |
|---|---|
| `dataset_generation_code/world_gen_rpg.py` | Generates the RPG world JSONs, including visible catalogs, hidden simulator configs, neutral policies, oracle scores, validators, and gold answers. This is the mechanism source of truth. |
| `dataset_generation_code/audit_rpg.py` | Quality gate for the dataset. It checks archetype/subtype balance, visibility contracts, difficulty signatures, sample-row validity, and optional fresh oracle rechecks. |
| `framework_code/schemas_rpg.py` | Defines typed RPG queries/results/world info without pretending the worlds are Bayesian networks. This is the contract between scientist, world model, and simulator. |
| `framework_code/simulator_rpg.py` | Runtime simulator. It loads one RPG world, validates typed trajectory experiments, calls the generator rollout function, and returns longitudinal dataframes. |
| `framework_code/world_model_rpg.py` | Agent-facing interface. It parses natural language or JSON into RPG queries, enforces visible cell/query budgets and visibility, executes the simulator, and writes CSV results. |
| `framework_code/notebooks/rpg_world_demo.ipynb` | Human inspection notebook. It shows public context, parsed queries, accepted rollouts, hidden-variable rejection, slide examples, and oracle-only checks. |
| planned `framework_code/run_agent_batch_rpg.py` | Future Bedrock Opus 4.8 batch runner for scientist agents on RPG worlds. It should keep the BN runner stable while we iterate. |
| planned `framework_code/evaluate_rpg.py` | Future answer extractor/scorer for RPG answer types: policy ids, variable names, dose choices, and eventually structured policy descriptions. |

## Example 1: Delayed Harm / Post-Discharge Follow-Up

World file:
`dataset_generation_code/all_out_rpg/out_rpg_v1/world_rpg_delayed_harm_post_discharge_follow_up_intensity_seed29703.json`

Question:

> Which candidate policy should be deployed for 8 post-discharge weeks to
> maximize final RecoveryStatusReport while keeping the risk of severe
> CareBurdenSurvey below 15%? Answer with one policy_id.

Visible/hidden split:

- Visible measurements: `ReadmissionRiskScore`, `CareBurdenSurvey`,
  `RecoveryStatusReport`, `VisitCompletionLog`, `BaselineProfileScore`,
  `ResilienceIndex`, `ResourceUseIndex`, `ContactLoadIndex`,
  `ShortTermResponseSignal`, `BurdenTrendSignal`.
- Visible actions: `FollowupIntensity`, `NavigationSupport`.
- Hidden latents: `ReadmissionRisk`, `CareBurden`, `FollowupParticipation`,
  `HomeRecoveryStatus`.
- Gold answer: `policy_C`.
- Gold margin: 5.48 final target points over the next safe policy.

Candidate policies:

| Policy | Mechanism |
|---|---|
| `policy_A` | Standard action level every period. |
| `policy_B` | Standard level plus support every period. |
| `policy_C` | High level while latest `CareBurdenSurvey` is below a threshold, otherwise standard plus support. |
| `policy_D` | High action level every period. |
| `policy_E` | High level for two periods, then standard. |

Ideal successful rollout:

1. The scientist first checks short-horizon temptation:

   ```text
   Compare policy_A, policy_B, policy_C, policy_D, and policy_E for
   40 patients over 2 periods measuring RecoveryStatusReport and
   CareBurdenSurvey seed 2101
   ```

   At period 2, the high-intensity policies look similarly good and none
   violates the severe-burden threshold. This is useful but not decisive:

   | Policy | Final RecoveryStatusReport | Severe burden rate |
   |---|---:|---:|
   | `policy_C` | 63.86 | 0.000 |
   | `policy_E` | 63.40 | 0.000 |
   | `policy_D` | 62.23 | 0.000 |
   | `policy_B` | 59.33 | 0.000 |
   | `policy_A` | 55.49 | 0.000 |

2. The scientist then runs the decision horizon and includes the safety
   measurement:

   ```text
   Compare policy_B, policy_C, and policy_D for
   40 patients over 8 periods measuring RecoveryStatusReport and
   CareBurdenSurvey seed 2102
   ```

   At period 8, `policy_D` has the largest target but violates the safety
   constraint for essentially all patients. `policy_C` is the best safe policy:

   | Policy | Final RecoveryStatusReport | Mean CareBurdenSurvey | Severe burden rate |
   |---|---:|---:|---:|
   | `policy_D` | 83.43 | 75.71 | 1.000 |
   | `policy_C` | 77.03 | 33.47 | 0.000 |
   | `policy_B` | 72.18 | 1.65 | 0.000 |

3. The scientist spends a final focused query to check the other plausible
   safe option:

   ```text
   Compare policy_B, policy_C, and policy_E for
   40 patients over 8 periods measuring RecoveryStatusReport and
   CareBurdenSurvey seed 2103
   ```

   | Policy | Final RecoveryStatusReport | Mean CareBurdenSurvey | Severe burden rate |
   |---|---:|---:|---:|
   | `policy_C` | 80.26 | 33.04 | 0.000 |
   | `policy_B` | 73.88 | 1.16 | 0.000 |
   | `policy_E` | 71.06 | 37.75 | 0.025 |

Successful conclusion:

> `policy_C`, because it preserves most of the recovery gain while satisfying
> the severe-burden constraint.

Failing rollout to show on slides:

- Failure mode 1: stop after the 2-period query and choose `policy_D` or
  `policy_E` because the delayed burden has not appeared yet.
- Failure mode 2: request hidden `HomeRecoveryStatus`; the world correctly
  rejects it with `unknown or hidden measurements requested`.

## Example 2: Dose Response / Job Training Hours

World file:
`dataset_generation_code/all_out_rpg/out_rpg_v1/world_rpg_dose_response_job_training_hours_seed6525.json`

Question:

> Which dose policy should be used for 8 training weeks to maximize final
> ReadinessAssessment? Explore the dose levels rather than assuming more is
> always better. Answer with one policy_id.

Visible/hidden split:

- Visible measurements: `SkillCheckScore`, `ScheduleStrainSurvey`,
  `ReadinessAssessment`, `WorkshopAttendance`, `BaselineReadinessScore`,
  `ScheduleLoadIndex`, `ResourceUseIndex`, `ShortTermGainSignal`,
  `RecoveryWindowIndex`, `AttendanceTrace`.
- Visible action: `TrainingDose`.
- Hidden latents: `JobSkillGrowth`, `ScheduleStrain`, `PlacementReadiness`.
- Subvariant: `inverted_u`.
- Gold answer: `policy_C`.
- Gold margin: 18.67 final target points over the runner-up.

Candidate policies:

| Policy | Dose |
|---|---|
| `policy_B` | none |
| `policy_A` | low |
| `policy_C` | medium |
| `policy_D` | high |

Ideal successful rollout:

```text
Compare all policies for 35 participants
over 8 periods measuring ReadinessAssessment and ScheduleStrainSurvey
seed 2202
```

Final-period estimates:

| Policy | Final ReadinessAssessment | Final ScheduleStrainSurvey |
|---|---:|---:|
| `policy_C` | 79.81 | 30.39 |
| `policy_A` | 63.31 | 11.86 |
| `policy_D` | 52.69 | 72.83 |
| `policy_B` | 43.27 | 6.81 |

Successful conclusion:

> `policy_C`, because the response curve is inverted-U: medium training dose
> creates the best readiness, while high dose creates strain and lower final
> readiness.

Failing rollout to show on slides:

```text
Compare policy_A and policy_D for 35 participants over 8 periods measuring
ReadinessAssessment and ScheduleStrainSurvey seed 2203
```

This only compares low and high doses. It finds `policy_A` beats `policy_D`
but cannot discover the true optimum `policy_C`. The failure is not statistical
noise; it is bad experimental coverage.

## Example 3: Latent Mediator / Reentry Support Services

World file:
`dataset_generation_code/all_out_rpg/out_rpg_v1/world_rpg_latent_mediator_reentry_support_services_seed6222.json`

Question:

> Which intermediate measurement is on the actual pathway from ReentryProgram
> to final StabilityReview: HousingStabilityCheck or AppointmentRecallScore?
> Answer with the variable name.

Visible/hidden split:

- Visible measurements: `HousingStabilityCheck`, `AppointmentRecallScore`,
  `StabilityReview`, `BaselineStatusScore`, `SecondaryPathwaySurvey`,
  `AdministrativeFamiliarityScore`, `ProgramExposureLog`,
  `ShortTermOutcomeSignal`, `AdministrativeTraceIndex`.
- Visible action: `ReentryProgram`.
- Hidden latents: `HousingStability`, `AppointmentFamiliarity`,
  `CommunityStability`.
- Subvariant: `mediated_only`.
- Gold answer: `HousingStabilityCheck`.
- Gold margin: 35.05 outcome-effect points over the decoy pathway.

Candidate policies:

| Policy | Mechanism |
|---|---|
| `policy_B` | Current process unchanged. |
| `policy_C` | Primary program variant X every period. |
| `policy_A` | Decoy-focused program variant Y every period. |

Ideal successful rollout:

1. Compare current process against the primary program:

   ```text
   Compare policy_B and policy_C for 40 clients over 8 periods measuring
   HousingStabilityCheck, AppointmentRecallScore, and StabilityReview seed 2303
   ```

   Both candidate intermediates move under the primary program, so this query
   alone is not enough to identify the pathway:

   | Policy | Final StabilityReview | Final HousingStabilityCheck | Final AppointmentRecallScore |
   |---|---:|---:|---:|
   | `policy_B` | 24.41 | 24.26 | 28.10 |
   | `policy_C` | 61.39 | 82.91 | 81.84 |

2. Add the decoy-focused program:

   ```text
   Compare policy_A, policy_B, and policy_C for 40 clients over 8 periods
   measuring HousingStabilityCheck, AppointmentRecallScore, and
   StabilityReview seed 2304
   ```

   Now the mechanism becomes identifiable:

   | Policy | Final StabilityReview | Final HousingStabilityCheck | Final AppointmentRecallScore |
   |---|---:|---:|---:|
   | `policy_A` | 29.36 | 27.61 | 94.25 |
   | `policy_B` | 28.56 | 24.98 | 29.30 |
   | `policy_C` | 61.18 | 82.84 | 80.58 |

Successful conclusion:

> `HousingStabilityCheck`, because `policy_C` moves housing and final
> stability together, while the decoy-focused `policy_A` moves
> `AppointmentRecallScore` strongly without improving `StabilityReview`.

Failing rollout to show on slides:

- Stop after comparing `policy_B` and `policy_C`; both intermediates move, so
  the pathway is ambiguous.
- Pick `AppointmentRecallScore` just because it is highly responsive under the
  primary program; the decoy-focused policy shows that responsiveness alone is
  not causal pathway evidence.
