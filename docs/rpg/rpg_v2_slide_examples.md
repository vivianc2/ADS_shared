# RPG v2 (static) — lessons, fixes, and watchpoints

Companion to [worldgen_rpg_plan_static_partial_observation.md](worldgen_rpg_plan_static_partial_observation.md).
This file captures the things that were non-obvious or that we got wrong on
the first pass while implementing the static, partially-observed RPG
generator in [world_gen_rpg.py](world_gen_rpg.py) (`SCHEMA_VERSION_STATIC =
"rpg_static_v2"`). Use this as a checklist before adding new archetypes.

## All-six RPG static run: slide-ready archetype summary

Source run: `results/rpg_static_all6_opus/rpg_static_all6_opus.json`
with per-world logs in `results/rpg_static_all6_opus/rpg_agent_logs/`.
This is the first full run with all six static archetypes represented
six times each.

| Archetype | Scientific question being tested | Answer schema | Run result | How Opus approached it in the logs | Current lesson |
|---|---|---|---:|---|---|
| `hidden_cause` | Which intervention improves the underlying condition when several observed proxies and semantically plausible decoys exist? | `intervention_with_hypothesis` | 6/6 | Starts with observational target/proxy data, then tests candidate interventions one by one and chooses the largest target shift. | Mechanically healthy, but some examples may be easy once the right-looking action is tested. |
| `confounded_action` | Is the conventional observational belief wrong because treatment is assigned to harder cases? | `intervention_with_hypothesis` | 6/6 | Checks passive assignment/outcome/severity, then randomizes treatment levels under `do(.)` and reverses the observational conclusion. | Strong scientific behavior; the agent uses interventions rather than passive correlations. |
| `mechanism_chain` | Which stage in a sequential pipeline is the bottleneck, and why are the other stages not the leverage point? | `intervention_with_hypothesis` | 6/6 | Reads stage means, identifies the lowest stage, then confirms by intervening on the suspected bottleneck and alternatives. | Valid, but can collapse toward "pick the lowest visible stage" unless decoys are strengthened. |
| `negative_control` | Are heavily discussed interventions actually useful, or are observational benefits selection bias? | `intervention_with_hypothesis` with `{}` allowed | 6/6 | Tests trendy/conventional/evidence-labeled interventions under randomized assignment and returns no intervention when effects are tiny. | Good no-action behavior; the agent does not force an intervention just because knobs exist. |
| `hidden_subtype` | Can the agent learn a conditional policy where different subgroups need opposite interventions? | `conditional_policy` | 6/6 | Runs each intervention while measuring the screen variable, compares screen-outcome correlations, and picks opposite actions above/below a threshold. | Strong and semantically meaningful; currently tests estimating the matching rule more than discovering the branch variable. |
| `anomaly_discovery` | Can the agent infer a rare feature-signature rule from observational samples? | `anomaly_identification` | 0/6 | Uses many observational samples, reasons from pairwise correlations and tails, but lacks row-level computation over candidate rules. | Useful failure mode: this needs a coder agent or richer anomaly summaries. |

Overall: 30/36 accepted. All intervention/policy archetypes were solved;
all anomaly worlds failed. That split is useful for slides because it shows
the pipeline works while also exposing a clear limitation of the plain
scientist agent.

## Example 1: `hidden_cause` — orchard yield shortfall

- **Actual world:** `world_rpg_static_hidden_cause_stagnant_orchard_yield_seed5404`
- **Outcome:** accepted.
- **Gold:** `{"FieldTreatmentF": "on"}`
- **Agent answer:** `{"intervention": {"FieldTreatmentF": "on"}}`

The question asks which single intervention most improves the underlying
condition reflected by high `YieldShortfall` and related target
measurements. In the log, the agent first measured `YieldShortfall`,
`SoilMoistureDeficit`, and `PruningCompliance` under current practice, then
tested four candidate interventions:

| Query | Mode | Intervention | Key evidence from log |
|---:|---|---|---|
| 1 | observational | none | `YieldShortfall` mean about 61.0; decoy-looking variables correlate with the target. |
| 2 | interventional | `IrrigationBoost=on` | `YieldShortfall` worsened to about 65.9. |
| 3 | interventional | `FieldTreatmentF=on` | `YieldShortfall` improved to about 52.8; `LeafCanopyDamage` and `FruitQualityIndex` also moved in the right direction. |
| 4 | interventional | `PruningProgram=on` | `YieldShortfall` worsened to about 66.2. |
| 5 | interventional | `FoliarFeed=high` | modest improvement only, `YieldShortfall` about 57.4. |

**How the LLM approached it.** It compared the interventional mean shifts,
not just observational correlations. Its final reasoning was that
`FieldTreatmentF` likely addresses a soil/root-zone problem, because it
improved the primary outcome and the related canopy/fruit-quality
measurements together. This is a good trajectory: it treats variable names
as hypotheses and uses `do(.)` samples to reject plausible decoys.

**Slide takeaway.** This archetype works as intended when the agent tests
multiple candidates. The risk is that some true-action names are still
semantically suggestive, so future variants should keep decoys similarly
plausible and require intervention evidence.

## Example 2: `confounded_action` — ICU sedation intensity

- **Actual world:** `world_rpg_static_confounded_action_icu_sedation_intensity_seed5909`
- **Outcome:** accepted.
- **Gold:** `{"Sedation": "high", "EarlyMobilization": "on"}`
- **Agent answer:** `{"intervention": {"Sedation": "high", "EarlyMobilization": "on"}}`

The question says passive records make aggressive sedation look harmful.
The agent began with an observational query over `RecoveryScore`,
`AssignedSedationLevel`, and `APACHEProxy`, then ran randomized
interventional comparisons:

| Query | Mode | Intervention | Key evidence from log |
|---:|---|---|---|
| 1 | observational | none | `APACHEProxy` was strongly negatively correlated with `RecoveryScore` (`corr` about -0.52), indicating severity confounding. |
| 2 | interventional | `Sedation=high`, `EarlyMobilization=on` | `RecoveryScore` mean about 65.4. |
| 3 | interventional | `Sedation=off`, `EarlyMobilization=on` | `RecoveryScore` mean about 55.8. |
| 4 | interventional | `Sedation=low`, `EarlyMobilization=on` | `RecoveryScore` mean about 60.0. |

**How the LLM approached it.** It explicitly described the observational
belief as confounding by indication: sicker patients get more sedation, and
sicker patients recover worse regardless of sedation. The randomized
samples reversed the passive association, so the model chose high sedation
plus early mobilization.

**Slide takeaway.** This is one of the cleanest examples of the desired
"scientist agent" behavior: passive data generates a hypothesis, but the
answer comes from targeted interventions.

## Example 3: `mechanism_chain` — SaaS activation funnel

- **Actual world:** `world_rpg_static_mechanism_chain_saas_user_activation_funnel_seed26461`
- **Outcome:** accepted.
- **Gold:** `{"SignUpFlowOptimization": "on"}`
- **Agent answer:** `{"intervention": {"SignUpFlowOptimization": "on"}}`

The question asks for the bottleneck in a three-stage funnel with final
`RetentionScore`. The agent first inspected the three stage rates, then
tested the suspected bottleneck and both alternatives:

| Query | Mode | Intervention | Key evidence from log |
|---:|---|---|---|
| 1 | observational | none | `SignUpCompletionRate` about 44.3, while onboarding and first-week engagement were about 75. |
| 2 | observational | none | confirmed low sign-up and final `RetentionScore` about 48.9. |
| 3 | interventional | `SignUpFlowOptimization=on` | sign-up rose to about 64.8 and `RetentionScore` to about 68.4. |
| 4 | interventional | `FirstWeekEngagementCampaign=on` | `RetentionScore` fell to about 42.2. |
| 5 | interventional | `OnboardingProgramBoost=on` | `RetentionScore` fell to about 40.2. |

**How the LLM approached it.** It used the stage means to nominate the
bottleneck, then used interventions to verify that only the bottleneck
unlocked the downstream outcome. Its final hypothesis also explained why
the other stages were not the right target: they were already high and
their interventions did not improve retention.

**Slide takeaway.** The trajectory is valid, but this archetype is at risk
of being too readable from a single table of stage means. It may need
stronger observational traps, e.g. non-bottleneck stages that look
correlated with the outcome or bottlenecks that are not simply the lowest
mean in every visible sample.

## Example 4: `negative_control` — outpatient adjunct therapies

- **Actual world:** `world_rpg_static_negative_control_popular_outpatient_adjunct_therapies_seed6919`
- **Outcome:** accepted.
- **Gold:** `{}`
- **Agent answer:** `{"intervention": {}}`

The question asks whether any heavily discussed intervention should be
applied uniformly. The model initially made one invalid query by trying to
measure an intervention knob as if it were an observed measurement, then
recovered and ran the right randomized checks:

| Query | Mode | Intervention | Key evidence from log |
|---:|---|---|---|
| 1 | invalid | none | requested hidden/non-measurement `PopularAdjunctRegimen`; rejected by world model. |
| 2 | observational | none | `RecoveryDelayIndex` about 71.1; `HealthLiteracyScore` and `AppointmentEngagementIndex` correlated with better recovery. |
| 3 | interventional | `PopularAdjunctRegimen=on` | delay worsened to about 72.9. |
| 4 | interventional | `ClinicStandardOfCareAdjunct=on` | no meaningful effect, delay about 71.3. |
| 5 | interventional | `PublishedEvidenceProtocol=high` | delay worsened to about 72.5. |
| 6 | interventional | `PublishedEvidenceProtocol=low` | no meaningful effect, delay about 71.6. |

**How the LLM approached it.** It compared randomized effects against the
observational baseline and concluded that all apparent benefits were
selection bias: patients with more engagement/health literacy self-select
into adjuncts and recover faster regardless of the adjunct. It returned an
empty intervention dict.

**Slide takeaway.** This is a strong negative-control example. The agent is
not biased toward always acting, and the query validator catches the
intervention-as-measurement mistake cleanly.

## Example 5: `hidden_subtype` — headache treatment matching

- **Actual world:** `world_rpg_static_hidden_subtype_headache_management_subtype_matching_seed7424`
- **Outcome:** accepted.
- **Gold:** threshold 50 on `HeadachePhenotypeScreen`, above gets
  `MuscleRelaxantRegimen=on`, below gets `VasoactiveAgentRegimen=on`.
- **Agent answer:** threshold 42 on `HeadachePhenotypeScreen`, above gets
  `MuscleRelaxantRegimen=on`, below gets `VasoactiveAgentRegimen=on`.

The question asks for a conditional policy keyed on an observable screen.
The agent measured the screen and outcome under each treatment:

| Query | Mode | Intervention | Key evidence from log |
|---:|---|---|---|
| 1 | observational | none | `HeadachePhenotypeScreen` median about 58.4; weak overall relationship with outcome. |
| 2 | interventional | `VasoactiveAgentRegimen=on` | `corr(HeadachePhenotypeScreen, PainReductionScore)` about -0.47. |
| 3 | interventional | `MuscleRelaxantRegimen=on` | `corr(HeadachePhenotypeScreen, PainReductionScore)` about +0.52. |
| 4 | interventional | `BalancedCombinationRegimen=on` | near-zero interaction, mean outcome lower than targeted policy would imply. |

**How the LLM approached it.** It looked for treatment-effect
heterogeneity rather than one global best arm. It inferred that low-screen
patients benefit from vasoactive treatment and high-screen patients benefit
from muscle relaxant treatment. Its estimated threshold was 42 rather than
the stored gold threshold 50, but the policy was within oracle tolerance.

**Slide takeaway.** This archetype tests a richer scientific behavior than
single-action selection. It is working, though the branch variable is
currently quite visible in the question, so the hard part is estimating the
direction and threshold rather than discovering which variable matters.

## Example 6: `anomaly_discovery` — payment account anomaly hunt

- **Actual world:** `world_rpg_static_anomaly_discovery_payment_account_anomaly_hunt_seed8030`
- **Outcome:** rejected.
- **Gold:** `TransactionAmountQuantile > 60 AND DeviceTrustScore < 36`
- **Agent answer:** `DeviceTrustScore < 25 AND SecondaryRiskSignal > 75`
- **Score:** precision 1.0, recall about 0.14; rejected for low recall.

The question asks for a rare anomaly signature from observational samples.
The agent spent six successful queries on overlapping triples of features:

| Query pattern | What the agent saw | What happened |
|---|---|---|
| transaction/device/geo triples | weak negative correlation between `TransactionAmountQuantile` and `DeviceTrustScore` | This was actually part of the gold rule, but the agent did not lock onto it. |
| device/risk/geo triples | weak negative correlation between `DeviceTrustScore` and `SecondaryRiskSignal` | The agent interpreted this as the anomaly signature. |
| repeated overlapping samples | means around 47-51, SDs around 12-15, weak pairwise correlations | Summaries were not enough to search row-level conjunctions well. |

**How the LLM approached it.** It reasoned from domain semantics and
pairwise correlations: low device trust plus high secondary risk sounds
like fraud, so it proposed a very specific high-risk rule. That rule had
perfect precision on the audit batch but only caught a small slice of the
true anomaly group.

**Slide takeaway.** This is the clearest failure mode of the current plain
scientist agent. The world/oracle are not obviously broken; the interface
is weak for anomaly discovery because the model sees summaries instead of
computing over full rows. This motivates a coder agent or richer automatic
summaries such as tail co-occurrence counts and candidate-rule sweeps.

## QA snapshot of v2 (initial 12-world stress test)

- Archetypes shipped: `hidden_cause` and `confounded_action`. Three
  domain templates each, three mixture weights each (0.0, 0.10, 0.20).
- Validators passed: 12/12.
- Gold margin range: 4.2 to 27.0 (threshold `STATIC_MIN_GOLD_MARGIN = 4.0`).
- Recoverability band, across 30 seeds per world:
  - Small-budget observational naive hit rate: 0–7% (cap 40%).
  - Medium-budget interventional naive hit rate: 100% (floor 70%).
- No hidden-variable name leakage in `visible.story` /
  `visible.observed_variables[*].description` / `visible.question`.

## v2.1 redesign (2026-05-29)

A round of audit on the v2 pilot surfaced four real problems beyond the
ones from §"Bugs that fired" below. v2.1 fixes all four; the resulting
mechanism is meaningfully harder and resistant to LLM pattern-matching.

### v2.1 / 1 — Decoys were observationally uncorrelated with the target

**Symptom.** In the v2 pilot, `corr(latent_driver_proxy, target) ≈ 0.74`
across all hidden_cause worlds, while `corr(decoy_proxy_a, target) ≈ 0.00`.
A naive analyst with even a tiny observational sample could read off the
right proxy from a simple correlation table. That violates the plan's
intent that decoys be "just as visible" observationally.

**Root cause.** The decoy states were driven only by `BackgroundStrain`,
which had no path to the target. So decoy observations had zero causal or
confounded correlation with the target.

**Fix.** Introduced a fourth hidden variable, `BurdenSubstrate`, that
drives BOTH decoy states AND BaselineSeverity. Now both decoy proxies are
genuinely correlated with the target observationally (~0.20-0.35) through
the BurdenSubstrate confound — but `do(decoy_knob_a=on)` still has zero
target effect because the knob only moves the decoy state, not the
substrate. See `_static_hidden_cause_default_params` (new
`substrate_*` and `decoy_*_substrate_loading` params) and the rewritten
`_static_hidden_cause_sample_hidden`.

### v2.1 / 2 — Names leaked the answer to any LLM with general knowledge

**Symptom.** `SerologyMarker` ↔ `AntibioticCourse`,
`RootTissueAssay` ↔ `RootFungicideTreatment`,
`DeviceTLSFingerprintRarity` ↔ `TLSFingerprintGate` are so semantically
pair-matched that a strong LLM could "solve" the world without sampling
anything. That violates the §"Hard design rules" #7 ("no leakage through
names").

**Fix.** Renamed the latent_driver_proxy and the true_lever_knob to
neutral, non-pairmatching forms:

| Domain | Old latent proxy → true lever | New (v2.1) |
|---|---|---|
| Chronic GI | SerologyMarker → AntibioticCourse | IntakeSerumPanelB → OralRegimenM |
| Orchard | RootTissueAssay → RootFungicideTreatment | OrchardLabPanel3 → FieldTreatmentF |
| Payments | DeviceTLSFingerprintRarity → TLSFingerprintGate | RiskSignalCluster7 → RiskRuleR4 |

The decoy proxies and decoy knobs continue to share the prevailing-theory
vocabulary (StressInventoryScore ↔ StressReductionProgram, etc.) — so an
LLM that pattern-matches names is *actively misled* toward a decoy.

### v2.1 / 3 — Binary LatentDriverPresent → continuous LatentBurden

**Symptom.** Previous mechanism had `LatentDriverPresent` as a binary
indicator; the proxy was effectively `70 * indicator + noise`. That's
unrealistic and makes the recoverability discrete.

**Fix.** `LatentBurden` is now continuous on `[0, 100]`. Target severity
gets a sigmoid-thresholded contribution
`burden_effect * σ((burden - threshold)/soft) * (burden/100)`, so the
true lever's effect is biggest when burden sits in the active band and
tapers off when burden is low. The lever now reduces burden
*multiplicatively* (e.g. 80%-off) instead of full clearance.

The validator `latent_driver_present_rate ≥ 0.35` is replaced by
`latent_burden_active_band_rate ∈ [0.30, 0.85]` — the share of the
population sitting above the activation threshold.

### v2.1 / 4 — Latent_driver_proxy was too clean

**Symptom.** Even with the new continuous mechanism, the
latent_driver_proxy correlation with target was ~0.85 because the
observation noise SD (~14) was small relative to LatentBurden's range.

**Fix.** Bumped `latent_driver_proxy_sd` from `[11, 16]` to `[18, 24]`.
Observed correlations now land in `[0.30, 0.45]`. A new validator
`true_proxy_calibrated` rejects worlds whose proxy↔target correlation
exceeds 0.75 in absolute value.

### v2.1 pilot QA snapshot

12 worlds, regenerated in `out_rpg_static_v2_pilot_12/`:

- 12/12 pass the validation playbook including the new
  `decoys_tempting_obs` and `true_proxy_calibrated` checks.
- Latent-proxy↔target correlations: range `[0.35, 0.41]`.
- Decoy-proxy↔target correlations: range `[0.15, 0.34]`.
- Gold margins: range `[4.2, 16.0]`.
- Recoverability: small `0.00-0.03`, medium `1.00`.
- Gold-knob position distribution (anchored shuffle now wired through
  every level of the call chain):
  - hidden_cause: positions 0/1/2/3 each get 2/2/1/1 worlds.
  - confounded_action: positions 0/1 each get 3/3 worlds.

### `audit_rpg.py --static`

Added a new audit mode that reads `schema_version == "rpg_static_v2"`
worlds and runs:

1. Schema check.
2. Hidden-name leakage substring check on every visible-text field.
3. Public/hidden split verification (no hidden variable names in
   `visible.*`).
4. All-validators-passed re-check against the stored signature.
5. Recoverability-band bounds (`small ≤ 0.40`, `medium ≥ 0.70`).
6. Proxy-correlation calibration (`|true_proxy_corr| ≤ 0.80`, max decoy
   correlation `≥ 0.15`).
7. Fresh oracle recheck (re-runs `_static_oracle_score` with a different
   seed and asserts the gold intervention is stable).
8. Per-archetype gold-position distribution across the dataset.

Run with `python audit_rpg.py --outdir <path> --static --summary-only`
(static mode is auto-selected when the first world in the dir has
`schema_version == "rpg_static_v2"`).

### `framework_code/simulator_rpg.py` — runtime is ready

A `StaticRPGSimulator` class is in place (selected by `schema_version`).
Public surface:

```python
sim = StaticRPGSimulator.from_json("world_*.json")
sim.public_world()                      # the visible block only
sim.run_query(StaticRPGParsedQuery(...))  # returns StaticRPGQueryResult
sim.score_answer({"intervention": {...}, "hypothesis": "..."})
```

Smoke-tested against a v2.1 hidden_cause world:
- `observational_sample` returns rows with `query_mode` + the requested
  measurement columns under current-practice assignment.
- `interventional_sample` returns rows with `do_<knob_name>` columns plus
  the requested measurements.
- `inspect_unit` returns one freshly-sampled unit (case-conditional).
- `score_answer` accepts the gold intervention (`OralRegimenM=on`) and
  rejects decoys (`StressReductionProgram=on`).
- Submitting a non-intervenable name (`NoSuchKnob`) is rejected with
  a clear error message.

---

## Bugs that fired and how they were fixed

### 1. Naive small-budget baseline was running uniform `do()`, so every world looked trivially easy

**Symptom.** First smoke test:
`recoverability_small_budget=1.0` on every world. The validator failed
because gold was recovered every time.

**Root cause.** The original `_static_recoverability_band` called
`_static_apply` (uniform `do(iv)`) with `STATIC_RECOVER_SMALL_N = 80` units
per candidate. With a 14-unit-or-larger gold margin and ~8-unit observation
SD, SE per candidate is `8 / sqrt(80) ≈ 0.9`, and the gap dwarfs SE. Naive
mean over `do()` samples picks gold every time.

**Fix.** Split the band so the small-budget naive samples
*observational* data (current-practice assignment) and the medium-budget
naive samples *interventional* `do(iv)` data per candidate. The small
analyst now eats the same confounding the agent faces, and the band
asymmetry becomes the actual difficulty signal. Helpers introduced:
`_static_observational_outcomes`, `_static_obs_naive_pick`,
`_static_intv_naive_pick`.

**Watch for in future archetypes.** Any "recoverability" check must
distinguish what data the baseline gets, not just how many rows. Don't
re-use a single helper for both rungs of the band.

### 2. Decoy "looks tempting in obs" validator was checking the wrong direction

**Symptom.** `decoy_tempting_in_observational` failed with
`{"decoy_a": -3.3, "true_lever": -20.6}`. The decoy apparent effect on the
primary outcome was much weaker than the true lever's apparent effect, so
the validator (which wanted decoy ≤ true_lever in obs) rejected the world.

**Root cause.** The first hidden-cause draft had no confounder: assignment
of the decoy knob depended on `DecoyState_A`, which was independent of
`BaselineSeverity`. Under `do(decoy)` the target barely moved (designed),
and the apparent effect in obs was likewise small. Meanwhile the true
lever genuinely cleared the latent driver, so even with rare assignment its
apparent obs effect was large.

**Fix.** Introduced a hidden `HealthSeekingTrait` variable that:

- raises uptake of *every* non-baseline knob in current practice
  (`_static_hidden_cause_assignment` lines around the `assign_base_*` /
  `assign_slope_*` parameters), and
- lowers `BaselineSeverity` (via `baseline_healthseek_loading` in
  `_static_hidden_cause_sample_hidden`).

This makes every common-practice intervention look beneficial in
observational data while the true lever stays rare. The
`decoy_tempting_in_observational` validator was then redundant with the
recoverability band, so it was deleted.

**Watch for in future archetypes.** If the validator and the recoverability
band say the same thing, drop the validator — the band is harder to fool
because it is end-to-end.

### 3. Subtle Python bug in the naive-pick comparison

**Symptom.** Both archetypes died with
`'<' not supported between instances of 'float' and 'NoneType'`.

**Root cause.** I had:

```python
improvement = (score > best_score) if higher_better else (score < best_score)
if best_score is None or improvement:
    ...
```

Python evaluates `improvement` eagerly before the `if`, and `best_score` is
`None` on the first iteration, blowing up the comparison.

**Fix.** Split the conditional into an explicit `if best_score is None:`
branch in `_static_obs_naive_pick` and `_static_intv_naive_pick`.

**Watch for in future archetypes.** Any "pick the best" loop must
unconditionally accept the first candidate before doing any comparison.
This is generic enough that we should probably factor the pattern into a
shared helper if a third archetype needs it.

### 4. Param-removed-but-still-referenced

**Symptom.** Would have raised `KeyError: 'target_decoy_a_weight'` at runtime.

**Root cause.** While rewriting `_static_hidden_cause_default_params` I
removed `target_decoy_a_weight` (decoys no longer route through the target)
but the apply function still used it.

**Fix.** Removed the corresponding term in
`_static_hidden_cause_apply`.

**Watch for.** When tightening the mechanism, grep for every parameter key
you remove. Better still: the param dict and the apply function should be
edited in the same commit and both run through the smoke test before
moving on.

## Design traps I avoided (worth carrying forward)

### A. "Pick policy A/B/C/D" collapsed into a menu

The v1 dynamic generator gives agents a fixed `allowed_policies` list of
~5 named policies. The static design intentionally exposes
`intervenable_variables` instead and forces agents to build a `do(.)` dict.
This is a non-cosmetic change: it means the answer schema becomes a *dict*
that the oracle evaluates by running the simulator under the agent's
intervention, not a lookup.

If a future archetype is tempted to ship a "pick from these 4 packaged
interventions" question, that is a regression — it collapses the
discovery task into ranking. See §1 of the plan md.

### B. Primary observation == latent target

In `rpg_hidden_cause`, the symptom proxy (`SymptomReport` etc.) is *not*
the latent target the oracle scores. It's a noisy proxy with an additional
palliative channel: the weak knob (antacid-style) suppresses the
observation without moving the latent. This was deliberate — it means
agents who only ever sample one proxy can still get fooled even at the
medium budget if a future world strengthens the palliative effect.

For v2 the palliative was sized so the gold (antibiotic-style) still wins
on the primary obs at medium budget. If a future archetype wants a
metric-hacking-flavored variant, push the palliative larger and route the
medium-budget naive baseline through a *cross-check* proxy
(`SecondaryFraudSignal` / `EndoscopyFindingScore` / etc.).

### C. Mixture prior is for outlier robustness, not for partial observability

Tempting to crank `mixture_weight` to make worlds harder. Don't. The
mixture is bounded at `STATIC_MIXTURE_WEIGHTS = [0.0, 0.10, 0.20]`
because beyond ~0.20 the structured signature dilutes enough that even the
oracle starts misranking interventions. Partial observability is delivered
by noise SDs and multi-cause proxies. Tune those if you want more
difficulty.

### D. Common random numbers in the oracle

`_static_oracle_score` shares a single `hidden = sample_hidden(...)` across
all candidate interventions and only varies the seed for the apply step.
This gives much tighter pairwise comparisons (gold vs runner-up) than
independent draws would, and is what lets us run with
`STATIC_DEFAULT_ORACLE_N = 50000` instead of needing 200k+ to get the SE
under the margin. **Do not break this** in future archetypes — if an
archetype needs different hidden draws per candidate (e.g. selection-bias
archetypes that re-sample the population), add an explicit branch but
keep the default code path on CRN.

### E. Story names the wrong theory out loud

The hidden-cause stories ("Current guidelines focus on stress, diet, and
acid suppression." / "The risk team currently blames device-quality,
browser fingerprint, and account-age factors.") explicitly name the
entrenched but wrong theory. This is the H. pylori effect:
the prior is *seductive*, not absent. An archetype that drops this and
just describes a "population with a problem" loses most of the discovery
test.

## Things still on the watchlist

### Single-knob-only `intervention_with_hypothesis`

`max_intervention_knobs = 1` for hidden_cause and `2` for
confounded_action. The runtime simulator (not yet written for v2) must
enforce this in the agent's submitted answer. If an agent submits
`{"AntibioticCourse": "on", "StressReductionProgram": "on"}` for a
hidden_cause world, the scoring code must reject it as malformed, *not*
silently score it (which would let multi-knob brute-force win).

### Hypothesis is recorded but not scored

`answer.hypothesis` is preserved in the trajectory but not auto-scored in
v1. We should not pretend to score it with regex or keyword matching —
that ranges from useless to actively misleading. The plan reserves
hypothesis scoring for an LLM-judge or human-judge pass; that pass needs
to happen before "discovery quality" gets reported as a metric.

### Recoverability band uses primary obs only

Both the small-obs and medium-intv naive baselines only look at the
primary target observation. This is fine for v2 because in our two
archetypes the gold wins on that proxy alone at medium budget. But:

- If a future archetype's gold *doesn't* win on the primary proxy (e.g. a
  metric-hacking world where the primary obs is suppressed by the wrong
  intervention), the medium-budget band will reject every world. The band
  needs an `archetype_recoverability_measurements` field so each archetype
  can declare which proxies the naive baseline should consult.

### `_static_observational_outcomes` is O(unique assignment tuples)

We stratify by unique per-unit assignment tuple. With 4 binary + 1
ternary knob = 48 possible tuples in `hidden_cause`. For 4000 obs units
that's fine. For a future archetype with many knobs (or continuous knobs)
this explodes. If it ever matters, vectorize by computing
`apply_interventions` over the full unit set once per *unique candidate*
intervention and indexing out the rows that received it — same as the
oracle does but per-unit assignment-matched.

### Runtime simulator and audit script

Not yet written for `rpg_static_v2`:

- `framework_code/simulator_rpg.py` needs a `StaticRPGSimulator` class
  selected by `schema_version` that handles `observational_sample`,
  `interventional_sample`, `inspect_unit` modes, enforces
  `max_intervention_knobs`, rejects non-intervenable variables with a
  verbose error, and accounts cells against the budget.
- `dataset_generation_code/audit_rpg.py` needs a `--static` mode that
  re-derives gold with a fresh seed, re-runs the band, and asserts no
  schema/visibility regressions.

These were intentionally deferred so v2 generation could be validated end
to end first. The format is stable; the runtime layer should not require
breaking changes once it lands.

## Optional LLM hooks (`--llm-polish`, `--llm-extra-templates`)

Two narrow uses of Opus 4.8 (Bedrock) were added on top of the
deterministic core, both off by default and both gated by a single rule:
**the LLM never sees oracle scores, role assignments, or hidden state**.
Documented in §10b of the plan; surface-level details and gotchas here.

### Why polish at all

The code-written stories and questions are correct but read like a schema:

> "A population shows persistently high SymptomReport under current
> practice. Using the available query modes and budget, determine which
> single intervention, applied uniformly to a freshly sampled population,
> most reduces SymptomReport. Submit one intervention as a `do(.)` dict
> over the intervenable variables, plus a one-paragraph hypothesis
> explaining the underlying mechanism."

That works for a benchmark loader but is dry for human reviewers and
loses the "this scenario could really happen" feel. Opus 4.8 rewrites
the same content into a more human briefing without touching mechanism
or answer.

### Hook 1: `--llm-polish` (narrative rewrite)

Runs after `_static_build_<archetype>` produces an accepted world. The
LLM gets only the **visible** block plus the archetype brief
(`_STATIC_ARCHETYPE_BRIEF`). It returns rewritten story, question, and
per-variable descriptions.

Defenses (all enforced in `_static_polish_validate` and
`_static_leakage_check`):

- Variable names must round-trip exactly. A renamed observed variable
  rejects the polish.
- The question must still contain `do(` or `intervention` AND the word
  `hypothesis`. Drops to fallback otherwise.
- A substring blacklist rejects internal names (`LatentDriver`,
  `HealthSeekingTrait`, `DecoyState`, …) and role words (`decoy`,
  `red herring`, `true lever`).
- On any rejection after `max_retries`, the un-polished world is
  written with `meta.llm_polish = {"applied": false, "rejected_reason":
  "..."}`. Generation never *fails* because polish failed.

### Hook 2: `--llm-extra-templates N` (domain diversity)

Runs before generation. For each archetype the LLM proposes N new
domain templates (subdomain + neutral variable names + per-variable
short descriptions). Templates that survive `_static_template_validate`
are appended to `STATIC_TEMPLATES[archetype]` in memory for the rest of
the run.

The mechanism (per-archetype `_apply` / `_observe` / params) doesn't
care which template was used — it reads role keys from `template["names"]`
and never sees variable name surface forms. That is why we can grow
diversity through templates without touching mechanism code.

### Watchpoints if you turn either hook on

0. **Per-role semantic briefs at proposal time, not at polish time.**
   This is the gap I almost shipped without. The proposer needs to know
   what each role *means* in the mechanism, otherwise the LLM might
   wire an invented `Stage1ProductivityTraining` to a role that the
   mechanism code treats as a palliative knob. The fix is
   `_static_role_briefs(archetype)` — per-role plain-English
   descriptions that the proposer reads to pick coherent names. The
   briefs are never written into the world JSON; the leakage check
   rejects any output that echoes brief vocabulary (`decoy`,
   `palliative`, `true lever`). **When adding a new archetype, add a
   brief for every role key.** Without it, the LLM's output may be
   syntactically valid but semantically nonsense, and you only catch
   it during manual review.

1. **Opus genuinely knows the H. pylori story.** If you tell Opus "a
   population has chronic GI symptoms and one of the knobs is
   AntibioticCourse", it will be tempted to gush about bacterial
   colonization. The polish system prompt has an explicit instruction to
   *not* lean into textbook answers, and the leakage check rejects the
   word `bacterial` if it co-occurs with hidden-construct vocabulary —
   but the easiest mitigation is to keep the archetype brief abstract:
   it says "one knob acts on a hidden construct", not "one knob is the
   true cause." Don't expand the brief if you don't need to.

2. **Leakage check is a substring match, not semantics.** A polish that
   says "the device-quality theory is convenient but probably wrong" will
   pass the leakage check, but it does leak the answer. We accepted this
   risk because Opus 4.8 in practice follows the system prompt's rule
   "do not state or imply which intervention is the correct answer."
   When you inspect polished worlds during dataset review, spot-check for
   this kind of soft leakage and report back so we can tighten the
   prompt.

3. **Template proposal can quietly fail half the requests.** The schema
   check is strict (CamelCase variable names, role keys all present,
   description maps complete). Opus sometimes returns templates with
   one missing measurement description. Failures are printed
   (`[llm-template skip] hidden_cause: ...`) but don't abort the run.
   If you ask for 4 extras and get only 2, that's expected. Re-running
   with a different seed usually fills the rest.

4. **The polish hook does not re-run mechanism validators.** It only
   rewrites narrative. So polishing cannot break the recoverability band
   or the gold margin — those are settled before polish even runs.
   Conversely, polish cannot *fix* a world that failed validation; it
   simply never runs for rejected worlds.

5. **Hypothesis is still not auto-scored.** The polished question asks
   for a hypothesis paragraph because it makes the agent's reasoning
   explicit and auditable, not because we score it. Watch for downstream
   readers assuming hypothesis quality is part of the metric. It isn't.

6. **AWS credentials must be present when the flags are on.** With no
   creds, the BedrockLLM client constructs successfully (boto3 lazy
   binding) but every `generate` call throws. Polish failures fall back
   to un-polished worlds gracefully; template-proposal failures print a
   `[llm-template skip]` line and the generator falls back to the
   hand-written templates. Watch for the `applied: false` records
   piling up if you forgot to export creds.

### CLI summary

```bash
# Pure deterministic, no AWS dependency:
python world_gen_rpg.py --static --outdir out_rpg_static_v2/

# Polish narratives via Opus 4.8:
python world_gen_rpg.py --static --outdir out_rpg_static_v2_polished/ \
    --llm-polish

# Grow domain diversity to 5 templates per archetype, then polish:
python world_gen_rpg.py --static --outdir out_rpg_static_v2_diverse/ \
    --llm-extra-templates 2 --llm-polish

# Use a different Bedrock model:
python world_gen_rpg.py --static --outdir out_dbg/ --llm-polish \
    --llm-model us.anthropic.claude-sonnet-4-8-v1
```

## Quick reference: where things live in `world_gen_rpg.py`

| Concern | Symbol |
|---|---|
| Schema version | `SCHEMA_VERSION_STATIC` |
| Default budget caps | `STATIC_MAX_*` constants |
| Recoverability thresholds | `STATIC_RECOVER_*` constants |
| Per-world parameter draws | `_static_<archetype>_default_params` |
| Hidden state sampler | `_static_<archetype>_sample_hidden` |
| Structural equations under `do(.)` | `_static_<archetype>_apply` |
| Observation model | `_static_<archetype>_observe` |
| Current-practice assignment for obs sampling | `_static_<archetype>_assignment` |
| Oracle (CRN over candidates) | `_static_oracle_score` |
| Small/medium naive baselines | `_static_obs_naive_pick`, `_static_intv_naive_pick` |
| Recoverability band | `_static_recoverability_band` |
| Validators | `_static_validate` |
| World JSON assembler | `_static_assemble_world` |
| Per-archetype builders | `_static_build_<archetype>` |
| Top-level builder dispatcher | `STATIC_BUILDERS`, `static_rpg_generate_world` |
| Batch driver | `static_rpg_generate_dataset` |
| CLI entry | `python world_gen_rpg.py --static ...` |
| Optional Bedrock LLM wrapper | `_StaticRPGLLM` |
| LLM JSON helpers | `_static_extract_first_json`, `_static_llm_json` |
| Archetype briefs for the LLM | `_STATIC_ARCHETYPE_BRIEF` |
| Leakage substring blacklist | `_static_leakage_terms`, `_static_leakage_check` |
| Polish step | `_static_llm_polish_world`, `_static_polish_validate` |
| Template proposal | `_static_llm_propose_template`, `_static_template_validate` |
| Template-extension batch | `_static_extend_templates_via_llm` |
| CLI flags for LLM | `--llm-polish`, `--llm-extra-templates`, `--llm-model` |
