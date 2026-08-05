# RPG v3 Slide Examples and Qualitative Audit

## One-slide verdict

- The current static-plus-latent run is useful as a pipeline test, but it is not yet the large-action-space scientific-discovery benchmark we want.
- Six of the seven question families still mostly reduce to: read the named variables, run a few controlled samples, compare mean outcomes, choose the best named intervention.
- The new latent-regime family is the closest to the intended direction: the agent has to infer two hidden response regimes and a conditional policy. It did infer two regimes in all scored attempts, but it failed the best action for the high-regime branch.
- The anomaly family is hard, but not in the right way. It asks for unit-level anomaly IDs while the agent mostly sees summaries and small previews, so failures reflect interface mismatch as much as scientific reasoning.
- Bottom line for the PI: the benchmark currently tests whether an LLM can run small randomized comparisons. The next version should test whether it can form, revise, and distinguish competing latent explanations under an ambiguous action and measurement space.

## Evaluation files

- `framework_code/evaluations/rpg/eval_rpg_static_plus_latent_opus.json`
  - Same scoring as the trace file.
  - Includes per-world details, score fields, failure bucket, resource usage, final answer, and gold answer.
  - Does not include compact per-query traces.

- `framework_code/evaluations/rpg/eval_rpg_static_plus_latent_opus_traces.json`
  - Same `29/42 = 69.0%` accuracy and same per-archetype metrics.
  - Adds `details[].log_summary.query_trace`.
  - This is the better file for qualitative trajectory analysis because it shows query mode, sample size, measurements, interventions, and invalid-query errors.

## Run-level results

| Question family | Worlds | Accuracy | Avg. queries | Avg. sample cells | Main observed pattern |
|---|---:|---:|---:|---:|---|
| Hidden cause | 6 | 6/6 | 4.17 | 9600 | Solved by testing the named plausible treatments. |
| Confounded action | 6 | 6/6 | 4.33 | 9600 | Solved by randomized comparisons after being told observational data are confounded. |
| Mechanism bottleneck | 6 | 6/6 | 3.83 | 8333 | Solved by finding the lowest stage and testing the corresponding intervention. |
| Negative control | 6 | 5/6 | 4.33 | 9733 | Mostly solved by testing the few candidates and returning no intervention. |
| Hidden subtype policy | 6 | 6/6 | 3.83 | 8200 | Solved by testing 2-3 interventions and reading the sign of response by the named screen. |
| Latent response regimes | 6 | 0/6 | 4.33 | 9600 | Inferred two regimes, but missed or did not test the best high-regime action. |
| Anomaly discovery | 6 | 0/6 | 6.50 | 11934 | Failed to identify units/rules; hard partly because the interface shows summaries, not enough raw unit analysis. |

## Large-action-space assessment

| Question family | Observed measurements | Public actions | Candidate interventions in oracle | Does this feel large? |
|---|---:|---:|---:|---|
| Hidden cause | 6 | 4 | 6 | No. A few named interventions, one best. |
| Confounded action | 5 | 2 | 6 | No. It is basically a small factorial over sedation/mobilization or equivalent. |
| Mechanism bottleneck | 5 | 3 | 4 | No. The question names the three stages and the three matching actions. |
| Negative control | 5 | 3 | 5 | No. Test each candidate, choose none if no mean improvement. |
| Hidden subtype policy | 5 | 3 | 4 | Not really. The question names the screen variables and intervention choices. |
| Latent response regimes | 10 | 8 | 10 | Somewhat. This is the only family with enough distractors to start feeling scientific, but the prompt still exposes the action catalog and the answer form. |
| Anomaly discovery | 5 | 0 | 1 | Not an action-space problem. It is a distribution/rule-finding problem. |

Interpretation:

The current setup has a larger *surface vocabulary* than the earlier pilot, but not a genuinely large *scientific action space*. Most actions are already semantically labeled, few in number, and directly tied to the question. The model usually does not need to invent a latent variable or decide what kind of experiment would discriminate hypotheses; it can run a short checklist.

## Example 1: Hidden Cause

Plain question:

Something is making a population have persistently bad symptoms. Which treatment actually addresses the underlying cause?

Actual example:

- World: `world_rpg_static_hidden_cause_chronic_upper_gi_symptoms_seed5000`
- Topic: hospital data, chronic upper-GI symptoms
- Question asks the agent to improve `SymptomReport` and related measurements.
- Public measurements include symptom reports, endoscopy findings, quality-of-life index, diet, stress, and a serum panel.
- Public actions include `OralRegimenM`, `DietModification`, `StressReductionProgram`, and `AntacidDose`.
- Gold answer: turn on `OralRegimenM`.

What the key thing is:

The hidden cause is an infection-like driver. The correct treatment is the one that addresses the root cause, not just the symptom surface.

Why it is hard, in principle:

A real version would require distinguishing several plausible latent causes: infection, acid exposure, stress, diet, adherence, or measurement bias.

What experiments are needed:

- Start with symptom, endoscopy, and quality-of-life measurements.
- Test root-cause treatment versus symptom-control treatment.
- Confirm the root-cause treatment improves multiple related measurements, not just one noisy proxy.

What the agent actually did:

- Observed symptoms/endoscopy/quality of life.
- Tested `OralRegimenM`.
- Tested high antacid dose.
- Chose `OralRegimenM` because it improved all target measurements more.

Assessment:

This is too easy right now. The action names and domain semantics make `OralRegimenM` sound like the causal treatment. The agent can solve it with two intervention queries.

## Example 2: Confounded Action

Plain question:

Records say an aggressive treatment looks harmful, but maybe sicker people were more likely to receive it. What treatment should we choose if we randomize?

Actual example:

- World: `world_rpg_static_confounded_action_icu_sedation_intensity_seed15579`
- Topic: ICU sedation intensity
- Question says passive records make high sedation look bad.
- Public actions are `Sedation` with values `off/low/high` and `EarlyMobilization` with values `off/on`.
- Gold answer: `Sedation=high` and `EarlyMobilization=on`.

What the key thing is:

The agent must distinguish observational association from interventional effect.

Why it is hard, in principle:

In real science, confounding by indication is subtle. The hard part is deciding which baseline severity variables matter, which outcomes are delayed, and whether the treatment has different effects in subgroups.

What experiments are needed:

- Observe treatment assignment, severity proxy, and outcome.
- Randomize the suspected treatment levels.
- Compare outcomes under the same mobilization setting.

What the agent actually did:

- Observed `RecoveryScore`, assigned sedation, and severity proxy.
- Tested high, low, and off sedation with mobilization on.
- Picked the arm with the highest mean recovery.

Assessment:

This currently collapses. The question tells the model the observational belief is confounded, names the two intervention knobs, and the agent just runs a small factorial comparison.

## Example 3: Mechanism Bottleneck

Plain question:

A process has several stages. Which stage is blocking the final outcome, and where should we intervene?

Actual example:

- World: `world_rpg_static_mechanism_chain_credential_program_pipeline_seed6313`
- Topic: credential program pipeline
- Measurements are foundation pass rate, apprenticeship completion, capstone assessment, job placement, and certification likelihood.
- Public actions are `FoundationSupportProgram`, `ApprenticeshipMentoring`, and `CapstonePrepWorkshop`.
- Gold answer: `CapstonePrepWorkshop=on`.

What the key thing is:

The capstone stage is the bottleneck. Fixing earlier stages does not help as much because the later stage constrains the final outcome.

Why it is hard, in principle:

A real bottleneck problem would require deciding whether a low stage is a cause, a symptom, or a measurement artifact, and whether improving an upstream stage propagates downstream.

What experiments are needed:

- Measure all stages and the final outcome.
- Intervene on the suspected bottleneck.
- Compare against at least one non-bottleneck intervention.

What the agent actually did:

- Observed all three stages.
- Saw capstone was much lower than the other two.
- Tested capstone prep and apprenticeship mentoring.
- Chose capstone prep.

Assessment:

This is very legible and too scaffolded. The question literally names the three stages and asks for the bottleneck. It is good for sanity testing, not for v3.

## Example 4: Negative Control

Plain question:

Several popular interventions are available. Do any of them actually help, or are the observational patterns misleading?

Actual example:

- World: `world_rpg_static_negative_control_job_search_support_packages_seed7020`
- Topic: job-search support packages
- Outcome is `UnemploymentSpellLength`, lower is better.
- Actions include `EvidenceBasedPlacementSupport`, `ConventionalResumePolish`, and `TrendingSelfDevelopmentWorkshop`.
- Gold answer: empty intervention `{}`.

What the key thing is:

None of the available interventions reliably improves the outcome under controlled assignment.

Why it is hard, in principle:

This could be a meaningful scientific question if the model had to decide whether apparent improvement is selection, placebo, measurement shift, short-term proxy movement, or a real long-term effect.

What experiments are needed:

- Observe the natural correlations.
- Test each plausible intervention under controlled assignment.
- Return no intervention if the controlled effects are small or harmful.

What the agent actually did:

- Observed job-search initiative, engagement, and unemployment length.
- Tested the three named interventions.
- Returned `{}` because none clearly improved the target.

Assessment:

This mostly collapses into “try the three candidates.” It is useful only if we care about whether the model can say no.

## Example 5: Hidden Subtype Policy

Plain question:

Different people respond to different interventions. Use a measured screen to decide which intervention each person should get.

Actual example:

- World: `world_rpg_static_hidden_subtype_growth_campaign_audience_segmentation_seed7626`
- Topic: user retention campaign
- Screen variables are `AudienceSegmentSignal` and `SecondaryAudienceIndex`.
- Actions are `DiscountLedCampaign`, `BalancedHybridCampaign`, and `FeatureLedCampaign`.
- Gold answer: if `AudienceSegmentSignal` is high, use `FeatureLedCampaign`; if low, use `DiscountLedCampaign`.

What the key thing is:

The population has two response types. Low-screen users respond better to discount messaging; high-screen users respond better to feature messaging.

Why it is hard, in principle:

A real version would not tell the agent which screen matters or that there are two clean response groups. The model would need to discover effect heterogeneity from competing possible screens.

What experiments are needed:

- Observe the distribution of screen variables and outcome.
- Test each campaign while measuring the screen and outcome.
- Look for treatment-response reversal by screen value.
- Choose a threshold and branch actions.

What the agent actually did:

- Observed the two screens and retention.
- Tested discount, feature, and hybrid campaigns.
- Saw discount had negative correlation with the screen and feature had positive correlation with the screen.
- Proposed the correct conditional policy.

Assessment:

This is scientifically closer than the simple one-action worlds, but the question gives away too much. It says the policy should be keyed on one of two named screens and names the three interventions. The agent does not need to search broadly.

## Example 6: Latent Response Regimes

Plain question:

Clinicians disagree whether one noisy syndrome is actually two hidden patient types. Figure out whether there are hidden response groups and propose a treatment rule.

Actual example:

- World: `world_rpg_static_latent_regime_discovery_acute_inflammatory_response_regimes_seed9040`
- Topic: acute inflammatory response
- Public measurements include recovery stability, inflammation panel, complement shift, organ stress, medication exposure, viral pattern, tolerability, and others.
- Public actions include `SignalModulatorAlpha`, `SignalModulatorBeta`, `BroadStabilizer`, supportive care, medication hold, microbial coverage, symptom relief, and monitoring escalation.
- Gold answer: two regimes; branch on `InflammationPanelA` around 50; high branch gets `SignalModulatorBeta`; low branch gets `SignalModulatorAlpha`.

What the key thing is:

There are two hidden response regimes. One regime benefits from alpha modulation, the other from beta modulation. A single uniform treatment is worse than a regime-aware policy.

Why it is hard, in principle:

This is the direction we want. The model must infer that average effects hide opposite subgroup effects, identify a measured proxy for the latent regime, and choose a treatment rule. The problem should require forming competing hypotheses like “one broad inflammatory axis,” “two opposing immune regimes,” “site practice confounding,” or “medication exposure artifact.”

What experiments are needed:

- Observe recovery and multiple candidate proxies to look for clusters or wide/bimodal distributions.
- Test one action that helps one suspected group.
- Check whether treatment response reverses across a proxy.
- Test the complementary action for the other suspected group.
- Compare a broad uniform treatment against a branch-specific policy.

What the agent actually did:

- It usually observed recovery plus inflammation/complement/organ-stress variables.
- It tested `SignalModulatorAlpha` and `BroadStabilizer`.
- It inferred two regimes and often picked the right branch variable.
- It failed because it did not test `SignalModulatorBeta`, so it used `BroadStabilizer` for the high-inflammation branch.

Representative failure:

- Agent answer: branch on `InflammationPanelA`; low branch uses `SignalModulatorAlpha`; high branch uses `BroadStabilizer`.
- Gold answer: branch on `InflammationPanelA`; low branch uses `SignalModulatorAlpha`; high branch uses `SignalModulatorBeta`.
- The model got the latent structure mostly right but stopped one experiment too early.

Assessment:

This is the best current v3 seed. It creates meaningful scientific behavior: the model hypothesizes a latent split, uses correlation changes under intervention, and proposes a conditional policy. But it still exposes all actions and states the latent-regime frame in the question. It is not yet broad enough.

## Example 7: Anomaly Discovery

Plain question:

Most units follow a normal pattern across several measurements. A small fraction do not. Find the unusual units and describe the rule that makes them unusual.

Actual example:

- World: `world_rpg_static_anomaly_discovery_clinical_case_anomaly_hunt_seed8131`
- Topic: clinical anomaly hunt
- Measurements include acute vital anomaly, baseline lab composite, comorbidity, medication breadth, and adverse event likelihood.
- No interventions.
- Gold rule: high `AcuteVitalAnomalyIndex` together with low `BaselineLabComposite`.

What the key thing is:

Anomalous units violate the usual joint relationship between two features.

Why it is hard, in principle:

This is a real scientific task if the model must discover a rare subgroup from raw data, decide which joint pattern is abnormal, and distinguish rare-but-normal tails from true anomalies.

What experiments are needed:

- Sample enough units with the relevant measurements measured together.
- Inspect raw rows, not just aggregate summaries.
- Fit or reason about the typical joint distribution.
- Flag units violating the learned relationship.

What the agent actually did:

- It repeatedly sampled different measurement combinations.
- It relied heavily on summaries and correlations.
- It guessed a rule in the wrong direction for one clinical case: high acute vital anomaly and high baseline lab composite, while the gold was high acute vital anomaly and low baseline lab composite.
- It also wasted turns near the cell budget limit.

Assessment:

This is hard, but currently not a clean LLM scientist test. The answer asks for unit IDs, while the agent does not have a good workflow for reading and analyzing the full CSV tables. This should move to coder-agent or explicit table-analysis tools if we keep unit-level anomaly tasks.

## What this says about our current benchmark

- It is not yet testing “large action space” in the PI sense.
- It often tests small randomized A/B comparisons.
- The question wording often gives away the ontology:
  - “hidden subtype”
  - “conditional policy”
  - “bottleneck”
  - “conventional wisdom is confounded”
  - “multiple hidden response regimes”
- The public intervention names are too semantically revealing.
- The action catalog is too short for most worlds.
- The correct experimental program is often obvious at step 1.
- The agent can perform well without maintaining multiple competing latent hypotheses.

## What should change for v3

### 1. Ask for a scientific diagnosis, not just an action

Better plain question:

“Patients are not responding consistently to current treatment. You can order measurements and run limited controlled trials. What is the best explanation of the population structure, and what would you do next?”

This forces:

- latent hypothesis formation;
- deciding what kind of hidden variable might exist;
- deciding whether to treat, stratify, measure more, or reject the premise;
- explaining uncertainty.

### 2. Hide the answer frame

Do not say:

- “there are hidden response regimes”
- “propose a conditional policy”
- “keyed on this screen”
- “identify the bottleneck”

Instead ask:

- “Why are average treatment effects unstable?”
- “Why do two sites report opposite outcomes?”
- “Why does the treatment help in one dataset and harm in another?”
- “Which follow-up experiment would most reduce uncertainty?”

### 3. Make the action space genuinely larger

Current latent-regime has 8 actions, but the model sees them all as named candidate interventions. Better:

- include 15-30 possible actions;
- include diagnostic tests, treatments, timing changes, dose changes, measurement choices, and follow-up windows;
- make many actions plausible but irrelevant;
- include actions that reveal information without directly improving the outcome;
- allow joint actions, but make some combinations harmful or uninterpretable.

### 4. Require sequential hypothesis revision

A good v3 problem should not have the full experimental plan obvious at step 1.

Example structure:

- Early data suggest one broad syndrome.
- A targeted measurement reveals two modes.
- First intervention helps on average but increases variance.
- Stratified analysis suggests response reversal.
- A second intervention tests the alternative branch.
- The final answer must state the latent explanation and the next policy.

### 5. Add latent discovery tasks that are not just treatment selection

Possible v3 question types:

- “How many hidden groups are there?”
- “Which measured variables are proxies for the hidden state?”
- “Which intervention distinguishes the competing latent explanations?”
- “Is this a true subgroup or a measurement artifact?”
- “What experiment should be run next, and why?”

### 6. Separate evaluation of explanation and action

For latent worlds, score separately:

- Did the agent infer the number of regimes?
- Did it identify the right proxy or family of proxies?
- Did it test the action that distinguishes regimes?
- Did it choose a good policy?
- Did its explanation mention the correct causal/latent structure?

The current latent eval already starts this with `regime_count_correct`, `branch_variable_matches_gold`, `branch_actions_match_gold`, and utility gap. We should extend it with explicit “tested decisive experiment” checks.

## Best slide-friendly framing

For PI:

“Our current RPG benchmark proves the pipeline works, but most tasks are still too close to controlled A/B testing over a small named action set. The new latent-regime family is the right direction: the model inferred two hidden regimes but failed to test the complementary treatment, which is a meaningful scientific failure. v3 should expand this idea: hide the ontology, enlarge the action/measurement space, and evaluate whether the model forms and revises latent hypotheses, not just whether it compares means.”

