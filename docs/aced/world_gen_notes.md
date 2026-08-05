Question types:

# 1. Best intervention under safety constraint

## Core question

> The clinic wants to reduce avoidable FollowUpVisits, but it must not increase MissedComplication. Which intervention would you recommend?

This is probably the **most stable and valuable** question type.

It tests whether the agent can:

1. identify the target,
2. test candidate interventions,
3. separately test side effects,
4. avoid the intervention with the best target effect if it causes harm.

## More examples

> A school wants to reduce DropoutRisk without lowering AcademicConfidence. Which policy should it adopt?

> A platform wants to reduce ChurnRisk without increasing ComplaintRate. Which product change should it choose?

> A court wants to reduce ReoffenseRisk without increasing UnnecessaryDetention. Which policy intervention is safest?

## World skeleton

Use a target, a side-effect, 3–5 candidate interventions, and mediators.

```text
BaselineSeverity → FollowUpVisits
BaselineSeverity → MissedComplication

HomeMonitoring → BloodPressureControl → FollowUpVisits
HomeMonitoring → MissedComplication

NurseCallFrequency → PatientAnxiety → FollowUpVisits
NurseCallFrequency → MissedComplication

AppointmentPolicy → FollowUpVisits
AppointmentPolicy → MissedComplication
```

You want at least one intervention that:

* improves target and does not worsen side effect;
* improves target more but worsens side effect;
* has little effect.

## Probability design

Make the target effect and side-effect effect separable. For example:

* `HomeMonitoring=Yes` strongly improves `BloodPressureControl`, reducing `FollowUpVisits`, but slightly increases `MissedComplication` detection/reporting.
* `NurseCallFrequency=High` moderately reduces `FollowUpVisits` and does not increase `MissedComplication`.
* `AppointmentPolicy=Sparse` reduces `FollowUpVisits` but increases `MissedComplication`.

Do **not** make all intervention effects monotonic in the same direction. Otherwise the question collapses into argmin.

## Reliable evaluation

Compute:

```text
E[target | do(X=v)]
E[side_effect | do(X=v)]
```

Gold can be:

```text
argmin target among actions satisfying side_effect <= baseline + epsilon
```

or Pareto set:

```text
not dominated on target and side_effect
```

I would prefer **set-valued gold** here, because multiple interventions may be acceptable.

## Failure modes

This question becomes unstable if:

* no action satisfies the safety constraint;
* all safe actions are statistically tied;
* side effect is too rare;
* target and side effect are both descendants of the same mediator with identical direction, making no real tradeoff.

Use your existing threshold-guard idea: keep actions away from the accept/reject boundary so finite samples do not flip the answer. Your current code already has a guard to shift thresholds away from near-ties in budget/side-effect settings. 

## Adaptive experiment design value

Very high. The agent should first estimate baseline, then test high-priority interventions, then check side effects before committing.

---

# 2. Observational association reversal / confounding trap

## Core question

> Observational data suggest that HomeMonitoring is associated with more FollowUpVisits. Does HomeMonitoring actually increase FollowUpVisits, or is the association explained by confounding?

This is one of the best tests of scientific reasoning because it punishes agents that only analyze passive samples.

## More examples

> Students assigned to IntensiveTutoring have lower final grades in observational data. Does tutoring hurt grades?

> Defendants receiving StrictSupervision have higher TechnicalViolation rates. Is supervision causing violations, or are higher-risk defendants assigned to it?

> Patients receiving HighDosage have worse RecoveryStatus. Is high dosage harmful, or are sicker patients more likely to receive it?

## World skeleton

Classic confounding:

```text
BaselineSeverity → HomeMonitoring
BaselineSeverity → FollowUpVisits
HomeMonitoring → BloodPressureControl → FollowUpVisits
```

The intended pattern:

```text
P(FollowUpVisits high | HomeMonitoring=Yes) > P(FollowUpVisits high | HomeMonitoring=No)
```

but:

```text
E[FollowUpVisits | do(HomeMonitoring=Yes)] < E[FollowUpVisits | do(HomeMonitoring=No)]
```

## Probability design

Make `BaselineSeverity` strongly affect both treatment assignment and outcome.

Example:

* severe patients are much more likely to receive monitoring;
* severe patients have many follow-up visits;
* monitoring itself reduces follow-ups within each severity stratum.

This creates Simpson-style reversal.

## Reliable evaluation

Gold is the sign comparison:

```text
observational association sign != interventional effect sign
```

The answer can be:

> Observationally it looks harmful, but interventionally it is beneficial.

You can also evaluate whether the agent names the confounder, but the primary score should be based on the correct causal conclusion.

## Failure modes

Unstable if:

* confounding is too weak;
* treatment effect is too weak;
* finite samples do not reveal reversal;
* the treatment variable is almost deterministic given severity, causing poor overlap.

Avoid near-deterministic treatment assignment. Keep every treatment level with at least maybe 10–20% probability in each severity group. Your current code already noticed that near-deterministic roots can mute or distort downstream dependence, so similar anti-degeneracy checks matter here too. 

## Adaptive experiment design value

Very high. A good agent should notice observational association, hypothesize confounding, then request interventional samples or stratified observational analysis.

---

# 3. Direct vs mediated mechanism

## Core question

> Does TreatmentStatus influence FollowUpVisits only through BloodPressureRange, or is there also a direct effect?

This is stable if you generate it from a controlled skeleton.

## More examples

> Does TutoringProgram improve FinalGrade only through HomeworkCompletion, or also directly through TestPreparation?

> Does AppNotification reduce ChurnRisk only through EngagementLevel, or is there also a direct effect?

> Does RehabilitationProgram reduce ReoffenseRisk only through EmploymentStatus, or also through SubstanceUseStability?

## World skeleton

You should generate three balanced classes:

### Mediated only

```text
TreatmentStatus → BloodPressureRange → FollowUpVisits
```

### Direct + mediated

```text
TreatmentStatus → BloodPressureRange → FollowUpVisits
TreatmentStatus → FollowUpVisits
```

### Not through proposed mediator

```text
TreatmentStatus → MedicationAdherence → FollowUpVisits
TreatmentStatus → BloodPressureRange
```

The third class is important. Otherwise models learn that the named mediator is usually relevant.

## Probability design

For mediated-only, the intervention effect of `TreatmentStatus` on target should mostly disappear when intervening to fix the mediator.

For direct+mediated, fixing the mediator should reduce but not eliminate the treatment effect.

For “not through mediator,” `TreatmentStatus` and the proposed mediator may be associated, but the mediator should not lie on the active directed path to target.

## Reliable evaluation

Use graph structure for gold:

```text
exists directed path X → ... → M → ... → Y
exists directed path X → ... → Y not passing through M
```

Then classify:

```text
mediated_only / direct_and_mediated / not_mediated
```

This matches your current mediator-class direction, where the advanced code already tracks `mediator_only`, `mediator_both`, and `mediator_not`. 

## Failure modes

Potential issue: from samples alone, distinguishing direct vs mediated can be hard if the direct effect is small or if mediator is noisy.

So for direct+mediated, direct effect must be large enough. For mediated-only, do not include alternate paths from X to Y unless they are intentionally part of the class.

## Adaptive experiment design value

High. The agent should test:

1. `do(X)`;
2. `do(M)`;
3. maybe `do(X)` while stratifying/conditioning on M, depending on available sample interface;
4. compare whether X still moves Y beyond M.

---

# 4. Identify the actual mediator among candidates

## Core question

> Which variable best explains how DosageAmount affects FollowUpVisits?

This is better than asking “which variables mediate X on Y” in an open-ended way, because open-ended mediator discovery can become noisy. Give 3–5 candidate variables.

## More examples

> Which pathway explains how TutoringProgram affects GraduationStatus: Attendance, HomeworkCompletion, SchoolBelonging, or ParentContact?

> Which variable mediates the effect of NotificationFrequency on ChurnRisk: DailyEngagement, UserAnnoyance, PurchaseIntent, or SupportTickets?

> Which pathway explains how RehabilitationProgram affects ReoffenseRisk: EmploymentStatus, HousingStability, SubstanceUseStability, or OfficerContact?

## World skeleton

```text
X → M_true → Y

X → M_decoy1
M_decoy2 → Y
C → X
C → Y
```

Decoys should be realistic:

* one affected by X but not affecting Y;
* one affecting Y but not affected by X;
* one associated through confounding;
* maybe one collider trap.

## Probability design

The true mediator should carry a substantial fraction of the effect. Decoys should show some marginal association but fail under intervention tests.

Example:

```text
DosageAmount → BloodPressureRange → FollowUpVisits
DosageAmount → SideEffectSymptoms
BaselineSeverity → DosageAmount
BaselineSeverity → PatientAnxiety → FollowUpVisits
```

`SideEffectSymptoms` may correlate with dosage, and `PatientAnxiety` may correlate with visits, but only `BloodPressureRange` lies on the directed path.

## Reliable evaluation

Gold:

```text
candidate M is valid if M lies on a directed path from X to Y
```

Optional stronger gold:

```text
mediated effect through M exceeds threshold
```

I would start with structural gold, then filter worlds by effect size using exact inference. This avoids “structural mediator but numerically irrelevant.”

## Failure modes

* Multiple candidates are true mediators unless you intend set-valued answer.
* True mediator effect is tiny.
* Decoys are too obviously irrelevant by name.
* Agent can answer from semantic priors without sampling.

To avoid semantic giveaway, keep all candidate names plausible.

## Adaptive experiment design value

High. Good agents should test candidate mediators, not just target.

---

# 5. Find an acceptable intervention, not necessarily the best one

## Core question

> The hospital wants any intervention that reduces ReadmissionRisk by a meaningful amount. Which interventions meet this standard?

This differs from argmin. It tests satisficing and set-valued reasoning.

## More examples

> Find any policy that reduces DropoutRisk to an acceptable level.

> Which product changes reduce ChurnRisk by at least a meaningful amount?

> Which treatment options improve RecoveryStatus enough to justify deployment?

## World skeleton

```text
InterventionA → M1 → Target
InterventionB → M2 → Target
InterventionC → Target
InterventionD unrelated or weak
```

You want buckets:

* no acceptable intervention;
* one or two acceptable interventions;
* many acceptable interventions.

Your current advanced code already has a satisficing generator with empty/few/many buckets and threshold robustness. 

## Probability design

Effects should be separated around the threshold.

Example expected improvements:

```text
A: 0.42
B: 0.38
C: 0.09
D: 0.01
threshold: 0.30
```

Do not set threshold at 0.40 if A and B are close; finite sampling may flip.

## Reliable evaluation

Gold is a set:

```text
{(X,v): improvement >= threshold}
```

But I would avoid putting the numeric threshold in the natural-language question unless you want mathy benchmark behavior. You can phrase it as:

> “meaningfully reduce” / “clinically meaningful improvement”

while storing a hidden threshold in metadata.

For evaluation, accept any gold action, or ask for all acceptable actions depending on difficulty.

## Failure modes

* Threshold too close to action effect.
* Target variable not scoreable.
* Too many candidate action states, making exploration combinatorially expensive.

Limit candidate interventions to maybe 4 variables × 2–3 states.

## Adaptive experiment design value

Medium-high. The agent does not need exhaustive search if it finds a good enough action, but strong agents can use early stopping.

---

# 6. Robust intervention across subgroups

## Core question

> Which intervention reduces DropoutRisk across both LowIncome and HigherIncome students, rather than only improving the average?

This is excellent for adaptive experiment design, but slightly less stable than the first five.

## More examples

> Which treatment reduces FollowUpVisits for both younger and older patients?

> Which platform change reduces ChurnRisk for both new and long-term users?

> Which reentry policy reduces ReoffenseRisk for both first-time and repeat defendants?

## World skeleton

```text
Group → Target
Group → Mediator
InterventionA → Target          # strong average effect, mostly one group
InterventionB → Mediator → Target  # moderate effect in both groups
InterventionC → Target          # harmful in one group
```

You need effect modification:

```text
Intervention × Group → Target
```

But a normal BN does not have interaction nodes unless encoded through CPD of `Target` with parents `[Intervention, Group]`.

So this is feasible: set `Target` parents to include both intervention and group, with group-specific CPD effects.

## Probability design

Example:

* `PolicyA` helps high-resource group a lot, does nothing for low-resource group.
* `PolicyB` helps both groups moderately.
* Average effect of A may be larger because high-resource group is more common.
* Robust answer is B.

## Reliable evaluation

Gold:

```text
for each subgroup g:
  E[target | do(X=v), Group=g] improves over baseline for Group=g

choose action maximizing min_g improvement
```

or:

```text
acceptable if improvement_g >= threshold for all g
```

## Failure modes

* Need enough samples per subgroup.
* If group is rare, finite samples are noisy.
* If agent only requests global samples, it may miss heterogeneity.
* If subgroup variable is morally sensitive, be careful with domain framing.

Use non-sensitive-ish variables first: `PriorAchievementLevel`, `BaselineSeverity`, `UserTenureGroup`, `InitialRiskLevel`.

## Adaptive experiment design value

Very high. The agent must decide to stratify and collect enough subgroup data.

---

# 7. Choose the next experiment / value of information

## Core question

> Before choosing a final intervention, which variable should the researcher experimentally manipulate next to learn the most about reducing FollowUpVisits?

This is closest to “adaptive experiment design,” but it is harder to evaluate reliably.

A safer version:

> The researcher is uncertain whether FollowUpVisits are driven more by BloodPressureRange or PatientAnxiety. Which experiment would distinguish these mechanisms?

## More examples

> Which intervention would best distinguish whether tutoring works through Attendance or HomeworkCompletion?

> Which experiment would tell us whether ChurnRisk is mainly driven by EngagementLevel or UserAnnoyance?

> Which variable should be intervened on next to decide between two competing causal hypotheses?

## World skeleton

This requires two or more plausible pathways:

```text
X → M1 → Y
X → M2 → Y
C → M1
C → M2
```

But since the true graph is fixed, “value of information” must be defined relative to hidden uncertainty induced by samples. If the agent does not know the graph, then intervening on M1 or M2 reveals mechanism.

## Probability design

Make M1 and M2 both observationally correlated with Y, but only one is a strong causal mediator.

Example:

```text
Treatment → BloodPressureRange → FollowUpVisits
Treatment → PatientAnxiety
BaselineSeverity → PatientAnxiety
BaselineSeverity → FollowUpVisits
```

Observationally both BloodPressureRange and PatientAnxiety correlate with visits, but only BloodPressureRange is the main causal mediator.

## Reliable evaluation

This is the hardest one.

I would not evaluate open-ended “most informative” at first. Instead, use a multiple-choice candidate set:

```text
candidate experiments:
do(BloodPressureRange=Normal)
do(PatientAnxiety=Low)
do(BaselineSeverity=Low)  # non-intervenable, invalid
do(AppointmentPolicy=Sparse)
```

Gold can be:

```text
experiment whose result would maximally separate candidate hypotheses
```

But implementing true expected information gain is heavier.

A simpler reliable version:

> Which experiment would directly test whether M is causal for Y?

Gold:

```text
do(M=v)
```

## Failure modes

* Requires formal hypothesis set.
* Hard to score free-form answers.
* Agents may answer semantically instead of experimentally.

## Adaptive experiment design value

Extremely high, but implementation complexity is medium-high. I would include this as a smaller slice of the dataset after the other types work.

---

# 8. Invalid causal premise / non-intervenable or wrong-direction question

## Core question

> An analyst proposes changing AgeGroup to reduce FollowUpVisits. Is this a valid intervention? If not, what manipulable variable should be considered instead?

This is very useful because real scientists must reject invalid questions.

## More examples

> Which variables mediate the effect of DosageAmount on AgeGroup, if any?

> Can we reduce RecidivismRisk by intervening on PriorConvictions?

> A report claims FinalGrade causes PriorAchievementLevel. Is this causal direction plausible in the world?

## World skeleton

Wrong-direction:

```text
AgeGroup → BaselineSeverity → DosageAmount
AgeGroup → FollowUpVisits
DosageAmount → BloodPressureRange → FollowUpVisits
```

Question asks about:

```text
DosageAmount → AgeGroup
```

which should be rejected.

Non-intervenable:

```text
AgeGroup → Target
PriorHistory → Target
IntervenablePolicy → Mediator → Target
```

The invalid variable has real causal influence but cannot be manipulated.

## Probability design

Make the invalid variable genuinely predictive, so a naive observational model wants to use it. But since it is non-intervenable or temporally upstream, the correct answer rejects it.

Example:

* `AgeGroup` strongly predicts follow-up visits;
* but cannot be assigned;
* `HomeMonitoring` is a manipulable descendant/parallel policy that can reduce visits.

## Reliable evaluation

Gold can be structural/metadata-based:

```text
is X non_intervenable?
is Y ancestor of X, making X → Y impossible?
does no directed path exist from X to Y?
```

You already have non-intervenable detection in the generator, with examples like age, gender, genetic conditions, and historical variables marked as impossible/unethical/impractical to manipulate. 

## Failure modes

* If the answer is too obvious from the variable name, models may not need samples.
* If the question asks “what should be considered instead,” there may be many valid alternatives.

So I would split into two stages:

1. binary/short answer: “valid intervention or not?”
2. optional: “name one valid alternative intervention that improves target.”

## Adaptive experiment design value

Medium. It tests scientific judgment more than exploration, but it is important for avoiding nonsensical interventions.

---

# My recommended final 8-question benchmark set

If I were designing the dataset, I would use these 8:

| # | Question type                                          |   Stability | Adaptive value | Gold type                      |
| - | ------------------------------------------------------ | ----------: | -------------: | ------------------------------ |
| 1 | Best intervention with safety constraint               |   Very high |      Very high | set / constrained argmin       |
| 2 | Observational reversal due to confounding              |   Very high |      Very high | causal conclusion + confounder |
| 3 | Direct vs mediated mechanism                           |        High |           High | 3-class label                  |
| 4 | Identify mediator among candidates                     |        High |           High | set / single mediator          |
| 5 | Satisficing acceptable intervention                    |   Very high |    Medium-high | set                            |
| 6 | Robust subgroup intervention                           | Medium-high |      Very high | maximin / set                  |
| 7 | Choose discriminating next experiment                  |      Medium |      Very high | candidate experiment           |
| 8 | Invalid causal premise / non-intervenable intervention |        High |         Medium | reject / valid alternative     |

The first five are the most stable. Six and seven are the most aligned with your abstract but need more careful implementation. Eight is less about adaptive sampling, but it protects the benchmark from becoming “always assume the question is valid.”

---

# How I would generate worlds for these

The key is to stop relying on random graph discovery for the central structure. Instead:

## Step 1: choose an archetype

Example:

```text
confounding_reversal
safety_constrained_policy
mediator_identification
subgroup_robust_policy
```

## Step 2: instantiate semantic roles

For `safety_constrained_policy`:

```text
Target = FollowUpVisits
SideEffect = MissedComplication
InterventionGood = NurseCallFrequency
InterventionRisky = AppointmentPolicy
InterventionWeak = PatientPortalMessage
MediatorGood = BloodPressureControl
MediatorRisky = DelayedCare
Confounder = BaselineSeverity
```

## Step 3: hard-code required edges

```text
BaselineSeverity → Target
BaselineSeverity → SideEffect
InterventionGood → MediatorGood → Target
InterventionRisky → Target
InterventionRisky → SideEffect
InterventionWeak → Target
```

## Step 4: add realistic distractor variables

Let LLM generate extra variables and plausible edges, but enforce:

* DAG;
* max parents ≤ 3;
* path length bounded;
* no accidental alternate path that changes the intended answer class;
* no extra parent that destroys the designed mechanism.

Your current advanced generator already recognizes why bounded path length, connectedness, v-structures, and max in-degree matter for advanced questions. 

## Step 5: CPD construction should be archetype-aware

For central edges, do not use purely random logistic CPDs. Use controlled CPDs with desired effect sizes.

Then for background edges, random strong logistic CPDs are fine.

You need exact post-generation checks:

```text
effect_size(target, best_action) > min_effect
top1 - top2 gap > min_gap, unless set-valued
side_effect gap not near threshold
mediator effect > min_mediated_effect
confounding reversal actually appears
subgroup effects separated enough
```

This matters because faithfulness alone is not enough. A graph can be faithful and still produce a question that is practically unanswerable from finite samples because the relevant effect is tiny.

---

# Probability / CPD issues to watch

## 1. Faithfulness is necessary but not sufficient

Your current code checks faithfulness and also uses stronger CPDs because weak probabilities can make long-chain dependence vanish.  But for this benchmark you also need **task faithfulness**, meaning the specific causal contrast in the question must be empirically visible.

For every generated question, verify the exact estimand:

```text
target intervention gap
side-effect safety gap
observational-vs-interventional reversal
direct-vs-mediated residual effect
subgroup-specific improvement
```

## 2. Avoid deterministic policies

If `Treatment=Yes` almost always occurs when `Severity=High`, observational data becomes hard to analyze and interventional estimates may be fine but observational overlap is bad. Keep all states with nontrivial support.

## 3. Avoid too many states for key variables

For adaptive experiment design, key interventions should usually be binary or ternary. If `DosageAmount` has 20 levels, the agent burns budget estimating a dose-response curve instead of reasoning.

Good:

```text
DosageAmount = Low / Medium / High
NurseCallFrequency = None / Monthly / Weekly
```

Bad for first version:

```text
DosageAmount = 0mg / 5mg / 10mg / ... / 100mg
```

## 4. Avoid semantic giveaways

If the question is:

> Which variable mediates Treatment → Recovery: BloodPressureControl or FavoriteColor?

then models can answer from names.

All candidates should sound plausible.

## 5. Avoid requiring conditioning-only causal inference if the agent can request interventions

Since your agent can request interventional samples, the best questions should reward active experimentation. Do not make everything solvable by observational adjustment.

---

# The most important design principle

For each question, define both:

## A. The causal role pattern

Example:

```text
confounder C causes both X and Y
X causally improves Y
observational association points opposite direction
```

## B. The empirical signature

Example:

```text
E[Y | X=1] worse than E[Y | X=0]
E[Y | do(X=1)] better than E[Y | do(X=0)]
```

Then reject worlds where the empirical signature does not hold by a comfortable margin.

That is what makes the dataset reliable.

---

The questions should not just ask for causal facts. They should create situations where a good strategy looks different from a bad one.

For example:

* Bad agent: only samples observationally, gets fooled by confounding.
* Good agent: notices possible confounding, requests `do(Treatment)`.
* Bad agent: optimizes target only, chooses harmful intervention.
* Good agent: checks side-effect before recommending.
* Bad agent: tests only global average.
* Good agent: stratifies by subgroup.
* Bad agent: assumes named mediator is correct.
* Good agent: intervenes on mediator candidates.

That gives you a much stronger story than “models answer causal graph questions.” It becomes: **agents differ in experimental strategy**, and your benchmark can measure that difference.

Yes, I think you should **keep the topics**, but stop treating them as the primary generator. They should become the **outer semantic domain**, not the thing that determines the causal structure.

Right now your pipeline is roughly:

> topic → variables → graph → CPDs → story → questions

For your new goal, I would change it to:

> topic → scientific archetype → causal role plan → variables → required graph skeleton → extra graph context → controlled CPDs → validation → story/question → save world

The important shift is: **the question archetype controls the graph and probabilities**, not the other way around.

Your current advanced generator already has some good implementation lessons: it uses connected advanced topologies, bounded path length, max in-degree, v-structure requirements, stronger CPDs, root probability floors, and faithfulness checks because shallow/disconnected/weak-signal graphs produce bad advanced questions.  The new version should keep those ideas, but make the core graph **question-driven** rather than random-topology-driven.

---

# Should you keep the old topics?

Your old topics are still good:

```python
TOPICS = [
    "Screening & diagnosis",
    "Treatment effectiveness",
    "Hospital data",
    "Education",
    "Social Science",
    "Labor & Policy",
    "User Behavior",
    "Criminal Justice",
]
```

But I would reorganize them into **topic × archetype compatibility**.

Some archetypes work better in some topics:

| Archetype                                   | Best topics                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| safety-constrained intervention             | Screening & diagnosis, Treatment effectiveness, Hospital data, Criminal Justice |
| confounding reversal                        | Treatment effectiveness, Education, Criminal Justice, Labor & Policy            |
| direct-vs-mediated mechanism                | all topics                                                                      |
| mediator identification                     | all topics                                                                      |
| satisficing intervention                    | all topics                                                                      |
| subgroup-robust intervention                | Education, Labor & Policy, User Behavior, Hospital data                         |
| next experiment / mechanism discrimination  | Treatment effectiveness, Screening & diagnosis, User Behavior, Education        |
| invalid intervention / wrong causal premise | all topics                                                                      |

So topic selection should be:

1. choose archetype;
2. choose compatible topic;
3. generate a domain-specific instantiation.

Not:

1. choose topic;
2. randomly generate graph;
3. hope good questions exist.

---

# Overall generation pipeline

## Step 0: Choose generation mode

Expected outcome:

```json
{
  "topic": "Treatment effectiveness",
  "archetype": "safety_constrained_intervention",
  "n_nodes": 20,
  "seed": 123
}
```

You should sample from a controlled distribution, not uniformly random forever.

For example:

```text
25% safety-constrained intervention
15% confounding reversal
15% direct-vs-mediated
15% mediator identification
10% satisficing intervention
10% subgroup robustness
5% next experiment
5% invalid premise
```

At first, I would over-sample the stable types until the pipeline is reliable.

Expected outcome: every world has a known scientific purpose before variables or edges are generated.

---

# Step 1: Ask LLM for a scenario-role plan, not a graph

This is the first important LLM call.

Do **not** ask the LLM for CPDs. Do **not** ask it to generate the final graph. Ask it to instantiate semantic roles for the selected topic and archetype.

For example, for `safety_constrained_intervention`, ask:

```text
Topic: Treatment effectiveness.
Archetype: A researcher must choose one intervention that improves a target outcome without worsening a safety outcome.

Generate a realistic causal study plan with:
- one target outcome, ordinal or binary, with clear preferred direction
- one safety/side-effect outcome, ordinal or binary, with clear preferred direction
- 3-5 manipulable intervention variables, each binary or ternary
- 2-4 mediators
- 2-3 baseline confounders or context variables
- 2-5 extra realistic variables
- mark which variables are intervenable and non-intervenable
- give values for each variable
- give a one-sentence description
- do not give probabilities
- do not give final edges except role-level causal expectations
```

Expected LLM output:

```json
{
  "topic": "Treatment effectiveness",
  "archetype": "safety_constrained_intervention",
  "study_name": "outpatient hypertension medication adjustment study",
  "roles": {
    "target": "FollowUpVisits",
    "safety_outcome": "MissedComplication",
    "good_intervention": "NurseCallFrequency",
    "risky_intervention": "AppointmentSpacingPolicy",
    "weak_intervention": "PatientPortalReminder",
    "mediators": ["BloodPressureControl", "PatientAnxiety", "SymptomReporting"],
    "confounders": ["BaselineSeverity", "ComorbidityBurden"],
    "non_intervenable": ["AgeGroup", "BaselineSeverity", "ComorbidityBurden"]
  },
  "variables": [...]
}
```

The LLM’s job is naming and realism. Your code’s job is structure and probability.

Issues to watch:

* Reject variables with too many states for key interventions.
* Reject target/side-effect variables without clear direction.
* Reject semantically impossible roles, like `AgeGroup` as an intervention.
* Reject duplicate meanings under different names.
* Reject overly obvious decoys like `FavoriteColor`.

Your existing code already has variable generation and sanitization logic that asks for plausible discrete variables and cleans names/values, but for this new pipeline, that prompt should be role-aware rather than chunk-only. 

---

# Step 2: Normalize variable specs

Before graph building, code should normalize the role plan.

Expected outcome:

```json
{
  "variables": [
    {
      "name": "FollowUpVisits",
      "values": ["None", "One", "TwoOrMore"],
      "role": "target",
      "intervenable": false,
      "preferred_low": true,
      "cardinality": 3
    },
    {
      "name": "NurseCallFrequency",
      "values": ["None", "Monthly", "Weekly"],
      "role": "good_intervention",
      "intervenable": true,
      "preferred_low": null,
      "cardinality": 3
    }
  ]
}
```

Controls here:

* Key interventions: 2–3 values.
* Target and safety: 2–4 ordered values.
* Mediators: 2–4 ordered values.
* Confounders: 2–4 values.
* Extra categorical variables: allowed, but should not be central targets.

For this benchmark, I would **not** allow 20–30 state variables in central roles. Your current older generator allows up to 30 values for realism, which is fine for generic variables, but it can make adaptive experimental search too expensive and noisy. 

Expected outcome: a clean variable table where central roles are small-cardinality and scoreable.

---

# Step 3: Build a required causal skeleton from the archetype

This is where you control the meaningful question.

Do not let the LLM decide these core edges. Use deterministic archetype skeletons.

## Example A: safety-constrained intervention

Required skeleton:

```text
BaselineSeverity → Target
BaselineSeverity → SafetyOutcome

GoodIntervention → GoodMediator → Target
RiskyIntervention → Target
RiskyIntervention → SafetyOutcome
WeakIntervention → Target

Optional:
GoodIntervention → SafetyOutcome with zero or tiny safe effect
RiskyIntervention → Mediator → Target
```

Expected outcome:

```json
{
  "required_edges": [
    ["BaselineSeverity", "FollowUpVisits"],
    ["BaselineSeverity", "MissedComplication"],
    ["NurseCallFrequency", "BloodPressureControl"],
    ["BloodPressureControl", "FollowUpVisits"],
    ["AppointmentSpacingPolicy", "FollowUpVisits"],
    ["AppointmentSpacingPolicy", "MissedComplication"],
    ["PatientPortalReminder", "FollowUpVisits"]
  ],
  "forbidden_edges": [
    ["NurseCallFrequency", "MissedComplication"],
    ["MissedComplication", "AppointmentSpacingPolicy"],
    ["FollowUpVisits", "BaselineSeverity"]
  ]
}
```

## Example B: confounding reversal

Required skeleton:

```text
Confounder → Treatment
Confounder → Target
Treatment → Mediator → Target
```

Expected signature:

```text
observational Treatment appears worse
interventional Treatment is better
```

## Example C: direct vs mediated

Required skeleton depends on answer class.

Mediated only:

```text
X → M → Y
forbid X → Y
forbid X → other_mediator → Y
```

Direct + mediated:

```text
X → M → Y
X → Y
```

Not mediated through proposed M:

```text
X → TrueMediator → Y
X → ProposedM
forbid ProposedM → Y
```

## Example D: subgroup robustness

Required skeleton:

```text
Group → Target
Group → Mediator
InterventionA → Target
InterventionB → Mediator → Target
```

And `Target` CPD must have parents including both `Group` and intervention, so you can encode effect modification.

Expected outcome of Step 3: a partial DAG skeleton with required edges, forbidden edges, role labels, and intended empirical signature.

---

# Step 4: Add extra variables and background edges

Now you can ask the LLM for plausible background edges, but only among non-central or allowed pairs.

Prompt:

```text
Given these variables and this required causal skeleton, propose additional plausible causal edges that make the world realistic.

Rules:
- Do not contradict required or forbidden edges.
- Do not create cycles.
- Do not add edges into baseline immutable variables such as AgeGroup or PriorHistory.
- Keep the graph sparse.
- Prefer edges that create realistic confounding, mediation, or side pathways.
- Do not add alternate paths that would change the intended answer.
```

Expected outcome:

```json
{
  "candidate_extra_edges": [
    ["ComorbidityBurden", "BloodPressureControl"],
    ["HealthLiteracy", "MedicationAdherence"],
    ["MedicationAdherence", "BloodPressureControl"],
    ["PatientAnxiety", "SymptomReporting"]
  ]
}
```

Then code enforces:

* DAG acyclicity.
* max in-degree.
* no forbidden edges.
* no central-answer-breaking path.
* no too-long directed path.
* no disconnected central component.

Your older generator already uses LLM edge candidates but code enforces acyclicity, max parents, target edge count, and connectedness.  Keep that pattern, but add archetype-specific constraints.

Expected outcome: a full graph with meaningful central skeleton plus realistic background structure.

---

# Step 5: Decide graph-level desired features

## Connected or disconnected?

For your use case, I recommend:

> The **central task subgraph must be weakly connected**.
> The full graph should usually be weakly connected.
> Allow at most one small disconnected distractor component only if you deliberately want irrelevant variables.

For most worlds, keep it connected.

Reason: your scientist agent receives CSV samples. If there are disconnected variables, they mostly add noise and token cost, not useful adaptive reasoning. Your current advanced generator intentionally moved toward single-component DAGs because advanced questions need depth and branching; it rejected disconnected graphs and required nontrivial path length and colliders.  I agree with that.

## Desired graph size

Have the option for 10, 15, 20 nodes. I will first test with 10 nodes.

## Desired edge density

Use mean in-degree around 1.5–2.0.

For 20 nodes: around 28–35 edges.

But central variables should not have too many parents. Keep target parents ≤ 3 or 4. CPD tables grow with parent cardinality, and too many parents make exact patterns harder to infer from finite samples. Your current advanced code caps max in-degree because CPD size grows exponentially. 

## Desired path length

For 20 nodes:

* minimum longest directed path: 4
* maximum longest directed path: 6

For 30 nodes:

* minimum: 5
* maximum: 7

This matches the logic already in your advanced topology: enough depth for mediation and multi-step reasoning, but not so long that dependence attenuates and faithfulness fails. 

## Desired structures

Each world should contain at least:

* one intervention → mediator → target chain;
* one confounder structure: `C → X` and `C → Y`;
* one competing intervention;
* one side path or decoy mediator;
* one non-intervenable variable;
* optionally one collider, but do not overemphasize collider puzzles unless the question needs it.

Expected outcome: a graph where the answer is discoverable through active sampling, not obvious from structure or impossible due to weak effects.

---

# Step 6: Build CPDs in two layers

This is the most important implementation change.

Your current strong CPD builder samples CPDs to satisfy faithfulness. It fixes two real problems: softmax saturation where a class becomes ~0.9 regardless of parents, and near-deterministic roots that mute downstream dependence. It uses narrower biases, stronger weights, and root probability floors. 

For the new benchmark, I would not use purely random strong CPDs for central variables. Use:

1. **controlled CPDs for archetype-critical variables**;
2. **random strong CPDs for background variables**;
3. **post-hoc validation using exact inference**.

## Layer 1: root/base CPDs

For root variables like `BaselineSeverity`, `AgeGroup`, `PriorRisk`, use non-degenerate marginals.

Expected:

```text
P(BaselineSeverity=Low/Medium/High) = 0.35 / 0.40 / 0.25
```

Avoid:

```text
0.92 / 0.06 / 0.02
```

because rare states create unreliable subgroup/intervention estimates.

Your existing root floor idea is good; keep it. 

## Layer 2: assignment/treatment CPDs

For confounding reversal:

```text
BaselineSeverity → Treatment
```

Make treatment more likely for severe cases, but not deterministic.

Example:

```text
P(HomeMonitoring=Yes | Severity=Low) = 0.20
P(HomeMonitoring=Yes | Severity=Medium) = 0.45
P(HomeMonitoring=Yes | Severity=High) = 0.70
```

Expected outcome: enough confounding, but enough overlap.

Avoid:

```text
Low: 0.01
High: 0.99
```

because observational stratification becomes unstable.

## Layer 3: mediator CPDs

For a good intervention:

```text
NurseCallFrequency → BloodPressureControl
```

Make mediator response visible.

Example:

```text
P(BloodPressureControl=Good | NurseCall=None) = 0.30
P(BloodPressureControl=Good | NurseCall=Monthly) = 0.50
P(BloodPressureControl=Good | NurseCall=Weekly) = 0.70
```

Expected outcome: the agent can discover that intervention moves mediator.

## Layer 4: target CPDs

For `FollowUpVisits`, combine baseline severity and mediator.

Example preferred-low target:

```text
BloodPressureControl=Good lowers FollowUpVisits.
BaselineSeverity=High raises FollowUpVisits.
AppointmentSpacingPolicy=Sparse lowers measured FollowUpVisits but may raise MissedComplication.
```

Expected outcome: target changes under `do(intervention)` by a visible margin.

## Layer 5: side-effect CPDs

For safety-constrained questions, side effect must be **separable** from target.

This means:

* one action improves target but worsens safety;
* one action improves target moderately and does not worsen safety;
* one action weakly improves or does nothing.

Example:

```text
AppointmentSpacingPolicy=Sparse:
  FollowUpVisits improves by 0.45
  MissedComplication worsens by 0.25

NurseCallFrequency=Weekly:
  FollowUpVisits improves by 0.30
  MissedComplication changes by -0.03 or 0.00

PortalReminder=On:
  FollowUpVisits improves by 0.08
  MissedComplication changes by 0.00
```

This is exactly where your earlier probability design should be controlled: **inside the controlled CPD generator for target and side-effect variables**, not in the LLM prompt and not after random CPD sampling.

Expected outcome: the best target-only intervention is not necessarily the best constrained intervention.

---

# Step 7: Archetype-specific probability signatures

Each archetype should have a required empirical signature. After building CPDs, exact inference must verify it.

## 1. Safety-constrained intervention

Expected validation:

```text
Best target-only action worsens side effect.
At least one safe action improves target.
Safe best action has clear gap over other safe actions.
```

Reject if:

* all actions are safe and monotonic;
* no action is safe;
* target-only best is also safe and dominates everything;
* side-effect differences are tiny.

## 2. Confounding reversal

Expected validation:

```text
E[Y | X=treated] worse than E[Y | X=untreated]
E[Y | do(X=treated)] better than E[Y | do(X=untreated)]
```

Reject if:

* observational and interventional signs match;
* sign reversal margin is small;
* treatment assignment has poor overlap.

## 3. Direct vs mediated

Expected validation:

Mediated-only:

```text
X changes M.
M changes Y.
X changes Y.
After fixing/intervening on M, residual X effect is near zero.
```

Direct+mediated:

```text
X changes M.
M changes Y.
X changes Y.
Residual direct X effect remains above threshold.
```

Reject if:

* mediator barely changes;
* direct and mediated effects cancel;
* alternate paths accidentally create ambiguity.

## 4. Mediator identification

Expected validation:

```text
True mediator has nontrivial mediated effect.
Decoys have weaker or no mediated effect.
All candidates are semantically plausible.
```

Reject if:

* multiple mediators are equally valid unless question allows set answer;
* true mediator is obvious by name;
* decoys are independent and too easy.

## 5. Satisficing intervention

Expected validation:

```text
At least one action above hidden threshold, or intentionally empty-gold world.
No action lies close to threshold.
```

Your advanced code already uses a guard idea to keep thresholds away from effect values. Keep that. 

## 6. Subgroup-robust intervention

Expected validation:

```text
Action A best on average but fails one subgroup.
Action B improves all subgroups.
Gold action maximizes minimum subgroup improvement.
```

Reject if:

* subgroup too rare;
* all actions behave the same across groups;
* global best and robust best are identical in every generated world.

## 7. Next experiment / mechanism discrimination

Expected validation:

```text
Two candidate mechanisms are observationally plausible.
One intervention would clearly distinguish them.
The gold experiment has higher expected contrast than alternatives.
```

I would initially make this multiple-choice among candidate interventions. Free-form “most informative” is too hard to evaluate reliably.

## 8. Invalid premise / non-intervenable

Expected validation:

```text
Proposed variable is predictive but non-intervenable, or proposed causal direction is impossible.
At least one valid manipulable alternative exists.
```

Reject if:

* invalidity is only semantic and not encoded in metadata;
* no good alternative intervention exists;
* proposed invalid variable is irrelevant, making the question trivial.

---

# Step 8: Faithfulness check

After CPDs are built, run your existing global faithfulness check, but treat it as one layer, not the final guarantee.

Expected outcome:

```text
model.check_model passes
global faithfulness passes
task-specific empirical signature passes
```

Your current advanced code builds a BN, checks `model.check_model()`, then checks faithfulness with an epsilon and possibly sampled triples for larger graphs.  Keep this.

But add:

```text
task_validity_check(world, archetype)
```

because a globally faithful BN can still produce a bad benchmark item.

---

# Step 9: Generate story after graph and roles are fixed

This is another LLM call.

The story should be generated **after** the graph exists, because it must be consistent with variables and central causal logic.

Prompt:

```text
Write a 3-5 sentence realistic research scenario.
Do not mention DAGs or Bayesian networks.
Include the institutional setting, what data can be sampled, and what decision the researcher wants to make.
Be consistent with these variables and causal roles.
Do not reveal the answer.
```

Expected outcome:

```json
{
  "story": "You are a researcher at a community hypertension clinic studying how follow-up policies affect avoidable visits and patient safety..."
}
```

Your current generator already creates a story after variables, edges, and CPDs are built, which is the right location.  The difference is that now the story should also receive the archetype role plan.

---

# Step 10: Generate the natural-language question

This should mostly be template-based, with LLM lightly polishing if needed.

For example, for safety-constrained intervention:

```text
The clinic wants to reduce avoidable FollowUpVisits without increasing MissedComplication. Based on experiments you can run in this world, which intervention would you recommend, and why?
```

Do not include:

* exact budget per question;
* “under 5 steps”;
* “calculate E[Y|do(X)]”;
* hidden threshold values;
* too much graph language.

Expected outcome:

```json
{
  "question": "...",
  "question_type": "safety_constrained_intervention",
  "metadata": {
    "target": "FollowUpVisits",
    "safety_outcome": "MissedComplication",
    "candidate_interventions": [...],
    "evaluation_rule": "constrained_argmin_or_pareto",
    "hidden_gold": null,
    "lazy_eval": true
  }
}
```

This matches your preference: the global budget belongs to the agent loop, not the individual question text.

---

# Step 11: Lazy gold evaluation

You can avoid storing one fixed answer string, but you should store enough metadata to recompute the gold.

Expected metadata:

```json
{
  "archetype": "safety_constrained_intervention",
  "target": "FollowUpVisits",
  "target_direction": "minimize",
  "safety_outcome": "MissedComplication",
  "safety_direction": "minimize",
  "candidate_actions": [
    ["NurseCallFrequency", "Weekly"],
    ["AppointmentSpacingPolicy", "Sparse"],
    ["PortalReminder", "On"]
  ],
  "evaluation_rule": {
    "type": "constrained_argmin",
    "safety_tolerance": 0.03,
    "tie_tolerance": 0.05
  }
}
```

Expected outcome: during evaluation, you compute exact `do` effects from the saved BN and accept:

* exact best;
* all near-ties within tolerance;
* all Pareto-valid actions if the question asks for acceptable options.

This is better than storing a fragile single answer.

---

# Step 12: Final world rejection checks

Before saving, each world should pass a checklist.

## General graph checks

Expected:

```text
DAG: yes
central component connected: yes
full graph weakly connected: usually yes
max in-degree: ≤ 3, maybe ≤ 4 for target in subgroup worlds
longest path: 4-6 for N=20
at least one mediator chain
at least one confounder/fork
at least one non-intervenable variable
no central forbidden edges
```

## General probability checks

Expected:

```text
root state min probability ≥ 0.08 or 0.10
no central variable has state probability < 0.05 under baseline
no central CPD row has probability exactly 0 or 1
key intervention effects ≥ min_effect
gold-vs-runner-up gap ≥ min_gap unless set-valued
threshold margins ≥ guard_eps
```

## Agent-answerability checks

Expected:

```text
answer discoverable from observational/interventional CSV samples
does not require seeing graph
does not require hidden metadata
does not require impossible conditioning on continuous values
candidate action space not too large
```

## Anti-degeneration checks

Reject if:

* all interventions point in the same direction on all outcomes;
* best action can be guessed from variable name alone;
* no real tradeoff exists;
* target is not scoreable;
* side effect is independent of all interventions;
* the direct effect is too small to detect;
* subgroup has too few samples;
* a decoy mediator is obviously irrelevant;
* the question text reveals the intended answer.

---

# Where exactly do we control probability design?

For the example you quoted:

> HomeMonitoring improves BloodPressureControl but slightly increases MissedComplication; NurseCallFrequency moderately reduces FollowUpVisits safely; AppointmentPolicy reduces FollowUpVisits but increases MissedComplication.

This should be controlled in **three places**:

## 1. Role-plan stage

The LLM names the roles:

```json
{
  "good_intervention": "NurseCallFrequency",
  "risky_intervention": "AppointmentSpacingPolicy",
  "weak_intervention": "PatientPortalReminder",
  "good_mediator": "BloodPressureControl",
  "target": "FollowUpVisits",
  "safety_outcome": "MissedComplication"
}
```

It does not set probabilities.

## 2. Required skeleton stage

Code creates the causal paths:

```text
NurseCallFrequency → BloodPressureControl → FollowUpVisits
AppointmentSpacingPolicy → FollowUpVisits
AppointmentSpacingPolicy → MissedComplication
PatientPortalReminder → FollowUpVisits
BaselineSeverity → FollowUpVisits
BaselineSeverity → MissedComplication
```

This ensures the right variables can causally affect the right outcomes.

## 3. Controlled CPD stage

Code sets CPD parameters so the empirical signature actually holds:

```text
do(NurseCallFrequency=Weekly):
  target improves moderately
  safety does not worsen

do(AppointmentSpacingPolicy=Sparse):
  target improves strongly
  safety worsens

do(PatientPortalReminder=On):
  target improves weakly
  safety neutral
```

Then validation checks exact effects.

This is the key point: **the LLM gives semantic plausibility; code enforces causal and statistical behavior.**

---

# Suggested final architecture

I would implement the generator as these conceptual modules:

## 1. `ArchetypeSpec`

Expected content:

```json
{
  "name": "confounding_reversal",
  "compatible_topics": [...],
  "required_roles": [...],
  "required_edges_template": "...",
  "forbidden_edges_template": "...",
  "cpd_signature": "...",
  "validation_checks": [...]
}
```

## 2. `RolePlanGenerator`

LLM-based.

Input:

```text
topic + archetype + n_nodes
```

Output:

```text
variables + roles + values + descriptions
```

## 3. `SkeletonBuilder`

Code-based.

Input:

```text
role plan
```

Output:

```text
required edges, forbidden edges, central variables
```

## 4. `BackgroundGraphGenerator`

LLM + code.

Input:

```text
role plan + required skeleton
```

Output:

```text
full DAG
```

## 5. `ControlledCPDBuilder`

Mostly code.

Input:

```text
full DAG + archetype CPD signature
```

Output:

```text
BN with CPDs
```

## 6. `Validator`

Exact inference.

Checks:

```text
faithfulness
task signature
effect gaps
sample support
answerability
```

## 7. `QuestionWriter`

Template-based, optional LLM polish.

Output:

```text
natural question + metadata
```

## 8. `LazyGoldEvaluator`

Exact inference at evaluation time.

Output:

```text
accepted answers / scores
```

---

# Expected generated example

For topic `Treatment effectiveness`, archetype `safety_constrained_intervention`, expected final world:

```json
{
  "topic": "Treatment effectiveness",
  "archetype": "safety_constrained_intervention",
  "story": "You are a researcher at an outpatient hypertension clinic...",
  "variables": [
    "AgeGroup",
    "BaselineSeverity",
    "ComorbidityBurden",
    "NurseCallFrequency",
    "AppointmentSpacingPolicy",
    "PatientPortalReminder",
    "BloodPressureControl",
    "PatientAnxiety",
    "SymptomReporting",
    "FollowUpVisits",
    "MissedComplication",
    "EmergencyVisit",
    "MedicationAdherence",
    "HealthLiteracy",
    "CareAccess"
  ],
  "question": "The clinic wants to reduce avoidable FollowUpVisits without increasing MissedComplication. Which intervention would you recommend, and why?",
  "metadata": {
    "target": "FollowUpVisits",
    "safety_outcome": "MissedComplication",
    "candidate_interventions": [
      "NurseCallFrequency",
      "AppointmentSpacingPolicy",
      "PatientPortalReminder"
    ],
    "evaluation_rule": "constrained_argmin_with_ties"
  }
}
```

Expected empirical behavior:

```text
AppointmentSpacingPolicy=Sparse:
  strongest reduction in FollowUpVisits
  worsens MissedComplication

NurseCallFrequency=Weekly:
  moderate reduction in FollowUpVisits
  does not worsen MissedComplication

PatientPortalReminder=On:
  weak reduction
  safe but not best
```

The scientist agent can discover this by:

1. sampling baseline observational data;
2. testing `do(AppointmentSpacingPolicy=Sparse)`;
3. checking side effect;
4. testing `do(NurseCallFrequency=Weekly)`;
5. comparing target and safety;
6. answering with reasoning.

That directly tests the ability to sequence experiments.

---

# My strongest implementation advice

Do **not** try to make the LLM generate a complete good BN.

Use the LLM for:

* realistic variable names;
* plausible values;
* domain story;
* plausible extra edges;
* natural question wording.

Use code for:

* role constraints;
* required/forbidden edges;
* graph validity;
* CPDs;
* effect sizes;
* faithfulness;
* task validity;
* lazy gold.

This division will give you worlds that are both realistic and reliable.
