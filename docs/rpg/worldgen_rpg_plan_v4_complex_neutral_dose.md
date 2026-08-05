# RPG v4 — Harder Latent Discovery with Neutral, Non-Binary Actions

> Extends `worldgen_rpg_plan_static_partial_observation.md` (the v2/v3 static
> plan) and the v3 `story_hidden_cause_discovery` archetype shipped in
> `world_gen_rpg.py`. This plan does **not** introduce a new archetype or new
> role contract — it hardens the existing archetype so the worlds are no longer
> solvable by name-matching, and so interventions are no longer all binary.
> Everything stays role-driven, so `simulator_rpg.py`, `world_model_rpg.py`,
> `evaluate_rpg.py`, and `audit_rpg.py` keep working with at most a one-line
> change for continuous knobs.

## 0. Why this revision

The v3 gutter pilot (`out_rpg_v3_story_hidden_v1`) passed 2/2 with Opus, but it
is **too legible** and **too uniform** (see `rpg_v3_story_hidden_slide_materials.md`,
slides 7 and 16):

1. **Name leakage.** The whole design hides the latent cause in the *story*.
   But the action catalog re-exposes it: `ClearRearGutters`, `FlushDownspout`,
   `RunRoofEdgeFlowTest`, and proxy names like `DownspoutDischargeDelay` /
   `RoofEdgeOverflowScore` let a strong model read the answer off the labels and
   solve it in one intervention. The hiding in the story is undone by the menu.
2. **All interventions are binary (`off`/`on`).** The only "choice" is which
   single knob to flip. There is no dose, no setpoint, no genuine
   right-amount decision — so the scientist's action space has no internal
   structure.
3. **The one-shot example is too easy.** The gutter world has an obvious
   surface clue (leaves), an obvious named fix, and a monotone "more is better"
   action. Nothing forces real scientific deliberation.

### Design goals for v4

- **The core task stays latent-variable discovery + mechanism explanation.**
  The scientist must still infer an *unobserved* story-plausible cause and
  explain the mechanism. The richer action space is in service of that — it is
  not a best-arm bandit.
- **No name leakage.** Action names are clinically/operationally **neutral**
  (e.g. `RegimenB`, `SupportiveInfusionRate`) and never reveal which knob
  targets the hidden cause. The agent must learn the mapping from *queried
  data*, not from labels. Mechanism-proxy names are likewise neutral.
- **Non-binary interventions.** Knobs are dose-valued (categorical with many
  levels, e.g. `{none, low, standard, high}`) and at least one is a
  **continuous setpoint** (`0–100`). "On/off" becomes a special case.
- **Actual complexity in the choice.** The right answer is *right-drug +
  right-dose*, where:
  - dose-response **saturates** (low dose is sub-therapeutic; benefit plateaus
    at the "standard" dose), and
  - the top dose carries an **over-treatment side-effect**, so "max everything"
    is *not* optimal — the gold dose is an interior level, and
  - a **palliative continuous knob** lowers the visible outcome proxy without
    touching the mechanism proxies or the hidden cause (the classic
    treat-the-symptom trap).

## 1. The "many actions vs. no actions" decision

The professor's framing was: either give *many* actions or *none*, because a
short, descriptively-named menu leaks the answer.

**Chosen direction: many neutral, non-binary actions.** Rationale:

- Keeping an intervention surface preserves the existing scoring + trajectory
  machinery (`latent_cause_hypothesis` requires a mechanism-verifying
  interventional query). Removing actions entirely would gut the part of the
  benchmark that proves the agent *did an experiment*, not just guessed from
  the story.
- The leakage the professor worried about comes from **descriptive names**, not
  from the *existence* of actions. Neutral names (`RegimenA…F`, infusion-rate
  setpoint) plus non-binary doses defeat name-matching while keeping the
  combinatorial richness the professor wanted ("bigger action space").
- The effective joint-action space is large: 10 neutral knobs, several with 4
  dose levels and one continuous, queried up to 3 at a time → far more than the
  176 binary candidates of v3, and none of them is labelled "this is the fix".

The **"no named fixes"** variant (diagnostics + a single continuous control
dial) is recorded here as a considered alternative; it is cleaner against
leakage but weakens the action-complexity dimension and the trajectory check,
so it is not the v4 default.

## 2. Role contract — unchanged

v4 reuses the v3 `story_hidden_cause_discovery` role contract verbatim
(`REQUIRED_OBSERVED_ROLES`, `REQUIRED_ACTION_ROLES`). What changes is **per-knob
metadata** and **the mechanism's reading of action values**, not the roles. This
is deliberate: the simulator/world-model/evaluator/audit are all role-driven, so
no new dispatch is needed.

### New per-knob metadata (template-level)

Each action may now declare:

```json
{
  "role": "targeted_fix_primary",
  "name": "RegimenB",
  "value_type": "dose",                       // "binary" | "dose" | "continuous"
  "values": ["none", "low", "standard", "high"],
  "default": "none",
  "description": "A daily oral regimen used for upper-GI mucosal support."
}
```

- `value_type: "binary"` → `values: ["off","on"]` (the v3 default; unchanged).
- `value_type: "dose"` → ordered categorical `values` list, low→high. The first
  entry is the baseline/"none".
- `value_type: "continuous"` → numeric setpoint with `min`/`max` (default
  `0`/`100`) and an optional `oracle_grid` the oracle sweeps.

### Dose magnitude

The mechanism converts any submitted value to a **dose fraction** `d ∈ [0,1]`:

- binary / dose categorical: `d = index(value) / (len(values) - 1)`
  (`off`/`none` → 0, top level → 1).
- continuous: `d = clip((value - min) / (max - min), 0, 1)`.

Every v3 effect that was gated `if knob == "on"` becomes scaled by `d`. For a
binary knob `d ∈ {0,1}`, reproducing v3 behaviour exactly — **the gutter world
is bit-for-bit unchanged.**

## 3. Mechanism generalization (`_apply_story_hidden`)

All changes are backward-compatible (binary ⇒ identical to v3) and gated by
template-supplied `mechanism_params` (defaults make non-medical worlds behave
exactly as before).

- **Targeted fixes** (`targeted_fix_primary`, `targeted_fix_secondary`) reduce
  the hidden cause with a **saturating** dose-response and an **over-treatment**
  penalty:
  ```text
  benefit_fraction = min(1, d / dose_saturation)          # plateaus at standard
  hidden_cause   -= base_reduction * benefit_fraction
  overtreat       = overtreat_penalty * max(0, d - dose_saturation)
  outcome        += overtreat_outcome * overtreat          # high dose hurts a bit
  secondary_out  += overtreat_secondary * overtreat        # high dose hurts more
  ```
  Defaults: `dose_saturation = 1.0`, `overtreat_penalty = 0.0` → linear, no
  penalty → v3 behaviour. Medical template sets `dose_saturation ≈ 0.66` (so the
  "standard" level captures full benefit) and `overtreat_penalty > 0` (so "high"
  is strictly worse) → **gold dose = standard**, an interior optimum.
- **Palliative continuous knob** (`symptom_mitigation`): `outcome -=
  palliative_outcome * d`, but it does **not** touch the hidden cause or the
  mechanism proxies. So it lowers the visible outcome reading while leaving the
  mechanism evidence unchanged — the symptom-vs-cause trap.
- All other v3 knob effects (`alternative_fix_*`, `partial_reroute`,
  `weak_buffer`, `distractor_check`, `cosmetic_action`, `diagnostic_test`)
  become `d`-scaled versions of their v3 forms; binary knobs are unaffected.

## 4. Oracle scoring over dose grids

`_candidate_story_interventions` now enumerates **value grids**, not just
on/off:

- each action contributes its non-baseline levels (`values[1:]` for
  categorical; an `oracle_grid` for continuous, default `[33, 66, 100]`);
- candidates = baseline `{}` + all level-combinations of action subsets up to
  `oracle_max_joint`.

`oracle_max_joint` is template-configurable to bound cost:

- gutter (binary): `oracle_max_joint = 3` → `1 + C(10,1)+C(10,2)+C(10,3) = 176`
  (unchanged, since each binary grid has size 1).
- medical (dose): `oracle_max_joint = 2`. Singles sweep full dose grids
  (~30 candidates); pairs are level×level combinations (~few hundred). The gold
  is a *single* regimen at the right dose, so singles + pairs comfortably
  contain it. The agent-facing `max_intervention_knobs` stays 3; any 3-knob
  answer the agent submits is rescored by fresh Monte Carlo
  (`_score_intervention_answer` already does this for unseen keys).

## 5. The new hard one-shot example — medical dose-response

`_template_medical_dose()` (neutral names, non-binary, latent-discovery core).

- **Domain.** A fatigue/anemia clinic. Patients show a rising
  `AnemiaSeverityIndex` (outcome). The prevailing theories are diet, thyroid,
  and sleep — all decoys.
- **Hidden cause.** `OccultBloodLossBurden` — an unrecognized slow internal
  blood loss that standard panels miss. plain_name "an unrecognized slow
  internal blood loss"; aliases: occult/internal/GI/slow bleeding etc. It is
  mentioned only obliquely in the story (a casually-listed antiplatelet use, a
  long gap since GI evaluation) and is **never** an observed column or an action
  name.
- **Surface clue (visible_trigger).** `AntiplateletExposureIndex` — predicts the
  hidden loss but is not itself the cause (treating it doesn't stop an
  established bleed).
- **Mechanism proxies (neutral).** `MarrowCompensationIndex`,
  `IronStoreDepletionMarker`, and the diagnostic `OccultSourceAssaySignal`
  (informative only when the assay action is run). These move with the hidden
  cause — the agent discovers the latent state by querying them under
  interventions.
- **Alternative proxies (decoys).** `DietaryIronAdequacy`, `ThyroidActivityIndex`,
  `SleepDebtIndex`.
- **Actions (neutral, non-binary).**
  | role | name | value_type | values |
  |---|---|---|---|
  | targeted_fix_primary | `RegimenB` | dose | none/low/standard/high |
  | targeted_fix_secondary | `RegimenD` | dose | none/low/standard/high |
  | diagnostic_test | `OrderSourceAssay` | binary | off/on |
  | alternative_fix_primary | `RegimenA` | dose | none/low/standard/high |
  | symptom_mitigation | `SupportiveInfusionRate` | continuous | 0–100 |
  | partial_reroute | `RegimenE` | dose | none/low/high |
  | alternative_fix_secondary | `RegimenF` | dose | none/low/high |
  | distractor_check | `SleepProtocolAdjust` | binary | off/on |
  | weak_buffer | `MicronutrientAdjunct` | dose | none/low/high |
  | cosmetic_action | `WellnessCoachingTier` | dose | none/basic/premium |
  None names the bleed; the agent must learn from data which regimen reduces the
  mechanism proxies + outcome, and at which dose.
- **Gold.** Hidden cause = occult blood loss; gold action ≈ `RegimenB=standard`
  (full saturated benefit, no over-treatment penalty), with `OrderSourceAssay`
  as the decisive diagnostic. `SupportiveInfusionRate` is the palliative trap;
  `RegimenA/F` (diet/thyroid) are the decoys the agent must rule out.
- **Why it's harder.** The cause is genuinely unobserved (no column, no action
  names it); the surface clue is a confound, not the cause; the gold requires an
  *interior* dose; and the most tempting wrong move (crank the infusion) lowers
  the visible outcome while leaving every mechanism proxy elevated.

The gutter template stays in the file and stays selectable; the **medical
template becomes the built-in default** (`--builtin-template medical_dose`,
default; `yard_flooding` still available).

## 6. LLM template prompt update

`STORY_TEMPLATE_PROMPT` is updated so LLM-proposed worlds follow the same rules:

- Action names must be **neutral** and must **not** hint which knob targets the
  hidden cause (no "Clear…", "Fix…", "Stop…" verbs naming the mechanism).
- Mechanism-proxy names must be neutral too.
- Actions must be **non-binary**: most are dose-valued (`value_type:"dose"`,
  ordered `values`), at least one is `value_type:"continuous"` (a setpoint), and
  one diagnostic stays binary.
- The core remains latent discovery: the hidden cause is unobserved and only
  casually present in the story; the answer is a mechanism explanation, not an
  arm ranking.

## 7. Minimal runtime touches (so continuous works end-to-end)

Dose-categorical knobs already validate against each knob's `values` list, so
they work with **zero** runtime change. Continuous needs two small edits:

- `framework_code/simulator_rpg.py::validate_intervention` — when a knob's
  `value_type == "continuous"`, accept any numeric value within `[min,max]`
  (keep it numeric) instead of checking membership in a `values` list.
- `framework_code/schemas_rpg.py::get_intervention_catalog` — display the range
  for continuous knobs (and `value_type`) so the agent sees a setpoint, not an
  empty `values` list.

No change to `world_model_rpg.py` (it passes intervention values through
verbatim) or `evaluate_rpg.py` (it delegates rescoring to the simulator).

## 8. Validators

The generic acceptance checks (`corr_trigger_outcome`, `corr_context_outcome`,
`corr_hidden_cause_mechanism_proxy`, `targeted_beats_alternative_margin`) are
kept. The only change: the validator/diagnostic probe must use a **valid
therapeutic value** for each knob (the saturation-dose level for dose knobs,
`"on"` for binary, `max` for continuous) instead of the literal `"on"`, since
`"on"` is not a legal value for a dose knob. A `_therapeutic_probe_value(spec)`
helper returns this. For binary knobs it returns `"on"` → gutter checks
unchanged.

## 9. Backward-compatibility checklist

- Gutter world (`_template_yard_flooding`, all binary, `dose_saturation=1.0`,
  `overtreat_penalty=0`) produces **identical** oracle candidates (176) and
  identical mechanism outputs.
- Schema version, benchmark name, answer schema (`latent_cause_hypothesis`),
  and the role contract are unchanged.
- `_static_intervention_key` already maps `none`/`off`/`""` to baseline, so
  dose `none` is treated as "knob not used".

## 10. Smoke commands

Generate one medical world (built-in template, no LLM):

```bash
python3 dataset_generation_code/world_gen_rpg.py \
  --outdir /tmp/rpg_v4_medical_smoke \
  --builtin-template medical_dose \
  --distribution '{"story_hidden_cause_discovery":1}' \
  --oracle-n 8000 --max-attempts-per-world 3
```

Confirm gutter still validates unchanged:

```bash
python3 dataset_generation_code/world_gen_rpg.py \
  --outdir /tmp/rpg_v4_gutter_smoke \
  --builtin-template yard_flooding \
  --distribution '{"story_hidden_cause_discovery":1}' \
  --oracle-n 8000 --max-attempts-per-world 3
```

Audit:

```bash
python3 dataset_generation_code/audit_rpg.py \
  --outdir /tmp/rpg_v4_medical_smoke --static --summary-only --recheck-oracle-n 0
```

## 11. Open follow-ups (not in this change)

- Make the right dose depend on a **hidden subtype** (some patients need low,
  some high) so the optimal answer is a *conditional* dose policy — a true
  contextual decision. (Bridges to `latent_regime_policy`.)
- Expand to a larger neutral catalog (15–20 knobs) once oracle cost is sharded.
- Add mechanism families beyond the single shared role graph so agents can't
  learn one generic pattern (slide 16 risk).
