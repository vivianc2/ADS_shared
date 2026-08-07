# RPG v7 — structural sampling for scientific-reasoning worlds

v7 turns the RPG benchmark from *4 hand-authored worlds × seeds* into
*unlimited draws from parameterized structural families*. The v6 engine, oracle,
audits, resolver, sim, and runners are all structure-agnostic (they operate on
any `{world_id, domain, scenario, scm, ground_truth}` dict); v7 adds a **sampler**
that emits randomized-but-valid world dicts and lets the audit suite filter the
bad draws.

## Diversity axes

A generated world is a draw over:

- **skin** (domain / topic) — 10 available: `bioprocess, datacenter,
  watertreatment, agronomy, clinical, semiconductor, aquaculture, battery,
  catalysis, fermentation`. Skins are pure data in `skins.py` (role-keyed name
  banks + scenario template). Names are meaningful but **non-leaking**: knowing
  *what* a variable is must not reveal *whether* it is the cause.
- **archetype** (role-wiring / reasoning skill) — 3 in `ARCHETYPES`:
  - `confounded_chain` — break a confound + trace a mediation chain (the backbone).
  - `collider_selection` — a decoy correlates with the outcome **only** because
    the historical record is conditioned on a collider; the correlation vanishes
    under intervention. Skill: don't trust an observational correlation —
    verify it interventionally.
  - `hidden_subtype` — a treatment helps one hidden subgroup and harms the other
    (population-average ≈ 0). The ideal answer is a **conditional policy**
    (stratify on an observable marker, treat only the subgroup it helps), not a
    single dose. Graded by the conditional-policy path.
- **structure params** — chain depth (2–4), #confounders (1–2), #decoys (1–2),
  #distractors (8–12), #inert knobs (4–6).
- **difficulty features** — `sign_flip, interior_dose, two_cause, symptom_trap`.

## Run

```bash
conda activate ADS-rpg
# generate (mixed archetypes/skins), or restrict with --archetype / --skins / --require-feature
python generate_v7.py --outdir out_v7_batch --n 60 --seed 300000
# collider is filtered more often (weak-correlation draws rejected); for a
# balanced set, generate per-archetype and merge:
python generate_v7.py --outdir out_v7_collider --n 20 --seed 1 --archetype collider_selection
python generate_v7.py --outdir out_v7_subtype  --n 20 --seed 2 --archetype hidden_subtype

# solvability smoke (mock plays the computed gold; want partA≈1.00, 0 artifacts)
python run_batch_v6.py --worlds-dir out_v7_batch --backend mock --outdir results_v7/mock

# the real read (needs AWS_BEARER_TOKEN_BEDROCK + AWS_DEFAULT_REGION=us-west-2)
python run_batch_v6.py --worlds-dir out_v7_batch --backend bedrock \
    --model us.anthropic.claude-opus-4-8 --outdir results_v7/opus -v
python analyze_results.py --run opus=results_v7/opus --out results_v7/report.md
```

### Resuming an interrupted batch

If a run breaks or you stop it, re-run the **same command** with `--resume`: it
reads `--outdir` for existing `result_<world_id>.json` files, skips those worlds
(no re-spent API), runs only the rest, and rebuilds `summary.json` over all of
them. If every world is already done, it rebuilds the summary without building an
LLM client at all (so no API creds are needed just to re-aggregate). Use
`--force` to re-run everything regardless.

```bash
python run_batch_v6.py --worlds-dir out_v7_batch --backend bedrock \
    --model us.anthropic.claude-opus-4-8 --outdir results_v7/opus -v --resume
```

`manifest.json` records `skin_distribution`, `archetype_distribution`,
`feature_distribution`, and `rejected_by_gate`.

## What each archetype guarantees (verified)

- Every generated world's **gold passes its own audits** (solvable + self-
  consistent). Mock (gold-replaying) batches hit partA=1.00 across all
  archetypes.
- `collider_selection`: selection decoy has strong observational corr (~0.4)
  with the outcome but **zero do-effect**, and is never in the gold. Weak-corr
  draws are rejected by `decoy_audit` (hence lower acceptance for this archetype).
- `hidden_subtype`: uniform dosing nets ~0 (fails counterintuitiveness as a
  naive move); the conditional policy (± the chain fix) recovers full benefit.
  Gold is the **combined** chain-fix + stratified-policy; a single-dose answer
  cannot pass part A.

## Engine additions (all additive, backward-compatible)

- `WorldSCM.selection` + `sample(..., select=True)` — selection/conditioning on a
  collider. Applied only to **observational** draws; auto-disabled under any
  intervention, so the oracle's interventional truth is never selection-distorted.
- `subtype_effect` mechanism — treatment effect whose sign depends on a hidden
  subtype.
- Conditional (per-unit) `set` actuator via `{"policy": {stratifier, threshold,
  dose_if_ge, dose_if_lt}}` — lets a stratified policy be executed and graded.
- `oracle_v6.optimal_gold(world)` — world-aware gold (policy for subtype worlds,
  actuator-combo optimum otherwise); `grade()` accepts a `recommended_policy`
  and judges part A by fraction-of-benefit-recovered (≥0.90).

## Not yet built

- Feedback/homeostasis archetype (needs cyclic / time-unrolled sampling).
- Instrument-only-cause archetype (true cause not directly actuable).
- Arbitrary-DAG (tier-3) sampling.
