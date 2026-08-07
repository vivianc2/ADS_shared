# RPG v6 — Complete Documentation

Authoritative record of RPG v6: design, implementation, evaluation methodology,
results, every bug found and fixed, and known limitations. Written as v6 is
frozen and work moves to v7. Snapshot: 2026-08-05.

Companion docs (all under `docs/rpg/`):
- `worldgen_rpg_plan_v6_open_scenario.md` — the original design plan.
- `rpg_v6_pipeline_overview.md` — system map.
- `rpg_v6_slides.md` — advisor deck + the final results table.
- `rpg_v6_batch20_audit.md` — the fairness audit that drove the token/turn fixes.
- `rpg_v6_datacenter_run_analysis.md` — a single-run close reading.

Code: `dataset_generation_code/rpg_v6_prototype/`.

---

## 1. What v6 is

An LLM "scientist" agent is dropped into a partially-observed system whose true
mechanics are a fully-specified structural causal model (SCM) we own. It reads a
plain-language situation report — **no variable list, no action menu** — and must
identify the hidden cause of a degraded outcome and prescribe an intervention,
within a bounded number of experiments. Because we own the SCM, the correct
answer (best intervention + the causal structure) is computed, and the agent is
graded against it.

Three design goals, and the failure mode each guards against:
- measure **hypothesis formation + experiment design + mechanism recovery** under
  a budget — not recall;
- **no leakage**: names are meaningful (so world knowledge is usable) but the
  world is large enough that knowing *what* a variable is does not reveal *whether
  it matters*;
- **exact grading**: the answer is scored against SCM-computed quantities, never
  fuzzy text matching.

v6's core bet, validated: the engine/oracle/audits are **structure-agnostic**, so
the whole difficulty comes from the world's causal structure, and worlds are
hand-authored SCM specs. (v7 replaces hand-authoring with a structural sampler.)

---

## 2. World representation

A world = a DAG of typed nodes + a set of actuators + a prose scenario + hidden
ground-truth role labels.

**Node kinds:** `latent` (no assay), `observable` (assay + noise), `outcome`.
Each non-root node computes from its parents via one **mechanism** (closed
library); exogenous roots carry a distribution.

**Mechanism library** (`engine.py`; `x`/`a`/`b` are parent values):

| Form | Formula | Role |
|---|---|---|
| `linear` | `intercept + Σ wᵢ·parentᵢ` | weighted sum |
| `saturating` | `gain·x/(x+k)` | diminishing returns |
| `hill` | `vmax·xⁿ/(kⁿ+xⁿ)` | sigmoid / soft threshold |
| `soft_threshold` | `gain·σ((x−θ)/w)` | logistic switch |
| `interaction` | `intercept + gain·(a/s)·(b/s)` | bilinear (a's effect depends on b) |
| `sign_flip` | `intercept + lo·min(x,knee) + hi·max(x−knee,0)` | effect reverses at a knee |
| `gated_and` | `intercept + vmax·σ((a−tₐ)/wₐ)·σ((b−t_b)/w_b)` | AND-gate: high only if BOTH inputs pass |
| `abs` | `intercept + gain·abs(x−center)` | distance from an optimum (V-shape) |

**Actuators** are the only way to intervene (no free `do(X:=x)`). Each is a
record `{id, aliases, target, op, dtype, range/values, default, expr,
side_effect, description}` with one typed operation:

| op | Effect on `target` | Touches true state? | Models |
|---|---|---|---|
| `set` | force `target` to the value (overrides its mechanism) | yes | controller / clamp |
| `scale` | multiply `target` by `expr(d)` | yes | dosed treatment |
| `add` | add `expr(d)` to `target` | yes | dosed additive (unused by the 4 worlds) |
| `mask` | bias only the *measured reading* of `target` | **no** | symptom-masking trap |

Dose fraction `d∈[0,1]` = `(value−min)/(max−min)` continuous, `index/(len−1)`
discrete. `set/scale/add` edit the structural value and descendants re-propagate;
`mask` is applied only in `measure()`. Typing says whether the agent can SEE a
variable; the actuator set says whether it can CHANGE it — independent axes, so a
hidden cause with no actuator is reachable only via a downstream lever.

---

## 3. The four world families (topologies)

`worlds_v6.py`. Each stresses a different reasoning failure mode; sizes are
(variables / actuators).

| World | Size | Structure | The trap / required insight |
|---|---|---|---|
| **Bioreactor** titer loss | 25 / 14 | mediation chain + confound | trace outcome ← carbon-flux ← ROS ← copper past a DO confound; interior chelator dose. Easiest by construction (single lever). |
| **Datacenter** throughput | 23 / 11 | sign reversal | the obvious fix is *backwards* — more cooling → condensation → worse. True fix: dehumidify. |
| **Greenhouse** yield | 19 / 9 | two interacting causes (AND-gate) | neither lever works alone; fix needs iron **and** pH together. |
| **Clinic** readmission | 20 / 7 | latent-subtype heterogeneity | drug effect ≈0 on average, opposite-signed by subgroup; best is an interior dose. |

Each world also carries a **confounded decoy** (correlates with the outcome, zero
do-effect), many **inert distractors** (real but causally inert vars/knobs so
brute force is infeasible), and a **symptom-mask trap** (a `mask` actuator that
lifts the reading without changing the true state).

---

## 4. Agent protocol

`run_agent_v6.py`. The agent sees only: prose scenario, outcome name + direction,
budget. It issues **free-text** requests; a resolver maps them. Four actions, one
per turn:

- `measure(requests)` — resolve to observables, return per-unit rows (CSV) +
  summary stats + correlations. Costs 1 experiment.
- `intervene(actions, measure)` — resolve to actuators + doses, apply jointly
  (≤3), return post-intervention readings + rows. Costs 1 experiment.
- `code(...)` — sandboxed pandas/numpy/scipy over the collected CSVs. **Free** (no
  budget cost) — the agent can fit dose curves, regress out confounds, etc.
- `answer(...)` — structured final answer; ends the run.

**Defaults (current):** experiment budget 15; **max productive turns 60**
(raised from 32 — see §7); per-turn output tokens = **model max** (32768 Qwen/
gpt-oss, 65536 deepseek, 8192 Opus); joint interventions ≤3; 400 units/query.

**The resolver** (`resolver.py`): free text → canonical variable/actuator, or a
typed rejection (`no_assay` / `no_actuator` / `not_in_world`). Lexical scoring
(stem match, distinctiveness weighting via document frequency, request-coverage
gate) with an **LLM disambiguation fallback** (on by default, fixed strong model
independent of the agent). Every mapping is echoed to the agent and logged, so a
resolution miss is always visible and separable from a reasoning miss.

**The sandbox** (`sandbox.py`): spawned subprocess, pandas/numpy/scipy, hard
timeout, stdout captured, small vars carried across turns.

**Backends** (`build_llm`): `bedrock` (Opus), `nautilus`/`openai`
(OpenAI-compatible, e.g. Qwen `qwen3-small` on NRP Nautilus
`https://ellm.nrp-nautilus.io/v1`), `mock` (replays gold; wiring test only).

---

## 5. Golden answer + grading (both computed from the SCM)

`oracle_v6.py`. **Accept = Part A ∧ Part B.**

**Gold intervention:** screen each actuator's marginal effect, add a synergy pass
(pairs that only help jointly — catches the AND-gate), search combinations over
the active set (≤3), refine continuous doses by golden-section. Monte-Carlo
utility with common random numbers.

**Part A — found the fix:** recompute the agent's recommended intervention's
utility; require **benefit recovered ≥ 0.90**, where
`benefit = (rec − baseline) / (gold − baseline)`. (A fraction, not an absolute
tolerance — comparable across worlds of different utility ranges.)

**Part B — understood the structure (≥ 0.8 of items):** the agent names the true
mechanism proxy, the confounded decoys, and the sign of each actuator's effect.
- **proxy** credited against the SET of valid proxies (any measurable the true
  lever moves), not one hardcoded string; verbose parentheticals are stripped
  before resolving;
- **decoys**: must flag the true confounder(s), must not mislabel a real proxy;
- **signs**: increasing-direction convention (with a flip for "reduce X"
  phrasing); **non-monotone (interior-optimum) actuators are marked `skip`** and
  not scored — no single sign is correct for them.

**Per-world audits (a world ships only if all pass):**
- **decoy inertness** — confounder correlates (|r|≥0.3) but forcing it moves the
  outcome ≤0.6;
- **proxy signal** — true proxy informative: observational |r|∈[0.35,0.75] OR
  shifts >1 SD under the targeted intervention (for dormant-at-baseline worlds);
- **distractor inertness** — every non-active actuator ≈0 marginal + interaction;
- **gold self-consistency** — no single-actuator perturbation beats gold;
- **counterintuitiveness** — declared naive interventions (the obvious first move)
  gain < 3.0 over baseline.
Calibration auto-tunes proxy assay-noise and confounder loading to hit the bands
before auditing.

---

## 6. Results (batch20: 4 topologies × 5 seeds = 20 worlds)

Both models, same 20 audited worlds, identical protocol and grader. Qwen from the
fixed re-run (`batch20_qwen_2`). Runs done at max-turns 32 (before the raise to
60).

**Overall:**

| Model | n | Accepted | Accuracy [95% CI] | Part-A | Mean Part-B |
|---|---|---|---|---|---|
| Opus-4.8 | 20 | 4 | 0.20 [0.08, 0.42] | 0.50 | 0.65 |
| Qwen3.6-27B (`qwen3-small`) | 20 | 2 | 0.10 [0.03, 0.30] | 0.40 | 0.61 |

**By topology (accept; part-A / mean part-B):**

| Topology | Opus | Qwen |
|---|---|---|
| Bioreactor | 4/5 · 1.00 / 0.92 | 2/5 · 0.60 / 0.87 |
| Datacenter | 0/5 · 0.00 / 0.65 | 0/5 · 0.00 / 0.67 |
| Greenhouse | 0/5 · 0.40 / 0.66 | 0/5 · 0.20 / 0.55 |
| Clinic | 0/5 · 0.60 / 0.37 | 0/5 · 0.80 / 0.35 |

**Findings (defensible):**
- Both models do the hard *qualitative* steps (break confounds, reject the
  obvious-wrong fix) but repeatedly **stop one inferential hop short** of the full
  mechanism or the exact fix. Same shape across four distinct structures.
- **Bioreactor** (single-lever mediation chain) is the only family either model
  solves reliably.
- Common failure: **"acts right, explains wrong"** — a reasonable intervention
  (part A) with a mislabeled mechanism (part B), e.g. naming a cause/knob as the
  proxy.
- Opus ≥ Qwen overall and on bioreactor, but **directional only** — at n=5 the
  CIs overlap; the reliable signal is the per-structure failure pattern, not the
  exact rates.

---

## 7. Bugs found and fixed (the audit trail — important for trusting §6)

Three times a scary number turned out to be the harness, not the model. Standing
rule adopted: **a 0/N or a flagged failure is never read as difficulty until the
artifact check clears.** `run_batch_v6.py` auto-flags suspect results;
`_artifact_check` fires on "part-A passed but B failed", "recommended
intervention resolved to nothing", or "named proxy didn't resolve".

Grading fixes (in `oracle_v6.py` / `run_agent_v6.py`):
1. **String-match → computed grading** (the original v3/v4 → v5 change).
2. **Absolute tolerance → fraction-of-benefit (Part A).** ±2.0 utility was ~5% of
   range on wide worlds but ~35% on the narrow clinic — inconsistent. Now recover
   ≥90% of achievable benefit everywhere.
3. **Proxy credited as a SET + parenthetical stripping (Part B).** A correct
   proxy described verbosely ("LeafGreenness (interveinal chlorosis…)") was being
   dropped by the resolver; now stripped and any genuine proxy is credited.
4. **Interior-optimum sign bug.** Actuator signs were computed at the *extreme*
   dose, mislabeling a titrated fix (helps at moderate dose, hurts at max) as
   harmful. Now non-monotone actuators are `skip` (not scored).

Resolver fixes (`resolver.py`): stem matching (chelating/chelation cluster),
document-frequency distinctiveness (generic words like "speed" don't license a
match), request-coverage gate ("network link speed" ↛ "fan speed"), and the LLM
fallback for the uncertain band.

Harness fairness fixes (found via the Qwen batch, `rpg_v6_batch20_audit.md`):
5. **Output truncation.** Qwen (a thinking model) blew the old 2500-token cap →
   26 empty + 6 truncated responses (8% of turns) vs Opus 0%. Fix: per-turn
   output tokens default to the **model's max**. Re-run: parse errors 47 → 2 of
   514 turns.
6. **Turn-cap starvation.** Qwen hit the 32-turn cap on 9/20 (Opus 0/20); 4
   worlds never answered while actively solving. Fix: (a) **unproductive turns
   (empty/truncated) are retried free**, not charged; (b) **max-turns raised
   32 → 60**. Re-run: 0 turn-cap no-answers.

After all fixes, the batch20 failures were audited world-by-world and confirmed
**genuine reasoning errors**, not artifacts. The Opus↔Qwen comparison is fair
(same worlds, same grader, symmetric near-cap answering: Opus 14/20 vs Qwen 13/20
answered at turn ≥30).

---

## 8. Known limitations (carry into v7 / any writeup)

1. **n = 5 per topology.** CIs are wide and overlap; report per-structure failure
   *modes*, not exact accuracy gaps.
2. **Clinic construction mismatch.** Its ideal answer is a **conditional policy**
   (treat the responsive subgroup); the grader scores only a single dose. Clinic's
   0/5 for both models partly reflects this skill/grader mismatch — not equivalent
   evidence to datacenter's 0/5. A conditional-policy answer schema would fix it.
3. **Within a topology, worlds differ only by parameter jitter** (same graph,
   same roles). Seeds give within-structure *reliability*, not breadth. → this is
   exactly what v7's structural sampler addresses.
4. **Batch20 numbers were run at max-turns 32**, then the default was raised to 60.
   The 32-cap was symmetric (both models), so the comparison is fair, but both
   scores may be mildly suppressed; a re-run at 60 would make the cap provably
   non-binding.
5. One residual resolver miss ("Interface/network error count (IEC)",
   slash+abbrev) is verdict-neutral (that world fails part-A anyway).
6. Grading Monte-Carlo uses a fixed seed, not the world's seed → possible ±1-world
   wobble at the 90%-benefit boundary. Pin per-world for exactness.

---

## 9. File index (`dataset_generation_code/rpg_v6_prototype/`)

| File | Role |
|---|---|
| `engine.py` | SCM evaluator + mechanism library; sample()/measure()/utility |
| `worlds_v6.py` | the 4 hand-authored world families |
| `oracle_v6.py` | gold search, counterfactual battery, grader, 5 audits, calibration |
| `resolver.py` | free-text → variable/actuator mapping (+ LLM fallback) |
| `sim_v6.py` | runtime simulator; serves measure/intervene, writes raw CSVs |
| `sandbox.py` | sandboxed Python for the `code` action |
| `run_agent_v6.py` | single-world agent loop; backends, resolver wiring, token/turn logic, grading, artifact check |
| `run_batch_v6.py` | batch runner; artifact flagging; `--resume` |
| `analyze_results.py` | multi-model aggregate report (Wilson CIs, per-topology, decomposition) |
| `regrade.py` | re-score existing results with the current grader (`--fresh-oracle`) |
| `openai_llm.py` | OpenAI-compatible client + per-model presets + max-output-tokens |
| `generate_v6.py` | world generator (jitter one of the 4 templates + audit + emit) |
| `demo_v6.py` | no-LLM scripted expert walkthrough |

Data: world banks `out_v6_batch*`; results `results_v6/batch20_opus`,
`results_v6/batch20_qwen_2` (current), plus superseded `batch1/2/3`,
`batch20_qwen` (pre-fix).

**Reproduce:**
```bash
# generate a batch (or use out_v6_batch20)
python generate_v6.py --outdir out_v6_batch20 --n 20 --seed 40000
# run a model
python run_batch_v6.py --worlds-dir out_v6_batch20 --backend nautilus \
    --model qwen3-small --outdir results_v6/<name> -v
# compare
python analyze_results.py --run opus4.8=results_v6/batch20_opus \
    --run qwen3-small=results_v6/batch20_qwen_2 --out results_v6/report.md
```
