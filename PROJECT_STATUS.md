# Project Status

Written for a new collaborator joining at the RPG stage. Snapshot date: **2026-08-04**.

---

## 1. Part one: ACED-Bench (done, being written up)

### The idea

An LLM "scientist" is shown a story and a variable list, but **not** the causal
graph. It must request observational and interventional samples from a Bayesian
network simulator (budget: **10 successful data queries**) and then answer a
question about the hidden structure.

Two question families:

- **ACED-Struct** — structural questions (ancestry, marginal and conditional
  independence). World family `4_19_big`. The paper discusses a 30-world /
  180-question subset.
- **ACED-Decision** — causal decision questions across five archetypes:
  `invalid_premise`, `mediator_structure`, `safety_constrained`, `satisficing`,
  `subgroup_robust`. World family `adv_v3`.

Three agent conditions, all sharing one orchestrator interface:

| Agent | File | Shape |
|---|---|---|
| `agent` | `scientist_agent_causal.py` | `<action type="query\|answer\|give_up">` |
| `coder` | `scientist_coder_agent.py` | adds sandboxed `<action type="code">` (pandas/numpy/scipy) |
| `coder_new` | `scientist_coder_agent_new.py` | **modular**: INIT / CODE / ANALYSIS / DESIGN turns |

`coder_new` is the headline condition in the paper.

### Headline finding

Agentic querying beats zero-shot by a wide margin, and the modular `coder_new`
decomposition beats the monolithic coder — but the gap between frontier and
open-weight models is much larger on **Decision** than on **Struct**. Struct is
close to saturated for frontier models; Decision is where the benchmark still
discriminates, especially on `satisficing` and `subgroup_robust`.

The most solid single anchor: **Opus `coder_new` on ACED-Struct = 164/173 =
0.948**, consistent across every source checked.

### ⚠️ Do not trust the ACED result numbers without re-deriving them

The unrestricted Decision runs have **mixed provenance** — historical 48-row
runs, 54-row `changed24` hybrids, and 60-row reruns all coexist, and different
files disagree. Concretely, for GPT-OSS-120B `coder_new` on ACED-Decision:

- the old handoff notes claim `45/60 = 0.750`
- `evaluations/for_paper/eval_oss120_coder_new_adv_v3.json` says `32/48 = 0.667`
- `evaluations/eval_oss120_coder_new_adv_v3.json` (the newer one) says `51/60 = 0.850`

These are three different runs, not three views of one run. **Always read
`scores.overall.total` and check the row count before quoting anything.** The
structural GPT-OSS run (`oss120_coder_new_4_19_big`) never finished, and the
manuscript currently carries placeholder values for it.

Since we are not rerunning ACED, this matters only if a number gets cited. The
raw JSONs live on Vivian's machine under `framework_code/evaluations/`.

### Protocol constants (fixed across all ACED runs)

- Query budget: 10 successful data queries
- World parser: Claude Opus-4.8 via Bedrock, temp `0.1`, max output `512`
- Evidence-ledger annotator: same model, temp `0.0`, max output `600`, ≤3 retries
- Decision scoring tolerance: `0.05` expected-state-index units
- Structural dependency threshold: total variation ε = `0.02`
- Expected-state-index uses the ordered categorical state list, zero-based.
  Lower outcome index = better unless a world says otherwise.

---

## 2. Part two: RPG (active work)

### What changed and why

ACED-Bench always *hands the agent a question*. RPG removes that scaffold.

An RPG world shows a population that already exhibits a problem. The agent gets:

- a set of **intervenable knobs**
- a set of **observable measurements**
- **no menu of candidate policies**

It must decide what to measure, interpret noisy signals, form a hypothesis about
the latent cause, and propose a `do(.)` dict. The motivating example is
H. pylori: the prevailing theory (stress, acid, lifestyle) is a red herring, and
the agent has to discover that the real driver is bacterial — and that the
intervention conventional wisdom calls irrelevant is the one that works.

Current archetype: **`story_hidden_cause_discovery`**. Current schema:
**`rpg_static_v3`**.

### Version history

- **v1** — dynamic, time-based, fixed policy menus. Superseded; notes retained in
  `docs/rpg/world_gen_rpg_agent_pipeline_notes.md` §A.
- **v2** — static, partially observed, no policy menu, agent submits `do(.)`.
  Design in `docs/rpg/worldgen_rpg_plan_static_partial_observation.md`.
- **v3** — `story_hidden_cause_discovery` archetype; hides the latent cause in
  the *story*.
- **v4** — hardening pass. Design in
  `docs/rpg/worldgen_rpg_plan_v4_complex_neutral_dose.md`.

### The v3 → v4 story (this is the important part)

The v3 pilot **passed 2/2 with Opus** — but that was a bad sign, not a good one.
It was too legible and too uniform, for two reasons:

1. **Name leakage.** The design hides the latent cause in the story, but the
   *action catalog gave it away*: names like `ClearRearGutters`,
   `FlushDownspout`, and proxies like `DownspoutDischargeDelay` let a strong
   model read the answer off the labels and solve it in one intervention.
2. **All interventions were binary** (`off`/`on`). The only choice was which
   knob to flip — no dose, no setpoint, no genuine "right amount" decision.

v4 addresses both: neutral, non-leaking names and non-binary/dose actions. It
deliberately does **not** add a new archetype or change the role contract, so
`simulator_rpg.py`, `world_model_rpg.py`, `evaluate_rpg.py`, and `audit_rpg.py`
keep working with at most a one-line change for continuous knobs.

### Current RPG results (`results_rpg/`, Opus)

| Run | Worlds | Accepted | Accuracy | Avg queries |
|---|---|---|---|---|
| v3 story-hidden pilot | 2 | 2 | **1.00** | 3.5 |
| **v4 LLM-templated** | 8 | 0 | **0.00** | 3.6 |
| **v4 mixed** | 6 | 4 | **0.67** | 3.2 |

v4 failure buckets:

- v4 LLM: `latent_cause_missing_or_surface_proxy` ×6, `latent_cause_no_decisive_test` ×1, `latent_cause_weak_evidence` ×1
- v4 mixed: `latent_cause_missing_or_surface_proxy` ×2

**Read this carefully before acting on it.** v4 hardening worked — arguably too
well on the LLM-templated set. Opus went from 2/2 to 0/8. The dominant failure is
`latent_cause_missing_or_surface_proxy`: the agent settles on a surface proxy
instead of the true latent cause. Two readings, and we have not yet distinguished
them:

- **Good reading** — we removed the name-matching shortcut, and the task is now
  genuinely hard, which is what we wanted.
- **Bad reading** — the LLM-templated worlds are *unsolvable* or the grader is
  too strict about what counts as identifying the latent cause. `avg_queries` is
  only ~3.6 against a 12-turn budget, which is suspicious: the agent is stopping
  early rather than exhausting its budget. That is more consistent with the agent
  believing it is done than with it fighting a hard problem.

Note also the sample sizes are tiny (8 and 6 worlds). None of these numbers are
statistically meaningful yet.

### Suggested next steps

1. **Disambiguate the 0/8.** Hand-solve two or three v4 LLM worlds. If a careful
   human with the same query interface cannot recover the latent cause, the
   worlds are broken, not hard.
2. **Check the grader separately from the generator.** Re-score the v4 traces by
   hand to see whether "surface proxy" answers are ever defensibly correct.
3. **Explain the LLM vs mixed gap** (0/8 vs 4/6). These differ in how templates
   were produced; that is the cleanest available lever on difficulty.
4. **Investigate early stopping.** ~3.6 of 12 turns used suggests the agent is
   not being pushed to test its hypothesis.
5. **Scale up** only once 1–4 are settled — current n is far too small.

---

## 3. Known-stale references (cleaned up in this branch)

The previous handoff notes pointed at several things that do not exist:

- `codebase_guideline.md`, `submission_notes.md`, `notebook_instructions.md` — never found in the repo
- `framework_code/run_evidence_ledger_current_parallel.sh` — does not exist anywhere
- `evaluations/round/` and `evaluations/samp/` — the resource-variant evals are under `framework_code/logs/` instead
- `paper_polished.tex` cites `Acharyaetal25` but **no `references.bib` exists** in the repo; the bibliography source must be found before the paper will compile

Terminology: current names are **ACED-Bench**, **ACED-Struct**, **ACED-Decision**.
Older prose terms (`PGM-Struct`, `PGM-Decision`, `Basic`, `Advanced`,
`guess-shot`) are retired, though they still appear in some filenames and
variable names — acceptable as paths, not as prose.
