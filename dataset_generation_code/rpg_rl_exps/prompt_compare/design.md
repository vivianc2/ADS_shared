# Design: System Prompt and Reward Function Comparison

## 0. Prerequisite
Our project: The codebase is under /rpg. Review the codebase carefully.

## 1. Objective

Measure Qwen3.5-9B on the same RPG worlds under a 3 × 3 matrix of:

- three system prompts (`p1`, `p2`, `p3`); and
- three terminal reward functions (`r1`, `r2`, `r3`).

Each prompt runs eight independent rollouts on each world, matching GRPO group size
`G=8`. After each rollout terminates, all three reward functions score the same
terminal answer. The nine prompt/reward configuration IDs are analytical views over
those shared prompt rollouts, not nine separate inference conditions. The experiment
compares prompt-driven task performance and reward-function behavior without training
the model.

Important interpretation: the reward is computed only after an episode ends, so it
cannot change model behavior in this experiment. Evaluation metrics compare system
prompts; reward metrics compare how the three reward functions score terminal
answers.

## 2. Scope and access rules

Implementation may create or modify files only under:

`dataset_generation_code/rpg_rl_exps/prompt_compare/`

The only permitted edit outside that directory is:

`dataset_generation_code/rpg_rl/env.py`

Do not edit any other existing file. Existing modules in `rpg_rl` and `rpg_v9` may be
imported. Use repository-relative path discovery; do not hard-code a machine-specific
repository root.

## 3. Experiment matrix

Use all nine archetypes exposed by `rpg_v9/sampler.py`:

1. `confounded_chain`
2. `collider_selection`
3. `hidden_subtype`
4. `surrogate_trap`
5. `instrument_only`
6. `competing_causes`
7. `synergy_pair`
8. `dose_window`
9. `confounded_reversal`

Generate exactly 10 accepted, audited worlds per archetype: 90 unique worlds total.
Generate them once and reuse them for every configuration.

Configuration IDs are the Cartesian product:

`p1_r1`, `p1_r2`, `p1_r3`, `p2_r1`, ..., `p3_r3`.

Each prompt runs 90 worlds × 8 rollouts = 720 episodes. The complete run has 2,160
model episodes and 6,480 terminal reward evaluations (2,160 terminal answers × three
reward functions). A rollout is a full multi-turn `RPGEnv` episode, not a single
model completion.

## 4. Reproducibility and fairness

- Accept one master seed at the CLI and record it in the run manifest.
- Resolve imports explicitly from `dataset_generation_code/rpg_rl` followed by
  `dataset_generation_code/rpg_v9`. Set and record `RPG_PROTO=rpg_v9`; do not allow
  Python to resolve `sampler`, `engine`, or `oracle_v6` from an older RPG directory.
- Set and record `RPG_SYNERGY_SOFT=20` before importing or generating worlds. This is
  part of the world-generation specification because it changes `synergy_pair`
  topology even when the seed is unchanged.
- Generate and audit worlds deterministically using `rpg_v9.sample_world`,
  `rpg_v9.generate_v7.audit`, and `to_record`-compatible serialization.
- Persist accepted worlds before model inference starts.
- Keep model, chat template, thinking mode, sampling parameters, token limits,
  environment budget, and maximum turns fixed across configurations.
- Derive a stable rollout seed from `(master_seed, world_id, prompt_id,
  rollout_index)`. Reward ID is absent because reward functions are applied post hoc
  to the same terminal answer.
- Pass the seed on every model request and record the resolved sampling settings.
- Match the canonical POPE rollout settings: thinking on, temperature `1.0`, top-p
  `1.0`, top-k `-1` (disabled), and at most 8,192 generated tokens per assistant turn.
  Record all of these settings in the scientific fingerprint. The generation cap
  includes model thinking when the server returns thinking in ordinary assistant
  content.
- Before every model request, render the complete chat with the same model tokenizer,
  generation prompt, and thinking setting used by vLLM. If that prompt exceeds 18,432
  tokens, do not send it: record a transparent zero-reward soft terminal with
  `finish_reason: length` and `termination_reason: input_length`. This matches SkyRL's
  `generator.max_input_length=18432` behavior and leaves 6,144 tokens of headroom under
  `max_model_len=32768` even when an allowed prompt uses the full 8,192-token generation
  budget.
- Maintain a complete, episode-local OpenAI message history: system prompt, initial
  catalog-bearing user observation, and every subsequent assistant response and user
  observation. Send that accumulated history on every turn. Use the exact assistant
  text passed to `RPGEnv.step` as the assistant history message.
- Never mutate a module-level prompt while concurrent rollouts are active.
- Never share a message-history object between concurrent rollouts.
- Treat a vLLM context-length HTTP 400 as non-retryable. It is a last-resort fallback
  for tokenizer/server disagreement after the client-side 18,432-token check. Convert
  it into a transparent zero-reward terminal with `termination_reason: context_limit`,
  retain the server error in the JSONL, and count it in `stats.json`; do not let one
  oversized history make the run permanently incomplete.

There is exactly one raw output for each `(prompt, world, rollout_index)`. All reward
variants therefore share the transcript by construction rather than relying on
backend determinism to reproduce it.

## 5. Minimal environment change

Make `rpg_rl/env.py` support per-instance configuration while preserving existing
callers:

- Add `system_prompt: str = SYSTEM_PROMPT` to `RPGEnv`.
- Add an injectable terminal `reward_fn`, defaulting to the existing
  `compute_reward`.
- In `_terminal`, call the instance reward function with the same inputs currently
  passed to `compute_reward`.

The experiment runner must use `env.system_prompt` when calling the model. Do not
change the global `SYSTEM_PROMPT` between jobs. Because the shared `rollout.py` cannot
be edited, implement the small rollout loop inside `prompt_compare`. `RPGEnv` may use
`r1` for the terminal scalar required by its step API, but the runner must explicitly
apply `r1`, `r2`, and `r3` to the same terminal answer after the episode ends. The
post-hoc `r1` value must agree with the environment return.

Treat `budget` as a hard upper bound on simulator experiments. Once `_used == budget`,
later `measure` or `intervene` actions must not call the simulator or increment
`_used`; return a recoverable observation directing the model to answer from existing
evidence. Free `code` turns and terminal `answer`/`give_up` actions remain available.

Each candidate reward function must accept the existing `compute_reward` inputs and
return a dictionary containing at least a finite numeric `reward`. Candidate reward
functions live inside `prompt_compare`; do not modify `rpg_rl/reward.py`.

## 6. Evaluation definitions

Keep evaluation independent of the selected reward function. Re-grade every terminal
answer with the existing strict oracle and the current evidence rule:

- `part_a`: clipped recovered utility in `[0, 1]`;
- `part_b`: strict counterfactual-battery fraction in `[0, 1]`;
- if no intervention was applied, both parts are zero; and
- `score = 0.5 * part_a + 0.5 * part_b`.

For each `(configuration, world)` analytical view, select the named reward from the
eight shared prompt rollouts and calculate:

- `reward_mean = mean(reward)`;
- `reward_variance = mean((reward - reward_mean)^2)` (population variance, `ddof=0`);
- `avg_score = mean(score)`;
- `best_of_8_score = max(score)`;
- `avg_part_a = mean(part_a)`; and
- `avg_part_b = mean(part_b)`.

For each `(configuration, archetype)`, calculate `avg_score`, `avg_part_a`, and
`avg_part_b` across its 80 rollouts. Also calculate the same overall metrics across
all 720 rollouts in a configuration. Overall `best_of_8_score` means the average of
the 90 per-world best-of-eight scores. Overall within-group reward variance means the
average of the 90 per-world reward variances, not variance across all 720 rollouts.

Do not substitute candidate reward for evaluation score. Store full-precision values
in statistics and round only for display.

## 7. Storage layout

Use one self-contained run directory:

```text
prompt_compare/
  design.md
  implementation_detail.md
  run_experiment.py
  candidates.py
  ...
  visualization/
    plot_results.py
  runs/<run_id>/
    manifest.json
    worlds/
      <archetype>/world_<world_id>.json
    outputs/
      <archetype>/<world_id>/<prompt_id>/rollout_00.jsonl
      <archetype>/<world_id>/<prompt_id>/rollout_01.jsonl
      ...
      <archetype>/<world_id>/<prompt_id>/rollout_07.jsonl
    stats.json
    figures/
    logs/
```

Worlds and model outputs are separate. Do not copy oracle/ground-truth data into
model prompts or raw response records.

Each rollout JSONL contains one record per turn with identifiers, request seed,
observation, raw model response, parsed action type, timing, rendered prompt-token
count, request-message count, and a hash of the exact accumulated message list sent
to vLLM. The turn records must
be sufficient to reconstruct and verify every request history without duplicating the
full history in every JSONL row. Its final record is a terminal summary containing the
`candidate_rewards` map (`r1`, `r2`, and `r3`), fixed evaluation score, part A, part B,
termination reason, optional context-limit error, intervention count, and
`complete: true`.

Store the three candidate reward values even though they can be derived at rollout
time. They are the terminal measurements consumed by aggregation, and keeping them
makes resume, statistics regeneration, and audit independent of rerunning oracle code
or of later changes to reward implementations. Prompt and reward source hashes in the
manifest bind the stored measurements to the exact definitions used.

Write rollout files atomically. On `--resume`, skip only files whose final record has
`complete: true` and whose entire JSONL passes the same per-file validator used by
aggregation. That validator checks record/turn counts, transcript hash, the complete
request-history hash chain, prompt-token bounds, synthetic-action/termination
consistency, identifiers, sampling settings, and environment limits. Incomplete or
invalid completed files are rerun with the same seed.

`stats.json` is the sole canonical statistics file. It contains:

- run metadata and configuration definitions/hashes;
- `per_world`: 810 records (9 configurations × 90 worlds);
- `per_archetype`: 81 records (9 configurations × 9 archetypes);
- `overall`: 9 records; and
- error counts, post-hoc reward-application provenance, and completeness checks.

Retryable transport/server failures may be retried up to three times with the same
seed. Non-retryable HTTP 4xx responses must not be retried with an unchanged payload;
context-length 400 responses use the recorded soft-terminal behavior above.
Malformed model actions are valid model behavior and must not be retried. Do not
silently compute final statistics from fewer than eight completed rollouts; fail the
aggregation and list the missing paths.

## 8. Execution and GPU use

Provide one CLI entrypoint with an `all` command that performs world generation,
server startup, rollouts, aggregation, validation, and plotting:

```bash
python run_experiment.py all \
  --model Qwen/Qwen3.5-9B \
  --gpus 5,6,7 \
  --seed 7000000 \
  --run-id prompt_compare_v3
```

Use three independent vLLM OpenAI-compatible workers, one on each of exactly three
unique GPU IDs supplied by the user, after a preflight confirms the configured model
fits on one GPU. The canonical launch uses GPUs 5, 6, and 7, but hosts numbered 0, 1,
and 2 must be supported with `--gpus 0,1,2`. Assign complete world/prompt groups to
workers and submit the eight trajectories concurrently so vLLM can batch active
turns. Keep ports configurable, wait for health checks, save server logs, and stop
only servers started by this command. If the one-GPU fit preflight fails, stop with a
clear error instead of silently changing tensor parallelism.

The entrypoint must support `--resume` and a small `--smoke-test` mode. It must print
the run directory and final completeness summary.

## 9. Visualization

All plotting code must live in `prompt_compare/visualization/`. It reads only
`stats.json` and writes figures to the run's `figures/` directory.

Produce at least:

- a prompt-level score sheet with average score, part A, part B, and average
  per-world best-of-eight score for `p1`, `p2`, and `p3`;
- a prompt-level reward sheet with reward mean and within-group variance for each
  of `r1`, `r2`, and `r3`; and
- archetype-by-prompt heatmaps for average score, part A, and part B, with exactly
  `p1`, `p2`, and `p3` on the x-axis because evaluation scores are invariant to
  post-hoc reward selection.

Write both sheets as rendered PNG tables and CSV files.

Use fixed axes and color ranges across comparable plots.

## 10. Validation and completion criteria

Before the full run, tests and smoke mode must verify prompt/reward injection,
concurrent isolation, complete multi-turn message history, deterministic seed
derivation, metric calculations, atomic resume behavior, and JSON schema
completeness.

The run is complete only when:

- there are 90 unique audited worlds, exactly 10 per archetype;
- all nine configuration IDs are present;
- every prompt/world group has exactly eight completed rollouts (2,160 episodes);
- each terminal has all three candidate rewards (6,480 reward evaluations);
- all 810 per-world and 81 per-archetype statistics records exist;
- every stored metric is finite and within its expected range;
- every terminal has `experiment_count <= budget`;
- `stats.json` can be regenerated from saved rollout files; and
- every required figure is generated from `stats.json`.

## 11. Required implementation handoff

CodeX must add `implementation_detail.md` inside `prompt_compare`. It should list the
implemented files, environment change, dependencies, CLI arguments/defaults, output
schema, resume/failure behavior, validation commands, and the exact single launch
command for GPUs 5, 6, and 7.

## Appendix A — Candidate System Prompts

- `p1`: CURRENT SYSTEM PROMPT
- `p2`: 
```
You are a scientist diagnosing a failing industrial system. You interact ONLY through the catalog of ids given each turn — every measurement, control, and answer refers to those ids (m0, m1, ... for measurable signals; a0, a1, ... for controls). You do NOT know which signal or control matters; you must find out from data by measuring and (crucially) by intervening.

Each turn, output exactly:
<reasoning>your scientific thinking</reasoning>
<action type="measure|intervene|code|answer|give_up">JSON</action>
<memory>notes to carry forward</memory>

Action payloads (JSON, ids only):
- measure:   {"ids": ["m3","m0"]}                         # read those signals
- intervene: {"actions":[{"actuator":"a2","value":66}], "measure":["m3"]}   # set controls, then read
- code:      raw Python analysis; does not cost budget. Each experiment's data is preloaded as a pandas DataFrame named experiment_<n>_df (use it directly, e.g. experiment_1_df.describe()); its file path is also available as experiment_<n>_csv. pandas as pd, numpy as np, scipy.stats as stats are imported. Each code turn runs in a FRESH namespace — the experiment_<n>_df variables are always available, but variables you define do NOT persist to the next code turn, so recompute what you need.
- answer:    {"actions":[{"actuator":"a2","value":66}],
              "policy":{"treatment":"a2","stratifier":"m1","threshold":50,"dose_if_ge":100,"dose_if_lt":0},
              "proxy":"m3", "decoys":["m0"], "signs":{"a2":"+"}}
- give_up:   {}

Observation and code alone CANNOT establish causation — you must INTERVENE to test a cause and find what improves the outcome. Submit "answer" once you know the fix AND the mechanism (which signal is the true proxy, which are decoys, and each control's effect sign +/-/0). For a world where a treatment helps only a sub-population, use "policy" to stratify on the marker signal.

Do not trust a control just because the outcome reading went up — verify it actually changed the CAUSE (the mechanism proxy moves). A control that only lifts the reading without changing the cause has sign 0, and recommending it as the fix is wrong.
```

- `p3`:
```
You are a scientist diagnosing a failing industrial system. You interact ONLY through the catalog of ids given each turn — every measurement, control, and answer refers to those ids (m0, m1, ... for measurable signals; a0, a1, ... for controls). You do NOT know which signal or control matters; you must find out from data by measuring and (crucially) by intervening.

Each turn, output exactly:
<reasoning>your scientific thinking</reasoning>
<action type="measure|intervene|code|answer|give_up">JSON</action>
<memory>notes to carry forward</memory>

Action payloads (JSON, ids only):
- measure:   {"ids": ["m3","m0"]}                         # read those signals
- intervene: {"actions":[{"actuator":"a2","value":66}], "measure":["m3"]}   # set controls, then read
- code:      raw Python analysis; does not cost budget. Each experiment's data is preloaded as a pandas DataFrame named experiment_<n>_df (use it directly, e.g. experiment_1_df.describe()); its file path is also available as experiment_<n>_csv. pandas as pd, numpy as np, scipy.stats as stats are imported. Each code turn runs in a FRESH namespace — the experiment_<n>_df variables are always available, but variables you define do NOT persist to the next code turn, so recompute what you need.
- answer:    {"actions":[{"actuator":"a2","value":66}],
              "policy":{"treatment":"a2","stratifier":"m1","threshold":50,"dose_if_ge":100,"dose_if_lt":0},
              "proxy":"m3", "decoys":["m0"], "signs":{"a2":"+"}}
- give_up:   {}

Observation and code alone CANNOT establish causation — you must INTERVENE to test a cause and find what improves the outcome. Submit "answer" once you know the fix AND the mechanism (which signal is the true proxy, which are decoys, and each control's effect sign +/-/0). For a world where a treatment helps only a sub-population, use "policy" to stratify on the marker signal.
```


## Appendix B — Candidate Reward Functions
- `r1`: CURRENT REWARD FUNCTIONS
- `r2`: `part_a - 0.25 * invalid_id_fraction`
- `r3`: `part_b - 0.25 * invalid_id_fraction`
