# Prompt/Reward Comparison Implementation

## Implemented files

- `run_experiment.py`: the single CLI entrypoint. Its `all` command generates worlds,
  starts and health-checks vLLM, runs rollouts, aggregates, validates, plots, and prints
  the run directory plus completeness summary. It also exposes `aggregate`, `plot`, and
  `validate` maintenance commands.
- `bootstrap.py`: discovers `dataset_generation_code` relative to this file, sets
  `RPG_PROTO=rpg_v9` and `RPG_SYNERGY_SOFT=20`, puts `rpg_rl` before `rpg_v9` on
  `sys.path`, and asserts that collision-prone imports came from `rpg_v9`.
- `candidates.py`: exact `p1`/`p2`/`p3` prompts, `r1`/`r2`/`r3` terminal rewards,
  prompt/configuration hashes, explicit post-hoc application of every reward, and the
  reward-independent strict evaluation function.
- `pipeline.py`: deterministic audited-world generation, immutable run manifests,
  prompt-level seed derivation, isolated full-history episode execution, atomic JSONL
  output, canonical post-hoc reward aggregation, and completion/regeneration
  validation.
- `servers.py`: one-GPU empirical fit preflight, one independently owned vLLM worker
  per configured GPU, health checks, logs, scoped shutdown, and a seed-aware
  full-message OpenAI-compatible HTTP client with transport retries.
- `storage.py`: canonical hashes, JSON/JSONL parsing, finite-number validation, and
  same-directory temporary-file plus `os.replace` atomic writes.
- `visualization/plot_results.py`: reads only `stats.json` and produces two
  prompt-level summary sheets (PNG and CSV) plus three prompt-only archetype
  heatmaps with fixed score scales.
- `tests/test_prompt_compare.py`: prompt/reward injection, concurrent isolation,
  complete two-turn chat history, candidate formulas, v9 import resolution,
  deterministic seed behavior, population variance, terminal schema, and atomic
  resume tests.
- `tests/test_plot_results.py`: stats-only plotting integration test.
- `tests/test_topology.py`: three-unique-GPU/port validation, server startup,
  and complete three-worker group scheduling tests.

## Shared environment change

`dataset_generation_code/rpg_rl/env.py` now has two backward-compatible dataclass
fields:

```python
system_prompt: str = SYSTEM_PROMPT
reward_fn: Callable[..., Dict[str, Any]] = compute_reward
```

`RPGEnv._terminal` calls `self.reward_fn` with the same arguments formerly passed to
`compute_reward`. Existing callers that supply neither field retain the original
prompt and reward. The experiment calls the model with `env.system_prompt`; it never
mutates the module-level `SYSTEM_PROMPT`. `RPGEnv.step` also enforces `budget` before
calling the simulator: once exhausted, later measure/intervene requests are no-ops
that do not increment `_used`, while code and terminal actions remain usable.

## Experiment execution

The full run deterministically generates and audits 10 worlds for each of the nine
v9 archetypes, persists their `to_record`-compatible JSON plus hashes in
`manifest.json`, and only then starts inference. The same 90 files are reused by all
nine configurations.

Whole `(prompt, world)` groups are assigned round-robin across the configured vLLM
workers. Each GPU owns one full model replica. In a full group, all eight complete
`RPGEnv` episodes run concurrently so vLLM can continuously batch their active turns.
One GPU processes one group at a time; additional GPUs process additional groups in
parallel. The stable request seed is a SHA-256 derivation of `(master_seed, world_id,
prompt_id, rollout_index)`. That same seed is sent on every request in the episode.

Every episode owns an independent OpenAI message list. Turn one sends the system
prompt and catalog-bearing initial observation. Each later turn sends that entire
history plus the preceding assistant response and new environment observation. The
assistant history content is exactly the text passed to `RPGEnv.step`, including the
reasoning wrapper synthesized when vLLM returns reasoning separately. No history
object is shared between the eight concurrently executing trajectories.

`RPGEnv` uses `r1` only to satisfy its terminal `step()` return contract. Once the
episode is terminal, the runner explicitly calls all three candidate reward functions
on the same answer and verifies that post-hoc `r1` equals the environment return. This
produces 2,160 model episodes and 6,480 reward evaluations in a full run. The nine
prompt/reward configurations are statistics views over shared prompt transcripts.

Smoke mode retains the complete 3 × 3 analysis matrix and `G=8`, but generates one
world per archetype: 9 worlds, 216 model episodes, and 648 reward evaluations. It is a
pipeline validation run, not a scientific result.

The default request settings are:

- thinking enabled through `chat_template_kwargs.enable_thinking`, matching the
  Qwen3.5 tokenizer default used by POPE;
- temperature `1.0`, top-p `1.0`, top-k `-1` (disabled), min-p `0.0`;
- 8,192 maximum generated tokens per turn, matching POPE's
  `generator.sampling_params.max_generate_length=8192` override;
- an 18,432-token maximum rendered input prompt, checked client-side before every
  request with the model chat template and the same thinking setting;
- 32 environment turns and a 15-experiment budget;
- bfloat16 serving, 32,768 model context, and GPU memory utilization `0.80`;
- unused image/video capacity disabled for this text-only experiment.

All settings above are fixed across configurations and included in the scientific
fingerprint. A resume with a different fingerprint is rejected.

## CLI arguments and defaults

`all` requires `--model`, exactly three unique physical GPU IDs in `--gpus`, `--seed`,
and `--run-id`. For example, both `--gpus 0,1,2` and `--gpus 5,6,7` are valid; one- and
two-GPU launch topologies and duplicate IDs are rejected. `nvidia-smi` inventory
validation then confirms that every requested ID exists on the current host.

Operational arguments include:

- `--ports 18005,18006,18007`: exactly three unique ports
- `--host 127.0.0.1`
- `--dtype bfloat16`
- `--max-model-len 32768`
- `--max-input-tokens 18432`
- `--max-new-tokens 8192`
- `--gpu-memory-utilization 0.80`
- `--health-timeout 1800`
- `--request-timeout 900`
- `--transport-retries 3` (three retries after the initial attempt)
- `--vllm-executable vllm`
- `--[no-]thinking` (default enabled)
- `--[no-]disable-multimodal` (default enabled)
- `--resume`
- `--smoke-test`
- `--runs-root`, which defaults to this directory's `runs/` via relative discovery.

The first vLLM worker is launched on one configured GPU and must become healthy. This
is the empirical one-GPU fit preflight. Only then are any additional workers launched.
A failure aborts instead of changing tensor parallelism or GPU topology. Process-group
shutdown targets only workers started by the current command. Ports and captured GPU
inventory are operational metadata; the scientific fingerprint covers settings that
affect worlds, prompts, sampling, and evaluation.

The worker command targets the installed vLLM 0.23 CLI. It intentionally does not pass
the removed `--disable-log-requests` option; request logging is already off by default
in that version.

## Artifact and output schema

Each run is self-contained:

```text
runs/<run_id>/
  manifest.json
  worlds/<archetype>/world_<world_id>.json
  outputs/<archetype>/<world_id>/<prompt_id>/rollout_00.jsonl
  stats.json
  figures/*.png
  figures/*.csv
  logs/vllm_gpu_<gpu>_port_<port>.log
  logs/rollout_errors.json
  logs/episode_data/...
```

`manifest.json` records the seed, protocol variables, resolved imports, model/chat
settings, full system-prompt text and hashes, conversation-history mode,
prompt/reward hashes, expected cardinalities, world audit metadata, world hashes, and
launch topology.

Every rollout JSONL turn record contains run/world/prompt identifiers, rollout and
turn indices, request seed, exact observation, rendered request prompt-token count,
raw response content, separately
returned reasoning content when present, parsed action type, finish reason, latency,
transport-attempt count, API usage, request-message count, and a SHA-256 hash of the
exact message list sent to vLLM. It contains no copied oracle or ground-truth record.
Validation reconstructs the accumulated system/user/assistant history from the JSONL
and rejects any message-count or hash mismatch. It also records whether an action was
synthetically supplied after a context-limit rejection and preserves the server error.

The final JSONL object has `record_type="terminal"` and includes a
`candidate_rewards` map with `r1`, `r2`, and `r3`, candidate diagnostics,
independently re-graded `score`, `part_a`, `part_b`, strict acceptance, termination
reason, intervention/experiment/turn counts, transcript hash, and `complete=true`.
If the rendered prompt exceeds 18,432 tokens, no HTTP request is made; a transparent
synthetic turn records `finish_reason="length"`, zero transport attempts, the measured
token count, and terminal `termination_reason="input_length"`. If vLLM nevertheless
rejects a request for context length, the last-resort terminal instead has
`termination_reason="context_limit"`. Both paths preserve their detail in
`context_limit_error` and receive zero candidate/evaluation credit rather than leaving
the rollout file incomplete.
The small reward map is stored so resume and exact `stats.json` regeneration do not
rerun the oracle or silently adopt later reward-code changes; manifest source hashes
identify the definitions that produced it.

`stats.json` is the canonical statistics artifact. A complete full run has 810
`per_world` rows, 81 `per_archetype` rows, and 9 `overall` rows. Per-world variance is
population variance. Overall best-of-eight and within-group reward variance are means
of the 90 per-world values. Evaluation metrics always come from the fixed strict
oracle/evidence rule, never from the selected candidate reward.

## Resume and failure behavior

- JSONL is written to a same-directory temporary file, flushed/fsynced, and atomically
  replaced only after a terminal summary exists.
- `--resume` skips only files whose final record is a completed terminal summary.
  Resume and aggregation call the same full-record validator over the same one-time
  JSONL read. It verifies turn count, transcript hash, every accumulated request hash,
  prompt-token limits, synthetic length-terminal consistency, run identity, sampling,
  and environment limits. Missing, malformed, incomplete, or invalid completed files
  rerun with the same derived seed.
- A rendered prompt over 18,432 tokens stops client-side before transport and becomes
  a recorded zero-reward `input_length` terminal, matching SkyRL's soft length stop.
- Retryable HTTP/server/transport failures receive up to three retries with the
  identical seed. Non-retryable HTTP 4xx responses are attempted once. A
  context-length HTTP 400 becomes a recorded synthetic `give_up` terminal with zero
  reward, so it cannot deterministically break aggregation or resume. A successfully
  returned malformed model action goes directly to `RPGEnv` and is never retried.
- Episode failures are retained in `logs/rollout_errors.json`. Aggregation writes an
  incomplete `stats.json` containing errors and every missing path, then fails rather
  than computing partial metrics.
- `stats.json.errors.input_length_terminations` and
  `server_context_length_terminations` separate normal client-side input stops from
  last-resort server context rejections; `context_length_terminations` is their total.
- All three rewards are read from each single terminal record, so they share a
  transcript by construction. Completeness requires every terminal reward key.
- `validate` reconstructs statistics from the saved JSONL and compares every canonical
  section to `stats.json`.

## Validation commands

From `dataset_generation_code/rpg_rl_exps/prompt_compare` in the project environment:

```bash
python -m unittest discover -v -s tests

python run_experiment.py all \
  --model Qwen/Qwen3.5-9B \
  --gpus 0,1,2 \
  --seed 7000000 \
  --run-id prompt_compare_smoke_v3 \
  --smoke-test

python run_experiment.py aggregate --run-dir runs/prompt_compare_v3
python run_experiment.py plot --run-dir runs/prompt_compare_v3
python run_experiment.py validate --run-dir runs/prompt_compare_v3
```

## Exact full launch command

Run this from `dataset_generation_code/rpg_rl_exps/prompt_compare` in an environment
that provides vLLM, Transformers, NumPy, pandas, SciPy, and Matplotlib:

```bash
python run_experiment.py all \
  --model Qwen/Qwen3.5-9B \
  --gpus 5,6,7 \
  --seed 7000000 \
  --run-id prompt_compare_v3
```

This fixed-count experiment topology starts one full model replica on each of the
three requested GPUs. It does not use tensor parallelism across the cards.
