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
  configuration hashes, and the reward-independent strict evaluation function.
- `pipeline.py`: deterministic audited-world generation, immutable run manifests,
  paired seed derivation, concurrent episode execution, atomic JSONL output, canonical
  aggregation, pairing warnings, and completion/regeneration validation.
- `servers.py`: one-GPU empirical fit preflight, three independently owned vLLM
  workers, health checks, logs, scoped shutdown, and a seed-aware OpenAI-compatible
  HTTP client with transport retries.
- `storage.py`: canonical hashes, JSON/JSONL parsing, finite-number validation, and
  same-directory temporary-file plus `os.replace` atomic writes.
- `visualization/plot_results.py`: reads only `stats.json` and produces all six
  required figures with fixed score scales.
- `tests/test_prompt_compare.py`: prompt/reward injection, concurrent isolation,
  candidate formulas, v9 import resolution, deterministic seed behavior, population
  variance, terminal schema, and atomic resume tests.
- `tests/test_plot_results.py`: stats-only plotting integration test.

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
mutates the module-level `SYSTEM_PROMPT`.

## Experiment execution

The full run deterministically generates and audits 10 worlds for each of the nine
v9 archetypes, persists their `to_record`-compatible JSON plus hashes in
`manifest.json`, and only then starts inference. The same 90 files are reused by all
nine configurations.

Whole `(configuration, world)` groups are assigned round-robin to one of three vLLM
workers. In a full group, all eight complete `RPGEnv` episodes run concurrently so
vLLM can continuously batch their active turns. The stable request seed is a SHA-256
derivation of `(master_seed, world_id, prompt_id, rollout_index)`; reward id is not an
input. That same seed is sent on every request in the episode.

Smoke mode retains the complete 3 × 3 matrix and `G=8`, but generates one world per
archetype: 9 worlds and 648 episodes. It is a pipeline validation run, not a scientific
result.

The default request settings are:

- thinking enabled through `chat_template_kwargs.enable_thinking`;
- temperature `1.0`, top-p `0.95`, top-k `20`, min-p `0.0`;
- 8,192 maximum generated tokens per turn;
- 32 environment turns and a 15-experiment budget;
- bfloat16 serving, 32,768 model context, and GPU memory utilization `0.80`;
- unused image/video capacity disabled for this text-only experiment.

All settings above are fixed across configurations and included in the scientific
fingerprint. A resume with a different fingerprint is rejected.

## CLI arguments and defaults

`all` requires `--model`, the fixed `--gpus 5,6,7` topology, `--seed`, and
`--run-id`.

Operational arguments include:

- `--ports 18005,18006,18007`
- `--host 127.0.0.1`
- `--dtype bfloat16`
- `--max-model-len 32768`
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
is the empirical one-GPU fit preflight. Only then are workers two and three launched.
A failure aborts instead of changing tensor parallelism or GPU topology. Process-group
shutdown targets only workers started by the current command.

## Artifact and output schema

Each run is self-contained:

```text
runs/<run_id>/
  manifest.json
  worlds/<archetype>/world_<world_id>.json
  outputs/<archetype>/<world_id>/<config_id>/rollout_00.jsonl
  stats.json
  figures/*.png
  logs/vllm_gpu_<gpu>_port_<port>.log
  logs/rollout_errors.json
  logs/episode_data/...
```

`manifest.json` records the seed, protocol variables, resolved imports, model/chat
settings, prompt/reward hashes, expected cardinalities, world audit metadata, world
hashes, and launch topology.

Every rollout JSONL turn record contains run/world/configuration identifiers, rollout
and turn indices, request seed, exact observation, raw response content, separately
returned reasoning content when present, parsed action type, finish reason, latency,
transport-attempt count, and API usage. It contains no copied oracle or ground-truth
record.

The final JSONL object has `record_type="terminal"` and includes candidate reward,
candidate diagnostics, independently re-graded `score`, `part_a`, `part_b`, strict
acceptance, termination reason, intervention/experiment/turn counts, transcript hash,
and `complete=true`.

`stats.json` is the canonical statistics artifact. A complete full run has 810
`per_world` rows, 81 `per_archetype` rows, and 9 `overall` rows. Per-world variance is
population variance. Overall best-of-eight and within-group reward variance are means
of the 90 per-world values. Evaluation metrics always come from the fixed strict
oracle/evidence rule, never from the selected candidate reward.

## Resume and failure behavior

- JSONL is written to a same-directory temporary file, flushed/fsynced, and atomically
  replaced only after a terminal summary exists.
- `--resume` skips only files whose final record is a completed terminal summary.
  Missing, malformed, or incomplete files rerun with the same derived seed.
- HTTP/server/transport failures receive up to three retries with the identical seed.
  A successfully returned malformed model action goes directly to `RPGEnv` and is
  never retried.
- Episode failures are retained in `logs/rollout_errors.json`. Aggregation writes an
  incomplete `stats.json` containing errors and every missing path, then fails rather
  than computing partial metrics.
- Transcript hashes for `r1`/`r2`/`r3` are compared within each paired prompt/world/
  rollout tuple. Mismatches are warnings in `stats.json`; no data is discarded.
- `validate` reconstructs statistics from the saved JSONL and compares every canonical
  section to `stats.json`.

## Validation commands

From `dataset_generation_code/rpg_rl_exps/prompt_compare` in the project environment:

```bash
python -m unittest discover -v -s tests

python run_experiment.py all \
  --model Qwen/Qwen3.5-9B \
  --gpus 5,6,7 \
  --seed 7000000 \
  --run-id prompt_compare_smoke_v1 \
  --smoke-test

python run_experiment.py aggregate --run-dir runs/prompt_compare_v1
python run_experiment.py plot --run-dir runs/prompt_compare_v1
python run_experiment.py validate --run-dir runs/prompt_compare_v1
```

## Exact full launch command

Run this from `dataset_generation_code/rpg_rl_exps/prompt_compare` in an environment
that provides vLLM, NumPy, pandas, SciPy, and Matplotlib:

```bash
python run_experiment.py all \
  --model Qwen/Qwen3.5-9B \
  --gpus 5,6,7 \
  --seed 7000000 \
  --run-id prompt_compare_v1
```
