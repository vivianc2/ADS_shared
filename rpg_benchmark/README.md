# Causal discovery benchmark

An agent diagnoses a failing system it has never seen. Each world is a hidden structural causal model with ~25
adjustable controls and ~26 measurable signals, described in a plain-language situation report. The report names
every measurable signal (with its unit) and every control (with its usual level or dose and its range); the hidden
variables are not named. The agent has 15 experiments (measure, or intervene on any combination of controls) plus
free analysis code, and must find what restores the true underlying objective and explain the mechanism. There are
no ids or action menus: the agent writes requests in its own words, and a resolver model maps each request to the
world.

**Version:** harness v10.1 (see `CHANGELOG.md`). Every result file records `harness_version`. The default agent
prompt is the built-in v10 prompt (`prompts/agent_prompt_v10.txt`).

## Data

| split | worlds | |
|---|---|---|
| `data/test` | 540 | the benchmark: 10 domains, 60 worlds of each of 9 causal structures |
| `data/validation` | 270 | optional, for development: 30 worlds of each structure |
| `data/core108` | 108 | subset of `test`: 12 worlds of each structure, every (structure, domain) pair at least once |
| `data/core54` | 54 | subset of `core108`: 6 worlds of each structure |
| `data/core54_anon` | 54 | `core54` with neutral variable names (ablation) |

`index.json` in each split lists the domain and causal structure of every file. A world
file holds the situation report, the causal model, and the precomputed best intervention used for grading.
The core subsets hold byte-identical copies of `data/test` files; `python data/build_core_subsets.py` rebuilds
`core108` and checks both subsets against their selection rule (described in that script).

## Run

```
pip install -r requirements.txt

# smoke test, no API (a scripted agent that answers with the gold intervention)
python run_benchmark.py --worlds-dir data/validation --backend mock --outdir out/mock

# a model on AWS Bedrock (needs AWS credentials or AWS_BEARER_TOKEN_BEDROCK)
python run_benchmark.py --worlds-dir data/test --backend bedrock --model <bedrock model id> --outdir out/<name> \
    --concurrency 8 --resume

# an OpenAI-compatible endpoint (OPENAI_API_KEY / OPENAI_BASE_URL, or VLLM_BASE_URL for --backend vllm)
python run_benchmark.py --worlds-dir data/test --backend openai --model <model> --outdir out/<name>
```

Other options: `--budget` (15), `--max-turns` (32), `--max-new-tokens` (16000), `--resolver-model`,
`--system-prompt-file`. One world: `python agent_loop.py --world-file <file> --backend ... --out <result.json>`.

The resolver uses a fixed Bedrock model (`--resolver-model`) when Bedrock credentials are available, otherwise the
agent's own model.

World files contain the answers. The agent's analysis code runs in a separate process but is not isolated, so run
the benchmark where that code cannot read `data/` (for example in a container).

## Output and scores

`<outdir>/result_<world>.json` holds the full trajectory (every turn, what the agent saw, resolutions, code output)
and the grade; `<outdir>/summary.json` aggregates. Per world:

- `benefit_recovered`: improvement of the true objective achieved by the recommended intervention, as a fraction of
  the best achievable (1.0 = optimal, 0 = no improvement).
- `part_a_utility_ok`: benefit_recovered >= 0.9.
- `battery_fraction`: the causal account -- mechanism marker, confounded decoys, and the effect sign of every control
  that acts on the hidden cause. `part_b_battery_ok`: >= 0.8.
- `accepted`: part A and part B.
