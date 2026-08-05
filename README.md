# ACED — Agentic Causal Evaluation & Discovery

A benchmark and agent framework where an LLM "scientist" must infer hidden causal
structure by *actively querying* a simulator, rather than answering from a story
alone.

There are two generations of work in here:

| | **ACED-Bench** (part 1) | **RPG** (part 2) |
|---|---|---|
| Status | Complete; written up in `docs/aced/paper_polished.tex` | **Active development** |
| World format | Bayesian network, explicit question per world | Static partially-observed world, one open discovery task |
| Agent's job | Answer a posed structural / decision question | Decide what to measure, form a hypothesis, propose `do(.)` |
| Answer space | Fixed (Yes/No, argmin, ranking, adjustment set) | Open — no menu of candidate policies |
| Code prefix | `*_causal.py`, `*_coder_agent*.py` | `*_rpg.py` |

**If you are joining now, the work is on RPG.** ACED-Bench is included because RPG
reuses its simulator concepts, agent loop shape, and evaluation harness — read it
as background, not as an active task. Start with
[PROJECT_STATUS.md](PROJECT_STATUS.md).

---

## Setup

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
```

Python version is pinned in `py_version.txt`; the torch build used is in
`torch_build.txt`.

### Credentials

**No keys are committed to this repo, and none should ever be.** Every backend
reads from the environment:

| Backend | Env var | Used by |
|---|---|---|
| AWS Bedrock (primary) | `AWS_BEARER_TOKEN_BEDROCK`, or standard `AWS_*` / `~/.aws` credentials | `bedrock_llm.py` |
| OpenAI / vLLM | `OPENAI_API_KEY` (use `EMPTY` for a local vLLM server) | `world_model_causal.py`, `run_zero_shot.py` |
| Google Gemini | `GEMINI_API_KEY` | `gemini_llm.py` |

Put them in a shell profile or an untracked `.env`. `.gitignore` already excludes
`.env`, `*.pem`, and credential files — please keep it that way.

### Running without any API access

`run_agent_batch_rpg.py` ships a `mock` backend, so you can exercise the whole
RPG loop offline:

```bash
cd framework_code
python run_agent_batch_rpg.py --world-json ../examples/rpg_world/world_rpg_static_story_hidden_cause_apiary_night_mass_loss_2_seed5202.json \
  --scientist-backend mock -v
```

---

## What is *not* in this branch

This branch is a clean, shareable slice. Deliberately excluded:

- **Generated world datasets** (`all_out_bn/`, `all_out_rpg/`) — ~320 MB, and
  regenerable. One representative RPG world is included under `examples/` so you
  can see the format without cloning gigabytes.
- **Raw agent trajectories** (`framework_code/results/`) — ~7 GB.
- **ACED-Bench evaluation JSONs and figures** — we are not rerunning ACED, so
  only the headline findings are carried over, in `PROJECT_STATUS.md`.
- **Rebuttal / audit material** — one-off, superseded.
- Legacy generators and agent variants (`world_gen_2.py`, `scientist_agent_confidence.py`, …).

All of the above still exists on Vivian's machine in the full `ADS` repo if a
specific number needs to be traced back.

---

## Repo map

```
framework_code/
  # --- RPG (active) ---
  schemas_rpg.py            data contracts for static RPG worlds
  simulator_rpg.py          RPG world engine (observation + do-intervention)
  world_model_rpg.py        NL -> structured query translator for RPG
  scientist_agent_rpg.py    the RPG scientist agent
  orchestrator_rpg.py       RPG loop controller
  run_agent_batch_rpg.py    entry point: run agent over RPG worlds
  evaluate_rpg.py           scoring + failure bucketing

  # --- ACED-Bench (background) ---
  schemas.py  simulator.py  world_model_causal.py
  scientist_agent_causal.py            <action query|answer|give_up>
  scientist_coder_agent.py             + sandboxed <action code>
  scientist_coder_agent_new.py         modular INIT/CODE/ANALYSIS/DESIGN turns
  orchestrator.py  run_agent_batch.py  run_zero_shot.py
  evaluate_zero_shot.py  evaluate_advanced.py
  evidence_ledger_analysis.py  compute_cis.py
  json_converter.py                    world JSON -> BIF for pgmpy

  # --- LLM backends ---
  bedrock_llm.py  gemini_llm.py
  serve_qwen36.sh  serve_gpt_oss_120b.sh     local vLLM serving

dataset_generation_code/
  world_gen_rpg.py     audit_rpg.py           # RPG (active)
  world_gen_rpg_old.py                        # NOT legacy - live static engine, see note below
  world_gen_advanced.py  advanced_utils.py    # ACED decision worlds
  world_gen_causal.py  world_gen.py           # ACED structural worlds + shared graph utils
  run_many*.py  audit_advanced.py  check_faithfulness.py  validate_dataset.py

docs/
  rpg/    design plans, pipeline notes, slide walkthroughs   <- read these
  aced/   ARCHITECTURE.md, paper_polished.tex, world_gen_notes.md

examples/rpg_world/    one full v4 RPG world + its manifest
results_rpg/           RPG eval outputs (v3 pilot, v4 llm, v4 mixed)
```

---

## RPG quick reference

Generate worlds (needs Bedrock):

```bash
cd dataset_generation_code
python world_gen_rpg.py --outdir all_out_rpg/out_rpg_v4_llm \
  --distribution '{"story_hidden_cause_discovery": 8}' \
  --use-llm-templates --llm-model us.anthropic.claude-opus-4-7 --start-seed 5000
python audit_rpg.py --outdir all_out_rpg/out_rpg_v4_llm
```

Run the agent and score it:

```bash
cd framework_code
python run_agent_batch_rpg.py --worlds-dir ../dataset_generation_code/all_out_rpg/out_rpg_v4_llm \
  --scientist-backend bedrock --scientist-model us.anthropic.claude-opus-4-7 \
  --max-turns 12 -v -o results/rpg_v4_llm.json

python evaluate_rpg.py results/rpg_v4_llm.json --details -o evaluations/rpg/eval_rpg_v4_llm.json
```

Worlds are accepted by the runner only if `schema_version` is `rpg_static_v2` or
`rpg_static_v3`. Current generation is `rpg_static_v3`.

### Two filenames that lie

- **`world_gen_rpg_old.py` is live code.** `world_gen_rpg.py` imports it at
  module load for the static engine (`_static_observe`, `_static_apply`,
  `_static_sample_hidden`, `_static_utility_from_outcomes`, `rollout`). Removing
  it breaks the whole RPG import chain down to `simulator_rpg.py`.
- **`world_gen.py` is live code**, imported by `world_gen_causal.py`,
  `advanced_utils.py`, `audit_advanced.py`, and `check_faithfulness.py` for
  `find_ancestors` / `is_d_separated`, despite older notes describing it as
  legacy.

Also: `run_many.py` has no `__main__` guard and runs on import; it is a template
you edit in place, and it points at a placeholder `world_gen_xxx.py`.

Every module in `framework_code/` and `dataset_generation_code/` (except
`run_many.py`) has been import-checked in this tree, and the
mock-run → `evaluate_rpg.py` chain has been run end to end.
