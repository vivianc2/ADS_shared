# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Orientation

Read [PROJECT_STATUS.md](PROJECT_STATUS.md) first. Short version:

- **ACED-Bench** (`*_causal.py`, `*_coder_agent*.py`) is finished work, kept as
  background for the paper in `docs/aced/paper_polished.tex`.
- **RPG** (`*_rpg.py`) is the active line of work. Default to assuming a request
  is about RPG unless ACED is named.

## Hard rules

- **Never commit credentials.** All backends read from env vars
  (`AWS_BEARER_TOKEN_BEDROCK`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). Do not add a
  key to a file, a default argument, a notebook cell, or a shell script. Do not
  weaken the `.gitignore` entries for `.env`, `*.pem`, or credential files.
- **This branch is shared.** Do not add generated datasets, raw agent
  trajectories, or multi-megabyte result dumps. If output needs to be inspected,
  summarize it in a doc instead of committing the artifact.
- **Do not quote ACED result numbers from memory or from old notes.** The
  unrestricted Decision runs have mixed provenance and different files
  genuinely disagree. Read `scores.overall.total` from a named JSON and check
  the row count, or say the number is unverified. See PROJECT_STATUS.md §1.

## Terminology

Current: **ACED-Bench**, **ACED-Struct**, **ACED-Decision**.
Retired in prose: `PGM-Struct`, `PGM-Decision`, `Basic`, `Advanced`,
`guess-shot`. These still appear in filenames and variable names — fine as
paths, not as prose.

## RPG specifics

- Current archetype: `story_hidden_cause_discovery`; current schema:
  `rpg_static_v3`. `run_agent_batch_rpg.py` silently skips worlds whose
  `schema_version` is not `rpg_static_v2` / `rpg_static_v3`.
- v4 hardening exists to kill two shortcuts: **name leakage** (action/proxy names
  that reveal the latent cause) and **all-binary interventions**. When adding
  worlds, templates, or variable names, preserve both properties — a
  descriptive-but-leaky name silently undoes the benchmark.
- v4 deliberately keeps the role contract stable so `simulator_rpg.py`,
  `world_model_rpg.py`, `evaluate_rpg.py`, and `audit_rpg.py` need at most a
  one-line change. Prefer changes that hold that line.
- `run_agent_batch_rpg.py --scientist-backend mock` runs the full loop with no
  API access — use it for smoke tests.
- **`world_gen_rpg_old.py` is not legacy despite its name.** It holds the live
  static engine (`_static_observe`, `_static_apply`, `_static_sample_hidden`,
  `_static_utility_from_outcomes`, `rollout`) that `world_gen_rpg.py` imports at
  module load. Delete it and the entire RPG import chain breaks, including
  `simulator_rpg.py`. Same trap with `world_gen.py`, which `world_gen_causal.py`,
  `advanced_utils.py`, `audit_advanced.py`, and `check_faithfulness.py` all
  import even though old notes call it legacy.
- `run_many.py` has no `if __name__ == "__main__"` guard and executes on import;
  it is an edit-the-constants-inside template that references a placeholder
  `world_gen_xxx.py`. Do not import it.

## Commands

```bash
# RPG: generate -> audit -> run -> evaluate
cd dataset_generation_code
python world_gen_rpg.py --outdir all_out_rpg/<name> \
  --distribution '{"story_hidden_cause_discovery": 8}' --use-llm-templates
python audit_rpg.py --outdir all_out_rpg/<name>

cd ../framework_code
python run_agent_batch_rpg.py --worlds-dir ../dataset_generation_code/all_out_rpg/<name> \
  --scientist-backend bedrock --scientist-model us.anthropic.claude-opus-4-7 -v -o results/<name>.json
python evaluate_rpg.py results/<name>.json --details -o evaluations/rpg/eval_<name>.json
```

```bash
# ACED (background; agent runs need --llm-extract at eval time)
cd framework_code
python run_agent_batch.py --worlds-dir <dir> --agent-type coder_new \
  --scientist-backend bedrock --scientist-model us.anthropic.claude-opus-4-7 -v
python evaluate_zero_shot.py results/agent_<ts>.json --details --llm-extract -o evaluations/eval_<name>.json
```

## Protocol constants (ACED, fixed across runs)

- Query budget 10; parser Opus-4.8 @ temp `0.1` / 512 tok; ledger annotator
  temp `0.0` / 600 tok / ≤3 retries
- Decision scoring tolerance `0.05` expected-state-index units
- Structural dependency threshold TV ε = `0.02`
- Expected-state-index uses the ordered categorical state list, zero-based;
  lower outcome index is better unless a world says otherwise

## Architecture

```
run_agent_batch_rpg.py                    run_agent_batch.py / run_zero_shot.py
        |                                          |
   orchestrator_rpg.py                      orchestrator.py
        |                                          |
  scientist_agent_rpg.py               scientist_agent_causal.py
  world_model_rpg.py                   | scientist_coder_agent.py
        |                              | scientist_coder_agent_new.py
  simulator_rpg.py                     world_model_causal.py
                                             |
                                       simulator.py  (pgmpy obs + do-calculus)
```

Full ACED detail: `docs/aced/ARCHITECTURE.md`.
RPG design docs: `docs/rpg/` — start with `worldgen_rpg_plan_v4_complex_neutral_dose.md`.
