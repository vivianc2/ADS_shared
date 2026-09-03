# ADS_shared (RPG) — repository guide

This repo holds **RPG**: procedurally-generated scientific-reasoning ("causal-discovery")
worlds, an oracle/grader for them, and an RL setup that trains a policy to solve them. It
also carries the free-text evaluation harness and a lot of committed run outputs.

**If you read one thing:** the current pipeline is **v9**, and the step-by-step to
reproduce it on a fresh box is in [`SETUP_NEW_SERVER.md`](SETUP_NEW_SERVER.md).

## Which version is current

The world generator evolved `v6 → v7 → v8 → v9`, each a copy-forward of the last.
**v9 is the only current one.** Everything else is kept for provenance, not for running.

| Path | What it is | Status |
|---|---|---|
| `dataset_generation_code/rpg_v9/` | Current world generator (sampler, oracle, engine) + the committed de-leaked dataset in `data_v9_deleaked/{train,validation}.parquet` | **current** |
| `dataset_generation_code/rpg_rl/` | RL layer: id-space environment, reward, world stream, splits, id-space eval | **current** |
| `dataset_generation_code/skyrl_rpg/` | SkyRL integration: dataset builder, env registration, and the `run_rpg.sh` launcher | **current** (the trainer) |
| `framework_code/` | Free-text / API evaluation harness (`evaluate_advanced.py`, `evaluate_zero_shot.py`) | **current** (eval) |
| `dataset_generation_code/rpg_v6_prototype/`, `rpg_v7_prototype/`, `rpg_v8/` | Earlier generations of the generator | frozen / historical |
| `docs/rpg/` | v6-era write-ups and slides | historical (describe v6, not v9) |
| `dataset_generation_code/rpg_rl/train_grpo.py` | Standalone HF-generate GRPO prototype | superseded by `skyrl_rpg` |
| `dataset_generation_code/rpg_curriculum/` | Easy-first curriculum scheduler | draft, not yet wired into training |

### A naming heads-up
Inside `rpg_v9/` you will see files named `oracle_v6.py`, `sim_v6.py`, `generate_v7.py`,
and docstrings that say things like "v6 engine" or "v7 sampler". Those version numbers
name the **design generation of that component** (carried forward unchanged) — they are
**not** the dataset version. The dataset version is the directory name: `rpg_v9`.

## The pipeline (what actually runs)

1. **Build the RL dataset** — `skyrl_rpg/rpg_dataset.py` samples and audits worlds with the
   `rpg_v9` generator and writes train/validation parquet. The canonical de-leaked set is
   already committed at `rpg_v9/data_v9_deleaked/`, so this step is optional.
2. **Train** — SkyRL GRPO + LoRA via `skyrl_rpg/run_rpg.sh` (→ `skyrl_rpg/main_rpg.py`; the
   env is registered from `skyrl_rpg/env.py`, and reward/env logic lives in `rpg_rl/`).
3. **Evaluate** — RL-comparable id-space eval: `rpg_rl/id_space_eval.py`. Free-text / API
   benchmark: `framework_code/evaluate_advanced.py`.

`RPG_PROTO` selects the generator directory and **must** match the one the dataset was
built with. It defaults to `rpg_v9` everywhere; don't point it at an older generation.

## Notes

- Large per-run outputs (`results_*/`, `beliefs/`, per-world CSVs) are committed for
  provenance. They are regenerable and are **not** needed to run anything.
- Credentials (`key.txt`, `wandb_key.txt`) are never committed — see `SETUP_NEW_SERVER.md`
  for what you need to obtain and where it goes.
