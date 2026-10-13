# ACED-Bench: run new models

This folder runs a model on ACED-Bench with the same protocol, worlds, parser and scorer as the paper's main figures
(`fig9_yesno_round4` = ACED-Qual, `fig9_other_round4` = ACED-Core). It then prints the model next to the paper's 9 models.

- **Code:** `code/`, copied from `framework_code/` (the code that produced the paper runs). One fix is in
  `code/bedrock_llm.py`: newer Bedrock models (Opus/Sonnet 5.x, GPT-6) reject `temperature`, and their replies can start
  with a reasoning block.
- **Worlds:** `data/`, copied from the ACED repo `ADS/dataset_generation_code/all_out_bn/`. World files contain the
  answers.

| folder | from | worlds / questions | used for |
|---|---|---|---|
| `data/core_struct` | `out_bn_4_19_big_60` | 10 / 60 (50 yes/no, 10 list) | 3 agents, 4-turn budget |
| `data/core_decision` | `out_bn_adv_v3_60_balance` | 60 / 60 (12 per archetype) | 3 agents, 4-turn budget |
| `data/guess_struct` | `out_bn_4_19_big_subset` | 30 / 180 | guess-only |
| `data/guess_decision` | `out_bn_adv_v3` | 48 / 48 | guess-only |

Settings, all fixed by `run_model.sh`:
- 4 outer turns, using `--limit-variant rounds`;
- world parser and answer extractor: `us.anthropic.claude-opus-4-6-v1` (Bedrock). The paper's eval JSONs label it
  "opus-4-8-v1", but the raw run files show 4-6-v1, and no "4-8-v1" id exists on Bedrock;
- output cap 65536 tokens (`MAX_TOKENS`) for the agents and for guess-only (the paper's guess-only runs used 4096;
  a higher cap only removes truncation);
- shards: 10 for the structural cell, 20 for the decision cell, as in the original runs.

## Setup (yggdrasil)

```bash
cd ~/ADS_folder/ADS_shared/aced_benchmark
source ~/ADS_folder/key.txt                 # Bedrock (always needed: the parser runs on Bedrock) + Nautilus key
export PYTHON=~/miniconda3/envs/ADS/bin/python   # has pgmpy, boto3, openai, pandas, scipy
```

## Run one model

```bash
nohup bash run_model.sh NAME BACKEND MODEL > NAME.log 2>&1 &
```

The script runs 6 cells (agent / coder / coder_new × struct / decision) and 2 guess-only runs, all in parallel. It then
scores them and prints the table. It is safe to stop and restart, because finished questions are skipped. To re-score
only, run `STAGES=eval bash run_model.sh NAME BACKEND MODEL`.

Models not in the paper:

| model | command |
|---|---|
| Claude Opus 5.5 | `bash run_model.sh opus55 bedrock us.anthropic.claude-opus-5-5` |
| Claude Sonnet 5.5 | `bash run_model.sh sonnet55 bedrock us.anthropic.claude-sonnet-5-5` |
| GPT-6.1 Sol | `bash run_model.sh gpt61sol bedrock us.openai.gpt-6.1-sol` |
| GPT-6 Astra | `bash run_model.sh gpt6astra bedrock us.openai.gpt-6-astra` |
| Qwen 3.8 27B | `BASE_URL=https://ellm.nrp-nautilus.io/v1 API_KEY=$NAUTILUS_API_KEY bash run_model.sh qwen38 openai qwen3-small` |
| 9B (ours) | start a vLLM OpenAI server first, then `BASE_URL=http://localhost:8000/v1 API_KEY=EMPTY bash run_model.sh ours9b openai <served name>` |

Notes on these models:
- **Bedrock ids:** all four were checked to answer on 2026-10-13.
- **Qwen 3.8 27B:** `qwen3-small` is the Nautilus alias that served Qwen3.6-27B for the paper (`qwen36`). Check which model
  it serves now before labelling the run.
- **Already in the paper, no need to rerun:**
  - GPT-OSS-120B (`gpt-oss`, Nautilus);
  - Gemini 3.1 Pro preview (`gemini` backend, `GEMINI_API_KEY`);
  - Llama 4 Maverick and Llama 3.3 70B (Bedrock; need `MAX_TOKENS=8000`);
  - DeepSeek-V4-Pro (`openai` backend, `BASE_URL=https://api.deepseek.com`);
  - GPT-4o.

## Output (`out/NAME/`, not tracked by git)

```
out/NAME/
  round4/NAME_<agent>_<set>_round4/
    NAME_<agent>_<set>_round4.json   all answers of the cell (written by the final collect pass)
    shard_<i>.json                   per-shard partial files
    agent_logs/experiment_*.json     one full trajectory per question (used for resume)
    agent_query_data/                the data each query returned
  guess/zero_shot_guess_struct_{easy,medium}.json, zero_shot_guess_decision_hard.json
  eval/eval_*.json                   scored files (same format as the paper's evaluations/)
  logs/                              one log per shard and per collect pass
```

To print the comparison table: `python summarize.py out/NAME [out/NAME2 ...]`. Groups follow the paper notebook:
- **ACED-Qual:** the yes/no structural questions;
- **ACED-Core:** the list structural questions plus all decision questions.

Paper cells come from `paper_reference.json`, which was built from `ADS/framework_code/evaluations/{round,for_paper}`. One
known difference: the figure overrides Opus modular Qual to 0.90; the JSON gives 0.64 (a stale file, per the notebook).

## Time

Measured on 2026-05 (paper runs):
- one cell took 0.2–5 h;
- the 6 cells run at the same time, so one model takes about **2–5 h wall**;
- guess-only adds no wall time (it runs alongside).

One model starts 98 processes. Every run also calls the Opus 4.6 parser on Bedrock, so 2–3 models at once is the
practical limit.

## Caveat for the camera-ready

These world sets reproduce the paper's figures. They are not the partition stated in the paper text (Core = 60 list
+ 120 decision, Qual = 150); see `personal_docs/camera_ready/03_RUN_LIST.md` A0/A1. If the camera-ready freezes a new
manifest, point `run_model.sh` at those world folders instead.
