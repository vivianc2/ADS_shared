# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **causal discovery benchmark** where an LLM "Scientist" agent must infer properties of a hidden causal graph by requesting observational/interventional data samples from a Bayesian Network simulator. The framework compares agent-based causal reasoning vs. zero-shot baselines across multiple LLM backends (local HuggingFace, OpenAI-compatible API / vLLM, AWS Bedrock).

All framework scripts run from `framework_code/`:
```bash
cd /home/vivianchen/ADS/framework_code
```

## Key Commands

### Dataset Generation
```bash
cd dataset_generation_code
python run_many.py                        # Generate 60 worlds (edit OUTDIR/SEED_BASE inside)
python validate_dataset.py all_out_bn/out_bn_3_4  # Validate a generated dataset
```

`run_many.py` calls `world_gen_causal.py` to produce 20 worlds at each size (10, 20, 30 nodes), 60 total.

### Running Experiments

**Zero-shot baseline** (no data queries — LLM answers from domain knowledge alone):
```bash
# Local HuggingFace model
python run_zero_shot.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 -v

# OpenAI API
python run_zero_shot.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --backend openai --model gpt-4o-mini -v

# AWS Bedrock
python run_zero_shot.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --backend bedrock --model us.anthropic.claude-opus-4-0-20250514-v1:0 -v
```

**Zero-shot with sub-prompt** (uses scientist-like prompt structure, but no data access):
```bash
python run_zero_shot_sub_prompt.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --backend openai --model gpt-4o-mini -v
```

**Agent batch** (scientist queries data iteratively):
```bash
# Local Qwen
python run_agent_batch.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --scientist-backend local --scientist-model Qwen/Qwen2.5-7B-Instruct -v

# vLLM server (launch separately: vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --gpu-memory-utilization 0.5)
python run_agent_batch.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --scientist-backend openai --scientist-model meta-llama/Llama-3.1-8B-Instruct --scientist-base-url http://localhost:8000/v1 --scientist-api-key EMPTY -v

# OpenAI API
export OPENAI_API_KEY=sk-...
python run_agent_batch.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --scientist-backend openai --scientist-model gpt-4o-mini -v

# AWS Bedrock
python run_agent_batch.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --scientist-backend bedrock --scientist-model us.anthropic.claude-opus-4-0-20250514-v1:0 -v

# Coder agent variant (adds Python execution tool — add --agent-type coder to any of the above)
python run_agent_batch.py --worlds-dir ../dataset_generation_code/all_out_bn/out_bn_3_4 --scientist-backend openai --scientist-model gpt-4o-mini --agent-type coder -v
```

### Evaluation
```bash
# Zero-shot results
python evaluate_zero_shot.py results/zero_shot_<timestamp>.json --details

# Agent results (use --llm-extract to have an LLM re-extract answers from verbose reasoning)
python evaluate_zero_shot.py results/agent_<timestamp>.json --details --llm-extract -o evaluations/eval_output.json
```

### Failure Analysis
```bash
# Analyze failures using Bedrock Claude (reads per-experiment logs)
python analyze_failures.py --results-dir results/opus_agent_3_4 --eval-file evaluations/eval_opus_agent_3_4.json --output-dir analysis_output/opus_agent_3_4
```

## Key `run_agent_batch.py` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--world-model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for world model query parser (always local) |
| `--scientist-backend` | `local` | `local` (HuggingFace), `openai` (OpenAI-compatible API / vLLM), or `bedrock` (AWS) |
| `--scientist-model` | same as `--world-model` | Model name for scientist agent |
| `--agent-type` | `agent` | `agent` (ScientistAgent) or `coder` (CoderScientistAgent with Python execution) |
| `--max-queries` | `10` | Query budget per question |
| `--temperature` | `0.3` | Sampling temperature for scientist |
| `--causal` | always on | Causal variants (`world_model_causal` + `scientist_agent_causal`) are always used |

## Architecture

```
run_agent_batch.py (primary entry point)
         │
    orchestrator.py (manages interaction loop, logs turns, saves JSON results)
         │
    ┌────┴─────────────────────────┐
    ▼                              ▼
scientist_agent_causal.py    world_model_causal.py
  (or scientist_coder_agent.py)  (NL query → ParsedQuery → simulator → XML result)
(LLM reasons about causal            │
 structure, queries or answers)  simulator.py (pgmpy wrapper, observational + do-calculus sampling)
```

**Core modules:**
- `schemas.py`: Data contracts (`ParsedQuery`, `QueryResult`, `WorldInfo`, `Question`, error types)
- `simulator.py`: Bayesian Network engine; implements do-operator by mutilating graph (removes incoming edges, fixes CPD to point mass)
- `world_model_causal.py`: LLM-powered NL→structured query translator + validator; supports `QwenLLM` (local), `OpenAILLM` (API), and `BedrockLLM` (AWS)
- `scientist_agent_causal.py`: Causal reasoning agent; outputs `<action type="query/answer/give_up">` XML; maintains scientist memory and query history
- `scientist_coder_agent.py`: Extends agent with `<action type="code">` for Python execution (pandas/scipy analysis); max 8 code rounds per turn, 30s timeout
- `scientist_agent_confidence.py`: Agent variant that requires explicit confidence level before answering
- `orchestrator.py`: Loop controller; enforces budget, logs all turns, evaluates answers, writes `results/agent_<timestamp>.json`
- `json_converter.py`: Converts world JSON files to BIF format for pgmpy
- `bedrock_llm.py`: AWS Bedrock LLM backend (Converse API)

**Zero-shot scripts** (no orchestrator/scientist/world_model — direct LLM calls):
- `run_zero_shot.py`: Standard zero-shot baseline
- `run_zero_shot_sub_prompt.py`: Zero-shot using reduced scientist-like prompt structure

**Analysis tools:**
- `evaluate_zero_shot.py`: Evaluates both zero-shot and agent results; `--llm-extract` re-extracts answers from verbose output
- `analyze_failures.py`: LLM-based failure categorization using Bedrock
- `demo_coder_agent.py`: Standalone demo for debugging agent prompts and responses

**Deprecated (still present but unused by active code paths):**
- `world_model.py`: Non-causal world model variant
- `scientist_agent.py`: Non-causal scientist agent variant
- `run_experiment.py`: Single-world entry point (imports non-causal versions)

## World JSON Format

Each world has `meta`, `story`, `variables`, `edges`, `cpds`, `questions`, and `non_intervenable_variables`. The scientist sees variables + story but NOT edges/CPDs.

```json
{
  "meta": {"topic": "Criminal Justice", "n_nodes": 10, "seed": 1010, ...},
  "story": "In a criminal justice system...",
  "variables": [{"name": "Arrest", "desc": "...", "values": ["yes", "no"]}, ...],
  "edges": [["var1", "var2"], ...],
  "cpds": [...],
  "questions": [
    {"id": 0, "question": "Does X cause Y?", "question_type": "causal_effect", "answer": "Yes", "difficulty": "easy"},
    ...
  ],
  "non_intervenable_variables": {"VarName": "reason why not manipulable"}
}
```

## Question Types

Each world has 6 questions across 3 difficulty groups (2 per group):

**Group 1 — Causal Structure (easy):**
- `causal_effect`: "Does A cause B?" (Yes/No — tests if A is ancestor of B)
- `all_causes_of` or `all_effects_of`: "What are all causes/effects of X?" (list of ancestors/descendants)

**Group 2 — Marginal Independence (medium):**
- Questions about whether two variables are marginally independent/dependent
- Post-hoc classified by structural motif: `direct_marginal`, `chain_marginal`, `fork_marginal`, `v_structure_marginal`, `other_marginal`

**Group 3 — Conditional Independence (medium):**
- Questions about whether two variables are independent/dependent given conditioning set
- Post-hoc classified: `chain_conditional`, `fork_conditional`, `v_structure_conditional`, `other_conditional`

Yes/No balance is guaranteed by answer-first generation with 50/50 stratification.

## Dataset

Primary benchmark dataset: `dataset_generation_code/all_out_bn/out_bn_3_4/`
- 60 worlds: 20 × 10-node, 20 × 20-node, 20 × 30-node
- 8 topics: Criminal Justice, Education, Hospital data, Labor & Policy, Screening & diagnosis, Social Science, Treatment effectiveness, User Behavior
- 360 total questions (6 per world): 120 easy + 240 medium

## Output Artifacts

- `results/agent_<timestamp>.json`: Full agent interaction history (queries, responses, reasoning)
- `results/zero_shot_<timestamp>.json`: Zero-shot results (split by difficulty group)
- `results/agent_logs/`: Per-experiment JSON logs from orchestrator
- `results/agent_query_data/`: CSV files with sampled data from simulator
- `evaluations/eval_*.json`: Accuracy summaries with per-question-type breakdowns
- `analysis_output/`: LLM-based failure analysis reports
