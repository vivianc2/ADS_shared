# RPG v7 runbook — what to rerun, and Qwen3-8B via vLLM

**Date:** 2026-08-07 · **Location:** ADS_shared (canonical)
All paths are relative to `dataset_generation_code/rpg_v7_prototype/`. Activate the env
first: `conda activate ADS-rpg`.

---

## 0. Why a rerun is needed (read first)

The V4 fix added `_selection_nodes` to ground_truth, which changed how names are drawn from
the pools → **same seeds now produce slightly different worlds → world_ids shifted.** So:

- The regenerated `out_v7/{chain,collider,subtype,mixed9}` are the **correct current sets**
  (V4/V5 baked in: 0 valid-proxy leaks, `lenient_mechanism_proxies` present).
- The old `results_v7/mixed9_opus/` live run references **pre-regeneration** worlds (4/9 of
  its world files no longer exist) and predates the grader fixes. **Do not compare it to the
  new worlds.** It stays only as a historical record.

**Bottom line:** to get a clean, comparable headline number under the fixed grader, rerun
the agent on the regenerated `out_v7/mixed9/` (or a larger set). Both models below run
against the same worlds so Opus vs Qwen3-8B is apples-to-apples.

---

## 1. WHAT TO RERUN

### 1a. Opus 4.8 on the regenerated mixed9 (the refreshed headline)
```bash
export AWS_BEARER_TOKEN_BEDROCK=...        # your token
export AWS_DEFAULT_REGION=us-west-2
python run_batch_v6.py --worlds-dir out_v7/mixed9 \
    --backend bedrock --model us.anthropic.claude-opus-4-8 \
    --outdir results_v7/mixed9_opus_v2 -v
```
- Writes to `mixed9_opus_v2` (NOT the stale `mixed9_opus`) so the old record is preserved.
- If it breaks or you stop it, rerun the **same command with `--resume`** — it skips worlds
  that already have a `result_<wid>.json`.
- ~40 min for 9 worlds (matches the first run).

### 1b. (optional) A bigger, balanced set for a firmer number
The 9-world set is a smoke read; per-archetype n=3 is noisy. For a real result, run the full
24-per-archetype sets you already generated:
```bash
for d in chain collider subtype; do
  python run_batch_v6.py --worlds-dir out_v7/$d \
    --backend bedrock --model us.anthropic.claude-opus-4-8 \
    --outdir results_v7/${d}_opus -v --resume
done
python analyze_results.py --run opus_chain=results_v7/chain_opus \
    --run opus_collider=results_v7/collider_opus \
    --run opus_subtype=results_v7/subtype_opus --out results_v7/report_opus.md
```
72 worlds × ~5 min ≈ 6 h. Use `--resume` freely; push results back when done.

---

## 2. Qwen3-8B via vLLM (self-hosted, Linux server)

A `vllm` backend was added. vLLM serves an OpenAI-compatible API, so the runner talks to it
exactly like any OpenAI endpoint.

### 2a. Launch the vLLM server (on the GPU box)
```bash
pip install vllm            # once, in the server env
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
# (add --enable-reasoning --reasoning-parser qwen3  if your vLLM build separates
#  <think> into reasoning_content; the client captures it either way.)
```
Leave it running (or use `screen`/`tmux`). Sanity check:
```bash
curl http://localhost:8000/v1/models        # should list Qwen/Qwen3-8B
```

### 2b. Point the runner at it and run the SAME worlds
```bash
export VLLM_BASE_URL=http://localhost:8000/v1     # default; set if server is elsewhere
# (VLLM_API_KEY is optional; vLLM ignores it. Defaults to "EMPTY".)

python run_batch_v6.py --worlds-dir out_v7/mixed9 \
    --backend vllm --model Qwen/Qwen3-8B \
    --outdir results_v7/mixed9_qwen8b -v --resume
```
- `--model` must match the string vLLM served (`Qwen/Qwen3-8B`). The Qwen3 sampling preset
  (temp 1.0 / top_p 0.95 / top_k 20) applies automatically.
- **Resolver note:** with no Bedrock creds, the resolver falls back to **reusing the agent
  model** (Qwen3-8B resolves its own free-text). That makes resolution quality model-
  dependent. Two options:
  - **Recommended for a fair Opus-vs-Qwen comparison:** give the resolver a *fixed* strong
    model by setting `AWS_BEARER_TOKEN_BEDROCK` (the resolver auto-uses Opus, agent stays
    Qwen) — resolution held constant across both runs.
  - **Fully self-hosted (no Bedrock):** accept Qwen-as-its-own-resolver, or add
    `--no-resolver-llm` to use lexical-only (fast, but weaker on verbose answers).
- Thinking model → long outputs. Default `--max-new-tokens` is fine; if you see truncated
  `<action>` tags, raise it (`--max-new-tokens 8192`).

### 2c. Compare Opus vs Qwen3-8B on identical worlds
```bash
python analyze_results.py \
    --run opus=results_v7/mixed9_opus_v2 \
    --run qwen8b=results_v7/mixed9_qwen8b \
    --out results_v7/report_opus_vs_qwen8b.md
```

---

## 3. Interpreting results (guardrails from the last run)

- **Check `artifact_suspects` in the summary first.** The batch prints a loud warning; a
  flagged failure may be a resolver/grader artifact, not reasoning. The grader fixes
  (verbose-proxy resolution, alt-fix fairness) should keep this low now — but verify.
- **Read part A and part B separately.** part A = found a utility-optimal fix (benefit ≥
  0.90); part B = named the mechanism proxy / rejected decoys / signs (the counterfactual
  battery). The known pattern is "acts right (A), explains wrong (B)".
- **Qwen3-8B is a debug-scale model** — expect lower part A than Opus. The point of this run
  is (a) confirm the vLLM path works end-to-end, and (b) get a cheap second data point, not
  a definitive number.
- If Qwen produces many empty/truncated turns, that's the output-cap issue seen before
  (thinking models blow the token budget): raise `--max-new-tokens`. Unproductive turns
  don't consume the query budget (the runner retries), but they waste wall-clock.

---

## 4. Push/pull workflow

- All code + regenerated worlds are in `ADS_shared` (this repo). Commit + push, pull on the
  server, run, push results back.
- **Do NOT commit large result dumps** to the shared branch (CLAUDE.md rule). The
  `results_v7/*/result_*.json` + `*_data/` CSVs are multi-MB — keep them local/gitignored or
  summarize into a doc. `summary.json` + `analyze_results` reports are small and fine to
  share.
- Quick local smoke before pushing (no GPU/API): `python test_reward_integrity.py` and
  `python run_batch_v6.py --worlds-dir out_v7/mixed9 --backend mock --outdir /tmp/x`.
