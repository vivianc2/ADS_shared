#!/usr/bin/env bash
#
# serve_and_run_muse_glimmer.sh
#
# One-shot: serve Meta's Muse-Glimmer-30B with vLLM (OpenAI-compatible) and run
# the v8 72-world validation set through the existing run_batch_v6.py harness
# (--backend vllm). Meant to be pulled onto a GPU server and run as-is.
#
#   cd .../dataset_generation_code/rpg_v8
#   bash serve_and_run_muse_glimmer.sh
#
# Everything is overridable via env vars (see the CONFIG block). The harness
# talks to the model over HTTP only, so this script starts vLLM, waits for it to
# be healthy, runs the 72 worlds, prints the summary, then shuts the server down.
#
# Prereqs on the server (NOT installed here):
#   - Python env with: vllm (recent enough to know model_type "muse_glimmer";
#     if not, set MODEL_IMPL=transformers to use vLLM's transformers backend),
#     plus the harness deps (boto3, openai). transformers>=5.15 is required.
#   - GPU(s) with enough memory for a 30B bf16 model (~60GB weights + KV cache).
#     Single 80GB card works for modest context; for the full 128K context or
#     smaller cards, use TENSOR_PARALLEL>=2 or a quantized MODEL (see README).
#   - HF access to the (likely gated) repo: `huggingface-cli login` or HF_TOKEN.
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# CONFIG (override by exporting before running)
# --------------------------------------------------------------------------- #
MODEL="${MODEL:-meta-models/Muse-Glimmer-30B}"      # HF repo id (or local path / quant repo)
SERVED_NAME="${SERVED_NAME:-muse-glimmer}"          # name vLLM advertises == --model passed to harness
PORT="${PORT:-8000}"                                # vLLM default; matches VLLM_DEFAULT_BASE_URL
WORLDS_DIR="${WORLDS_DIR:-rpg_v8_fast_worlds}"       # the 72-world set (8x9 archetypes)
OUTDIR="${OUTDIR:-out_muse_glimmer_72}"             # per-world results + summary.json land here
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"            # full model context (avoid truncation). Lower to save KV memory.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"            # generation cap per turn (avoid clipping long answers)
CONCURRENCY="${CONCURRENCY:-8}"                     # worlds in flight; exploits vLLM continuous batching
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MODEL_IMPL="${MODEL_IMPL:-}"                          # set to "transformers" if your vLLM lacks native muse_glimmer
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"          # set to 1 only if the repo requires it
SKIP_SERVE="${SKIP_SERVE:-0}"                         # 1 = a vLLM server is already up at VLLM_BASE_URL
NO_RESOLVER="${NO_RESOLVER:-0}"                       # 1 = pass --no-resolver-llm (see README on grading)
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"         # how long to wait for weights to load + server ready

# tensor parallelism: default to every visible GPU
_gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || echo 1)
[ -z "$_gpus" ] || [ "$_gpus" -lt 1 ] && _gpus=1
TENSOR_PARALLEL="${TENSOR_PARALLEL:-$_gpus}"

PY="${PYTHON:-python}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

echo "=========================================================="
echo " Muse-Glimmer-30B  ->  v8 72-world validation set"
echo "  model            : $MODEL  (served as '$SERVED_NAME')"
echo "  worlds           : $WORLDS_DIR  ($(ls "$WORLDS_DIR"/world_*.json 2>/dev/null | wc -l | tr -d ' ') worlds)"
echo "  outdir           : $OUTDIR"
echo "  tensor-parallel  : $TENSOR_PARALLEL GPU(s)"
echo "  max-model-len    : $MAX_MODEL_LEN   max-new-tokens: $MAX_NEW_TOKENS"
echo "  concurrency      : $CONCURRENCY"
echo "  base url         : $VLLM_BASE_URL"
echo "  resolver         : $([ "$NO_RESOLVER" = 1 ] && echo disabled || echo 'harness default (Bedrock Opus if AWS_BEARER_TOKEN_BEDROCK set, else reuse agent)')"
echo "=========================================================="

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">> stopping vLLM server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# 1. Serve
# --------------------------------------------------------------------------- #
if [ "$SKIP_SERVE" != "1" ]; then
  serve_args=(
    serve "$MODEL"
    --served-model-name "$SERVED_NAME"
    --port "$PORT"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
  )
  [ "$TRUST_REMOTE_CODE" = "1" ] && serve_args+=(--trust-remote-code)
  [ -n "$MODEL_IMPL" ] && serve_args+=(--model-impl "$MODEL_IMPL")

  echo ">> launching: vllm ${serve_args[*]}"
  vllm "${serve_args[@]}" > vllm_server.log 2>&1 &
  SERVER_PID=$!
  echo ">> vLLM pid $SERVER_PID, logs -> $(pwd)/vllm_server.log"

  echo ">> waiting for server health (up to ${HEALTH_TIMEOUT_S}s while weights load)..."
  waited=0
  until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!! vLLM exited during startup. Last 40 log lines:"; tail -n 40 vllm_server.log; exit 1
    fi
    sleep 10; waited=$((waited+10))
    if [ "$waited" -ge "$HEALTH_TIMEOUT_S" ]; then
      echo "!! server not healthy after ${HEALTH_TIMEOUT_S}s. Last 40 log lines:"; tail -n 40 vllm_server.log; exit 1
    fi
  done
  echo ">> server healthy after ${waited}s."
else
  echo ">> SKIP_SERVE=1: assuming a vLLM server is already up at $VLLM_BASE_URL"
fi

# --------------------------------------------------------------------------- #
# 2. Run the 72-world batch through the existing harness
# --------------------------------------------------------------------------- #
run_args=(
  run_batch_v6.py
  --backend vllm
  --model "$SERVED_NAME"
  --worlds-dir "$WORLDS_DIR"
  --outdir "$OUTDIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --concurrency "$CONCURRENCY"
  -v
)
[ "$NO_RESOLVER" = "1" ] && run_args+=(--no-resolver-llm)

echo ">> running: $PY ${run_args[*]}"
"$PY" "${run_args[@]}"

echo ">> done. Results in $OUTDIR/ (summary.json + result_<world>.json)."
