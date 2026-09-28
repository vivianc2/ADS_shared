#!/usr/bin/env bash
#
# serve_and_run_muse_glimmer.sh
#
# One-shot: serve Muse-Glimmer-30B with vLLM (OpenAI-compatible) and run the v8
# 72-world validation set through the existing run_batch_v6.py harness
# (--backend vllm).
#
#   cd .../dataset_generation_code/rpg_v8
#   bash serve_and_run_muse_glimmer.sh
#
# Everything is overridable via env vars (see the CONFIG block). The harness
# talks to the model over HTTP only, so this script starts vLLM, waits for it to
# be healthy, runs the 72 worlds, prints the summary, then shuts the server down.
#
# ---------------------------------------------------------------------------
# THIS BOX IS SHARED. Read before changing GPU settings.
# ---------------------------------------------------------------------------
# The 8x L40S here are mostly other people's jobs. This script pins itself to
# CUDA_DEVICES (default "1") and refuses to start if that GPU is already busy.
# Do NOT set TENSOR_PARALLEL to "all visible GPUs" -- vLLM grabs
# GPU_MEM_UTIL of each card's TOTAL memory and will OOM into co-tenants.
#
# Prereqs (built by the `glimmer` conda env -- see README notes at bottom):
#   - vllm >= 0.27.1 AND transformers >= 5.15. vLLM has NO native muse_glimmer
#     kernel (checked v0.27.1 and main), so MODEL_IMPL=transformers is REQUIRED,
#     not optional. transformers 5.15 is what actually ships the modeling code.
#   - Enough VRAM. Full bf16 is ~60GB (the model card targets 64GB VRAM), so it
#     needs the 2x48GB L40S pair with TENSOR_PARALLEL=2. It does NOT fit one
#     card -- for a single GPU use RedHatAI/Muse-Glimmer-30B-FP8-block (~30GB).
#     NVFP4 repos are Blackwell-only -- useless on Ada L40S.
set -euo pipefail

# --------------------------------------------------------------------------- #
# CONFIG (override by exporting before running)
# --------------------------------------------------------------------------- #
MODEL="${MODEL:-meta-models/Muse-Glimmer-30B}"      # full bf16, ~60GB -- needs the 2x48GB pair below.
                                                    # On ONE card instead, use
                                                    # RedHatAI/Muse-Glimmer-30B-FP8-block (~30GB).
SERVED_NAME="${SERVED_NAME:-muse-glimmer}"          # name vLLM advertises == --model passed to harness
                                                     # (also what openai_llm.py's preset matches on)
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"                 # physical GPU(s) to use. ONLY use cards you own.
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"             # must equal the number of ids in CUDA_DEVICES
PORT="${PORT:-8020}"                                # 8000/8001/8077 are in use on this box
WORLDS_DIR="${WORLDS_DIR:-rpg_v8_fast_worlds}"      # the 72-world set (8x9 archetypes)
OUTDIR="${OUTDIR:-out_muse_glimmer_72}"             # per-world results + summary.json land here
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"             # ~36GB left for KV after bf16 weights across 2 cards.
                                                    # Model ceiling is 131072; raise if runs truncate.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"            # per-turn generation cap. Glimmer is a reasoning model;
                                                    # 2500 (the old harness default) clips it mid-think.
CONCURRENCY="${CONCURRENCY:-8}"                     # worlds in flight; exploits vLLM continuous batching
DTYPE="${DTYPE:-auto}"                              # auto: respect the checkpoint's own dtype
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"                # fraction of the PINNED card only
MODEL_IMPL="${MODEL_IMPL:-transformers}"            # REQUIRED: no native vLLM muse_glimmer support
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"                       # 1 = a vLLM server is already up at VLLM_BASE_URL
NO_RESOLVER="${NO_RESOLVER:-0}"                     # 1 = pass --no-resolver-llm (see grading notes)
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"        # weights load + torch.compile on a cold cache is slow
FORCE_GPU="${FORCE_GPU:-0}"                         # 1 = skip the "GPU is busy" guard. Think first.
CONDA_ENV="${CONDA_ENV:-glimmer}"

# Use the glimmer env unless the caller already pointed PYTHON somewhere.
if [ -z "${PYTHON:-}" ]; then
  _conda_sh="/home/vivianchen/miniconda3/etc/profile.d/conda.sh"
  [ -f "$_conda_sh" ] && . "$_conda_sh" && conda activate "$CONDA_ENV"
fi
PY="${PYTHON:-python}"

# CUDA forward compatibility. vLLM >= 0.21 pins a torch built for CUDA 13,
# which normally demands driver >= 580 -- this box runs 570.153.02 (CUDA 12.8),
# so torch.cuda.is_available() is False without this. cuda-compat-13-0 ships a
# userspace libcuda.so.580 that works against the older kernel module (supported
# on datacenter GPUs; L40S qualifies). Purely user-local: no root, no driver
# change, nothing visible to the other tenants on this box.
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/home/vivianchen/opt/cuda-13.0-compat/compat}"
if [ -f "$CUDA_COMPAT_DIR/libcuda.so.1" ]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
else
  echo "!! CUDA_COMPAT_DIR=$CUDA_COMPAT_DIR missing libcuda.so.1."
  echo "   Re-extract with:"
  echo "     curl -sLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb"
  echo "     dpkg-deb -x cuda-compat-13-0_*.deb /tmp/cc && cp -r /tmp/cc/usr/local/cuda-13.0 ~/opt/cuda-13.0-compat"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

_n_worlds=$(ls "$WORLDS_DIR"/world_*.json 2>/dev/null | wc -l | tr -d ' ')

echo "=========================================================="
echo " Muse-Glimmer-30B  ->  v8 72-world validation set"
echo "  model            : $MODEL  (served as '$SERVED_NAME')"
echo "  model-impl       : $MODEL_IMPL"
echo "  worlds           : $WORLDS_DIR  ($_n_worlds worlds)"
echo "  outdir           : $OUTDIR"
echo "  GPU(s)           : CUDA_VISIBLE_DEVICES=$CUDA_DEVICES  tensor-parallel=$TENSOR_PARALLEL"
echo "  max-model-len    : $MAX_MODEL_LEN   max-new-tokens: $MAX_NEW_TOKENS"
echo "  concurrency      : $CONCURRENCY"
echo "  base url         : $VLLM_BASE_URL"
echo "  python           : $($PY -c 'import sys;print(sys.executable)')"
echo "  resolver         : $([ "$NO_RESOLVER" = 1 ] && echo disabled || echo 'harness default (Bedrock Opus if AWS_BEARER_TOKEN_BEDROCK set, else reuse agent)')"
echo "=========================================================="

# --------------------------------------------------------------------------- #
# 0. Preflight
# --------------------------------------------------------------------------- #
if [ "$_n_worlds" -ne 72 ]; then
  echo "!! expected 72 worlds in $WORLDS_DIR, found $_n_worlds"; exit 1
fi

_n_dev=$(echo "$CUDA_DEVICES" | tr ',' '\n' | grep -c .)
if [ "$_n_dev" -ne "$TENSOR_PARALLEL" ]; then
  echo "!! TENSOR_PARALLEL=$TENSOR_PARALLEL but CUDA_DEVICES lists $_n_dev GPU(s). These must match."
  exit 1
fi

if [ "$SKIP_SERVE" != "1" ] && [ "$FORCE_GPU" != "1" ]; then
  for dev in $(echo "$CUDA_DEVICES" | tr ',' ' '); do
    used=$(nvidia-smi --id="$dev" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "${used:-0}" -gt 2000 ]; then
      echo "!! GPU $dev already has ${used}MiB in use -- someone (maybe you) is on it:"
      nvidia-smi --id="$dev" --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
      echo "   Free it, pick another card you own, or re-run with FORCE_GPU=1."
      exit 1
    fi
  done
fi

if [ "$SKIP_SERVE" != "1" ] && (command -v ss >/dev/null && ss -tln 2>/dev/null | grep -q ":${PORT} "); then
  echo "!! port $PORT is already bound. Pick another PORT."; exit 1
fi

# --------------------------------------------------------------------------- #
# 1. Serve
# --------------------------------------------------------------------------- #
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">> stopping vLLM server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$SKIP_SERVE" != "1" ]; then
  serve_args=(
    serve "$MODEL"
    --served-model-name "$SERVED_NAME"
    --port "$PORT"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$CONCURRENCY"
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

# Smoke-test one completion before committing to a 72-world run.
echo ">> smoke test..."
curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}],\"max_tokens\":2048}" \
  | head -c 600 || { echo "!! smoke test failed"; exit 1; }
echo

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
