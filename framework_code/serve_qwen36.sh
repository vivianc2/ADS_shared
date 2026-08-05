#!/usr/bin/env bash
# Serve Qwen/Qwen3.6-27B (thinking mode) via vLLM on an OpenAI-compatible endpoint.
#
# Usage:
#   ./serve_qwen36.sh                    # default GPUs 1,2 (TP=2)
#   ./serve_qwen36.sh 4,5,6,7            # use those GPUs (TP=4)
#   PORT=8001 ./serve_qwen36.sh 1,2      # custom port
#   MAX_LEN=65536 ./serve_qwen36.sh 1,2  # bigger context
#
# After it's up:
#   curl http://localhost:${PORT:-8000}/v1/models   # smoke check
#   then point your runner at it:
#     --scientist-backend openai \
#     --scientist-model Qwen/Qwen3.6-27B \
#     --scientist-base-url http://localhost:${PORT:-8000}/v1 \
#     --scientist-api-key EMPTY
#
# Notes:
#   - --reasoning-parser qwen3 routes <think> content into reasoning_content,
#     so .content is just the final answer (matches our XML answer extractor).
#   - --language-model-only skips the vision encoder (text-only task → more KV cache).
#   - Sampling params (temp=1.0, top_p=0.95, top_k=20) are baked into OpenAILLM
#     via the model preset — no need to set them here.

set -euo pipefail

GPUS="${1:-1,2}"
TP=$(echo "$GPUS" | awk -F, '{print NF}')
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"

# Activate the ADS conda env if not already.
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "ADS" ]]; then
  source /home/vivianchen/miniconda3/etc/profile.d/conda.sh
  conda activate ADS
fi

echo "Serving $MODEL on GPUs=$GPUS (TP=$TP) port=$PORT max_len=$MAX_LEN"

CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --reasoning-parser qwen3 \
  --language-model-only
