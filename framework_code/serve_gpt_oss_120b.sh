#!/usr/bin/env bash
# Serve openai/gpt-oss-120b via vLLM on an OpenAI-compatible endpoint.
#
# Usage:
#   ./serve_gpt_oss_120b.sh                # default GPUs 4,5,6,7 (TP=4)
#   ./serve_gpt_oss_120b.sh 0,1,2,3,4,5,6,7  # full 8-GPU split (TP=8)
#   PORT=8002 ./serve_gpt_oss_120b.sh 4,5,6,7
#   MAX_LEN=65536 ./serve_gpt_oss_120b.sh 4,5,6,7
#
# After it's up:
#   curl http://localhost:${PORT:-8002}/v1/models
#   then point your runner at it:
#     --scientist-backend openai \
#     --scientist-model openai/gpt-oss-120b \
#     --scientist-base-url http://localhost:${PORT:-8002}/v1 \
#     --scientist-api-key EMPTY
#
# Notes:
#   - gpt-oss-120b ships in MXFP4. On L40S (Ada, no native FP4) vLLM dequantizes
#     via Marlin/equivalent kernels. TP=4 is the practical minimum on L40S 48GB;
#     drop to TP=8 if you OOM.
#   - --reasoning-parser openai_gptoss separates harmony reasoning channel from
#     the final-answer channel into reasoning_content (kept out of .content).
#   - Sampling params come from the gpt-oss preset in OpenAILLM.

set -euo pipefail

GPUS="${1:-4,5,6,7}"
TP=$(echo "$GPUS" | awk -F, '{print NF}')
PORT="${PORT:-8002}"
MAX_LEN="${MAX_LEN:-32768}"
MODEL="${MODEL:-openai/gpt-oss-120b}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "ADS" ]]; then
  source /home/vivianchen/miniconda3/etc/profile.d/conda.sh
  conda activate ADS
fi

echo "Serving $MODEL on GPUs=$GPUS (TP=$TP) port=$PORT max_len=$MAX_LEN"

CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --reasoning-parser openai_gptoss
