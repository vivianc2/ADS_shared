#!/usr/bin/env bash
# Base Qwen3.5-9B (or MODEL=<merged checkpoint>), one vLLM server per GPU, ADS env vLLM 0.19.1, max-model-len = the
# model maximum (262144) for full history. In git since 2026-10-24 (was ~/rpgrl/rl_v10/vllm/serve_base9b.sh).
#   bash serve_base9b.sh "2 3 4 5 6 7" 8201
# THINKING=off (default since 2026-10-24: the 9B RL line runs thinking off) serves with enable_thinking=false and no
# reasoning parser (all output is content); THINKING=on uses the reasoning parser qwen3 (the v10.2 gate setting).
# SERVE_FLAGS, if set, overrides both.
THINKING=${THINKING:-off}
case "$THINKING" in
  off) DEF_FLAGS='--default-chat-template-kwargs {"enable_thinking":false}' ;;
  on)  DEF_FLAGS='--reasoning-parser qwen3' ;;
  *) echo "THINKING must be on or off" >&2; exit 1 ;;
esac
GPUS=($1); P0=${2:-8201}; M=${MODEL:-/home/vivianchen/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a/}
NAME=${NAME:-qwen3.5-9b}; L=~/rpgrl/rl_v10/vllm
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}; p=$((P0+i))
  CUDA_VISIBLE_DEVICES=$g nohup ~/miniconda3/envs/ADS/bin/vllm serve $M --served-model-name $NAME --port $p \
    --max-model-len 262144 --gpu-memory-utilization 0.90 ${SERVE_FLAGS:-$DEF_FLAGS} --max-num-seqs 64 \
    --enable-prefix-caching > $L/${NAME}_gpu${g}_p${p}.log 2>&1 &
  echo "gpu $g port $p pid $!"
done
