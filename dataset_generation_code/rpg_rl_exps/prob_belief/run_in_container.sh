#!/usr/bin/env bash
set -euo pipefail

# Host-side entrypoint for the repository's Python 3.12/CUDA environment.
# Override only when the established container has a different local name.
CONTAINER_NAME="${PROB_BELIEF_CONTAINER:-skyrl-pc}"
CONTAINER_DIR="/work/ADS_shared/dataset_generation_code/rpg_rl_exps/prob_belief"
PYTHON_BIN="/work/SkyRL/.venv/bin/python"
VLLM_BIN="/work/SkyRL/.venv/bin/vllm"

if [ "$#" -eq 0 ]; then
  echo "usage: bash run_in_container.sh <inspect|all|aggregate|validate> [arguments...]" >&2
  exit 2
fi

args=("$@")
if [ "$1" = "all" ]; then
  has_vllm_executable=false
  for arg in "$@"; do
    if [ "$arg" = "--vllm-executable" ]; then
      has_vllm_executable=true
      break
    fi
  done
  if [ "$has_vllm_executable" = false ]; then
    args+=(--vllm-executable "$VLLM_BIN")
  fi
fi

exec docker exec \
  --workdir "$CONTAINER_DIR" \
  --env HOME=/work/home \
  --env HF_HOME=/work/hf_cache \
  --env XDG_CACHE_HOME=/work/home/.cache \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env RPG_PROTO=rpg_v9 \
  --env RPG_SYNERGY_SOFT=20 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$CONTAINER_NAME" \
  "$PYTHON_BIN" run_experiment.py "${args[@]}"
