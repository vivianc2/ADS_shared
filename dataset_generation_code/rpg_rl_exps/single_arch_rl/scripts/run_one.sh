#!/usr/bin/env bash
# Launch ONE single-archetype run. Executes INSIDE the skyrl-pc container.
#
#   bash scripts/run_one.sh easy      # 96 dose_window worlds,         GPU 0
#   bash scripts/run_one.sh hard      # 96 confounded_reversal worlds, GPU 4
#
# Any extra arguments are appended as SkyRL overrides. All configuration (env vars,
# overrides, output paths) comes from single_arch_rl/config.py, so this script holds no
# settings of its own.
#
# Re-running the SAME command after a crash resumes from the newest checkpoint
# (trainer.resume_mode=latest) and reuses the same W&B run id, so the chart continues
# rather than forking.
set -euo pipefail

RUN_ID="${1:?usage: run_one.sh <easy|hard> [extra SkyRL overrides...]}"
shift

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_PARENT="$(dirname "$PKG_DIR")"
SKYRL_DIR="${SA_SKYRL_DIR:-/work/SkyRL}"

# config.py is importable with the stdlib only.
run_cfg() { PYTHONPATH="$PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}" python3 -m single_arch_rl.config "$@"; }

# ---- materialize the per-run environment ---------------------------------------------
while IFS= read -r line; do
  [ -z "$line" ] && continue
  export "$line"
done <<< "$(run_cfg env "$RUN_ID")"

# expandable_segments is incompatible with the CuMemAllocator pool vLLM re-maps on every
# colocate_all sleep/wake cycle; main.py's pre-flight refuses to start with it set.
unset PYTORCH_CUDA_ALLOC_CONF || true

# ---- create this run's isolated directories -------------------------------------------
while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  mkdir -p "$value"
done <<< "$(run_cfg paths "$RUN_ID")"

RUN_DIR="$SA_RUN_DIR"
run_cfg manifest "$RUN_ID" > "$RUN_DIR/run_manifest.json"

# ---- required inputs must already exist ------------------------------------------------
TRAIN_PARQUET="$(python3 - "$RUN_ID" <<'PY'
import sys
from single_arch_rl.config import ExperimentConfig
print(ExperimentConfig().train_parquet(sys.argv[1]))
PY
)"
if [ ! -f "$TRAIN_PARQUET" ]; then
  echo "missing dataset $TRAIN_PARQUET -- run: python -m single_arch_rl.build_dataset" >&2
  exit 1
fi

# ---- W&B credentials (never echoed) ------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /work/wandb_key.txt ]; then
  set -a; . /work/wandb_key.txt; set +a
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is unset and /work/wandb_key.txt did not provide one." >&2
  exit 1
fi

# NUL-separated so paths and regex overrides survive without any shell re-quoting.
mapfile -t -d '' OVERRIDES < <(run_cfg args0 "$RUN_ID")
LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "[single_arch_rl] run=$RUN_ID archetype=$SA_TRAIN_ARCHETYPE gpu=$CUDA_VISIBLE_DEVICES"
echo "[single_arch_rl] wandb=$WANDB_PROJECT/$WANDB_RUN_ID (resume=$WANDB_RESUME)"
echo "[single_arch_rl] run_dir=$RUN_DIR"
echo "[single_arch_rl] log=$LOG_FILE"

cd "$SKYRL_DIR"
# The SkyRL venv is used directly (not `uv run --isolated`) so the project-local FLA 0.5.2
# and vLLM partial-wake overlays on PYTHONPATH are the ones that load; an isolated build
# would resolve fla from its own tree instead.
"$SKYRL_DIR/.venv/bin/python" -m single_arch_rl.main "${OVERRIDES[@]}" "$@" 2>&1 | tee "$LOG_FILE"
