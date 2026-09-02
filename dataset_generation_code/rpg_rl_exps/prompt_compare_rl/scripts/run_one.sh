#!/usr/bin/env bash
# Launch ONE prompt-comparison run. Executes INSIDE the `skyrl` container.
#
#   bash scripts/run_one.sh p1
#
# All configuration (env vars, SkyRL overrides, output paths) comes from
# prompt_compare_rl/config.py, so this script contains no settings of its own.
set -euo pipefail

PROMPT_ID="${1:?usage: run_one.sh <p1|p2|p3> [extra SkyRL overrides...]}"
shift

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_PARENT="$(dirname "$PKG_DIR")"
SKYRL_DIR="${PC_SKYRL_DIR:-/work/SkyRL}"

# config.py itself is importable with the stdlib only.
CFG_PY=("python3" "-m" "prompt_compare_rl.config")
run_cfg() { PYTHONPATH="$PKG_PARENT${PYTHONPATH:+:$PYTHONPATH}" "${CFG_PY[@]}" "$@"; }

# ---- materialize the per-run environment ------------------------------------------
ENV_LINES="$(run_cfg env "$PROMPT_ID")"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  export "$line"
done <<< "$ENV_LINES"

# ---- create this run's isolated directories ---------------------------------------
while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  mkdir -p "$value"
done <<< "$(run_cfg paths "$PROMPT_ID")"

RUN_DIR="$PC_RUN_DIR"
run_cfg manifest "$PROMPT_ID" > "$RUN_DIR/run_manifest.json"

# ---- required inputs must already exist -------------------------------------------
if [ ! -f "$RPG_SYSTEM_PROMPT_FILE" ]; then
  echo "missing prompt file $RPG_SYSTEM_PROMPT_FILE -- run build_dataset.py first" >&2
  exit 1
fi
TRAIN_PARQUET="$(python3 - "$PROMPT_ID" <<'PY'
import sys
from prompt_compare_rl.config import ExperimentConfig
print(ExperimentConfig().train_parquet(sys.argv[1]))
PY
)"
if [ ! -f "$TRAIN_PARQUET" ]; then
  echo "missing dataset $TRAIN_PARQUET -- run: python -m prompt_compare_rl.build_dataset" >&2
  exit 1
fi

# ---- W&B credentials (never echoed) ------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /work/wandb_key.txt ]; then
  set -a; . /work/wandb_key.txt; set +a
fi

# NUL-separated so paths and regex overrides survive without any shell re-quoting.
mapfile -t -d '' OVERRIDES < <(run_cfg args0 "$PROMPT_ID")
LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "[prompt_compare_rl] prompt=$PROMPT_ID gpu=$CUDA_VISIBLE_DEVICES wandb_run_id=$WANDB_RUN_ID"
echo "[prompt_compare_rl] run_dir=$RUN_DIR"
echo "[prompt_compare_rl] log=$LOG_FILE"

cd "$SKYRL_DIR"
uv run --isolated --extra fsdp -m prompt_compare_rl.main "${OVERRIDES[@]}" "$@" 2>&1 | tee "$LOG_FILE"
