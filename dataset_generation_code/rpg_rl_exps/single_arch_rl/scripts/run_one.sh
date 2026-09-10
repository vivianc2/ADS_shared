#!/usr/bin/env bash
# Launch ONE single-archetype run. Executes INSIDE the skyrl-pc container.
#
#   bash scripts/run_one.sh easy      # 96 dose_window worlds,         GPUs 5,6 (default)
#   bash scripts/run_one.sh hard      # 96 confounded_reversal worlds, GPUs 5,6 (default)
#
# Any extra arguments are appended as SkyRL overrides. All configuration (env vars,
# overrides, output paths) comes from single_arch_rl/config.py, so this script holds no
# settings of its own.
#
# Re-running the same command after a crash resumes from the newest checkpoint and opens
# a fresh W&B attempt in the same group, preventing rollback-step rejection.
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

# vLLM's multiprocessing workers can outlive Ray after both successful and failed runs.
# In this long-lived container they are reparented to PID 1 and retain their CUDA/shared-
# memory allocations, which can starve the next sequential run. Clean only orphaned
# processes owned by this uid whose inherited experiment tag exactly matches this run.
cleanup_run_orphans() {
  local run_status=$?
  local pid ppid
  local -a tagged_orphans=()
  local -a survivors=()

  trap - EXIT
  set +e
  while read -r pid ppid; do
    [ "$ppid" = "1" ] || continue
    [ -r "/proc/$pid/environ" ] || continue
    if { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null \
        | grep -Fx -- "SA_EXP_TAG=$SA_EXP_TAG" >/dev/null; then
      tagged_orphans+=("$pid")
    fi
  done < <(ps -u "$(id -u)" -o pid=,ppid=)

  if [ "${#tagged_orphans[@]}" -gt 0 ]; then
    echo "[single_arch_rl] cleaning ${#tagged_orphans[@]} tagged orphan process(es)"
    kill -TERM "${tagged_orphans[@]}" 2>/dev/null || true
    for _ in {1..10}; do
      survivors=()
      for pid in "${tagged_orphans[@]}"; do
        kill -0 "$pid" 2>/dev/null && survivors+=("$pid")
      done
      [ "${#survivors[@]}" -eq 0 ] && break
      sleep 1
    done
    [ "${#survivors[@]}" -eq 0 ] || kill -KILL "${survivors[@]}" 2>/dev/null || true
  fi

  exit "$run_status"
}
trap cleanup_run_orphans EXIT

# Every process launch gets a new W&B history. A checkpoint resume can therefore start
# again at an earlier trainer/global step without W&B rejecting it as non-monotonic.
# Attempts remain together in the stable group emitted by config.py.
WANDB_ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export WANDB_RUN_ID="${WANDB_RUN_ID}-${WANDB_ATTEMPT}"
export WANDB_NAME="${WANDB_NAME}-${WANDB_ATTEMPT}"
export WANDB_RESUME=never

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
