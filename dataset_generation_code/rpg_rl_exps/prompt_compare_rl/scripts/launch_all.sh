#!/usr/bin/env bash
# One command that starts all three prompt runs (requirement 10).
#
# Run this ON THE HOST. It drives the already-running `skyrl` docker container:
#
#   bash scripts/launch_all.sh                 # build datasets, smoke test, launch p1/p2/p3
#   bash scripts/launch_all.sh --sequential    # same, but one run at a time (low host RAM)
#   bash scripts/launch_all.sh --skip-build    # datasets already built
#   bash scripts/launch_all.sh --dry-run       # print what would happen and exit
#   PC_GPUS=5,6,7 bash scripts/launch_all.sh   # use other cards (default 0,1,2)
#
# p1 -> GPU 0, p2 -> GPU 1, p3 -> GPU 2 by default (PC_GPUS overrides, one card per
# run). Every job gets its own Ray cluster
# (RAY_ADDRESS=local + private RAY_TMPDIR), checkpoint / export / log directory,
# LoRA-sync directory, compiler caches and W&B run id, so the jobs cannot interfere.
set -euo pipefail

CONTAINER="${PC_CONTAINER:-skyrl}"
HOST_PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The container mounts the repo root at /work.
HOST_REPO_ROOT="${PC_HOST_REPO_ROOT:-/home/sts004/rpg}"
CONTAINER_PKG_DIR="/work${HOST_PKG_DIR#$HOST_REPO_ROOT}"

SEQUENTIAL=0
SKIP_BUILD=0
SKIP_SMOKE=0
DRY_RUN=0
STAGGER="${PC_STAGGER_SECONDS:-600}"
PROMPTS="${PC_PROMPTS:-p1 p2 p3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --sequential) SEQUENTIAL=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --dry-run)    DRY_RUN=1 ;;
    --stagger)    STAGGER="$2"; shift ;;
    --prompts)    PROMPTS="$2"; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# Forward the experiment-scoping vars so the container-side steps resolve the same
# paths and settings as the host-side pre-flight. `docker exec -e VAR` (no value) passes
# the variable through only when it is set in this shell.
in_container() {
  docker exec -e PC_GPUS -e PC_EXP_TAG -e PC_CKPT_ROOT -e PC_CKPT_INTERVAL \
    "$CONTAINER" bash -lc "$1"
}

echo "== prompt_compare_rl launcher =="
echo "container:       $CONTAINER"
echo "package (host):  $HOST_PKG_DIR"
echo "package (cont.): $CONTAINER_PKG_DIR"
echo "prompts:         $PROMPTS"
echo "GPUs:            ${PC_GPUS:-0,1,2 (default)}"
echo "mode:            $([ "$SEQUENTIAL" = 1 ] && echo sequential || echo "concurrent (stagger ${STAGGER}s)")"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container '$CONTAINER' is not running." >&2
  exit 1
fi

# ---- resource budget (single source of truth: config.py) ----------------------------
PKG_PARENT="$(dirname "$HOST_PKG_DIR")"
eval "$(PYTHONPATH="$PKG_PARENT" python3 -m prompt_compare_rl.config budget p1 | sed 's/^/PC_BUDGET_/')"
njobs="$(echo "$PROMPTS" | wc -w)"

# ---- pre-flight: GPUs ---------------------------------------------------------------
# A job needs vLLM's share (gpu_memory_utilization x total) plus headroom for the bf16
# policy, gradients, activations and the CUDA graph pool. A GPU that is already busy OOMs
# ~20 minutes into startup rather than failing fast, so check before launching.
# The mapping comes from config.py (PC_GPUS), so it is defined in exactly one place.
gpu_for_prompt() {
  PYTHONPATH="$PKG_PARENT" python3 -m prompt_compare_rl.config env "$1" \
    | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
}

gpu_busy=0
for pid in $PROMPTS; do
  gpu="$(gpu_for_prompt "$pid")"
  [ -n "$gpu" ] || { echo "could not resolve a GPU for $pid" >&2; exit 2; }
  read -r total_mib free_mib <<< "$(nvidia-smi --id="$gpu" \
      --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ',')"
  need_mib="$(awk -v t="${total_mib:-0}" -v u="$PC_BUDGET_GPU_MEM_UTIL" -v h="$PC_BUDGET_GPU_GB_HEADROOM" \
              'BEGIN{printf "%d", t*u + h*1024}')"
  echo "GPU $gpu ($pid): ${free_mib:-?} MiB free, run needs about ${need_mib} MiB"
  if [ "${free_mib:-0}" -lt "$need_mib" ]; then
    echo "WARNING: GPU $gpu does not have enough free memory for $pid." >&2
    gpu_busy=1
  fi
done
if [ "$gpu_busy" = 1 ]; then
  echo "         Wait for the GPUs to free up, lower PC_GPU_MEM_UTIL, or set PC_FORCE=1." >&2
  if [ "${PC_FORCE:-0}" != "1" ] && [ "$DRY_RUN" = 0 ]; then
    exit 1
  fi
fi

# ---- pre-flight: host RAM -----------------------------------------------------------
# Read the EFFECTIVE cgroup limit, not /proc/meminfo. Docker here runs inside an outer LXC
# cgroup, and /proc/meminfo inside the container reports the physical machine (377 GB)
# rather than the ~96 GiB this container tree may actually use.
effective_memory_limit_gb() {
  local rel dir value best=""
  rel="$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup 2>/dev/null)"
  dir="/sys/fs/cgroup${rel:-}"
  while : ; do
    if [ -r "$dir/memory.max" ]; then
      value="$(tr -d '[:space:]' <"$dir/memory.max")"
      case "$value" in
        ''|*[!0-9]*) : ;;                                  # "max" == no limit
        *) if [ -z "$best" ] || [ "$value" -lt "$best" ]; then best="$value"; fi ;;
      esac
    fi
    [ "$dir" = "/sys/fs/cgroup" ] && break
    dir="${dir%/*}"
    [ -z "$dir" ] && dir="/sys/fs/cgroup"
  done
  [ -n "$best" ] && awk -v b="$best" 'BEGIN{printf "%d", b/1073741824}'
}

mem_limit_gb="$(effective_memory_limit_gb || true)"
if [ -n "${mem_limit_gb:-}" ] && [ -r /sys/fs/cgroup/memory.current ]; then
  mem_used_gb="$(awk '{printf "%d", $1/1073741824}' /sys/fs/cgroup/memory.current)"
  avail_gb=$(( mem_limit_gb - mem_used_gb ))
  echo "host RAM: ${mem_limit_gb} GB cgroup limit, ${mem_used_gb} GB in use, ${avail_gb} GB free"
else
  avail_gb="$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)"
  echo "host RAM: no finite cgroup limit found; MemAvailable = ${avail_gb} GB"
fi
# Each concurrent job holds a bf16 policy in host RAM during generation; only ONE can be
# inside save_checkpoints() at a time (main.py holds a lock shared by the three runs).
need_gb=$(( njobs * PC_BUDGET_HOST_GB_PER_JOB + PC_BUDGET_CKPT_SPIKE_GB ))
echo "concurrent runs need roughly ${need_gb} GB (${njobs} x ${PC_BUDGET_HOST_GB_PER_JOB} GB + ${PC_BUDGET_CKPT_SPIKE_GB} GB checkpoint spike)"
if [ "$SEQUENTIAL" = 0 ] && [ "${avail_gb:-0}" -lt "$need_gb" ]; then
  echo "WARNING: not enough free host RAM for ${njobs} concurrent runs." >&2
  echo "         Options: --sequential, PC_CKPT_INTERVAL=8 (one save per run)," >&2
  echo "         or PC_FORCE=1 to proceed anyway." >&2
  if [ "${PC_FORCE:-0}" != "1" ] && [ "$DRY_RUN" = 0 ]; then
    exit 1
  fi
fi

# ---- pre-flight: checkpoint volume ---------------------------------------------------
if [ "${PC_BUDGET_CKPT_INTERVAL:-0}" -gt 0 ]; then
  ckpt_root="$PC_BUDGET_CKPT_ROOT"
  ckpt_need_gb=$(( njobs * 19 * (8 / PC_BUDGET_CKPT_INTERVAL) ))
  if ! mkdir -p "$ckpt_root" 2>/dev/null || [ ! -w "$ckpt_root" ]; then
    echo "ERROR: checkpoint root is not writable from the host: $ckpt_root" >&2
    echo "       Create the container with the /data mount (container/create_skyrl_pc.sh)," >&2
    echo "       point PC_CKPT_ROOT elsewhere, or set PC_CKPT_INTERVAL=0." >&2
    if [ "$DRY_RUN" = 0 ]; then exit 1; fi
  else
    ckpt_free_gb="$(df -BG --output=avail "$ckpt_root" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "checkpoints: ${ckpt_root} has ${ckpt_free_gb:-?} GB free, run needs about ${ckpt_need_gb} GB"
    if [ "${ckpt_free_gb:-0}" -lt "$ckpt_need_gb" ]; then
      echo "WARNING: not enough space for checkpoints at $ckpt_root." >&2
      if [ "${PC_FORCE:-0}" != "1" ] && [ "$DRY_RUN" = 0 ]; then exit 1; fi
    fi
  fi
fi

if [ "$DRY_RUN" = 1 ]; then
  echo "(dry run) would build datasets, run the smoke test, then launch: $PROMPTS"
  exit 0
fi

# ---- 1. datasets (prompt injection) --------------------------------------------------
if [ "$SKIP_BUILD" = 0 ]; then
  echo "== building per-prompt datasets =="
  in_container "cd $CONTAINER_PKG_DIR && bash scripts/in_container.sh python -m prompt_compare_rl.build_dataset"
fi

# ---- 2. smoke test -------------------------------------------------------------------
if [ "$SKIP_SMOKE" = 0 ]; then
  echo "== smoke test =="
  in_container "cd $CONTAINER_PKG_DIR && bash scripts/in_container.sh python -m prompt_compare_rl.smoke_test"
fi

# ---- 3. launch -----------------------------------------------------------------------
first=1
for pid in $PROMPTS; do
  if [ "$SEQUENTIAL" = 1 ]; then
    echo "== running $pid (foreground) =="
    docker exec -e PC_GPUS -e PC_EXP_TAG -e PC_CKPT_ROOT -e PC_CKPT_INTERVAL \
      "$CONTAINER" bash -lc "cd $CONTAINER_PKG_DIR && bash scripts/run_one.sh $pid"
  else
    if [ "$first" = 0 ] && [ "${STAGGER:-0}" -gt 0 ]; then
      echo "-- staggering ${STAGGER}s before $pid --"
      sleep "$STAGGER"
    fi
    echo "== launching $pid in the background =="
    docker exec -d -e PC_GPUS -e PC_EXP_TAG -e PC_CKPT_ROOT -e PC_CKPT_INTERVAL \
      "$CONTAINER" bash -lc "cd $CONTAINER_PKG_DIR && bash scripts/run_one.sh $pid"
    first=0
  fi
done

echo
echo "All requested runs started. Follow them with:"
echo "  docker exec $CONTAINER bash -lc 'tail -f \$(ls -t /work/data/rpg_rl_exps/prompt_compare_rl/*/runs/*/logs/train_*.log | head -3)'"
echo "  docker exec $CONTAINER bash -lc 'cd $CONTAINER_PKG_DIR && bash scripts/in_container.sh python -m prompt_compare_rl.report_eval'"
