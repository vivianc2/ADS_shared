#!/usr/bin/env bash
# THE entrypoint for the single-archetype experiment. Run this ON THE HOST.
#
#   bash scripts/launch.sh                  # build datasets -> pre-flight -> launch `easy`
#   bash scripts/launch.sh --dry-run        # print every check and exit, touching nothing
#   bash scripts/launch.sh --runs hard      # the other archetype, once easy is done
#   bash scripts/launch.sh --skip-build     # datasets already built (the resume path)
#   SA_GPUS=0,6 bash scripts/launch.sh      # different cards (default 0,4)
#
#   easy -> 96 dose_window worlds
#   hard -> 96 confounded_reversal worlds
#
# ONE run at a time, using ALL of SA_GPUS as one tensor-parallel job. Two concurrent
# single-GPU runs need ~165 GiB of host RAM against this box's 96 GiB ceiling; see
# docs/gpu_sizing.md. Asking for both runs here is therefore sequential by construction.
#
# Re-running this command after a crash RESUMES from the newest checkpoint
# (trainer.resume_mode=latest) into the SAME W&B run id, so the chart continues instead
# of forking. --skip-build is the fast path for that.
#
# The run gets its own Ray cluster (RAY_ADDRESS=local + private RAY_TMPDIR), checkpoint /
# export / log directory, LoRA-sync directory, compiler caches and W&B run id, so a later
# run of the other archetype cannot disturb this one's outputs.
set -euo pipefail

CONTAINER="${SA_CONTAINER:-skyrl-pc}"
HOST_PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The container mounts the repo root at /work.
HOST_REPO_ROOT="${SA_HOST_REPO_ROOT:-/home/sts004/rpg}"
CONTAINER_PKG_DIR="/work${HOST_PKG_DIR#$HOST_REPO_ROOT}"
PKG_PARENT="$(dirname "$HOST_PKG_DIR")"

SKIP_BUILD=0
SKIP_PREFLIGHT=0
DRY_RUN=0
RUNS="${SA_RUNS:-easy}"

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build)     SKIP_BUILD=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --dry-run)        DRY_RUN=1 ;;
    --runs)           RUNS="$2"; shift ;;
    -h|--help)        sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# Forward the experiment-scoping vars so the container-side steps resolve the same paths
# and settings as the host-side pre-flight. `docker exec -e VAR` (no value) passes the
# variable through only when it is set in this shell.
FORWARD=(-e SA_GPUS -e SA_EXP_TAG -e SA_CKPT_ROOT -e SA_CKPT_INTERVAL -e SA_EVAL_INTERVAL
         -e SA_MAX_STEPS -e SA_GROUP_SIZE -e SA_EVAL_N -e SA_GPU_MEM_UTIL -e SA_MODEL
         -e SA_OUT_ROOT -e SA_WANDB_PROJECT -e SA_SEED -e SA_LR
         # The outer cgroup limit is only readable from the host; inside the container
         # /sys/fs/cgroup/memory.max is "max" and /proc/meminfo is the physical machine.
         -e SA_CGROUP_MEMORY_BYTES)
in_container() { docker exec "${FORWARD[@]}" "$CONTAINER" bash -lc "$1"; }

cfg() { PYTHONPATH="$PKG_PARENT" python3 -m single_arch_rl.config "$@"; }

echo "== single_arch_rl launcher =="
echo "container:       $CONTAINER"
echo "package (host):  $HOST_PKG_DIR"
echo "package (cont.): $CONTAINER_PKG_DIR"
echo "runs:            $RUNS  (sequential: they share the GPUs)"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container '$CONTAINER' is not running." >&2
  echo "       Create it with prompt_compare_rl/container/create_skyrl_pc.sh (it mounts /data)." >&2
  exit 1
fi

# ---- resource budget (single source of truth: config.py) ------------------------------
eval "$(cfg budget | sed 's/^/SA_BUDGET_/')"
njobs="$(echo "$RUNS" | wc -w)"
echo "steps:           $SA_BUDGET_MAX_STEPS (checkpoint every $SA_BUDGET_CKPT_INTERVAL)"
echo "GPUs:            ${SA_BUDGET_GPUS} (one tensor-parallel job)"

# ---- pre-flight: GPUs -------------------------------------------------------------------
# The job needs, ON EVERY CARD, vLLM's share (gpu_memory_utilization x total) plus its
# share of the fp32 policy, gradients, activations and the CUDA graph pool. A GPU that is
# already busy OOMs ~20 minutes into startup rather than failing fast, so check first.
# Other containers' allocations do not appear in this container's process table, so
# compare free framebuffer rather than the process list.
gpu_busy=0
for gpu in ${SA_BUDGET_GPUS//,/ }; do
  read -r total_mib free_mib <<< "$(nvidia-smi --id="$gpu" \
      --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ',')"
  need_mib="$(awk -v t="${total_mib:-0}" -v u="$SA_BUDGET_GPU_MEM_UTIL" -v h="$SA_BUDGET_GPU_GB_HEADROOM" \
              'BEGIN{printf "%d", t*u + h*1024}')"
  echo "GPU $gpu: ${free_mib:-?} MiB free, the job needs about ${need_mib} MiB here"
  if [ "${free_mib:-0}" -lt "$need_mib" ]; then
    echo "WARNING: GPU $gpu does not have enough free memory." >&2
    gpu_busy=1
  fi
done
if [ "$gpu_busy" = 1 ]; then
  echo "         Wait for the card to free up, pick others with SA_GPUS=a,b," >&2
  echo "         lower SA_GPU_MEM_UTIL, or set SA_FORCE=1 to launch anyway." >&2
  if [ "${SA_FORCE:-0}" != "1" ] && [ "$DRY_RUN" = 0 ]; then exit 1; fi
fi

# ---- pre-flight: host RAM ----------------------------------------------------------------
# Read the EFFECTIVE cgroup limit, not /proc/meminfo: Docker here runs inside an outer LXC
# cgroup and /proc/meminfo reports the physical machine, not what this tree may use.
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
  # memory.current counts the page cache, which is reclaimable under pressure -- after a
  # dataset build or a model load that is tens of GB of cached file pages that would be
  # evicted rather than cause an OOM. Charge only the unreclaimable part (anon + kernel).
  mem_cache_gb="$(awk '$1=="file" {printf "%d", $2/1073741824}' /sys/fs/cgroup/memory.stat 2>/dev/null || echo 0)"
  mem_committed_gb=$(( mem_used_gb - ${mem_cache_gb:-0} ))
  [ "$mem_committed_gb" -lt 0 ] && mem_committed_gb=0
  avail_gb=$(( mem_limit_gb - mem_committed_gb ))
  # Forward the real limit: Ray cannot discover it from inside the container.
  export SA_CGROUP_MEMORY_BYTES=$(( mem_limit_gb * 1073741824 ))
  echo "host RAM: ${mem_limit_gb} GB cgroup limit, ${mem_committed_gb} GB committed" \
       "(+${mem_cache_gb:-0} GB reclaimable page cache), ${avail_gb} GB free"
else
  avail_gb="$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)"
  echo "host RAM: no finite cgroup limit found; MemAvailable = ${avail_gb} GB"
fi
# Host RAM is this experiment's binding constraint, not GPU memory: vLLM's shared sleep
# buffer holds the base weights (LoRA sync forbids dropping them) and the FSDP ranks keep
# the fp32 policy on the host. Measured; see docs/gpu_sizing.md.
need_gb=$SA_BUDGET_HOST_GB_PER_JOB
echo "the run needs roughly ${need_gb} GB of host RAM at peak"
if [ "${avail_gb:-0}" -lt "$need_gb" ]; then
  echo "WARNING: not enough free host RAM for this run." >&2
  echo "         Free memory on this container tree, or set SA_FORCE=1 to proceed anyway." >&2
  if [ "${SA_FORCE:-0}" != "1" ] && [ "$DRY_RUN" = 0 ]; then exit 1; fi
fi

# ---- pre-flight: checkpoint volume ---------------------------------------------------------
# Requirement (3): the container filesystem has ~49 GB free; a checkpoint is ~19 GB. They
# must land on the 31 TB /data volume, which the skyrl-pc container mounts at the same path.
if [ "${SA_BUDGET_CKPT_INTERVAL:-0}" -gt 0 ]; then
  ckpt_root="$SA_BUDGET_CKPT_ROOT"
  ckpt_need_gb=$(( njobs * SA_BUDGET_CKPT_GB_PER_RUN ))
  if ! mkdir -p "$ckpt_root" 2>/dev/null || [ ! -w "$ckpt_root" ]; then
    echo "ERROR: checkpoint root is not writable from the host: $ckpt_root" >&2
    echo "       Point SA_CKPT_ROOT elsewhere on /data, or set SA_CKPT_INTERVAL=0." >&2
    if [ "$DRY_RUN" = 0 ]; then exit 1; fi
  else
    ckpt_free_gb="$(df -BG --output=avail "$ckpt_root" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "checkpoints: ${ckpt_root} has ${ckpt_free_gb:-?} GB free, runs need about ${ckpt_need_gb} GB"
    if [ "${ckpt_free_gb:-0}" -lt "$ckpt_need_gb" ]; then
      echo "WARNING: not enough space for checkpoints at $ckpt_root." >&2
      if [ "${SA_FORCE:-0}" != "1" ] && [ "$DRY_RUN" = 0 ]; then exit 1; fi
    fi
  fi
fi

# ---- pre-flight: W&B credentials --------------------------------------------------------
if [ ! -r "$HOST_REPO_ROOT/wandb_key.txt" ]; then
  echo "ERROR: $HOST_REPO_ROOT/wandb_key.txt is missing; the runs log to W&B." >&2
  if [ "$DRY_RUN" = 0 ]; then exit 1; fi
fi

for run in $RUNS; do
  env_lines="$(PYTHONPATH="$PKG_PARENT" python3 -m single_arch_rl.config env "$run")"
  wid="$(echo "$env_lines" | sed -n 's/^WANDB_RUN_ID=//p')"
  arch="$(echo "$env_lines" | sed -n 's/^SA_TRAIN_ARCHETYPE=//p')"
  echo "run '$run': archetype=$arch  wandb_run_id=$wid (resumed on relaunch)"
done

if [ "$DRY_RUN" = 1 ]; then
  echo "(dry run) would build datasets, run the CPU checks, then launch: $RUNS"
  exit 0
fi

# ---- 1. datasets --------------------------------------------------------------------------
if [ "$SKIP_BUILD" = 0 ]; then
  echo "== building the single-archetype datasets =="
  in_container "cd $CONTAINER_PKG_DIR && bash scripts/in_container.sh python -m single_arch_rl.build_dataset"
fi

# ---- 2. CPU checks (unit tests + full launch path without Ray) ----------------------------
if [ "$SKIP_PREFLIGHT" = 0 ]; then
  echo "== CPU checks =="
  in_container "cd $CONTAINER_PKG_DIR && bash scripts/run_tests.sh"
fi

# ---- 3. launch -----------------------------------------------------------------------------
# Sequential by construction: the runs share every GPU, so a second one cannot start until
# the first exits. With a single run (the default) this is one backgrounded docker exec.
first=1
for run in $RUNS; do
  if [ "$first" = 0 ]; then
    echo "== waiting for the previous run to finish before starting $run =="
    while docker exec "$CONTAINER" pgrep -f "single_arch_rl.main" >/dev/null 2>&1; do sleep 60; done
  fi
  echo "== launching $run in the background =="
  docker exec -d "${FORWARD[@]}" "$CONTAINER" bash -lc "cd $CONTAINER_PKG_DIR && bash scripts/run_one.sh $run"
  first=0
done

EXP_DIR_HOST="${SA_BUDGET_EXP_DIR/\/work/$HOST_REPO_ROOT}"
echo
echo "All requested runs started. Follow them with:"
echo "  tail -f \$(ls -t $EXP_DIR_HOST/runs/*/logs/train_*.log | head -1)"
echo "  docker exec $CONTAINER bash -lc 'cd $CONTAINER_PKG_DIR && bash scripts/in_container.sh python -m single_arch_rl.report_eval'"
echo "W&B: project $(cfg env easy | sed -n 's/^WANDB_PROJECT=//p'), runs $(for r in $RUNS; do cfg env "$r" | sed -n 's/^WANDB_RUN_ID=//p' | tr '\n' ' '; done)"
