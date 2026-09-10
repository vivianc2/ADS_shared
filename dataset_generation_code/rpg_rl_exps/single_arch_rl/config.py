#!/usr/bin/env python3
"""Single source of truth for the single-archetype RL experiment.

Two GRPO runs that differ in **one thing only**: which archetype their 96 training
worlds are drawn from.

    easy -> dose_window            (96 worlds)
    hard -> confounded_reversal    (96 worlds)

Model, seeds, reward, validation set (``rpg_v9/data_v9_deleaked/validation_small.parquet``)
and every hyper-parameter are identical, so a difference in the learning curves is
attributable to the training archetype. Everything that MUST differ (archetype, GPU,
output directories, Ray temp dir, LoRA-sync dir, W&B run id) is derived from the run id
alone, which ``tests/test_config.py`` asserts.

The bash launchers hold no settings of their own; they call

    python -m single_arch_rl.config env   <run_id>    # KEY=VALUE lines
    python -m single_arch_rl.config args0 <run_id>    # NUL-separated SkyRL overrides

so there is exactly one copy of the configuration.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------------------
# Fixed experiment constants (identical across the two runs)
# --------------------------------------------------------------------------------------

RUN_IDS: Tuple[str, ...] = ("easy", "hard")

#: Requirement (2): the ONLY difference between the runs.
TRAIN_ARCHETYPE: Dict[str, str] = {
    "easy": "dose_window",
    "hard": "confounded_reversal",
}

#: Spec (1): 96 training worlds per run, drawn from that run's archetype only.
TRAIN_WORLDS_PER_RUN = 96

#: The GPUs given to THE run being launched -- all of them, to one run.
#:
#: The spec asked for two concurrent runs, one GPU each. That is not possible on this
#: box and no GPU-memory setting changes it: the constraint is HOST memory. A trainable
#: policy is materialized in fp32 (SkyRL passes ``bf16=policy.inference_only_init``,
#: false for the policy), and on a single GPU FSDP shards nothing, so one run holds a
#: 37.6 GB fp32 policy plus vLLM's ~23 GB sleep buffer -- LoRA weight sync forces sleep
#: level 1, which must keep the base weights. Measured peak for one single-GPU run:
#: 95.6 GiB against a 96 GiB cgroup ceiling. Two would need ~165 GiB.
#: With both GPUs on one run, FSDP shards the policy across the two ranks and rank 1
#: uses meta-init, which is the shape that fits. See docs/gpu_sizing.md.
DEFAULT_GPUS: Tuple[int, ...] = (5, 6)

#: Runs launched by default. The two runs share the GPUs now, so they are sequential;
#: `hard` is launched the same way once `easy` finishes.
DEFAULT_RUNS: Tuple[str, ...] = ("easy",)

#: ``train_batch_size == policy_mini_batch_size`` with ``update_epochs_per_batch == 1``
#: makes one SkyRL global step exactly one optimizer step, so "checkpoint every 4 steps"
#: is unambiguous. 96 worlds / 2 = 48 steps per epoch.
TRAIN_BATCH_SIZE = 2
POLICY_MINI_BATCH_SIZE = 2
UPDATE_EPOCHS_PER_BATCH = 1
STEPS_PER_EPOCH = TRAIN_WORLDS_PER_RUN // TRAIN_BATCH_SIZE  # 48

MODEL_PATH = "Qwen/Qwen3.5-9B"
RPG_PROTO = "rpg_v9"

ENV_ID = "rpg_single_arch"
ENV_ENTRY_POINT = "single_arch_rl.sky_env:SingleArchRPGEnv"

# Fresh output/checkpoint namespace for the batch-2, 30-step run. The completed sarl_v1
# tree is never reused or overwritten.
DEFAULT_EXP_TAG = "sarl_v3_bs2_s30_ckpt4"
DEFAULT_OUT_ROOT = "/work/data/rpg_rl_exps/single_arch_rl"
#: Requirement (3): the container filesystem has ~49 GB free and one checkpoint is
#: ~19 GB. Checkpoints go to the 31 TB NFS volume, mounted at the SAME path inside the
#: `skyrl-pc` container, so a path printed in a log works verbatim from the host shell.
DEFAULT_CKPT_ROOT = "/data/rpg_rl_exps/single_arch_rl"
DEFAULT_RPG_SRC = "/work/ADS_shared/dataset_generation_code"
DEFAULT_SKYRL_DIR = "/work/SkyRL"
DEFAULT_WANDB_PROJECT = "rpg_single_arch"

#: Requirement (2): both training sets are sampled out of the committed de-leaked v9
#: train split, and BOTH runs evaluate on the same validation_small.parquet.
SOURCE_TRAIN = "rpg_v9/data_v9_deleaked/train.parquet"
SOURCE_VAL = "rpg_v9/data_v9_deleaked/validation_small.parquet"

#: The 8 non-reserved skins (splits.py). The 96 worlds are drawn 12 per skin so the two
#: training sets differ in archetype and in nothing else that we can control.
WORLDS_PER_SKIN = 12

# --- Blackwell (sm120) runtime contract ------------------------------------------------
# Qwen3.5's Gated DeltaNet layers run through flash-linear-attention. The version in the
# SkyRL venv is 0.5.1, whose Blackwell ``prepare_wy_repr_bwd`` Triton autotune config is
# broken; 0.5.2 restricts it to the stable configuration. We ship 0.5.2 as a project-local
# overlay (``runtime/fla-0.5.2``) rather than mutating the shared venv, and SkyRL's own
# Qwen3.5 Blackwell recipe additionally requires FLA_TILELANG=0.
REQUIRED_FLA_VERSION = "0.5.2"
#: vLLM 0.23.0's ``EngineCore.wake_up`` resumes the scheduler even on a weights-only
#: partial wake, while the KV cache is still unmapped. SkyRL's colocated path does exactly
#: that (``wake_up(tags=["weights"])`` then ``wake_up(tags=["kv_cache"])``). The overlay in
#: ``runtime/skyrl_patches`` adds the missing sleeping-state check -- and loads the frozen
#: policy base in bf16, without which the run does not fit in host memory at all.
REQUIRED_VLLM_VERSION = "0.23.0"
#: Fail an attempt instead of allowing vLLM's TP worker wake collective to wait forever.
VLLM_WAKE_TIMEOUT_SECONDS = 300

# --- resource budget, used by the launcher pre-flight ----------------------------------
#: Host RAM for one run, MEASURED on this stack (3-second cgroup sampling, see
#: docs/gpu_sizing.md): vLLM's shared sleep buffer holding the base weights (~23 GB, not
#: droppable because LoRA sync forces sleep level 1) plus the fp32 policy the FSDP ranks
#: keep on the host (~38 GB), plus the Ray driver, the vLLM engine process and the object
#: store. The single-GPU shape peaked at 95.6 GiB of a 96 GiB ceiling; sharding the policy
#: over two ranks is what brings this down.
APPROX_HOST_GB_PER_JOB = 78
#: Extra host RAM while one run materializes its state dict inside ``save_checkpoints``.
#: Only one run can be in that window at a time (the shared lock in ``main.py``).
CKPT_SAVE_SPIKE_GB = 18
#: GPU headroom PER CARD beyond vLLM's ``gpu_memory_utilization`` share. Measured on this
#: stack: with vLLM awake for generation, one card sat at 35.7 GB against a 33.5 GB vLLM
#: reservation, the policy being offloaded in that phase. Training is the other half of
#: the cycle -- vLLM asleep, the bf16 policy shard (~9.4 GB per rank) plus LoRA optimizer
#: state, activations and the CUDA graph pool back on the card. 16 GB covers both phases
#: with margin; it was 26 GB while the policy was loaded in fp32.
APPROX_GPU_GB_HEADROOM = 16
#: One checkpoint of this stack (bf16 policy + AdamW state + LoRA adapter + tokenizer).
APPROX_CKPT_GB = 19
#: Ray sizes its plasma store at 30% of what it thinks node memory is. Inside the
#: container ``/proc/meminfo`` reports the PHYSICAL host, not our cgroup, so the default
#: would be capped only by a 16 GiB /dev/shm that both runs share. Cap it explicitly.
RAY_OBJECT_STORE_BYTES = 4 * 1024**3

#: The outer LXC cgroup limit this container tree actually lives in.
#:
#: It cannot be read from where the trainer runs: inside the container
#: ``/sys/fs/cgroup/memory.max`` is ``max`` and ``/proc/meminfo`` reports the physical
#: 377 GB machine. ``launch.sh`` reads the real limit on the host and forwards it as
#: ``SA_CGROUP_MEMORY_BYTES``; this is the fallback when a run is started by hand from
#: inside the container.
DEFAULT_CGROUP_MEMORY_BYTES = 96 * 1024**3

#: Logical CPUs to give Ray. Left alone Ray sizes its worker pool from the machine's 144
#: cores and pre-starts a corresponding number of idle workers, each tens of MB of anon
#: memory, against a host-RAM budget that is this experiment's binding constraint. The
#: actors here ask for fractional CPU (0.2 per placement-group bundle), so a small number
#: is ample.
RAY_NUM_CPUS = 16

#: Host RAM left outside Ray's logical budget for the things Ray does not account for:
#: the vLLM engine process, the driver, and the ~18 GiB policy that ``colocate_all``
#: parks in host memory during every generation phase.
RAY_MEMORY_RESERVE_BYTES = 20 * 1024**3


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _parse_gpus(value: str) -> Tuple[int, ...]:
    """Parse ``SA_GPUS`` into the physical GPU ids for the run being launched."""
    try:
        gpus = tuple(int(part) for part in value.split(",") if part.strip() != "")
    except ValueError:
        raise SystemExit(f"SA_GPUS must be comma-separated integers, got {value!r}")
    if not gpus:
        raise SystemExit("SA_GPUS must name at least one GPU")
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f"SA_GPUS must not repeat a GPU: {value!r}")
    if any(g < 0 for g in gpus):
        raise SystemExit(f"SA_GPUS must be non-negative: {value!r}")
    return gpus


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment-wide settings (identical for both runs)."""

    exp_tag: str = field(default_factory=lambda: _env("SA_EXP_TAG", DEFAULT_EXP_TAG))
    out_root: str = field(default_factory=lambda: _env("SA_OUT_ROOT", DEFAULT_OUT_ROOT))
    ckpt_root: str = field(default_factory=lambda: _env("SA_CKPT_ROOT", DEFAULT_CKPT_ROOT))
    gpus: Tuple[int, ...] = field(
        default_factory=lambda: _parse_gpus(_env("SA_GPUS", ",".join(str(g) for g in DEFAULT_GPUS)))
    )
    rpg_src: str = field(default_factory=lambda: _env("RPG_SRC", DEFAULT_RPG_SRC))
    skyrl_dir: str = field(default_factory=lambda: _env("SA_SKYRL_DIR", DEFAULT_SKYRL_DIR))
    model: str = field(default_factory=lambda: _env("SA_MODEL", MODEL_PATH))
    wandb_project: str = field(default_factory=lambda: _env("SA_WANDB_PROJECT", DEFAULT_WANDB_PROJECT))

    # Knobs that are deliberately exposed but must stay equal across the two runs.
    max_training_steps: int = field(default_factory=lambda: _env_int("SA_MAX_STEPS", 30))
    #: Checkpoint every 4 optimizer steps. SkyRL additionally always saves at an
    #: epoch boundary (every 48 steps here) and once more at the end of training.
    ckpt_interval: int = field(default_factory=lambda: _env_int("SA_CKPT_INTERVAL", 4))
    eval_interval: int = field(default_factory=lambda: _env_int("SA_EVAL_INTERVAL", 2))
    seed: int = field(default_factory=lambda: _env_int("SA_SEED", 42))
    lr: float = field(default_factory=lambda: _env_float("SA_LR", 1.0e-5))
    num_warmup_steps: int = field(default_factory=lambda: _env_int("SA_WARMUP_STEPS", 2))
    lora_rank: int = field(default_factory=lambda: _env_int("SA_LORA_RANK", 16))
    lora_alpha: int = field(default_factory=lambda: _env_int("SA_LORA_ALPHA", 32))
    #: GRPO group size. Also the number of samples the per-step group_reward_var /
    #: group_reward_mean statistics are computed over (log requirement 1).
    n_samples_per_prompt: int = field(default_factory=lambda: _env_int("SA_GROUP_SIZE", 8))
    eval_n_samples_per_prompt: int = field(default_factory=lambda: _env_int("SA_EVAL_N", 2))
    #: vLLM's share of the card. This box is shared, so the runs deliberately do NOT
    #: claim half of a 96 GB card each. 0.35 leaves vLLM ~33 GB: ~18.5 GB of bf16 weights
    #: plus ~14 GB of KV / GDN-state cache, which is far more than the 8-16 concurrent
    #: sequences a single-GPU rollout phase actually keeps in flight. See
    #: `docs/gpu_sizing.md` for the measurement this number comes from.
    gpu_memory_utilization: float = field(default_factory=lambda: _env_float("SA_GPU_MEM_UTIL", 0.35))
    #: Qwen3.5 is a HYBRID model: 24 of its 32 layers are Gated DeltaNet, whose recurrent
    #: state vLLM allocates one slot per `max_num_seqs`, not per token. At fp32 that is
    #: ~48 MB per slot, so the stock 512 would reserve ~24 GB of pure bookkeeping before a
    #: single KV block exists. 64 slots is more concurrency than one card can decode.
    max_num_seqs: int = field(default_factory=lambda: _env_int("SA_MAX_NUM_SEQS", 64))
    max_prompt_length: int = field(default_factory=lambda: _env_int("SA_MAX_PROMPT_LEN", 18432))
    max_generate_length: int = field(default_factory=lambda: _env_int("SA_MAX_GEN_LEN", 8192))
    max_tokens_per_microbatch: int = field(default_factory=lambda: _env_int("SA_MAX_TOK_PER_MICROBATCH", 8192))
    max_turns: int = field(default_factory=lambda: _env_int("SA_MAX_TURNS", 33))
    #: vLLM context window. Must be >= max_prompt_length + max_generate_length (26624).
    max_model_len: int = field(default_factory=lambda: _env_int("SA_MAX_MODEL_LEN", 32768))
    #: Requirement (5): "latest" from the very first launch. With no checkpoint present
    #: SkyRL starts from step 0; after a crash the same command resumes from the newest
    #: checkpoint. Each process launch gets a fresh W&B attempt run (grouped under the
    #: experiment id), so replaying steps after rollback cannot be rejected by W&B.
    resume_mode: str = field(default_factory=lambda: _env("SA_RESUME_MODE", "latest"))
    max_env_workers: int = field(default_factory=lambda: _env_int("SA_MAX_ENV_WORKERS", 16))
    #: Forwarded by launch.sh from the host; see DEFAULT_CGROUP_MEMORY_BYTES.
    cgroup_memory_bytes: int = field(
        default_factory=lambda: _env_int("SA_CGROUP_MEMORY_BYTES", DEFAULT_CGROUP_MEMORY_BYTES)
    )

    @property
    def ray_logical_memory_bytes(self) -> int:
        """Ray's schedulable-memory budget for the run.

        Left to itself Ray would advertise the physical host's ~377 GB to actors living
        inside a 96 GiB cgroup. Only one run is in flight at a time, so it gets the whole
        cgroup minus the object store and the reserve.
        """
        return max(1, self.cgroup_memory_bytes - RAY_OBJECT_STORE_BYTES - RAY_MEMORY_RESERVE_BYTES)

    # -- derived ------------------------------------------------------------------------

    @property
    def epochs(self) -> int:
        """Enough passes over the 96 worlds to reach ``max_training_steps``."""
        return max(1, math.ceil(self.max_training_steps / STEPS_PER_EPOCH))

    @property
    def exp_dir(self) -> Path:
        return Path(self.out_root) / self.exp_tag

    @property
    def dataset_root(self) -> Path:
        return self.exp_dir / "datasets"

    @property
    def source_train(self) -> Path:
        return Path(self.rpg_src) / SOURCE_TRAIN

    @property
    def source_val(self) -> Path:
        return Path(self.rpg_src) / SOURCE_VAL

    def train_parquet(self, run_id: str) -> Path:
        return self.dataset_root / run_id / "train.parquet"

    @property
    def val_parquet(self) -> Path:
        """Shared by both runs: the same 45 held-out worlds, requirement (2)."""
        return self.dataset_root / "validation_small.parquet"

    # -- per-run, mutually exclusive locations ------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        return self.exp_dir / "runs" / run_id

    @property
    def ckpt_exp_dir(self) -> Path:
        return Path(self.ckpt_root) / self.exp_tag

    @property
    def ckpt_lock_path(self) -> Path:
        """Shared across the runs on purpose: it serializes their checkpoint saves."""
        return self.exp_dir / "checkpoint_save.lock"

    def run_paths(self, run_id: str) -> Dict[str, Path]:
        run = self.run_dir(run_id)
        return {
            "run_dir": run,
            # On the NFS volume, not under out_root -- ~19 GB per checkpoint.
            "ckpt_path": self.ckpt_exp_dir / "runs" / run_id / "checkpoints",
            "export_path": run / "exports",
            # The ~196 MB of each checkpoint that is actually trained content, copied out
            # so the deltas survive independently of the 19 GB blobs.
            "adapters_dir": run / "exports" / "adapters",
            "log_path": run / "logs",
            # NOT under run_dir: Ray builds its plasma-store socket as
            # <ray_tmpdir>/ray/session_<stamp>/sockets/plasma_store, and AF_UNIX caps the
            # whole path at 107 bytes. The session suffix alone is ~65 bytes, so a temp dir
            # nested inside the experiment tree overflows it. Local disk, not /data: a Unix
            # socket on NFS does not work.
            "ray_tmpdir": self.ray_tmpdir(run_id),
            "lora_sync_path": run / "lora_sync",
            "episode_scratch": run / "episode_scratch",
            "wandb_dir": run / "wandb",
            "triton_cache": run / "cache" / "triton",
            "inductor_cache": run / "cache" / "inductor",
            "vllm_cache": run / "cache" / "vllm",
        }

    #: Ray appends ``/ray/session_<date>_<time>_<us>_<pid>/sockets/plasma_store`` (~68
    #: bytes) to RAY_TMPDIR, and AF_UNIX rejects the result past 107 bytes.
    RAY_TMPDIR_MAX_LEN = 38

    def ray_tmpdir(self, run_id: str) -> Path:
        """A SHORT, local, per-run Ray temp dir. See RAY_TMPDIR_MAX_LEN."""
        path = Path("/tmp") / f"ray-{self.exp_tag}-{run_id}"
        if len(str(path)) > self.RAY_TMPDIR_MAX_LEN:
            raise SystemExit(
                f"SA_EXP_TAG={self.exp_tag!r} makes the Ray temp dir {str(path)!r} "
                f"({len(str(path))} chars) too long: Ray's plasma-store socket path would "
                f"exceed the 107-byte AF_UNIX limit. Use an exp tag of at most "
                f"{self.RAY_TMPDIR_MAX_LEN - len('/tmp/ray--') - max(len(r) for r in RUN_IDS)} "
                "characters."
            )
        return path

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(str(g) for g in self.gpus)

    def archetype_for(self, run_id: str) -> str:
        return TRAIN_ARCHETYPE[run_id]

    def run_name(self, run_id: str) -> str:
        return f"{self.exp_tag}_{run_id}_{TRAIN_ARCHETYPE[run_id]}"

    def wandb_run_id(self, run_id: str) -> str:
        """Stable W&B group/id prefix; run_one.sh adds a unique attempt suffix."""
        return f"{self.exp_tag}-{run_id}"


def validate_run_id(run_id: str) -> str:
    if run_id not in RUN_IDS:
        raise SystemExit(f"unknown run id {run_id!r}; expected one of {list(RUN_IDS)}")
    return run_id


# --------------------------------------------------------------------------------------
# Environment for one run
# --------------------------------------------------------------------------------------


def runtime_dir() -> Path:
    return Path(__file__).resolve().parent / "runtime"


def run_env(cfg: ExperimentConfig, run_id: str) -> Dict[str, str]:
    """Environment variables for one run: isolation + protocol + Blackwell runtime."""
    validate_run_id(run_id)
    paths = cfg.run_paths(run_id)
    package_parent = str(Path(__file__).resolve().parent.parent)
    overlays = [str(runtime_dir() / "skyrl_patches"), str(runtime_dir() / "fla-0.5.2")]
    # Deduplicate: run_one.sh already puts the package parent on PYTHONPATH in order to
    # import this module, and the value is re-exported into every Ray worker.
    seen = overlays + [package_parent]
    entries = seen + [e for e in os.environ.get("PYTHONPATH", "").split(":") if e and e not in seen]
    return {
        # -- which archetype this run trains on ------------------------------------------
        "SA_RUN_ID": run_id,
        "SA_TRAIN_ARCHETYPE": cfg.archetype_for(run_id),
        "SA_EXP_TAG": cfg.exp_tag,
        "SA_OUT_ROOT": cfg.out_root,
        "SA_RUN_DIR": str(paths["run_dir"]),
        # -- RPG protocol / sources -------------------------------------------------------
        "RPG_SRC": cfg.rpg_src,
        "RPG_PROTO": RPG_PROTO,
        # Per-episode CSV scratch for the `code` tool, kept inside this run's tree.
        "RPG_DATA_ROOT": str(paths["episode_scratch"]),
        # -- checkpointing -----------------------------------------------------------------
        # SHARED by both runs: main.py takes this lock around save_checkpoints() so only
        # one run at a time materializes an ~18 GiB state dict.
        "SA_CKPT_LOCK": str(cfg.ckpt_lock_path),
        "SA_ADAPTER_EXPORT_DIR": str(paths["adapters_dir"]),
        # -- Ray / runtime isolation --------------------------------------------------------
        # RAY_ADDRESS=local forces ray.init() to start a FRESH local cluster instead of
        # attaching to the sibling run's cluster; RAY_TMPDIR gives it its own session tree.
        "RAY_ADDRESS": "local",
        "RAY_TMPDIR": str(paths["ray_tmpdir"]),
        "CUDA_VISIBLE_DEVICES": cfg.cuda_visible_devices,
        # Compiler caches are per-run so two concurrent JIT compilations cannot race.
        "TRITON_CACHE_DIR": str(paths["triton_cache"]),
        "TORCHINDUCTOR_CACHE_DIR": str(paths["inductor_cache"]),
        "VLLM_CACHE_ROOT": str(paths["vllm_cache"]),
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(RAY_OBJECT_STORE_BYTES),
        "SA_CGROUP_MEMORY_BYTES": str(cfg.cgroup_memory_bytes),
        "SA_RAY_OBJECT_STORE_BYTES": str(RAY_OBJECT_STORE_BYTES),
        "SA_RAY_LOGICAL_MEMORY_BYTES": str(cfg.ray_logical_memory_bytes),
        "SA_RAY_NUM_CPUS": str(RAY_NUM_CPUS),
        # Ray's memory monitor reads /proc/meminfo, which inside this container is the
        # PHYSICAL 377 GB machine shared with other tenants -- it sat at 95% before this
        # experiment started anything, so the monitor immediately killed our FSDP policy
        # worker and the vLLM actor as "the node running low on memory". That pool is
        # neither ours to measure nor ours to influence; our real ceiling is the outer
        # 96 GiB cgroup, which the kernel enforces on its own. Disable the monitor and
        # budget explicitly (SA_RAY_LOGICAL_MEMORY_BYTES + the launcher's pre-flight).
        "RAY_memory_monitor_refresh_ms": "0",
        # -- import path: overlays first, then the package, forwarded into Ray workers -----
        "PYTHONPATH": ":".join(entries),
        "SKYRL_PYTHONPATH_EXPORT": "1",
        # -- Blackwell / Qwen3.5 runtime contract ------------------------------------------
        # SkyRL's Qwen3.5-on-Blackwell recipe: the TileLang GDN backend is disabled and the
        # FLA overlay above supplies 0.5.2. main.py asserts both before Ray starts.
        "FLA_TILELANG": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTHONFAULTHANDLER": "1",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        # Load the FROZEN base in bf16 instead of SkyRL's fp32 (runtime/skyrl_patches).
        # The fp32 materialization is 37.6 GB on the host for a 9.4 B model and is what
        # puts this run over the container tree's 96 GiB ceiling; the trained LoRA tensors
        # are promoted back to fp32. Set to "fp32" to restore SkyRL's default.
        "SA_POLICY_LOAD_DTYPE": _env("SA_POLICY_LOAD_DTYPE", "bf16"),
        "SA_VLLM_WAKE_TIMEOUT_SECONDS": _env(
            "SA_VLLM_WAKE_TIMEOUT_SECONDS", str(VLLM_WAKE_TIMEOUT_SECONDS)
        ),
        "HF_HOME": _env("HF_HOME", "/work/hf_cache"),
        # -- W&B: run_one.sh turns this stable prefix into a unique attempt id ------------
        "WANDB_PROJECT": cfg.wandb_project,
        "WANDB_RUN_ID": cfg.wandb_run_id(run_id),
        "WANDB_NAME": cfg.run_name(run_id),
        "WANDB_RUN_GROUP": cfg.wandb_run_id(run_id),
        "WANDB_RESUME": "never",
        "WANDB_DIR": str(paths["wandb_dir"]),
    }


#: Deliberately NOT set. ``expandable_segments`` is incompatible with the CuMemAllocator
#: pool vLLM unmaps and re-maps on every sleep/wake cycle under ``colocate_all``
#: (pytorch/pytorch#147851). vLLM neutralizes the setting only inside its own context
#: manager, so activations, the CUDA graph pool and LoRA buffers still came from
#: expandable segments while their physical pages were being remapped; on this box that
#: killed the vLLM worker on the 5th wake-up in three consecutive runs. ``run_one.sh``
#: unsets it.
UNSET_IN_RUN_ENV: Tuple[str, ...] = ("PYTORCH_CUDA_ALLOC_CONF",)


# --------------------------------------------------------------------------------------
# SkyRL overrides for one run
# --------------------------------------------------------------------------------------


def run_overrides(cfg: ExperimentConfig, run_id: str) -> List[str]:
    """The full SkyRL CLI override list for one run.

    Identical for `easy` and `hard` except for the training parquet and the run-scoped
    output paths -- which is the fairness condition the comparison needs.
    """
    validate_run_id(run_id)
    paths = cfg.run_paths(run_id)
    return [
        # ---- data: the run's 96 single-archetype worlds; SHARED validation set ---------
        f"data.train_data=['{cfg.train_parquet(run_id)}']",
        f"data.val_data=['{cfg.val_parquet}']",
        # ---- model --------------------------------------------------------------------
        f"trainer.policy.model.path={cfg.model}",
        "trainer.strategy=fsdp",
        f"trainer.policy.model.lora.rank={cfg.lora_rank}",
        f"trainer.policy.model.lora.alpha={cfg.lora_alpha}",
        # The LoRA adapter handoff dir MUST be per-run: SkyRL's default is a single
        # shared /tmp path, and two concurrent jobs would silently load each other's
        # adapter between the trainer writing it and vLLM reading it back.
        f"trainer.policy.model.lora.lora_sync_path={paths['lora_sync_path']}",
        # ---- placement: one GPU per job -----------------------------------------------
        "trainer.placement.colocate_all=true",
        f"trainer.placement.policy_num_gpus_per_node={cfg.num_gpus}",
        f"trainer.placement.ref_num_gpus_per_node={cfg.num_gpus}",
        # ONE tensor-parallel engine, not one data-parallel engine per GPU. Under
        # colocate_all every engine keeps its own host-side sleep buffer holding the base
        # weights (~23 GB, and LoRA sync forbids dropping them), so N data-parallel
        # engines cost N copies. Tensor parallelism splits the weights instead: one
        # buffer, halved per GPU. On this box that is the difference between fitting in
        # the 96 GiB host budget and not -- see docs/gpu_sizing.md.
        "generator.inference_engine.num_engines=1",
        f"generator.inference_engine.tensor_parallel_size={cfg.num_gpus}",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.distributed_executor_backend=mp",
        "generator.inference_engine.weight_sync_backend=nccl",
        f"generator.inference_engine.gpu_memory_utilization={cfg.gpu_memory_utilization}",
        f"generator.inference_engine.max_num_batched_tokens={cfg.max_prompt_length}",
        f"generator.inference_engine.max_num_seqs={cfg.max_num_seqs}",
        # SkyRL does not pass max_model_len, so vLLM would default to Qwen3.5's
        # max_position_embeddings of 262144 and size its KV budget for a 256k context that
        # this env never produces (max_input_length + max_generate_length = 26624). Pinning
        # it is what makes the reduced gpu_memory_utilization above safe.
        f"generator.inference_engine.engine_init_kwargs.max_model_len={cfg.max_model_len}",
        # ---- algorithm (Dr.GRPO-style: no std normalization, no batch normalization) ---
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.algorithm.grpo_norm_by_std=false",
        "trainer.algorithm.advantage_batch_normalize=false",
        "trainer.algorithm.loss_reduction=token_mean",
        "trainer.algorithm.use_kl_loss=false",
        "trainer.algorithm.use_entropy_loss=false",
        # Keep zero-variance groups: dropping them would silently change the effective
        # batch size differently for the two archetypes, and group_reward_var is exactly
        # the quantity this experiment is measuring.
        "trainer.algorithm.zero_variance_filter=false",
        # ---- optimization --------------------------------------------------------------
        f"trainer.policy.optimizer_config.lr={cfg.lr}",
        "trainer.policy.optimizer_config.max_grad_norm=1.0",
        "trainer.policy.optimizer_config.scheduler=constant_with_warmup",
        f"trainer.policy.optimizer_config.num_warmup_steps={cfg.num_warmup_steps}",
        f"trainer.seed={cfg.seed}",
        # ---- batching: 1 global step == 1 optimizer step --------------------------------
        f"trainer.epochs={cfg.epochs}",
        f"trainer.max_training_steps={cfg.max_training_steps}",
        f"trainer.train_batch_size={TRAIN_BATCH_SIZE}",
        f"trainer.policy_mini_batch_size={POLICY_MINI_BATCH_SIZE}",
        f"trainer.update_epochs_per_batch={UPDATE_EPOCHS_PER_BATCH}",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.micro_forward_batch_size_per_gpu=1",
        f"trainer.max_tokens_per_microbatch={cfg.max_tokens_per_microbatch}",
        "trainer.remove_microbatch_padding=false",
        f"trainer.max_prompt_length={cfg.max_prompt_length}",
        # ---- generation -----------------------------------------------------------------
        "generator.batched=false",
        f"generator.n_samples_per_prompt={cfg.n_samples_per_prompt}",
        f"generator.max_input_length={cfg.max_prompt_length}",
        f"generator.max_turns={cfg.max_turns}",
        "generator.sampling_params.temperature=1.0",
        "generator.sampling_params.top_p=1.0",
        "generator.sampling_params.top_k=-1",
        f"generator.sampling_params.max_generate_length={cfg.max_generate_length}",
        # eval_sampling_params does NOT inherit the training values once any of its
        # fields is overridden, so every field is stated explicitly.
        "generator.eval_sampling_params.temperature=1.0",
        "generator.eval_sampling_params.top_p=1.0",
        "generator.eval_sampling_params.top_k=-1",
        f"generator.eval_sampling_params.max_generate_length={cfg.max_generate_length}",
        f"generator.eval_n_samples_per_prompt={cfg.eval_n_samples_per_prompt}",
        # ---- environment ------------------------------------------------------------------
        f"environment.env_class={ENV_ID}",
        f"environment.skyrl_gym.max_env_workers={cfg.max_env_workers}",
        # ---- evaluation (log requirement 2): per-archetype avg score / part_a / part_b ---
        "trainer.eval_before_train=true",
        f"trainer.eval_interval={cfg.eval_interval}",
        "trainer.dump_eval_results=true",
        # ---- checkpoint / logging / isolation ---------------------------------------------
        f"trainer.ckpt_interval={cfg.ckpt_interval}",
        # Keep every checkpoint: the NFS volume has 31 TB free, and pruning would delete
        # the earlier checkpoints this experiment wants to re-evaluate.
        "trainer.max_ckpts_to_keep=-1",
        "trainer.hf_save_interval=-1",
        f"trainer.resume_mode={cfg.resume_mode}",
        f"trainer.ckpt_path={paths['ckpt_path']}",
        f"trainer.export_path={paths['export_path']}",
        f"trainer.log_path={paths['log_path']}",
        "trainer.logger=wandb",
        f"trainer.project_name={cfg.wandb_project}",
        f"trainer.run_name={cfg.run_name(run_id)}",
    ]


def run_manifest(cfg: ExperimentConfig, run_id: str) -> Dict[str, object]:
    """Everything needed to reproduce or audit one run."""
    paths = cfg.run_paths(run_id)
    return {
        "run_id": run_id,
        "train_archetype": cfg.archetype_for(run_id),
        "train_worlds": TRAIN_WORLDS_PER_RUN,
        "exp_tag": cfg.exp_tag,
        "gpus": list(cfg.gpus),
        "tensor_parallel_size": cfg.num_gpus,
        "run_name": cfg.run_name(run_id),
        "wandb_run_id": cfg.wandb_run_id(run_id),
        "wandb_project": cfg.wandb_project,
        "model": cfg.model,
        "rpg_proto": RPG_PROTO,
        "max_training_steps": cfg.max_training_steps,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": cfg.epochs,
        "ckpt_interval": cfg.ckpt_interval,
        "eval_interval": cfg.eval_interval,
        "source_train": str(cfg.source_train),
        "source_val": str(cfg.source_val),
        "train_parquet": str(cfg.train_parquet(run_id)),
        "val_parquet": str(cfg.val_parquet),
        "ckpt_path": str(paths["ckpt_path"]),
        "adapters_dir": str(paths["adapters_dir"]),
        "env": run_env(cfg, run_id),
        "overrides": run_overrides(cfg, run_id),
    }


def _main(argv: List[str]) -> int:
    if len(argv) < 1:
        print(__doc__)
        print("usage: python -m single_arch_rl.config {env|args|args0|manifest|paths|budget} <run_id>")
        return 2
    what = argv[0]
    cfg = ExperimentConfig()
    if what == "budget":
        # Single source of truth for the launcher's pre-flight arithmetic.
        n_ckpts = cfg.max_training_steps // max(1, cfg.ckpt_interval) + cfg.epochs + 1
        print(f"HOST_GB_PER_JOB={APPROX_HOST_GB_PER_JOB}")
        print(f"CKPT_SPIKE_GB={CKPT_SAVE_SPIKE_GB if cfg.ckpt_interval > 0 else 0}")
        print(f"GPU_GB_HEADROOM={APPROX_GPU_GB_HEADROOM}")
        print(f"GPU_MEM_UTIL={cfg.gpu_memory_utilization}")
        print(f"CKPT_ROOT={cfg.ckpt_exp_dir}")
        print(f"CKPT_INTERVAL={cfg.ckpt_interval}")
        print(f"CKPT_GB_PER_RUN={APPROX_CKPT_GB * n_ckpts if cfg.ckpt_interval > 0 else 0}")
        print(f"GPUS={','.join(str(g) for g in cfg.gpus)}")
        print(f"MAX_STEPS={cfg.max_training_steps}")
        print(f"EXP_DIR={cfg.exp_dir}")
        print(f"DATASET_ROOT={cfg.dataset_root}")
        return 0
    if len(argv) < 2:
        raise SystemExit(f"subcommand {what!r} needs a run id: {list(RUN_IDS)}")
    run_id = validate_run_id(argv[1])
    if what == "env":
        for key, value in run_env(cfg, run_id).items():
            print(f"{key}={value}")
    elif what == "args":
        # Human-readable form. Shell callers must use `args0`: quoting inside a variable
        # is not re-interpreted by word splitting, so a `$(...)` capture of this would
        # pass literal quote characters through to OmegaConf.
        print(" ".join(shlex.quote(a) for a in run_overrides(cfg, run_id)))
    elif what == "args0":
        sys.stdout.write("".join(a + "\0" for a in run_overrides(cfg, run_id)))
    elif what == "manifest":
        print(json.dumps(run_manifest(cfg, run_id), indent=2))
    elif what == "paths":
        for key, value in cfg.run_paths(run_id).items():
            print(f"{key}={value}")
    else:
        raise SystemExit(f"unknown subcommand {what!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
