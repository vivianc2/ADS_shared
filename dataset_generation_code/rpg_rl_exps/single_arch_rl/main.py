#!/usr/bin/env python3
"""SkyRL training entrypoint for ONE single-archetype run (`easy` or `hard`).

Mirrors the shipped ``skyrl_rpg/main_rpg.py`` -- register the environment inside the Ray
entrypoint task, then run the standard GRPO loop -- and adds four things:

1. **Pre-flight assertions** that turn the experiment's invariants into hard failures at
   startup instead of silent drift 20 minutes into a GPU run.
2. **The Blackwell runtime contract.** Qwen3.5's Gated DeltaNet layers go through
   flash-linear-attention; the SkyRL venv ships 0.5.1, whose Blackwell
   ``prepare_wy_repr_bwd`` autotune config is broken, and vLLM 0.23.0 resumes its
   scheduler on a weights-only partial wake while the KV cache is still unmapped. Both are
   fixed by the project-local overlays in ``runtime/``; this asserts they actually loaded.
3. **The per-step group statistics** (log requirement 1): ``reward/group_reward_mean``
   and ``reward/group_reward_var``, computed from the same rewards and uids SkyRL uses
   for ``reward/avg_raw_reward``.
4. **A checkpoint guard**: the two archetype runs serialize their ~18 GiB state-dict
   saves behind one file lock, and each saved LoRA adapter is copied out of the 19 GB
   blob so the trained delta survives on its own.

Launched by ``scripts/run_one.sh``; the override list comes from ``single_arch_rl.config``
so there is exactly one definition of the run settings.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import sys
import time

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register

from single_arch_rl.config import (
    ENV_ENTRY_POINT,
    ENV_ID,
    POLICY_MINI_BATCH_SIZE,
    REQUIRED_FLA_VERSION,
    REQUIRED_VLLM_VERSION,
    RPG_PROTO,
    RUN_IDS,
    TRAIN_ARCHETYPE,
    TRAIN_BATCH_SIZE,
    TRAIN_WORLDS_PER_RUN,
)
from single_arch_rl.metrics import group_reward_stats


# --------------------------------------------------------------------------------------
# Blackwell / Qwen3.5 runtime contract
# --------------------------------------------------------------------------------------


def _check_runtime_overlays() -> list:
    """Verify the project-local runtime overlays are the ones actually imported.

    Both are silent when missing -- FLA 0.5.1 fails much later inside a backward pass, and
    the vLLM partial wake corrupts scheduling rather than raising -- so check eagerly, on
    the driver, before any GPU work starts.
    """
    problems = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return [f"torch is not importable: {exc!r}"]

    capabilities = tuple(
        torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
    )
    is_blackwell = any(major in {10, 11, 12} for major, _minor in capabilities)
    if not is_blackwell:
        return problems  # the contract is Blackwell-specific

    if os.environ.get("FLA_TILELANG") != "0":
        problems.append(
            "Qwen3.5 on Blackwell requires FLA_TILELANG=0; got "
            f"{os.environ.get('FLA_TILELANG')!r}"
        )
    try:
        import fla

        version = str(getattr(fla, "__version__", "unknown"))
        if version != REQUIRED_FLA_VERSION:
            problems.append(
                f"flash-linear-attention {REQUIRED_FLA_VERSION} required on Blackwell "
                f"(the 0.5.1 in the venv has a broken prepare_wy_repr_bwd autotune "
                f"config); imported {version} from {fla.__file__}. Put runtime/fla-0.5.2 "
                "first on PYTHONPATH -- scripts/run_one.sh does this."
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not import fla: {exc!r}")

    try:
        import vllm
        from vllm.device_allocator.cumem import CuMemAllocator
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        if vllm.__version__ != REQUIRED_VLLM_VERSION:
            problems.append(
                f"the partial-wake guard is reviewed only for vLLM {REQUIRED_VLLM_VERSION}; "
                f"found {vllm.__version__}"
            )
        elif not getattr(EngineCore.wake_up, "_pope_partial_wake_guard", False):
            problems.append(
                "vLLM's partial-wake guard did not load: EngineCore.wake_up would resume "
                "the scheduler on wake_up(['weights']) while the KV cache is still "
                "unmapped. Put runtime/skyrl_patches on PYTHONPATH so its "
                "sitecustomize.py runs -- scripts/run_one.sh does this."
            )
        if not getattr(CuMemAllocator.wake_up, "_sa_cumem_state_guard", False):
            problems.append(
                "vLLM's CuMem sleep/wake guard did not load; repeated colocated wake cycles "
                "can wedge a tensor-parallel worker"
            )
        if not getattr(MultiprocExecutor.wake_up, "_sa_wake_timeout", False):
            problems.append(
                "vLLM's multiprocess wake timeout did not load; a wedged TP worker would "
                "block sync_weights forever"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not inspect vLLM EngineCore: {exc!r}")
    return problems


# --------------------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------------------


def _check_policy_load_patch() -> list:
    """The bf16 base load must be armed, or the run OOMs during policy init."""
    requested = os.environ.get("SA_POLICY_LOAD_DTYPE", "bf16").lower()
    if requested not in ("bf16", "bfloat16"):
        return []   # explicitly opted out; the operator accepted the memory cost
    overlay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "skyrl_patches")
    if overlay not in os.environ.get("PYTHONPATH", "").split(":"):
        return [
            f"{overlay} must be on PYTHONPATH so its sitecustomize.py loads the policy "
            "base in bf16; without it the fp32 materialization exceeds the host memory "
            "ceiling and the policy worker is killed during init"
        ]
    try:
        from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

        if not getattr(HFModelWrapper.__init__, "_sa_bf16_load", False):
            return ["the bf16 policy-load patch did not apply to HFModelWrapper"]
    except Exception as exc:  # noqa: BLE001
        return [f"could not verify the bf16 policy-load patch: {exc!r}"]
    return []


def _preflight(cfg: SkyRLTrainConfig) -> None:
    """Assert the properties the comparison depends on, before any GPU work starts."""
    problems = []

    run_id = os.environ.get("SA_RUN_ID")
    if run_id not in RUN_IDS:
        problems.append(f"SA_RUN_ID must be one of {list(RUN_IDS)}, got {run_id!r}")
    else:
        expected_arch = TRAIN_ARCHETYPE[run_id]
        if os.environ.get("SA_TRAIN_ARCHETYPE") != expected_arch:
            problems.append(
                f"SA_TRAIN_ARCHETYPE must be {expected_arch!r} for run {run_id!r}, got "
                f"{os.environ.get('SA_TRAIN_ARCHETYPE')!r}"
            )
        # The whole experiment is "this run saw only this archetype". Check the parquet
        # the trainer is actually about to read, not the one config.py would have named.
        for path in cfg.data.train_data:
            problems.extend(_check_train_parquet(path, expected_arch))
    for path in list(cfg.data.train_data) + list(cfg.data.val_data):
        problems.extend(_check_parquet_env_class(path))

    if os.environ.get("RPG_PROTO") != RPG_PROTO:
        problems.append(f"RPG_PROTO must be {RPG_PROTO!r}, got {os.environ.get('RPG_PROTO')!r}")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        problems.append(
            "PYTORCH_CUDA_ALLOC_CONF must be unset: expandable_segments is incompatible "
            "with the CuMemAllocator pool vLLM re-maps on every colocate_all sleep/wake "
            f"cycle (got {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r})"
        )

    # One global step == one optimizer step, so "checkpoint every 4 steps" is unambiguous.
    if cfg.trainer.train_batch_size != TRAIN_BATCH_SIZE:
        problems.append(f"trainer.train_batch_size must be {TRAIN_BATCH_SIZE}")
    if cfg.trainer.policy_mini_batch_size != POLICY_MINI_BATCH_SIZE:
        problems.append(f"trainer.policy_mini_batch_size must be {POLICY_MINI_BATCH_SIZE}")
    if cfg.trainer.update_epochs_per_batch != 1:
        problems.append("trainer.update_epochs_per_batch must be 1")
    if cfg.generator.n_samples_per_prompt < 2:
        problems.append("generator.n_samples_per_prompt must be >= 2 for group_reward_var")

    if not cfg.trainer.eval_before_train:
        problems.append("trainer.eval_before_train must be true to get the step-0 evaluation")
    if cfg.trainer.eval_interval <= 0:
        problems.append("trainer.eval_interval must be > 0")
    if cfg.environment.env_class != ENV_ID:
        problems.append(f"environment.env_class must be {ENV_ID!r}")
    # Requirement (5): the same command must resume a crashed run rather than silently
    # restarting its training state from step 0. W&B attempts are intentionally separate.
    if str(cfg.trainer.resume_mode) not in ("ResumeMode.LATEST", "latest"):
        problems.append(f"trainer.resume_mode must be 'latest', got {cfg.trainer.resume_mode!r}")
    if not os.environ.get("WANDB_RUN_ID"):
        problems.append("WANDB_RUN_ID is unset; a resumed run would start a new W&B chart")

    # Ray cannot see the outer cgroup from in here (see
    # _init_ray_with_a_cgroup_aware_budget); without these it sizes itself from, and
    # OOM-kills our workers over, the physical host shared with other tenants.
    if os.environ.get("RAY_memory_monitor_refresh_ms") != "0":
        problems.append(
            "RAY_memory_monitor_refresh_ms must be '0': Ray's memory monitor reads the "
            "physical 377 GB host, which other tenants already hold at ~95%, and kills "
            "this run's workers on startup"
        )
    if not os.environ.get("SA_RAY_LOGICAL_MEMORY_BYTES"):
        problems.append("SA_RAY_LOGICAL_MEMORY_BYTES is unset; Ray would advertise the host's memory")

    # Checkpointing: a ~19 GB write that must land on a writable volume with room, and
    # must be serialized across the two concurrent runs.
    if cfg.trainer.ckpt_interval > 0:
        if not os.environ.get("SA_CKPT_LOCK"):
            problems.append("SA_CKPT_LOCK is unset; concurrent checkpoint saves would not be serialized")
        ckpt_parent = os.path.dirname(cfg.trainer.ckpt_path.rstrip("/")) or "/"
        probe = cfg.trainer.ckpt_path if os.path.isdir(cfg.trainer.ckpt_path) else ckpt_parent
        if not os.path.isdir(probe) or not os.access(probe, os.W_OK):
            problems.append(
                f"trainer.ckpt_path is not writable: {cfg.trainer.ckpt_path}. Checkpoints "
                "need the /data volume mounted (the skyrl-pc container), or set "
                "SA_CKPT_INTERVAL=0 to disable them."
            )
        else:
            free_gb = shutil.disk_usage(probe).free / 2**30
            needed_gb = 19 * (cfg.trainer.max_training_steps // cfg.trainer.ckpt_interval + 2)
            if free_gb < needed_gb:
                problems.append(
                    f"only {free_gb:.0f} GB free at {probe}; this run writes about "
                    f"{needed_gb} GB of checkpoints"
                )

    # The policy-load patch is what keeps the run inside the host-memory ceiling; if the
    # overlay did not load, the fp32 materialization silently returns and the policy
    # worker is killed 90 seconds in. Assert the overlay is importable and armed.
    problems.extend(_check_policy_load_patch())
    problems.extend(_check_runtime_overlays())

    if problems:
        raise SystemExit("single_arch_rl pre-flight failed:\n  - " + "\n  - ".join(problems))


def _check_parquet_env_class(path: str) -> list:
    """Every row must name THIS experiment's env.

    SkyRL constructs each episode's environment from the row's ``env_class``, not from
    ``environment.env_class``. A parquet built by copying the source's ``"rpg"`` therefore
    routes every episode to the shipped env, which this process never registers, and the
    run dies at the first evaluation.
    """
    try:
        import pandas as pd

        classes = set(pd.read_parquet(path, columns=["env_class"])["env_class"])
    except Exception as exc:  # noqa: BLE001
        return [f"could not read env_class from {path}: {exc!r}"]
    if classes != {ENV_ID}:
        return [f"{path} must have env_class={ENV_ID!r} on every row, found {sorted(classes)}"]
    return []


def _check_train_parquet(path: str, expected_archetype: str) -> list:
    """The training parquet must be exactly ``TRAIN_WORLDS_PER_RUN`` worlds of one archetype."""
    problems = []
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return [f"could not read train parquet {path}: {exc!r}"]
    archetypes = {dict(row).get("archetype") for row in frame["extra_info"]}
    if archetypes != {expected_archetype}:
        problems.append(
            f"{path} must contain only {expected_archetype!r} worlds, found {sorted(archetypes)}"
        )
    if len(frame) != TRAIN_WORLDS_PER_RUN:
        problems.append(f"{path} must contain {TRAIN_WORLDS_PER_RUN} worlds, found {len(frame)}")
    return problems


# --------------------------------------------------------------------------------------
# Trainer patches
# --------------------------------------------------------------------------------------


def _install_group_reward_metrics() -> None:
    """Log ``reward/group_reward_mean`` and ``reward/group_reward_var`` every step.

    ``postprocess_generator_output`` is the one place that still holds trajectory-level
    rewards next to their uids -- it converts them to per-token rewards in place -- so the
    statistics are computed on the way in, from exactly the arrays SkyRL uses for
    ``reward/avg_raw_reward``. A failure here is swallowed: a metrics hook must never be
    the reason a step dies.
    """
    from skyrl.train.trainer import RayPPOTrainer

    original = RayPPOTrainer.postprocess_generator_output
    if getattr(original, "_sa_group_metrics", False):
        return

    def wrapped(self, generator_output, uids, metrics_generator_output=None, metrics_uids=None):
        # Mirror SkyRL's own choice: when a superset is supplied for metrics (dropped
        # groups under sample_full_batch), report over that superset.
        source = metrics_generator_output if metrics_generator_output is not None else generator_output
        source_uids = metrics_uids if metrics_uids is not None else uids
        try:
            stats = group_reward_stats(source["rewards"], source_uids)
        except Exception as exc:  # noqa: BLE001
            print(f"[single_arch_rl] group reward stats failed: {exc!r}", flush=True)
            stats = {}
        out = original(self, generator_output, uids, metrics_generator_output, metrics_uids)
        if stats:
            self.all_metrics.update(stats)
            print(
                f"[single_arch_rl] step {self.global_step}: "
                f"group_reward_mean={stats['reward/group_reward_mean']:.4f} "
                f"group_reward_var={stats['reward/group_reward_var']:.4f} "
                f"nondegenerate={stats['reward/frac_nondegenerate_groups']:.0%} "
                f"({int(stats['reward/num_groups'])} groups)",
                flush=True,
            )
        return out

    wrapped._sa_group_metrics = True
    RayPPOTrainer.postprocess_generator_output = wrapped


def _install_checkpoint_guard() -> None:
    """Serialize checkpoint saves across the two concurrent runs, and export adapters.

    **Memory.** ``save_checkpoints`` materializes the policy state dict (~18 GiB bf16)
    before writing it. Both runs hit step 2, 4, 6 ... at the same step numbers, so without
    coordination both can be inside that window at once -- ~+36 GiB on top of the steady
    state, against a 96 GiB cgroup limit. This is the exact call an earlier RPG run died
    inside (``ActorDiedError`` / SIGTERM in ``save_checkpoints``). A file lock shared by
    the runs (``SA_CKPT_LOCK``, placed at the experiment level by ``config.py``) makes at
    most one save happen at a time. A waiting run is not doing GPU work, so the cost is
    wall-clock on the slower run, not correctness.

    **Durability of the useful part.** Only ``policy/lora_adapter/`` (~196 MB) is trained
    content; the other ~18 GiB is a frozen copy of Qwen3.5-9B that already exists in the
    HF cache. Copy the adapter into the run's export tree so the deltas survive
    independently of the big blobs.
    """
    from skyrl.train.trainer import RayPPOTrainer

    lock_path = os.environ.get("SA_CKPT_LOCK")
    adapter_root = os.environ.get("SA_ADAPTER_EXPORT_DIR")
    if not lock_path:
        return
    if getattr(RayPPOTrainer.save_checkpoints, "_sa_guarded", False):
        return

    original = RayPPOTrainer.save_checkpoints
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    def _export_adapter(ckpt_dir: str, global_step: int) -> None:
        source = os.path.join(ckpt_dir, "policy", "lora_adapter")
        if not adapter_root or not os.path.isdir(source):
            return
        destination = os.path.join(adapter_root, f"global_step_{global_step}")
        try:
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)
            # The tokenizer/config next to it make the adapter directly loadable.
            hf_dir = os.path.join(ckpt_dir, "policy", "huggingface")
            if os.path.isdir(hf_dir):
                shutil.copytree(hf_dir, os.path.join(destination, "huggingface"))
            print(f"[single_arch_rl] exported LoRA adapter -> {destination}", flush=True)
        except Exception as exc:  # noqa: BLE001 - never fail training over a copy
            print(f"[single_arch_rl] adapter export failed: {exc!r}", flush=True)

    def guarded_save_checkpoints(self):
        started = time.monotonic()
        with open(lock_path, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            waited = time.monotonic() - started
            if waited > 1.0:
                print(
                    f"[single_arch_rl] waited {waited:.0f}s for the shared checkpoint lock "
                    "(the other archetype run was saving)",
                    flush=True,
                )
            try:
                ckpt_dir = original(self)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        _export_adapter(ckpt_dir, self.global_step)
        return ckpt_dir

    guarded_save_checkpoints._sa_guarded = True
    RayPPOTrainer.save_checkpoints = guarded_save_checkpoints


def _init_ray_with_a_cgroup_aware_budget(cfg: SkyRLTrainConfig) -> None:
    """Start Ray with OUR memory budget, not the physical host's.

    Neither ``/proc/meminfo`` nor ``/sys/fs/cgroup/memory.max`` inside this container
    describes the limit that actually applies: both report the 377 GB machine, while the
    real ceiling is the outer LXC cgroup (96 GiB, shared with the sibling run). Left
    alone Ray therefore advertises ~377 GB of schedulable memory and sizes its plasma
    store from it. ``RAY_memory_monitor_refresh_ms=0`` in the run environment separately
    stops Ray's OOM monitor from killing our workers over other tenants' usage.
    """
    object_store_memory = int(os.environ.get("SA_RAY_OBJECT_STORE_BYTES", 4 * 1024**3))
    logical_memory = int(os.environ.get("SA_RAY_LOGICAL_MEMORY_BYTES", 32 * 1024**3))
    original = ray.init

    num_cpus = int(os.environ.get("SA_RAY_NUM_CPUS", 16))

    def cgroup_aware_ray_init(*args, **kwargs):
        kwargs.setdefault("object_store_memory", object_store_memory)
        # Ray 2.56 takes `_memory` as the node's logical schedulable-memory budget.
        kwargs.setdefault("_memory", logical_memory)
        # Ray sizes its idle worker pool from the node's CPU count -- 144 here, each
        # worker tens of MB of anon memory that this box's host-RAM budget cannot spare.
        kwargs.setdefault("num_cpus", num_cpus)
        return original(*args, **kwargs)

    print(
        f"[single_arch_rl] ray budget: logical={logical_memory / 2**30:.0f} GiB "
        f"object_store={object_store_memory / 2**30:.0f} GiB cpus={num_cpus} "
        f"(cgroup {int(os.environ.get('SA_CGROUP_MEMORY_BYTES', 0)) / 2**30:.0f} GiB shared "
        f"by {len(RUN_IDS)} runs); memory monitor "
        f"{'DISABLED' if os.environ.get('RAY_memory_monitor_refresh_ms') == '0' else 'ON'}",
        flush=True,
    )
    ray.init = cgroup_aware_ray_init
    try:
        initialize_ray(cfg)
    finally:
        ray.init = original


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(id=ENV_ID, entry_point=ENV_ENTRY_POINT)
    _install_group_reward_metrics()
    _install_checkpoint_guard()
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    _preflight(cfg)
    if os.environ.get("SA_PREFLIGHT_ONLY") == "1":
        # Exercises the entire launch path (env, config, overrides, env-class import,
        # runtime overlays) on a machine with no free GPU.
        register(id=ENV_ID, entry_point=ENV_ENTRY_POINT)
        from skyrl_gym.envs.registration import load_env_creator

        load_env_creator(ENV_ENTRY_POINT)
        _install_group_reward_metrics()
        _install_checkpoint_guard()
        print(
            "[single_arch_rl] pre-flight OK for "
            f"{os.environ.get('SA_RUN_ID')} ({os.environ.get('SA_TRAIN_ARCHETYPE')}) on GPU "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}; exiting before Ray init "
            "(SA_PREFLIGHT_ONLY=1)"
        )
        return
    _init_ray_with_a_cgroup_aware_budget(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
