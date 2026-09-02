#!/usr/bin/env python3
"""SkyRL training entrypoint for one prompt-comparison run.

Mirrors the shipped ``skyrl_rpg/main_rpg.py``: register the environment inside the Ray
entrypoint task, then run the standard GRPO experiment loop. The only differences are
the env id / entry point (so this experiment cannot collide with an ``rpg`` run started
from the same SkyRL checkout) and a set of pre-flight assertions that turn the
experiment's invariants into hard failures at startup rather than silent drift.

Launched by ``scripts/run_one.sh``; the override list comes from
``prompt_compare_rl.config`` so there is exactly one definition of the run settings.
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

ENV_ID = "rpg_prompt_compare"
ENV_ENTRY_POINT = "prompt_compare_rl.sky_env:PromptCompareRPGEnv"


def _preflight(cfg: SkyRLTrainConfig) -> None:
    """Assert the properties the comparison depends on, before any GPU work starts."""
    from prompt_compare_rl.config import EVAL_INTERVAL, MAX_TRAINING_STEPS, RPG_PROTO

    problems = []

    if os.environ.get("RPG_PROTO") != RPG_PROTO:
        problems.append(f"RPG_PROTO must be {RPG_PROTO!r}, got {os.environ.get('RPG_PROTO')!r}")
    if not os.environ.get("PC_PROMPT_SHA256"):
        problems.append("PC_PROMPT_SHA256 is unset; the env cannot verify prompt arrival")

    # Requirement 3: exactly 8 optimizer steps.
    if cfg.trainer.max_training_steps != MAX_TRAINING_STEPS:
        problems.append(f"trainer.max_training_steps must be {MAX_TRAINING_STEPS}")
    if cfg.trainer.epochs != 1:
        problems.append("trainer.epochs must be 1")
    if cfg.trainer.train_batch_size != cfg.trainer.policy_mini_batch_size:
        problems.append(
            "train_batch_size must equal policy_mini_batch_size so one global step is one "
            f"optimizer step (got {cfg.trainer.train_batch_size} vs {cfg.trainer.policy_mini_batch_size})"
        )
    if cfg.trainer.update_epochs_per_batch != 1:
        problems.append("trainer.update_epochs_per_batch must be 1")

    # Requirement 5: evaluate at 0, 4, 8.
    if not cfg.trainer.eval_before_train:
        problems.append("trainer.eval_before_train must be true to get the step-0 evaluation")
    if cfg.trainer.eval_interval != EVAL_INTERVAL:
        problems.append(f"trainer.eval_interval must be {EVAL_INTERVAL}")

    # Requirement 8.
    if not cfg.generator.inference_engine.language_model_only:
        problems.append("generator.inference_engine.language_model_only must be true (text-only)")
    wrap = (cfg.trainer.policy.fsdp_config.wrap_policy or {}).get("transformer_layer_cls_to_wrap")
    if wrap != "Qwen3_5DecoderLayer":
        problems.append(f"FSDP wrap class must be Qwen3_5DecoderLayer, got {wrap!r}")
    if cfg.generator.chat_template_kwargs.get("enable_thinking") is not True:
        problems.append("generator.chat_template_kwargs.enable_thinking must be explicitly true")
    if cfg.environment.env_class != ENV_ID:
        problems.append(f"environment.env_class must be {ENV_ID!r}")

    # Checkpointing: a ~19 GB write that must land on a writable volume with room, and
    # must be serialized across the concurrent runs.
    if cfg.trainer.ckpt_interval > 0:
        if not os.environ.get("PC_CKPT_LOCK"):
            problems.append("PC_CKPT_LOCK is unset; concurrent checkpoint saves would not be serialized")
        ckpt_parent = os.path.dirname(cfg.trainer.ckpt_path.rstrip("/")) or "/"
        probe = cfg.trainer.ckpt_path if os.path.isdir(cfg.trainer.ckpt_path) else ckpt_parent
        if not os.path.isdir(probe) or not os.access(probe, os.W_OK):
            problems.append(
                f"trainer.ckpt_path is not writable: {cfg.trainer.ckpt_path}. Checkpoints "
                "need the NFS volume mounted (see container/create_skyrl_pc.sh), or set "
                "PC_CKPT_INTERVAL=0 to disable them."
            )
        else:
            free_gb = shutil.disk_usage(probe).free / 2**30
            needed_gb = 19 * (MAX_TRAINING_STEPS // cfg.trainer.ckpt_interval)
            if free_gb < needed_gb:
                problems.append(
                    f"only {free_gb:.0f} GB free at {probe}; this run writes about "
                    f"{needed_gb} GB of checkpoints"
                )

    if problems:
        raise SystemExit("prompt_compare_rl pre-flight failed:\n  - " + "\n  - ".join(problems))


def _install_checkpoint_guard() -> None:
    """Serialize checkpoint saves across the three concurrent runs, and export adapters.

    Two problems this solves.

    **Memory.** ``save_checkpoints`` materializes the policy state dict (~18 GiB bf16)
    before writing it. Three runs reach step 4 and step 8 at the same step numbers, so
    without coordination all three can be inside that window at once -- roughly +54 GiB
    on top of the steady-state footprint, against a 96 GiB cgroup limit. This is the exact
    call the previous RPG run died inside (``ActorDiedError`` / SIGTERM in
    ``save_checkpoints``). A file lock shared by the three runs (``PC_CKPT_LOCK``, which
    ``config.py`` deliberately places at the experiment level rather than per-run) makes
    at most one save happen at a time. A run that is waiting is not doing GPU work, so the
    cost is wall-clock on the slower runs, not correctness.

    **Durability of the useful part.** Only ``policy/lora_adapter/`` (~175 MB) is trained
    content; the other ~18 GiB is a frozen copy of the base model that already exists in
    the HF cache. Copy the adapter into the run's export tree so the deltas survive
    independently of the big blobs (and are small enough to publish).
    """
    from skyrl.train.trainer import RayPPOTrainer

    lock_path = os.environ.get("PC_CKPT_LOCK")
    adapter_root = os.environ.get("PC_ADAPTER_EXPORT_DIR")
    if not lock_path:
        return
    if getattr(RayPPOTrainer.save_checkpoints, "_pc_guarded", False):
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
            print(f"[prompt_compare_rl] exported LoRA adapter -> {destination}", flush=True)
        except Exception as exc:  # noqa: BLE001 - never fail training over a copy
            print(f"[prompt_compare_rl] adapter export failed: {exc!r}", flush=True)

    def guarded_save_checkpoints(self):
        started = time.monotonic()
        with open(lock_path, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            waited = time.monotonic() - started
            if waited > 1.0:
                print(
                    f"[prompt_compare_rl] waited {waited:.0f}s for the shared checkpoint "
                    "lock (another prompt run was saving)",
                    flush=True,
                )
            try:
                ckpt_dir = original(self)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        _export_adapter(ckpt_dir, self.global_step)
        return ckpt_dir

    guarded_save_checkpoints._pc_guarded = True
    RayPPOTrainer.save_checkpoints = guarded_save_checkpoints


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(id=ENV_ID, entry_point=ENV_ENTRY_POINT)
    _install_checkpoint_guard()
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    _preflight(cfg)
    if os.environ.get("PC_PREFLIGHT_ONLY") == "1":
        # Used by scripts/run_tests.sh to exercise the entire launch path (env, config,
        # overrides, env-class import) on a machine with no free GPU.
        register(id=ENV_ID, entry_point=ENV_ENTRY_POINT)
        from skyrl_gym.envs.registration import load_env_creator

        load_env_creator(ENV_ENTRY_POINT)
        print(
            "[prompt_compare_rl] pre-flight OK for "
            f"{os.environ.get('PC_PROMPT_ID')} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}; "
            "exiting before Ray init (PC_PREFLIGHT_ONLY=1)"
        )
        return
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
