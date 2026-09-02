#!/usr/bin/env python3
"""Single source of truth for the prompt-comparison RL experiment.

Everything the three runs share (model, data, seeds, reward, every SkyRL
hyper-parameter) is defined ONCE here, and everything that must differ (prompt id, GPU,
output directories, Ray temp dir, LoRA sync dir, W&B run id) is derived from the prompt
id alone. ``run_overrides()`` therefore produces three override lists that are
byte-identical except for the run-scoped paths -- which is exactly the fairness
condition the experiment needs, and which ``tests/test_config.py`` asserts.

The bash launchers do not duplicate any of this: they call

    python -m prompt_compare_rl.config env  <prompt_id>     # KEY=VALUE lines
    python -m prompt_compare_rl.config args <prompt_id>     # SkyRL CLI overrides

so there is no second copy of the configuration to drift out of sync.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------------------
# Fixed experiment constants (identical across p1/p2/p3)
# --------------------------------------------------------------------------------------

PROMPT_IDS: Tuple[str, ...] = ("p1", "p2", "p3")

#: Default GPU assignment: p1 -> GPU 0, p2 -> GPU 1, p3 -> GPU 2. Override with
#: ``PC_GPUS`` (comma-separated, one physical GPU per prompt in PROMPT_IDS order) when
#: those cards are busy, e.g. ``PC_GPUS=5,6,7``. Whatever the mapping, the three runs get
#: one dedicated GPU each and no two share a card.
DEFAULT_GPUS: Tuple[int, ...] = (0, 1, 2)

#: Requirement (3): exactly eight optimizer steps.
MAX_TRAINING_STEPS = 8
#: Requirement (5): eval at step 0 (eval_before_train) and then every 4 steps -> 0, 4, 8.
EVAL_INTERVAL = 4
EVAL_STEPS: Tuple[int, ...] = (0, 4, 8)

#: ``train_batch_size == policy_mini_batch_size`` and ``update_epochs_per_batch == 1``
#: makes one SkyRL global step exactly one optimizer step, so "8 steps" is unambiguous.
TRAIN_BATCH_SIZE = 32
POLICY_MINI_BATCH_SIZE = 32
UPDATE_EPOCHS_PER_BATCH = 1

MODEL_PATH = "Qwen/Qwen3.5-9B"
#: Qwen3.5's text decoder block. The HF class exposes
#: ``_no_split_modules == {"Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"}``; pinning the
#: text block keeps FSDP2 from wrapping the (unused, text-only run) vision tower.
FSDP_TRANSFORMER_LAYER_CLS = "Qwen3_5DecoderLayer"
#: Text-only: PEFT targets every linear EXCEPT the vision tower, and vLLM zeroes every
#: multimodal limit. See README "Qwen3.5 / text-only" for why the trainer-side
#: ``policy.language_model_only`` stays False (it would break LoRA name mapping).
LORA_EXCLUDE_MODULES = ".*visual.*"

RPG_PROTO = "rpg_v9"

DEFAULT_EXP_TAG = "pcrl_v1"
DEFAULT_OUT_ROOT = "/work/data/rpg_rl_exps/prompt_compare_rl"
#: Checkpoints go to the 30 TB NFS volume, which is mounted at the SAME path inside the
#: `skyrl-pc` container (see container/create_skyrl_pc.sh), so a path printed in a log is
#: usable verbatim from the host shell. The container filesystem has ~49 GB free and a
#: single checkpoint is ~19 GB, so it cannot hold this experiment's six.
DEFAULT_CKPT_ROOT = "/data/rpg_rl_exps/prompt_compare_rl"
DEFAULT_RPG_SRC = "/work/ADS_shared/dataset_generation_code"
DEFAULT_SKYRL_DIR = "/work/SkyRL"
DEFAULT_WANDB_PROJECT = "rpg_prompt_compare"

#: Source datasets, requirement (1).
SOURCE_TRAIN = "rpg_v9/data_v9_deleaked/train.parquet"
SOURCE_VAL = "rpg_v9/data_v9_deleaked/validation_small.parquet"

# --- resource budget, used by the launcher pre-flight ---------------------------------
# Measured from a real checkpoint of this exact stack (POPE's global_step_12):
# model_world_size_1_rank_0.pt is 17.72 GiB = 9.41 B params in **bfloat16** plus 0.051 B
# fp32 (the LoRA parameters). The policy is therefore bf16, not fp32 -- roughly 18 GiB,
# not the 36 GiB a naive fp32 reading suggests.

#: Host RAM per concurrent job: ~18 GiB bf16 policy parked in RAM by ``colocate_all``
#: during every generation phase, plus the vLLM engine process, the Ray driver/entrypoint
#: and the object store. An estimate, not a measurement -- see README "Host RAM".
APPROX_HOST_GB_PER_JOB = 26

#: Extra host RAM while one run materializes its state dict inside ``save_checkpoints``.
#: Only one run can be in that window at a time (see the lock in ``main.py``).
CKPT_SAVE_SPIKE_GB = 18

#: GPU headroom a job needs beyond vLLM's ``gpu_memory_utilization`` share: the bf16
#: policy plus gradients, activations and the CUDA graph pool.
APPROX_GPU_GB_HEADROOM = 22

#: Ray sizes its plasma store at 30% of what it thinks node memory is. Inside the
#: container ``/proc/meminfo`` reports the PHYSICAL host (377 GB), not our 96 GiB cgroup,
#: so the default would be capped only by /dev/shm -- ~15 GiB per Ray cluster out of a
#: 16 GiB shared /dev/shm, which three concurrent clusters cannot satisfy. Cap it
#: explicitly; Ray reads this env var when computing the store size.
RAY_OBJECT_STORE_BYTES = 4 * 1024**3


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _parse_gpus(value: str) -> Tuple[int, ...]:
    """Parse ``PC_GPUS`` into one physical GPU id per prompt, in PROMPT_IDS order."""
    try:
        gpus = tuple(int(part) for part in value.split(",") if part.strip() != "")
    except ValueError:
        raise SystemExit(f"PC_GPUS must be comma-separated integers, got {value!r}")
    if len(gpus) != len(PROMPT_IDS):
        raise SystemExit(
            f"PC_GPUS lists {len(gpus)} GPUs but there are {len(PROMPT_IDS)} prompts "
            f"({list(PROMPT_IDS)}); got {value!r}"
        )
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f"PC_GPUS must not repeat a GPU (the runs must not share a card): {value!r}")
    if any(g < 0 for g in gpus):
        raise SystemExit(f"PC_GPUS must be non-negative: {value!r}")
    return gpus


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment-wide settings (identical for all three runs)."""

    exp_tag: str = field(default_factory=lambda: _env("PC_EXP_TAG", DEFAULT_EXP_TAG))
    out_root: str = field(default_factory=lambda: _env("PC_OUT_ROOT", DEFAULT_OUT_ROOT))
    #: Where the ~19 GB checkpoints go. Defaults to the NFS volume; set it to a path
    #: under ``out_root`` only if you have deliberately reduced ``PC_CKPT_INTERVAL``.
    ckpt_root: str = field(default_factory=lambda: _env("PC_CKPT_ROOT", DEFAULT_CKPT_ROOT))
    gpus: Tuple[int, ...] = field(
        default_factory=lambda: _parse_gpus(_env("PC_GPUS", ",".join(str(g) for g in DEFAULT_GPUS)))
    )
    rpg_src: str = field(default_factory=lambda: _env("RPG_SRC", DEFAULT_RPG_SRC))
    skyrl_dir: str = field(default_factory=lambda: _env("PC_SKYRL_DIR", DEFAULT_SKYRL_DIR))
    model: str = field(default_factory=lambda: _env("PC_MODEL", MODEL_PATH))
    wandb_project: str = field(default_factory=lambda: _env("PC_WANDB_PROJECT", DEFAULT_WANDB_PROJECT))

    # Knobs that are deliberately exposed but must stay equal across the three runs.
    seed: int = field(default_factory=lambda: _env_int("PC_SEED", 42))
    lr: float = field(default_factory=lambda: _env_float("PC_LR", 1.0e-5))
    num_warmup_steps: int = field(default_factory=lambda: _env_int("PC_WARMUP_STEPS", 1))
    lora_rank: int = field(default_factory=lambda: _env_int("PC_LORA_RANK", 16))
    lora_alpha: int = field(default_factory=lambda: _env_int("PC_LORA_ALPHA", 32))
    n_samples_per_prompt: int = field(default_factory=lambda: _env_int("PC_GROUP_SIZE", 8))
    eval_n_samples_per_prompt: int = field(default_factory=lambda: _env_int("PC_EVAL_N", 2))
    gpu_memory_utilization: float = field(default_factory=lambda: _env_float("PC_GPU_MEM_UTIL", 0.5))
    max_prompt_length: int = field(default_factory=lambda: _env_int("PC_MAX_PROMPT_LEN", 18432))
    max_generate_length: int = field(default_factory=lambda: _env_int("PC_MAX_GEN_LEN", 8192))
    max_model_len: int = field(default_factory=lambda: _env_int("PC_MAX_MODEL_LEN", 32768))
    max_tokens_per_microbatch: int = field(default_factory=lambda: _env_int("PC_MAX_TOK_PER_MICROBATCH", 8192))
    #: Checkpoint every N steps; 4 gives one at step 4 and one at step 8, matching the
    #: evaluation cadence. ``0`` disables checkpoint writing entirely. Each checkpoint is
    #: ~19 GB (bf16 policy + optimizer + adapter), so this requires ``ckpt_root`` to
    #: point at the NFS volume -- see README "Checkpoints".
    ckpt_interval: int = field(default_factory=lambda: _env_int("PC_CKPT_INTERVAL", 4))
    max_turns: int = field(default_factory=lambda: _env_int("PC_MAX_TURNS", 33))
    #: "none" (default) restarts a re-launched run from step 0; pair "latest" with a
    #: non-zero PC_CKPT_INTERVAL if you want a crash to resume mid-training instead.
    resume_mode: str = field(default_factory=lambda: _env("PC_RESUME_MODE", "none"))
    max_env_workers: int = field(default_factory=lambda: _env_int("PC_MAX_ENV_WORKERS", 16))

    @property
    def exp_dir(self) -> Path:
        return Path(self.out_root) / self.exp_tag

    @property
    def prompt_dir(self) -> Path:
        return self.exp_dir / "prompts"

    @property
    def dataset_root(self) -> Path:
        return self.exp_dir / "datasets"

    @property
    def source_train(self) -> Path:
        return Path(self.rpg_src) / SOURCE_TRAIN

    @property
    def source_val(self) -> Path:
        return Path(self.rpg_src) / SOURCE_VAL

    def dataset_dir(self, prompt_id: str) -> Path:
        return self.dataset_root / prompt_id

    def train_parquet(self, prompt_id: str) -> Path:
        return self.dataset_dir(prompt_id) / "train.parquet"

    def val_parquet(self, prompt_id: str) -> Path:
        return self.dataset_dir(prompt_id) / "validation_small.parquet"

    def prompt_file(self, prompt_id: str) -> Path:
        return self.prompt_dir / f"{prompt_id}.txt"

    # -- per-run, mutually exclusive locations (requirement 7) --------------------------

    def run_dir(self, prompt_id: str) -> Path:
        return self.exp_dir / "runs" / prompt_id

    @property
    def ckpt_exp_dir(self) -> Path:
        return Path(self.ckpt_root) / self.exp_tag

    @property
    def ckpt_lock_path(self) -> Path:
        """Shared across the three runs on purpose: it serializes their checkpoint saves."""
        return self.exp_dir / "checkpoint_save.lock"

    def run_paths(self, prompt_id: str) -> Dict[str, Path]:
        run = self.run_dir(prompt_id)
        return {
            "run_dir": run,
            # On the NFS volume, not under out_root -- ~19 GB per checkpoint.
            "ckpt_path": self.ckpt_exp_dir / "runs" / prompt_id / "checkpoints",
            "export_path": run / "exports",
            # The 175 MB of each checkpoint that is actually trained content, copied out
            # so the deltas survive independently of the 19 GB blobs.
            "adapters_dir": run / "exports" / "adapters",
            "log_path": run / "logs",
            "ray_tmpdir": run / "ray_tmp",
            "lora_sync_path": run / "lora_sync",
            "episode_scratch": run / "episode_scratch",
            "wandb_dir": run / "wandb",
            "triton_cache": run / "cache" / "triton",
            "inductor_cache": run / "cache" / "inductor",
            "vllm_cache": run / "cache" / "vllm",
        }

    def gpu_for(self, prompt_id: str) -> int:
        return self.gpus[PROMPT_IDS.index(prompt_id)]

    def run_name(self, prompt_id: str) -> str:
        return f"{self.exp_tag}_{prompt_id}"

    def wandb_run_id(self, prompt_id: str) -> str:
        """Deterministic W&B id: a re-launch of the same (exp_tag, prompt) resumes it."""
        return f"{self.exp_tag}-{prompt_id}"


def validate_prompt_id(prompt_id: str) -> str:
    if prompt_id not in PROMPT_IDS:
        raise SystemExit(f"unknown prompt id {prompt_id!r}; expected one of {list(PROMPT_IDS)}")
    return prompt_id


def prompt_sha256(cfg: ExperimentConfig, prompt_id: str) -> str:
    from prompt_compare_rl.prompts import EXPECTED_SHA256

    return EXPECTED_SHA256[prompt_id]


# --------------------------------------------------------------------------------------
# Environment for one run
# --------------------------------------------------------------------------------------


def run_env(cfg: ExperimentConfig, prompt_id: str) -> Dict[str, str]:
    """Environment variables for one run: isolation + protocol + prompt selection."""
    validate_prompt_id(prompt_id)
    paths = cfg.run_paths(prompt_id)
    package_parent = str(Path(__file__).resolve().parent.parent)
    # Deduplicate: run_one.sh already puts the package parent on PYTHONPATH in order to
    # import this module, and the value is re-exported into every Ray worker.
    entries = [package_parent] + [
        e for e in os.environ.get("PYTHONPATH", "").split(":") if e and e != package_parent
    ]
    pythonpath = ":".join(entries)
    env = {
        # -- which prompt, and the digest every consumer re-checks -----------------------
        "PC_PROMPT_ID": prompt_id,
        "PC_PROMPT_SHA256": prompt_sha256(cfg, prompt_id),
        "PC_EXP_TAG": cfg.exp_tag,
        "PC_OUT_ROOT": cfg.out_root,
        "PC_RUN_DIR": str(paths["run_dir"]),
        # -- RPG protocol / sources (requirement 8) --------------------------------------
        "RPG_SRC": cfg.rpg_src,
        "RPG_PROTO": RPG_PROTO,
        "RPG_SYSTEM_PROMPT_FILE": str(cfg.prompt_file(prompt_id)),
        "RPG_SYSTEM_PROMPT_SHA256": prompt_sha256(cfg, prompt_id),
        # Per-episode CSV scratch for the `code` tool, kept inside this run's tree.
        "RPG_DATA_ROOT": str(paths["episode_scratch"]),
        # -- checkpointing ---------------------------------------------------------------
        # SHARED by all three runs: main.py takes this lock around save_checkpoints() so
        # only one run at a time materializes an ~18 GiB state dict.
        "PC_CKPT_LOCK": str(cfg.ckpt_lock_path),
        "PC_ADAPTER_EXPORT_DIR": str(paths["adapters_dir"]),
        # -- Ray / runtime isolation (requirement 7) -------------------------------------
        # RAY_ADDRESS=local forces ray.init() to start a FRESH local cluster instead of
        # attaching to a sibling job's cluster; RAY_TMPDIR gives it its own session tree.
        "RAY_ADDRESS": "local",
        "RAY_TMPDIR": str(paths["ray_tmpdir"]),
        "CUDA_VISIBLE_DEVICES": str(cfg.gpu_for(prompt_id)),
        # Compiler caches are per-run so three concurrent JIT compilations cannot race.
        "TRITON_CACHE_DIR": str(paths["triton_cache"]),
        "TORCHINDUCTOR_CACHE_DIR": str(paths["inductor_cache"]),
        "VLLM_CACHE_ROOT": str(paths["vllm_cache"]),
        # -- import path for the env class, forwarded into Ray workers -------------------
        "PYTHONPATH": pythonpath,
        "SKYRL_PYTHONPATH_EXPORT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Cap Ray's plasma store; the default is derived from the physical host's memory
        # and would claim ~15 GiB of a 16 GiB /dev/shm that all three runs share.
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(RAY_OBJECT_STORE_BYTES),
        # -- W&B: distinct id per prompt, stable across re-launches (caution 1) ----------
        "WANDB_PROJECT": cfg.wandb_project,
        "WANDB_RUN_ID": cfg.wandb_run_id(prompt_id),
        "WANDB_NAME": cfg.run_name(prompt_id),
        "WANDB_RESUME": "allow",
        "WANDB_DIR": str(paths["wandb_dir"]),
    }
    return env


# --------------------------------------------------------------------------------------
# SkyRL overrides for one run
# --------------------------------------------------------------------------------------


def run_overrides(cfg: ExperimentConfig, prompt_id: str) -> List[str]:
    """The full SkyRL CLI override list for one run."""
    validate_prompt_id(prompt_id)
    paths = cfg.run_paths(prompt_id)
    return [
        # ---- data (requirement 1): the de-leaked v9 train set + validation_small ------
        f"data.train_data=['{cfg.train_parquet(prompt_id)}']",
        f"data.val_data=['{cfg.val_parquet(prompt_id)}']",
        # ---- model -------------------------------------------------------------------
        f"trainer.policy.model.path={cfg.model}",
        "trainer.strategy=fsdp",
        # Requirement 8: pin the Qwen3.5 TEXT decoder block for FSDP2 wrapping.
        f"trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap={FSDP_TRANSFORMER_LAYER_CLS}",
        f"trainer.policy.model.lora.rank={cfg.lora_rank}",
        f"trainer.policy.model.lora.alpha={cfg.lora_alpha}",
        f"trainer.policy.model.lora.target_modules=all-linear",
        # Requirement 8, text-only training scope: no LoRA on the vision tower.
        f"trainer.policy.model.lora.exclude_modules={LORA_EXCLUDE_MODULES}",
        # Requirement 7: the LoRA adapter handoff dir MUST be per-run. The SkyRL default
        # is a single shared /tmp path -- three concurrent jobs would overwrite each
        # other's adapter between the trainer writing it and vLLM loading it.
        f"trainer.policy.model.lora.lora_sync_path={paths['lora_sync_path']}",
        # ---- placement: one GPU per job ----------------------------------------------
        "trainer.placement.colocate_all=true",
        "trainer.placement.policy_num_gpus_per_node=1",
        "trainer.placement.ref_num_gpus_per_node=1",
        "generator.inference_engine.num_engines=1",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.weight_sync_backend=nccl",
        f"generator.inference_engine.gpu_memory_utilization={cfg.gpu_memory_utilization}",
        # Requirement 8, text-only inference: zeroes every multimodal limit so vLLM does
        # no image/video profiling and reserves no encoder cache.
        "generator.inference_engine.language_model_only=true",
        f"generator.inference_engine.max_num_batched_tokens={cfg.max_prompt_length}",
        "generator.inference_engine.max_num_seqs=512",
        f"generator.inference_engine.engine_init_kwargs.max_model_len={cfg.max_model_len}",
        # ---- algorithm (identical across runs, requirement 4) -------------------------
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.algorithm.grpo_norm_by_std=false",
        "trainer.algorithm.advantage_batch_normalize=false",
        "trainer.algorithm.loss_reduction=token_mean",
        "trainer.algorithm.use_kl_loss=false",
        # ---- optimization ------------------------------------------------------------
        f"trainer.policy.optimizer_config.lr={cfg.lr}",
        "trainer.policy.optimizer_config.max_grad_norm=1.0",
        "trainer.policy.optimizer_config.scheduler=constant_with_warmup",
        f"trainer.policy.optimizer_config.num_warmup_steps={cfg.num_warmup_steps}",
        f"trainer.seed={cfg.seed}",
        # ---- batching: 1 global step == 1 optimizer step (requirement 3) -------------
        "trainer.epochs=1",
        f"trainer.max_training_steps={MAX_TRAINING_STEPS}",
        f"trainer.train_batch_size={TRAIN_BATCH_SIZE}",
        f"trainer.policy_mini_batch_size={POLICY_MINI_BATCH_SIZE}",
        f"trainer.update_epochs_per_batch={UPDATE_EPOCHS_PER_BATCH}",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.micro_forward_batch_size_per_gpu=1",
        f"trainer.max_tokens_per_microbatch={cfg.max_tokens_per_microbatch}",
        "trainer.remove_microbatch_padding=false",
        f"trainer.max_prompt_length={cfg.max_prompt_length}",
        # ---- generation --------------------------------------------------------------
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
        # Requirement 8: thinking mode stated explicitly rather than left to the
        # template default (Qwen3.5 emits "<think>\n" unless enable_thinking is false).
        "generator.chat_template_kwargs.enable_thinking=true",
        # ---- environment -------------------------------------------------------------
        "environment.env_class=rpg_prompt_compare",
        f"environment.skyrl_gym.max_env_workers={cfg.max_env_workers}",
        # ---- evaluation cadence (requirement 5): steps 0, 4, 8 ------------------------
        "trainer.eval_before_train=true",
        f"trainer.eval_interval={EVAL_INTERVAL}",
        "trainer.dump_eval_results=true",
        # ---- checkpoint / logging / isolation ----------------------------------------
        f"trainer.ckpt_interval={cfg.ckpt_interval}",
        # Keep every checkpoint: the NFS volume has 30 TB free, and pruning would delete
        # the step-4 checkpoint the moment step 8 is written.
        "trainer.max_ckpts_to_keep=-1",
        "trainer.hf_save_interval=-1",
        f"trainer.resume_mode={cfg.resume_mode}",
        f"trainer.ckpt_path={paths['ckpt_path']}",
        f"trainer.export_path={paths['export_path']}",
        f"trainer.log_path={paths['log_path']}",
        "trainer.logger=wandb",
        f"trainer.project_name={cfg.wandb_project}",
        f"trainer.run_name={cfg.run_name(prompt_id)}",
    ]


def run_manifest(cfg: ExperimentConfig, prompt_id: str) -> Dict[str, object]:
    """Everything needed to reproduce or audit one run."""
    return {
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha256(cfg, prompt_id),
        "prompt_file": str(cfg.prompt_file(prompt_id)),
        "exp_tag": cfg.exp_tag,
        "gpu": cfg.gpu_for(prompt_id),
        "run_name": cfg.run_name(prompt_id),
        "wandb_run_id": cfg.wandb_run_id(prompt_id),
        "wandb_project": cfg.wandb_project,
        "model": cfg.model,
        "rpg_proto": RPG_PROTO,
        "max_training_steps": MAX_TRAINING_STEPS,
        "eval_steps": list(EVAL_STEPS),
        "source_train": str(cfg.source_train),
        "source_val": str(cfg.source_val),
        "ckpt_path": str(cfg.run_paths(prompt_id)["ckpt_path"]),
        "ckpt_interval": cfg.ckpt_interval,
        "adapters_dir": str(cfg.run_paths(prompt_id)["adapters_dir"]),
        "train_parquet": str(cfg.train_parquet(prompt_id)),
        "val_parquet": str(cfg.val_parquet(prompt_id)),
        "env": run_env(cfg, prompt_id),
        "overrides": run_overrides(cfg, prompt_id),
    }


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("usage: python -m prompt_compare_rl.config {env|args|args0|manifest|paths|budget} <prompt_id>")
        return 2
    what, prompt_id = argv[0], validate_prompt_id(argv[1])
    cfg = ExperimentConfig()
    if what == "env":
        for key, value in run_env(cfg, prompt_id).items():
            print(f"{key}={value}")
    elif what == "args":
        # Human-readable form. Shell callers must use `args0` instead: quoting inside a
        # variable is not re-interpreted by word splitting, so a `$(...)` capture of this
        # would pass literal quote characters through to OmegaConf.
        print(" ".join(shlex.quote(a) for a in run_overrides(cfg, prompt_id)))
    elif what == "args0":
        payload = "".join(a + "\0" for a in run_overrides(cfg, prompt_id))
        sys.stdout.write(payload)
    elif what == "manifest":
        print(json.dumps(run_manifest(cfg, prompt_id), indent=2))
    elif what == "budget":
        # Single source of truth for the launcher's pre-flight arithmetic.
        print(f"HOST_GB_PER_JOB={APPROX_HOST_GB_PER_JOB}")
        print(f"CKPT_SPIKE_GB={CKPT_SAVE_SPIKE_GB if cfg.ckpt_interval > 0 else 0}")
        print(f"GPU_GB_HEADROOM={APPROX_GPU_GB_HEADROOM}")
        print(f"GPU_MEM_UTIL={cfg.gpu_memory_utilization}")
        print(f"CKPT_ROOT={cfg.ckpt_exp_dir}")
        print(f"CKPT_INTERVAL={cfg.ckpt_interval}")
        print(f"GPUS={','.join(str(g) for g in cfg.gpus)}")
    elif what == "paths":
        for key, value in cfg.run_paths(prompt_id).items():
            print(f"{key}={value}")
    else:
        raise SystemExit(f"unknown subcommand {what!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
