"""The three runs differ only in prompt-scoped values, and the SkyRL config they resolve
to satisfies the experiment's requirements."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prompt_compare_rl.config import (
    EVAL_INTERVAL,
    EVAL_STEPS,
    DEFAULT_GPUS,
    MAX_TRAINING_STEPS,
    PROMPT_IDS,
    ExperimentConfig,
    run_env,
    run_manifest,
    run_overrides,
)

RUN_SCOPED_PREFIXES = (
    "data.train_data",
    "data.val_data",
    "trainer.ckpt_path",
    "trainer.export_path",
    "trainer.log_path",
    "trainer.run_name",
    "trainer.policy.model.lora.lora_sync_path",
)


def _shared(overrides):
    return [o for o in overrides if not o.startswith(RUN_SCOPED_PREFIXES)]


def test_hyperparameters_identical_across_runs(cfg):
    reference = _shared(run_overrides(cfg, "p1"))
    for pid in PROMPT_IDS[1:]:
        assert _shared(run_overrides(cfg, pid)) == reference


def test_every_run_scoped_override_is_actually_present(cfg):
    for pid in PROMPT_IDS:
        overrides = run_overrides(cfg, pid)
        for prefix in RUN_SCOPED_PREFIXES:
            assert any(o.startswith(prefix + "=") for o in overrides), f"{pid} missing {prefix}"


def test_gpu_assignment_default(cfg):
    """The documented default is p1->0, p2->1, p3->2."""
    assert DEFAULT_GPUS == (0, 1, 2)
    assert cfg.gpus == DEFAULT_GPUS
    for index, pid in enumerate(PROMPT_IDS):
        assert cfg.gpu_for(pid) == DEFAULT_GPUS[index]
        assert run_env(cfg, pid)["CUDA_VISIBLE_DEVICES"] == str(DEFAULT_GPUS[index])


def test_gpu_assignment_override(monkeypatch):
    """PC_GPUS moves the runs to other cards, one dedicated GPU each."""
    monkeypatch.setenv("PC_GPUS", "5,6,7")
    other = ExperimentConfig()
    assert other.gpus == (5, 6, 7)
    assert [run_env(other, pid)["CUDA_VISIBLE_DEVICES"] for pid in PROMPT_IDS] == ["5", "6", "7"]
    assert run_manifest(other, "p3")["gpu"] == 7
    # Nothing else may change when only the GPU mapping does.
    base = ExperimentConfig.__new__(ExperimentConfig)
    del base
    assert _shared(run_overrides(other, "p1")) == _shared(run_overrides(ExperimentConfig(), "p1"))


@pytest.mark.parametrize("value", ["5,6", "5,6,7,8", "5,5,6", "a,b,c", "-1,2,3"])
def test_gpu_override_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv("PC_GPUS", value)
    with pytest.raises(SystemExit):
        ExperimentConfig()


def test_wandb_ids_are_distinct_and_deterministic(cfg):
    ids = {pid: run_env(cfg, pid)["WANDB_RUN_ID"] for pid in PROMPT_IDS}
    assert len(set(ids.values())) == len(PROMPT_IDS)
    # Same inputs -> same id, so a re-launch after a crash resumes the same W&B run.
    again = {pid: ExperimentConfig().wandb_run_id(pid) for pid in PROMPT_IDS}
    assert ids == again
    for pid in PROMPT_IDS:
        assert run_env(cfg, pid)["WANDB_RESUME"] == "allow"


def test_run_paths_are_pairwise_disjoint(cfg):
    seen = {}
    for pid in PROMPT_IDS:
        for key, value in cfg.run_paths(pid).items():
            resolved = str(Path(value).resolve())
            assert resolved not in seen, f"{pid}.{key} collides with {seen.get(resolved)}"
            seen[resolved] = f"{pid}.{key}"


def test_ray_and_cache_isolation(cfg):
    envs = {pid: run_env(cfg, pid) for pid in PROMPT_IDS}
    for key in ("RAY_TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "VLLM_CACHE_ROOT", "RPG_DATA_ROOT", "WANDB_DIR"):
        values = [envs[pid][key] for pid in PROMPT_IDS]
        assert len(set(values)) == len(PROMPT_IDS), key
    for pid in PROMPT_IDS:
        assert envs[pid]["RAY_ADDRESS"] == "local"
        assert envs[pid]["SKYRL_PYTHONPATH_EXPORT"] == "1"
        assert envs[pid]["RPG_PROTO"] == "rpg_v9"


def test_pythonpath_points_at_the_package_parent(cfg):
    parent = str(Path(__file__).resolve().parent.parent.parent)
    assert run_env(cfg, "p1")["PYTHONPATH"].split(":")[0] == parent


def test_resolved_skyrl_config_meets_requirements(cfg, monkeypatch):
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils.utils import validate_cfg

    monkeypatch.setenv("WANDB_API_KEY", "test-placeholder")
    resolved = SkyRLTrainConfig.from_cli_overrides(run_overrides(cfg, "p1"))
    validate_cfg(resolved)

    # exactly 8 optimizer steps
    assert resolved.trainer.max_training_steps == MAX_TRAINING_STEPS == 8
    assert resolved.trainer.epochs == 1
    assert resolved.trainer.update_epochs_per_batch == 1
    assert resolved.trainer.train_batch_size == resolved.trainer.policy_mini_batch_size

    # eval at 0 / 4 / 8
    assert resolved.trainer.eval_before_train is True
    assert resolved.trainer.eval_interval == EVAL_INTERVAL
    assert EVAL_STEPS == (0, 4, 8)

    # required Qwen3.5 / SkyRL settings
    assert resolved.trainer.strategy == "fsdp"
    assert resolved.trainer.policy.fsdp_config.wrap_policy == {
        "transformer_layer_cls_to_wrap": "Qwen3_5DecoderLayer"
    }
    assert resolved.generator.inference_engine.language_model_only is True
    assert resolved.generator.chat_template_kwargs == {"enable_thinking": True}
    assert resolved.generator.batched is False  # chat_template_kwargs requires it
    assert resolved.trainer.policy.model.path == "Qwen/Qwen3.5-9B"

    # reward / algorithm identical and explicit
    assert resolved.trainer.algorithm.advantage_estimator == "grpo"
    assert resolved.trainer.algorithm.use_kl_loss is False
    assert resolved.trainer.seed == cfg.seed

    # eval sampling params are stated in full (they do not inherit from sampling_params)
    assert resolved.generator.eval_sampling_params.temperature == 1.0
    assert resolved.generator.eval_sampling_params.max_generate_length == cfg.max_generate_length

    # single GPU per job
    assert resolved.trainer.placement.policy_num_gpus_per_node == 1
    assert resolved.generator.inference_engine.num_engines == 1
    assert resolved.trainer.placement.colocate_all is True


def test_lora_sync_path_is_not_the_shared_default(cfg):
    from skyrl.train.config import SkyRLTrainConfig

    default = SkyRLTrainConfig.from_cli_overrides([]).trainer.policy.model.lora.lora_sync_path
    for pid in PROMPT_IDS:
        resolved = SkyRLTrainConfig.from_cli_overrides(run_overrides(cfg, pid))
        assert resolved.trainer.policy.model.lora.lora_sync_path != default


def test_pythonpath_reaches_ray_workers(cfg, monkeypatch):
    """SkyRL only forwards PYTHONPATH when SKYRL_PYTHONPATH_EXPORT is set."""
    import importlib

    env = run_env(cfg, "p1")
    monkeypatch.setenv("SKYRL_PYTHONPATH_EXPORT", env["SKYRL_PYTHONPATH_EXPORT"])
    monkeypatch.setenv("PYTHONPATH", env["PYTHONPATH"])
    monkeypatch.setenv("WANDB_API_KEY", "test-placeholder")

    import skyrl.env_vars as env_vars
    importlib.reload(env_vars)
    import skyrl.train.utils.utils as utils
    importlib.reload(utils)

    from skyrl.train.config import SkyRLTrainConfig

    resolved = SkyRLTrainConfig.from_cli_overrides(run_overrides(cfg, "p1"))
    forwarded = utils.prepare_runtime_environment(resolved)
    assert forwarded.get("PYTHONPATH") == env["PYTHONPATH"]


def test_manifest_is_self_describing(cfg):
    manifest = run_manifest(cfg, "p2")
    assert manifest["prompt_id"] == "p2"
    assert manifest["gpu"] == 1
    assert manifest["max_training_steps"] == 8
    assert manifest["eval_steps"] == [0, 4, 8]
    assert manifest["rpg_proto"] == "rpg_v9"
    assert manifest["env"]["PC_PROMPT_SHA256"] == manifest["prompt_sha256"]


def test_checkpoints_go_to_the_nfs_volume(cfg):
    """~19 GB per checkpoint x 6 cannot fit on the container filesystem, so ckpt_path
    must live under ckpt_root (the /data mount), separate from out_root."""
    assert cfg.ckpt_interval == 4
    out_root = Path(cfg.out_root).resolve()
    for pid in PROMPT_IDS:
        ckpt = Path(cfg.run_paths(pid)["ckpt_path"]).resolve()
        assert Path(cfg.ckpt_root).resolve() in ckpt.parents
        assert out_root not in ckpt.parents, f"{pid}: checkpoints must not land under out_root"
        assert f"trainer.ckpt_path={ckpt}" in run_overrides(cfg, pid)
    # Keep both step-4 and step-8; pruning would delete step 4 when step 8 is written.
    assert "trainer.max_ckpts_to_keep=-1" in run_overrides(cfg, "p1")


def test_checkpoint_paths_are_per_run_but_the_save_lock_is_shared(cfg):
    """The lock is the one thing the three runs MUST share -- it serializes the ~18 GiB
    state-dict materialization that would otherwise happen three times at once."""
    ckpt_paths = {str(Path(cfg.run_paths(p)["ckpt_path"]).resolve()) for p in PROMPT_IDS}
    assert len(ckpt_paths) == len(PROMPT_IDS)
    locks = {run_env(cfg, p)["PC_CKPT_LOCK"] for p in PROMPT_IDS}
    assert len(locks) == 1, f"the checkpoint lock must be shared, got {locks}"
    adapters = {run_env(cfg, p)["PC_ADAPTER_EXPORT_DIR"] for p in PROMPT_IDS}
    assert len(adapters) == len(PROMPT_IDS)


def test_ray_object_store_is_capped(cfg):
    """Ray sizes its plasma store from the physical host's memory, which would claim
    ~15 GiB of a 16 GiB /dev/shm shared by all three runs."""
    from prompt_compare_rl.config import RAY_OBJECT_STORE_BYTES

    for pid in PROMPT_IDS:
        value = run_env(cfg, pid)["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"]
        assert int(value) == RAY_OBJECT_STORE_BYTES
        assert int(value) * len(PROMPT_IDS) < 16 * 1024**3, "three stores must fit in /dev/shm"


def test_budget_constants_reflect_the_measured_bf16_policy(cfg):
    """The saved policy is bf16 (~18 GiB), not fp32 (~36 GiB); the launcher pre-flight
    thresholds derive from that."""
    from prompt_compare_rl.config import (
        APPROX_GPU_GB_HEADROOM,
        APPROX_HOST_GB_PER_JOB,
        CKPT_SAVE_SPIKE_GB,
    )

    assert 18 <= CKPT_SAVE_SPIKE_GB <= 20
    assert APPROX_HOST_GB_PER_JOB >= CKPT_SAVE_SPIKE_GB
    assert APPROX_GPU_GB_HEADROOM >= 18
