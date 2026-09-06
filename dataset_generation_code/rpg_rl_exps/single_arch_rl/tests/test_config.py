"""The two runs must differ in the archetype and in nothing else that matters."""

from __future__ import annotations

import re

import pytest

from single_arch_rl import config as cfgmod
from single_arch_rl.config import (
    RUN_IDS,
    STEPS_PER_EPOCH,
    TRAIN_ARCHETYPE,
    TRAIN_WORLDS_PER_RUN,
    ExperimentConfig,
    run_env,
    run_manifest,
    run_overrides,
)


def _strip_run_scoped(overrides, cfg, run_id):
    """Drop the overrides that legitimately differ: the train parquet and run paths."""
    run_dir = str(cfg.run_dir(run_id))
    ckpt_dir = str(cfg.ckpt_exp_dir / "runs" / run_id)
    kept = []
    for item in overrides:
        if run_dir in item or ckpt_dir in item or str(cfg.train_parquet(run_id)) in item:
            continue
        if item.startswith("trainer.run_name="):
            continue
        kept.append(item)
    return kept


def test_the_two_runs_are_the_two_archetypes():
    assert RUN_IDS == ("easy", "hard")
    assert TRAIN_ARCHETYPE == {"easy": "dose_window", "hard": "confounded_reversal"}


def test_overrides_differ_only_in_data_and_run_paths():
    cfg = ExperimentConfig()
    easy = _strip_run_scoped(run_overrides(cfg, "easy"), cfg, "easy")
    hard = _strip_run_scoped(run_overrides(cfg, "hard"), cfg, "hard")
    assert easy == hard


def test_each_run_points_at_its_own_train_parquet_and_the_shared_validation_set():
    cfg = ExperimentConfig()
    easy, hard = run_overrides(cfg, "easy"), run_overrides(cfg, "hard")
    assert f"data.train_data=['{cfg.train_parquet('easy')}']" in easy
    assert f"data.train_data=['{cfg.train_parquet('hard')}']" in hard
    assert cfg.train_parquet("easy") != cfg.train_parquet("hard")
    # Requirement (2): the SAME validation set for both runs.
    assert f"data.val_data=['{cfg.val_parquet}']" in easy
    assert f"data.val_data=['{cfg.val_parquet}']" in hard
    assert cfg.val_parquet.name == "validation_small.parquet"


def test_validation_source_is_the_committed_validation_small():
    cfg = ExperimentConfig()
    assert cfg.source_val.as_posix().endswith("rpg_v9/data_v9_deleaked/validation_small.parquet")
    assert cfg.source_train.as_posix().endswith("rpg_v9/data_v9_deleaked/train.parquet")


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_run_scoped_paths_are_pairwise_disjoint(run_id):
    cfg = ExperimentConfig()
    mine = cfg.run_paths(run_id)
    other_id = [r for r in RUN_IDS if r != run_id][0]
    theirs = cfg.run_paths(other_id)
    for key, path in mine.items():
        assert path != theirs[key], f"{key} is shared between the runs"
    # The checkpoint lock is shared ON PURPOSE -- it serializes the 18 GiB saves.
    assert cfg.ckpt_lock_path == ExperimentConfig().ckpt_lock_path


def test_all_gpus_go_to_one_tensor_parallel_run():
    """Host RAM, not GPU memory, forces this: two concurrent single-GPU runs need
    ~165 GiB against a 96 GiB ceiling, and N data-parallel engines would each keep their
    own ~23 GB host sleep buffer. See docs/gpu_sizing.md."""
    cfg = ExperimentConfig()
    assert cfg.gpus == (0, 4)
    assert cfg.num_gpus == 2
    for run_id in RUN_IDS:
        assert run_env(cfg, run_id)["CUDA_VISIBLE_DEVICES"] == "0,4"
    overrides = run_overrides(cfg, "easy")
    assert "trainer.placement.policy_num_gpus_per_node=2" in overrides
    assert "trainer.placement.ref_num_gpus_per_node=2" in overrides
    # ONE engine, tensor-parallel over the cards -- not one engine per card.
    assert "generator.inference_engine.num_engines=1" in overrides
    assert "generator.inference_engine.tensor_parallel_size=2" in overrides


def test_gpus_env_var_is_validated(monkeypatch):
    monkeypatch.setenv("SA_GPUS", "5,5")
    with pytest.raises(SystemExit, match="repeat"):
        ExperimentConfig()
    monkeypatch.setenv("SA_GPUS", "")
    assert ExperimentConfig().gpus == (0, 4)          # empty falls back to the default
    monkeypatch.setenv("SA_GPUS", "6")
    cfg = ExperimentConfig()
    assert cfg.gpus == (6,) and cfg.num_gpus == 1
    assert "generator.inference_engine.tensor_parallel_size=1" in run_overrides(cfg, "easy")
    monkeypatch.setenv("SA_GPUS", "1,2,3")
    assert ExperimentConfig().gpus == (1, 2, 3)


def test_checkpoint_every_two_steps():
    cfg = ExperimentConfig()
    assert cfg.ckpt_interval == 2                        # spec (3)
    assert "trainer.ckpt_interval=2" in run_overrides(cfg, "easy")
    # Nothing may prune the earlier checkpoints.
    assert "trainer.max_ckpts_to_keep=-1" in run_overrides(cfg, "easy")


def test_one_global_step_is_one_optimizer_step():
    cfg = ExperimentConfig()
    overrides = run_overrides(cfg, "easy")
    assert f"trainer.train_batch_size={cfgmod.TRAIN_BATCH_SIZE}" in overrides
    assert f"trainer.policy_mini_batch_size={cfgmod.POLICY_MINI_BATCH_SIZE}" in overrides
    assert cfgmod.TRAIN_BATCH_SIZE == cfgmod.POLICY_MINI_BATCH_SIZE
    assert "trainer.update_epochs_per_batch=1" in overrides
    assert STEPS_PER_EPOCH * cfgmod.TRAIN_BATCH_SIZE == TRAIN_WORLDS_PER_RUN


def test_epochs_cover_the_requested_number_of_steps(monkeypatch):
    monkeypatch.setenv("SA_MAX_STEPS", "12")
    cfg = ExperimentConfig()
    assert cfg.epochs * STEPS_PER_EPOCH >= cfg.max_training_steps
    monkeypatch.setenv("SA_MAX_STEPS", "7")
    assert ExperimentConfig().epochs == 3


def test_wandb_run_id_is_deterministic_and_resumes():
    """Requirement (5): a relaunch keeps the same W&B run so monitoring continues."""
    cfg = ExperimentConfig()
    for run_id in RUN_IDS:
        env = run_env(cfg, run_id)
        assert env["WANDB_RUN_ID"] == f"{cfg.exp_tag}-{run_id}"
        assert env["WANDB_RESUME"] == "allow"
        assert run_env(ExperimentConfig(), run_id)["WANDB_RUN_ID"] == env["WANDB_RUN_ID"]
    assert run_env(cfg, "easy")["WANDB_RUN_ID"] != run_env(cfg, "hard")["WANDB_RUN_ID"]
    # ... and the trainer must actually be asked to resume.
    assert "trainer.resume_mode=latest" in run_overrides(cfg, "easy")


def test_checkpoints_go_to_the_data_volume_not_the_container_filesystem():
    """Requirement (3): ~19 GB per checkpoint, /work has ~49 GB free."""
    cfg = ExperimentConfig()
    for run_id in RUN_IDS:
        ckpt = str(cfg.run_paths(run_id)["ckpt_path"])
        assert ckpt.startswith("/data/"), ckpt
        assert f"trainer.ckpt_path={ckpt}" in run_overrides(cfg, run_id)


def test_gpu_footprint_stays_modest_on_a_shared_box():
    """The two runs must not each claim most of a 96 GB card.

    vLLM's cost is gpu_memory_utilization x total PLUS one Gated-DeltaNet state slot per
    max_num_seqs (Qwen3.5 is hybrid: 24 of 32 layers are GDN), so both knobs are pinned.
    """
    cfg = ExperimentConfig()
    assert cfg.gpu_memory_utilization <= 0.4
    assert cfg.max_num_seqs <= 128
    overrides = run_overrides(cfg, "easy")
    assert f"generator.inference_engine.gpu_memory_utilization={cfg.gpu_memory_utilization}" in overrides
    assert f"generator.inference_engine.max_num_seqs={cfg.max_num_seqs}" in overrides


def test_context_window_is_pinned_and_large_enough():
    """Unpinned, vLLM would size its KV budget for Qwen3.5's 262144-token default."""
    cfg = ExperimentConfig()
    assert cfg.max_model_len >= cfg.max_prompt_length + cfg.max_generate_length
    assert (f"generator.inference_engine.engine_init_kwargs.max_model_len={cfg.max_model_len}"
            in run_overrides(cfg, "easy"))


def test_eight_optimizer_steps_by_default():
    cfg = ExperimentConfig()
    assert cfg.max_training_steps == 8
    assert "trainer.max_training_steps=8" in run_overrides(cfg, "easy")
    # 3 steps per epoch -> ckpt at 2, 3, 4, 6, 8 and eval at 0, 2, 4, 6, 8.
    assert cfg.epochs == 3


def test_group_size_supports_group_variance():
    cfg = ExperimentConfig()
    assert cfg.n_samples_per_prompt >= 2
    assert f"generator.n_samples_per_prompt={cfg.n_samples_per_prompt}" in run_overrides(cfg, "easy")
    # Dropping zero-variance groups would distort the very statistic being compared.
    assert "trainer.algorithm.zero_variance_filter=false" in run_overrides(cfg, "easy")


def test_evaluation_is_enabled_from_step_zero():
    cfg = ExperimentConfig()
    overrides = run_overrides(cfg, "easy")
    assert "trainer.eval_before_train=true" in overrides
    assert f"trainer.eval_interval={cfg.eval_interval}" in overrides
    assert "trainer.dump_eval_results=true" in overrides


def test_ray_is_given_our_cgroup_budget_not_the_hosts():
    """Inside the container both /proc/meminfo and cgroup/memory.max report the physical
    377 GB machine, which other tenants hold at ~95%. Ray must not size itself from it,
    and its OOM monitor must not kill our workers over it."""
    cfg = ExperimentConfig()
    for run_id in RUN_IDS:
        env = run_env(cfg, run_id)
        assert env["RAY_memory_monitor_refresh_ms"] == "0"
        logical = int(env["SA_RAY_LOGICAL_MEMORY_BYTES"])
        store = int(env["SA_RAY_OBJECT_STORE_BYTES"])
        # The run gets the cgroup minus the object store and the reserve -- never more.
        assert 0 < logical < cfg.cgroup_memory_bytes
        assert store == cfgmod.RAY_OBJECT_STORE_BYTES
        assert env["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == str(store)
    assert cfg.ray_logical_memory_bytes + cfgmod.RAY_OBJECT_STORE_BYTES <= cfg.cgroup_memory_bytes


def test_cgroup_budget_is_overridable_from_the_host(monkeypatch):
    monkeypatch.setenv("SA_CGROUP_MEMORY_BYTES", str(64 * 1024**3))
    cfg = ExperimentConfig()
    assert cfg.cgroup_memory_bytes == 64 * 1024**3
    assert cfg.ray_logical_memory_bytes < 64 * 1024**3


def test_blackwell_runtime_contract_is_in_the_run_environment():
    cfg = ExperimentConfig()
    for run_id in RUN_IDS:
        env = run_env(cfg, run_id)
        assert env["FLA_TILELANG"] == "0"
        first_two = env["PYTHONPATH"].split(":")[:2]
        assert any("skyrl_patches" in p for p in first_two)
        assert any("fla-0.5.2" in p for p in first_two)
        assert env["SA_POLICY_LOAD_DTYPE"] == "bf16"
        # expandable_segments must never be exported by the config.
        assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert cfgmod.UNSET_IN_RUN_ENV == ("PYTORCH_CUDA_ALLOC_CONF",)


def test_runtime_overlays_are_present_on_disk():
    runtime = cfgmod.runtime_dir()
    assert (runtime / "fla-0.5.2" / "fla" / "__init__.py").exists()
    patches = (runtime / "skyrl_patches" / "sitecustomize.py")
    assert patches.exists()
    text = patches.read_text(encoding="utf-8")
    # One sitecustomize, both patches: Python imports at most one.
    assert "vllm.v1.engine.core" in text and "model_wrapper" in text
    version = (runtime / "fla-0.5.2" / "fla" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r"__version__\s*=\s*['\"]0\.5\.2['\"]", version)


def test_ray_tmpdir_is_short_enough_for_the_plasma_socket(monkeypatch):
    """Ray builds <ray_tmpdir>/ray/session_<stamp>/sockets/plasma_store; AF_UNIX caps the
    whole path at 107 bytes, and the session suffix alone is ~68."""
    cfg = ExperimentConfig()
    suffix = "/ray/session_2026-09-03_10-19-28_776234_1234567/sockets/plasma_store"
    for run_id in RUN_IDS:
        path = str(cfg.run_paths(run_id)["ray_tmpdir"])
        assert len(path + suffix) < 107, path
        # It must be on local disk: a Unix socket on the NFS /data volume does not work.
        assert not path.startswith("/data"), path
    monkeypatch.setenv("SA_EXP_TAG", "a" * 64)
    with pytest.raises(SystemExit, match="AF_UNIX"):
        ExperimentConfig().run_paths("easy")


def test_ray_and_lora_sync_are_isolated_per_run():
    cfg = ExperimentConfig()
    for run_id in RUN_IDS:
        env = run_env(cfg, run_id)
        assert env["RAY_ADDRESS"] == "local"
        assert env["RAY_TMPDIR"] == str(cfg.run_paths(run_id)["ray_tmpdir"])
        sync = str(cfg.run_paths(run_id)["lora_sync_path"])
        assert f"trainer.policy.model.lora.lora_sync_path={sync}" in run_overrides(cfg, run_id)
        assert not sync.startswith("/tmp/")


def test_manifest_records_what_the_run_actually_did():
    manifest = run_manifest(ExperimentConfig(), "hard")
    assert manifest["train_archetype"] == "confounded_reversal"
    assert manifest["train_worlds"] == TRAIN_WORLDS_PER_RUN
    assert manifest["gpus"] == [0, 4]
    assert manifest["tensor_parallel_size"] == 2
