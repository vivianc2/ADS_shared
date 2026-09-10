"""The trainer patches: group metrics every step, serialized + exported checkpoints."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("skyrl.train.trainer")

from single_arch_rl import main as main_mod  # noqa: E402


class _FakeTrainer:
    """Stands in for RayPPOTrainer: only the attributes the patches touch."""

    global_step = 3

    def __init__(self, ckpt_dir="/tmp/does-not-exist/global_step_3"):
        self.all_metrics = {}
        self.saved = 0
        self._ckpt_dir = ckpt_dir


def _original_postprocess(self, generator_output, uids,
                          metrics_generator_output=None, metrics_uids=None):
    # The real method rewrites rewards into per-token form in place.
    generator_output["rewards"] = [[0.0, r] for r in generator_output["rewards"]]
    return generator_output, uids


def _original_save(self):
    self.saved += 1
    return self._ckpt_dir


@pytest.fixture
def trainer_cls(monkeypatch):
    """Reset RayPPOTrainer to unpatched stand-ins before each test.

    The guards are idempotent by design (they tag the function they installed), so each
    test must start from a clean, untagged pair or only the first would install.
    """
    from skyrl.train.trainer import RayPPOTrainer

    monkeypatch.setattr(RayPPOTrainer, "postprocess_generator_output",
                        _original_postprocess, raising=False)
    monkeypatch.setattr(RayPPOTrainer, "save_checkpoints", _original_save, raising=False)
    return RayPPOTrainer


def test_group_metrics_are_added_without_disturbing_the_original(trainer_cls):
    main_mod._install_group_reward_metrics()
    trainer = _FakeTrainer()
    output = {"rewards": [0.0, 1.0, 0.5, 0.5]}
    result, uids = trainer_cls.postprocess_generator_output(trainer, output, ["a", "a", "b", "b"])
    assert result["rewards"] == [[0.0, 0.0], [0.0, 1.0], [0.0, 0.5], [0.0, 0.5]]
    assert uids == ["a", "a", "b", "b"]
    assert trainer.all_metrics["reward/group_reward_mean"] == pytest.approx(0.5)
    assert trainer.all_metrics["reward/group_reward_var"] == pytest.approx(0.125)
    assert trainer.all_metrics["reward/num_groups"] == 2


def test_stats_are_computed_before_the_per_token_rewrite(trainer_cls):
    """Reading `rewards` after the original ran would double-count the conversion."""
    main_mod._install_group_reward_metrics()
    trainer = _FakeTrainer()
    trainer_cls.postprocess_generator_output(trainer, {"rewards": [0.0, 1.0]}, ["a", "a"])
    assert trainer.all_metrics["reward/group_reward_mean"] == pytest.approx(0.5)


def test_installing_twice_does_not_stack_wrappers(trainer_cls):
    main_mod._install_group_reward_metrics()
    once = trainer_cls.postprocess_generator_output
    main_mod._install_group_reward_metrics()
    assert trainer_cls.postprocess_generator_output is once


def test_a_metrics_failure_never_breaks_the_step(trainer_cls, monkeypatch):
    monkeypatch.setattr(main_mod, "group_reward_stats",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    main_mod._install_group_reward_metrics()
    trainer = _FakeTrainer()
    result, _ = trainer_cls.postprocess_generator_output(trainer, {"rewards": [1.0]}, ["a"])
    assert result["rewards"] == [[0.0, 1.0]]
    assert trainer.all_metrics == {}


def test_checkpoint_saves_take_the_shared_lock(trainer_cls, tmp_path, monkeypatch):
    lock = tmp_path / "exp" / "checkpoint_save.lock"
    monkeypatch.setenv("SA_CKPT_LOCK", str(lock))
    monkeypatch.setenv("SA_ADAPTER_EXPORT_DIR", str(tmp_path / "adapters"))
    main_mod._install_checkpoint_guard()
    trainer = _FakeTrainer()
    assert trainer_cls.save_checkpoints(trainer).endswith("global_step_3")
    assert trainer.saved == 1
    assert lock.exists()


def test_checkpoint_guard_exports_the_lora_adapter(trainer_cls, tmp_path, monkeypatch):
    """~196 MB of trained delta, copied out of the ~19 GB blob so it survives on its own."""
    ckpt = tmp_path / "checkpoints" / "global_step_3"
    (ckpt / "policy" / "lora_adapter").mkdir(parents=True)
    (ckpt / "policy" / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "policy" / "huggingface").mkdir(parents=True)
    (ckpt / "policy" / "huggingface" / "tokenizer_config.json").write_text("{}")

    adapters = tmp_path / "adapters"
    monkeypatch.setenv("SA_CKPT_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SA_ADAPTER_EXPORT_DIR", str(adapters))
    main_mod._install_checkpoint_guard()

    trainer_cls.save_checkpoints(_FakeTrainer(ckpt_dir=str(ckpt)))
    exported = adapters / "global_step_3"
    assert (exported / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (exported / "huggingface" / "tokenizer_config.json").exists()


def test_checkpoint_guard_is_a_no_op_without_the_lock(trainer_cls, monkeypatch):
    monkeypatch.delenv("SA_CKPT_LOCK", raising=False)
    before = trainer_cls.save_checkpoints
    main_mod._install_checkpoint_guard()
    assert trainer_cls.save_checkpoints is before


def _minimal_cfg(**overrides):
    """A stand-in config with just the fields ``_preflight`` reads."""
    trainer = dict(
        train_batch_size=2, policy_mini_batch_size=2, update_epochs_per_batch=1,
        eval_before_train=True, eval_interval=2, resume_mode="latest",
        ckpt_interval=0, ckpt_path="/nonexistent", max_training_steps=30,
    )
    trainer.update(overrides)
    return types.SimpleNamespace(
        data=types.SimpleNamespace(train_data=[], val_data=[]),
        trainer=types.SimpleNamespace(**trainer),
        generator=types.SimpleNamespace(n_samples_per_prompt=8),
        environment=types.SimpleNamespace(env_class="rpg_single_arch"),
    )


@pytest.fixture
def sane_env(monkeypatch):
    monkeypatch.setenv("SA_RUN_ID", "easy")
    monkeypatch.setenv("SA_TRAIN_ARCHETYPE", "dose_window")
    monkeypatch.setenv("RPG_PROTO", "rpg_v9")
    monkeypatch.setenv("WANDB_RUN_ID", "sarl_v3_bs2_s30_ckpt4-easy-attempt")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def test_preflight_rejects_expandable_segments(sane_env, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(SystemExit, match="PYTORCH_CUDA_ALLOC_CONF"):
        main_mod._preflight(_minimal_cfg())


def test_preflight_rejects_a_mismatched_archetype(sane_env, monkeypatch):
    monkeypatch.setenv("SA_TRAIN_ARCHETYPE", "confounded_reversal")
    with pytest.raises(SystemExit, match="SA_TRAIN_ARCHETYPE"):
        main_mod._preflight(_minimal_cfg())


def test_preflight_rejects_a_run_that_would_not_resume(sane_env, monkeypatch):
    with pytest.raises(SystemExit, match="resume_mode"):
        main_mod._preflight(_minimal_cfg(resume_mode="none"))
    monkeypatch.delenv("WANDB_RUN_ID")
    with pytest.raises(SystemExit, match="WANDB_RUN_ID"):
        main_mod._preflight(_minimal_cfg())


def test_preflight_rejects_a_group_size_too_small_for_variance(sane_env):
    cfg = _minimal_cfg()
    cfg.generator.n_samples_per_prompt = 1
    with pytest.raises(SystemExit, match="group_reward_var"):
        main_mod._preflight(cfg)


def test_preflight_rejects_a_training_set_of_the_wrong_archetype(sane_env, tmp_path):
    pd = pytest.importorskip("pandas")
    path = tmp_path / "train.parquet"
    pd.DataFrame({"extra_info": [{"archetype": "synergy_pair"}] * 96}).to_parquet(path)
    cfg = _minimal_cfg()
    cfg.data.train_data = [str(path)]
    with pytest.raises(SystemExit, match="dose_window"):
        main_mod._preflight(cfg)


def test_preflight_rejects_a_parquet_that_names_the_wrong_env(sane_env, tmp_path):
    """SkyRL builds each episode from the ROW's env_class; "rpg" would be unregistered."""
    pd = pytest.importorskip("pandas")
    path = tmp_path / "val.parquet"
    pd.DataFrame({"env_class": ["rpg"] * 45}).to_parquet(path)
    cfg = _minimal_cfg()
    cfg.data.val_data = [str(path)]
    with pytest.raises(SystemExit, match="env_class"):
        main_mod._preflight(cfg)
