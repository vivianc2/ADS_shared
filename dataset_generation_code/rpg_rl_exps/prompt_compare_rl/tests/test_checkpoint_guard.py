"""The checkpoint guard installed by ``main.py``.

Two behaviors matter and both are testable without a GPU:

1. It takes an EXCLUSIVE lock that the three concurrent runs share, so only one of them
   can be inside ``save_checkpoints`` (and holding an ~18 GiB state dict) at a time.
2. It copies the LoRA adapter -- the only trained content in a 19 GB checkpoint -- into
   the run's export tree.
"""

from __future__ import annotations

import fcntl
import json
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture()
def guard(monkeypatch, tmp_path):
    """Install the guard over a stub ``save_checkpoints`` and return the pieces."""
    from skyrl.train.trainer import RayPPOTrainer

    from prompt_compare_rl import main as pc_main

    lock_path = tmp_path / "checkpoint_save.lock"
    adapters = tmp_path / "adapters"
    monkeypatch.setenv("PC_CKPT_LOCK", str(lock_path))
    monkeypatch.setenv("PC_ADAPTER_EXPORT_DIR", str(adapters))

    calls = []

    def stub_save(self):
        ckpt = tmp_path / "ckpts" / f"global_step_{self.global_step}"
        (ckpt / "policy" / "lora_adapter").mkdir(parents=True, exist_ok=True)
        (ckpt / "policy" / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"weights")
        (ckpt / "policy" / "lora_adapter" / "adapter_config.json").write_text(json.dumps({"r": 16}))
        (ckpt / "policy" / "huggingface").mkdir(parents=True, exist_ok=True)
        (ckpt / "policy" / "huggingface" / "config.json").write_text("{}")
        (ckpt / "policy" / "model_world_size_1_rank_0.pt").write_bytes(b"x" * 1024)
        calls.append(self.global_step)
        return str(ckpt)

    original = RayPPOTrainer.save_checkpoints
    monkeypatch.setattr(RayPPOTrainer, "save_checkpoints", stub_save, raising=True)
    pc_main._install_checkpoint_guard()
    guarded = RayPPOTrainer.save_checkpoints
    assert guarded is not stub_save, "the guard did not wrap save_checkpoints"

    class FakeTrainer:
        global_step = 4

        save_checkpoints = guarded

    yield FakeTrainer(), lock_path, adapters, calls
    monkeypatch.setattr(RayPPOTrainer, "save_checkpoints", original, raising=True)


def test_adapter_is_exported(guard):
    trainer, _lock, adapters, calls = guard
    trainer.save_checkpoints()
    assert calls == [4]
    exported = adapters / "global_step_4"
    assert (exported / "adapter_model.safetensors").read_bytes() == b"weights"
    assert json.loads((exported / "adapter_config.json").read_text())["r"] == 16
    assert (exported / "huggingface" / "config.json").exists()
    # The 19 GB blob is NOT copied -- only the trained delta.
    assert not (exported / "model_world_size_1_rank_0.pt").exists()


def test_save_is_serialized_by_the_shared_lock(guard):
    """While another run holds the lock, a save must wait rather than run concurrently."""
    trainer, lock_path, _adapters, calls = guard
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    finished = threading.Event()

    def run_save():
        trainer.save_checkpoints()
        finished.set()

    # Stand in for another prompt run that is already inside save_checkpoints().
    with open(lock_path, "w") as other_run:
        fcntl.flock(other_run, fcntl.LOCK_EX)
        worker = threading.Thread(target=run_save, daemon=True)
        worker.start()
        assert not finished.wait(timeout=1.5), "save proceeded while another run held the lock"
        assert calls == []
        fcntl.flock(other_run, fcntl.LOCK_UN)

    assert finished.wait(timeout=30), "save did not proceed after the lock was released"
    assert calls == [4]
    worker.join(timeout=5)


def test_guard_is_idempotent(guard, monkeypatch):
    """Installing twice must not stack two locks (which would deadlock)."""
    from skyrl.train.trainer import RayPPOTrainer

    from prompt_compare_rl import main as pc_main

    first = RayPPOTrainer.save_checkpoints
    pc_main._install_checkpoint_guard()
    assert RayPPOTrainer.save_checkpoints is first


def test_guard_is_inert_without_the_lock_env_var(monkeypatch):
    """Outside a prompt-compare launch the trainer must be left alone."""
    from skyrl.train.trainer import RayPPOTrainer

    from prompt_compare_rl import main as pc_main

    monkeypatch.delenv("PC_CKPT_LOCK", raising=False)
    sentinel = RayPPOTrainer.save_checkpoints
    pc_main._install_checkpoint_guard()
    assert RayPPOTrainer.save_checkpoints is sentinel
