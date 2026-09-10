"""Project-local SkyRL/vLLM patches, applied in EVERY process including Ray workers.

``sitecustomize`` is the only hook that reaches a Ray actor: the trainer, the policy
workers and the vLLM engine are separate processes, and SkyRL re-exports ``PYTHONPATH``
into all of them (``SKYRL_PYTHONPATH_EXPORT=1``), so a module placed here runs in each.
All patches below are lazy -- an import hook fires only if the target module is actually
imported -- so no process pays for one it does not need.

Python imports at most ONE ``sitecustomize``, which is why both live in this file.

1. vLLM 0.23.0 sleep/wake guards
--------------------------------
``EngineCore.wake_up`` resumes the scheduler even on a weights-only partial wake, while
the KV cache is still unmapped. SkyRL's colocated path does exactly that --
``wake_up(tags=["weights"])`` then ``wake_up(tags=["kv_cache"])``. Adds the missing
sleeping-state check. (Byte-identical in behavior to the guard the POPE experiment
validated on this box.)

The 0.23.0 ``CuMemAllocator`` also blindly unmaps/maps every allocation, calls Python GC
while the mappings are absent, and does not synchronize CUDA streams before changing the
virtual-memory mappings. On a hot second wake this can wedge one TP worker; EngineCore then
prints ``No available shared memory broadcast block`` forever. Track which pointers are
actually unmapped, drain CUDA work around VMM changes, and omit the unsafe sleeping GC.

Finally, 0.23.0 calls the worker ``wake_up`` collective with no timeout. Put a five-minute
bound on that RPC so a native allocator failure terminates the attempt instead of leaving
an immortal Ray/vLLM process tree.

2. bf16 policy load
-------------------
``fsdp_worker.py`` builds the policy with ``bf16=cfg.policy.inference_only_init``, which
is False for a *trainable* policy, so ``HFModelWrapper`` calls ``from_pretrained`` with
``torch_dtype=torch.float32``. For Qwen3.5-9B that is a **37.6 GB fp32 materialization on
the host**, and it happens on rank 0 *after* vLLM is already holding its ~24 GB host sleep
buffer (which LoRA weight sync forbids dropping). Measured on this box: 92 GB of a 96 GiB
container-tree ceiling, and the policy worker is killed every time -- with one GPU or two,
since the load is complete on rank 0 before FSDP shards anything.

Loading the base in bf16 halves that to ~19 GB. This is a deliberate, documented
deviation, not an optimization:

* The base is FROZEN. Only the LoRA parameters are trained, and this patch casts them
  back to fp32 after PEFT wraps the model, so the optimizer still updates in fp32.
* bf16 is the dtype Qwen3.5-9B ships in (``config.json``: ``dtype: bfloat16``) and the
  dtype vLLM generates in. Training against an fp32 base while sampling from a bf16 one
  is a train/inference mismatch; this removes it.
* FSDP2 already runs compute in bf16 here (``MixedPrecisionPolicy(param_dtype=bf16)``).

Set ``SA_POLICY_LOAD_DTYPE=fp32`` to disable this patch and restore SkyRL's default. The
run then needs a host-memory ceiling above ~96 GiB.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import time
from types import ModuleType
from typing import Any

_VLLM_TARGET = "vllm.v1.engine.core"
_VLLM_CUMEM_TARGET = "vllm.device_allocator.cumem"
_VLLM_MP_TARGET = "vllm.v1.executor.multiproc_executor"
_POLICY_TARGET = "skyrl.backends.skyrl_train.workers.model_wrapper"


# --------------------------------------------------------------------------------------
# 1. vLLM sleep/wake guards
# --------------------------------------------------------------------------------------


def _require_vllm_023(patch_name: str) -> None:
    vllm_module = sys.modules["vllm"]
    version = getattr(vllm_module, "__version__", None)
    if version != "0.23.0":
        raise RuntimeError(f"the {patch_name} is reviewed only for vLLM 0.23.0; found {version}")


def _patch_cumem_allocator(module: ModuleType) -> None:
    """Make CuMem sleep/wake stateful and synchronize around CUDA VMM mutations.

    This is deliberately a Python-only guard. It prevents the invalid repeated map/unmap
    calls and sleeping-window GC that lead to the sticky native allocator error in 0.23.0,
    without mutating the shared SkyRL virtualenv.
    """
    _require_vllm_023("CuMem sleep/wake guard")
    allocator = module.CuMemAllocator
    if getattr(allocator.wake_up, "_sa_cumem_state_guard", False):
        return

    original_init = allocator.__init__
    original_malloc = allocator._python_malloc_callback

    def guarded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._sa_unmapped_ptrs = set()

    def guarded_malloc(self, allocation_handle):
        original_malloc(self, allocation_handle)
        # A new allocation is mapped by construction. This also handles allocations made
        # between the weights-only and KV-cache wake phases.
        self._sa_unmapped_ptrs.discard(allocation_handle[2])

    def guarded_sleep(self, offload_tags=None) -> None:
        import torch

        if offload_tags is None:
            offload_tags = (allocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        assert isinstance(offload_tags, tuple)

        unmapped = getattr(self, "_sa_unmapped_ptrs", None)
        if unmapped is None:
            unmapped = self._sa_unmapped_ptrs = set()

        devices = {data.handle[0] for ptr, data in self.pointer_to_data.items() if ptr not in unmapped}
        for device in devices:
            torch.cuda.synchronize(device)

        total_bytes = 0
        backup_bytes = 0
        for ptr, data in tuple(self.pointer_to_data.items()):
            if ptr in unmapped:
                continue
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                backup = torch.empty(
                    handle[1],
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=module.is_pin_memory_available(),
                )
                module.libcudart.cudaMemcpy(backup.data_ptr(), ptr, handle[1])
                data.cpu_backup_tensor = backup
            module.unmap_and_release(handle)
            unmapped.add(ptr)

        module.logger.info(
            "CuMemAllocator guarded sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )
        # Do not gc.collect()/empty_cache() here. A finalizer firing while mappings are
        # absent can call the native free callback on an already-unmapped allocation.

    def guarded_wake_up(self, tags=None) -> None:
        import torch

        unmapped = getattr(self, "_sa_unmapped_ptrs", None)
        if unmapped is None:
            unmapped = self._sa_unmapped_ptrs = set(self.pointer_to_data)

        touched_devices = set()
        for ptr, data in tuple(self.pointer_to_data.items()):
            if ptr not in unmapped or (tags is not None and data.tag not in tags):
                continue
            module.create_and_map(data.handle)
            touched_devices.add(data.handle[0])
            if data.cpu_backup_tensor is not None:
                backup = data.cpu_backup_tensor
                size_in_bytes = backup.numel() * backup.element_size()
                module.libcudart.cudaMemcpy(ptr, backup.data_ptr(), size_in_bytes)
                data.cpu_backup_tensor = None
            unmapped.remove(ptr)

        for device in touched_devices:
            torch.cuda.synchronize(device)

    guarded_init._sa_cumem_state_guard = True
    guarded_malloc._sa_cumem_state_guard = True
    guarded_sleep._sa_cumem_state_guard = True
    guarded_wake_up._sa_cumem_state_guard = True
    allocator.__init__ = guarded_init
    allocator._python_malloc_callback = guarded_malloc
    allocator.sleep = guarded_sleep
    allocator.wake_up = guarded_wake_up


def _patch_multiproc_executor(module: ModuleType) -> None:
    """Bound the TP-worker wake collective; vLLM 0.23.0 otherwise waits forever."""
    _require_vllm_023("multiprocess wake timeout")
    executor = module.MultiprocExecutor
    if getattr(executor.wake_up, "_sa_wake_timeout", False):
        return

    def guarded_wake_up(self, tags=None):
        if not self.is_sleeping:
            module.logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    module.logger.warning("Tag %s is not in sleeping tags %s", tag, self.sleeping_tags)
                    return

        timeout = float(os.environ.get("SA_VLLM_WAKE_TIMEOUT_SECONDS", "300"))
        started = time.perf_counter()
        self.collective_rpc("wake_up", timeout=timeout, kwargs=dict(tags=tags))
        module.logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time.perf_counter() - started,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    guarded_wake_up._sa_wake_timeout = True
    executor.wake_up = guarded_wake_up


def _patch_engine_core(module: ModuleType) -> None:
    _require_vllm_023("partial-wake guard")

    engine_core = module.EngineCore

    def _safe_wake_up(self: Any, tags: list | None = None) -> None:
        """Wake pools without scheduling against a sleeping KV cache."""
        if tags is not None and "scheduling" in tags:
            tags = [tag for tag in tags if tag != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # A weights-only wake leaves this true because KV remains unmapped.
        if not self.model_executor.is_sleeping:
            self.resume_scheduler()

    _safe_wake_up._pope_partial_wake_guard = True   # name kept: main.py and POPE assert on it
    _safe_wake_up._sa_partial_wake_guard = True
    engine_core.wake_up = _safe_wake_up


# --------------------------------------------------------------------------------------
# 2. bf16 policy load
# --------------------------------------------------------------------------------------


def _patch_model_wrapper(module: ModuleType) -> None:
    if os.environ.get("SA_POLICY_LOAD_DTYPE", "bf16").lower() not in ("bf16", "bfloat16"):
        return

    import torch

    wrapper = module.HFModelWrapper
    if getattr(wrapper.__init__, "_sa_bf16_load", False):
        return
    original_init = wrapper.__init__

    def patched_init(self, *args, **kwargs):
        # `bf16` reaches HFModelWrapper as a keyword from fsdp_worker.py; forcing it True
        # selects torch_dtype=bfloat16 on both the from_pretrained and the meta-init path.
        kwargs["bf16"] = True
        original_init(self, *args, **kwargs)
        # Keep the TRAINED parameters (the LoRA adapters) in fp32 so the optimizer's
        # updates and moments stay full precision; only the frozen base becomes bf16.
        promoted = 0
        model = getattr(self, "model", None)
        if model is not None:
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.dtype == torch.bfloat16:
                    parameter.data = parameter.data.float()
                    promoted += 1
        print(
            f"[single_arch_rl] policy base loaded in bf16; promoted {promoted} trainable "
            "(LoRA) tensors back to fp32",
            flush=True,
        )

    patched_init._sa_bf16_load = True
    wrapper.__init__ = patched_init


# --------------------------------------------------------------------------------------
# Lazy import hook
# --------------------------------------------------------------------------------------

_PATCHES = {
    _VLLM_TARGET: _patch_engine_core,
    _VLLM_CUMEM_TARGET: _patch_cumem_allocator,
    _VLLM_MP_TARGET: _patch_multiproc_executor,
    _POLICY_TARGET: _patch_model_wrapper,
}


class _GuardedLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, patch) -> None:
        self._wrapped = wrapped
        self._patch = patch

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        self._patch(module)


class _GuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        patch = _PATCHES.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _GuardedLoader(spec.loader, patch)
        return spec


sys.meta_path.insert(0, _GuardFinder())
