"""Project-local SkyRL/vLLM patches, applied in EVERY process including Ray workers.

``sitecustomize`` is the only hook that reaches a Ray actor: the trainer, the policy
workers and the vLLM engine are separate processes, and SkyRL re-exports ``PYTHONPATH``
into all of them (``SKYRL_PYTHONPATH_EXPORT=1``), so a module placed here runs in each.
Both patches below are lazy -- an import hook fires only if the target module is actually
imported -- so no process pays for one it does not need.

Python imports at most ONE ``sitecustomize``, which is why both live in this file.

1. vLLM 0.23.0 partial-wake guard
---------------------------------
``EngineCore.wake_up`` resumes the scheduler even on a weights-only partial wake, while
the KV cache is still unmapped. SkyRL's colocated path does exactly that --
``wake_up(tags=["weights"])`` then ``wake_up(tags=["kv_cache"])``. Adds the missing
sleeping-state check. (Byte-identical in behavior to the guard the POPE experiment
validated on this box.)

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
from types import ModuleType
from typing import Any

_VLLM_TARGET = "vllm.v1.engine.core"
_POLICY_TARGET = "skyrl.backends.skyrl_train.workers.model_wrapper"


# --------------------------------------------------------------------------------------
# 1. vLLM partial-wake guard
# --------------------------------------------------------------------------------------


def _patch_engine_core(module: ModuleType) -> None:
    vllm_module = sys.modules["vllm"]
    version = getattr(vllm_module, "__version__", None)
    if version != "0.23.0":
        raise RuntimeError(
            "the partial-wake guard is reviewed only for vLLM 0.23.0; " f"found {version}"
        )

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

_PATCHES = {_VLLM_TARGET: _patch_engine_core, _POLICY_TARGET: _patch_model_wrapper}


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
