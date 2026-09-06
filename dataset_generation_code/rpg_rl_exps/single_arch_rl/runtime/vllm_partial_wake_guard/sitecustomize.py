"""POPE-local, lazy vLLM 0.23 partial-wake safety guard.

The import hook patches only processes that actually import EngineCore. It
does not eagerly import Torch/vLLM into Ray utility and policy processes.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any


_TARGET = "vllm.v1.engine.core"


def _patch_engine_core(module: ModuleType) -> None:
    vllm_module = sys.modules["vllm"]
    version = getattr(vllm_module, "__version__", None)
    if version != "0.23.0":
        raise RuntimeError(
            "POPE's partial-wake guard is reviewed only for vLLM 0.23.0; "
            f"found {version}"
        )

    engine_core = module.EngineCore

    def _pope_safe_wake_up(self: Any, tags: list[str] | None = None) -> None:
        """Wake pools without scheduling against a sleeping KV cache."""
        if tags is not None and "scheduling" in tags:
            tags = [tag for tag in tags if tag != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # weights-only wake leaves this true because KV remains unmapped.
        if not self.model_executor.is_sleeping:
            self.resume_scheduler()

    _pope_safe_wake_up._pope_partial_wake_guard = True
    engine_core.wake_up = _pope_safe_wake_up


class _GuardedLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_engine_core(module)


class _GuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _GuardedLoader(spec.loader)
        return spec


sys.meta_path.insert(0, _GuardFinder())
