"""Make ``single_arch_rl`` importable and keep the tests free of ambient environment.

Every ``SA_*`` variable is cleared for the duration of a test module so a shell that
happens to export ``SA_GPUS`` or ``SA_MAX_STEPS`` cannot change what the tests assert.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PKG_PARENT = str(Path(__file__).resolve().parents[2])
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    yield
