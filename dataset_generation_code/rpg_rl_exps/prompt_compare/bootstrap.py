"""Repository-relative import bootstrap for the prompt comparison."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_CODE_DIR = EXPERIMENT_DIR.parents[1]
RPG_RL_DIR = DATASET_CODE_DIR / "rpg_rl"
RPG_V9_DIR = DATASET_CODE_DIR / "rpg_v9"


def configure_imports() -> None:
    """Pin the protocol and import search order before any RPG imports occur."""
    os.environ["RPG_PROTO"] = "rpg_v9"
    os.environ["RPG_SYNERGY_SOFT"] = "20"
    wanted = [str(RPG_RL_DIR), str(RPG_V9_DIR)]
    sys.path[:] = [entry for entry in sys.path if entry not in wanted]
    sys.path[:0] = wanted


def assert_v9_imports() -> dict[str, str]:
    """Fail loudly if collision-prone modules resolved outside rpg_v9."""
    resolved = {}
    expected = RPG_V9_DIR.resolve()
    for name in ("sampler", "engine", "oracle_v6"):
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        if source.parent != expected:
            raise RuntimeError(
                f"unsafe import resolution: {name} came from {source}, expected {expected}"
            )
        resolved[name] = str(source)
    return resolved

