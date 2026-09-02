"""Shared fixtures.

Every test here runs on CPU inside the SkyRL container:

    bash scripts/in_container.sh python -m pytest -q prompt_compare_rl/tests
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(_PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR.parent))

os.environ.setdefault("RPG_SRC", "/work/ADS_shared/dataset_generation_code")
os.environ.setdefault("RPG_PROTO", "rpg_v9")
# A test must never inherit a prompt override from an interactive shell.
os.environ.pop("RPG_SYSTEM_PROMPT_FILE", None)
os.environ.pop("RPG_SYSTEM_PROMPT_SHA256", None)

from prompt_compare_rl.config import ExperimentConfig  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.fixture(scope="session")
def datasets_built(cfg: ExperimentConfig) -> ExperimentConfig:
    if not cfg.train_parquet("p1").exists():
        pytest.skip(
            f"datasets not built at {cfg.dataset_root}; "
            "run `python -m prompt_compare_rl.build_dataset` first"
        )
    return cfg


@pytest.fixture(scope="session")
def rpg_src() -> str:
    src = os.environ["RPG_SRC"]
    if not Path(src, "rpg_rl", "env.py").exists():
        pytest.skip(f"RPG sources not available at {src}")
    return src
