"""The shell layer must stay a thin wrapper over config.py, and must be safe to re-run."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from single_arch_rl.config import RUN_IDS, ExperimentConfig, run_env, run_overrides

_PKG = Path(__file__).resolve().parents[1]
_SCRIPTS = _PKG / "scripts"


@pytest.mark.parametrize("name", ["launch.sh", "run_one.sh", "in_container.sh", "run_tests.sh"])
def test_scripts_exist_are_executable_and_parse(name):
    path = _SCRIPTS / name
    assert path.exists(), path
    assert os.stat(path).st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_launch_is_the_single_entrypoint_and_drives_the_data_mounted_container():
    text = (_SCRIPTS / "launch.sh").read_text(encoding="utf-8")
    assert 'CONTAINER="${SA_CONTAINER:-skyrl-pc}"' in text
    for run_id in RUN_IDS:
        assert run_id in text
    # It must call the build, the checks and run_one.sh -- not reimplement them.
    assert "single_arch_rl.build_dataset" in text
    assert "scripts/run_tests.sh" in text
    assert "scripts/run_one.sh $run" in text


def test_launch_never_starts_two_runs_at_once():
    """They share every GPU, and two concurrent runs do not fit in host RAM anyway."""
    text = (_SCRIPTS / "launch.sh").read_text(encoding="utf-8")
    assert 'RUNS="${SA_RUNS:-easy}"' in text
    # A second run waits for the first to exit.
    assert "waiting for the previous run to finish" in text
    assert "--sequential" not in text


def test_run_one_holds_no_settings_of_its_own():
    """Every SkyRL flag comes from config.py, so the two runs cannot drift apart."""
    text = (_SCRIPTS / "run_one.sh").read_text(encoding="utf-8")
    assert "args0" in text
    stray = [line for line in text.splitlines()
             if re.search(r"^\s*(trainer|generator|data|environment)\.", line)]
    assert stray == []


def test_run_one_unsets_expandable_segments():
    """It is incompatible with the CuMemAllocator pool vLLM re-maps on every wake."""
    text = (_SCRIPTS / "run_one.sh").read_text(encoding="utf-8")
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in text


def test_run_one_uses_the_venv_python_so_the_overlays_load():
    text = (_SCRIPTS / "run_one.sh").read_text(encoding="utf-8")
    assert '"$SKYRL_DIR/.venv/bin/python" -m single_arch_rl.main' in text
    # `uv run --isolated` would resolve fla from its own ephemeral tree, defeating the
    # 0.5.2 overlay. Check the executable lines, not the comment that explains this.
    code = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any("uv run" in line for line in code)


def test_config_cli_matches_the_python_api():
    cfg = ExperimentConfig()
    env = dict(os.environ, PYTHONPATH=str(_PKG.parent))
    for run_id in RUN_IDS:
        args = subprocess.run(
            [sys.executable, "-m", "single_arch_rl.config", "args0", run_id],
            check=True, capture_output=True, env=env,
        ).stdout.decode().split("\0")[:-1]
        assert args == run_overrides(cfg, run_id)

        lines = subprocess.run(
            [sys.executable, "-m", "single_arch_rl.config", "env", run_id],
            check=True, capture_output=True, env=env,
        ).stdout.decode().splitlines()
        emitted = dict(line.split("=", 1) for line in lines if line)
        assert emitted == run_env(cfg, run_id)


def test_budget_subcommand_reports_what_the_launcher_needs():
    env = dict(os.environ, PYTHONPATH=str(_PKG.parent))
    out = subprocess.run(
        [sys.executable, "-m", "single_arch_rl.config", "budget"],
        check=True, capture_output=True, env=env,
    ).stdout.decode()
    keys = {line.split("=", 1)[0] for line in out.splitlines() if line}
    assert {"HOST_GB_PER_JOB", "GPU_GB_HEADROOM", "GPU_MEM_UTIL", "CKPT_ROOT",
            "CKPT_INTERVAL", "CKPT_GB_PER_RUN", "GPUS", "MAX_STEPS", "EXP_DIR"} <= keys
