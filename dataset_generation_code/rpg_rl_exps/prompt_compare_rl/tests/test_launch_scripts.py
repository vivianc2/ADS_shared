"""The shell entry points are syntactically valid and stay in sync with config.py."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from prompt_compare_rl.config import PROMPT_IDS, run_env, run_overrides

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.mark.parametrize("name", ["run_one.sh", "launch_all.sh", "in_container.sh"])
def test_shell_syntax(name):
    script = SCRIPTS / name
    assert script.exists(), script
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("name", ["run_one.sh", "launch_all.sh", "in_container.sh"])
def test_scripts_are_executable(name):
    assert (SCRIPTS / name).stat().st_mode & 0o111


def test_config_cli_matches_the_python_api(cfg):
    """The launchers consume `config.py env/args`; those must equal the API output."""
    package_parent = str(Path(__file__).resolve().parent.parent.parent)
    for pid in PROMPT_IDS:
        env_out = subprocess.run(
            [sys.executable, "-m", "prompt_compare_rl.config", "env", pid],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": package_parent},
        ).stdout
        parsed = dict(line.split("=", 1) for line in env_out.splitlines() if line)
        expected = run_env(cfg, pid)
        # PYTHONPATH depends on the caller's environment; compare the rest exactly.
        for key, value in expected.items():
            if key == "PYTHONPATH":
                continue
            assert parsed[key] == value, key

        args_out = subprocess.run(
            [sys.executable, "-m", "prompt_compare_rl.config", "args", pid],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": package_parent},
        ).stdout
        assert shlex.split(args_out) == run_overrides(cfg, pid)


def test_launch_all_pins_the_required_gpu_mapping():
    text = (SCRIPTS / "launch_all.sh").read_text()
    assert "p1 -> GPU 0, p2 -> GPU 1, p3 -> GPU 2" in text
    assert "--sequential" in text


def test_run_one_never_hardcodes_hyperparameters():
    """All settings must come from config.py so the three runs cannot diverge."""
    text = (SCRIPTS / "run_one.sh").read_text()
    for forbidden in ("max_training_steps", "train_batch_size", "lora.rank", "eval_interval"):
        assert forbidden not in text, f"{forbidden} is duplicated in run_one.sh"


def test_args0_is_nul_separated_and_equals_the_api(cfg):
    """`run_one.sh` consumes the NUL-separated form; it must match exactly."""
    package_parent = str(Path(__file__).resolve().parent.parent.parent)
    for pid in PROMPT_IDS:
        raw = subprocess.run(
            [sys.executable, "-m", "prompt_compare_rl.config", "args0", pid],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": package_parent},
        ).stdout
        assert raw.endswith("\0")
        assert raw.split("\0")[:-1] == run_overrides(cfg, pid)


@pytest.mark.parametrize("prompt_id", list(PROMPT_IDS))
def test_run_one_reaches_preflight(prompt_id, datasets_built):
    """End-to-end launch path without a GPU: env, dirs, uv, SkyRL config, env import.

    PC_PREFLIGHT_ONLY makes `main.py` stop right before `initialize_ray`.
    """
    script = SCRIPTS / "run_one.sh"
    if not Path("/work/SkyRL").exists():
        pytest.skip("SkyRL checkout not mounted at /work/SkyRL")
    result = subprocess.run(
        ["bash", str(script), prompt_id],
        capture_output=True, text=True,
        env={
            "PATH": "/home/ray/.local/bin:/home/ray/anaconda3/bin:/usr/bin:/bin",
            "HOME": "/work/home",
            "HF_HOME": "/work/hf_cache",
            "PC_PREFLIGHT_ONLY": "1",
        },
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert f"pre-flight OK for {prompt_id}" in result.stdout
