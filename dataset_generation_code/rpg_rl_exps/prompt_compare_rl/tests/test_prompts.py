"""The three prompts are the ones the inference-side experiment defines, and p1 is the
prompt already baked into the source parquet files."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prompt_compare_rl import prompts as prompt_lib


def test_expected_digests_match_the_source_of_truth():
    """`prompt_compare/candidates.py` still yields the three pinned strings."""
    loaded = prompt_lib.load_prompts_from_source()
    assert set(loaded) >= set(prompt_lib.PROMPT_IDS)
    for pid in prompt_lib.PROMPT_IDS:
        assert prompt_lib.sha256_text(loaded[pid]) == prompt_lib.EXPECTED_SHA256[pid]


def test_p1_is_the_env_module_default(rpg_src):
    """p1 must be `rpg_rl/env.py`'s DEFAULT_SYSTEM_PROMPT, with no override applied."""
    code = (
        "import sys, hashlib;"
        f"sys.path.insert(0, {os.path.join(rpg_src, 'rpg_rl')!r});"
        f"sys.path.insert(1, {os.path.join(rpg_src, 'rpg_v9')!r});"
        "import env;"
        "print(hashlib.sha256(env.DEFAULT_SYSTEM_PROMPT.encode()).hexdigest());"
        "print(hashlib.sha256(env.SYSTEM_PROMPT.encode()).hexdigest())"
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RPG_SYSTEM_PROMPT")}
    clean["RPG_PROTO"] = "rpg_v9"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=clean, check=True)
    default_sha, resolved_sha = out.stdout.split()
    assert default_sha == prompt_lib.EXPECTED_SHA256["p1"]
    # With no override set, SYSTEM_PROMPT is byte-identical to the default: the edit to
    # env.py must not have changed behavior for any existing caller.
    assert resolved_sha == default_sha


def test_env_module_honors_the_file_override(rpg_src, tmp_path: Path):
    """`RPG_SYSTEM_PROMPT_FILE` replaces SYSTEM_PROMPT; a wrong digest is a hard error."""
    prompt_file = tmp_path / "custom.txt"
    prompt_file.write_text("a custom system prompt\n", encoding="utf-8")
    expected = hashlib.sha256(prompt_file.read_bytes()).hexdigest()

    code = (
        "import sys, hashlib;"
        f"sys.path.insert(0, {os.path.join(rpg_src, 'rpg_rl')!r});"
        f"sys.path.insert(1, {os.path.join(rpg_src, 'rpg_v9')!r});"
        "import env;"
        "print(hashlib.sha256(env.SYSTEM_PROMPT.encode()).hexdigest())"
    )
    env = dict(os.environ, RPG_PROTO="rpg_v9", RPG_SYSTEM_PROMPT_FILE=str(prompt_file))

    ok = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        env=dict(env, RPG_SYSTEM_PROMPT_SHA256=expected))
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.strip() == expected

    bad = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(env, RPG_SYSTEM_PROMPT_SHA256="0" * 64))
    assert bad.returncode != 0
    assert "sha256 mismatch" in bad.stderr


def test_materialize_round_trips(cfg, tmp_path: Path):
    paths = prompt_lib.materialize(tmp_path)
    assert set(paths) == set(prompt_lib.PROMPT_IDS)
    reread = prompt_lib.read_materialized(tmp_path)
    for pid in prompt_lib.PROMPT_IDS:
        assert prompt_lib.sha256_text(reread[pid]) == prompt_lib.EXPECTED_SHA256[pid]
    assert (tmp_path / "prompts_manifest.json").exists()


def test_read_materialized_rejects_a_tampered_file(tmp_path: Path):
    prompt_lib.materialize(tmp_path)
    (tmp_path / "p2.txt").write_text("not the real p2", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt p2 changed"):
        prompt_lib.read_materialized(tmp_path)


def test_source_parquet_carries_p1(cfg):
    """The baked-in prompt of the required source datasets is exactly p1."""
    pd = pytest.importorskip("pandas")
    for source in (cfg.source_train, cfg.source_val):
        if not source.exists():
            pytest.skip(f"source dataset missing: {source}")
        frame = pd.read_parquet(source)
        digests = {hashlib.sha256(r[0]["content"].encode()).hexdigest() for r in frame["prompt"]}
        assert digests == {prompt_lib.EXPECTED_SHA256["p1"]}
