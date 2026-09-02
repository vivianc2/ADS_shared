"""The SkyRL environment: prompt verification, metric contract, and a real episode.

No GPU and no model are involved -- a scripted "policy" drives the verified RPG
simulator, which is enough to prove the metric plumbing the evaluation report depends
on.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from prompt_compare_rl import prompts as prompt_lib
from prompt_compare_rl.config import PROMPT_IDS


@pytest.fixture(scope="module")
def sky_env_module(rpg_src):
    os.environ["RPG_SRC"] = rpg_src
    os.environ["RPG_PROTO"] = "rpg_v9"
    from prompt_compare_rl import sky_env

    return sky_env


@pytest.fixture()
def world_extra(datasets_built):
    pd = pytest.importorskip("pandas")
    frame = pd.read_parquet(datasets_built.dataset_dir("p1") / "validation_small.parquet")
    row = frame.iloc[0]
    return dict(row["extra_info"]), row["prompt"]


def test_registered_entry_point_resolves(sky_env_module):
    """The exact string `main.py` registers must import to our class."""
    from skyrl_gym.envs.registration import load_env_creator

    from prompt_compare_rl.main import ENV_ENTRY_POINT

    assert load_env_creator(ENV_ENTRY_POINT) is sky_env_module.PromptCompareRPGEnv


def test_prompt_assertion_accepts_and_rejects(sky_env_module, datasets_built, monkeypatch):
    texts = prompt_lib.read_materialized(datasets_built.prompt_dir)
    for pid in PROMPT_IDS:
        monkeypatch.setenv("PC_PROMPT_ID", pid)
        monkeypatch.setenv("PC_PROMPT_SHA256", prompt_lib.sha256_text(texts[pid]))
        sky_env_module._verify_system_prompt(
            [{"role": "system", "content": texts[pid]}, {"role": "user", "content": "obs"}]
        )
        for other in PROMPT_IDS:
            if other == pid:
                continue
            with pytest.raises(RuntimeError, match="SYSTEM PROMPT MISMATCH"):
                sky_env_module._verify_system_prompt(
                    [{"role": "system", "content": texts[other]}, {"role": "user", "content": "obs"}]
                )


def test_prompt_assertion_requires_a_system_message(sky_env_module, monkeypatch):
    monkeypatch.setenv("PC_PROMPT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="system"):
        sky_env_module._verify_system_prompt([{"role": "user", "content": "obs"}])


def test_prompt_assertion_is_inert_without_the_env_var(sky_env_module, monkeypatch):
    """Outside a prompt-compare run the class must behave like a plain RPG env."""
    monkeypatch.delenv("PC_PROMPT_SHA256", raising=False)
    sky_env_module._verify_system_prompt([{"role": "user", "content": "obs"}])


def _make_env(sky_env_module, extra, tmp_path):
    os.environ["RPG_DATA_ROOT"] = str(tmp_path)
    return sky_env_module.PromptCompareRPGEnv(extras={"extra_info": extra})


def test_truncated_episode_metrics(sky_env_module, world_extra, tmp_path, monkeypatch):
    """A trajectory abandoned mid-episode reports truncated=1 and zeroed scores."""
    monkeypatch.delenv("PC_PROMPT_SHA256", raising=False)
    extra, prompt = world_extra
    env = _make_env(sky_env_module, extra, tmp_path)
    env.init(list(prompt))
    out = env.step('<reasoning>look</reasoning><action type="measure">{"ids": ["m0"]}</action>')
    assert out["done"] is False

    metrics = env.get_metrics()
    assert metrics["truncated"] == 1.0 and metrics["completed"] == 0.0
    assert metrics["score"] == 0.0 and metrics["part_a"] == 0.0 and metrics["part_b"] == 0.0
    assert metrics["turns"] == 1.0
    assert metrics["archetype"] == extra["archetype"]


def test_completed_episode_metrics(sky_env_module, world_extra, tmp_path, monkeypatch):
    """A terminal episode reports the graded parts and turn/intervention counts."""
    monkeypatch.delenv("PC_PROMPT_SHA256", raising=False)
    extra, prompt = world_extra
    env = _make_env(sky_env_module, extra, tmp_path)
    env.init(list(prompt))

    env.step('<reasoning>read</reasoning><action type="measure">{"ids": ["m0","m1"]}</action>')
    env.step(
        '<reasoning>test</reasoning>'
        '<action type="intervene">{"actions":[{"actuator":"a0","value":80}],"measure":["m0"]}</action>'
    )
    out = env.step(
        '<reasoning>answer</reasoning>'
        '<action type="answer">{"actions":[{"actuator":"a0","value":80}],'
        '"proxy":"m0","decoys":["m1"],"signs":{"a0":"+"}}</action>'
    )
    assert out["done"] is True

    metrics = env.get_metrics()
    assert metrics["truncated"] == 0.0 and metrics["completed"] == 1.0
    assert metrics["turns"] == 3.0
    assert metrics["interventions"] >= 1.0
    assert metrics["queries_used"] == 2.0
    for key in ("score", "part_a", "part_b", "accepted", "invalid_id_fraction", "forced_no_answer"):
        assert isinstance(metrics[key], float)
    assert 0.0 <= metrics["part_a"] <= 1.0 and 0.0 <= metrics["part_b"] <= 1.0
    assert metrics["score"] == pytest.approx(float(out["reward"]))


def test_metrics_cover_every_reported_column(sky_env_module, world_extra, tmp_path, monkeypatch):
    """report_eval.py reads these five keys; the env must always emit them."""
    monkeypatch.delenv("PC_PROMPT_SHA256", raising=False)
    extra, prompt = world_extra
    env = _make_env(sky_env_module, extra, tmp_path)
    env.init(list(prompt))
    env.step('<reasoning>x</reasoning><action type="give_up">{}</action>')
    metrics = env.get_metrics()
    for key in ("score", "part_a", "part_b", "truncated", "turns"):
        assert key in metrics


def test_aggregate_metrics_means_and_archetypes(sky_env_module):
    episodes = [
        {"score": 1.0, "part_a": 1.0, "part_b": 0.0, "truncated": 0.0, "turns": 4.0, "archetype": "a"},
        {"score": 0.0, "part_a": 0.0, "part_b": 0.0, "truncated": 1.0, "turns": 8.0, "archetype": "a"},
        {"score": 0.5, "part_a": 0.5, "part_b": 0.5, "truncated": 0.0, "turns": 6.0, "archetype": "b"},
    ]
    aggregated = sky_env_module.PromptCompareRPGEnv.aggregate_metrics(episodes)
    assert aggregated["score"] == pytest.approx(0.5)
    assert aggregated["truncated"] == pytest.approx(1 / 3)
    assert aggregated["turns"] == pytest.approx(6.0)
    assert aggregated["archetype/a/score"] == pytest.approx(0.5)
    assert aggregated["archetype/b/turns"] == pytest.approx(6.0)
    assert aggregated["archetype/a/episodes"] == 2.0
    # the non-numeric field must not leak into the flat metrics
    assert "archetype" not in aggregated


def test_episode_scratch_dir_is_inside_the_run_tree(sky_env_module, world_extra, tmp_path, monkeypatch):
    """The `code` tool's CSVs must land in this run's directory, not a shared /tmp."""
    monkeypatch.delenv("PC_PROMPT_SHA256", raising=False)
    extra, _prompt = world_extra
    env = _make_env(sky_env_module, extra, tmp_path)
    assert str(tmp_path) in env._data_dir
