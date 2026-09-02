"""The evaluation report reads SkyRL's dumps and produces the five required numbers."""

from __future__ import annotations

import json

from prompt_compare_rl import report_eval
from prompt_compare_rl.config import PROMPT_IDS


def _write_dump(cfg, prompt_id, step, aggregated, episodes):
    directory = cfg.run_paths(prompt_id)["export_path"] / "dumped_evals" / f"global_step_{step}_evals"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aggregated_results.jsonl").write_text(json.dumps(aggregated) + "\n", encoding="utf-8")
    with (directory / "rpg_v7.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")


def _cfg_in(tmp_path, monkeypatch):
    monkeypatch.setenv("PC_OUT_ROOT", str(tmp_path))
    monkeypatch.setenv("PC_EXP_TAG", "unittest")
    from prompt_compare_rl.config import ExperimentConfig

    return ExperimentConfig()


def test_collects_all_five_metrics(tmp_path, monkeypatch):
    cfg = _cfg_in(tmp_path, monkeypatch)
    aggregated = {
        "eval/all/avg_score": 0.25,
        "eval/all/environment/part_a": 0.4,
        "eval/all/environment/part_b": 0.1,
        "eval/all/environment/truncated": 0.2,
        "eval/all/environment/turns": 9.5,
        "eval/all/environment/archetype/dose_window/score": 0.3,
    }
    episodes = [
        {"score": 0.5, "stop_reason": "stop"},
        {"score": 0.0, "stop_reason": "length"},
    ]
    _write_dump(cfg, "p1", 4, aggregated, episodes)

    report = report_eval.collect(cfg, ["p1"], [0, 4, 8])
    row = report["runs"]["p1"]["by_step"][4]
    assert row["status"] == "ok"
    assert row["avg_score"] == 0.25
    assert row["part_a"] == 0.4
    assert row["part_b"] == 0.1
    assert row["truncation"] == 0.2
    assert row["mean_turns"] == 9.5
    assert row["episodes"] == 2
    assert row["avg_score_from_episodes"] == 0.25
    assert row["truncation_from_stop_reason"] == 0.5
    assert row["per_archetype"] == {"dose_window": 0.3}

    assert report["runs"]["p1"]["by_step"][0]["status"] == "missing"
    assert report["runs"]["p1"]["by_step"][8]["status"] == "missing"


def test_table_renders_missing_and_present_steps(tmp_path, monkeypatch):
    cfg = _cfg_in(tmp_path, monkeypatch)
    _write_dump(cfg, "p2", 0, {"eval/all/avg_score": 0.1}, [{"score": 0.1, "stop_reason": "stop"}])
    report = report_eval.collect(cfg, list(PROMPT_IDS), [0, 4, 8])
    table = report_eval.render_table(report)
    assert "p2" in table and "0.1000" in table
    assert "no eval dump yet" in table
    assert "avg_score" in table and "truncation" in table and "mean_turns" in table


def test_metric_keys_match_the_environment_metric_names():
    """The report must read exactly the keys SkyRL builds from get_metrics()."""
    keys = dict(report_eval.METRIC_COLUMNS)
    assert keys["part_a"] == "eval/all/environment/part_a"
    assert keys["part_b"] == "eval/all/environment/part_b"
    assert keys["truncation"] == "eval/all/environment/truncated"
    assert keys["mean_turns"] == "eval/all/environment/turns"
    assert keys["avg_score"] == "eval/all/avg_score"
