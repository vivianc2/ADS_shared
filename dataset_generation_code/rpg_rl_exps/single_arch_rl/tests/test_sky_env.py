"""The env reports avg score / part_a / part_b per archetype (log requirement 2).

``SingleArchRPGEnv`` is only importable with skyrl_gym present; the aggregation logic is
tested through the class so the test fails if the real base class changes shape.
"""

from __future__ import annotations

import pytest

pytest.importorskip("skyrl_gym")

from single_arch_rl.sky_env import ARCHETYPE_METRIC_KEYS, SingleArchRPGEnv  # noqa: E402


def _episode(archetype, score, part_a, part_b, truncated=0.0, turns=5.0):
    return {
        "archetype": archetype, "score": score, "part_a": part_a, "part_b": part_b,
        "truncated": truncated, "turns": turns, "accepted": 0.0, "completed": 1.0,
        "reward_error": 0.0, "invalid_id_fraction": 0.0, "forced_no_answer": 0.0,
        "interventions": 2.0, "queries_used": 3.0, "parse_failures": 0.0,
    }


def test_per_archetype_breakdown_of_the_three_headline_numbers():
    metrics = [
        _episode("dose_window", 0.4, 0.6, 0.2),
        _episode("dose_window", 0.6, 0.8, 0.4),
        _episode("confounded_reversal", 0.0, 0.0, 0.0),
    ]
    out = SingleArchRPGEnv.aggregate_metrics(metrics)
    assert out["archetype/dose_window/score"] == pytest.approx(0.5)
    assert out["archetype/dose_window/part_a"] == pytest.approx(0.7)
    assert out["archetype/dose_window/part_b"] == pytest.approx(0.3)
    assert out["archetype/dose_window/episodes"] == 2
    assert out["archetype/confounded_reversal/score"] == pytest.approx(0.0)
    assert out["archetype/confounded_reversal/episodes"] == 1


def test_every_documented_key_is_broken_out():
    out = SingleArchRPGEnv.aggregate_metrics([_episode("dose_window", 0.4, 0.6, 0.2)])
    for key in ARCHETYPE_METRIC_KEYS:
        assert f"archetype/dose_window/{key}" in out


def test_flat_keys_keep_skyrls_default_meaning():
    """The pooled numbers must stay where SkyRL's own aggregator would have put them."""
    from skyrl_gym.metrics import default_aggregate_metrics

    metrics = [_episode("dose_window", 0.4, 0.6, 0.2), _episode("synergy_pair", 0.0, 0.0, 0.0)]
    default = default_aggregate_metrics([{k: v for k, v in m.items() if k != "archetype"}
                                         for m in metrics])
    out = SingleArchRPGEnv.aggregate_metrics(metrics)
    for key, value in default.items():
        assert out[key] == pytest.approx(value), key


def test_a_missing_archetype_label_is_bucketed_not_dropped():
    out = SingleArchRPGEnv.aggregate_metrics([{"score": 0.5, "part_a": 0.5, "part_b": 0.5,
                                               "truncated": 0.0, "turns": 3.0}])
    assert out["archetype/unknown/episodes"] == 1
