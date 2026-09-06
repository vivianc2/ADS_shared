"""group_reward_mean / group_reward_var, the two numbers log requirement (1) names."""

from __future__ import annotations

import math

import pytest

from single_arch_rl.metrics import group_reward_stats, group_rewards


def _uids(n_groups, group_size):
    return [f"w{g}" for g in range(n_groups) for _ in range(group_size)]


def test_grouping_is_by_uid_and_order_preserving():
    groups = group_rewards([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"])
    assert list(groups) == ["a", "b"]
    assert groups["a"] == [1.0, 3.0] and groups["b"] == [2.0, 4.0]


def test_mean_and_variance_on_a_hand_computed_case():
    # group a: [0, 1] -> mean 0.5, population var 0.25
    # group b: [1, 1] -> mean 1.0, population var 0.0
    stats = group_reward_stats([0.0, 1.0, 1.0, 1.0], ["a", "a", "b", "b"])
    assert stats["reward/group_reward_mean"] == pytest.approx(0.75)
    assert stats["reward/group_reward_var"] == pytest.approx(0.125)
    assert stats["reward/group_reward_var_max"] == pytest.approx(0.25)
    assert stats["reward/group_reward_var_min"] == pytest.approx(0.0)
    assert stats["reward/frac_nondegenerate_groups"] == pytest.approx(0.5)
    assert stats["reward/num_groups"] == 2


def test_group_reward_mean_equals_the_flat_mean_for_equal_sized_groups():
    """It must agree with SkyRL's own reward/avg_raw_reward, or the pair misleads."""
    rewards = [0.1, 0.9, 0.3, 0.4, 0.0, 1.0]
    uids = _uids(3, 2)
    stats = group_reward_stats(rewards, uids)
    assert stats["reward/group_reward_mean"] == pytest.approx(sum(rewards) / len(rewards))


def test_all_equal_rewards_give_zero_variance_and_no_gradient_groups():
    stats = group_reward_stats([0.0] * 16, _uids(2, 8))
    assert stats["reward/group_reward_var"] == 0.0
    assert stats["reward/group_reward_var_between"] == 0.0
    assert stats["reward/frac_nondegenerate_groups"] == 0.0


def test_between_group_variance_separates_uniform_from_bimodal():
    """Same flat mean, same within-group spread, different world-to-world spread."""
    uniform = group_reward_stats([0.0, 1.0, 0.0, 1.0], ["a", "a", "b", "b"])
    bimodal = group_reward_stats([0.5, 1.5, -0.5, 0.5], ["a", "a", "b", "b"])
    assert uniform["reward/group_reward_mean"] == pytest.approx(bimodal["reward/group_reward_mean"])
    assert uniform["reward/group_reward_var"] == pytest.approx(bimodal["reward/group_reward_var"])
    assert uniform["reward/group_reward_var_between"] == 0.0
    assert bimodal["reward/group_reward_var_between"] > 0.0


def test_token_level_rewards_are_summed_like_skyrl_does():
    stats = group_reward_stats([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], ["a", "a"])
    assert stats["reward/group_reward_mean"] == pytest.approx(0.5)
    assert stats["reward/group_reward_var"] == pytest.approx(0.25)


def test_empty_batch_is_not_an_error():
    assert group_reward_stats([], []) == {}


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        group_reward_stats([1.0, 2.0], ["a"])


def test_singleton_group_has_zero_variance_not_nan():
    stats = group_reward_stats([0.7], ["a"])
    assert stats["reward/group_reward_var"] == 0.0
    assert not math.isnan(stats["reward/group_reward_mean"])
