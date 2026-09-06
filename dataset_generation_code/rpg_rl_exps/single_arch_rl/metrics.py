#!/usr/bin/env python3
"""Per-step GRPO group statistics (log requirement 1).

SkyRL logs ``reward/avg_raw_reward`` (the mean over *trajectories*) but nothing about
how reward is distributed *within* a GRPO group. That within-group spread is the entire
gradient signal: an all-equal group contributes exactly zero to the policy update, so a
run whose ``group_reward_var`` collapses is not learning regardless of how its mean
moves. For an easy-vs-hard archetype comparison it is the number to watch.

A "group" is the ``n_samples_per_prompt`` rollouts SkyRL generated for one dataset row
(one world), identified by a shared uid.

    group_reward_mean = mean over groups of the group's mean reward
    group_reward_var  = mean over groups of the group's population variance

``group_reward_mean`` equals ``reward/avg_raw_reward`` whenever every group has the same
size (it always does here); it is logged under its own name because the pair is only
interpretable together. This module is pure so ``tests/test_metrics.py`` can check it
without a trainer.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Union

Reward = Union[float, Sequence[float]]

PREFIX = "reward/"


def _scalar(reward: Reward) -> float:
    """One trajectory's scalar reward.

    SkyRL hands ``rewards`` over either as one float per trajectory or, for token-level
    rewards, as a list per trajectory. ``get_metrics_from_generator_output`` sums the
    token list for its own ``avg_score``; do the same so the two agree.
    """
    if isinstance(reward, (list, tuple)):
        return float(sum(reward))
    return float(reward)


def group_rewards(rewards: Sequence[Reward], uids: Sequence[str]) -> "OrderedDict[str, List[float]]":
    """Bucket trajectory rewards by uid, preserving first-seen order."""
    if len(rewards) != len(uids):
        raise ValueError(f"rewards ({len(rewards)}) and uids ({len(uids)}) must be the same length")
    groups: "OrderedDict[str, List[float]]" = OrderedDict()
    for reward, uid in zip(rewards, uids):
        groups.setdefault(uid, []).append(_scalar(reward))
    return groups


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pvariance(values: Sequence[float]) -> float:
    """Population variance -- the spread GRPO actually centers on, not a sample estimate."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def group_reward_stats(
    rewards: Sequence[Reward],
    uids: Sequence[str],
    *,
    prefix: str = PREFIX,
    tol: float = 1e-9,
) -> Dict[str, float]:
    """Group statistics for one training step.

    Returns an empty dict for an empty batch rather than raising: a metrics hook must
    never be the reason a training step fails.
    """
    if not len(rewards):
        return {}
    groups = group_rewards(rewards, uids)
    means = [_mean(v) for v in groups.values()]
    variances = [_pvariance(v) for v in groups.values()]
    sizes = [len(v) for v in groups.values()]
    nondegenerate = [v for v in variances if v > tol]
    return {
        # -- the two required numbers -----------------------------------------------------
        f"{prefix}group_reward_mean": _mean(means),
        f"{prefix}group_reward_var": _mean(variances),
        # -- context that makes them readable ---------------------------------------------
        # Between-group variance: how much the *worlds* differ from each other. Paired with
        # group_reward_var it separates "this archetype is uniformly hard" from "some worlds
        # are solved and others never are".
        f"{prefix}group_reward_var_between": _pvariance(means),
        f"{prefix}group_reward_var_max": max(variances),
        f"{prefix}group_reward_var_min": min(variances),
        # Fraction of groups with any spread at all -- the groups that produce a gradient.
        f"{prefix}frac_nondegenerate_groups": len(nondegenerate) / len(groups),
        f"{prefix}num_groups": float(len(groups)),
        f"{prefix}group_size_min": float(min(sizes)),
        f"{prefix}group_size_max": float(max(sizes)),
    }
