#!/usr/bin/env python3
"""SkyRL entrypoint with a binary any-causal-intervention reward."""

import sys
from typing import Any, Dict

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register

from examples.train.rpg.env import RPGSkyEnv, RewardConfig


class RPGAnyInterventionEnv(RPGSkyEnv):
    """Reward an answer iff it names at least one causal intervention.

    Dose quality, mechanism fields, extra causal interventions, invalid IDs, and
    the trajectory's experiment history do not change the scalar reward.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__(env_config=env_config, extras=extras)
        self._rpg.reward_cfg = RewardConfig(
            w_a=0.0,
            w_b=0.0,
            c_invalid=0.0,
            strict_part_b=True,
            require_evidence=False,
            c_no_evidence=0.0,
            lever_gate=False,
            lever_only=True,
            lever_mode="any",
            lever_bonus=0.0,
            lever_exact=False,
            lever_precision=False,
            lever_max_extra=-1,
            lever_scale="none",
            w_id=0.0,
        )
        # Keep the trajectory reward strictly terminal and binary even if a caller
        # happens to export RPG_BELIEF_SHAPING.
        self._belief_shaping = False


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="rpg",
        entry_point="rl_new_reward_500_steps.main_rpg_long:RPGAnyInterventionEnv",
    )
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
