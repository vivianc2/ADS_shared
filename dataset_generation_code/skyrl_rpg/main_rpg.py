#!/usr/bin/env python3
"""SkyRL training entrypoint for the RPG environment (mirrors examples/train/multiply).

Registers env_class "rpg" -> RPGSkyEnv inside the Ray entrypoint task (no fork of
skyrl/skyrl-gym needed), then runs the standard PPO/GRPO experiment loop.

Launch (inside the container, from the SkyRL repo root, with this package symlinked to
examples/train/rpg):
    uv run --isolated --extra fsdp -m examples.train.rpg.main_rpg <config overrides>
"""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="rpg",
        entry_point="examples.train.rpg.env:RPGSkyEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
