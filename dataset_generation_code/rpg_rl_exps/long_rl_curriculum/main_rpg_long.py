#!/usr/bin/env python3
"""SkyRL entrypoint for the two-stage RPG reward curriculum.

Training and evaluation rollouts through global step 150 use only reward part A.
Rollouts at steps 151-300 use only reward part B. The existing invalid-ID penalty
and evidence gate remain active in both stages.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

import ray
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register

from examples.train.rpg.env import RPGSkyEnv, RewardConfig


PART_A_LAST_STEP = 150


def curriculum_weights(global_step: int) -> tuple[float, float]:
    """Return (part-A weight, part-B weight) for a SkyRL global step."""
    return (1.0, 0.0) if global_step <= PART_A_LAST_STEP else (0.0, 1.0)


class RPGCurriculumEnv(RPGSkyEnv):
    """The working RPG environment with its terminal reward stage selected."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        self.curriculum_global_step = int(extras.get("curriculum_global_step", 0))
        self.curriculum_training_phase = str(extras.get("curriculum_training_phase", "train"))
        super().__init__(env_config=env_config, extras=extras)

        w_a, w_b = curriculum_weights(self.curriculum_global_step)
        self._rpg.reward_cfg = RewardConfig(w_a=w_a, w_b=w_b)
        # This run is explicitly terminal part-A/part-B curriculum training.
        self._belief_shaping = False
        self._terminal_reward_metrics: Dict[str, float] = {}

    def step(self, action: str):
        output = super().step(action)
        if output["done"]:
            metadata = output.get("metadata", {})
            self._terminal_reward_metrics = {
                "raw_part_a": float(metadata.get("part_a") or 0.0),
                "raw_part_b": float(metadata.get("part_b") or 0.0),
                "active_reward": float(output["reward"]),
            }
        return output

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        w_a, w_b = curriculum_weights(self.curriculum_global_step)
        metrics.update(
            {
                "curriculum_global_step": self.curriculum_global_step,
                "curriculum_part_a_weight": w_a,
                "curriculum_part_b_weight": w_b,
                **self._terminal_reward_metrics,
            }
        )
        return metrics


class CurriculumGenerator(SkyRLGymGenerator):
    """Pass the trainer's authoritative step number into every RPG rollout."""

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        batch_metadata = input_batch.get("batch_metadata")
        global_step = int(getattr(batch_metadata, "global_step", 0))
        training_phase = str(getattr(batch_metadata, "training_phase", "train"))
        for extras in input_batch.get("env_extras") or []:
            extras["curriculum_global_step"] = global_step
            extras["curriculum_training_phase"] = training_phase
        return await super().generate(input_batch)


class CurriculumPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return CurriculumGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="rpg",
        entry_point="long_rl_curriculum.main_rpg_long:RPGCurriculumEnv",
    )
    CurriculumPPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
