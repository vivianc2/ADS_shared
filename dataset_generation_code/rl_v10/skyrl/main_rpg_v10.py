#!/usr/bin/env python3
"""SkyRL entry point for RL on RPG v10: registers env_class "rpg_v10", forwards RLV10_*/RPG_* env vars to the Ray
workers (SkyRL forwards only an allowlist; see skyrl_rpg/main_rpg.py for why this matters), and trains with
RLv10Trainer (trainer_rlv10.py: overlong turn-sequences dropped from the loss, 2026-10-25)."""
import os
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


class RLv10Exp(BasePPOExp):
    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator,
                    colocate_pg):
        from examples.train.rpg_v10.trainer_rlv10 import RLv10Trainer
        return RLv10Trainer(cfg=cfg, tracker=tracker, tokenizer=tokenizer, train_dataset=train_dataset,
                            eval_dataset=eval_dataset, inference_engine_client=inference_engine_client,
                            generator=generator, colocate_pg=colocate_pg)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(id="rpg_v10", entry_point="examples.train.rpg_v10.env:RPGv10Env")
    exp = RLv10Exp(cfg)
    exp.run()


def _env_vars() -> dict:
    return {k: v for k, v in os.environ.items()
            if k.startswith(("RLV10_", "RPG_")) or k in ("HF_HOME", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")}


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ev = _env_vars()
    print("[main_rpg_v10] env forwarded to ray workers: " + repr(sorted(ev.items())), flush=True)
    ray.get(skyrl_entrypoint.options(runtime_env={"env_vars": ev}).remote(cfg))


if __name__ == "__main__":
    main()
