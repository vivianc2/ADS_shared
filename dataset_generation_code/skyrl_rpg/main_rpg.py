#!/usr/bin/env python3
"""SkyRL training entrypoint for the RPG environment (mirrors examples/train/multiply).

Registers env_class "rpg" -> RPGSkyEnv inside the Ray entrypoint task (no fork of
skyrl/skyrl-gym needed), then runs the standard PPO/GRPO experiment loop.

Launch (inside the container, from the SkyRL repo root, with this package symlinked to
examples/train/rpg):
    uv run --isolated --extra fsdp -m examples.train.rpg.main_rpg <config overrides>
"""

import os
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


def _rpg_env_vars() -> dict:
    """The RPG_* knobs the reward/env read from os.environ, forwarded to every Ray worker.

    WHY (2026-09-22). RewardConfig reads RPG_LEVER_GATE / RPG_LEVER_SCALE / RPG_W_ID / ... at
    construction time, inside the Ray env actors. SkyRL's prepare_runtime_environment forwards
    only a curated allowlist (NCCL/VLLM/WANDB/MLFLOW), so RPG_* reaches workers ONLY by
    inheritance from the raylet's environment. That holds when ray.init() boots the cluster as a
    child of the launcher, but NOT when the job attaches to a cluster started separately
    (`ray start --head`) -- which is required on a shared box where /tmp/ray already hosts
    another tenant's cluster. There the workers see NO RPG_* and the run silently computes the
    DEFAULT reward (r1) while every log line claims the gate is on. Verified: a probe task on
    such a cluster returned {} for RPG_*.

    Forwarding here makes the reward configuration a property of the JOB, not of how the cluster
    happened to be started. Ray propagates a task's runtime_env to the actors it creates.
    """
    return {k: v for k, v in os.environ.items()
            if k.startswith("RPG_") or k in ("HF_HOME", "PYTORCH_CUDA_ALLOC_CONF")}


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    rpg_env = _rpg_env_vars()
    print("[main_rpg] RPG_* forwarded to ray workers: "
          + repr({k: v for k, v in sorted(rpg_env.items()) if k.startswith("RPG_")}), flush=True)
    ray.get(skyrl_entrypoint.options(runtime_env={"env_vars": rpg_env}).remote(cfg))


if __name__ == "__main__":
    main()
