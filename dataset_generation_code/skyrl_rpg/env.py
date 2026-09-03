#!/usr/bin/env python3
"""RPGSkyEnv — SkyRL-Gym `BaseTextEnv` wrapper over our verified `RPGEnv`.

SkyRL drives a multi-turn rollout: it shows the model the dataset `prompt`, the model
emits a turn (a string), SkyRL calls `env.step(action)`; the env returns the next
observation (as chat messages), a reward, and `done`. Our `RPGEnv` already has exactly
this shape (`reset()`/`step(text)->(obs, reward, done, info)`) with a PURE id-space
terminal reward, so this wrapper is thin.

Each sample corresponds to ONE world. The dataset row carries the world's identity in
`extra_info` (seed, skin, archetype); we rebuild the exact world deterministically
(sample_world -> audit) and attach its precomputed gold/battery so the env never
recomputes the oracle per step (decision V9). Because generation is deterministic in the
seed, the env's `reset()` observation equals the observation rendered into the dataset
prompt at build time.

Reward is TERMINAL (0 on measure/intervene/code turns, the graded scalar at
answer/give_up/turn-cap). SkyRL sums per-turn rewards, so terminal-only yields the correct
trajectory reward for GRPO.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

# --- make the verified science + RL-env code importable in ANY context, including Ray
# --- workers that run from a copied working_dir (_ray_pkg) where realpath(__file__) no
# --- longer points at our repo. Use the stable absolute mount path (override via RPG_SRC). ---
_BASE = os.environ.get("RPG_SRC", "/work/ADS_shared/dataset_generation_code")
for _p in (os.path.join(_BASE, "rpg_rl"), os.path.join(_BASE, os.environ.get("RPG_PROTO", "rpg_v9"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput  # noqa: E402

from env import RPGEnv, SYSTEM_PROMPT       # rpg_rl/env.py  (RL environment)   # noqa: E402
from sampler import sample_world            # rpg_v9/sampler.py                  # noqa: E402
from generate_v7 import audit               # rpg_v9/generate_v7.py              # noqa: E402


def build_rpg_env(seed: int, skin: str, archetype: str,
                  max_turns: int = 32, budget: int = 15,
                  data_dir: str = None) -> RPGEnv:
    """Deterministically rebuild one world and wrap it in a fresh RPGEnv (gold/battery
    precomputed by the audit, so no per-episode oracle recompute).

    ``data_dir`` MUST be a per-episode-unique path: the sim writes experiment_<n>.csv
    there for the code tool. If left None the code tool is inert (the experiment_<n>_csv
    variables never exist) — that was the training-path bug where the prompt advertised a
    dead tool. G=8 rollouts of the same seed run concurrently, so the dir must be unique
    per rollout instance (not keyed by seed) to avoid CSV filename collisions."""
    world = sample_world(int(seed), skin=skin, archetype=archetype)
    world["ground_truth"]["_seed"] = int(seed)
    res = audit(world)                       # returns gold + battery regardless of ok
    return RPGEnv(world=world, gold=res["gold"], battery=res["battery"],
                  catalog_seed=int(seed), max_turns=int(max_turns), budget=int(budget),
                  data_dir=data_dir)


class RPGSkyEnv(BaseTextEnv):
    """One episode over one RPG world, exposed through the SkyRL-Gym interface."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        info = extras.get("extra_info", extras) or {}
        assert "seed" in info and "skin" in info and "archetype" in info, \
            "extra_info must carry {seed, skin, archetype} for the RPG world"
        # Per-episode-unique scratch dir so the code tool can read experiment CSVs.
        # Unique per rollout instance (mkdtemp) so the G concurrent samples of one seed
        # don't clobber each other's experiment_<n>.csv. Cleaned up when the episode ends.
        import tempfile
        self._data_dir = tempfile.mkdtemp(prefix="rpg_ep_",
                                          dir=os.environ.get("RPG_DATA_ROOT") or None)
        self._rpg = build_rpg_env(
            seed=info["seed"], skin=info["skin"], archetype=info["archetype"],
            max_turns=int(info.get("max_turns", 32)), budget=int(info.get("budget", 15)),
            data_dir=self._data_dir)
        self.max_turns = self._rpg.max_turns
        # reset now so the very first step advances the correct world state; the first
        # observation is identical to the one baked into the dataset prompt (determinism).
        self._first_obs = self._rpg.reset()

    def init(self, prompt):
        # The dataset prompt already holds [system(SYSTEM_PROMPT), user(first_obs)];
        # nothing to modify.
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        obs, reward, done, info = self._rpg.step(action)
        # RPG_NO_THINK=1 -> append Qwen3's /no_think directive to every turn (forces
        # thinking-off for the debug loop; SkyRL has no chat-template thinking flag).
        if not done and os.environ.get("RPG_NO_THINK"):
            obs = obs + "\n/no_think"
        if done:  # episode over -> remove the per-episode scratch dir
            import shutil
            shutil.rmtree(getattr(self, "_data_dir", None) or "/nonexistent", ignore_errors=True)
        observations = [] if done else [{"role": "user", "content": obs}]
        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=bool(done),
            metadata={k: info.get(k) for k in
                      ("part_a", "part_b", "accepted", "turn_type", "n_interventions",
                       "reward_error")},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"turns": self._rpg._turn, "interventions": self._rpg._n_interv}
