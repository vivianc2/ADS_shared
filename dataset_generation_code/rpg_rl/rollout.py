#!/usr/bin/env python3
"""Multi-turn rollouts over RPGEnv, and group rollouts for GRPO.

An episode is: reset() -> [gen turn -> env.step]* -> terminal reward. We record each
turn's (observation shown, completion produced, parsed action type) so a trainer can
later build per-turn (prompt, completion, advantage) examples — the multi-turn -> GRPO
bridge for GRPO (phase-1 RL design).

`rollout_group` runs G episodes on the SAME world (same seed => same catalog) — that is
the GRPO group. Rollouts are independent, so we thread them (vLLM continuous batching
turns the G concurrent request streams into ~one wall-clock).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from env import RPGEnv, SYSTEM_PROMPT

GenFn = Callable[[str, str, Optional[int]], str]   # (system, user, max_new_tokens) -> text


@dataclass
class Turn:
    obs: str                    # the observation (user prompt) shown this turn
    completion: str             # the model's raw output for this turn
    action_type: Optional[str]  # parsed action ("measure"/"intervene"/"code"/"answer"/...)


@dataclass
class Trajectory:
    world_id: str
    turns: List[Turn]
    reward: float               # terminal reward
    part_a: float
    part_b: float
    accepted: bool
    n_turns: int
    info: Dict[str, Any] = field(default_factory=dict)


def rollout_episode(env: RPGEnv, gen_fn: GenFn, max_new_tokens: int = 768,
                    max_turns_hard: int = 64) -> Trajectory:
    """Run one full episode. `env` must be freshly constructed (or reset-able)."""
    obs = env.reset()
    turns: List[Turn] = []
    reward, info, done = 0.0, {}, False
    guard = 0
    while not done and guard < max_turns_hard:
        guard += 1
        completion = gen_fn(SYSTEM_PROMPT, obs, max_new_tokens)
        nobs, reward, done, info = env.step(completion)
        atype = env.turns[-1].get("action_type") if env.turns else None
        turns.append(Turn(obs=obs, completion=completion, action_type=atype))
        obs = nobs
    return Trajectory(
        world_id=env.cat.world_id, turns=turns, reward=float(reward),
        part_a=float(info.get("part_a", 0.0)), part_b=float(info.get("part_b", 0.0)),
        accepted=bool(info.get("accepted", False)), n_turns=len(turns), info=info)


def rollout_group(bundle, gen_fn: GenFn, G: int = 8, max_new_tokens: int = 768,
                  max_workers: Optional[int] = None,
                  env_kwargs: Optional[Dict[str, Any]] = None) -> List[Trajectory]:
    """G independent episodes on one world (a GRPO group). Threaded for vLLM batching."""
    env_kwargs = env_kwargs or {}

    def one(_i):
        env = bundle.make_env(**env_kwargs)
        return rollout_episode(env, gen_fn, max_new_tokens)

    with ThreadPoolExecutor(max_workers=max_workers or G) as ex:
        return list(ex.map(one, range(G)))
