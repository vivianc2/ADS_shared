"""Scripted screening prefill (2026-10-01, week-5 L4 / plan Y3).

Runs the first k experiments of a fixed systematic screen through a live RPGEnv and returns the
(assistant, user) message pairs, so an episode can start with that evidence already in context:

    plan = [measure every signal] + [each control alone at 100] + [each control alone at 0]
    (capped at budget - 2 experiments, so the policy always keeps >= 2 experiments of its own)

This is the same screen as personal_docs/tooling/box2_eval_harness/driver.py::_prefill_screen
(the week-4 U6 probe: base 9B part_a +0.195 when handed the full screen). k = len(plan) ("full")
gives pure evidence-use episodes (NEXT_WEEK S2); smaller k gives a reverse curriculum (L4).

Knob: RPG_PREFILL_K = "" / "0" (default: off, behaviour unchanged) | "<int>" | "full" | "rand"
("rand" = k drawn uniformly from 0..full, seeded by the WORLD seed only: reproducible, and every sample of one world
gets the same k, so a GRPO group compares like with like; k varies across worlds).
The prefilled turns go through env.step(), so the env's budget, turn and intervention counters
include them, exactly as if the policy had taken those actions.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple


def screening_plan(env) -> List[Tuple[str, dict]]:
    mids = env.cat.measurable_ids()
    plan = [("measure", {"ids": mids})]
    plan += [("intervene", {"actions": [{"actuator": a, "value": 100}], "measure": mids})
             for a in env.cat.actuator_ids()]
    plan += [("intervene", {"actions": [{"actuator": a, "value": 0}], "measure": mids})
             for a in env.cat.actuator_ids()]
    return plan[:max(0, env.budget - 2)]


def resolve_k(spec: Optional[str], env, rng: Optional[random.Random] = None) -> int:
    """Number of screening experiments to prefill for this episode (0 = none)."""
    spec = (spec or "").strip().lower()
    full = len(screening_plan(env))
    if spec in ("", "0"):
        return 0
    if spec == "full":
        return full
    if spec == "rand":
        return (rng or random).randint(0, full)
    return max(0, min(int(spec), full))


def action_text(typ: str, payload: dict) -> str:
    what = ("baseline: measure every signal" if typ == "measure" else
            f"screen {payload['actions'][0]['actuator']} alone at {payload['actions'][0]['value']}")
    return (f"<reasoning>Systematic screening ({what}).</reasoning>\n"
            f'<action type="{typ}">{json.dumps(payload)}</action>\n<memory>screening in progress</memory>')


def prefill(env, k: int, obs_suffix: str = "") -> Tuple[List[Dict[str, str]], bool]:
    """Run the first k screening experiments in `env` (already reset). Returns (messages, done):
    messages = [assistant, user, assistant, user, ...]; done=True if the episode ended (should not
    happen with the budget-2 cap; callers must treat it as an error)."""
    msgs: List[Dict[str, str]] = []
    for typ, payload in screening_plan(env)[:k]:
        text = action_text(typ, payload)
        msgs.append({"role": "assistant", "content": text})
        obs, _r, done, _i = env.step(text)
        if done:
            return msgs, True
        msgs.append({"role": "user", "content": obs + obs_suffix})
    return msgs, False


def spec_from_env() -> str:
    return os.environ.get("RPG_PREFILL_K", "")
