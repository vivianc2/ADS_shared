#!/usr/bin/env python3
"""Fuzz RPGEnv.step with malformed model output — no input may raise (2026-10-01).

Why: one malformed action ({"actuator": {...}}) raised inside RPGEnv.step and killed a 25-update SkyRL run (pando v7).
Every action type is fed payloads with wrong JSON types in every field; the env must answer with an error observation
or a terminal reward, never an exception.

Run:  PYTHONPATH=../rpg_v9 python test_env_robustness.py
"""
from __future__ import annotations
import itertools, json, sys, traceback
from sampler import sample_world
from generate_v7 import audit
from env import RPGEnv

BAD = [None, 1, 1.5, True, "a0", "zzz", [], ["a0"], [{"x": 1}], {}, {"id": "a0"}, {"actuator": "a0"}, "1e309", float("nan")]


def payloads():
    for v in BAD:
        yield "measure", {"ids": v}
        yield "measure", {"ids": [v]}
        yield "intervene", {"actions": v, "measure": ["m0"]}
        yield "intervene", {"actions": [v], "measure": ["m0"]}
        yield "intervene", {"actions": [{"actuator": v, "value": 50}], "measure": ["m0"]}
        yield "intervene", {"actions": [{"actuator": "a0", "value": v}], "measure": [v]}
        yield "answer", {"actions": [{"actuator": v, "value": v}], "proxy": v, "decoys": v, "signs": v, "policy": v}
        yield "answer", {"actions": v, "policy": {"treatment": v, "stratifier": v, "threshold": v,
                                                   "dose_if_ge": v, "dose_if_lt": v}, "signs": {"a0": v}}
        yield "code", {"code": v}
    for raw in ['<action type="intervene">[1,2]</action>', '<action type="measure">"m0"</action>',
                '<action type="answer">null</action>', '<action type="intervene">{"actions":[{"actuator":["a0"]}]}</action>',
                '<action type="nonsense">{}</action>', '<action>{}</action>', '</think></think><action type="measure">{}</action>']:
        yield None, raw


def main():
    w = None
    for s in range(100020, 100040):
        cand = sample_world(s, skin="bioprocess", archetype="confounded_chain")
        res = audit(cand)
        if res["ok"]:
            w, gold, battery, seed = cand, res["gold"], res["battery"], s
            break
    fails, n = [], 0
    for atype, payload in payloads():
        env = RPGEnv(world=w, gold=gold, battery=battery, catalog_seed=seed); env.reset()
        text = payload if atype is None else f'<action type="{atype}">{json.dumps(payload, allow_nan=True)}</action>'
        n += 1
        try:
            env.step(text)
            if not env._done:                     # a follow-up valid turn must still work
                env.step('<action type="measure">{"ids":["m0"]}</action>')
        except Exception as e:
            fails.append((text[:120], f"{type(e).__name__}: {e}"))
    for t, e in fails[:10]:
        print("FAIL", e, "<-", t)
    print(f"{n} malformed turns, {len(fails)} raised")
    return 1 if fails else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(2)
