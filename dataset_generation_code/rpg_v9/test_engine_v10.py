#!/usr/bin/env python3
"""Unit tests for the v10 engine opt-ins (propagate_side_effects, partial expr). Run from rpg_v9/:
    python test_engine_v10.py"""
import numpy as np
from engine import WorldSCM, eval_expr


def world(propagate):
    V = {
        "cause": {"aliases": ["c"], "kind": "latent", "dist": {"normal": [50, 0]}},
        "goal": {"aliases": ["g"], "kind": "latent", "parents": ["cause"],
                 "mech": {"form": "linear", "weights": {"cause": -1.0}, "intercept": 100}},
        "out": {"aliases": ["o"], "kind": "outcome", "parents": ["goal"],
                "mech": {"form": "linear", "weights": {"goal": 1.0}, "intercept": 0}, "measurable": True},
    }
    A = {"fix": {"aliases": ["f"], "target": "cause", "op": "scale", "dtype": "continuous", "range": [0, 100],
                 "default": 0, "expr": "1-sat(d;k=0.5)",
                 "side_effect": {"target": "goal", "expr": "-overstrip(d;thr=0.5,gain=40)"}}}
    return WorldSCM(variables=V, actuators=A, outcome="out", goal="goal", propagate_side_effects=propagate)


def main():
    fails = 0
    for prop in (False, True):
        w = world(prop)
        for dose, want_goal in ((0, 50.0), (50, 100.0), (100, 80.0)):
            v = w.sample(10, {"fix": dose}, seed=1)
            g, o = float(v["goal"].mean()), float(v["out"].mean())
            # goal: cause removed fully at dose >= 50; overdose harm 40 * (d - 0.5) above 50
            want_out = want_goal if prop else (want_goal if dose <= 50 else 100.0)
            ok = abs(g - want_goal) < 1e-6 and abs(o - want_out) < 1e-6
            fails += not ok
            print(f"propagate={prop} dose={dose}: goal {g:.1f} (want {want_goal}) outcome {o:.1f} (want {want_out}) "
                  f"{'ok' if ok else 'FAIL'}")
    # serialisation round-trip
    d = world(True).to_dict(); fails += d.get("propagate_side_effects") is not True
    fails += WorldSCM.from_dict(d).propagate_side_effects is not True
    fails += "propagate_side_effects" in world(False).to_dict()
    # partial expr
    for d_, want in ((0.0, 1.0), (0.5, 0.8), (1.0, 0.6)):
        got = eval_expr("partial(d;p=0.4)", d_); ok = abs(got - want) < 1e-9; fails += not ok
        print(f"partial(p=0.4) at d={d_}: {got:.3f} (want {want}) {'ok' if ok else 'FAIL'}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
