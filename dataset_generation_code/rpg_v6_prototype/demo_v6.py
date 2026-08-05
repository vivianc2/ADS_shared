#!/usr/bin/env python3
"""RPG v6 milestone demo (no LLM).

Proves, on the large open-scenario bioreactor world:
  1. faithful SCM over ~20 variables / ~14 actuators;
  2. actuators-only intervention (no do() by fiat);
  3. combinations matter (best answer may pair remove-source + clear-contaminant);
  4. distractors are provably inert (audit) so brute force is wasteful;
  5. the world is solvable within budget by a reasoning-driven expert.

Run:  python demo_v6.py
"""

from __future__ import annotations

import json
import numpy as np

from worlds_v6 import ALL_WORLDS_V6
from oracle_v6 import (audit_world, calibrate, optimal_intervention,
                       counterfactual_battery, grade, expected_utility, screen_actuators)


def hr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def main():
    world = ALL_WORLDS_V6["bioreactor_titer_loss_v6"]()
    scm = world["scm"]
    gt = world["ground_truth"]

    hr("WORLD SIZE")
    print(f"variables: {len(scm.variables)}  (measurable: {len(scm.measurable_vars())}, "
          f"hidden: {len([v for v in scm.variables.values() if not (v['kind'] in ('observable','outcome') or v.get('measurable'))])})")
    print(f"actuators: {len(scm.actuators)}")
    print(f"outcome: {scm.outcome} (higher_is_better={scm.higher_is_better})")
    print("hidden chain root:", gt["true_root"], "-> ... ->", scm.outcome)

    hr("CALIBRATION + AUDITS")
    calib = calibrate(world)
    print("calibration:", calib)
    gold = optimal_intervention(scm)
    print("\nGOLD intervention:", gold["intervention"])
    print(f"  baseline util {gold['baseline_utility']:.1f} -> gold util {gold['expected_utility']:.1f}")
    print("  active actuators (survived screening):", gold["active_actuators"])
    from oracle_v6 import (distractor_inertness_audit, decoy_audit,
                           proxy_signal_audit, gold_selfconsistency_audit)
    di = distractor_inertness_audit(world, gold)
    print(f"\ndistractor inertness: passed={di['passed']} "
          f"({di['n_inert_checked']} inert actuators checked, {len(di['violations'])} violations)")
    dc = decoy_audit(world); print("decoy audit:", dc)
    ps = proxy_signal_audit(world); print("proxy signal:", ps)
    gs = gold_selfconsistency_audit(world, gold); print("gold self-consistency:", gs)

    hr("BRUTE FORCE IS INFEASIBLE")
    n_act = len(scm.actuators)
    import math
    combos = sum(math.comb(n_act, r) for r in (1, 2, 3))
    print(f"{n_act} actuators, joint up to 3 -> {combos} actuator SUBSETS (before dosing grids).")
    print("With a ~15-experiment budget, the agent cannot sweep them all;")
    print("it must reason from the scenario about WHERE to look.")

    hr("SCRIPTED EXPERT TRAJECTORY (reasoning-driven, actuators-only)")
    proxy, decoy = gt["true_mechanism_proxy"], gt["confounded_decoys"][0]
    do_ctrl = "do_controller"; chel = gt["targeted_actuator"]; flow = "feed_flow_controller"
    trap = gt["symptom_trap_actuator"]
    q = 0

    def measure(names, iv=None, tag=""):
        nonlocal q; q += 1
        v = scm.sample(400, intervention=iv or {}, seed=100 + q)
        o = scm.measure(v, names, seed=500 + q, intervention=iv or {})
        print(f"  [q{q}] measure {names} {('under '+str(iv)) if iv else '(observational)'} {tag}")
        print("        ", {k: round(float(np.mean(val)), 2) for k, val in o.items()})
        return {k: float(np.mean(val)) for k, val in o.items()}

    print("\n1) Observe: titer, the O2 the operators blame, and the cloudiness clue.")
    measure([scm.outcome, decoy, proxy])
    print("   -> O2 tracks titer (tempting) but broth turbidity is high & tracks low titer;")
    print("      an oxygen-starvation story does not explain cell lysis.")

    print("\n2) Break the confound: use the DO controller to HOLD oxygen high vs low.")
    hi = measure([scm.outcome], {do_ctrl: 80}, "(force O2 high)")
    lo = measure([scm.outcome], {do_ctrl: 20}, "(force O2 low)")
    print(f"   -> titer barely moves ({hi[scm.outcome]-lo[scm.outcome]:+.2f}) => O2 is a bystander (confounded by seed age).")

    print("\n3) Follow the real clue (cloudy broth + replaced fitting -> feed-borne toxin).")
    print("   Sweep the feed-water flow controller and watch titer + turbidity.")
    for f in [0, 40, 100]:
        measure([scm.outcome, proxy], {flow: f})
    print("   -> higher flow => lower titer, higher turbidity (sign flip): the FEED is the source.")

    print("\n4) Decisive test + dose: a chelating additive should mop up a feed-borne metal.")
    best_d, best_u = None, -1e9
    for d in [0, 33, 66, 100]:
        o = measure([scm.outcome, proxy], {chel: d})
        if o[scm.outcome] > best_u:
            best_u, best_d = o[scm.outcome], d
    print(f"   -> titer recovers & turbidity falls; coarse peak near {best_d}. Refine around it:")
    for d in [55, 62, 70]:
        measure([scm.outcome], {chel: d})
    print("   -> interior optimum ~66 (over-dose strips a nutrient).")

    print("\n5) Reject the trap: a 'stabilizer' additive lifts the READING only.")
    measure([scm.outcome, proxy], {trap: 100})
    print("   -> titer reading rises but turbidity unchanged => cosmetic, not a fix.")

    print("\n6) Combination check: remove source AND clear existing contaminant.")
    print(f"   expected utility of combos (n=30000):")
    for iv, name in [({chel: 66}, "chelator@66"),
                     ({flow: 0}, "flow@0"),
                     ({chel: 66, flow: 0}, "chelator@66 + flow@0")]:
        print(f"     {name:28s}: {expected_utility(scm, iv, n=30000, seed=999):.2f}")

    print(f"\n  expert used {q} experiments (budget ~15).")

    hr("GRADING: correct vs surface-proxy answer")
    battery = counterfactual_battery(world)
    correct = {
        "recommended_intervention": gold["intervention"],
        "structured": {"true_mechanism_proxy": proxy,
                       "confounded_decoys": [decoy],
                       "actuator_sign_predictions": battery["actuator_sign_predictions"]},
        "explanation": gt["latent_plain_name"],
    }
    surface = {
        "recommended_intervention": {do_ctrl: 90},
        "structured": {"true_mechanism_proxy": decoy, "confounded_decoys": [],
                       "actuator_sign_predictions": {do_ctrl: "+", chel: "0"}},
        "explanation": "dissolved-oxygen control drift",
    }
    print("\nCORRECT :", json.dumps(grade(world, correct, gold, battery), default=str))
    print("\nSURFACE :", json.dumps(grade(world, surface, gold, battery), default=str))


if __name__ == "__main__":
    main()
