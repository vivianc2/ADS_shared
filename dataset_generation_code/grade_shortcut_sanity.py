"""Adversarial design validation: does the reward CREDIT the archetype shortcut?

For each RL world we build 'shortcut' answers a lazy/confounded policy would give and
confirm the reward does NOT accept them:

  TRAP-FIX   : recommend the symptom_trap_actuator (a mask lever that moves the
               proxy/symptom but has no true utility effect) at max dose. Part A must
               stay ~0 (treating the symptom is not the fix).
  PROXY-SWAP : name a confounded decoy as the true_mechanism_proxy and label the true
               proxy as a decoy. Part B proxy + decoy items must fail.
  DECOY-FIX  : recommend a confounded-decoy actuator (if the decoy names an actuator)
               at max dose. Part A must stay ~0.

A shortcut that scores high (accepted, or benefit>=0.5, or B>=0.8) is a reward-hack
surface — exactly what an RL policy would exploit. Prints any such leak.

Run:  PYTHONPATH=rpg_v8:rpg_rl python3 grade_shortcut_sanity.py
"""
import json
from collections import defaultdict
from sampler import sample_world
from generate_v7 import audit
from oracle_v6 import grade

MAN = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/manifest.json"


def _max_dose(scm, aid):
    a = scm.actuators[aid]
    if a.get("dtype") == "continuous":
        return float(a["range"][1])
    vals = a.get("values", ["off", "on"])
    return vals[-1]


def main():
    man = json.load(open(MAN))
    rows = []
    for m in man:
        w = sample_world(m["seed"], skin=m["skin"], archetype=m["archetype"])
        res = audit(w)
        gold, battery = res["gold"], res["battery"]
        scm = w["scm"]
        gt = w["ground_truth"]
        trap = gt.get("symptom_trap_actuator")
        proxy = battery["true_mechanism_proxy"]
        decoys = list(battery["confounded_decoys"])

        # TRAP-FIX: recommend the symptom-trap lever at max dose.
        trap_ans = {"recommended_intervention": {}, "structured": {}}
        if trap in scm.actuators:
            trap_ans["recommended_intervention"] = {trap: _max_dose(scm, trap)}
        gTrap = grade(w, trap_ans, gold, battery, strict=True)

        # PROXY-SWAP: decoy named as the mechanism proxy; true proxy labeled a decoy.
        swap_ans = {"recommended_intervention": {},
                    "structured": {"true_mechanism_proxy": (decoys[0] if decoys else "unknown"),
                                   "confounded_decoys": [proxy]}}
        gSwap = grade(w, swap_ans, gold, battery, strict=True)

        # DECOY-FIX: if a confounded decoy is itself an actuator, recommend it.
        decoy_act = next((d for d in decoys if d in scm.actuators), None)
        gDecoy = None
        if decoy_act:
            gDecoy = grade(w, {"recommended_intervention": {decoy_act: _max_dose(scm, decoy_act)},
                               "structured": {}}, gold, battery, strict=True)

        rows.append({"arch": m["archetype"], "wid": w["world_id"], "trap": trap,
                     "trap_in_scm": trap in scm.actuators,
                     "trap_benefit": gTrap.get("benefit_recovered"), "trap_A": gTrap["part_a_utility_ok"],
                     "trap_B": gTrap["battery_fraction"], "trap_acc": gTrap["accepted"],
                     "swap_B": gSwap["battery_fraction"], "swap_partB": gSwap["part_b_battery_ok"],
                     "swap_items": dict(gSwap["battery_items"]),
                     "decoy_act": decoy_act,
                     "decoy_benefit": (gDecoy or {}).get("benefit_recovered") if gDecoy else None,
                     "decoy_A": gDecoy["part_a_utility_ok"] if gDecoy else None})

    by = defaultdict(list)
    for r in rows:
        by[r["arch"]].append(r)

    print(f"{'archetype':22s} {'n':>2s} {'trap_in':>7s} {'trapBenμ':>8s} {'trapA#':>6s} "
          f"{'trapB μ':>7s} {'trapAcc':>7s} {'swapB μ':>7s} {'swapB#':>6s}")
    for a in sorted(by):
        rs = by[a]; n = len(rs)
        tin = sum(r["trap_in_scm"] for r in rs)
        tben = sum((r["trap_benefit"] or 0) for r in rs) / n
        tA = sum(bool(r["trap_A"]) for r in rs)
        tB = sum(r["trap_B"] for r in rs) / n
        tacc = sum(r["trap_acc"] for r in rs)
        sB = sum(r["swap_B"] for r in rs) / n
        sBok = sum(bool(r["swap_partB"]) for r in rs)
        print(f"{a:22s} {n:>2d} {tin:>4d}/{n:<2d} {tben:>8.3f} {tA:>4d}/{n:<1d} "
              f"{tB:>7.2f} {tacc:>4d}/{n:<2d} {sB:>7.2f} {sBok:>4d}/{n:<1d}")

    print("\n--- SHORTCUT LEAKS (reward credits a shortcut — should be none) ---")
    leaks = []
    for r in rows:
        if r["trap_acc"] or (r["trap_benefit"] or 0) >= 0.5:
            leaks.append(("TRAP-FIX", r, f"benefit={r['trap_benefit']} A={r['trap_A']}"))
        if r["swap_partB"]:
            leaks.append(("PROXY-SWAP", r, f"B={r['swap_B']:.2f} items={r['swap_items']}"))
        if r["decoy_A"]:
            leaks.append(("DECOY-FIX", r, f"benefit={r['decoy_benefit']} act={r['decoy_act']}"))
    if not leaks:
        print("  none — no shortcut answer is accepted, and no shortcut recovers >=50% benefit or passes part B.")
    for kind, r, detail in leaks:
        print(f"  [{kind}] {r['arch']:20s} {r['wid']}  {detail}")

    json.dump(rows, open("/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/shortcut_sanity.json", "w"), indent=2)
    print(f"\nwrote shortcut_sanity.json ({len(rows)} worlds); leaks={len(leaks)}")


if __name__ == "__main__":
    main()
