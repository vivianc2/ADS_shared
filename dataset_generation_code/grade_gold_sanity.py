"""Design validation, Opus-independent: grade the GOLD answer and an EMPTY answer
against each of the exact RL worlds in the sampled set.

Expectation (if the reward/oracle are correctly designed on the live training data):
  - GOLD  -> part_a True, benefit_recovered ~1, battery_fraction 1.0, part_b True.
  - EMPTY -> part_a False, benefit ~0, battery_fraction low, part_b False.

Any RL world whose own gold does NOT ace its own grader is a reward/oracle bug on the
data the policy is actually trained on. Prints per-archetype worst cases.

Run:  PYTHONPATH=rpg_v8:rpg_rl python3 grade_gold_sanity.py
"""
import json
from collections import defaultdict
from sampler import sample_world
from generate_v7 import audit
from oracle_v6 import grade

MAN = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/manifest.json"


def gold_answer(world, gold, battery):
    """Canonical-name gold answer (bypasses the free-text resolver)."""
    gt = world["ground_truth"]
    # recommended intervention: scalar entries only; a conditional policy is passed
    # separately via recommended_policy (dict-valued entries are the policy form).
    rec = {k: v for k, v in gold["intervention"].items() if not isinstance(v, dict)}
    signs = {}
    for aid in gold.get("active_actuators", []):
        s = battery["actuator_sign_predictions"].get(aid, "0")
        if s in ("+", "-"):
            signs[aid] = s
    ans = {"recommended_intervention": rec,
           "structured": {"true_mechanism_proxy": battery["true_mechanism_proxy"],
                          "confounded_decoys": list(battery["confounded_decoys"]),
                          "actuator_sign_predictions": signs}}
    if gold.get("is_conditional_policy") and gold.get("policy"):
        sp = gt["subtype_policy"]
        gp = gold["policy"]
        ans["recommended_policy"] = {"treatment": sp["treatment_actuator"],
                                     "stratifier": sp["marker"],
                                     "threshold": gp["threshold"],
                                     "dose_if_ge": gp["dose_if_ge"],
                                     "dose_if_lt": gp["dose_if_lt"]}
    return ans


def main():
    man = json.load(open(MAN))
    rows = []
    for m in man:
        w = sample_world(m["seed"], skin=m["skin"], archetype=m["archetype"])
        res = audit(w)
        gold, battery = res["gold"], res["battery"]
        gA = grade(w, gold_answer(w, gold, battery), gold, battery, strict=True)
        eA = grade(w, {"recommended_intervention": {}, "structured": {}}, gold, battery, strict=True)
        rows.append({"arch": m["archetype"], "skin": m["skin"], "wid": w["world_id"],
                     "gold_benefit": gA.get("benefit_recovered"),
                     "gold_partA": gA["part_a_utility_ok"], "gold_B": gA["battery_fraction"],
                     "gold_partB": gA["part_b_battery_ok"], "gold_accepted": gA["accepted"],
                     "gold_items": dict(gA["battery_items"]),
                     "empty_benefit": eA.get("benefit_recovered"), "empty_B": eA["battery_fraction"],
                     "empty_accepted": eA["accepted"],
                     "is_policy": gA["is_conditional_policy"]})

    by = defaultdict(list)
    for r in rows:
        by[r["arch"]].append(r)

    print(f"{'archetype':22s} {'n':>2s} {'gold_acc':>8s} {'gold_A':>6s} {'goldBenμ':>8s} "
          f"{'gold_B':>6s} {'empty_acc':>9s} {'emptyBμ':>7s}")
    all_gold_ok = True
    for a in sorted(by):
        rs = by[a]
        n = len(rs)
        gacc = sum(r["gold_accepted"] for r in rs)
        gA = sum(r["gold_partA"] for r in rs)
        gben = sum((r["gold_benefit"] or 0) for r in rs) / n
        gB = sum(r["gold_B"] for r in rs) / n
        eacc = sum(r["empty_accepted"] for r in rs)
        eB = sum(r["empty_B"] for r in rs) / n
        print(f"{a:22s} {n:>2d} {gacc:>4d}/{n:<3d} {gA:>3d}/{n:<2d} {gben:>8.3f} "
              f"{gB:>6.2f} {eacc:>5d}/{n:<3d} {eB:>7.2f}")
        if gacc < n:
            all_gold_ok = False

    # Flag any world where gold fails to ace its own grader.
    print("\n--- GOLD self-grade failures (bug candidates) ---")
    fails = [r for r in rows if not r["gold_accepted"]]
    if not fails:
        print("  none — every gold answer is accepted (A and B) on its own world.")
    for r in fails:
        bad_items = [k for k, v in r["gold_items"].items() if not v]
        print(f"  {r['arch']:20s} {r['wid']}  A={r['gold_partA']}(ben={r['gold_benefit']}) "
              f"B={r['gold_B']:.2f} failed_items={bad_items} policy={r['is_policy']}")

    # Flag any empty answer that was accepted (master-key leak).
    leaks = [r for r in rows if r["empty_accepted"] or (r["empty_benefit"] or 0) >= 0.9]
    print("\n--- EMPTY-answer leaks (should be none) ---")
    print("  none" if not leaks else "")
    for r in leaks:
        print(f"  {r['arch']:20s} {r['wid']}  empty_benefit={r['empty_benefit']} empty_B={r['empty_B']:.2f}")

    json.dump(rows, open("/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/gold_sanity.json", "w"), indent=2)
    print(f"\nwrote gold_sanity.json ({len(rows)} worlds); all_gold_accepted={all_gold_ok}")


if __name__ == "__main__":
    main()
