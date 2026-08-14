"""Analyze the Opus-4.8 batch results against the RL worlds' gold/battery.

Produces, per archetype, the signals needed to judge whether Part A ('found the fix')
and Part B ('understood the mechanism') scored CORRECTLY:

  - health: did every episode ANSWER and avoid the turn cap? (a capped/truncated run is
    a harness artifact, not a reasoning signal — must not be read as a reward failure).
  - Part A: gold vs recommended intervention, benefit_recovered, part_a bool, gap.
  - Part B: battery_fraction, which battery items failed, the proxy/decoys Opus named.
  - resolver-artifact flag: any 'sign:<x>' battery item whose <x> is NOT a real actuator
    (free-text the resolver kept) — informational; the RL reward path uses ids, not text.
  - candidates for manual trace reading:
      * FALSE-NEG suspect: benefit < 0.9 but the final reasoning names the gold lever(s).
      * SHORTCUT suspect: recommended the symptom-trap / a decoy but benefit >= 0.5.

Run:  PYTHONPATH=rpg_v8 python3 analyze_opus_results.py [results_dir]
"""
import json, glob, os, sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ec2-user/SageMaker/vivian/ADS_shared/dataset_generation_code/results_v8_validation_opus"
MAN = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/manifest.json"


def load_manifest_index():
    man = json.load(open(MAN))
    return {m["world_id"]: m for m in man}


def actuator_names(world_json):
    scm = world_json.get("scm", {})
    acts = scm.get("actuators", {})
    return set(acts.keys())


def main():
    idx = load_manifest_index()
    files = sorted(glob.glob(os.path.join(RES, "result_*.json")))
    if not files:
        print(f"no result_*.json in {RES} yet")
        return
    rows = []
    for f in files:
        r = json.load(open(f))
        wid = r["world_id"]
        arch = idx.get(wid, {}).get("archetype", "?")
        g = r.get("grade", {})
        wfile = r.get("world_file")
        acts = set()
        gt = {}
        if wfile and os.path.exists(wfile):
            wj = json.load(open(wfile))
            acts = actuator_names(wj)
            gt = wj.get("ground_truth", {})
        # final answer turn
        turns = r.get("turns", [])
        ans_turn = next((t for t in reversed(turns) if t.get("action_type") == "answer"), None)
        reasoning = (ans_turn or {}).get("reasoning", "") if ans_turn else ""
        answer_raw = (ans_turn or {}).get("answer_raw", "")
        # resolver artifacts: battery sign items on non-actuators
        items = dict(g.get("battery_items", []))
        artifact_signs = [k for k in items
                          if k.startswith("sign:") and k[5:].split("(")[0] not in acts]
        # gold vs recommended
        gold_iv = g.get("gold_intervention", {})
        rec_iv = g.get("recommended_intervention", {})
        gold_acts = set(gold_iv.keys())
        rec_acts = set(rec_iv.keys())
        trap = gt.get("symptom_trap_actuator")
        decoys = set(gt.get("confounded_decoys", []))
        benefit = g.get("benefit_recovered")
        # false-neg suspect: low benefit but final reasoning mentions a gold lever
        names_gold = any(a.lower() in reasoning.lower() for a in gold_acts) if gold_acts else False
        fneg = (benefit is not None and benefit < 0.9 and names_gold and not g.get("part_a_utility_ok"))
        # shortcut suspect: recommended trap or a decoy actuator, yet benefit >= 0.5
        used_shortcut = (trap in rec_acts) or bool(rec_acts & decoys)
        shortcut_hi = used_shortcut and (benefit or 0) >= 0.5
        rows.append({
            "arch": arch, "wid": wid,
            "answered": r.get("answered"), "hit_cap": r.get("hit_turn_cap"),
            "iv_run": r.get("interventions_run"), "q_used": r.get("queries_used"),
            "gold_u": g.get("gold_utility"), "rec_u": g.get("recommended_utility"),
            "base_u": g.get("baseline_utility"), "benefit": benefit,
            "gap": g.get("utility_gap"), "partA": g.get("part_a_utility_ok"),
            "B": g.get("battery_fraction"), "partB": g.get("part_b_battery_ok"),
            "accepted": g.get("accepted"), "gold_iv": gold_iv, "rec_iv": rec_iv,
            "failed_items": [k for k, v in items.items() if not v],
            "artifact_signs": artifact_signs, "names_gold": names_gold,
            "fneg_suspect": fneg, "shortcut_suspect": shortcut_hi,
            "reasoning": reasoning[:400], "answer_raw": str(answer_raw)[:400],
        })

    by = defaultdict(list)
    for r in rows:
        by[r["arch"]].append(r)

    print(f"=== {len(rows)} worlds analyzed ===\n")
    print(f"{'archetype':22s} {'n':>2s} {'ans':>4s} {'cap':>3s} {'acc':>5s} {'A#':>5s} "
          f"{'benμ':>6s} {'B#':>5s} {'Bμ':>5s} {'artf':>4s} {'fneg':>4s} {'sc':>3s}")
    for a in sorted(by):
        rs = by[a]; n = len(rs)
        ans = sum(bool(x["answered"]) for x in rs)
        cap = sum(bool(x["hit_cap"]) for x in rs)
        acc = sum(bool(x["accepted"]) for x in rs)
        A = sum(bool(x["partA"]) for x in rs)
        benm = sum((x["benefit"] or 0) for x in rs) / n
        Bn = sum(bool(x["partB"]) for x in rs)
        Bm = sum((x["B"] or 0) for x in rs) / n
        artf = sum(1 for x in rs if x["artifact_signs"])
        fneg = sum(1 for x in rs if x["fneg_suspect"])
        sc = sum(1 for x in rs if x["shortcut_suspect"])
        print(f"{a:22s} {n:>2d} {ans:>3d}/{n:<1d} {cap:>3d} {acc:>2d}/{n:<2d} {A:>2d}/{n:<2d} "
              f"{benm:>6.2f} {Bn:>2d}/{n:<2d} {Bm:>5.2f} {artf:>4d} {fneg:>4d} {sc:>3d}")

    print("\n--- FALSE-NEGATIVE suspects (low benefit but reasoning names a gold lever) ---")
    for r in rows:
        if r["fneg_suspect"]:
            print(f"  {r['arch']:20s} {r['wid']}  benefit={r['benefit']} gap={r['gap']} "
                  f"gold={r['gold_iv']} rec={r['rec_iv']}")
            print(f"      reasoning: {r['reasoning'][:220]}")

    print("\n--- SHORTCUT suspects (recommended trap/decoy but benefit >= 0.5) ---")
    scs = [r for r in rows if r["shortcut_suspect"]]
    print("  none" if not scs else "")
    for r in scs:
        print(f"  {r['arch']:20s} {r['wid']}  benefit={r['benefit']} rec={r['rec_iv']}")

    print("\n--- RESOLVER-ARTIFACT battery signs (free-text kept as an actuator; RL id-path immune) ---")
    arts = [r for r in rows if r["artifact_signs"]]
    print("  none" if not arts else "")
    for r in arts:
        print(f"  {r['arch']:20s} {r['wid']}  {r['artifact_signs']}")

    json.dump(rows, open(os.path.join(RES, "analysis.json"), "w"), indent=2)
    print(f"\nwrote analysis.json ({len(rows)} worlds)")


if __name__ == "__main__":
    main()
