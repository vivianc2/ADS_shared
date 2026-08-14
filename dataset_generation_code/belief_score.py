"""Score reconstructed belief snapshots against the true SCM, per turn.

For each snapshot we score the agent's causal belief vs ground truth:
  - cause_ok   : believed-fix actuator is the true targeted actuator (or a co-actuator
                 in two_cause worlds).
  - proxy_ok   : believed mechanism marker is in the world's valid_mechanism_proxies.
  - decoy_f1   : F1 of flagged decoys vs the true confounded_decoys.
  - trap_ok    : did NOT name the symptom-trap lever as the cause; did NOT label the
                 true proxy/true fix as a decoy (the characteristic trap errors).
  - graph_score: mean(cause_ok, proxy_ok, decoy_f1)  -- the structural-accuracy summary.
Also diffs consecutive snapshots into symbolic EDITS (cause/proxy/decoy/sign changes).

Aggregates graph_score and its components by ACTION-TURN ORDINAL (1st action, 2nd, ...)
with carry-forward, per archetype and overall.

Output: <results_dir>/beliefs_scored.json  (per-world scored snapshots + edits + aggregates).
Run:  PYTHONPATH=rpg_v8 python3 belief_score.py <results_dir>
"""
import json, glob, os, sys
from collections import defaultdict
from sampler import sample_world
from generate_v7 import audit

MAN = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/manifest.json"
KMAX = 20   # action-turn ordinals to align on


def f1(pred, gold):
    pred, gold = set(pred or []), set(gold or [])
    if not gold and not pred:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo = int(i)
    return xs[lo] if lo == len(xs) - 1 else xs[lo] + (xs[lo + 1] - xs[lo]) * (i - lo)


def score_world(bel, gt, battery):
    tgt = gt["targeted_actuator"]
    co = set(gt.get("co_actuators") or [])
    causes_ok = {tgt} | co
    valid_prox = set(battery.get("valid_mechanism_proxies", [battery["true_mechanism_proxy"]]))
    gt_decoys = set(battery["confounded_decoys"])
    trap = gt.get("symptom_trap_actuator")
    true_prox = battery["true_mechanism_proxy"]
    scored = []
    for s in bel["snapshots"]:
        cause, proxy = s.get("cause"), s.get("proxy")
        decoys = s.get("decoys") or []
        cause_ok = 1.0 if cause in causes_ok else 0.0
        proxy_ok = 1.0 if proxy in valid_prox else 0.0
        df1 = f1(decoys, gt_decoys)
        trap_err = (cause == trap) or (true_prox in decoys) or (tgt in decoys)
        scored.append({
            "turn": s.get("turn"), "cause": cause, "proxy": proxy, "decoys": decoys,
            "signs": s.get("signs", {}), "ruled_out": s.get("ruled_out", []),
            "cause_ok": cause_ok, "proxy_ok": proxy_ok, "decoy_f1": round(df1, 3),
            "trap_ok": 0.0 if trap_err else 1.0,
            "graph_score": round((cause_ok + proxy_ok + df1) / 3, 3),
        })
    # symbolic edits between consecutive snapshots
    edits = []
    for a, b in zip(scored, scored[1:]):
        e = []
        if a["cause"] != b["cause"]:
            e.append(f"cause:{a['cause']}→{b['cause']}")
        if a["proxy"] != b["proxy"]:
            e.append(f"proxy:{a['proxy']}→{b['proxy']}")
        add = set(b["decoys"]) - set(a["decoys"])
        rem = set(a["decoys"]) - set(b["decoys"])
        for d in add:
            e.append(f"+decoy:{d}")
        for d in rem:
            e.append(f"-decoy:{d}")
        if a["signs"] != b["signs"]:
            e.append("signΔ")
        edits.append({"turn": b["turn"], "edits": e})
    return scored, edits


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/ec2-user/SageMaker/vivian/ADS_shared/dataset_generation_code/results_v8_validation_opus"
    man_path = os.path.join(rd, "manifest.json") if os.path.exists(os.path.join(rd, "manifest.json")) else MAN
    man = {m["world_id"]: m for m in json.load(open(man_path))}
    bfiles = sorted(glob.glob(os.path.join(rd, "beliefs", "*.json")))
    per_world = []
    for bf in bfiles:
        bel = json.load(open(bf))
        t = man.get(bel["world_id"])
        if not t:
            continue
        w = sample_world(t["seed"], skin=t["skin"], archetype=t["archetype"])
        res = audit(w)
        scored, edits = score_world(bel, w["ground_truth"], res["battery"])
        # align to action-turn ordinal 1..KMAX (carry forward), for aggregate
        curve = {"graph": [], "cause": [], "proxy": [], "decoy": [], "trap": []}
        last = {"graph": 0.0, "cause": 0.0, "proxy": 0.0, "decoy": 0.0, "trap": 1.0}
        for k in range(KMAX):
            if k < len(scored):
                s = scored[k]
                last = {"graph": s["graph_score"], "cause": s["cause_ok"], "proxy": s["proxy_ok"],
                        "decoy": s["decoy_f1"], "trap": s["trap_ok"]}
            curve["graph"].append(last["graph"]); curve["cause"].append(last["cause"])
            curve["proxy"].append(last["proxy"]); curve["decoy"].append(last["decoy"]); curve["trap"].append(last["trap"])
        per_world.append({"world_id": bel["world_id"], "archetype": bel["archetype"],
                          "actuators": bel["actuators"], "measurables": bel["measurables"],
                          "gt": {"cause": w["ground_truth"]["targeted_actuator"],
                                 "proxy": res["battery"]["true_mechanism_proxy"],
                                 "decoys": list(res["battery"]["confounded_decoys"]),
                                 "trap": w["ground_truth"].get("symptom_trap_actuator"),
                                 "root": w["ground_truth"].get("true_root")},
                          "snapshots": scored, "edits": edits, "curve": curve,
                          "n_actions": len(scored)})

    def agg(rows, comp):
        out = []
        for k in range(KMAX):
            vals = [w["curve"][comp][k] for w in rows]
            out.append({"q": k + 1, "mean": round(sum(vals) / len(vals), 4),
                        "p25": round(pctl(vals, .25), 4), "p75": round(pctl(vals, .75), 4)})
        return out

    by = defaultdict(list)
    for w in per_world:
        by[w["archetype"]].append(w)
    result = {
        "model": "opus-4.8", "kmax": KMAX, "n_worlds": len(per_world),
        "overall": {c: agg(per_world, c) for c in ("graph", "cause", "proxy", "decoy", "trap")},
        "by_arch": {a: {c: agg(rows, c) for c in ("graph", "cause", "proxy", "decoy", "trap")}
                    for a, rows in by.items()},
        "per_world": per_world,
    }
    out = os.path.join(rd, "beliefs_scored.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"scored {len(per_world)} worlds -> {out}")
    gc = result["overall"]["graph"]
    print("overall graph_score by action-turn ordinal:")
    print("  q:    " + " ".join(f"{d['q']:>4d}" for d in gc[:15]))
    print("  mean: " + " ".join(f"{d['mean']:>4.2f}" for d in gc[:15]))
    for c in ("cause", "proxy", "decoy"):
        fin = result["overall"][c][-1]["mean"]
        print(f"  final {c}_ok mean = {fin:.2f}")


if __name__ == "__main__":
    main()
