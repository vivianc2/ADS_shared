"""Reconstruct per-query utility trajectories from a batch run's traces.

For every INTERVENE the agent executed, recompute the TRUE expected utility of that
intervention on the live SCM (regenerated from the world's seed), and express it as
benefit_recovered = (u - baseline) / (gold - baseline)  in [.,1], gold = 1.0.

x-axis = query index (experiment_id = the budget counter; measures and interventions
share it). We report, per world and query index k = 1..K:
  - best-so-far benefit (the search-efficiency envelope: best result observed by k)
This is the "how fast does the model climb toward gold" curve. Aggregates (mean, p25,
p75) are computed per archetype and overall, with carry-forward for finished episodes.

Output: results_dir/trajectories.json  (per-world points + best curves + aggregates),
consumed by build_trajectory_chart.py. Model-agnostic — run once per model's results dir.

Run:  PYTHONPATH=rpg_v8 python3 build_utility_trajectories.py <results_dir> <model_label>
"""
import json, glob, os, sys
from collections import defaultdict
from sampler import sample_world
from generate_v7 import audit
from oracle_v6 import expected_utility

RES = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ec2-user/SageMaker/vivian/ADS_shared/dataset_generation_code/results_v8_validation_opus"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "opus-4.8"
MAN = "/home/ec2-user/SageMaker/vivian/data/rpg_v8_fast_worlds/manifest.json"
KMAX = 15                     # budget
N_MC = 6000                   # MC samples for utility recompute (speed/precision trade-off)


def pctl(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(i)
    if lo == len(xs) - 1:
        return xs[lo]
    return xs[lo] + (xs[lo + 1] - xs[lo]) * (i - lo)


def main():
    man_path = os.path.join(RES, "manifest.json") if os.path.exists(os.path.join(RES, "manifest.json")) else MAN
    man = {m["world_id"]: m for m in json.load(open(man_path))}
    files = sorted(glob.glob(os.path.join(RES, "result_*.json")))
    per_world = []
    for f in files:
        r = json.load(open(f))
        wid = r["world_id"]
        t = man.get(wid)
        if not t:
            continue
        w = sample_world(t["seed"], skin=t["skin"], archetype=t["archetype"])
        res = audit(w)
        gold, scm = res["gold"], w["scm"]
        base, gu = gold["baseline_utility"], gold["expected_utility"]
        denom = (gu - base) if abs(gu - base) > 1e-9 else 1.0

        pts = []
        for tn in r["turns"]:
            if tn.get("action_type") != "intervene":
                continue
            rr = tn.get("result", {})
            iv = {k: v for k, v in (rr.get("applied_intervention", {}) or {}).items()
                  if k in scm.actuators}
            q = rr.get("experiment_id")
            if not iv or q is None:
                continue
            u = expected_utility(scm, iv, n=N_MC, seed=999)
            pts.append({"q": int(q), "u": round(u, 3),
                        "benefit": round((u - base) / denom, 4), "iv": iv})

        # best-so-far benefit curve over query index 1..KMAX (0 before first intervention)
        best = 0.0
        curve = []
        pi = 0
        pts_sorted = sorted(pts, key=lambda p: p["q"])
        for k in range(1, KMAX + 1):
            while pi < len(pts_sorted) and pts_sorted[pi]["q"] <= k:
                best = max(best, pts_sorted[pi]["benefit"])
                pi += 1
            curve.append(round(best, 4))
        per_world.append({
            "wid": wid, "arch": t["archetype"], "skin": t["skin"],
            "base_u": round(base, 3), "gold_u": round(gu, 3),
            "points": pts_sorted, "best_curve": curve,
            "final_benefit": (r.get("grade", {}) or {}).get("benefit_recovered"),
            "queries_used": r.get("queries_used"),
        })

    # aggregate: mean/p25/p75 of best-so-far benefit at each query index
    def agg(rows):
        out = []
        for k in range(KMAX):
            vals = [w["best_curve"][k] for w in rows]
            out.append({"q": k + 1, "mean": round(sum(vals) / len(vals), 4),
                        "p25": round(pctl(vals, 0.25), 4), "p75": round(pctl(vals, 0.75), 4),
                        "n": len(vals)})
        return out

    by_arch = defaultdict(list)
    for w in per_world:
        by_arch[w["arch"]].append(w)

    result = {
        "model": LABEL, "kmax": KMAX, "n_worlds": len(per_world),
        "overall": agg(per_world),
        "by_arch": {a: agg(rows) for a, rows in by_arch.items()},
        "per_world": per_world,
    }
    out = os.path.join(RES, "trajectories.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"{LABEL}: {len(per_world)} worlds -> {out}")
    print("overall best-so-far benefit by query:")
    print("  q:    " + " ".join(f"{d['q']:>5d}" for d in result["overall"]))
    print("  mean: " + " ".join(f"{d['mean']:>5.2f}" for d in result["overall"]))


if __name__ == "__main__":
    main()
