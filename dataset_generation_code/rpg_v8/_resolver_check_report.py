"""Compare the fixed-resolver run vs the original (lexical-only) run on the same worlds:
method usage, intervention reject rate, and per-world partA/benefit deltas."""
import json, glob, os
from collections import Counter

NEW = "/home/ec2-user/SageMaker/vivian/results_v9_resolver_check"
OLD = "/home/ec2-user/SageMaker/vivian/results_v9_validation_27b"


def scan(rdir):
    methods = Counter(); req_tot = req_rej = 0
    grades = {}
    for f in glob.glob(f"{rdir}/result_*.json"):
        d = json.load(open(f)); wid = d["world_id"]
        for t in d["turns"]:
            r = t.get("result")
            if isinstance(r, dict):
                for key in ("resolutions", "action_resolutions", "measurement_resolutions"):
                    for res in r.get(key, []):
                        methods[res.get("method", "?")] += 1
                for ar in (r.get("action_resolutions", []) if isinstance(r, dict) else []):
                    req_tot += 1; req_rej += 0 if ar.get("ok") else 1
        g = d.get("grade") or {}
        grades[wid] = (g.get("part_a_utility_ok"), g.get("benefit_recovered"),
                       bool(g.get("recommended_intervention")), g.get("battery_fraction"))
    return methods, req_tot, req_rej, grades


nm, nt, nr, ng = scan(NEW)
om, ot, orj, og = scan(OLD)
print("FIXED run (Opus resolver): methods=%s" % dict(nm))
print("  intervention reject rate: %d/%d (%.0f%%)" % (nr, nt, 100*nr/max(1, nt)))
print("ORIGINAL run (lexical only) on same worlds:")
oldsub = {w: og[w] for w in ng if w in og}
print("  (for reference) full-run methods were alias-only\n")
print("%-46s  %-22s  %-22s" % ("world", "OLD (partA,benefit,resolved)", "NEW (partA,benefit,resolved)"))
lift = []
for w in sorted(ng):
    o = og.get(w, ("?", "?", "?", "?")); n = ng[w]
    print("%-46s  %-22s  %-22s" % (w[:46], str(o[:3]), str(n[:3])))
    if isinstance(o[1], (int, float)) and isinstance(n[1], (int, float)):
        lift.append(n[1] - o[1])
if lift:
    print("\nmean benefit delta (NEW-OLD) over %d worlds: %+.3f" % (len(lift), sum(lift)/len(lift)))
