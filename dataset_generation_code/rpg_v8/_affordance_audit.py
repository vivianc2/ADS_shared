"""Across the 27B run: how much of the partA failure is free-text actuator-discovery
failure (harness artifact) vs genuine reasoning failure?

For each world count: intervene attempts, how many APPLIED (non-empty), the per-request
reject rate, and whether the FINAL recommended_intervention resolved to non-empty.
"""
import json, glob
from collections import Counter

rdir = "/work/results_v9_validation_27b"
man = json.load(open("/work/data/rpg_v9_val_worlds/manifest.json"))
arch_of = {w["world_id"]: w["archetype"] for w in man["worlds"]}

n = 0
worlds_zero_applied = 0          # never landed a single real intervention all episode
worlds_empty_final = 0           # final answer resolved to {} (no actuator)
req_total = req_rejected = 0     # per intervention-request resolution outcomes
by_arch_empty_final = Counter(); by_arch_n = Counter()
partA_true_but = 0

for f in glob.glob(f"{rdir}/result_*.json"):
    if "summary" in f:
        continue
    d = json.load(open(f))
    n += 1
    wid = d["world_id"]; a = arch_of.get(wid, "?")
    by_arch_n[a] += 1
    applied_any = False
    for t in d["turns"]:
        r = t.get("result")
        if isinstance(r, dict) and r.get("mode") == "intervene":
            if r.get("applied_intervention"):
                applied_any = True
            for ar in r.get("action_resolutions", []):
                req_total += 1
                if not ar.get("ok"):
                    req_rejected += 1
    if not applied_any:
        worlds_zero_applied += 1
    g = d.get("grade") or {}
    if not (g.get("recommended_intervention")):
        worlds_empty_final += 1
        by_arch_empty_final[a] += 1
    if g.get("part_a_utility_ok") and not g.get("recommended_intervention"):
        partA_true_but += 1

print(f"worlds: {n}")
print(f"intervention REQUESTS: {req_total} total, {req_rejected} rejected "
      f"({100*req_rejected/max(1,req_total):.0f}% reject rate)")
print(f"worlds that NEVER landed a single applied intervention: {worlds_zero_applied}/{n} "
      f"({100*worlds_zero_applied/n:.0f}%)")
print(f"worlds whose FINAL recommended_intervention resolved to EMPTY: {worlds_empty_final}/{n} "
      f"({100*worlds_empty_final/n:.0f}%)")
print("\nempty-final-recommendation rate by archetype:")
for a in sorted(by_arch_n):
    print(f"  {a:20s} {by_arch_empty_final[a]:>3}/{by_arch_n[a]:<3} "
          f"({100*by_arch_empty_final[a]/by_arch_n[a]:.0f}%)")
