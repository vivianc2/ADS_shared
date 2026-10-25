#!/usr/bin/env python3
"""Compare candidate reward formulas on (1) the synthetic audit answers, (2) the scripted baselines (test), (3) the
base-9B eval episodes (validation, n=4). Inputs from reward_audit.py and analyze_eval.py. 2026-10-23."""
import csv, glob, json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
K = 4
def clip(b): return 0.0 if b in (None, "", "None") else max(0.0, min(1.0, float(b)))
V = {
  "R0 acc+0.5ben":            lambda acc, pa, ben, n: acc + 0.5 * clip(ben),
  "R1 acc+0.5passA":          lambda acc, pa, ben, n: acc + 0.5 * pa,
  "R2 acc+0.5ben*min(1,4/n)": lambda acc, pa, ben, n: acc + 0.5 * clip(ben) * min(1.0, K / max(n, 1)),
  "R3 acc+0.25passA+0.25ben*min(1,4/n)": lambda acc, pa, ben, n: acc + 0.25 * pa + 0.25 * clip(ben) * min(1.0, K / max(n, 1)),
}
def table(title, groups):
    print(f"\n{title}\n{'':30s}" + "".join(f"{k.split()[0]:>7s}" for k in V) + "   n_rec")
    for g, R in groups.items():
        print(f"{g:30s}" + "".join(f"{np.mean([f(*r) for r in R]):7.3f}" for f in V.values()) + f"  {np.mean([r[3] for r in R]):6.1f}")
print("formulas:", list(V))
syn = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.expanduser("~/rpgrl/rl_v10/reward_audit/synthetic_validation.csv"))):
    syn[r["answer"]].append((float(r["accepted_core"]), float(r["part_a"]), r["benefit"], int(r["n_rec"])))
table("synthetic answers (validation 270)", syn)
B = os.path.expanduser("~/ADS_folder/personal_docs/camera_ready/analysis_rpg/baselines")
sc = collections.defaultdict(list)
for fn, key in (("baselines_episodes.csv", lambda r: f"{r['policy']}@{r['budget']}"), ("cd1_episodes_nocd.csv", lambda r: r["policy"])):
    for r in csv.DictReader(open(os.path.join(B, fn))):
        k = key(r)
        if fn.startswith("baselines") and k not in ("random@15", "informed@15", "oracle_levers@15", "blind_grid@120", "blind_group@15"): continue
        sc[k].append((float(r["accepted"]) if r.get("accepted") not in (None, "") else 0.0, float(r["part_a"]), r["benefit_recovered"], int(r["n_rec"] or 0)))
table("scripts (test 540; acc = release accepted where given)", sc)
sys.path.insert(0, HERE); os.environ.setdefault("RLV10_HARDEN_SANDBOX", "0")
from reward_v10 import reward_components
import gzip
ep = collections.defaultdict(list)
for f in glob.glob(os.path.expanduser("~/rpgrl/rl_v10/evals/base9b_full/validation_s*/shard_*/result_*.json")):
    r = json.load(open(f)); w = json.load(gzip.open(os.path.realpath(r["world_file"])))
    c = reward_components(r, {"ground_truth": w["ground_truth"]}, {})
    ep["base 9B (validation x4)"].append((c["accepted_core"], c["part_a"], c["benefit"], len(r["grade"]["recommended_intervention"])))
table("base 9B episodes", ep)
