#!/usr/bin/env python3
"""Lever-vs-inert |corr(level, outcome)| AUC of a cue-free build (pairing.jsonl). 0.5 = the cue is gone."""
import json, sys, collections
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
auc, top1, capped, by = [], [], 0, collections.defaultdict(list)
for r in rows:
    if "cuefree" not in r: continue
    c = r["cuefree"]; roles = c["roles"]; rr = c["realised_r"]
    lev = [abs(v) for t, v in rr.items() if roles.get(t) == "primary"]
    ine = [abs(v) for t, v in rr.items() if roles.get(t) == "inert"]
    capped += sum(1 for ch in c["changed"] if ch["target_r"] >= 0.97 * abs(c["rho_driver"]) - 1e-3)
    if not lev or not ine: continue
    for l in lev:
        a = np.mean([1.0 if l > x else 0.5 if l == x else 0.0 for x in ine]); auc.append(a); by[r["archetype"]].append(a)
    top1.append(max(lev) > max(ine))
print(f"worlds {len(rows)}; lever-vs-inert |r| AUC {np.mean(auc):.3f} (n {len(auc)}); lever = top-|r| setting in "
      f"{np.mean(top1):.2f} of worlds with a level lever; capped targets {capped}; driver R mean "
      f"{np.mean([r['cuefree']['rho_driver'] for r in rows if 'cuefree' in r]):.3f}")
for a, v in sorted(by.items()): print(f"  {a:20s} AUC {np.mean(v):.3f} (n {len(v)})")
