#!/usr/bin/env python3
"""Probe for the cue-free split design (2026-10-23): in standard v10 worlds, how strongly do (a) the PRIMARY level
levers, (b) inert level settings, (c) exogenous latents (root susceptibility, confounders) correlate with the
measured outcome in the observational batch an agent first sees (400 units, selection applied)?"""
import os, sys, json, collections
import numpy as np
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(D, "rpg_v9"), os.path.join(D, "rpg_rl")):
    sys.path.insert(0, p)
os.environ.setdefault("RPG_SYNERGY_SOFT", "20")
from sampler import sample_world, ARCHETYPES
from concurrent.futures import ProcessPoolExecutor

def levels(world):
    scm, roles = world["scm"], world["ground_truth"]["_v10_roles"]
    out = []
    for aid, a in scm.actuators.items():
        t = scm.variables.get(a.get("target"), {})
        if a.get("op") == "set" and t.get("kind") == "observable" and t.get("measurable"):
            out.append((aid, a["target"], roles.get(aid)))
    return out

def job(args):
    seed, arch, skin = args
    try:
        w = sample_world(seed, skin=skin, archetype=arch, version="v10")
    except Exception as e:
        return None
    scm = w["scm"]; gt = w["ground_truth"]
    vals = scm.sample(4000, seed=seed + 1, select=True)
    lv = levels(w)
    lat = [n for n, v in scm.variables.items() if v.get("kind") == "latent" and "dist" in v and not v.get("parents")]
    o = np.asarray(vals[scm.outcome], float)
    def r(x):
        x = np.asarray(x, float)
        return float(np.corrcoef(x, o)[0, 1]) if x.std() > 1e-9 and o.std() > 1e-9 else 0.0
    rec = {"seed": seed, "arch": arch, "lev": [(role, abs(r(vals[t]))) for _, t, role in lv],
           "lat": sorted([(n, abs(r(vals[n]))) for n in lat], key=lambda x: -x[1])[:4]}
    return rec

if __name__ == "__main__":
    from splits import train_skins
    skins = sorted(train_skins())
    jobs = [(70_000_000 + i, ARCHETYPES[i % 9], skins[(i // 9) % len(skins)]) for i in range(180)]
    with ProcessPoolExecutor(24) as ex:
        rows = [r for r in ex.map(job, jobs) if r]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for role, v in r["lev"]:
            by[r["arch"]][role].append(v)
        by[r["arch"]]["best_latent"].append(r["lat"][0][1] if r["lat"] else 0)
    print(f"{len(rows)} worlds; |r| with the outcome (observational, 4000 units, selection applied): mean [n]")
    for a in ARCHETYPES:
        d = by[a]
        print(f"  {a:20s} " + "  ".join(f"{k}={np.mean(v):.2f}[{len(v)}]" for k, v in sorted(d.items())))
    json.dump(rows, open(os.path.expanduser("~/rpgrl/rl_v10/cuefree/probe.json"), "w"))
