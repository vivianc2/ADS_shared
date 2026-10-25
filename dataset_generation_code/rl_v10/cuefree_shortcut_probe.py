#!/usr/bin/env python3
"""Passive-shortcut battery for a folder of release-format worlds (2026-10-24, camera-ready session).

Can one observational batch (400 units, selection applied, as the benchmark's first measurement) name the lever?
For each world with a primary level lever, every level setting whose level is measured is scored, and the lever is
compared with the other level settings (inert and harm roles):
  r_out     |corr(level, outcome)|                                         (the original |r| cue)
  odd       -mean |corr(level, other level settings)|                      (v2's shortcut: inert settings shared a driver)
  partial1  |corr(level, outcome | first principal component of the OTHER level settings)|
  partialA  |partial corr(level, outcome | ALL other level settings)|      (full linear adjustment)
Reported: AUC = P(lever scores above another setting) and top-1 = share of worlds where the lever scores highest.
Chance: AUC 0.5; top-1 about 1 / (number of level settings).

  RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 ~/miniconda3/envs/ADS/bin/python cuefree_shortcut_probe.py <world_dir> [limit]
"""
import collections
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/ADS_folder/ADS_shared/rpg_benchmark"))
from scm import WorldSCM  # noqa: E402

SCORES = ("r_out", "odd", "partial1", "partialA")


def _resid(y, X):
    X1 = np.column_stack([X, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def scores_for(world, n=400, seed=12345):
    scm = WorldSCM.from_dict(world["scm"])
    roles = world["ground_truth"].get("_v10_roles") or {}
    lev = []
    for aid, a in scm.actuators.items():
        t = scm.variables.get(a.get("target"), {})
        if a.get("op") == "set" and t.get("kind") == "observable" and t.get("measurable") and roles.get(aid) in (
                "primary", "inert", "harm"):
            lev.append((a["target"], roles[aid]))
    if not any(r == "primary" for _, r in lev) or len(lev) < 4:
        return None
    v = scm.sample(n, seed=seed, select=True)
    o = np.asarray(v[scm.outcome], float)
    X = np.column_stack([np.asarray(v[t], float) for t, _ in lev])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    oz = (o - o.mean()) / (o.std() + 1e-12)
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C)
    out = {k: [] for k in SCORES}
    for i in range(len(lev)):
        others = [j for j in range(len(lev)) if j != i]
        out["r_out"].append(abs(float(np.mean(X[:, i] * oz))))
        out["odd"].append(-float(np.mean(np.abs(C[i, others]))))
        Xo = X[:, others]
        u, _, _ = np.linalg.svd(Xo - Xo.mean(0), full_matrices=False)
        pc = u[:, :1]
        rx, ro = X[:, i] - pc @ (pc.T @ X[:, i]), oz - pc @ (pc.T @ oz)
        out["partial1"].append(abs(float(np.corrcoef(rx, ro)[0, 1])) if rx.std() > 1e-9 else 0.0)
        rx, ro = _resid(X[:, i], Xo), _resid(oz, Xo)
        out["partialA"].append(abs(float(np.corrcoef(rx, ro)[0, 1])) if rx.std() > 1e-9 else 0.0)
    return [r for _, r in lev], out


def main():
    d = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    files = sorted(glob.glob(os.path.join(d, "world_*.json.gz")))
    files = files[:lim] if lim else files
    auc, top1, nset = collections.defaultdict(list), collections.defaultdict(list), []
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        w = json.load(gzip.open(f))
        res = scores_for(w)
        if res is None:
            continue
        roles, sc = res
        nset.append(len(roles))
        arch = w["ground_truth"].get("_archetype")
        for k, s in sc.items():
            L = [s[i] for i in range(len(roles)) if roles[i] == "primary"]
            O = [s[i] for i in range(len(roles)) if roles[i] != "primary"]
            a = float(np.mean([[1.0 if l > x else 0.5 if l == x else 0.0 for x in O] for l in L]))
            auc[k].append(a)
            top1[k].append(max(L) > max(O))
            by[k][arch].append(a)
    print(f"{d}: worlds with a level lever {len(nset)}; level settings per world {np.mean(nset):.1f} "
          f"(chance top-1 about {np.mean([1 / x for x in nset]):.2f})")
    for k in SCORES:
        print(f"  {k:9s} AUC {np.mean(auc[k]):.3f}   lever ranked first in {np.mean(top1[k]):.2f}")
    print("  per structure (AUC):", ", ".join(
        f"{a} " + "/".join(f"{np.mean(by[k][a]):.2f}" for k in SCORES) for a in sorted(by["r_out"])))


if __name__ == "__main__":
    main()
