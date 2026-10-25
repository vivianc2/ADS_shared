#!/usr/bin/env python3
"""Paired comparison: scripted baselines on standard vs cue-free worlds (same world ids), 2026-10-23.

Reads ~/rpgrl/rl_v10/scripts/{validation,validation_cuefree}/{baselines,cd1}_episodes.csv; prints part A per policy
(and by structure for the key policies), the paired difference with a world-bootstrap 95% CI, and writes
~/rpgrl/rl_v10/scripts/cuefree_vs_standard.csv.
"""
import collections
import csv
import os
import sys

import numpy as np

R = os.path.expanduser("~/rpgrl/rl_v10/scripts")


def load(split):
    out = {}
    d = split if os.path.isabs(split) else os.path.join(R, split)
    for fn, key in (("baselines_episodes.csv", lambda r: f"{r['policy']}@{r['budget']}"),
                    ("cd1_episodes.csv", lambda r: r["policy"]), ("cd1_episodes_nocd.csv", lambda r: r["policy"])):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            out[(key(r), r["world_id"])] = r
    return out


def main():
    std, cf = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("validation", "validation_cuefree")
    a, b = load(std), load(cf)
    pols = sorted({k[0] for k in a} & {k[0] for k in b})
    rows = []
    rng = np.random.default_rng(0)
    print(f"{'policy':28s} {'standard':>9s} {'cue-free':>9s} {'diff':>7s} {'95% CI':>16s}  n")
    for p in pols:
        ws = sorted({k[1] for k in a if k[0] == p} & {k[1] for k in b if k[0] == p})
        x = np.array([float(a[(p, w)]["part_a"]) for w in ws])
        y = np.array([float(b[(p, w)]["part_a"]) for w in ws])
        d = y - x
        bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"policy": p, "n": len(ws), "standard": x.mean(), "cuefree": y.mean(), "diff": d.mean(),
                     "lo": lo, "hi": hi})
        print(f"{p:28s} {x.mean():9.3f} {y.mean():9.3f} {d.mean():+7.3f} [{lo:+.3f}, {hi:+.3f}]  {len(ws)}")
    with open(os.path.join(R, f"cuefree_vs_standard_{os.path.basename(cf.rstrip('/'))}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    key = [p for p in ("nocd_corr", "nocd_corr_screen", "cd_pc_rcit_ivbase", "informed@15", "oracle_levers@15",
                       "blind_grid@120", "random@15") if p in pols]
    arch = sorted({r["archetype"] for r in a.values()})
    print("\npart A by structure, standard -> cue-free")
    print(f"{'structure':20s}" + "".join(f"{p[:14]:>16s}" for p in key))
    for s in arch:
        cells = []
        for p in key:
            ws = [k[1] for k in a if k[0] == p and a[k]["archetype"] == s and (p, k[1]) in b]
            x = sum(int(a[(p, w)]["part_a"]) for w in ws)
            y = sum(int(b[(p, w)]["part_a"]) for w in ws)
            cells.append(f"{x:>7d} -> {y:<5d}")
        print(f"{s:20s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
