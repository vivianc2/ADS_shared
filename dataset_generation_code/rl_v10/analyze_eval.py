#!/usr/bin/env python3
"""Summarise an rl_v10 eval folder (launch_eval.sh layout: <split>_s<k>/shard_*/result_*.json), 2026-10-23.

Per split and structure: part A, accepted (release rule and D1 core), mean benefit, reward under RewardCfg, the share
of worlds whose n samples do not all get the same reward (a GRPO group with signal), and basic process numbers.
95% CIs: world bootstrap (samples of one world stay together). Writes <out>.csv (episodes) and prints a table.

  ~/miniconda3/envs/ADS/bin/python analyze_eval.py ~/rpgrl/rl_v10/evals/base9b_full [--worlds-root DIR]
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reward_v10 import reward_components, RewardCfg  # noqa: E402

DATA = os.path.expanduser("~/ADS_folder/ADS_shared/rpg_benchmark/data")


def boot_ci(per_world, n_boot=2000, seed=0):
    """per_world: list of per-world means. Returns (lo, hi) of the mean."""
    a = np.asarray(per_world, float)
    if len(a) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = RewardCfg()
    rows = []
    worlds = {}
    for f in sorted(glob.glob(os.path.join(args.root, "*_s*", "shard_*", "result_*.json"))):
        split, sample = os.path.basename(os.path.dirname(os.path.dirname(f))).rsplit("_s", 1)
        r = json.load(open(f))
        wid = r["world_id"]
        key = (split, wid)
        if key not in worlds:
            wf = r.get("world_file") or os.path.join(DATA, split, f"world_{wid}.json.gz")
            rec = json.load(gzip.open(os.path.realpath(wf)))
            worlds[key] = rec
        rec = worlds[key]
        c = reward_components(r, {"ground_truth": rec["ground_truth"]}, {})
        turns = [t for t in r["turns"] if t.get("turn") != "post"]
        ans_turn = next((t["turn"] for t in turns if t.get("action_type") == "answer"), None)
        rows.append(dict(split=split, sample=int(sample), world_id=wid, archetype=rec["ground_truth"]["_archetype"],
                         part_a=c["part_a"], accepted_release=c["accepted_release"], accepted_core=c["accepted_core"],
                         benefit=c["benefit"], battery_core=c["battery_core"], reward=cfg.reward(c),
                         queries=r["queries_used"], interventions=r["interventions_run"], turns=len(turns),
                         answer_turn=ans_turn, answered=int(c["answered"])))
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"reward = {cfg.as_dict()}; episodes {len(rows)}")
    for split in sorted({r["split"] for r in rows}):
        R = [r for r in rows if r["split"] == split]
        by_w = collections.defaultdict(list)
        for r in R:
            by_w[r["world_id"]].append(r)
        def rate(k, sub=None):
            ws = [v for v in by_w.values() if sub is None or v[0]["archetype"] == sub]
            pw = [np.mean([x[k] for x in v]) for v in ws]
            lo, hi = boot_ci(pw)
            return np.mean(pw) if pw else np.nan, lo, hi
        mixed = [len({round(x["reward"], 6) for x in v}) > 1 for v in by_w.values()]
        anypass = [any(x["part_a"] for x in v) for v in by_w.values()]
        print(f"\n== {split}: {len(by_w)} worlds x {len(R) / max(len(by_w), 1):.1f} samples")
        for k in ("part_a", "accepted_release", "accepted_core", "reward"):
            m, lo, hi = rate(k)
            print(f"  {k:17s} {m:.3f} [{lo:.3f}, {hi:.3f}]")
        ben = [r["benefit"] for r in R if r["benefit"] is not None]
        print(f"  mean benefit {np.mean(ben):.3f} (median {np.median(ben):.3f}); worlds with mixed rewards "
              f"{sum(mixed)}/{len(mixed)} ({np.mean(mixed):.2f}); worlds passed by >=1 sample {sum(anypass)}/{len(anypass)}")
        print(f"  experiments {np.mean([r['queries'] for r in R]):.1f}, interventions {np.mean([r['interventions'] for r in R]):.1f}, "
              f"turns {np.mean([r['turns'] for r in R]):.1f}, answer turn {np.mean([r['answer_turn'] for r in R if r['answer_turn'] is not None]):.1f}, "
              f"answered {np.mean([r['answered'] for r in R]):.3f}")
        print(f"  {'structure':22s} {'partA':>6s} {'acc_core':>8s} {'reward':>7s} {'mixed':>6s} {'anypass':>7s}")
        for a in sorted({r["archetype"] for r in R}):
            ws = [v for v in by_w.values() if v[0]["archetype"] == a]
            mx = np.mean([len({round(x["reward"], 6) for x in v}) > 1 for v in ws])
            ap_ = sum(any(x["part_a"] for x in v) for v in ws)
            print(f"  {a:22s} {rate('part_a', a)[0]:6.3f} {rate('accepted_core', a)[0]:8.3f} {rate('reward', a)[0]:7.3f} "
                  f"{mx:6.2f} {ap_:3d}/{len(ws)}")


if __name__ == "__main__":
    main()
