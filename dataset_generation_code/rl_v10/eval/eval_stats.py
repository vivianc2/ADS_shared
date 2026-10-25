#!/usr/bin/env python3
"""Quick health + length stats for an rl_v10 eval folder (all result_*.json under it)."""
import glob, json, os, sys, collections
import numpy as np
root = sys.argv[1]
fs = sorted(glob.glob(os.path.join(root, "*", "shard_*", "result_*.json")))
pt, ct, fin, n_t, ans, pa, acc, q, emp, unparsed, retries = [], [], collections.Counter(), [], 0, 0, 0, [], 0, 0, 0
by_split = collections.defaultdict(list)
for f in fs:
    r = json.load(open(f)); split = f.split(os.sep)[-3]
    turns = [t for t in r["turns"] if t.get("turn") != "post"]
    for t in turns:
        u = t.get("usage") or {}
        if u.get("prompt_tokens"): pt.append(u["prompt_tokens"])
        if u.get("completion_tokens"): ct.append(u["completion_tokens"])
        fin[t.get("finish_reason")] += 1
        if not (t.get("raw") or "").strip(): emp += 1
        if t.get("error") == "no action parsed": unparsed += 1
    n_t.append(len(turns)); g = r["grade"]
    a = not g.get("forced_no_answer", False); ans += a; pa += g["part_a_utility_ok"]; acc += g["accepted"]; q.append(r["queries_used"])
    by_split[split.rsplit("_s", 1)[0]].append(g["part_a_utility_ok"])
n = len(fs)
print(f"episodes {n}; answered {ans}; partA {pa} ({pa/max(n,1):.3f}); accepted(release) {acc}")
print("partA by split:", {k: f"{sum(v)}/{len(v)}" for k, v in by_split.items()})
print(f"turns/episode mean {np.mean(n_t):.1f} max {max(n_t)}; experiments mean {np.mean(q):.1f}")
print(f"prompt tokens per turn: median {np.median(pt):.0f} p95 {np.percentile(pt,95):.0f} max {max(pt)}")
print(f"completion tokens per turn: median {np.median(ct):.0f} p95 {np.percentile(ct,95):.0f} p99 {np.percentile(ct,99):.0f} max {max(ct)}")
print(f"finish reasons {dict(fin)}; empty replies {emp}; unparsed turns {unparsed}")
