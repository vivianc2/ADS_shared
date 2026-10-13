#!/usr/bin/env python3
"""Derive a4/ and hard3/ from a v10 all9/ split set (same convention as v9_splits_2026-09-26: archetype filters of the
all9 files; validation = dev_heldout_domain) and verify the whole set. ygg 2026-10-12.

  PYTHONPATH=rpg_rl:rpg_v9 RPG_SYNERGY_SOFT=20 python skyrl_rpg/derive_v10_subsets.py rpg_v9/experiment_datasets/v10_splits_2026-10-12
"""
import collections, json, os, subprocess, sys

import pyarrow as pa, pyarrow.parquet as pq

FAMILIES = {"a4": {"confounded_chain", "dose_window", "instrument_only"},
            "hard3": {"competing_causes", "synergy_pair", "hidden_subtype"}}
FILES = {"train": "train", "validation": "dev_heldout_domain", "dev_in_domain": "dev_in_domain",
         "test_in_domain": "test_in_domain", "test_heldout_domain": "test_heldout_domain"}


def key(r):
    e = r["extra_info"]
    return (int(e["seed"]), e["skin"], e["archetype"])


def main(root):
    all9 = {n: pq.read_table(os.path.join(root, "all9", f"{n}.parquet")).to_pylist() for n in set(FILES.values())}
    for fam, archs in FAMILIES.items():
        os.makedirs(os.path.join(root, fam), exist_ok=True)
        for out, src in FILES.items():
            rows = [r for r in all9[src] if r["extra_info"]["archetype"] in archs]
            pq.write_table(pa.Table.from_pylist(rows), os.path.join(root, fam, f"{out}.parquet"))
            print(f"{fam}/{out}: {len(rows)} rows {dict(collections.Counter(r['extra_info']['archetype'] for r in rows))}")
    problems = 0
    keys = {n: {key(r) for r in rows} for n, rows in all9.items()}
    for n, rows in all9.items():
        if len(keys[n]) != len(rows):
            print(f"DUPLICATE worlds in {n}"); problems += 1
        if len({r['prompt'][-1]['content'] for r in rows}) != len(rows):
            print(f"DUPLICATE prompts in {n}"); problems += 1
        bad = [r for r in rows if r["extra_info"].get("world_version") != "v10" or not r["extra_info"].get("oracle_json")]
        if bad:
            print(f"{n}: {len(bad)} rows without world_version v10 / oracle_json"); problems += 1
        cells = collections.Counter((r["extra_info"]["archetype"], r["extra_info"]["skin"]) for r in rows)
        per = collections.defaultdict(list)
        for (a, s), c in cells.items():
            per[a].append(c)
        print(f"{n}: {len(rows)} rows; skin count per archetype min-max {min(min(v) for v in per.values())}-{max(max(v) for v in per.values())}")
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = keys[names[i]] & keys[names[j]]
            if ov:
                print(f"OVERLAP {names[i]} / {names[j]}: {len(ov)}"); problems += 1
    v9 = os.path.join(os.path.dirname(root.rstrip("/")), "v9_splits_2026-09-26", "all9")
    v9keys = set()
    for f in os.listdir(v9):
        v9keys |= {key(r) for r in pq.read_table(os.path.join(v9, f)).to_pylist()}
    # same seeds as v9 are EXPECTED (same recipe, other version); report how many, for the record
    print("worlds sharing (seed, skin, archetype) with v9 splits:", {n: len(k & v9keys) for n, k in keys.items()})
    # stale-prompt check: every stored prompt equals env.reset() of the world it names
    for n in names:
        out = subprocess.run([sys.executable, "skyrl_rpg/rebuild_v9_sets.py", "verify",
                              os.path.join(root, "all9", f"{n}.parquet"), "--workers", "24"],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if "stale" in l]
        print(line[-1] if line else out.stdout[-300:] + out.stderr[-300:])
        if not line or ": 0/" not in line[-1]:
            problems += 1
    print("ALL CHECKS PASS" if problems == 0 else f"{problems} PROBLEM(S)")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if main(sys.argv[1]) else 0)
