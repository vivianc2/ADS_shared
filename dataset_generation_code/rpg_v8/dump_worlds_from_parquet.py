#!/usr/bin/env python3
"""Reconstruct the exact world_*.json files behind an RL parquet split.

The RL dataset builder stores each row's identity (seed, skin, archetype) in
`extra_info` but not the full world. `run_batch_v6.py` (the API-backend agent
runner) needs `world_*.json` files. Because a world is a deterministic function
of (seed, skin, archetype) via `sample_world` + `audit` (calibrate is seeded),
we regenerate each parquet row's world byte-for-byte and write it in the
generate_v7 record format. This lets a strong model (e.g. Qwen3.6-27B on Nautilus)
be evaluated on the SAME held-out set the RL run is scored on — directly comparable.

Usage:
    python dump_worlds_from_parquet.py <parquet> <outdir>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from sampler import sample_world
from generate_v7 import audit, to_record, _json_default


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    parquet, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet)
    rows = [e if isinstance(e, dict) else json.loads(e) for e in df["extra_info"]]

    manifest, arche_c, skin_c, fails = [], Counter(), Counter(), Counter()
    for i, info in enumerate(rows):
        seed, skin, arche = int(info["seed"]), info["skin"], info["archetype"]
        try:
            world = sample_world(seed, skin=skin, archetype=arche)
            world["ground_truth"]["_seed"] = seed
            res = audit(world)
        except Exception as e:
            fails[f"{type(e).__name__}"] += 1
            print(f"  [exc] seed={seed} {skin}/{arche}: {e}")
            continue
        if not res["ok"]:
            # should not happen (these rows were accepted at build time) — flag loudly
            fails["audit_not_ok"] += 1
            print(f"  [REJECT] seed={seed} {skin}/{arche} fails={res['fails']}")
            continue
        rec = to_record(world, res)
        fname = f"world_{world['world_id']}.json"
        with open(outdir / fname, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, default=_json_default)
        manifest.append({"file": fname, "world_id": world["world_id"],
                         "seed": seed, "skin": skin, "archetype": arche,
                         "gold_utility": round(res["gold"]["expected_utility"], 2)})
        arche_c[arche] += 1
        skin_c[skin] += 1

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"source_parquet": parquet, "n": len(manifest), "worlds": manifest,
                   "archetype_distribution": dict(arche_c),
                   "skin_distribution": dict(skin_c), "failures": dict(fails)}, f, indent=2)
    print(f"\nwrote {len(manifest)}/{len(rows)} worlds to {outdir}")
    print(f"archetypes: {dict(arche_c)}")
    print(f"skins: {dict(skin_c)}")
    if fails:
        print(f"FAILURES (should be empty): {dict(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
