"""Reconstruct the exact world_*.json behind an RL parquet split (deterministic: a world is a
pure function of (seed, skin, archetype) via sample_world + audit). Lets the free-text eval run on
precisely the worlds the RL run trains/evals against. Usage: python dump_worlds_from_parquet.py <parquet> <outdir>

Rows that carry extra_info["oracle_json"] (v10 splits) reuse that stored gold + battery (checked against the world id;
calibrate is replayed) instead of re-running the 10-40 s oracle; the full audit already ran when the split was built.
Set DUMP_REAUDIT=1 to re-audit anyway."""
import json, os, sys
from collections import Counter
from pathlib import Path
import pandas as pd
from sampler import sample_world, world_kwargs
from generate_v7 import audit, to_record, _json_default
from oracle_v6 import calibrate


def main():
    parquet, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [e if isinstance(e, dict) else json.loads(e) for e in pd.read_parquet(parquet)["extra_info"]]
    manifest, arche_c, skin_c, fails = [], Counter(), Counter(), Counter()
    for info in rows:
        seed, skin, arche = int(info["seed"]), info["skin"], info["archetype"]
        try:
            w = sample_world(seed, skin=skin, archetype=arche, **world_kwargs(info))
            w["ground_truth"]["_seed"] = seed
            if info.get("oracle_json") and not os.environ.get("DUMP_REAUDIT"):
                d = json.loads(info["oracle_json"])
                if d.get("world_id") != w["world_id"]:
                    raise RuntimeError(f"stored oracle is for {d.get('world_id')}, not {w['world_id']}")
                res = {"ok": True, "fails": [], "calib": calibrate(w), "gold": d["gold"], "battery": d["battery"],
                       "audits": {"note": "stored oracle_json; audited when the split was built"}}
            else:
                res = audit(w)
        except Exception as e:
            fails[type(e).__name__] += 1; print("  [exc]", seed, skin, arche, e); continue
        if not res["ok"]:
            fails["audit_not_ok"] += 1; print("  [REJECT]", seed, skin, arche, res["fails"]); continue
        rec = to_record(w, res)
        fn = f"world_{w['world_id']}.json"
        with open(outdir / fn, "w") as f:
            json.dump(rec, f, indent=2, default=_json_default)
        manifest.append({"file": fn, "world_id": w["world_id"], "seed": seed, "skin": skin, "archetype": arche,
                         "world_version": w["ground_truth"].get("_world_version", "v9"),
                         "gold_intervention": res["gold"]["intervention"]})
        arche_c[arche] += 1; skin_c[skin] += 1
    with open(outdir / "manifest.json", "w") as f:
        json.dump({"source_parquet": parquet, "n": len(manifest), "worlds": manifest,
                   "archetype_distribution": dict(arche_c), "skin_distribution": dict(skin_c),
                   "failures": dict(fails)}, f, indent=2)
    print(f"wrote {len(manifest)}/{len(rows)} worlds to {outdir}")
    print("archetypes:", dict(sorted(arche_c.items())))
    if fails:
        print("FAILURES (should be empty):", dict(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
