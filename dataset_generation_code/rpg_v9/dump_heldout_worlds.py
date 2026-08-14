#!/usr/bin/env python3
"""Dump the held-out (validation) world set as world_*.json for the API agent runner.

Drives the SAME deterministic WorldStream the RL dataset builder uses, so these are
byte-for-byte the worlds the RL run is evaluated on — a strong model (Qwen3.6-27B via
Nautilus) run over them is directly comparable to run-7's held-out eval. This is
INDEPENDENT of the RL parquet (which run_rpg.sh clears/rebuilds under its own output
dir), so it never races the training launch.

Records are written straight from each WorldBundle (world + precomputed gold/battery,
which is all run_batch_v6.load_world_file needs) — no re-audit, so no double-calibrate.

Usage:
    python dump_heldout_worlds.py <outdir> [n] [seed0]
      n     default 128 (matches the build's validation split)
      seed0 default 20000000 (the build's --val_seed0)
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/work/ADS_shared/dataset_generation_code/rpg_rl")
from world_stream import WorldStream               # noqa: E402
from generate_v7 import SCHEMA_VERSION, _json_default  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    outdir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000_000
    outdir.mkdir(parents=True, exist_ok=True)

    stream = WorldStream(split="heldout", seed0=seed0)
    manifest, arche_c, skin_c = [], Counter(), Counter()
    for _ in range(n):
        b = stream.next()
        w = b.world
        rec = {
            "schema_version": SCHEMA_VERSION,
            "world_id": w["world_id"],
            "domain": w["domain"],
            "meta": {"seed": b.seed, "skin": b.skin, "archetype": b.archetype,
                     "features": w["ground_truth"].get("_features")},
            "scenario": w["scenario"],
            "scm": w["scm"].to_dict(),
            "ground_truth": w["ground_truth"],
            "oracle": {"gold": b.gold, "counterfactual_battery": b.battery},
        }
        with open(outdir / f"world_{w['world_id']}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, default=_json_default)
        manifest.append({"file": f"world_{w['world_id']}.json", "world_id": w["world_id"],
                         "seed": b.seed, "skin": b.skin, "archetype": b.archetype,
                         "gold_utility": round(b.gold["expected_utility"], 2)})
        arche_c[b.archetype] += 1
        skin_c[b.skin] += 1

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"split": "heldout", "seed0": seed0, "n": len(manifest),
                   "acceptance": round(stream.acceptance(), 3), "worlds": manifest,
                   "archetype_distribution": dict(arche_c),
                   "skin_distribution": dict(skin_c)}, f, indent=2)
    print(f"wrote {len(manifest)} held-out worlds to {outdir} (stream acceptance {stream.acceptance():.0%})")
    print(f"archetypes: {dict(sorted(arche_c.items()))}")
    print(f"skins: {dict(sorted(skin_c.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
