#!/usr/bin/env python3
"""Build SkyRL parquets for the HARD archetypes (competing_causes, synergy_pair, hidden_subtype).

Why a separate builder: `rpg_dataset.py` routes through `WorldStream`, whose TRAIN split
excludes the reserved archetypes (`splits.HELDOUT_ARCHETYPES` = hidden_subtype, surrogate_trap,
competing_causes). This experiment trains ON those archetypes deliberately, so the split here is
by SKIN only:

    train = hard archetypes x train skins        (seed0 40_000_000)
    val   = hard archetypes x held-out skins     (seed0 41_000_000; clinical, fermentation)

so the eval still reads as domain transfer within the trained archetypes. Seeds are disjoint
from every range used before (10M/20M v9 fast, 60M/20M generation, 7M probes).

Rows are byte-compatible with `rpg_dataset.py` (same prompt rendering via RPGEnv.reset(),
same extra_info keys that RPGSkyEnv rebuilds the world from). Worlds are regenerated at train
time from (seed, skin, archetype) with the CURRENT rpg_v9 sampler (RPG_SYNERGY_SOFT=20 default,
positive-polarity synergy proxies) -- i.e. the latest world version, nothing is frozen here.

Run (host, any env with scipy + pyarrow):
    cd dataset_generation_code && PYTHONPATH=rpg_rl:rpg_v9 python skyrl_rpg/build_hard_arch_ds.py \
        --output_dir rpg_v9/experiment_datasets/rl_train/rl_hard_lever_ds --train_size 384 --val_size 48
Optionally `--dump_val_worlds <dir>` writes the val worlds as world_*.json for the clean eval harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

_BASE = os.environ.get("RPG_SRC", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# rpg_rl must precede this script's own dir: skyrl_rpg/env.py (the SkyRL adapter, needs skyrl_gym)
# would otherwise shadow rpg_rl/env.py (the plain RPGEnv we render prompts with).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (os.path.join(_BASE, os.environ.get("RPG_PROTO", "rpg_v9")), os.path.join(_BASE, "rpg_rl")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import pyarrow as pa                                  # noqa: E402
import pyarrow.parquet as pq                          # noqa: E402
from sampler import sample_world                      # noqa: E402
from generate_v7 import audit                         # noqa: E402
from splits import HELDOUT_SKINS, train_skins         # noqa: E402
from env import RPGEnv, SYSTEM_PROMPT                 # noqa: E402
from reward import lever_sets                         # noqa: E402

HARD = ["competing_causes", "synergy_pair", "hidden_subtype"]
SYSTEM_MSG = {"role": "system", "content": SYSTEM_PROMPT}


def _try(args):
    """Sample + audit one (seed, skin, archetype); return the row or None. Runs in a worker."""
    seed, skin, arch, max_turns, budget, dump_dir = args
    try:
        w = sample_world(seed, skin=skin, archetype=arch)
        res = audit(w)
    except Exception as e:  # noqa: BLE001
        return {"seed": seed, "ok": False, "err": f"{type(e).__name__}: {e}"}
    if not res["ok"]:
        return {"seed": seed, "ok": False, "fails": list(res.get("fails", []))}
    env = RPGEnv(world=w, gold=res["gold"], battery=res["battery"], catalog_seed=seed,
                 max_turns=max_turns, budget=budget)
    first_obs = env.reset()
    if os.environ.get("RPG_NO_THINK"):
        first_obs = first_obs + "\n/no_think"
    causal, must = lever_sets(w, res["gold"])
    if dump_dir:                                   # same record shape as dump_heldout_worlds.py
        from generate_v7 import SCHEMA_VERSION, _json_default
        rec = {"schema_version": SCHEMA_VERSION, "world_id": w["world_id"], "domain": w["domain"],
               "meta": {"seed": seed, "skin": skin, "archetype": arch,
                        "features": w["ground_truth"].get("_features")},
               "scenario": w["scenario"], "scm": w["scm"].to_dict(),
               "ground_truth": w["ground_truth"],
               "oracle": {"gold": res["gold"], "counterfactual_battery": res["battery"]}}
        with open(os.path.join(dump_dir, f"world_{w['world_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, default=_json_default)
    return {
        "seed": seed, "ok": True, "world_id": w["world_id"],
        "n_causal": len(causal), "n_must": len(must),
        "row": {
            "data_source": "rpg_v9",
            "prompt": [SYSTEM_MSG, {"role": "user", "content": first_obs}],
            "env_class": "rpg",
            "reward_spec": {"method": "rule", "ground_truth": ""},
            "extra_info": {"seed": int(seed), "skin": skin, "archetype": arch,
                           "max_turns": int(max_turns), "budget": int(budget),
                           "split": "hard_train" if skin not in HELDOUT_SKINS else "hard_heldout"},
        },
    }


def build(n: int, seed0: int, skins, max_turns, budget, workers, dump_dir=None):
    """Deterministic: seed i -> cell (skins x HARD)[i % ncells]; keep the first n accepted in
    seed order. Over-provision attempts (acceptance is ~85-100% on these archetypes)."""
    cells = [(s, a) for a in HARD for s in skins]
    attempts = int(n / 0.6) + 16
    jobs = [(seed0 + i, *cells[i % len(cells)], max_turns, budget, dump_dir) for i in range(attempts)]
    with Pool(workers) as pool:
        results = pool.map(_try, jobs, chunksize=2)
    ok_all = [r for r in results if r["ok"]]
    ok = ok_all[:n]
    if dump_dir:                                   # workers dumped every accepted world; keep only the n used
        for r in ok_all[n:]:
            try:
                os.remove(os.path.join(dump_dir, f"world_{r['world_id']}.json"))
            except OSError:
                pass
    rej = Counter(f for r in results if not r["ok"] for f in (r.get("fails") or [r.get("err", "?")]))
    dist = Counter((r["row"]["extra_info"]["skin"], r["row"]["extra_info"]["archetype"]) for r in ok)
    print(f"  accepted {len(ok_all)}/{len(results)} attempts (kept {len(ok)}); rejections={dict(rej)}")
    print(f"  per-archetype: {dict(Counter(a for _, a in dist.elements()))}")
    print(f"  lever sets: n_causal={dict(Counter(r['n_causal'] for r in ok))} n_must={dict(Counter(r['n_must'] for r in ok))}")
    assert len(ok) == n, f"only {len(ok)} accepted worlds for n={n}; raise attempts"
    return [r["row"] for r in ok]


def _write(rows, path):
    pq.write_table(pa.Table.from_pylist(rows), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_size", type=int, default=384)
    ap.add_argument("--val_size", type=int, default=48)
    ap.add_argument("--train_seed0", type=int, default=40_000_000)
    ap.add_argument("--val_seed0", type=int, default=41_000_000)
    ap.add_argument("--max_turns", type=int, default=32)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dump_val_worlds", default="", help="also write val worlds as world_*.json here")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dump_val_worlds:
        os.makedirs(args.dump_val_worlds, exist_ok=True)
    print(f"[train] {HARD} x {train_skins()} from seed {args.train_seed0}")
    train = build(args.train_size, args.train_seed0, train_skins(), args.max_turns, args.budget, args.workers)
    print(f"[val]   {HARD} x {HELDOUT_SKINS} from seed {args.val_seed0}")
    val = build(args.val_size, args.val_seed0, HELDOUT_SKINS, args.max_turns, args.budget, args.workers,
                dump_dir=args.dump_val_worlds or None)
    _write(train, os.path.join(args.output_dir, "train.parquet"))
    _write(val, os.path.join(args.output_dir, "validation.parquet"))
    with open(os.path.join(args.output_dir, "MANIFEST.json"), "w") as f:
        json.dump({"archetypes": HARD, "train_skins": train_skins(), "val_skins": HELDOUT_SKINS,
                   "train_seed0": args.train_seed0, "val_seed0": args.val_seed0,
                   "train_size": len(train), "val_size": len(val),
                   "synergy_soft": os.environ.get("RPG_SYNERGY_SOFT", "20 (default)"),
                   "note": "hard-archetype set for the lever-gate GRPO experiment (2026-09-21); "
                           "split is by SKIN only, archetype reservation deliberately ignored"}, f, indent=2)
    print(f"wrote train({len(train)}) + validation({len(val)}) to {args.output_dir}")


if __name__ == "__main__":
    main()
