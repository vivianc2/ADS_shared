#!/usr/bin/env python3
"""Build the cue-free evaluation split paired with an existing v10 split (2026-10-23).

For every row of a v10 split parquet (e.g. v10_benchmark_2026-10-12/dev_heldout_domain = data/validation):
  1. sample_world(seed, skin, archetype, v10)        -- identical to the standard world before the oracle
  2. calibrate(world)                                -- as generate_v7.audit does first (mutates the SCM)
  3. cuefree.make_confounded(world, seed) (v3, default) or make_cuefree (v2, --design v2)
  4. optimal_gold, counterfactual_battery, all five audits + v10_audit (generate_v7.audit's steps, same order)
  5. compare with the row's stored oracle (the standard world): gold intervention, expected / baseline utility,
     battery; written to pairing.jsonl. v2: the split is valid only if they match. v3: the gold intervention should
     match; baseline utility may differ (the lever's observational regime changes); the oracle is the new world's.
World files are written in the dump format (generate_v7.to_record), then converted with to_release_format.py.

  RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 ~/miniconda3/envs/ADS/bin/python build_cuefree.py \
      <split.parquet> <out_dump_dir> --workers 24 [--limit N]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
for p in (os.path.join(D, "rpg_v9"), os.path.join(D, "rpg_rl"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("RPG_SYNERGY_SOFT", "20")

import numpy as np  # noqa: E402
from sampler import sample_world  # noqa: E402
from generate_v7 import to_record, _json_default  # noqa: E402
from oracle_v6 import (calibrate, optimal_gold, counterfactual_battery, decoy_audit, proxy_signal_audit,  # noqa: E402
                       distractor_inertness_audit, gold_selfconsistency_audit, counterintuitiveness_audit,
                       is_v10, v10_audit)
from cuefree import make_cuefree, make_confounded  # noqa: E402
DESIGN = {"v2": make_cuefree, "v3": make_confounded}


def _default(o):
    """generate_v7._json_default plus numpy bools (the audit results hold them; the standard build never dumps
    audit results, so it never needed this)."""
    if isinstance(o, np.bool_):
        return bool(o)
    return _json_default(o)


V3_ATTEMPTS = 5


def _proxy_gold_shift(world, gold, n=20000, seed=222):
    """v3 proxy check: does the GOLD intervention move the true mechanism proxy by > 1 baseline SD? The release
    proxy_signal_audit accepts a proxy that is either visible in observational data or moved by setting the targeted
    control to the top of its range. In v3 a held lever makes the proxy invisible in passive data on purpose, and in
    v10 worlds the top of the range is often not where the mechanism switches (interior setpoints, gates near
    setpoints), so the release test misses proxies that the right experiment does move."""
    scm = world["scm"]
    proxy = world["ground_truth"]["true_mechanism_proxy"]
    base = scm.measure(scm.sample(n, seed=seed, select=True), [proxy], seed=seed + 3)[proxy]
    iv = gold["intervention"]
    do = scm.measure(scm.sample(n, intervention=iv, seed=seed + 5), [proxy], seed=seed + 6, intervention=iv)[proxy]
    return abs(float(np.mean(do) - np.mean(base))) / (float(np.std(base)) + 1e-9)


def audit_cuefree(world, seed, design="v3"):
    if design == "v3":
        import copy
        calib = calibrate(world)
        pristine = copy.deepcopy(world)
        res = None
        for attempt in range(V3_ATTEMPTS):
            w = copy.deepcopy(pristine)
            res = _audit_after(w, seed, design, calib, attempt)
            if res["ok"]:
                break
        world.clear()
        world.update(res.pop("_world"))
        return res
    calib = calibrate(world)
    return _audit_after(world, seed, design, calib, 0)


def _audit_after(world, seed, design, calib, attempt):
    cf = DESIGN[design](world, seed, attempt) if design == "v3" else DESIGN[design](world, seed)
    gold = optimal_gold(world)
    battery = counterfactual_battery(world)
    audits = {"decoy": decoy_audit(world), "proxy_signal": proxy_signal_audit(world),
              "distractor_inertness": distractor_inertness_audit(world, gold),
              "gold_selfconsistency": gold_selfconsistency_audit(world, gold),
              "counterintuitiveness": counterintuitiveness_audit(world, gold)}
    if is_v10(world):
        audits["v10"] = v10_audit(world, gold)
    if design == "v3" and not audits["proxy_signal"]["passed"]:
        shift = _proxy_gold_shift(world, gold)
        audits["proxy_signal"]["gold_shift_sd"] = round(shift, 2)
        audits["proxy_signal"]["passed"] = bool(shift > 1.0)
        audits["proxy_signal"]["passed_by"] = "gold_shift" if shift > 1.0 else None
    fails = [k for k, a in audits.items() if not a["passed"]]
    return {"ok": not fails, "fails": fails, "calib": calib, "gold": gold, "battery": battery, "audits": audits,
            "cuefree": cf, "_world": world}


def _canon(x):
    return json.loads(json.dumps(x, default=_default, sort_keys=True))


def job(args):
    info, out_dir, design = args
    seed, skin, arch = int(info["seed"]), info["skin"], info["archetype"]
    rec = {"seed": seed, "skin": skin, "archetype": arch}
    try:
        w = sample_world(seed, skin=skin, archetype=arch, version="v10")
        w["ground_truth"]["_seed"] = seed
        std = json.loads(info["oracle_json"])
        if std["world_id"] != w["world_id"]:
            raise RuntimeError(f"stored oracle is for {std['world_id']}, not {w['world_id']}")
        res = audit_cuefree(w, seed, design)
        res.pop("_world", None)
        g, sg = _canon(res["gold"]), std["gold"]
        b, sb = _canon(res["battery"]), std["battery"]
        rec.update(world_id=w["world_id"], ok=res["ok"], fails=res["fails"],
                   gold_iv_equal=g["intervention"] == sg["intervention"],
                   gold_u=g["expected_utility"], std_gold_u=sg["expected_utility"],
                   base_u=g.get("baseline_utility"), std_base_u=sg.get("baseline_utility"),
                   battery_equal=b == sb,
                   battery_diff_keys=sorted(k for k in set(b) | set(sb) if b.get(k) != sb.get(k)),
                   cuefree=res["cuefree"], attempt=res["cuefree"].get("attempt", 0),
                   proxy_passed_by=res["audits"]["proxy_signal"].get("passed_by"))
        if res["ok"]:
            fn = os.path.join(out_dir, f"world_{w['world_id']}.json")
            with open(fn, "w") as f:
                json.dump(to_record(w, res), f, indent=2, default=_default)
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["tb"] = traceback.format_exc()[-1500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--design", choices=sorted(DESIGN), default="v3",
                    help="v3 (default, 2026-10-24): per-setting held / free / responsive regimes; v2: shared driver")
    args = ap.parse_args()
    import pandas as pd
    rows = [e if isinstance(e, dict) else json.loads(e) for e in pd.read_parquet(args.parquet)["extra_info"]]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(job, [(r, args.out_dir, args.design) for r in rows]):
            out.append(r)
    with open(os.path.join(args.out_dir, "pairing.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, default=_default) + "\n")
    ok = [r for r in out if "error" not in r]
    n = len(ok)
    du = [abs(r["gold_u"] - r["std_gold_u"]) for r in ok]
    db = [abs((r["base_u"] or 0) - (r["std_base_u"] or 0)) for r in ok]
    print(f"rows {len(out)}, errors {len(out) - n}, audits passed {sum(r['ok'] for r in ok)}/{n} "
          f"(fails: {dict(__import__('collections').Counter(f for r in ok for f in r['fails']))})")
    print(f"gold intervention equal {sum(r['gold_iv_equal'] for r in ok)}/{n}; battery equal "
          f"{sum(r['battery_equal'] for r in ok)}/{n} (differing keys: "
          f"{dict(__import__('collections').Counter(k for r in ok for k in r['battery_diff_keys']))}); "
          f"max |gold utility diff| {max(du):.3f}, max |baseline utility diff| {max(db):.3f}")
    if args.design == "v3":
        reg = __import__('collections').Counter((c["role"], c["regime"]) for r in ok for c in r["cuefree"]["changed"])
        print("v3 regimes (role, regime):", dict(sorted(reg.items())))
        return
    lev = [x for r in ok for x in r["cuefree"]["lever_r"]]
    inert = [abs(v) for r in ok for t, v in r["cuefree"]["realised_r"].items() if r["cuefree"]["roles"].get(t) == "inert"]
    print(f"|r| with outcome after change: primary levers mean {np.mean(lev):.3f} (n {len(lev)}), inert settings mean "
          f"{np.mean(inert):.3f} (n {len(inert)}); settings rewired per world "
          f"{np.mean([len(r['cuefree']['changed']) for r in ok]):.1f}")


if __name__ == "__main__":
    main()
