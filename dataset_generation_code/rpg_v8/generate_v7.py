#!/usr/bin/env python3
"""RPG v7 world-batch generator (structural sampling + audit filter).

Unlike v6 (which jittered 4 hand-authored templates), v7 SAMPLES the causal
structure per seed (skin, chain depth, #confounders/decoys/distractors,
mechanism forms, difficulty features), audits it with the shared v6 audit suite,
and emits only worlds that pass. This produces N structurally-distinct worlds,
not N recolorings of a fixed template.

Usage:
    python generate_v7.py --outdir out_v7 --n 40 --seed 100000
    # optional: restrict skins / require a feature
    python generate_v7.py --outdir out_v7_signflip --n 20 --require-feature sign_flip
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sampler import sample_world, FEATURES, ARCHETYPES
from skins import skin_names
from oracle_v6 import (calibrate, optimal_gold, counterfactual_battery,
                       decoy_audit, proxy_signal_audit, distractor_inertness_audit,
                       gold_selfconsistency_audit, counterintuitiveness_audit)

SCHEMA_VERSION = "rpg_scm_v7"


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def audit(world) -> Dict[str, Any]:
    calib = calibrate(world)
    gold = optimal_gold(world)
    battery = counterfactual_battery(world)
    audits = {
        "decoy": decoy_audit(world),
        "proxy_signal": proxy_signal_audit(world),
        "distractor_inertness": distractor_inertness_audit(world, gold),
        "gold_selfconsistency": gold_selfconsistency_audit(world, gold),
        "counterintuitiveness": counterintuitiveness_audit(world, gold),
    }
    ok = all(a["passed"] for a in audits.values())
    fails = [k for k, a in audits.items() if not a["passed"]]
    return {"ok": ok, "fails": fails, "calib": calib, "gold": gold,
            "battery": battery, "audits": audits}


def to_record(world, res) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "world_id": world["world_id"],
        "domain": world["domain"],
        "meta": {"seed": world["ground_truth"].get("_seed"),
                 "features": world["ground_truth"].get("_features"),
                 "skin": world["ground_truth"].get("_skin"),
                 "depth": world["ground_truth"].get("_depth")},
        "scenario": world["scenario"],
        "scm": world["scm"].to_dict(),
        "ground_truth": world["ground_truth"],
        "oracle": {"gold": res["gold"], "counterfactual_battery": res["battery"],
                   "calibration": res["calib"], "audits": res["audits"]},
    }


def generate(outdir: str, n: int, seed0: int, skins: List[str],
             require_feature: str = None, archetype: str = None) -> None:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    accepted, attempts, manifest = 0, 0, []
    fail_counter = Counter()
    skin_counter = Counter()
    feat_counter = Counter()
    arche_counter = Counter()
    s = seed0
    max_attempts = n * 8

    while accepted < n and attempts < max_attempts:
        attempts += 1
        seed = s + attempts
        skin = skins[attempts % len(skins)] if skins else None
        try:
            world = sample_world(seed, skin=skin, archetype=archetype)
        except Exception as e:
            fail_counter[f"sample_exc:{type(e).__name__}"] += 1
            continue
        world["ground_truth"]["_seed"] = seed
        if require_feature and require_feature not in world["ground_truth"]["_features"]:
            continue
        try:
            res = audit(world)
        except Exception as e:
            fail_counter[f"audit_exc:{type(e).__name__}"] += 1
            continue
        if not res["ok"]:
            for f in res["fails"]:
                fail_counter[f] += 1
            continue
        rec = to_record(world, res)
        fname = f"world_{world['world_id']}.json"
        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, default=_json_default)
        gi = rec["oracle"]["gold"]
        manifest.append({"file": fname, "world_id": world["world_id"],
                         "skin": world["domain"], "features": world["ground_truth"]["_features"],
                         "archetype": world["ground_truth"].get("_archetype"),
                         "depth": world["ground_truth"]["_depth"],
                         "gold_intervention": gi["intervention"],
                         "gold_utility": round(gi["expected_utility"], 2)})
        skin_counter[world["domain"]] += 1
        arche_counter[world["ground_truth"].get("_archetype")] += 1
        for ft in world["ground_truth"]["_features"]:
            feat_counter[ft] += 1
        accepted += 1
        print(f"  [ok] {fname[:60]:60s} gold_util={gi['expected_utility']:.0f}")

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "n": accepted, "seed0": seed0,
                   "worlds": manifest,
                   "skin_distribution": dict(skin_counter),
                   "archetype_distribution": dict(arche_counter),
                   "feature_distribution": dict(feat_counter),
                   "rejected_by_gate": dict(fail_counter)}, f, indent=2)
    print(f"\nGenerated {accepted}/{n} worlds in {attempts} attempts "
          f"(acceptance {accepted/max(1,attempts):.0%}).")
    print(f"skins: {dict(skin_counter)}")
    print(f"archetypes: {dict(arche_counter)}")
    print(f"features: {dict(feat_counter)}")
    print(f"rejected by gate: {dict(fail_counter)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--skins", nargs="*", default=skin_names())
    ap.add_argument("--require-feature", default=None, choices=FEATURES)
    ap.add_argument("--archetype", default=None, choices=ARCHETYPES,
                    help="restrict to one structural archetype (default: sample both)")
    args = ap.parse_args()
    generate(args.outdir, args.n, args.seed, args.skins, args.require_feature, args.archetype)


if __name__ == "__main__":
    main()
