#!/usr/bin/env python3
"""Generate a batch of RPG v6 worlds from the topology templates.

Per world:
  1. pick a topology template (bioreactor / datacenter / ...);
  2. jitter mechanism params by seed (so worlds are not clones);
  3. shuffle the neutral (Regimen*-style) actuator labels where the template
     marks them shufflable, so no positional shortcut survives a batch;
  4. calibrate to difficulty bands, compute gold + counterfactual battery;
  5. run the full audit suite INCLUDING counterintuitiveness;
  6. emit only worlds that pass every audit as a self-contained JSON.

Emitted JSON is loadable by run_agent_v6.py via --world-file.

Usage:
    python generate_v6.py --outdir out_v6_batch --n 10 --seed 20260804
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from engine import WorldSCM
from worlds_v6 import ALL_WORLDS_V6
from oracle_v6 import (calibrate, optimal_intervention, counterfactual_battery,
                       decoy_audit, proxy_signal_audit, distractor_inertness_audit,
                       gold_selfconsistency_audit, counterintuitiveness_audit)

SCHEMA_VERSION = "rpg_scm_v6"

# neutral actuator id pool for label shuffling (only applied to actuators whose
# id is already generic, i.e. not a physically-descriptive control)
NEUTRAL_POOL = [f"regimen_{c}" for c in "abcdefgh"]


def _jitter_scm(scm: WorldSCM, rng: random.Random) -> None:
    """Perturb safe mechanism params in place so worlds are not identical."""
    for name, spec in scm.variables.items():
        mech = spec.get("mech", {})
        form = mech.get("form")
        if form == "hill":
            mech["k"] = round(mech["k"] * rng.uniform(0.85, 1.15), 2)
            mech["vmax"] = round(mech["vmax"] * rng.uniform(0.9, 1.1), 2)
        elif form == "interaction":
            mech["gain"] = round(mech["gain"] * rng.uniform(0.85, 1.15), 2)


def _serialize_world(world: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in world.items() if k != "scm"}
    out["scm"] = world["scm"].to_dict()
    return out


def build_and_audit(template_name: str, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Instantiate one world from a template, jitter, calibrate, audit."""
    rng = random.Random(seed)
    world = ALL_WORLDS_V6[template_name]()
    _jitter_scm(world["scm"], rng)
    world["world_id"] = f"{world['world_id']}_{seed}"

    calib = calibrate(world)
    gold = optimal_intervention(world["scm"])
    battery = counterfactual_battery(world)
    audits = {
        "decoy": decoy_audit(world),
        "proxy_signal": proxy_signal_audit(world),
        "distractor_inertness": distractor_inertness_audit(world, gold),
        "gold_selfconsistency": gold_selfconsistency_audit(world, gold),
        "counterintuitiveness": counterintuitiveness_audit(world, gold),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "world_id": world["world_id"],
        "domain": world["domain"],
        "template": template_name,
        "meta": {"seed": seed},
        "scenario": world["scenario"],
        "scm": world["scm"].to_dict(),
        "ground_truth": world["ground_truth"],
        "oracle": {"gold": gold, "counterfactual_battery": battery, "calibration": calib,
                   "audits": audits},
    }
    return record, audits


def audits_pass(audits: Dict[str, Any]) -> Tuple[bool, List[str]]:
    fails = []
    if not audits["decoy"]["passed"]:
        fails.append("decoy")
    if not audits["proxy_signal"]["passed"]:
        fails.append("proxy_signal")
    if not audits["distractor_inertness"]["passed"]:
        fails.append("distractor_inertness")
    if not audits["gold_selfconsistency"]["passed"]:
        fails.append("gold_selfconsistency")
    if not audits["counterintuitiveness"]["passed"]:
        fails.append("counterintuitiveness")
    return (len(fails) == 0, fails)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def generate(outdir: str, n: int, seed: int, templates: List[str]) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, accepted, attempts, idx = [], 0, 0, 0
    max_attempts = n * 6

    while accepted < n and attempts < max_attempts:
        attempts += 1
        template = templates[idx % len(templates)]
        wseed = seed + attempts * 101
        try:
            record, audits = build_and_audit(template, wseed)
        except Exception as e:
            print(f"  [err]  {template} seed={wseed}: {e}")
            continue
        ok, fails = audits_pass(audits)
        if not ok:
            print(f"  [skip] {record['world_id']}: failed {fails}")
            continue
        fname = f"world_{record['world_id']}.json"
        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=_json_default)
        gi = record["oracle"]["gold"]
        ci = record["oracle"]["audits"]["counterintuitiveness"]
        worst_naive = max((r["gain_over_baseline"] for r in ci["naive_results"]), default=0)
        manifest.append({
            "file": fname, "template": template, "world_id": record["world_id"],
            "gold_intervention": gi["intervention"],
            "gold_utility": round(gi["expected_utility"], 2),
            "baseline_utility": round(gi["baseline_utility"], 2),
            "worst_naive_gain": round(worst_naive, 2),
        })
        accepted += 1
        idx += 1
        print(f"  [ok]   {fname}  gold={gi['intervention']} "
              f"(base {gi['baseline_utility']:.1f}->{gi['expected_utility']:.1f}, "
              f"naive_gain {worst_naive:+.1f})")

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "n": accepted, "seed": seed,
                   "templates": templates, "worlds": manifest}, f, indent=2)
    print(f"\nGenerated {accepted}/{n} worlds ({attempts} attempts) into {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--templates", nargs="*", default=list(ALL_WORLDS_V6),
                    help="topology templates to draw from")
    args = ap.parse_args()
    generate(args.outdir, args.n, args.seed, args.templates)


if __name__ == "__main__":
    main()
