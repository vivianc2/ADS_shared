#!/usr/bin/env python3
"""Generate RPG v5 SCM worlds for a trial run.

Pipeline per world (doc `worldgen_rpg_plan_v5_scm_chain.md` §7):

1. pick a domain template + jitter its mechanism params by seed;
2. shuffle the *neutral* action labels so no positional shortcut survives across
   a batch (the targeted knob is not always "RegimenC");
3. auto-calibrate to the difficulty bands;
4. compute the golden answer (MC + golden-section) and counterfactual battery;
5. run faithfulness + solvability audits;
6. emit only worlds that pass every audit, as a JSON with a clean
   ``visible`` / ``hidden`` / ``oracle`` split.

Usage:
    python3 generate.py --outdir out_v5_trial --n 6 --seed 20260804
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from worlds import ALL_WORLDS, World
from scm import SCM
from oracle import (audit_world, calibrate_world, counterfactual_battery,
                    name_leakage_audit, decoy_audit, proxy_signal_audit,
                    optimal_intervention, solvability_certificate,
                    gold_optimality_audit)

SCHEMA_VERSION = "rpg_scm_v5"

NEUTRAL_POOL = [f"Regimen{c}" for c in "ABCDEFGH"]


# ---------------------------------------------------------------------------
# Relabeling: rename knobs consistently everywhere they appear.
# ---------------------------------------------------------------------------

def _rename_in_mech(mech: Dict[str, Any], mapping: Dict[str, str]) -> None:
    for key in ("of", "a", "b"):
        if key in mech and mech[key] in mapping:
            mech[key] = mapping[mech[key]]
    if "weights" in mech:
        mech["weights"] = {mapping.get(k, k): v for k, v in mech["weights"].items()}


def relabel_knobs(world: World, mapping: Dict[str, str]) -> World:
    """Return a copy of ``world`` with knobs renamed per ``mapping`` (old->new)."""
    w = World.from_dict(copy.deepcopy(world.to_dict()))
    scm = w.scm

    new_nodes: Dict[str, Dict[str, Any]] = {}
    for name, spec in scm.nodes.items():
        nn = mapping.get(name, name)
        spec = dict(spec)
        if "parents" in spec:
            spec["parents"] = [mapping.get(p, p) for p in spec["parents"]]
        if "mech" in spec:
            spec["mech"] = dict(spec["mech"])
            _rename_in_mech(spec["mech"], mapping)
        new_nodes[nn] = spec
    scm.nodes = new_nodes

    scm.knob_effects = {mapping.get(k, k): v for k, v in scm.knob_effects.items()}
    scm.obs_effects = {mapping.get(k, k): v for k, v in scm.obs_effects.items()}

    w.knobs = {mapping.get(k, k): v for k, v in w.knobs.items()}
    w.targeted_knob = mapping.get(w.targeted_knob, w.targeted_knob)
    w.symptom_trap_knob = mapping.get(w.symptom_trap_knob, w.symptom_trap_knob)
    return w


# ---------------------------------------------------------------------------
# Per-world jitter + neutral-label shuffle
# ---------------------------------------------------------------------------

def jitter_params(world: World, rng: random.Random) -> None:
    """Perturb a few safe mechanism params in place so worlds are not clones."""
    for name, spec in world.scm.nodes.items():
        mech = spec.get("mech", {})
        form = mech.get("form")
        if form == "hill":
            mech["k"] = round(mech["k"] * rng.uniform(0.85, 1.15), 2)
            mech["vmax"] = round(mech["vmax"] * rng.uniform(0.9, 1.1), 2)
        elif form == "interaction":
            mech["gain"] = round(mech["gain"] * rng.uniform(0.85, 1.15), 2)
        elif form == "sign_flip":
            mech["knee"] = round(mech["knee"] * rng.uniform(0.9, 1.1), 2)


def shuffle_neutral_labels(world: World, rng: random.Random) -> World:
    """Reassign the neutral (Regimen*) knob labels to random pool members so the
    targeted/trap knob is not in a fixed position across the batch."""
    neutral = [k for k in world.knobs if k.startswith("Regimen")]
    if not neutral:
        return world
    new_labels = rng.sample(NEUTRAL_POOL, len(neutral))
    mapping = dict(zip(neutral, new_labels))
    # avoid accidental identity collision with a non-neutral knob name
    return relabel_knobs(world, mapping)


# ---------------------------------------------------------------------------
# Emit a clean agent-facing / hidden / oracle split
# ---------------------------------------------------------------------------

# Neutral human-readable descriptions for observables (agent-facing). Kept
# generic on purpose: they describe what the sensor reads, never the mechanism.
DEFAULT_OBS_DESC = "a measured signal from the process; may be noisy and may be "\
                   "a downstream clue rather than a cause."


def build_visible(world: World) -> Dict[str, Any]:
    obs_catalog = []
    for m in world.observables:
        obs_catalog.append({"name": m, "description": DEFAULT_OBS_DESC})
    knob_catalog = []
    for name, spec in world.knobs.items():
        entry = {"name": name, "value_type": spec.get("dtype", "continuous")}
        if spec.get("dtype") == "continuous":
            entry["range"] = spec["range"]
        else:
            entry["values"] = spec.get("values", ["off", "on"])
        knob_catalog.append(entry)
    return {
        "story": world.story,
        "question": world.question,
        "answer_schema": "scm_latent_cause_v5",
        "observed_variables": obs_catalog,
        "action_variables": knob_catalog,
        "clampable_measurements": world.clampable,
        "allowed_query_modes": ["observational_sample", "interventional_sample",
                                 "sweep", "clamp"],
        "max_intervention_knobs": 3,
        "experiment_budget": {
            "max_queries": 12,
            "max_units_per_query": 500,
            "max_total_samples": 20000,
        },
        "outcome_name": world.scm.outcome,
        "outcome_direction": "higher_is_better" if world.scm.higher_is_better
                             else "lower_is_better",
    }


def build_world_record(world: World, *, seed: int, difficulty: str) -> Dict[str, Any]:
    calib = calibrate_world(world)
    gold = optimal_intervention(world)
    battery = counterfactual_battery(world)
    audits = {
        "name_leakage": name_leakage_audit(world),
        "decoy": decoy_audit(world),
        "proxy_signal": proxy_signal_audit(world),
        "gold_optimality": gold_optimality_audit(world, gold),
        "solvability": solvability_certificate(world, gold, battery),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "world_id": world.world_id,
        "domain": world.domain,
        "meta": {"seed": seed, "difficulty": difficulty},
        "visible": build_visible(world),
        "hidden": {
            "scm": world.scm.to_dict(),
            "true_root": world.true_root,
            "true_mechanism_proxy": world.true_mechanism_proxy,
            "confounded_decoys": world.confounded_decoys,
            "targeted_knob": world.targeted_knob,
            "symptom_trap_knob": world.symptom_trap_knob,
            "latent_plain_name": world.latent_plain_name,
            "notes": world.notes,
        },
        "oracle": {
            "gold_intervention": gold,
            "counterfactual_battery": battery,
            "calibration": calib,
            "audits": audits,
        },
    }, audits


def audits_pass(audits: Dict[str, Any]) -> Tuple[bool, List[str]]:
    fails = []
    if not audits["name_leakage"]["passed"]:
        fails.append("name_leakage")
    if not audits["decoy"]["passed"]:
        fails.append("decoy")
    if not audits["proxy_signal"]["passed"]:
        fails.append("proxy_signal")
    if not audits["gold_optimality"]["passed"]:
        fails.append("gold_optimality")
    if not audits["solvability"]["solvable"]:
        fails.append("solvability")
    return (len(fails) == 0, fails)


def generate(outdir: str, n: int, seed: int) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    domains = list(ALL_WORLDS)
    manifest = []
    accepted = 0
    attempts = 0
    idx = 0
    max_attempts = n * 5

    while accepted < n and attempts < max_attempts:
        attempts += 1
        rng = random.Random(seed + attempts * 101)
        domain = domains[idx % len(domains)]
        world = ALL_WORLDS[domain]()
        jitter_params(world, rng)
        world = shuffle_neutral_labels(world, rng)
        wseed = seed + attempts * 101
        world.world_id = f"{world.world_id}_{wseed}"
        record, audits = build_world_record(world, seed=wseed, difficulty="standard")
        ok, fails = audits_pass(audits)
        if not ok:
            print(f"  [skip] {world.world_id}: failed audits {fails}")
            continue
        fname = f"world_{world.world_id}.json"
        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=_json_default)
        gi = record["oracle"]["gold_intervention"]
        manifest.append({
            "file": fname,
            "domain": domain,
            "world_id": world.world_id,
            "gold_intervention": gi["intervention"],
            "gold_utility": round(gi["expected_utility"], 2),
            "baseline_utility": round(gi["baseline_utility"], 2),
            "targeted_knob": record["hidden"]["targeted_knob"],
            "symptom_trap_knob": record["hidden"]["symptom_trap_knob"],
            "min_queries": audits["solvability"]["n_queries"],
        })
        accepted += 1
        idx += 1
        print(f"  [ok]   {fname}  gold={gi['intervention']} "
              f"(util {record['oracle']['gold_intervention']['baseline_utility']:.1f}"
              f"->{gi['expected_utility']:.1f})")

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "n": accepted,
                   "seed": seed, "worlds": manifest}, f, indent=2)
    print(f"\nGenerated {accepted}/{n} worlds ({attempts} attempts) into {out}/")
    print(f"Manifest: {out}/manifest.json")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate RPG v5 SCM worlds.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()
    generate(args.outdir, args.n, args.seed)


if __name__ == "__main__":
    main()
