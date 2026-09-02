#!/usr/bin/env python3
"""End-to-end trust test for the RL env + reward (id-space, no resolver/LLM).

Proves the reward path is correct by having a GOLD PLAYER construct the exact
correct answer in ID SPACE (from the world's stored gold+battery, mapped to catalog
ids) and confirming the env returns reward ~1.0 — i.e. the id->canonical->grade()
path is faithful. Then confirms degenerate answers score ~0 (master-key) and that a
gold-equivalent answer beats a fix-only answer (variance / part-B actually bites).

Run:  PYTHONPATH=../rpg_v9 python test_env_reward.py
"""

from __future__ import annotations

import json
import sys
import traceback

from sampler import sample_world
from generate_v7 import audit
from catalog import build_catalog
from env import RPGEnv
from reward import compute_reward, RewardConfig

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def _gold_answer_ids(world, gold, battery, cat):
    """Build the correct answer in ID space from stored gold+battery."""
    struct = {"actions": [], "signs": {}}
    # scalar actions (skip the per-unit policy entry)
    for name, val in gold["intervention"].items():
        if isinstance(val, dict):
            continue
        aid = cat.actuator_id(name)
        if aid:
            struct["actions"].append({"actuator": aid, "value": val})
    # conditional policy for subtype worlds
    if gold.get("is_conditional_policy") and gold.get("policy"):
        sp = world["ground_truth"]["subtype_policy"]
        gp = gold["policy"]
        struct["policy"] = {
            "treatment": cat.actuator_id(sp["treatment_actuator"]),
            "stratifier": cat.measurable_id(sp["marker"]),
            "threshold": gp["threshold"], "dose_if_ge": gp["dose_if_ge"],
            "dose_if_lt": gp["dose_if_lt"]}
    # proxy + decoys + signs (canonical -> id)
    struct["proxy"] = cat.measurable_id(battery["true_mechanism_proxy"])
    struct["decoys"] = [cat.measurable_id(d) for d in battery["confounded_decoys"]
                        if cat.measurable_id(d)]
    for aid_name, s in battery["actuator_sign_predictions"].items():
        if s in ("+", "-"):
            cid = cat.actuator_id(aid_name)
            if cid:
                struct["signs"][cid] = s
    return struct


CASES = [
    (100020, "bioprocess", "confounded_chain"),
    (700001, "clinical", "collider_selection"),
    (800001, "clinical", "hidden_subtype"),
    (222, "catalysis", "confounded_chain"),
    (444, "agronomy", "hidden_subtype"),
]


def run():
    n = 0
    for seed, skin, arch in CASES:
        w = None
        for s in range(seed, seed + 15):
            cand = sample_world(s, skin=skin, archetype=arch)
            res = audit(cand)
            if res["ok"]:
                w, gold, battery = cand, res["gold"], res["battery"]
                seed = s
                break
        if w is None:
            print(f"[skip] no audited {arch}/{skin}")
            continue
        n += 1
        tag = f"{skin}/{arch}/{seed}"
        cat = build_catalog(w, w["scm"], seed=seed)

        # ---- 1. GOLD answer in id-space -> reward ~1.0 (reward path is faithful) ----
        gstruct = _gold_answer_ids(w, gold, battery, cat)
        rg = compute_reward(gstruct, w, cat, gold, battery)
        check(rg["reward"] >= 0.9,
              f"[{tag}] gold id-answer reward>=0.9 (got {rg['reward']:.2f}; A={rg['part_a']:.2f} B={rg['part_b']:.2f})")
        check(rg["invalid_id_fraction"] == 0.0,
              f"[{tag}] gold id-answer has no invalid ids ({rg['invalid_id_fraction']:.2f})")

        # ---- 2. degenerate answers -> ~0 (master-key) ----
        rempty = compute_reward({}, w, cat, gold, battery)
        check(rempty["reward"] <= 0.25,
              f"[{tag}] empty answer reward<=0.25 (got {rempty['reward']:.2f})")
        # invalid ids only
        rbad = compute_reward({"actions": [{"actuator": "a999", "value": 50}],
                               "proxy": "m999", "decoys": ["m998"]}, w, cat, gold, battery)
        check(rbad["reward"] < rg["reward"] and rbad["part_b"] == 0.0,
              f"[{tag}] all-invalid-id answer scores low (r={rbad['reward']:.2f}, B={rbad['part_b']:.2f})")

        # ---- 3. gold beats fix-only (part B actually contributes -> variance) ----
        fix_only = {"actions": gstruct["actions"]}
        if gstruct.get("policy"):
            fix_only["policy"] = gstruct["policy"]
        rfix = compute_reward(fix_only, w, cat, gold, battery)
        check(rg["reward"] > rfix["reward"] + 0.05,
              f"[{tag}] full gold ({rg['reward']:.2f}) > fix-only ({rfix['reward']:.2f}) — part B bites")

    print(f"\ntested {n} worlds; {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
