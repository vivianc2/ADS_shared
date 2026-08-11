#!/usr/bin/env python3
"""Phase-0 pre-RL harness checks for the world stream + env under parallelism.

Covers the pre-RL checklist items that were still [todo]:

  V12  on-demand world stream sustains acceptance and never collides world_ids
  V13  train/heldout split has ZERO leakage (skin/archetype reservation holds)
  V11  every episode terminates and emits a terminal reward (forced at turn cap)
  V10  rollouts are process- AND thread-safe: parallel grades == serial grades
  (also) end-to-end RPGEnv.reset()/step() smoke: a scripted GOLD policy -> reward ~1.0

The parallel workers regenerate each world deterministically from (seed, skin,
archetype) and replay the gold policy, so this also re-confirms determinism (V8):
the same world must grade identically no matter where/when it runs.

Run:  PYTHONPATH=../rpg_v7_prototype python test_stream_parallel.py
"""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from sampler import sample_world
from generate_v7 import audit
from catalog import build_catalog
from env import RPGEnv
from reward import RewardConfig
from splits import split_of, HELDOUT_SKINS, HELDOUT_ARCHETYPES
from world_stream import WorldStream

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


# ---------------------------------------------------------------------------
# scripted policies (turn the env into a deterministic, no-LLM test bed)
# ---------------------------------------------------------------------------

def _gold_answer_struct(world, gold, battery, cat):
    """The exact correct answer in ID space (same construction as test_env_reward)."""
    struct = {"actions": [], "signs": {}}
    for name, val in gold["intervention"].items():
        if isinstance(val, dict):
            continue
        aid = cat.actuator_id(name)
        if aid:
            struct["actions"].append({"actuator": aid, "value": val})
    if gold.get("is_conditional_policy") and gold.get("policy"):
        sp = world["ground_truth"]["subtype_policy"]
        gp = gold["policy"]
        struct["policy"] = {"treatment": cat.actuator_id(sp["treatment_actuator"]),
                            "stratifier": cat.measurable_id(sp["marker"]),
                            "threshold": gp["threshold"], "dose_if_ge": gp["dose_if_ge"],
                            "dose_if_lt": gp["dose_if_lt"]}
    struct["proxy"] = cat.measurable_id(battery["true_mechanism_proxy"])
    struct["decoys"] = [cat.measurable_id(d) for d in battery["confounded_decoys"]
                        if cat.measurable_id(d)]
    for aid_name, s in battery["actuator_sign_predictions"].items():
        if s in ("+", "-"):
            cid = cat.actuator_id(aid_name)
            if cid:
                struct["signs"][cid] = s
    return struct


def _answer_text(struct):
    return (f"<reasoning>gold</reasoning>\n"
            f'<action type="answer">{json.dumps(struct)}</action>\n'
            f"<memory>done</memory>")


def _measure_text():
    return ('<reasoning>look</reasoning>\n'
            '<action type="measure">{"ids": ["m0"]}</action>\n'
            "<memory>n</memory>")


def rollout_gold(seed, skin, arche, max_turns=32, budget=15):
    """Regenerate world deterministically, run the gold policy for one turn, return
    the terminal reward + accept flag. Top-level (picklable) so it runs in a worker."""
    world = sample_world(seed, skin=skin, archetype=arche)
    world["ground_truth"]["_seed"] = seed
    res = audit(world)
    if not res["ok"]:
        return {"world_id": world["world_id"], "audited": False}
    gold, battery = res["gold"], res["battery"]
    cat = build_catalog(world, world["scm"], seed=seed)
    env = RPGEnv(world=world, gold=gold, battery=battery, catalog_seed=seed,
                 max_turns=max_turns, budget=budget, reward_cfg=RewardConfig())
    env.reset()
    struct = _gold_answer_struct(world, gold, battery, cat)
    _obs, reward, done, info = env.step(_answer_text(struct))
    return {"world_id": world["world_id"], "audited": True, "reward": round(reward, 6),
            "accepted": bool(info.get("accepted")), "part_a": round(info["part_a"], 6),
            "part_b": round(info["part_b"], 6), "done": done}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_split_leakage():
    """V12 acceptance + no world_id collision; V13 zero split leakage."""
    print("\n[V12/V13] world stream + split leakage")
    tr = WorldStream(split="train", seed0=2_000_000)
    ho = WorldStream(split="heldout", seed0=3_000_000)
    tr_b = tr.take(30)
    ho_b = ho.take(30)

    check(tr.acceptance() > 0.5, f"train acceptance > 0.5 (got {tr.acceptance():.0%})")
    check(ho.acceptance() > 0.5, f"heldout acceptance > 0.5 (got {ho.acceptance():.0%})")

    # every train world is train-routed and touches NO reserved skin/archetype
    tr_ok = all(split_of(b.skin, b.archetype) == "train"
                and b.skin not in HELDOUT_SKINS
                and b.archetype not in HELDOUT_ARCHETYPES for b in tr_b)
    check(tr_ok, "every train world routes 'train' and uses no reserved skin/archetype")

    # every heldout world is heldout-routed (its skin OR archetype is reserved)
    ho_ok = all(split_of(b.skin, b.archetype) == "heldout"
                and (b.skin in HELDOUT_SKINS or b.archetype in HELDOUT_ARCHETYPES)
                for b in ho_b)
    check(ho_ok, "every heldout world routes 'heldout' (skin or archetype reserved)")

    # no world_id shared within or across the two streams
    ids = [b.world.get("world_id") for b in tr_b + ho_b]
    check(len(ids) == len(set(ids)), f"no world_id collisions across streams ({len(set(ids))}/{len(ids)} unique)")


def test_end_to_end_and_termination():
    """End-to-end reset/step with a gold policy (reward ~1.0); V11 forced termination."""
    print("\n[E2E + V11] gold rollout reaches reward~1.0; never-answer force-terminates")
    tr = WorldStream(split="train", seed0=4_000_000)
    for b in tr.take(4):
        r = rollout_gold(b.seed, b.skin, b.archetype)
        check(r["audited"] and r["reward"] >= 0.9 and r["accepted"],
              f"[{b.skin}/{b.archetype}/{b.seed}] gold rollout reward>=0.9 & accepted "
              f"(r={r.get('reward')}, A={r.get('part_a')}, B={r.get('part_b')})")

    # V11: a policy that never answers must be force-terminated at the turn cap,
    # with exactly one terminal reward emitted and the 'forced' flag set.
    b = tr.take(1)[0]
    env = b.make_env(max_turns=6, budget=15)
    env.reset()
    done, steps, term = False, 0, None
    while not done and steps < 20:
        _obs, reward, done, info = env.step(_measure_text())
        steps += 1
        if done:
            term = info
    check(done and steps <= 6, f"never-answer episode terminates by turn cap (steps={steps})")
    check(term is not None and term.get("turn_type") == "terminal" and term.get("forced") is True,
          f"terminal emitted with forced=True (info={ {k: term.get(k) for k in ('turn_type','forced')} if term else None})")
    check(term is not None and "reward" in term,
          "forced terminal still emits a scalar reward")


def test_parallel_matches_serial():
    """V10: process- and thread-parallel rollouts grade identically to serial (also V8)."""
    print("\n[V10] parallel (process + thread) grades == serial grades")
    tr = WorldStream(split="train", seed0=5_000_000)
    jobs = [(b.seed, b.skin, b.archetype) for b in tr.take(8)]

    serial = [rollout_gold(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=8) as ex:
        proc = list(ex.map(_star, jobs))
    with ThreadPoolExecutor(max_workers=8) as ex:
        thr = list(ex.map(_star, jobs))

    def key(rs):
        return [(r["world_id"], r.get("reward"), r.get("accepted"),
                 r.get("part_a"), r.get("part_b")) for r in rs]

    check(key(serial) == key(proc), "process-parallel grades identical to serial")
    check(key(serial) == key(thr), "thread-parallel grades identical to serial")
    check(all(r["audited"] and r["accepted"] for r in serial),
          "all 8 gold rollouts audited & accepted (sanity)")


def _star(job):
    return rollout_gold(*job)


def run():
    test_split_leakage()
    test_end_to_end_and_termination()
    test_parallel_matches_serial()
    print(f"\n{'ALL PHASE-0 HARNESS CHECKS PASSED' if not failures else f'{len(failures)} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
