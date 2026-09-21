#!/usr/bin/env python3
"""Tests for the lever-identification reward gate (reward.py, 2026-09-21).

Proves, on freshly generated audited worlds across archetypes:
  1. gate OFF  -> reward is bit-identical to the pre-gate reward (HEAD version from git)
  2. gold answer -> lever_ok, reward 1.0 in every mode
  3. right lever + wrong dose -> gate: 0 < r < 1 ; lever_only: exactly 1.0
  4. wrong lever + PERFECT battery -> gate/lever_only: no credit (r <= 0) ; old: >= 0.5*w_b
  5. source knob alone (single-cause world) counts as identification (fix and src both causal)
  6. inert knob / surrogate trap alone -> lever_ok False
  7. competing/synergy: one co-cause -> mode=full False, mode=any True
  8. hidden_subtype (conditional gold): policy-only answer -> full True; chain-fix-only -> full False
  9. evidence gate still wins: gold answer with 0 interventions -> 0 in every mode
 10. reward is total on malformed answers in every mode

Run:  PYTHONPATH=../rpg_v9 python test_lever_gate.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types

from sampler import sample_world
from generate_v7 import audit
from catalog import build_catalog
from reward import compute_reward, RewardConfig, lever_sets
from test_env_reward import _gold_answer_ids

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def _load_old_reward():
    """The reward module as committed at HEAD (pre-gate), for the bit-exact regression."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = subprocess.check_output(
        ["git", "show", "HEAD:dataset_generation_code/rpg_rl/reward.py"],
        cwd=here, text=True)
    mod = types.ModuleType("reward_old")
    sys.modules["reward_old"] = mod           # dataclass resolves annotations via sys.modules
    exec(compile(src, "reward_old.py", "exec"), mod.__dict__)
    return mod


def _cfg(**kw):
    base = dict(lever_gate=False, lever_only=False, lever_mode="full", lever_bonus=0.0,
                w_a=0.5, w_b=0.5)
    base.update(kw)
    return RewardConfig(**base)


def _find_world(seed, skin, arch, tries=40):
    for s in range(seed, seed + tries):
        w = sample_world(s, skin=skin, archetype=arch)
        res = audit(w)
        if res["ok"]:
            return s, w, res
    raise RuntimeError(f"no audited world for {skin}/{arch} from seed {seed}")


CASES = [
    (100020, "bioprocess", "confounded_chain"),
    (700001, "clinical", "collider_selection"),
    (800001, "clinical", "hidden_subtype"),
    (300001, "agronomy", "competing_causes"),
    (400001, "datacenter", "synergy_pair"),
    (500001, "agronomy", "surrogate_trap"),
]


def run():
    old = _load_old_reward()
    for seed, skin, arch in CASES:
        s, w, res = _find_world(seed, skin, arch)
        gold, battery = res["gold"], res["battery"]
        cat = build_catalog(w, w["scm"], seed=s)
        tag = f"[{skin}/{arch}/{s}]"
        causal, must = lever_sets(w, gold)
        gt = w["ground_truth"]
        scm = w["scm"]
        print(f"{tag} causal={sorted(causal)} must={sorted(must)} targeted={gt['targeted_actuator']}")

        gold_ans = _gold_answer_ids(w, gold, battery, cat)

        # --- answers used throughout ---
        inert = [a for a in scm.actuators if a not in causal and a != gt["symptom_trap_actuator"]]
        inert_id = cat.actuator_id(inert[0])
        wrong_perfect_B = dict(gold_ans)            # perfect proxy/decoys/signs, wrong knob
        wrong_perfect_B["actions"] = [{"actuator": inert_id, "value": 100}]
        wrong_perfect_B.pop("policy", None)

        modes = {
            "old(HEAD)": lambda a, n=5: old.compute_reward(a, w, cat, gold, battery, old.RewardConfig(), n_interventions=n),
            "gate_off": lambda a, n=5: compute_reward(a, w, cat, gold, battery, _cfg(), n_interventions=n),
            "gate": lambda a, n=5: compute_reward(a, w, cat, gold, battery, _cfg(lever_gate=True), n_interventions=n),
            "lever_only": lambda a, n=5: compute_reward(a, w, cat, gold, battery, _cfg(lever_only=True), n_interventions=n),
            "gate_any": lambda a, n=5: compute_reward(a, w, cat, gold, battery, _cfg(lever_gate=True, lever_mode="any"), n_interventions=n),
        }

        # 1. gate OFF == old reward, bit-exact, on a ladder of answers
        ladder = [gold_ans, wrong_perfect_B, {"actions": []}, {"actions": [{"actuator": "zz9", "value": 1}]},
                  {"proxy": gold_ans.get("proxy"), "decoys": gold_ans.get("decoys")}]
        same = all(abs(modes["gate_off"](a)["reward"] - modes["old(HEAD)"](a)["reward"]) < 1e-12 for a in ladder)
        check(same, f"{tag} gate OFF is bit-identical to HEAD reward on a 5-answer ladder")

        # 2. gold answer -> lever_ok + 1.0 everywhere
        for m in ("gate", "lever_only", "gate_any"):
            r = modes[m](gold_ans)
            check(r["lever_ok"] and r["reward"] >= 0.9, f"{tag} gold answer [{m}] lever_ok & r>=0.9 (r={r['reward']:.2f})")

        # 3. right lever, wrong dose (only meaningful if gold has a scalar lever)
        scal = [x for x in gold_ans["actions"]]
        if scal:
            bad_dose = dict(gold_ans)
            bad_dose["actions"] = [{"actuator": x["actuator"], "value": (5.0 if float(x["value"]) > 50 else 95.0)} for x in scal]
            rg, rl = modes["gate"](bad_dose), modes["lever_only"](bad_dose)
            check(rg["lever_ok"] and 0.0 < rg["reward"] < 1.0,
                  f"{tag} right lever/wrong dose [gate] 0<r<1 (r={rg['reward']:.2f}, A={rg['part_a']:.2f})")
            check(abs(rl["reward"] - 1.0) < 1e-9, f"{tag} right lever/wrong dose [lever_only] r==1.0")

        # 4. wrong lever + perfect battery: old pays part B; gated pays nothing
        ro, rg, rl = modes["old(HEAD)"](wrong_perfect_B), modes["gate"](wrong_perfect_B), modes["lever_only"](wrong_perfect_B)
        check(ro["reward"] >= 0.5 * 0.5, f"{tag} wrong lever/perfect B [old] r>=0.25 (r={ro['reward']:.2f}, B={ro['part_b']:.2f})")
        check((not rg["lever_ok"]) and rg["lever_gated"] and rg["reward"] <= 0.0 and rg["part_b"] == 0.0,
              f"{tag} wrong lever/perfect B [gate] r<=0 & part_b zeroed (r={rg['reward']:.2f})")
        check(rl["reward"] <= 0.0, f"{tag} wrong lever/perfect B [lever_only] r<=0 (r={rl['reward']:.2f})")

        # 5/6. single-cause worlds: source knob alone identifies; inert / trap alone does not
        if not must:
            for lever in sorted(causal):
                a = {"actions": [{"actuator": cat.actuator_id(lever), "value": 0}]}
                check(modes["lever_only"](a)["lever_ok"], f"{tag} causal lever alone '{lever}' -> lever_ok")
            trap = gt["symptom_trap_actuator"]
            if trap not in causal:
                a = {"actions": [{"actuator": cat.actuator_id(trap), "value": 100}]}
                check(not modes["lever_only"](a)["lever_ok"], f"{tag} trap alone '{trap}' -> NOT lever_ok")
        a = {"actions": [{"actuator": inert_id, "value": 100}]}
        check(not modes["lever_only"](a)["lever_ok"], f"{tag} inert knob alone -> NOT lever_ok")

        # 7. two-cause worlds: one co-cause -> full False, any True; both -> full True
        co = gt.get("co_actuators")
        if co:
            one = {"actions": [{"actuator": cat.actuator_id(co[0]), "value": 100}]}
            both = {"actions": [{"actuator": cat.actuator_id(c), "value": 100} for c in co]}
            check(not modes["lever_only"](one)["lever_ok"], f"{tag} one of {len(co)} co-causes [full] -> NOT lever_ok")
            check(modes["gate_any"](one)["lever_ok"], f"{tag} one of {len(co)} co-causes [any] -> lever_ok")
            check(modes["lever_only"](both)["lever_ok"], f"{tag} both co-causes [full] -> lever_ok")

        # 8. subtype: policy-only vs chain-fix-only
        if gold.get("is_conditional_policy"):
            pol_only = {"actions": [], "policy": gold_ans["policy"]}
            chain_only = {"actions": list(gold_ans["actions"])}
            check(modes["lever_only"](pol_only)["lever_ok"], f"{tag} policy-only answer (treatment via policy) [full] -> lever_ok")
            if chain_only["actions"]:
                check(not modes["lever_only"](chain_only)["lever_ok"], f"{tag} chain-fix-only answer [full] -> NOT lever_ok (treatment missing)")
                check(modes["gate_any"](chain_only)["lever_ok"], f"{tag} chain-fix-only answer [any] -> lever_ok")

        # 9. evidence gate dominates
        for m in ("gate", "lever_only", "gate_any"):
            r = modes[m](gold_ans, 0)
            check(r["reward"] <= 0.0 and r["evidence_gated"], f"{tag} gold answer with 0 interventions [{m}] -> r<=0")

        # 10. total on garbage
        garbage = [None, [], {"actions": "x"}, {"actions": [{"actuator": 3, "value": {}}], "policy": {"treatment": ["a"]}},
                   {"policy": {"treatment": gold_ans["actions"][0]["actuator"] if gold_ans["actions"] else "a0", "stratifier": "m0",
                               "threshold": "nan", "dose_if_ge": True}}]
        ok = True
        for m in modes:
            for a in garbage:
                try:
                    modes[m](a)
                except Exception as e:  # noqa: BLE001
                    ok = False
                    print(f"     raised in {m} on {a!r}: {type(e).__name__}: {e}")
        check(ok, f"{tag} reward is total on 5 malformed answers x {len(modes)} modes")

    print(f"\n{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run())
