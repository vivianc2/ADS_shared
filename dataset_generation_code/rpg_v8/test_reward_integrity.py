#!/usr/bin/env python3
"""Reward-integrity tests — the guard rail before this grader becomes an RL reward.

Two directions, both essential:

  MASTER-KEY (no false positives): trivial / degenerate / adversarial answers must
  score ~0. If a generic opener, an empty answer, an all-decoy label, or a
  single-knob answer to a conjunction world scores high, the reward is hackable and
  an RL policy WILL find it (cf. "One Token to Fool LLM-as-a-Judge", and our own
  two_cause oracle bug).

  ARTICULATE-CORRECT (no false negatives): a correct answer phrased VERBOSELY — the
  way a strong model actually writes ("LDH (lactate dehydrogenase release from cell
  lysis)") — must score high. The mixed9 run failed 0/9 because the grader punished
  articulate-correct answers; a reward with that bug trains terseness, not reasoning.

Run:  python test_reward_integrity.py
Exits non-zero if any assertion fails.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, List

import numpy as np

from sampler import sample_world
from generate_v7 import audit
from engine import WorldSCM
from sim_v6 import SimV6
from run_agent_v6 import (_resolve_answer_intervention, _resolve_answer_policy,
                          _translate_structured)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build(seed: int, skin: str, archetype: str):
    """Sample + audit a world; return (sim, gold, battery) or None if it didn't
    pass the audit gate (we only test on worlds that would enter training). The
    SimV6 gives us the SAME translate->grade path the runner/reward uses, so the
    test exercises free-text resolution, not just the bare grader."""
    w = sample_world(seed, skin=skin, archetype=archetype)
    res = audit(w)
    if not res["ok"]:
        return None
    sim = SimV6(w, resolver_llm=None, precomputed={"gold": res["gold"], "battery": res["battery"]})
    return w, sim, res["gold"], res["battery"]


def _alias(scm: WorldSCM, name: str) -> str:
    """First alias of a variable OR actuator (whichever holds the name)."""
    if name in scm.variables:
        return scm.variables[name].get("aliases", [name])[0]
    if name in scm.actuators:
        return scm.actuators[name].get("aliases", [name])[0]
    return name


def _grade(sim: SimV6, answer_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Full reward path: resolve free-text answer -> translate -> grade. This is
    exactly what run_agent_v6 (and the future RL reward) does, so the test scores
    answers the way training will."""
    iv, _ = _resolve_answer_intervention(sim, answer_raw.get("recommended_intervention_text", []))
    pol = _resolve_answer_policy(sim, answer_raw.get("recommended_policy_text"))
    answer = {"recommended_intervention": iv,
              "structured": _translate_structured(sim, answer_raw.get("structured", {})),
              "explanation": answer_raw.get("explanation", "")}
    if pol and not pol.get("_unresolved"):
        answer["recommended_policy"] = pol
    return sim.grade(answer)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

CASES = [
    # (seed, skin, archetype) — a spread across archetypes/skins
    (300020 - 300000 + 100000, "bioprocess", "confounded_chain"),
    (700001, "clinical", "collider_selection"),
    (800001, "clinical", "hidden_subtype"),
    (222, "catalysis", "confounded_chain"),
    (333, "battery", "collider_selection"),
    (444, "agronomy", "hidden_subtype"),
    (800001, "bioprocess", "surrogate_trap"),
    (700001, "datacenter", "surrogate_trap"),
    (610001, "bioprocess", "instrument_only"),
    (610004, "agronomy", "instrument_only"),
    (560001, "datacenter", "competing_causes"),
    (560006, "aquaculture", "competing_causes"),
    (481001, "bioprocess", "dose_window"),
    (492001, "datacenter", "confounded_reversal"),
    (470001, "datacenter", "synergy_pair"),
    (470005, "semiconductor", "synergy_pair"),
]

failures: List[str] = []


def check(cond: bool, msg: str):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


def run():
    n_worlds = 0
    for seed, skin, arch in CASES:
        built = _build(seed, skin, arch)
        if built is None:
            # try a couple of nearby seeds so a rejected draw doesn't blank the test
            for s2 in range(seed + 1, seed + 12):
                built = _build(s2, skin, arch)
                if built:
                    seed = s2
                    break
        if built is None:
            print(f"[skip] no audited {arch}/{skin} world near seed {seed}")
            continue
        world, sim, gold, battery = built
        scm: WorldSCM = world["scm"]
        n_worlds += 1
        tag = f"{skin}/{arch}/{seed}"
        print(f"\n=== {tag} (gold util {gold['expected_utility']:.1f}, base "
              f"{gold.get('baseline_utility', float('nan')):.1f}) ===")

        # ---- MASTER-KEY: degenerate answers must score ~0 ----
        empty = {"recommended_intervention_text": [], "structured": {}}
        g = _grade(sim, empty)
        check(not g["accepted"] and (g.get("benefit_recovered") or 0) < 0.5,
              f"[{tag}] empty answer rejected (benefit={g.get('benefit_recovered')}, B={g['battery_fraction']})")

        generic = {"recommended_intervention_text": [],
                   "structured": {"true_mechanism_proxy": "the main driver",
                                  "confounded_decoys": ["everything"],
                                  "actuator_sign_predictions": {}},
                   "explanation": "Let's think step by step. The system is complex."}
        g = _grade(sim, generic)
        check(not g["accepted"], f"[{tag}] generic 'let's think' answer rejected")

        # all-decoy: label the true proxy AS a decoy, name nothing useful
        proxy = battery["true_mechanism_proxy"]
        all_decoy = {"recommended_intervention_text": [],
                     "structured": {"true_mechanism_proxy": "unknown",
                                    "confounded_decoys": [_alias(scm, proxy)]}}
        g = _grade(sim, all_decoy)
        check(not g["part_b_battery_ok"],
              f"[{tag}] answer that calls the true proxy a decoy fails part B")

        # single-knob on a two_cause world must fail part A (the conjunction bug guard)
        if world["ground_truth"].get("co_actuators"):
            co = world["ground_truth"]["co_actuators"]
            one = {"recommended_intervention_text": [{"request": _alias(scm, co[0]), "value": 100}],
                   "structured": {}}
            g = _grade(sim, one)
            check(not g["part_a_utility_ok"],
                  f"[{tag}] single knob on a two_cause world fails part A "
                  f"(benefit={g.get('benefit_recovered')})")

        # ---- SURROGATE-ENDPOINT TRAP: the headline invariant of the redesign ----
        # In a surrogate_trap world the trap actuator adds a REAL structural edge to the
        # OBSERVED surrogate S (= scm.outcome) but has ZERO path to the LATENT goal G
        # (scm.goal, what utility() scores). So dosing the trap must: (1) move S, (2) leave
        # utility(G) unchanged, (3) leave the mechanism proxy P flat, and therefore (4) earn
        # ~0 reward. This is the entire point of the surrogate-endpoint redesign (it replaces
        # the old op="mask" reading-bias model); assert it hard so no future edit silently
        # reconnects the trap to the goal and reopens the Goodhart shortcut.
        if "symptom_trap" in world["ground_truth"].get("_features", []):
            trap_id = world["ground_truth"]["symptom_trap_actuator"]
            proxy_name = battery["true_mechanism_proxy"]
            N = 6000
            base = scm.sample(N, seed=seed + 4242)
            dosed = scm.sample(N, intervention={trap_id: 100}, seed=seed + 4242)
            dS = float(np.mean(dosed[scm.outcome])) - float(np.mean(base[scm.outcome]))
            dU = float(np.mean(scm.utility(dosed))) - float(np.mean(scm.utility(base)))
            dP = float(np.mean(dosed[proxy_name])) - float(np.mean(base[proxy_name]))
            check(abs(dS) > 1.0, f"[{tag}] trap moves the observed surrogate S (dS={dS:+.2f})")
            check(abs(dU) < 0.05, f"[{tag}] trap leaves utility(latent goal G) unchanged (dU={dU:+.3f})")
            check(abs(dP) < 0.05, f"[{tag}] trap leaves mechanism proxy P flat (dP={dP:+.3f})")
            trap_answer = {"recommended_intervention_text": [{"request": _alias(scm, trap_id), "value": 100}],
                           "structured": {}}
            g = _grade(sim, trap_answer)
            check(not g["part_a_utility_ok"] and (g.get("benefit_recovered") or 0) < 0.1,
                  f"[{tag}] dosing the trap earns ~0 reward "
                  f"(part_a={g['part_a_utility_ok']}, benefit={g.get('benefit_recovered')})")

        # ---- ARTICULATE-CORRECT: a verbose but correct answer must score high ----
        # Gold answer phrased the way a strong model writes: verbose proxy (acronym +
        # gloss), actions + signs by alias. Goes through the real translate->grade path.
        gold_iv = gold["intervention"]
        rec_text = [{"request": _alias(scm, k), "value": v}
                    for k, v in gold_iv.items() if not isinstance(v, dict)]
        proxy_alias = _alias(scm, proxy)
        verbose_proxy = f"{proxy_alias} (the key downstream marker of the true mechanism)"
        signs = {}
        for aid in gold.get("active_actuators", []):
            s = battery["actuator_sign_predictions"].get(aid, "0")
            if s in ("+", "-"):
                signs[_alias(scm, aid)] = s
        decoy_aliases = [_alias(scm, d) for d in battery["confounded_decoys"]]
        articulate = {
            "recommended_intervention_text": rec_text,
            "structured": {"true_mechanism_proxy": verbose_proxy,
                           "confounded_decoys": decoy_aliases,
                           "actuator_sign_predictions": signs},
        }
        if gold.get("is_conditional_policy") and gold.get("policy"):
            sp = world["ground_truth"]["subtype_policy"]
            gp = gold["policy"]
            articulate["recommended_policy_text"] = {
                "treatment": _alias(scm, sp["treatment_actuator"]),
                "stratifier": _alias(scm, sp["marker"]),
                "threshold": gp["threshold"], "dose_if_ge": gp["dose_if_ge"],
                "dose_if_lt": gp["dose_if_lt"]}
        g = _grade(sim, articulate)
        prox_ok = dict(g["battery_items"]).get("true_mechanism_proxy", False)
        check(prox_ok, f"[{tag}] verbose-but-correct proxy is credited (not punished for being articulate)")
        check((g.get("benefit_recovered") or 0) >= 0.85,
              f"[{tag}] gold-equivalent answer recovers benefit (got {g.get('benefit_recovered')})")

        # ---- V3: reward has within-group VARIANCE on a realistic quality spread ----
        # GRPO needs the group's rewards to differ, else advantage=0 -> no gradient.
        # Build 8 escalating-quality answers and confirm reward std > 0. (A SATURATED
        # group — all identical quality — correctly gives std 0; that case is handled
        # by DAPO dynamic sampling in the trainer, not the reward. See V3 in
        # rpg_v7_reward_contract_decisions.md.)
        def _reward(answer_raw, wA=0.5, wB=0.5):
            gg = _grade(sim, answer_raw)
            return wA * max(0.0, gg.get("benefit_recovered") or 0.0) + wB * gg["battery_fraction"]
        ladder = [
            {"recommended_intervention_text": [], "structured": {}},                       # q0 empty
            {"recommended_intervention_text": rec_text, "structured": {}},                 # q1 fix only
            {"recommended_intervention_text": rec_text,
             "structured": {"true_mechanism_proxy": verbose_proxy}},                        # q2 +proxy
            {"recommended_intervention_text": rec_text,
             "structured": {"true_mechanism_proxy": verbose_proxy,
                            "confounded_decoys": decoy_aliases}},                           # q3 +decoys
        ]
        if "recommended_policy_text" in articulate:
            for a in ladder[1:]:
                a["recommended_policy_text"] = articulate["recommended_policy_text"]
        group = [_reward(ladder[i % len(ladder)]) for i in range(8)]
        std = float(np.std(group))
        check(std > 0.05, f"[{tag}] realistic mixed-quality group has reward spread (std={std:.3f})")

    print(f"\n{'='*60}")
    print(f"tested {n_worlds} worlds; {len(failures)} assertion(s) failed")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL REWARD-INTEGRITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
