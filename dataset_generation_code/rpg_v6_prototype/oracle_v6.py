#!/usr/bin/env python3
"""RPG v6 oracle, grader, and audits — over an actuator-combination action space.

Golden search:
  1. screen every actuator's marginal effect at its extreme dose (CRN);
  2. keep the "active" set (|effect| > eps) -- distractors are pruned, which is
     provably safe (see distractor_inertness_audit);
  3. search combinations among active actuators up to size ``max_joint`` with
     golden-section refinement on continuous doses.

The counterfactual battery and grader mirror v5, generalized to actuators.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine import WorldSCM, actuator_dose


def expected_utility(scm: WorldSCM, intervention: Dict[str, Any], *, n: int, seed: int) -> float:
    vals = scm.sample(n, intervention=intervention, seed=seed)
    return float(np.mean(scm.utility(vals)))


def _golden_section(scm, act_id, other, a, b, *, n, seed, iters=22):
    gr = (np.sqrt(5) - 1) / 2

    def f(x):
        iv = dict(other); iv[act_id] = x
        return expected_utility(scm, iv, n=n, seed=seed)

    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def screen_actuators(scm: WorldSCM, *, n=15000, seed=12345, eps=1.0) -> Dict[str, Any]:
    base = expected_utility(scm, {}, n=n, seed=seed)
    marg = {}
    for aid, act in scm.actuators.items():
        if act.get("op") == "mask":
            hi = act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
            u = expected_utility(scm, {aid: hi}, n=n, seed=seed)  # mask doesn't change utility
            marg[aid] = u - base
            continue
        if act.get("dtype") == "continuous":
            lo, hi = act["range"]
            cand = [lo, (lo + hi) / 2, hi]
        else:
            cand = act.get("values", ["off", "on"])
        best = max(expected_utility(scm, {aid: v}, n=n, seed=seed) for v in cand)
        marg[aid] = best - base
    active = [aid for aid, m in marg.items() if abs(m) > eps]

    # Synergy pass: catch pairs where NEITHER actuator helps alone but TOGETHER
    # they do (two-interacting-causes topology). Without this, marginal-only
    # screening would prune both and the oracle would never find the gold combo.
    # Only check pairs of non-mask actuators not already active, at their extremes.
    non_mask = [aid for aid, act in scm.actuators.items() if act.get("op") != "mask"]
    def _extreme(aid):
        act = scm.actuators[aid]
        return act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
    inactive = [aid for aid in non_mask if aid not in active]
    for i in range(len(inactive)):
        for j in range(i + 1, len(inactive)):
            a, b = inactive[i], inactive[j]
            u = expected_utility(scm, {a: _extreme(a), b: _extreme(b)}, n=n, seed=seed)
            if (u - base) > eps:   # joint helps though neither did alone
                if a not in active:
                    active.append(a)
                if b not in active:
                    active.append(b)
    return {"baseline_utility": base, "marginal": marg, "active": active}


def optimal_intervention(scm: WorldSCM, *, n=15000, seed=12345, max_joint=3, coarse=7) -> Dict[str, Any]:
    scr = screen_actuators(scm, n=n, seed=seed)
    active = scr["active"]
    base = scr["baseline_utility"]
    best = {"intervention": {}, "expected_utility": base}

    # evaluate single + joint combos over active actuators
    for r in range(1, min(max_joint, len(active)) + 1):
        for combo in itertools.combinations(active, r):
            # coarse grid per continuous actuator; discrete values enumerated
            grids = []
            for aid in combo:
                act = scm.actuators[aid]
                if act.get("dtype") == "continuous":
                    lo, hi = act["range"]
                    grids.append([round(x, 2) for x in np.linspace(lo, hi, coarse)])
                else:
                    grids.append(act.get("values", ["off", "on"]))
            for point in itertools.product(*grids):
                iv = dict(zip(combo, point))
                u = expected_utility(scm, iv, n=n, seed=seed)
                if u > best["expected_utility"]:
                    best = {"intervention": iv, "expected_utility": u}

    # golden-section refine each continuous actuator in the winner
    refined = dict(best["intervention"])
    for aid in list(refined):
        act = scm.actuators[aid]
        if act.get("dtype") == "continuous":
            lo, hi = act["range"]
            other = {k: v for k, v in refined.items() if k != aid}
            x, u = _golden_section(scm, aid, other, lo, hi, n=n, seed=seed)
            refined[aid] = round(float(x), 2)
    best["intervention"] = refined
    best["expected_utility"] = expected_utility(scm, refined, n=n, seed=seed)
    best["baseline_utility"] = base
    best["active_actuators"] = active
    return best


def _sign(delta, eps):
    return "+" if delta > eps else ("-" if delta < -eps else "0")


def counterfactual_battery(world: Dict[str, Any], *, n=15000, seed=777) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    eps = 0.5
    base = expected_utility(scm, {}, n=n, seed=seed)
    act_signs = {}
    for aid, act in scm.actuators.items():
        if act.get("op") == "mask":
            act_signs[aid] = "0"  # no true-utility effect
            continue
        # Sign = effect on the outcome of INCREASING the actuator's setting
        # (the convention the agent is told, and that _translate_structured flips
        # for 'reduce/lower' phrasings). This is unambiguous for monotone
        # actuators: chelator '+', feed-flow '-', inert '0'.
        #
        # BUT a non-monotone actuator (interior optimum: helps at a moderate
        # dose, hurts at the extreme — e.g. a titrated drug) has no single
        # correct increasing-direction sign. Forcing one unfairly penalizes a
        # correct answer either way. We DETECT non-monotonicity via a dose sweep
        # and mark such actuators "skip" so the grader does not score their sign.
        if act.get("dtype") == "continuous":
            lo, hi = act["range"]
            grid = np.linspace(lo, hi, 7)
            us = [expected_utility(scm, {aid: float(v)}, n=n, seed=seed) for v in grid]
            best_i = int(np.argmax(us))
            extreme_gain = us[-1] - base
            interior_peak = (best_i not in (0, len(us) - 1)) and (max(us) - base > eps) \
                and (max(us) - us[-1] > eps)
            if interior_peak:
                act_signs[aid] = "skip"      # non-monotone -> not scored
            else:
                act_signs[aid] = _sign(extreme_gain, eps)
        else:
            top = act.get("values", ["off", "on"])[-1]
            act_signs[aid] = _sign(expected_utility(scm, {aid: top}, n=n, seed=seed) - base, eps)

    # Valid mechanism proxies = measurable signals that the TARGETED actuator (which
    # acts on the true cause) actually moves, but the confounder does not. Computed
    # empirically so ANY genuine proxy on the causal chain is credited, not one
    # hardcoded name (a world may expose several: e.g. interface errors AND a
    # dew-point sensor). Prevents marking a correct-but-alternative proxy wrong.
    tgt = gt["targeted_actuator"]
    tspec = scm.actuators[tgt]
    thi = tspec["range"][1] if tspec.get("dtype") == "continuous" else tspec.get("values", ["off", "on"])[-1]
    measurables = [nm for nm, s in scm.variables.items()
                   if (s["kind"] in ("observable",) or s.get("measurable")) and nm != scm.outcome]
    base_obs = scm.measure(scm.sample(n, seed=seed + 1), measurables, seed=seed + 2)
    do_obs = scm.measure(scm.sample(n, intervention={tgt: thi}, seed=seed + 1),
                         measurables, seed=seed + 2, intervention={tgt: thi})
    valid_proxies = []
    for nm in measurables:
        shift = abs(float(do_obs[nm].mean() - base_obs[nm].mean()))
        sd = float(base_obs[nm].std()) + 1e-9
        if shift / sd > 0.5:      # the true lever meaningfully moves this signal
            valid_proxies.append(nm)
    return {
        "true_mechanism_proxy": gt["true_mechanism_proxy"],
        "valid_mechanism_proxies": sorted(set(valid_proxies) | {gt["true_mechanism_proxy"]}),
        "confounded_decoys": sorted(gt["confounded_decoys"]),
        "actuator_sign_predictions": act_signs,
        "targeted_actuator": gt["targeted_actuator"],
        "symptom_trap_actuator": gt["symptom_trap_actuator"],
    }


def _score_battery(battery, answer):
    st = answer.get("structured", {}) or {}
    items = []
    # Accept ANY genuine proxy on the causal chain, not one hardcoded name.
    valid_proxies = set(battery.get("valid_mechanism_proxies", [battery["true_mechanism_proxy"]]))
    items.append(("true_mechanism_proxy", st.get("true_mechanism_proxy") in valid_proxies))
    agent_decoys = set(st.get("confounded_decoys", []))
    gt_decoys = set(battery["confounded_decoys"])
    # decoy check: must flag the true confounder(s), and must NOT mislabel any
    # genuine proxy as a decoy.
    items.append(("confounded_decoys",
                  gt_decoys.issubset(agent_decoys) and not (agent_decoys & valid_proxies)))
    pred = st.get("actuator_sign_predictions", {}) or {}
    trap = battery.get("symptom_trap_actuator")
    all_signs = battery["actuator_sign_predictions"]
    # only score actuators the agent could plausibly have identified; require the
    # targeted one and any the agent volunteered.
    scored = set([battery["targeted_actuator"]]) | set(pred)
    for aid in scored:
        gold = all_signs.get(aid, "0")
        # Non-monotone actuators (interior optimum) have no single correct
        # increasing-direction sign -> not scored, either party.
        if gold == "skip":
            continue
        if aid == trap:
            ok = pred[aid] in ("0", "+") if aid in pred else True
            items.append((f"sign:{aid}(trap)", ok))
        else:
            items.append((f"sign:{aid}", pred.get(aid) == gold))
    n_ok = sum(1 for _, ok in items if ok)
    return (n_ok / len(items), items) if items else (1.0, items)


def grade(world: Dict[str, Any], answer: Dict[str, Any], gold: Dict[str, Any],
          battery: Dict[str, Any], *, n=30000, seed=999,
          benefit_frac: float = 0.90, tolerance: float = 2.0) -> Dict[str, Any]:
    """Grade an answer. Part A (found the fix) is judged by the FRACTION of the
    achievable benefit recovered, not an absolute utility gap:

        benefit_recovered = (rec_util - baseline) / (gold_util - baseline)
        part_A = benefit_recovered >= benefit_frac   (default 0.90)

    Rationale: an absolute tolerance is not comparable across worlds whose
    utility RANGES differ (e.g. a bioreactor spans ~44 utility units, a clinic
    world only ~6). A fixed ±2.0 was ~5% of the range on wide worlds but ~35% on
    narrow ones — simultaneously too strict on some and too lenient on others.
    The fraction-of-benefit rule makes "found the fix" mean the same thing
    everywhere. A tiny absolute tolerance is still allowed so a numerically-tied
    optimum is not rejected on Monte-Carlo noise."""
    scm: WorldSCM = world["scm"]
    rec = answer.get("recommended_intervention", {}) or {}
    valid = {k: v for k, v in rec.items() if k in scm.actuators}
    u = expected_utility(scm, valid, n=n, seed=seed)
    gold_u = gold["expected_utility"]
    base_u = gold.get("baseline_utility")
    if base_u is None:
        base_u = expected_utility(scm, {}, n=n, seed=seed)
    rng = gold_u - base_u
    frac_recovered = (u - base_u) / rng if abs(rng) > 1e-9 else 1.0
    part_a = bool(frac_recovered >= benefit_frac or u >= gold_u - tolerance)
    frac, items = _score_battery(battery, answer)
    part_b = bool(frac >= 0.8)
    return {"accepted": bool(part_a and part_b), "part_a_utility_ok": part_a, "part_b_battery_ok": part_b,
            "recommended_intervention": valid, "recommended_utility": round(u, 3),
            "gold_intervention": gold["intervention"], "gold_utility": round(gold_u, 3),
            "baseline_utility": round(base_u, 3), "benefit_recovered": round(frac_recovered, 3),
            "utility_gap": round(gold_u - u, 3), "battery_fraction": round(frac, 3),
            "battery_items": items}


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------

def distractor_inertness_audit(world, gold, *, n=20000, seed=555, eps=1.0) -> Dict[str, Any]:
    """Every actuator NOT in the active set must have ~0 marginal AND ~0
    interaction with the outcome when paired with the targeted actuator.
    Justifies pruning them from the oracle search."""
    scm: WorldSCM = world["scm"]
    tgt = world["ground_truth"]["targeted_actuator"]
    active = set(gold["active_actuators"])
    tgt_val = gold["intervention"].get(tgt)
    if tgt_val is None:  # gold didn't use the targeted actuator; fall back to its mid dose
        a = scm.actuators[tgt]
        tgt_val = (a["range"][0] + a["range"][1]) / 2 if a.get("dtype") == "continuous" else a["values"][-1]
    base = expected_utility(scm, {}, n=n, seed=seed)
    u_tgt = expected_utility(scm, {tgt: tgt_val}, n=n, seed=seed)
    bad = []
    for aid, act in scm.actuators.items():
        if aid in active or aid == tgt:
            continue
        hi = act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
        marg = expected_utility(scm, {aid: hi}, n=n, seed=seed) - base
        joint = expected_utility(scm, {tgt: tgt_val, aid: hi}, n=n, seed=seed)
        interaction = joint - u_tgt - marg
        if abs(marg) > eps or abs(interaction) > eps:
            bad.append({"actuator": aid, "marginal": round(marg, 2), "interaction": round(interaction, 2)})
    return {"passed": len(bad) == 0, "n_inert_checked": len(scm.actuators) - len(active), "violations": bad}


def decoy_audit(world, *, n=20000, seed=333, eps_corr=0.3, eps_effect=0.6) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    vals = scm.sample(n, seed=seed)
    obs = scm.measure(vals, [scm.outcome] + gt["confounded_decoys"], seed=seed + 3)
    y = obs[scm.outcome]
    res, ok = {}, True
    for dec in gt["confounded_decoys"]:
        corr = float(np.corrcoef(obs[dec], y)[0, 1])
        vhi = scm.sample(n, intervention={_find_set_actuator(scm, dec): 80.0}, seed=seed + 7) if _find_set_actuator(scm, dec) else None
        # if a set-actuator exists for the decoy, forcing it should not move outcome
        if vhi is not None:
            vlo = scm.sample(n, intervention={_find_set_actuator(scm, dec): 20.0}, seed=seed + 7)
            eff = float(np.mean(scm.utility(vhi)) - np.mean(scm.utility(vlo)))
        else:
            eff = 0.0
        good = abs(corr) >= eps_corr and abs(eff) <= eps_effect
        ok = ok and good
        res[dec] = {"obs_corr": round(corr, 3), "clamp_do_effect": round(eff, 3), "ok": good}
    return {"passed": ok, "decoys": res}


def _find_set_actuator(scm, var):
    for aid, act in scm.actuators.items():
        if act.get("op") == "set" and act["target"] == var:
            return aid
    return None


def proxy_signal_audit(world, *, n=20000, seed=222, band=(0.35, 0.75)) -> Dict[str, Any]:
    """The true proxy must be an informative signal. Two valid ways to be
    informative:
      (a) OBSERVATIONAL: it correlates with the outcome in the baseline
          population within the band (visible but not decisive); OR
      (b) INTERVENTIONAL: in worlds whose mechanism is dormant at baseline (e.g.
          a two-required-causes AND-gate), observational correlation is
          legitimately ~0 and the agent must INTERVENE to see the proxy move.
          Then we require the targeted intervention to shift the proxy markedly.
    Passing either way is acceptable; failing both means the proxy is not a
    usable clue.
    """
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    proxy = gt["true_mechanism_proxy"]
    vals = scm.sample(n, seed=seed)
    obs = scm.measure(vals, [scm.outcome, proxy], seed=seed + 3)
    corr = float(np.corrcoef(obs[proxy], obs[scm.outcome])[0, 1])
    obs_ok = band[0] <= abs(corr) <= band[1]

    # interventional informativeness: does the targeted intervention move the proxy?
    tgt = gt["targeted_actuator"]
    co = gt.get("co_actuators")
    iv = {}
    if co:  # two-cause world: the informative intervention is the JOINT one
        for aid in co:
            a = scm.actuators[aid]
            iv[aid] = a["range"][1] if a.get("dtype") == "continuous" else a.get("values", ["off", "on"])[-1]
    else:
        a = scm.actuators[tgt]
        iv[tgt] = a["range"][1] if a.get("dtype") == "continuous" else a.get("values", ["off", "on"])[-1]
    base_p = obs[proxy]
    do_p = scm.measure(scm.sample(n, intervention=iv, seed=seed + 5), [proxy], seed=seed + 6, intervention=iv)[proxy]
    shift = abs(float(do_p.mean() - base_p.mean())) / (float(base_p.std()) + 1e-9)
    interv_ok = shift > 1.0

    return {"passed": bool(obs_ok or interv_ok), "proxy_outcome_corr": round(corr, 3),
            "obs_ok": obs_ok, "interventional_shift_sd": round(shift, 2), "interv_ok": interv_ok, "band": band}


def gold_selfconsistency_audit(world, gold, *, n=25000, seed=808, margin=1.0) -> Dict[str, Any]:
    """Spot-check that no simple perturbation of the gold beats it (guards the
    v5-style joint-exploit bug at scale)."""
    scm: WorldSCM = world["scm"]
    gu = gold["expected_utility"]
    beats = []
    # try adding each active actuator at extreme to the gold
    for aid in gold["active_actuators"]:
        if aid in gold["intervention"]:
            continue
        act = scm.actuators[aid]
        hi = act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
        iv = dict(gold["intervention"]); iv[aid] = hi
        u = expected_utility(scm, iv, n=n, seed=seed)
        if u > gu + margin:
            beats.append({"add": {aid: hi}, "utility": round(u, 2), "beats_by": round(u - gu, 2)})
    return {"passed": len(beats) == 0, "gold_utility": round(gu, 2), "beating_perturbations": beats}


def counterintuitiveness_audit(world, gold, *, n=25000, seed=606, help_margin=3.0) -> Dict[str, Any]:
    """The obvious first move must fail.

    The world's ``ground_truth['naive_interventions']`` lists the actuator
    settings an operator would try FIRST from the surface story (e.g. push the
    DO controller because the leading theory is oxygen drift). For the world to
    be genuinely counterintuitive we require:

      - each naive intervention does NOT meaningfully help
        (utility gain over baseline < help_margin; ideally <= 0), and
      - the gold is well separated from the best naive move.

    If any naive move recovers most of the gold's benefit, the "obvious"
    reasoning basically works and the world is not counterintuitive -> fail.
    """
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    base = expected_utility(scm, {}, n=n, seed=seed)
    gu = gold["expected_utility"]
    naive = gt.get("naive_interventions", [])
    results = []
    worst_gain = -1e9
    for iv in naive:
        valid = {k: v for k, v in iv.items() if k in scm.actuators}
        u = expected_utility(scm, valid, n=n, seed=seed)
        gain = u - base
        # fraction of the achievable (gold - base) benefit that the naive move captures
        frac = gain / (gu - base) if (gu - base) > 1e-6 else 0.0
        results.append({"naive": valid, "utility": round(u, 2), "gain_over_baseline": round(gain, 2),
                        "fraction_of_gold_benefit": round(frac, 3), "helps": gain >= help_margin})
        worst_gain = max(worst_gain, gain)
    passed = len(naive) > 0 and all(not r["helps"] for r in results)
    return {"passed": passed, "baseline_utility": round(base, 2), "gold_utility": round(gu, 2),
            "naive_results": results,
            "note": "obvious first move must not meaningfully help"}


def calibrate(world, *, n=20000, seed=333, proxy_band=(0.35, 0.75), decoy_min_corr=0.32) -> Dict[str, Any]:
    """Tune proxy assay-noise (hit proxy band) and confounder->outcome loading
    (make the decoy convincing). Mutates the SCM in place."""
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    proxy, decoy = gt["true_mechanism_proxy"], gt["confounded_decoys"][0]

    def pc():
        v = scm.sample(n, seed=seed); o = scm.measure(v, [scm.outcome, proxy], seed=seed + 3)
        return abs(float(np.corrcoef(o[proxy], o[scm.outcome])[0, 1]))

    def dc():
        v = scm.sample(n, seed=seed); o = scm.measure(v, [scm.outcome, decoy], seed=seed + 3)
        return abs(float(np.corrcoef(o[decoy], o[scm.outcome])[0, 1]))

    tgt = sum(proxy_band) / 2
    lo, hi = 0.5, 80.0
    for _ in range(26):
        mid = (lo + hi) / 2
        scm.variables[proxy]["assay_noise"] = {"normal": [0, mid]}
        if pc() > tgt:
            lo = mid
        else:
            hi = mid
    pnoise = round((lo + hi) / 2, 2)
    scm.variables[proxy]["assay_noise"] = {"normal": [0, pnoise]}

    conf = scm.variables[decoy]["parents"][0]
    ow = scm.variables[scm.outcome]["mech"]["weights"]
    if conf in ow:
        w0 = ow[conf]
        for s in (1, 2, 4, 6, 8, 10, 14):
            ow[conf] = w0 * s
            if dc() >= decoy_min_corr:
                break
    return {"proxy_assay_noise": pnoise, "proxy_corr": round(pc(), 3), "decoy_corr": round(dc(), 3)}


def audit_world(world) -> Dict[str, Any]:
    calib = calibrate(world)
    gold = optimal_intervention(world["scm"])
    battery = counterfactual_battery(world)
    return {
        "calibration": calib,
        "gold": gold,
        "battery": battery,
        "decoy": decoy_audit(world),
        "proxy_signal": proxy_signal_audit(world),
        "distractor_inertness": distractor_inertness_audit(world, gold),
        "gold_selfconsistency": gold_selfconsistency_audit(world, gold),
        "counterintuitiveness": counterintuitiveness_audit(world, gold),
    }
