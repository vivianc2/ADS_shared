#!/usr/bin/env python3
"""Expected utility, gold search, counterfactual battery, and grading."""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scm import WorldSCM, actuator_dose


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
            u = expected_utility(scm, {aid: hi}, n=n, seed=seed)
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

    non_mask = [aid for aid, act in scm.actuators.items() if act.get("op") != "mask"]
    def _extreme(aid):
        act = scm.actuators[aid]
        return act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
    solo = {aid: base + marg[aid] for aid in non_mask}
    inactive = [aid for aid in non_mask if aid not in active]
    for b in inactive:
        for a in non_mask:
            if a == b:
                continue
            u = expected_utility(scm, {a: _extreme(a), b: _extreme(b)}, n=n, seed=seed)
            if (u - solo[a]) > eps and (u - solo[b]) > eps and (u - base) > eps:
                for x in (a, b):
                    if x not in active:
                        active.append(x)
                break
    return {"baseline_utility": base, "marginal": marg, "active": active}


def optimal_intervention(scm: WorldSCM, *, n=15000, seed=12345, max_joint=3, coarse=7) -> Dict[str, Any]:
    scr = screen_actuators(scm, n=n, seed=seed)
    active = scr["active"]
    base = scr["baseline_utility"]
    best = {"intervention": {}, "expected_utility": base}

    for r in range(1, min(max_joint, len(active)) + 1):
        for combo in itertools.combinations(active, r):
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


def _policy_intervention(sp: Dict[str, Any], marker_thresh: float,
                         dose_ge: float, dose_lt: float) -> Dict[str, Any]:
    return {sp["treatment_actuator"]: {"policy": {
        "stratifier": sp["marker"], "threshold": marker_thresh,
        "dose_if_ge": dose_ge, "dose_if_lt": dose_lt}}}


def optimal_policy(scm: WorldSCM, sp: Dict[str, Any], *, n=15000, seed=12345,
                   coarse=7, base_intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_iv = dict(base_intervention or {})
    base = expected_utility(scm, base_iv, n=n, seed=seed)
    mvals = scm.sample(20000, seed=seed + 3, select=True)[sp["marker"]]
    thr_grid = [float(np.percentile(mvals, q)) for q in (30, 40, 50, 60, 70)]
    lo, hi = scm.actuators[sp["treatment_actuator"]]["range"]
    dose_grid = [round(x, 2) for x in np.linspace(lo, hi, coarse)]
    best = {"policy": None, "expected_utility": base}
    for thr in thr_grid:
        for dge in dose_grid:
            for dlt in dose_grid:
                iv = dict(base_iv)
                iv.update(_policy_intervention(sp, thr, dge, dlt))
                u = expected_utility(scm, iv, n=n, seed=seed)
                if u > best["expected_utility"]:
                    best = {"policy": {"marker": sp["marker"], "threshold": round(thr, 3),
                                       "dose_if_ge": dge, "dose_if_lt": dlt},
                            "expected_utility": u}
    best["baseline_utility"] = base
    return best


def is_v10(world: Dict[str, Any]) -> bool:
    return (world.get("ground_truth") or {}).get("_world_version") == "v10"


def _cont_grid(act, k):
    lo, hi = act["range"]
    return [round(float(x), 2) for x in np.linspace(lo, hi, k)]


def optimal_intervention_v10(scm: WorldSCM, *, n=15000, seed=12345, eps=1.0, margin=0.3,
                             screen_points=11, combo_points=5, max_joint=3, top_refine=5) -> Dict[str, Any]:
    base = expected_utility(scm, {}, n=n, seed=seed)
    acts = [a for a, act in scm.actuators.items() if act.get("op") != "mask"]
    best_single, best_val, worst_single = {}, {}, {}
    for a in acts:
        act = scm.actuators[a]
        grid = _cont_grid(act, screen_points) if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])
        us = [(expected_utility(scm, {a: v}, n=n, seed=seed), v) for v in grid]
        u, v = max(us, key=lambda t: t[0])
        best_single[a], best_val[a] = u, v
        worst_single[a] = min(t[0] for t in us)
    marg = {a: best_single[a] - base for a in acts}
    cand = [a for a in acts if marg[a] > eps]
    def _ext(a):
        act = scm.actuators[a]
        return act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
    for b in [a for a in acts if a not in cand]:
        for a in list(cand):
            u = expected_utility(scm, {a: best_val[a], b: _ext(b)}, n=n, seed=seed)
            if (u - best_single[a]) > eps and (u - base) > eps and (u - best_single[b]) > eps:
                cand.append(b)
                break

    def refine(iv):
        iv = dict(iv)
        for a in list(iv):
            act = scm.actuators[a]
            if act.get("dtype") == "continuous":
                lo, hi = act["range"]
                x, _ = _golden_section(scm, a, {k: v for k, v in iv.items() if k != a}, lo, hi, n=n, seed=seed)
                iv[a] = round(float(x), 2)
        return iv, expected_utility(scm, iv, n=n, seed=seed)

    scored = []
    for r in range(1, min(max_joint, len(cand)) + 1):
        for combo in itertools.combinations(cand, r):
            grids = [(_cont_grid(scm.actuators[a], combo_points) if scm.actuators[a].get("dtype") == "continuous"
                      else scm.actuators[a].get("values", ["off", "on"])) for a in combo]
            bu, bp = -1e18, None
            for point in itertools.product(*grids):
                u = expected_utility(scm, dict(zip(combo, point)), n=n, seed=seed)
                if u > bu:
                    bu, bp = u, point
            scored.append((bu, dict(zip(combo, bp))))
    scored.sort(key=lambda t: -t[0])
    best_iv, best_u = {}, base
    for _, iv in scored[:top_refine]:
        riv, ru = refine(iv)
        if ru > best_u:
            best_iv, best_u = riv, ru
    for _ in range(4):
        improved = False
        for a in cand:
            if a in best_iv:
                trial = {k: v for k, v in best_iv.items() if k != a}
                tu = expected_utility(scm, trial, n=n, seed=seed)
            else:
                trial, tu = refine(dict(best_iv, **{a: best_val[a]}))
            if tu > best_u + margin:
                best_iv, best_u, improved = trial, tu, True
        if not improved:
            break
    best_iv, best_u = refine(best_iv) if best_iv else (best_iv, best_u)
    tol = max(margin, 0.01 * max(0.0, best_u - base))
    pruned = True
    while pruned and len(best_iv) > 1:
        pruned = False
        for a in sorted(best_iv, key=lambda k: best_single[k] if k in best_single else 0.0):
            trial, tu = refine({k: v for k, v in best_iv.items() if k != a})
            if tu >= best_u - tol:
                best_iv, best_u, pruned = trial, tu, True
                break
    return {"intervention": best_iv, "expected_utility": best_u, "baseline_utility": base,
            "active_actuators": cand, "single_utility": {a: round(best_single[a] - base, 3) for a in acts},
            "single_range": {a: [round(worst_single[a] - base, 3), round(best_single[a] - base, 3)] for a in acts},
            "single_best_value": best_val}


def optimal_gold(world: Dict[str, Any], *, n=15000, seed=12345) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gold = (optimal_intervention_v10(scm, n=n, seed=seed) if is_v10(world)
            else optimal_intervention(scm, n=n, seed=seed))
    sp = world["ground_truth"].get("subtype_policy")
    if not sp:
        return gold
    base_iv = {k: v for k, v in gold["intervention"].items()
               if k != sp["treatment_actuator"]}
    pol = optimal_policy(scm, sp, n=n, seed=seed, base_intervention=base_iv)
    if pol["policy"] is not None and pol["expected_utility"] > gold["expected_utility"]:
        merged = dict(base_iv)
        merged.update(_policy_intervention(
            sp, pol["policy"]["threshold"],
            pol["policy"]["dose_if_ge"], pol["policy"]["dose_if_lt"]))
        return {"intervention": merged,
                "policy": pol["policy"],
                "base_intervention": base_iv,
                "expected_utility": pol["expected_utility"],
                "baseline_utility": gold["baseline_utility"],
                "active_actuators": gold["active_actuators"],
                **{k: gold[k] for k in ("single_utility", "single_range", "single_best_value") if k in gold},
                "is_conditional_policy": True}
    return gold


def _sign(delta, eps):
    return "+" if delta > eps else ("-" if delta < -eps else "0")


def _canon_sign(s):
    if s is None:
        return None
    if isinstance(s, bool):
        return "+" if s else "0"
    if isinstance(s, (int, float)):
        return "+" if s > 0 else ("-" if s < 0 else "0")
    t = str(s).strip().lower()
    if t in ("+", "1", "+1", "pos", "positive", "up", "increase", "increases", "higher"):
        return "+"
    if t in ("-", "-1", "neg", "negative", "down", "decrease", "decreases", "lower"):
        return "-"
    if t in ("0", "0.0", "none", "null", "no effect", "no_effect", "neutral", "flat", "zero", "skip"):
        return "0"
    return t


def counterfactual_battery(world: Dict[str, Any], *, n=15000, seed=777) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    eps = 0.5
    base = expected_utility(scm, {}, n=n, seed=seed)
    act_signs = {}
    for aid, act in scm.actuators.items():
        if act.get("op") == "mask":
            act_signs[aid] = "0"
            continue
        hi = act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
        if is_v10(world) and act.get("dtype", "continuous") == "continuous":
            lo_, hi_ = act["range"]
            us = [expected_utility(scm, {aid: lo_ + k * (hi_ - lo_) / 10.0}, n=n, seed=seed) - base for k in range(11)]
            act_signs[aid] = "+" if max(us) > eps else ("-" if min(us) < -eps else "0")
            if aid == ((gt.get("subtype_policy") or {}).get("treatment_actuator")):
                act_signs[aid] = "+"
            continue
        u = expected_utility(scm, {aid: hi}, n=n, seed=seed)
        act_signs[aid] = _sign(u - base, eps)

    tgt = gt["targeted_actuator"]
    tspec = scm.actuators[tgt]
    thi = tspec["range"][1] if tspec.get("dtype") == "continuous" else tspec.get("values", ["off", "on"])[-1]
    excluded = set(gt.get("_selection_nodes", []))
    excluded.add(scm.outcome)
    if scm.goal:
        excluded.add(scm.goal)
    measurables = [nm for nm, s in scm.variables.items()
                   if (s["kind"] in ("observable",) or s.get("measurable"))
                   and nm != scm.outcome and nm not in excluded]
    base_obs = scm.measure(scm.sample(n, seed=seed + 1), measurables, seed=seed + 2)
    do_obs = scm.measure(scm.sample(n, intervention={tgt: thi}, seed=seed + 1),
                         measurables, seed=seed + 2, intervention={tgt: thi})
    valid_proxies = []
    for nm in measurables:
        shift = abs(float(do_obs[nm].mean() - base_obs[nm].mean()))
        sd = float(base_obs[nm].std()) + 1e-9
        if shift / sd > 0.5:
            valid_proxies.append(nm)
    root = gt.get("true_root")
    downstream = scm._descendants({root}) if root in scm.variables else set()
    lenient = {nm for nm in measurables
               if nm in downstream and nm not in set(gt.get("confounded_decoys", []))}
    lenient |= set(valid_proxies) | {gt["true_mechanism_proxy"]}
    extra = {}
    if is_v10(world):
        extra["sign_scope"] = sorted(a for a, r in gt["_v10_roles"].items() if r == "primary")
    return {
        **extra,
        "true_mechanism_proxy": gt["true_mechanism_proxy"],
        "valid_mechanism_proxies": sorted(set(valid_proxies) | {gt["true_mechanism_proxy"]}),
        "lenient_mechanism_proxies": sorted(lenient),
        "confounded_decoys": sorted(gt["confounded_decoys"]),
        "actuator_sign_predictions": act_signs,
        "targeted_actuator": gt["targeted_actuator"],
        "symptom_trap_actuator": gt["symptom_trap_actuator"],
    }


def _score_battery(battery, answer, recommended=None, strict=True):
    st = answer.get("structured", {}) or {}
    items = []
    if strict:
        proxy_set = set(battery.get("valid_mechanism_proxies", [battery["true_mechanism_proxy"]]))
    else:
        proxy_set = set(battery.get("lenient_mechanism_proxies",
                                    battery.get("valid_mechanism_proxies", [battery["true_mechanism_proxy"]])))
    items.append(("true_mechanism_proxy", st.get("true_mechanism_proxy") in proxy_set))
    agent_decoys = set(st.get("confounded_decoys", []))
    gt_decoys = set(battery["confounded_decoys"])
    if agent_decoys & proxy_set:
        decoy_score = 0.0
    elif not gt_decoys and not agent_decoys:
        decoy_score = 1.0
    else:
        union = gt_decoys | agent_decoys
        decoy_score = len(gt_decoys & agent_decoys) / len(union) if union else 1.0
    items.append(("confounded_decoys", decoy_score))
    pred = st.get("actuator_sign_predictions", {}) or {}
    trap = battery.get("symptom_trap_actuator")
    gold_signs = battery["actuator_sign_predictions"]
    required = battery["targeted_actuator"]
    used_valid = [a for a in (recommended or {})
                  if gold_signs.get(a, "0") not in ("0", "skip")]
    if used_valid and pred.get(required) is None:
        required = used_valid[0]
    active = {aid for aid, s in gold_signs.items() if s in ("+", "-")}
    if "sign_scope" in battery:
        active &= set(battery["sign_scope"])
    # v10.2: only real controls are scored; a sign for a control that does not exist in this world used to count as
    # a correct "0" and could pad part B
    scored = {required} | active | {k for k in pred if k in gold_signs}
    for aid in scored:
        gold = gold_signs.get(aid, "0")
        items.append((f"sign:{aid}", _canon_sign(pred.get(aid)) == _canon_sign(gold)))
    score_sum = sum(float(ok) for _, ok in items)
    return score_sum / len(items), items


def _normalize_policy_answer(scm: WorldSCM, answer: Dict[str, Any],
                             valid: Dict[str, Any]) -> Dict[str, Any]:
    pol = answer.get("recommended_policy")
    if not pol:
        return valid
    treat = pol.get("treatment")
    strat = pol.get("stratifier")
    if treat not in scm.actuators or strat not in scm.variables:
        return valid
    out = dict(valid)
    out[treat] = {"policy": {"stratifier": strat,
                             "threshold": float(pol.get("threshold", 50.0)),
                             "dose_if_ge": float(pol.get("dose_if_ge", 0.0)),
                             "dose_if_lt": float(pol.get("dose_if_lt", 0.0))}}
    return out


def grade(world: Dict[str, Any], answer: Dict[str, Any], gold: Dict[str, Any],
          battery: Dict[str, Any], *, n=30000, seed=999, tolerance=2.0,
          strict=True) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    rec = answer.get("recommended_intervention", {}) or {}
    valid = {k: v for k, v in rec.items() if k in scm.actuators}
    valid = _normalize_policy_answer(scm, answer, valid)
    u = expected_utility(scm, valid, n=n, seed=seed)
    gold_u = gold["expected_utility"]
    base_u = gold.get("baseline_utility")
    if base_u is not None and (gold_u - base_u) > 1e-6:
        benefit = (u - base_u) / (gold_u - base_u)
        part_a = bool(benefit >= 0.90)
    else:
        benefit = None
        part_a = bool(u >= gold_u - tolerance)
    frac, items = _score_battery(battery, answer, recommended=valid, strict=strict)
    part_b = bool(frac >= 0.8)
    return {"accepted": bool(part_a and part_b), "part_a_utility_ok": part_a, "part_b_battery_ok": part_b,
            "recommended_intervention": valid, "recommended_utility": round(u, 3),
            "gold_intervention": gold["intervention"], "gold_utility": round(gold_u, 3),
            "baseline_utility": round(base_u, 3) if base_u is not None else None,
            "benefit_recovered": round(benefit, 3) if benefit is not None else None,
            "is_conditional_policy": bool(gold.get("is_conditional_policy")),
            "utility_gap": round(gold_u - u, 3), "battery_fraction": round(frac, 3),
            "battery_items": items}


def distractor_inertness_audit(world, gold, *, n=20000, seed=555, eps=1.0) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    tgt = gt["targeted_actuator"]
    trap = gt.get("symptom_trap_actuator")
    active = set(gold["active_actuators"])
    tgt_val = gold["intervention"].get(tgt)
    if tgt_val is None:
        a = scm.actuators[tgt]
        tgt_val = (a["range"][0] + a["range"][1]) / 2 if a.get("dtype") == "continuous" else a["values"][-1]
    base = expected_utility(scm, {}, n=n, seed=seed)
    u_tgt = expected_utility(scm, {tgt: tgt_val}, n=n, seed=seed)
    base_S = float(np.mean(scm.sample(n, seed=seed)[scm.outcome]))
    bad = []
    v10_roles = gt.get("_v10_roles") if gt.get("_world_version") == "v10" else None
    for aid, act in scm.actuators.items():
        if v10_roles is not None:
            if v10_roles.get(aid) not in ("inert", "decoy"):
                continue
        elif aid in active or aid == tgt or aid == trap:
            continue
        hi = act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
        marg = expected_utility(scm, {aid: hi}, n=n, seed=seed) - base
        joint = expected_utility(scm, {tgt: tgt_val, aid: hi}, n=n, seed=seed)
        interaction = joint - u_tgt - marg
        s_shift = float(np.mean(scm.sample(n, intervention={aid: hi}, seed=seed)[scm.outcome])) - base_S
        if abs(marg) > eps or abs(interaction) > eps or abs(s_shift) > eps:
            bad.append({"actuator": aid, "marginal": round(marg, 2),
                        "interaction": round(interaction, 2), "surrogate_shift": round(s_shift, 2)})
    return {"passed": len(bad) == 0, "n_inert_checked": len(scm.actuators) - len(active), "violations": bad}


def decoy_audit(world, *, n=20000, seed=333, eps_corr=0.3, eps_effect=0.6) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    if gt.get("_archetype") in ("competing_causes", "confounded_reversal"):
        eps_corr = min(eps_corr, 0.2)
    vals = scm.sample(n, seed=seed, select=True)
    obs = scm.measure(vals, [scm.outcome] + gt["confounded_decoys"], seed=seed + 3)
    y = obs[scm.outcome]
    res, ok = {}, True
    for dec in gt["confounded_decoys"]:
        corr = float(np.corrcoef(obs[dec], y)[0, 1])
        vhi = scm.sample(n, intervention={_find_set_actuator(scm, dec): 80.0}, seed=seed + 7) if _find_set_actuator(scm, dec) else None
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
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    if gt.get("_archetype") == "confounded_reversal":
        band = (0.2, 0.85)
    proxy = gt["true_mechanism_proxy"]
    vals = scm.sample(n, seed=seed, select=True)
    obs = scm.measure(vals, [scm.outcome, proxy], seed=seed + 3)
    corr = float(np.corrcoef(obs[proxy], obs[scm.outcome])[0, 1])
    obs_ok = band[0] <= abs(corr) <= band[1]

    tgt = gt["targeted_actuator"]
    co = gt.get("co_actuators")
    iv = {}
    if co:
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
    scm: WorldSCM = world["scm"]
    gu = gold["expected_utility"]
    beats = []
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
        frac = gain / (gu - base) if (gu - base) > 1e-6 else 0.0
        results.append({"naive": valid, "utility": round(u, 2), "gain_over_baseline": round(gain, 2),
                        "fraction_of_gold_benefit": round(frac, 3), "helps": gain >= help_margin})
        worst_gain = max(worst_gain, gain)
    passed = len(naive) > 0 and all(not r["helps"] for r in results)
    return {"passed": passed, "baseline_utility": round(base, 2), "gold_utility": round(gu, 2),
            "naive_results": results,
            "note": "obvious first move must not meaningfully help"}


def v10_audit(world, gold, *, n=20000, seed=4040, eps=1.0) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    roles = gt["_v10_roles"]
    base = gold["baseline_utility"]
    gb = gold["expected_utility"] - base
    rng_ = gold.get("single_range") or {}
    res = {}
    res["a_controls"] = len(scm.actuators) >= 22
    sym = [a for a, r in roles.items() if r == "symptomatic"]
    if sym:
        hi = {a: scm.actuators[a]["range"][1] for a in sym}
        frac = (expected_utility(scm, hi, n=n, seed=seed) - base) / gb if gb > 1e-6 else 0.0
        res["b_symptomatic_frac"] = round(frac, 3)
        res["b_symptomatic"] = frac < 0.80
    else:
        res["b_symptomatic"] = True
    gold_keys = set(k for k in gold["intervention"] if k in scm.actuators)
    graded = [a for a in scm.actuators if a not in gold_keys
              and max(abs(x) for x in rng_.get(a, [0.0, 0.0])) >= 0.10 * gb]
    res["c_graded_n"] = len(graded)
    res["c_graded"] = len(graded) >= 2
    harm_ok = True
    for a, r in roles.items():
        if r == "harm":
            d = expected_utility(scm, {a: scm.actuators[a]["range"][1]}, n=n, seed=seed) - base
            harm_ok = harm_ok and d <= -eps
    res["d_harm"] = harm_ok
    win_ok = True
    for a, act in scm.actuators.items():
        se = act.get("side_effect")
        if se and a in gold_keys and isinstance(gold["intervention"].get(a), (int, float)):
            iv_g = dict(gold["intervention"])
            iv_h = dict(iv_g); iv_h[a] = act["range"][1]
            if abs(iv_h[a] - iv_g[a]) < 5:
                continue
            o = lambda iv: scm.measure(scm.sample(n, intervention=iv, seed=seed + 1), [scm.outcome],
                                       seed=seed + 2, intervention=iv)[scm.outcome]
            og, oh = o(iv_g), o(iv_h)
            se_400 = float(np.std(og)) * np.sqrt(2.0 / 400.0)
            win_ok = win_ok and (float(og.mean()) - float(oh.mean())) >= 3.0 * se_400
    res["e_window_visible"] = win_ok
    lvl = lambda a: scm.variables.get(scm.actuators[a]["target"], {}).get("measurable", False)
    res["f_types"] = (any(lvl(a) and roles[a] != "inert" for a in roles)
                      and any((not lvl(a)) and roles[a] != "inert" for a in roles))
    res["g_gold_primary"] = all(roles.get(a) == "primary" for a in gold_keys)
    res["h_labels"] = not any((act.get("aliases") or [""])[0].lower().startswith("set ") for act in scm.actuators.values())
    checks = [k for k in res if k[0] in "abcdefgh" and isinstance(res[k], bool)]
    return {"passed": all(res[k] for k in checks), **res}


def calibrate(world, *, n=20000, seed=333, proxy_band=(0.35, 0.75), decoy_min_corr=0.32) -> Dict[str, Any]:
    scm: WorldSCM = world["scm"]
    gt = world["ground_truth"]
    if gt.get("_archetype") in ("competing_causes", "confounded_reversal"):
        decoy_min_corr = min(decoy_min_corr, 0.22)
    proxy, decoy = gt["true_mechanism_proxy"], gt["confounded_decoys"][0]

    def pc():
        v = scm.sample(n, seed=seed, select=True); o = scm.measure(v, [scm.outcome, proxy], seed=seed + 3)
        return abs(float(np.corrcoef(o[proxy], o[scm.outcome])[0, 1]))

    def dc():
        v = scm.sample(n, seed=seed, select=True); o = scm.measure(v, [scm.outcome, decoy], seed=seed + 3)
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
    loading_node = scm.goal or scm.outcome
    ow = scm.variables[loading_node]["mech"]["weights"]
    scales = ((1, 2, 4, 6, 8, 10, 14, 20, 28, 40)
              if gt.get("_archetype") == "competing_causes" else (1, 2, 4, 6, 8, 10, 14))
    if conf in ow:
        w0 = ow[conf]
        for s in scales:
            ow[conf] = w0 * s
            if dc() >= decoy_min_corr:
                break
    return {"proxy_assay_noise": pnoise, "proxy_corr": round(pc(), 3), "decoy_corr": round(dc(), 3)}


def audit_world(world) -> Dict[str, Any]:
    calib = calibrate(world)
    gold = optimal_gold(world)
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
