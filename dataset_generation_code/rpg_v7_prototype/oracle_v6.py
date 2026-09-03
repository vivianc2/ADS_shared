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

    # Synergy pass: rescue actuators whose SOLO effect is below the activation
    # threshold but that participate in a genuine INTERACTION — a pair whose
    # JOINT effect exceeds what EITHER member achieves alone. This covers
    # two-required-causes (AND-gate) topologies. The earlier version only checked
    # inactive x inactive pairs, which missed the common case where one co-cause
    # has a modest solo effect (so it lands in `active`) while its partner looks
    # inert solo (so it would be pruned) even though the true optimum needs BOTH.
    # We now test every inactive knob against every OTHER non-mask actuator
    # (active or inactive). Testing against an inert distractor is safe: inert
    # knobs have ~0 effect and ~0 interaction (distractor_inertness_audit), so a
    # joint with one never beats both singles and never triggers a rescue.
    non_mask = [aid for aid, act in scm.actuators.items() if act.get("op") != "mask"]
    def _extreme(aid):
        act = scm.actuators[aid]
        return act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
    solo = {aid: base + marg[aid] for aid in non_mask}   # best solo utility ~ base + marginal
    inactive = [aid for aid in non_mask if aid not in active]
    for b in inactive:
        for a in non_mask:
            if a == b:
                continue
            u = expected_utility(scm, {a: _extreme(a), b: _extreme(b)}, n=n, seed=seed)
            # genuine synergy: the joint must beat BOTH singles (so it is a real
            # interaction, not just `a` carrying the effect) and beat baseline.
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


def _policy_intervention(sp: Dict[str, Any], marker_thresh: float,
                         dose_ge: float, dose_lt: float) -> Dict[str, Any]:
    """Build the actuator intervention for a conditional (stratified) policy."""
    return {sp["treatment_actuator"]: {"policy": {
        "stratifier": sp["marker"], "threshold": marker_thresh,
        "dose_if_ge": dose_ge, "dose_if_lt": dose_lt}}}


def optimal_policy(scm: WorldSCM, sp: Dict[str, Any], *, n=15000, seed=12345,
                   coarse=7, base_intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Gold for a hidden-subtype world: the best CONDITIONAL policy that
    stratifies on the observable marker and treats only the subgroup it helps.

    Searches (marker threshold) x (dose for marker>=thr) x (dose for marker<thr)
    on a coarse grid. ``base_intervention`` is applied UNDERNEATH the policy (e.g.
    the confounded-chain fix, which also helps in a subtype world layered on the
    chain backbone), so the gold captures BOTH levers and self-consistency holds."""
    base_iv = dict(base_intervention or {})
    base = expected_utility(scm, base_iv, n=n, seed=seed)
    # marker thresholds: quantiles of the marker's observational distribution
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


def optimal_gold(world: Dict[str, Any], *, n=15000, seed=12345) -> Dict[str, Any]:
    """World-aware gold. For a hidden-subtype world, the gold is the best
    conditional policy (single doses cannot express it); otherwise the standard
    actuator-combination optimum. The record shape is unified: it always carries
    ``intervention`` (the actuator dict the engine can execute), ``expected_utility``,
    ``baseline_utility``, ``active_actuators``, and, for policy worlds, ``policy``."""
    scm: WorldSCM = world["scm"]
    gold = optimal_intervention(scm, n=n, seed=seed)
    sp = world["ground_truth"].get("subtype_policy")
    if not sp:
        return gold
    # the confounded-chain fix (if any) helps in a subtype world too; search the
    # conditional policy ON TOP of the standard optimum so the gold uses BOTH.
    # Exclude the treatment actuator from the base (the policy controls it).
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
                "is_conditional_policy": True}
    return gold


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
        hi = act["range"][1] if act.get("dtype") == "continuous" else act.get("values", ["off", "on"])[-1]
        u = expected_utility(scm, {aid: hi}, n=n, seed=seed)
        act_signs[aid] = _sign(u - base, eps)

    # Valid mechanism proxies = measurable signals that the TARGETED actuator (which
    # acts on the true cause) actually moves, but the confounder does not. Computed
    # empirically so ANY genuine proxy on the causal chain is credited, not one
    # hardcoded name (a world may expose several: e.g. interface errors AND a
    # dew-point sensor). Prevents marking a correct-but-alternative proxy wrong.
    tgt = gt["targeted_actuator"]
    tspec = scm.actuators[tgt]
    thi = tspec["range"][1] if tspec.get("dtype") == "continuous" else tspec.get("values", ["off", "on"])[-1]
    # SELECTION APPARATUS is excluded from the proxy scan: the collider node (and
    # selection-decoy driver's observable) sit downstream of the mediator BY
    # CONSTRUCTION (that is how conditioning opens the spurious path), so the
    # targeted actuator moves them and they would be mislabeled "valid proxies".
    # They are the selection machinery / decoys, not the mechanism proxy.
    excluded = set(gt.get("_selection_nodes", []))
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
        if shift / sd > 0.5:      # the true lever meaningfully moves this signal
            valid_proxies.append(nm)
    # LENIENT proxy set (used by the RL REWARD, not the strict eval): any measurable
    # that is causally DOWNSTREAM of the true root on the mechanism chain and is not
    # a decoy or selection node. This credits "understood the mechanism" even when
    # the agent names a downstream marker other than the exact sampled proxy
    # variable (the mixed9 "coulombic efficiency / interveinal chlorosis" case),
    # without crediting decoys/selection apparatus (V4). See reward-contract doc V5.
    root = gt.get("true_root")
    downstream = scm._descendants({root}) if root in scm.variables else set()
    lenient = {nm for nm in measurables
               if nm in downstream and nm not in set(gt.get("confounded_decoys", []))}
    lenient |= set(valid_proxies) | {gt["true_mechanism_proxy"]}
    return {
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
    # Proxy credit. STRICT (eval/benchmark): the exact sampled proxy or an
    # empirically-equivalent one (valid_mechanism_proxies). LENIENT (RL reward):
    # any measurable causally downstream of the true root that is not a decoy /
    # selection node -- rewards "understood the mechanism" without punishing an
    # articulate-but-non-canonical name. See reward-contract doc V5.
    if strict:
        proxy_set = set(battery.get("valid_mechanism_proxies", [battery["true_mechanism_proxy"]]))
    else:
        proxy_set = set(battery.get("lenient_mechanism_proxies",
                                    battery.get("valid_mechanism_proxies", [battery["true_mechanism_proxy"]])))
    items.append(("true_mechanism_proxy", st.get("true_mechanism_proxy") in proxy_set))
    agent_decoys = set(st.get("confounded_decoys", []))
    gt_decoys = set(battery["confounded_decoys"])
    # decoy check: must flag the true confounder(s), and must NOT mislabel any
    # genuine proxy as a decoy.
    items.append(("confounded_decoys",
                  gt_decoys.issubset(agent_decoys) and not (agent_decoys & proxy_set)))
    pred = st.get("actuator_sign_predictions", {}) or {}
    trap = battery.get("symptom_trap_actuator")
    gold_signs = battery["actuator_sign_predictions"]
    # ALT-FIX FAIRNESS: the required "primary fix" lever is normally the gold's
    # targeted actuator, but a world often admits >1 causally-valid fix (e.g.
    # reducing the source knob is equivalent to dosing down the root). If the
    # agent RECOMMENDED a different but causally-valid lever (non-'0'/'skip' true
    # sign) and did not opine on the gold lever, score THAT lever as the required
    # one — otherwise a correct alternative fix is unfairly docked for not
    # predicting a lever the agent never used.
    required = battery["targeted_actuator"]
    used_valid = [a for a in (recommended or {})
                  if gold_signs.get(a, "0") not in ("0", "skip")]
    if used_valid and pred.get(required) is None:
        required = used_valid[0]
    scored = {required} | set(pred)
    for aid in scored:
        gold = gold_signs.get(aid, "0")
        if aid == trap:
            ok = pred[aid] in ("0", "+") if aid in pred else True
            items.append((f"sign:{aid}(trap)", ok))
        else:
            items.append((f"sign:{aid}", pred.get(aid) == gold))
    n_ok = sum(1 for _, ok in items if ok)
    return n_ok / len(items), items


def _normalize_policy_answer(scm: WorldSCM, answer: Dict[str, Any],
                             valid: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a conditional-policy answer into the executable intervention dict.

    The agent expresses a stratified policy as:
        "recommended_policy": {"treatment": <actuator id or alias-resolved id>,
                               "stratifier": <marker var>, "threshold": <float>,
                               "dose_if_ge": <float>, "dose_if_lt": <float>}
    We convert it to the engine's per-unit set form and merge it in (overriding
    any scalar the agent also gave for that treatment actuator)."""
    pol = answer.get("recommended_policy")
    if not pol:
        return valid
    treat = pol.get("treatment")
    strat = pol.get("stratifier")
    if treat not in scm.actuators or strat not in scm.variables:
        return valid            # unresolvable policy -> leave scalar answer as-is
    out = dict(valid)
    out[treat] = {"policy": {"stratifier": strat,
                             "threshold": float(pol.get("threshold", 50.0)),
                             "dose_if_ge": float(pol.get("dose_if_ge", 0.0)),
                             "dose_if_lt": float(pol.get("dose_if_lt", 0.0))}}
    return out


def grade(world: Dict[str, Any], answer: Dict[str, Any], gold: Dict[str, Any],
          battery: Dict[str, Any], *, n=30000, seed=999, tolerance=2.0,
          strict=True) -> Dict[str, Any]:
    """Grade an answer. ``strict`` controls part-B proxy credit:
    - strict=True (DEFAULT, eval/benchmark): the exact sampled proxy (or an
      empirically-equivalent one). Precise, comparable — the reported number.
    - strict=False (RL reward): any measurable downstream of the true root that is
      not a decoy/selection node. Rewards mechanism understanding, not exact naming.
    See the reward-contract decisions (V5)."""
    scm: WorldSCM = world["scm"]
    rec = answer.get("recommended_intervention", {}) or {}
    valid = {k: v for k, v in rec.items() if k in scm.actuators}
    valid = _normalize_policy_answer(scm, answer, valid)
    u = expected_utility(scm, valid, n=n, seed=seed)
    gold_u = gold["expected_utility"]
    # judge part A by fraction of achievable benefit recovered (consistent bar
    # across topologies whatever the utility scale), with a small absolute floor.
    base_u = gold.get("baseline_utility")
    if base_u is not None and (gold_u - base_u) > 1e-6:
        benefit = (u - base_u) / (gold_u - base_u)
        part_a = bool(benefit >= 0.90 or u >= gold_u - tolerance)
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
    vals = scm.sample(n, seed=seed, select=True)     # observational (selected) record
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
    vals = scm.sample(n, seed=seed, select=True)     # observational (selected) record
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
