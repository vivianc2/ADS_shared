#!/usr/bin/env python3
"""Oracle, grader, solvability certificate, and faithfulness audits for RPG v5.

Everything here is computed from the SCM itself, so the golden answer is
mathematically derived rather than string-matched. Three products:

1. ``optimal_intervention`` — best knob + dose by common-random-number Monte
   Carlo with golden-section refinement on continuous knobs (part 4A).
2. ``counterfactual_battery`` — ground-truth answers to held-out interventional
   predictions used to grade *understanding* (part 4B).
3. audits — faithfulness/anti-leakage checks + a solvability certificate that
   confirms the true cause is separable from decoys within a query budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scm import SCM
from worlds import World


# ---------------------------------------------------------------------------
# Expected utility under an intervention (common random numbers)
# ---------------------------------------------------------------------------

def expected_utility(scm: SCM, intervention: Dict[str, Any], *, n: int, seed: int) -> float:
    vals = scm.sample(n, intervention=intervention, seed=seed)
    return float(np.mean(scm.utility(vals)))


def _grid(lo: float, hi: float, k: int) -> List[float]:
    return list(np.linspace(lo, hi, k))


def optimal_intervention(
    world: World, *, n: int = 20000, seed: int = 12345, coarse: int = 7, refine_iters: int = 25
) -> Dict[str, Any]:
    """Best single-knob intervention with golden-section refinement.

    We restrict the golden answer to single-knob settings (the world families
    are designed so the optimum is one targeted knob at the right dose). Joint
    settings the agent submits are re-scored on the fly by the grader.
    """
    scm = world.scm
    baseline_u = expected_utility(scm, {}, n=n, seed=seed)
    best = {"intervention": {}, "expected_utility": baseline_u, "knob": None, "value": None}

    for knob, spec in world.knobs.items():
        dtype = spec.get("dtype", "continuous")
        if dtype == "continuous":
            lo, hi = spec["range"]
            # coarse sweep
            cand = _grid(lo, hi, coarse)
            us = [expected_utility(scm, {knob: v}, n=n, seed=seed) for v in cand]
            j = int(np.argmax(us))
            # golden-section refine around best coarse point
            a = cand[max(0, j - 1)]
            b = cand[min(len(cand) - 1, j + 1)]
            v_star, u_star = _golden_section(scm, knob, a, b, n=n, seed=seed, iters=refine_iters)
            if u_star > best["expected_utility"]:
                best = {"intervention": {knob: round(float(v_star), 3)},
                        "expected_utility": u_star, "knob": knob, "value": round(float(v_star), 3)}
        else:
            for v in spec.get("values", []):
                u = expected_utility(scm, {knob: v}, n=n, seed=seed)
                if u > best["expected_utility"]:
                    best = {"intervention": {knob: v}, "expected_utility": u, "knob": knob, "value": v}

    best["baseline_utility"] = baseline_u
    return best


def _golden_section(
    scm: SCM, knob: str, a: float, b: float, *, n: int, seed: int, iters: int
) -> Tuple[float, float]:
    """Maximize expected utility over [a, b] for one continuous knob."""
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = expected_utility(scm, {knob: c}, n=n, seed=seed)
    fd = expected_utility(scm, {knob: d}, n=n, seed=seed)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = expected_utility(scm, {knob: c}, n=n, seed=seed)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = expected_utility(scm, {knob: d}, n=n, seed=seed)
    x = (a + b) / 2
    return x, expected_utility(scm, {knob: x}, n=n, seed=seed)


# ---------------------------------------------------------------------------
# Counterfactual battery (part 4B) — ground truth for grading understanding
# ---------------------------------------------------------------------------

def _sign(delta: float, eps: float) -> str:
    if delta > eps:
        return "+"
    if delta < -eps:
        return "-"
    return "0"


def counterfactual_battery(world: World, *, n: int = 20000, seed: int = 777) -> Dict[str, Any]:
    """Ground-truth interventional predictions the agent must reproduce.

    Signs are expressed *toward the outcome direction the agent cares about*
    (higher utility = better), so a "+" means the knob improves things.
    """
    scm = world.scm
    eps = 0.5  # utility units treated as "no effect"
    base_u = expected_utility(scm, {}, n=n, seed=seed)

    knob_signs: Dict[str, str] = {}
    for knob, spec in world.knobs.items():
        if spec.get("dtype") == "continuous":
            hi = spec["range"][1]
            u = expected_utility(scm, {knob: hi}, n=n, seed=seed)
        else:
            top = spec.get("values", ["off", "on"])[-1]
            u = expected_utility(scm, {knob: top}, n=n, seed=seed)
        knob_signs[knob] = _sign(u - base_u, eps)

    # clamp of the confounded decoy should not move the outcome
    clamp_effects: Dict[str, str] = {}
    for obs in world.clampable:
        # clamp to two levels and compare utility
        vals_hi = scm.sample(n, clamp={obs: 80.0}, seed=seed + 1)
        vals_lo = scm.sample(n, clamp={obs: 20.0}, seed=seed + 1)
        du = float(np.mean(scm.utility(vals_hi)) - np.mean(scm.utility(vals_lo)))
        clamp_effects[obs] = _sign(du, eps)

    return {
        "true_mechanism_proxy": world.true_mechanism_proxy,
        "confounded_decoys": sorted(world.confounded_decoys),
        "knob_sign_predictions": knob_signs,     # e.g. FeedWaterFlow: '-', RegimenC: '+'
        "clamp_effect_on_outcome": clamp_effects, # confounded decoy -> '0'
        "targeted_knob": world.targeted_knob,
        "symptom_trap_knob": world.symptom_trap_knob,
    }


def grade_answer(world: World, battery: Dict[str, Any], gold_iv: Dict[str, Any],
                 answer: Dict[str, Any], *, n: int = 20000, seed: int = 999,
                 tolerance: float = 2.0) -> Dict[str, Any]:
    """Grade an agent answer. Acceptance = utility-optimal (A) AND battery (B).

    ``answer`` shape (see doc §4B)::

        {"recommended_intervention": {"RegimenC": 66},
         "structured": {"true_mechanism_proxy": "...",
                        "confounded_decoys": ["..."],
                        "knob_sign_predictions": {"FeedWaterFlow": "-", ...}}}
    """
    scm = world.scm
    # ----- Part A: utility of the recommended intervention -----
    rec = answer.get("recommended_intervention", {}) or {}
    u = expected_utility(scm, rec, n=n, seed=seed)
    gold_u = gold_iv["expected_utility"]
    part_a = bool(u >= gold_u - tolerance)

    # ----- Part B: structured predictions vs battery -----
    st = answer.get("structured", {}) or {}
    b_items: List[Tuple[str, bool]] = []
    b_items.append(("true_mechanism_proxy",
                    st.get("true_mechanism_proxy") == battery["true_mechanism_proxy"]))
    b_items.append(("confounded_decoys",
                    set(st.get("confounded_decoys", [])) == set(battery["confounded_decoys"])))
    pred_signs = st.get("knob_sign_predictions", {}) or {}
    for knob, gold_sign in battery["knob_sign_predictions"].items():
        # the symptom-trap knob's "improvement" is transient; accept '0' or '+'
        if knob == battery["symptom_trap_knob"]:
            ok = pred_signs.get(knob) in ("0", "+", "0_or_transient", None) or True
            # only score if agent offered a prediction; treat trap leniently
            if knob in pred_signs:
                ok = pred_signs[knob] in ("0", "+", "0_or_transient")
            b_items.append((f"sign:{knob}(trap)", ok))
        else:
            b_items.append((f"sign:{knob}", pred_signs.get(knob) == gold_sign))

    n_ok = sum(1 for _, ok in b_items if ok)
    frac = n_ok / len(b_items)
    part_b = bool(frac >= 0.8)

    return {
        "accepted": bool(part_a and part_b),
        "part_a_utility_ok": part_a,
        "part_b_battery_ok": part_b,
        "recommended_utility": u,
        "gold_utility": gold_u,
        "utility_gap": gold_u - u,
        "battery_fraction": frac,
        "battery_items": b_items,
    }


def _score_battery(battery: Dict[str, Any], answer: Dict[str, Any]) -> Tuple[float, List[Tuple[str, bool]]]:
    """Battery matching only (no SCM needed). Shared by World- and record-based
    graders so the acceptance logic lives in one place."""
    st = answer.get("structured", {}) or {}
    items: List[Tuple[str, bool]] = []
    items.append(("true_mechanism_proxy",
                  st.get("true_mechanism_proxy") == battery["true_mechanism_proxy"]))
    # Decoy check: the agent must (a) flag the real confounder(s) as decoys and
    # (b) NOT mislabel the true mechanism proxy as a decoy. Listing additional
    # causally-inert observables (e.g. TemperatureReading) as decoys is CORRECT,
    # not a mistake, so we do not require exact set equality.
    agent_decoys = set(st.get("confounded_decoys", []))
    gt_decoys = set(battery["confounded_decoys"])
    decoy_ok = gt_decoys.issubset(agent_decoys) and \
        battery["true_mechanism_proxy"] not in agent_decoys
    items.append(("confounded_decoys", decoy_ok))
    pred = st.get("knob_sign_predictions", {}) or {}
    trap = battery.get("symptom_trap_knob")
    for knob, gold_sign in battery["knob_sign_predictions"].items():
        if knob == trap:
            ok = pred[knob] in ("0", "+", "0_or_transient") if knob in pred else True
            items.append((f"sign:{knob}(trap)", ok))
        else:
            items.append((f"sign:{knob}", pred.get(knob) == gold_sign))
    n_ok = sum(1 for _, ok in items if ok)
    return n_ok / len(items), items


def grade_answer_record(sim, answer: Dict[str, Any], *, n: int = 20000,
                        seed: int = 999, tolerance: float = 2.0) -> Dict[str, Any]:
    """Grade an agent answer from a loaded SimV5 (record-based). Acceptance =
    utility-optimal (A) AND battery >= 80% (B). This is the runtime grader."""
    scm = sim.scm
    oracle = sim.oracle
    battery = oracle["counterfactual_battery"]
    gold_u = float(oracle["gold_intervention"]["expected_utility"])

    rec = answer.get("recommended_intervention", {}) or {}
    # validate against the world's knobs; ignore unknown keys defensively
    valid = {k: v for k, v in rec.items() if k in sim.knobs}
    u = expected_utility(scm, valid, n=n, seed=seed)
    part_a = bool(u >= gold_u - tolerance)

    frac, items = _score_battery(battery, answer)
    part_b = bool(frac >= 0.8)
    return {
        "accepted": bool(part_a and part_b),
        "part_a_utility_ok": part_a,
        "part_b_battery_ok": part_b,
        "recommended_intervention": valid,
        "recommended_utility": round(u, 3),
        "gold_utility": round(gold_u, 3),
        "gold_intervention": oracle["gold_intervention"]["intervention"],
        "utility_gap": round(gold_u - u, 3),
        "battery_fraction": round(frac, 3),
        "battery_items": items,
    }


# ---------------------------------------------------------------------------
# Faithfulness audits + solvability certificate
# ---------------------------------------------------------------------------

_LEAK_RE = None


def name_leakage_audit(world: World) -> Dict[str, Any]:
    import re
    global _LEAK_RE
    if _LEAK_RE is None:
        # Only terms that would reveal the HIDDEN mechanism/cause. Operational
        # verbs (flush, feed) are fine — they name a legitimate control, not the
        # answer. The check targets names that hand the agent the latent.
        _LEAK_RE = re.compile(
            r"chelat|copper|\biron\b|toxin|contaminant|corros|robb|bacter|metal",
            re.IGNORECASE,
        )
    leaks = []
    # knob names and observable names must not name the mechanism
    for name in list(world.knobs) + world.observables:
        if _LEAK_RE.search(name):
            # allowed if it is itself an intended observable clue? No — be strict.
            leaks.append(name)
    return {"passed": len(leaks) == 0, "leaky_names": leaks}


def decoy_audit(world: World, *, n: int = 20000, seed: int = 555, eps_corr: float = 0.3,
                eps_effect: float = 0.5) -> Dict[str, Any]:
    """Each confounded decoy must be observationally correlated with the outcome
    yet causally inert (clamping it must not move the outcome)."""
    scm = world.scm
    vals = scm.sample(n, seed=seed)
    obs = scm.observe(vals, [world.scm.outcome] + world.confounded_decoys, seed=seed + 3)
    y = obs[scm.outcome]
    results = {}
    ok = True
    for decoy in world.confounded_decoys:
        corr = float(np.corrcoef(obs[decoy], y)[0, 1])
        vals_hi = scm.sample(n, clamp={decoy: 80.0}, seed=seed + 7)
        vals_lo = scm.sample(n, clamp={decoy: 20.0}, seed=seed + 7)
        eff = float(np.mean(scm.utility(vals_hi)) - np.mean(scm.utility(vals_lo)))
        decoy_ok = (abs(corr) >= eps_corr) and (abs(eff) <= eps_effect)
        ok = ok and decoy_ok
        results[decoy] = {"obs_corr_with_outcome": round(corr, 3),
                          "clamp_do_effect": round(eff, 3), "ok": decoy_ok}
    return {"passed": ok, "decoys": results}


def proxy_signal_audit(world: World, *, n: int = 20000, seed: int = 333,
                       band: Tuple[float, float] = (0.35, 0.75)) -> Dict[str, Any]:
    """The true mechanism proxy must correlate with the outcome inside a band:
    visible enough to notice, weak enough to require intervention."""
    scm = world.scm
    vals = scm.sample(n, seed=seed)
    obs = scm.observe(vals, [scm.outcome, world.true_mechanism_proxy], seed=seed + 3)
    corr = float(np.corrcoef(obs[world.true_mechanism_proxy], obs[scm.outcome])[0, 1])
    return {"passed": band[0] <= abs(corr) <= band[1],
            "proxy_outcome_corr": round(corr, 3), "band": band}


def solvability_certificate(world: World, gold_iv: Dict[str, Any], battery: Dict[str, Any],
                            *, n: int = 20000, seed: int = 111) -> Dict[str, Any]:
    """Confirm an oracle-informed querying strategy can separate the true cause
    from the best decoy within a small budget, and record the minimal
    discriminating query set (the efficiency yardstick).

    The certificate is a concrete sequence of interventions whose *results*
    uniquely fingerprint the true mechanism versus each decoy hypothesis.
    """
    scm = world.scm
    checks: List[Dict[str, Any]] = []

    # (1) sweeping the targeted knob moves the outcome AND the true proxy;
    #     sweeping it does NOT move a confounded decoy.
    tk = world.targeted_knob
    spec = world.knobs[tk]
    hi = spec["range"][1] if spec.get("dtype") == "continuous" else spec["values"][-1]
    vals_do = scm.sample(n, intervention={tk: hi}, seed=seed)
    vals_base = scm.sample(n, seed=seed)
    d_out = float(np.mean(scm.utility(vals_do)) - np.mean(scm.utility(vals_base)))
    obs_do = scm.observe(vals_do, [world.true_mechanism_proxy] + world.confounded_decoys, seed=seed + 2)
    obs_base = scm.observe(vals_base, [world.true_mechanism_proxy] + world.confounded_decoys, seed=seed + 2)
    d_proxy = float(np.mean(obs_do[world.true_mechanism_proxy]) - np.mean(obs_base[world.true_mechanism_proxy]))
    checks.append({"query": f"sweep {tk}", "outcome_moves": abs(d_out) > 1.0,
                   "true_proxy_moves": abs(d_proxy) > 1.0, "d_out": round(d_out, 2),
                   "d_proxy": round(d_proxy, 2)})

    # (2) clamping the decoy does not move the outcome (rules out the naive cause)
    for decoy, s in battery["clamp_effect_on_outcome"].items():
        checks.append({"query": f"clamp {decoy}", "outcome_moves": s != "0", "expected": "0", "got": s})

    # (3) the symptom trap raises the outcome but not the true proxy
    trap = world.symptom_trap_knob
    tspec = world.knobs[trap]
    thi = tspec["range"][1] if tspec.get("dtype") == "continuous" else tspec["values"][-1]
    vals_trap = scm.sample(n, intervention={trap: thi}, seed=seed + 5)
    obs_trap = scm.observe(vals_trap, [world.true_mechanism_proxy], seed=seed + 6)
    d_trap_proxy = float(np.mean(obs_trap[world.true_mechanism_proxy]) - np.mean(obs_base[world.true_mechanism_proxy]))
    checks.append({"query": f"sweep {trap} (trap)", "true_proxy_moves": abs(d_trap_proxy) > 1.0,
                   "d_proxy": round(d_trap_proxy, 2)})

    discriminable = (
        checks[0]["outcome_moves"] and checks[0]["true_proxy_moves"]
        and all((c.get("got", "0") == "0") for c in checks if c["query"].startswith("clamp"))
        and not checks[-1]["true_proxy_moves"]
    )
    return {"solvable": bool(discriminable),
            "minimal_query_set": [c["query"] for c in checks],
            "n_queries": len(checks),
            "checks": checks}


def calibrate_world(world: World, *, n: int = 20000, seed: int = 333,
                    proxy_band: Tuple[float, float] = (0.35, 0.75),
                    decoy_min_corr: float = 0.3) -> Dict[str, Any]:
    """Auto-tune two knobs so the world lands in its target difficulty bands
    (doc §7.2): the true proxy's obs-noise (to hit the proxy correlation band)
    and the confounder->outcome loading (so the decoy is observationally
    convincing). Mutates the world's SCM in place. This is a compact stand-in
    for the generation-time calibration sweep.
    """
    scm = world.scm
    proxy = world.true_mechanism_proxy
    decoy = world.confounded_decoys[0]

    def proxy_corr() -> float:
        vals = scm.sample(n, seed=seed)
        obs = scm.observe(vals, [scm.outcome, proxy], seed=seed + 3)
        return abs(float(np.corrcoef(obs[proxy], obs[scm.outcome])[0, 1]))

    def decoy_corr() -> float:
        vals = scm.sample(n, seed=seed)
        obs = scm.observe(vals, [scm.outcome, decoy], seed=seed + 3)
        return abs(float(np.corrcoef(obs[decoy], obs[scm.outcome])[0, 1]))

    # 1) proxy obs-noise: bisection on noise SD to hit the middle of the band.
    target = sum(proxy_band) / 2
    lo, hi = 0.5, 60.0
    for _ in range(24):
        mid = (lo + hi) / 2
        scm.nodes[proxy]["obs_noise"] = {"normal": [0, mid]}
        c = proxy_corr()
        # higher noise -> lower corr
        if c > target:
            lo = mid
        else:
            hi = mid
    proxy_noise = round((lo + hi) / 2, 2)
    scm.nodes[proxy]["obs_noise"] = {"normal": [0, proxy_noise]}

    # 2) confounder->outcome loading: find which outcome weight comes from the
    #    confounder's driver and scale it up until the decoy correlates enough.
    #    We locate the confounder latent that feeds the decoy, then its weight
    #    into the outcome.
    conf_latent = scm.nodes[decoy]["parents"][0]
    out_w = scm.nodes[scm.outcome]["mech"]["weights"]
    if conf_latent in out_w:
        w0 = out_w[conf_latent]
        for scale in (1, 2, 4, 6, 8, 10, 14):
            out_w[conf_latent] = w0 * scale
            if decoy_corr() >= decoy_min_corr:
                break
    return {"proxy_obs_noise": proxy_noise,
            "proxy_corr_after": round(proxy_corr(), 3),
            "decoy_corr_after": round(decoy_corr(), 3),
            "confounder_latent": conf_latent}


def gold_optimality_audit(world: World, gold: Dict[str, Any], *, n: int = 30000,
                          seed: int = 24680, margin: float = 2.0) -> Dict[str, Any]:
    """Verify the single-knob gold is not beaten by a *joint* intervention that
    pairs the targeted knob with each other knob at its extreme.

    This guards a subtle bug: if a decoy/nutrient knob has a residual direct
    benefit that a confound normally hides, then once the targeted knob removes
    the confound the pair can exceed single-knob gold -> the oracle (which
    searches single knobs) would mislabel the answer and the grader would accept
    "do the fix + crank the decoy". Any beating pair fails the audit.
    """
    scm = world.scm
    gold_u = gold["expected_utility"]
    tk = gold.get("knob") or world.targeted_knob
    tval = gold.get("value")
    beating = []
    for knob, spec in world.knobs.items():
        if knob == tk:
            continue
        if spec.get("dtype") == "continuous":
            trials = [spec["range"][0], spec["range"][1]]
        else:
            trials = list(spec.get("values", []))
        for v in trials:
            u = expected_utility(scm, {tk: tval, knob: v}, n=n, seed=seed)
            if u > gold_u + margin:
                beating.append({"pair": {tk: tval, knob: v}, "utility": round(u, 2),
                                "beats_gold_by": round(u - gold_u, 2)})
    return {"passed": len(beating) == 0, "gold_utility": round(gold_u, 2),
            "beating_joint_interventions": beating}


def audit_world(world: World, calibrate: bool = True) -> Dict[str, Any]:
    calib = calibrate_world(world) if calibrate else None
    gold = optimal_intervention(world)
    battery = counterfactual_battery(world)
    return {
        "world_id": world.world_id,
        "calibration": calib,
        "gold_intervention": gold,
        "battery": battery,
        "name_leakage": name_leakage_audit(world),
        "decoy": decoy_audit(world),
        "proxy_signal": proxy_signal_audit(world),
        "gold_optimality": gold_optimality_audit(world, gold),
        "solvability": solvability_certificate(world, gold, battery),
    }
