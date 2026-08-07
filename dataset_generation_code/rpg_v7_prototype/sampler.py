#!/usr/bin/env python3
"""RPG v7 structural world sampler.

The v6 engine/oracle/audits are structure-agnostic: they operate on any world
dict of the shape {world_id, domain, scenario, scm(WorldSCM), ground_truth}.
v6 only ever *hand-authored* four such dicts. v7 adds a generator that SAMPLES
the structure — chain depth, number of confounders / decoys / distractors, which
mechanism form each edge uses, sign patterns, and which difficulty FEATURES are
present — then dresses it with a domain skin. The existing v6 audits then filter
out any sampled world that is unsolvable, leaky, or not counterintuitive.

This is what makes the benchmark scale in a meaningful sense: not "4 problems ×
seeds" but "unlimited draws from parameterized structural families", each a
genuinely different causal graph, all graded by the same computed oracle.

A sampled world always has this backbone (a confounded mediation chain):

    source_knob --(sign)--> ROOT --hill--> M1 --...--> M_k --linear--> OUTCOME
                                    (PROXY attaches on a mediator, measurable)
    CONFOUNDER --> OUTCOME (weak) and --> DECOY (measurable)   [fake correlation]
    FIX actuator --scale--> ROOT (reduces the true cause; interior-optimum option)
    TRAP actuator --mask--> OUTCOME reading only
    many inert distractor variables + inert control actuators

Optional FEATURES layered on:
  - "sign_flip"     : the source knob helps then hurts (reversal).
  - "interior_dose" : fix has an over-treatment penalty -> best dose interior.
  - "two_cause"     : the root is an AND-gate of two knobs (neither alone works).
  - "symptom_trap"  : include the mask actuator (default on).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine import WorldSCM
from skins import SKINS, skin_names


# ---------------------------------------------------------------------------
# difficulty configuration
# ---------------------------------------------------------------------------

FEATURES = ["sign_flip", "interior_dose", "two_cause", "symptom_trap"]

# Structural archetypes. Each is a different role-wiring consumed by the SAME
# engine/oracle/audits, so each forces a DIFFERENT scientific-reasoning skill:
#   confounded_chain  : break a confound + trace a mediation chain (the backbone)
#   collider_selection: recognize selection bias -- a decoy correlates with the
#                       outcome ONLY because the historical record is conditioned
#                       on a collider; the correlation vanishes under intervention
#   hidden_subtype    : effect heterogeneity -- a treatment helps one hidden
#                       subgroup and harms the other (population-average ~0), so
#                       the ideal answer is a CONDITIONAL POLICY (stratify on an
#                       observable marker, treat only the subgroup it helps), not
#                       a single dose. Graded by the conditional-policy path.
ARCHETYPES = ["confounded_chain", "collider_selection", "hidden_subtype"]


def _choice(rng, seq):
    return seq[rng.randrange(len(seq))]


def _take(rng, pool: List[Dict[str, Any]], k: int, used: set) -> List[Dict[str, Any]]:
    """Draw k distinct entries whose names are not already used."""
    avail = [x for x in pool if x["name"] not in used]
    rng.shuffle(avail)
    picked = avail[:k]
    for x in picked:
        used.add(x["name"])
    return picked


def sample_world(seed: int, skin: Optional[str] = None,
                 features: Optional[List[str]] = None,
                 archetype: Optional[str] = None) -> Dict[str, Any]:
    """Sample one structurally-randomized world. Deterministic in `seed`.

    ``archetype`` selects the role-wiring family (see ARCHETYPES). Defaults to
    a random draw. All archetypes share the same engine/oracle/audits."""
    rng = random.Random(seed)
    skin_name = skin or _choice(rng, skin_names())
    S = SKINS[skin_name]
    arche = archetype or _choice(rng, ARCHETYPES)
    used: set = set()

    # ---- structural parameters (this is what varies across worlds) ----
    depth = rng.choice([2, 3, 4])                 # #mediators between root and outcome
    n_confounders = rng.choice([1, 1, 2])
    n_decoys = rng.choice([1, 1, 2])
    n_distractors = rng.choice([8, 10, 12])
    n_inert_knobs = rng.choice([4, 5, 6])
    feats = set(features if features is not None else
                [f for f in FEATURES if (f == "symptom_trap") or rng.random() < 0.5])
    # two_cause and sign_flip both act on the source->root edge; pick at most one
    if "two_cause" in feats and "sign_flip" in feats:
        feats.discard(rng.choice(["two_cause", "sign_flip"]))

    V: Dict[str, Dict[str, Any]] = {}
    A: Dict[str, Dict[str, Any]] = {}

    def add_var(entry, **kw):
        V[entry["name"]] = {"aliases": entry["aliases"], **kw}
        return entry["name"]

    def add_act(entry, **kw):
        A[entry["name"]] = {"aliases": entry["aliases"], **kw}
        return entry["name"]

    # ---- names ----
    outcome = S["outcome"]
    root = _take(rng, S["root_cause_pool"], 1, used)[0]
    mediators = _take(rng, S["mediator_pool"], depth, used)
    proxy = _take(rng, S["proxy_pool"], 1, used)[0]
    confs = _take(rng, S["confounder_pool"], n_confounders, used)
    decoys = _take(rng, S["decoy_pool"], n_decoys, used)
    src = _take(rng, S["source_knob_pool"], 1, used)[0]
    fix = _take(rng, S["fix_actuator_pool"], 1, used)[0]
    trap = _take(rng, S["trap_actuator_pool"], 1, used)[0]
    src2 = None
    if "two_cause" in feats:
        # need a second source knob; reuse an inert-var name as a second controllable
        src2 = _take(rng, S["inert_var_pool"], 1, used)[0]
    # reserve collider/selection names up-front so distractors don't exhaust the
    # inert pool before the augmentation block can draw them.
    sel_decoy = col_node = None
    if arche == "collider_selection":
        _sd = _take(rng, S["inert_var_pool"], 1, used)
        _col = _take(rng, S["inert_var_pool"], 1, used)
        if _sd and _col:
            sel_decoy, col_node = _sd[0], _col[0]

    # All sampled worlds are framed higher-is-better (a yield / throughput /
    # quality score). This avoids an entire class of sign-convention bugs that
    # arise when chain_sign, confounder loadings, and lower-is-better utility
    # interact. Domain framing handles the semantics (see the skin's outcome).
    higher_better = True

    # ---- exogenous latents: source-driver severity + confounders ----
    # a per-unit "severity" latent gives the root chain population variance
    sev_name = f"{root['name']}Susceptibility"
    add_var({"name": sev_name, "aliases": ["susceptibility", "batch susceptibility"]},
            kind="latent", dist={"normal": [55, 20]})
    for c in confs:
        add_var(c, kind="latent", dist={"normal": [50, 16]})

    # ---- source knob(s) (controllable, observable) ----
    add_var(src, kind="observable", dist={"normal": [40, 12]},
            measurable=True, assay_noise={"normal": [0, 3]})
    add_act(src, target=src["name"], op="set", dtype="continuous",
            range=[0, 100], default=40, description=f"controller for {src['name']}")
    if src2 is not None:
        add_var(src2, kind="observable", dist={"normal": [20, 8]},
                measurable=True, assay_noise={"normal": [0, 3]})
        add_act(src2, target=src2["name"], op="set", dtype="continuous",
                range=[0, 100], default=20, description=f"controller for {src2['name']}")

    # ---- ROOT node ----
    if "two_cause" in feats:
        # AND-gate: root high only if BOTH source knobs clear thresholds
        add_var(root, kind="latent", parents=[src["name"], src2["name"]],
                mech={"form": "gated_and", "a": src["name"], "b": src2["name"],
                      "ta": 55, "tb": 55, "wa": 9, "wb": 9, "vmax": 95.0, "intercept": 3.0})
        # for two_cause, "high root" is GOOD (it's the required-uptake style),
        # so the chain to outcome is positive; handled below via chain_sign.
        chain_sign = +1.0
    elif "sign_flip" in feats:
        # source helps then hurts: derived "harm axis" rises past a knee, times severity
        harm = f"{src['name']}HarmAxis"
        add_var({"name": harm, "aliases": ["harm axis"]},
                kind="latent", parents=[src["name"]],
                mech={"form": "sign_flip", "of": src["name"], "knee": 45,
                      "lo_gain": 0.05, "hi_gain": 0.9, "intercept": 2.0})
        add_var(root, kind="latent", parents=[harm, sev_name],
                mech={"form": "interaction", "a": harm, "b": sev_name,
                      "gain": 260.0, "scale": 100.0, "intercept": 3.0},
                noise={"normal": [0, 1.5]})
        chain_sign = -1.0   # more root -> worse outcome
    else:
        # plain: root rises with source × severity
        add_var(root, kind="latent", parents=[src["name"], sev_name],
                mech={"form": "interaction", "a": src["name"], "b": sev_name,
                      "gain": 260.0, "scale": 100.0, "intercept": 3.0},
                noise={"normal": [0, 1.5]})
        chain_sign = -1.0

    # ---- mediator chain: root -> M1 -> ... -> M_depth ----
    prev = root["name"]
    for i, med in enumerate(mediators):
        if i == 0:
            add_var(med, kind="latent", parents=[prev],
                    mech={"form": "hill", "of": prev, "vmax": 70, "k": 35, "n": 2})
        else:
            add_var(med, kind="latent", parents=[prev],
                    mech={"form": "linear", "weights": {prev: 0.9}, "intercept": 5})
        prev = med["name"]
    last_mediator = prev

    # ---- hidden-subtype augmentation (effect heterogeneity) ----
    # A treatment whose SIGN depends on a hidden subtype: it helps one subgroup
    # and harms the other, so treating EVERYONE nets ~0 (fails counterintuitive-
    # ness) and the ideal answer is a CONDITIONAL POLICY -- stratify on an
    # OBSERVABLE marker that reveals the subtype, treat only the subgroup it
    # helps. Built before the outcome so its effect node is an outcome parent.
    subtype_extra: Dict[str, float] = {}
    sub_info = None
    if arche == "hidden_subtype":
        picks = _take(rng, S["inert_var_pool"], 3, used)
        if len(picks) == 3:
            treat, marker, effnode_src = picks
            sub_name = f"{treat['name']}Subtype"
            dose_name = f"{treat['name']}Dose"
            eff_name = f"{treat['name']}Response"
            center = 50.0
            # hidden subtype latent, ~50/50 split about center
            add_var({"name": sub_name, "aliases": ["patient subtype", "hidden subtype", "responder class"]},
                    kind="latent", dist={"normal": [center, 18]})
            # observable stratifier marker: reveals the subtype (the breadcrumb)
            add_var(marker, kind="observable", parents=[sub_name],
                    mech={"form": "linear", "weights": {sub_name: 1.0}, "intercept": 0},
                    measurable=True, assay_noise={"normal": [0, 6]})
            # treatment dose target (exogenous, 0 at baseline; set by the actuator)
            add_var({"name": dose_name, "aliases": [f"{treat['aliases'][0]} dose"]},
                    kind="latent", dist={"normal": [0, 0]})
            add_act(treat, target=dose_name, op="set", dtype="continuous",
                    range=[0, 100], default=0,
                    description=f"apply the {treat['aliases'][0]} treatment")
            # heterogeneous response node -> added to the outcome
            add_var({"name": eff_name, "aliases": ["treatment response"]},
                    kind="latent", parents=[dose_name, sub_name],
                    mech={"form": "subtype_effect", "dose": dose_name, "subtype": sub_name,
                          "center": center, "gain": 46.0, "scale": 100.0})
            subtype_extra = {eff_name: 1.0}
            sub_info = {"subtype": sub_name, "marker": marker["name"], "dose": dose_name,
                        "treatment_actuator": treat["name"], "response": eff_name,
                        "center": center}

    # ---- outcome: driven by last mediator (chain_sign) + small confounder path ----
    weights = {last_mediator: 0.9 * chain_sign}
    for c in confs:
        weights[c["name"]] = 0.1 * (1 if higher_better else -1)
    weights.update(subtype_extra)
    intercept = 15 if chain_sign < 0 else 8
    add_var(outcome, kind="outcome", parents=[last_mediator] + [c["name"] for c in confs] + list(subtype_extra),
            mech={"form": "linear", "weights": weights, "intercept": intercept},
            measurable=True, assay_noise={"normal": [0, 3]})

    # ---- true mechanism proxy: attaches to a mediator (>=1 hop downstream) ----
    proxy_parent = mediators[min(1, len(mediators) - 1)]["name"]
    add_var(proxy, kind="observable", parents=[proxy_parent],
            mech={"form": "linear", "weights": {proxy_parent: 0.8}, "intercept": 5},
            measurable=True, assay_noise={"normal": [0, 12]})

    # ---- confounded decoys: driven by a confounder (zero do-effect on outcome) ----
    for d in decoys:
        cparent = _choice(rng, confs)["name"]
        add_var(d, kind="observable", parents=[cparent],
                mech={"form": "linear", "weights": {cparent: 0.7}, "intercept": 40},
                measurable=True, assay_noise={"normal": [0, 4]})

    # ---- FIX actuator: scales the root down (reduces the true cause) ----
    if "two_cause" in feats:
        # for the AND-gate, the "fix" is applying BOTH source knobs (co-actuators);
        # there is no separate scale-down fix. Mark co_actuators accordingly.
        fix_id = None
    else:
        side = None
        if "interior_dose" in feats:
            side = {"target": outcome["name"],
                    "expr": ("-overstrip(d;thr=0.66,gain=30)" if higher_better
                             else "overstrip(d;thr=0.66,gain=30)")}
        eff = {"target": root["name"], "op": "scale", "dtype": "continuous",
               "range": [0, 100], "default": 0, "expr": "1-sat(d;k=0.66)",
               "description": f"dosing to reduce {root['name']}"}
        if side:
            eff["side_effect"] = side
        fix_id = add_act(fix, **eff)

    # ---- TRAP actuator: masks the outcome reading only ----
    trap_id = None
    if "symptom_trap" in feats:
        expr = "transient_boost(d)" if higher_better else "-transient_boost(d)"
        trap_id = add_act(trap, target=outcome["name"], op="mask", dtype="continuous",
                          range=[0, 100], default=0, expr=expr,
                          description=f"a control that adjusts the {outcome['name']} readout")

    # ---- inert distractor variables + inert control actuators ----
    inert_vars = _take(rng, S["inert_var_pool"], n_distractors, used)
    for iv in inert_vars:
        add_var(iv, kind="observable", dist={"normal": [50, 12]},
                measurable=True, assay_noise={"normal": [0, 3]})
    # inert control actuators target a subset of inert vars (real handles, ~0 effect)
    for iv in inert_vars[:n_inert_knobs]:
        aid = f"set_{iv['name']}"
        A[aid] = {"aliases": [f"set {a}" for a in iv["aliases"][:2]] + [f"adjust {iv['name']}"],
                  "target": iv["name"], "op": "set", "dtype": "continuous",
                  "range": [0, 100], "default": 50, "description": f"control for {iv['name']}"}

    # ---- collider/selection archetype augmentation ----
    # A "selection decoy" is an observable with ZERO causal path to the outcome,
    # yet it correlates with the outcome in the HISTORICAL record because that
    # record was conditioned on a collider whose two parents are (a) the outcome-
    # driving chain signal and (b) the decoy's own driver. Conditioning on the
    # collider opens a spurious path decoy<->outcome that DISAPPEARS the moment
    # the agent intervenes (a controlled experiment is not selection-filtered).
    # The skill under test: don't trust an observational correlation; verify it
    # interventionally. The engine applies `selection` only to observational
    # draws, so the oracle/gold (interventional) are untouched.
    selection = None
    if arche == "collider_selection" and sel_decoy is not None:
        # decoy's exogenous driver (independent of the true cause)
        driver = f"{sel_decoy['name']}Driver"
        add_var({"name": driver, "aliases": ["latent driver"]},
                kind="latent", dist={"normal": [50, 16]})
        # the selection decoy: measurable, driven ONLY by its own driver
        add_var(sel_decoy, kind="observable", parents=[driver],
                mech={"form": "linear", "weights": {driver: 0.9}, "intercept": 5},
                measurable=True, assay_noise={"normal": [0, 4]})
        # collider: child of BOTH the outcome-driving mediator and the driver.
        # Its two parents live on very different scales across skins (a mediator
        # may have SD ~0.5 while the driver has SD ~16); an unweighted sum would
        # be dominated by one parent and conditioning would not open the spurious
        # path. Normalize each parent's contribution by its SD so BOTH matter,
        # which is what makes conditioning induce a real decoy<->outcome corr.
        _probe = WorldSCM(variables=dict(V), actuators={}, outcome=outcome["name"])
        pv = _probe._sample_raw(8000, seed=seed + 909)
        w_lm = 1.0 / (float(pv[last_mediator].std()) + 1e-9)
        w_dr = 1.0 / (float(pv[driver].std()) + 1e-9)
        add_var(col_node, kind="observable", parents=[last_mediator, driver],
                mech={"form": "linear",
                      "weights": {last_mediator: w_lm, driver: w_dr}, "intercept": 0},
                measurable=True, assay_noise={"normal": [0, 0.1]})
        # select the historical record on the collider (upper tail). soft
        # logistic selection so the observational sample is not razor-truncated.
        # thresh/soft are calibrated below to hit a target spurious correlation.
        selection = {"node": col_node["name"], "op": ">=", "thresh": 0.0, "soft": 4.0}

    scm = WorldSCM(variables=V, actuators=A, outcome=outcome["name"],
                   higher_is_better=higher_better, selection=selection)

    # Calibrate the selection so the collider-induced spurious correlation between
    # the selection decoy and the outcome lands in the decoy-audit band (target
    # ~0.45). Threshold is fixed at the collider's ~55th percentile (retain the
    # upper ~45%); we search the logistic width `soft` (sharper => stronger
    # induced dependence), robust across skins with different variance scales.
    if selection is not None:
        cnode, sdname, out_name = selection["node"], sel_decoy["name"], outcome["name"]
        base_vals = scm._sample_raw(20000, seed=seed + 4242)
        col_sd = float(np.std(base_vals[cnode])) + 1e-9
        selection["soft"] = 0.10 * col_sd     # moderate logistic sharpness
        # sweep the selection percentile (how heavily the record is tail-selected);
        # a higher percentile opens a stronger spurious path. Pick the percentile
        # whose induced |corr(decoy, outcome)| is closest to the target band center
        # while keeping selection realistic (<= 85th pct, retain >=15%).
        target, best_q, best_err = 0.45, 65.0, 1e9
        for q in (60, 65, 70, 75, 80):
            selection["thresh"] = float(np.percentile(base_vals[cnode], q))
            scm.selection = selection
            v = scm.sample(12000, seed=seed + 11, select=True)
            o = scm.measure(v, [sdname, out_name], seed=seed + 12)
            c = abs(float(np.corrcoef(o[sdname], o[out_name])[0, 1]))
            if abs(c - target) < best_err:
                best_err, best_q = abs(c - target), q
        selection["thresh"] = float(np.percentile(base_vals[cnode], best_q))
        scm.selection = selection

    # ---- ground-truth roles + naive interventions ----
    targeted = fix_id if fix_id else src["name"]
    if sub_info is not None:
        # for a hidden-subtype world the lever under test is the heterogeneous
        # treatment; the "obvious" move is to give it to EVERYONE, which nets ~0.
        targeted = sub_info["treatment_actuator"]
    co_acts = [src["name"], src2["name"]] if "two_cause" in feats else None
    naive = []
    # obvious moves = pushing the source knob the "intuitive" direction, and inert knobs
    if "sign_flip" in feats:
        naive.append({src["name"]: 100})       # "more of the input" (wrong: harms)
    if sub_info is not None:
        naive.append({sub_info["treatment_actuator"]: 100})   # treat everyone -> ~0
    for iv in inert_vars[:2]:
        naive.append({f"set_{iv['name']}": 100})

    all_decoys = [d["name"] for d in decoys]
    selection_nodes = []
    if sel_decoy is not None:
        # the selection decoy is causally inert on the outcome -> it is a decoy the
        # agent must NOT recommend acting on, and (if it has a handle) pushing it
        # is a naive move that fails.
        all_decoys.append(sel_decoy["name"])
        selection_nodes.append(sel_decoy["name"])
    if col_node is not None:
        # the COLLIDER node is causally downstream of the mediator by construction
        # (that is why conditioning on it opens the spurious path), so the targeted
        # actuator moves it and the empirical proxy scan would wrongly call it a
        # "valid mechanism proxy". It is NOT the mechanism proxy and NOT a classic
        # confounded decoy (a collider is both-parents->node, not confounder->both);
        # it is the SELECTION APPARATUS. Record it as a selection node so the proxy
        # scan excludes it, but do NOT force it into confounded_decoys (that would
        # fail the confounder-style decoy audit).
        selection_nodes.append(col_node["name"])

    gt = {
        "true_root": root["name"],
        "true_mechanism_proxy": proxy["name"],
        "confounded_decoys": all_decoys,
        "targeted_actuator": targeted,
        "symptom_trap_actuator": trap_id if trap_id else (fix_id or src["name"]),
        "co_actuators": co_acts,
        "selection_decoy": sel_decoy["name"] if sel_decoy is not None else None,
        "_selection_nodes": selection_nodes,   # selection apparatus: excluded from proxy scan
        "subtype_policy": sub_info,   # None unless hidden_subtype archetype
        "latent_plain_name": f"a hidden {root['aliases'][0]} that propagates through "
                             f"{', '.join(m['aliases'][0] for m in mediators)} to the outcome",
        "naive_interventions": naive or [{f"set_{inert_vars[0]['name']}": 100}],
        "_features": sorted(feats), "_skin": skin_name, "_depth": depth,
        "_archetype": arche,
    }
    scenario = _build_scenario(S, feats)
    _atag = {"confounded_chain": "chain", "collider_selection": "collider",
             "hidden_subtype": "subtype"}.get(arche, arche)
    wid = f"v7_{skin_name}_{_atag}_{'_'.join(sorted(feats))[:20]}_{seed}"
    return {"world_id": wid, "domain": skin_name, "scenario": scenario,
            "scm": scm, "ground_truth": gt}


def _build_scenario(S: Dict[str, Any], feats: set) -> str:
    return S["scenario"].format(**S["fills"])
