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
#   surrogate_trap    : a controllable-but-useless metric -- a measured SURROGATE
#                       correlates with the outcome (shared confounder) and has a
#                       handle that moves it directly, but that handle has ZERO
#                       do-effect on the true outcome. Skill: don't optimize a
#                       surrogate; verify the OUTCOME, not a proxy metric you can
#                       push. (REUSE: confounded decoy + set-handle, sign graded "0".)
#   instrument_only   : the true cause has NO direct handle -- the only lever is the
#                       UPSTREAM source/instrument knob; acting on the cause directly
#                       is impossible, so you must reason back to the instrument.
#                       (REUSE*: structural twist -- the fix actuator is removed.)
#   competing_causes  : TWO independent causes feed the outcome (unequal strength);
#                       neither alone recovers >=90% of the benefit, so you must find
#                       BOTH and not stop at the first. (REUSE: 2nd chain + 2nd fix.)
#   synergy_pair      : two levers that do NOTHING alone but TOGETHER produce a large
#                       effect (AND-gate). Skill: test COMBINATIONS, not one-at-a-time
#                       (a lever inert alone can be essential in combination). REUSE:
#                       the two_cause AND-gate as the core; the oracle's synergy-rescue
#                       pass finds the pair. NB structurally = the two_cause FEATURE ->
#                       for a clean held-out transfer test, exclude the two_cause
#                       feature from TRAINING worlds (or use synergy_pair as train-only).
#   dose_window       : the fix has a THERAPEUTIC WINDOW -- it helps as you reduce the
#                       cause, but OVER-treating past a point harms the outcome, so the
#                       optimum is INTERIOR (both rails fail). Skill: find the sweet
#                       spot, don't max the knob. (REUSE: forces the interior_dose
#                       feature as the central task; oracle golden-section finds it.)
#   confounded_reversal: the treatment's OBSERVATIONAL correlation with the outcome is
#                       the OPPOSITE sign of its true causal effect (a confounder drives
#                       both who-is-treated and the outcome). Skill: don't trust the
#                       observational sign -- INTERVENE; the aggregate can reverse.
#                       (REUSE: reversed-sign source lever; Simpson's / confounding-by-
#                       indication; graded by the interventional sign.)
ARCHETYPES = ["confounded_chain", "collider_selection", "hidden_subtype",
              "surrogate_trap", "instrument_only", "competing_causes", "synergy_pair",
              "dose_window", "confounded_reversal"]


def _choice(rng, seq):
    return seq[rng.randrange(len(seq))]


# Mechanism-proxy names whose natural polarity is HIGHER = BETTER, i.e. consistent with the
# engine's higher_better=True (the proxy tracks the root/outcome UPWARD with the correct fix).
# Bad-marker proxies (turbidity, defect density, CRP, leakage) are the OPPOSITE polarity: the
# engine forces them up with the fix, but a domain reasoner knows they should go DOWN, so it
# INVERTS the answer (traced: Opus signed the synergy co-actuators "-"/"0" and reduced them ->
# part_a 0.00 on watertreatment/semiconductor/clinical). For synergy (RPG_SYNERGY_SOFT) we
# restrict the mechanism proxy to this set so "proxy moves in the good direction" is honest.
_POSITIVE_PROXY = {
    "DewPointMargin", "LeafGreenness", "TissueNutrientAssay", "CoulombicEfficiency",
    "CatalystSurfaceArea", "ViabilityStain",
    "FiltrationIntegrityIndex", "TissuePerfusionIndex", "FilmUniformityIndex", "GillPerfusionIndex",
    "ViableCellDensity",
}


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
                [f for f in FEATURES if f != "symptom_trap" and rng.random() < 0.5])
    # two_cause and sign_flip both act on the source->root edge; pick at most one
    if "two_cause" in feats and "sign_flip" in feats:
        feats.discard(rng.choice(["two_cause", "sign_flip"]))
    # surrogate_trap needs the real root-fix lever present (two_cause removes it).
    if arche == "surrogate_trap":
        feats.discard("two_cause")
    # instrument_only removes the direct fix lever; interior_dose and two_cause both
    # act on that fix, so drop them (nothing to attach to).
    if arche == "instrument_only":
        feats.discard("two_cause")
        feats.discard("interior_dose")
    # competing_causes: keep cause 1 a plain single-fix chain (no source-knob lever;
    # removed below), no two_cause/sign_flip/interior_dose (they add source/fix
    # complications). The two causes then each have ONE clean lever (their fix).
    if arche == "competing_causes":
        feats.discard("two_cause"); feats.discard("sign_flip"); feats.discard("interior_dose")
    # synergy_pair IS the two-required-causes AND-gate: force two_cause on; drop
    # sign_flip (both act on the source->root edge).
    if arche == "synergy_pair":
        feats.add("two_cause"); feats.discard("sign_flip")
    # dose_window: force the therapeutic-window (interior_dose) penalty as the central
    # task; need the fix present, so drop two_cause.
    if arche == "dose_window":
        feats.discard("two_cause"); feats.add("interior_dose")
    # confounded_reversal: the source knob is the reversed-sign treatment; keep it a
    # clean single lever (no two_cause/sign_flip/interior_dose; the fix is removed below).
    if arche == "confounded_reversal":
        feats.discard("two_cause"); feats.discard("sign_flip"); feats.discard("interior_dose")
    # The Goodhart / symptom trap is the DEFINING mechanism of surrogate_trap, which is a
    # HELD-OUT archetype. Confine the trap to it — and keep it OUT of every TRAINING archetype
    # — so we can still measure TRANSFER of trap-resistance: a model that never trained on a
    # metric-gaming control must nonetheless resist one at eval. A universal trap would put
    # that skill in-distribution and forfeit the headline held-out claim.
    if arche == "surrogate_trap":
        feats.add("symptom_trap")
    else:
        feats.discard("symptom_trap")

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
    # For synergy (default ON; gated with the super-additive fix on RPG_SYNERGY_SOFT), require a
    # positive-polarity mechanism proxy so the honest "verify the proxy moved in the good
    # direction" check is not physically inverted (see _POSITIVE_PROXY). Fall back to the full
    # pool if a skin has no positive option. RPG_SYNERGY_SOFT=0 reproduces the old (buggy) design.
    _proxy_pool = S["proxy_pool"]
    if arche == "synergy_pair" and float(__import__("os").environ.get("RPG_SYNERGY_SOFT", "20")):
        _pos = [p for p in _proxy_pool if p["name"] in _POSITIVE_PROXY]
        if _pos:
            _proxy_pool = _pos
    proxy = _take(rng, _proxy_pool, 1, used)[0]
    confs = _take(rng, S["confounder_pool"], n_confounders, used)
    decoys = _take(rng, S["decoy_pool"], n_decoys, used)
    src = _take(rng, S["source_knob_pool"], 1, used)[0]
    fix = _take(rng, S["fix_actuator_pool"], 1, used)[0]
    trap = _take(rng, S["trap_actuator_pool"], 1, used)[0]
    # The TRUE GOAL is a LATENT objective — the oracle scores utility on it and the symptom
    # trap CANNOT touch it. `outcome` (above) is only its OBSERVED surrogate readout, which
    # the agent measures and the trap can move. Skins name the goal explicitly (deeper
    # objective the metric proxies); fall back to an internal latent name otherwise.
    goal = S.get("goal") or {"name": f"Underlying{outcome['name']}",
                             "aliases": ["underlying true objective"]}
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
    # NB: surrogate_trap no longer builds a SEPARATE controllable metric. Post-split the
    # PRIMARY observed metric (`outcome`) is itself a surrogate readout of the latent goal,
    # and the universal symptom trap moves it with a real edge -> that IS the
    # surrogate-endpoint challenge. Merging avoids a world with two surrogates.
    # reserve the reversal confounder up-front (confounded_reversal archetype)
    rev_conf = None
    if arche == "confounded_reversal":
        _rc = _take(rng, S["confounder_pool"], 1, used) or _take(rng, S["inert_var_pool"], 1, used)
        if _rc:
            rev_conf = _rc[0]

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
    if arche == "confounded_reversal" and rev_conf is not None:
        # Simpson's / confounding-by-indication: the source knob is OBSERVATIONALLY
        # driven by a hidden confounder that also (strongly, positively) drives the
        # outcome -- so observationally the knob looks BENEFICIAL, while its causal
        # effect (via the chain) is HARMFUL. Only intervention reveals the true sign.
        add_var(rev_conf, kind="latent", dist={"normal": [50, 16]})
        add_var(src, kind="observable", parents=[rev_conf["name"]],
                mech={"form": "linear", "weights": {rev_conf["name"]: 1.2}, "intercept": 0},
                measurable=True, assay_noise={"normal": [0, 3]})
    else:
        add_var(src, kind="observable", dist={"normal": [40, 12]},
                measurable=True, assay_noise={"normal": [0, 3]})
    if arche != "competing_causes":
        add_act(src, target=src["name"], op="set", dtype="continuous",
                range=[0, 100], default=40, description=f"controller for {src['name']}")
    # (competing_causes: no source-knob lever -- cause 1's ONLY lever is its fix, so
    #  the two causes have exactly one clean lever each.)
    if src2 is not None:
        add_var(src2, kind="observable", dist={"normal": [20, 8]},
                measurable=True, assay_noise={"normal": [0, 3]})
        add_act(src2, target=src2["name"], op="set", dtype="continuous",
                range=[0, 100], default=20, description=f"controller for {src2['name']}")

    # ---- ROOT node ----
    if "two_cause" in feats:
        # AND-gate: root high only if BOTH source knobs clear thresholds.
        # RPG_SYNERGY_SOFT=<ma>: super-additive redesign — each lever gives a modest
        # individual effect (findable, partial credit, below accept bar) + dominant joint
        # synergy. NOW DEFAULT 20 (the fix): the original hard AND (ma=0) was unsolvable
        # within the 15-experiment budget (1-of-28 pair) and, combined with the inverted
        # proxy polarity, gave Opus part_a ~0.13. Set RPG_SYNERGY_SOFT=0 to reproduce the
        # old hard-AND design for provenance. See box1_clean_eval/SYNERGY_REDESIGN_REPORT.md.
        import os as _os
        _ms = float(_os.environ.get("RPG_SYNERGY_SOFT", "20"))
        add_var(root, kind="latent", parents=[src["name"], src2["name"]],
                mech={"form": "gated_and", "a": src["name"], "b": src2["name"],
                      "ta": 55, "tb": 55, "wa": 9, "wb": 9,
                      "vmax": (55.0 if _ms else 95.0), "intercept": 3.0, "ma": _ms, "mb": _ms})
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

    # ---- competing-causes augmentation (two independent causes) ----
    # A SECOND, independent cause with its own controllable source, root, and fix,
    # feeding the outcome ADDITIVELY at a weaker weight than chain 1. Neither cause
    # alone recovers >=90% of the achievable benefit, so the agent must recognize
    # there are TWO causes and address BOTH -- stopping at the first one it finds
    # leaves most of the benefit on the table. Built before the outcome so root2 is
    # an outcome parent.
    # competing_causes: a SECOND, independent cause (exogenous latent + its own fix).
    # The initial weight below is a placeholder -- it is CALIBRATED after the SCM is
    # built (see the weight search) so that the dominant single cause recovers ~60%
    # of the benefit (both single fixes below the Part-A bar; both together ~all),
    # making the two causes jointly necessary. Design: one clean lever per cause;
    # the decoy band is relaxed for this archetype (oracle_v6) so cause 2 need not be
    # weakened to keep the confounder->decoy correlation.
    competing_extra: Dict[str, float] = {}
    cc_root2 = cc_fix2 = None
    if arche == "competing_causes":
        # Cause 2 is an EXOGENOUS latent driving the outcome, with a SINGLE lever
        # (its own fix). One clean lever per cause avoids the source-vs-fix
        # ambiguity; draw the name from the root pool, falling back to the (large)
        # inert pool so every skin can build a second cause.
        _r2 = _take(rng, S["root_cause_pool"], 1, used) or _take(rng, S["inert_var_pool"], 1, used)
        if _r2:
            r2 = _r2[0]
            cc_root2 = r2["name"]
            add_var(r2, kind="latent", dist={"normal": [55, 18]})
            # DE-LEAK (name leakage): the second cause's fix actuator must NOT name the cause.
            # It used aliases[0]=f"reduce {r2['aliases'][0]}" and description=f"dosing to reduce
            # {cc_root2}", so the catalog printed the true cause on the control label and the model
            # shortcut to it (18/25 competing_causes traces cited it before experimenting). Mirror the
            # MAIN fix: draw a NEUTRAL remedy name/alias from the skin's fix_actuator_pool (fallback to
            # the inert pool if exhausted) so the label describes a plausible treatment without
            # revealing which cause it addresses. Mechanics (target/op/expr) are unchanged.
            _f2 = _take(rng, S["fix_actuator_pool"], 1, used) or _take(rng, S["inert_var_pool"], 1, used)
            _f2e = _f2[0] if _f2 else {"name": f"Fix{cc_root2}", "aliases": ["adjustable process control"]}
            cc_fix2 = add_act(_f2e,
                              target=cc_root2, op="scale", dtype="continuous",
                              range=[0, 100], default=0, expr="1-sat(d;k=0.66)",
                              description="an adjustable process control")
            # placeholder; calibrated after the SCM is built (see the weight search).
            competing_extra = {cc_root2: 0.5 * chain_sign}

    # ---- GOAL (latent true objective) driven by last mediator (chain_sign) + small
    # confounder path. This is the TRUE causal chain terminus; the oracle scores utility
    # HERE. It is LATENT (no assay) — the trap cannot touch it and the agent cannot read it
    # directly; the agent must infer goal-recovery from the mechanism proxy.
    weights = {last_mediator: 0.9 * chain_sign}
    for c in confs:
        weights[c["name"]] = 0.1 * (1 if higher_better else -1)
    weights.update(subtype_extra)
    weights.update(competing_extra)
    # confounded_reversal: the reversal confounder drives the GOAL POSITIVELY and strongly
    # (weight calibrated after the SCM is built) so the source knob's observational
    # correlation with the observed surrogate is OPPOSITE its causal sign. rev_conf is a
    # common cause of src (observed) and the goal -> confounding by indication.
    reversal_extra = ({rev_conf["name"]: 1.0}
                      if (arche == "confounded_reversal" and rev_conf is not None) else {})
    weights.update(reversal_extra)
    intercept = 15 if chain_sign < 0 else 8
    add_var(goal, kind="latent",
            parents=[last_mediator] + [c["name"] for c in confs]
                    + list(subtype_extra) + list(competing_extra) + list(reversal_extra),
            mech={"form": "linear", "weights": weights, "intercept": intercept})
    # ---- OBSERVED SURROGATE: a faithful, noisy readout of the latent goal (goal -> outcome).
    # This is the metric the agent measures and optimizes; utility is NOT scored here. Under
    # an honest fix goal recovers -> surrogate recovers; the trap adds a REAL edge to THIS
    # node only (below), raising the surrogate with zero effect on the goal or the proxy.
    add_var(outcome, kind="outcome", parents=[goal["name"]],
            mech={"form": "linear", "weights": {goal["name"]: 1.0}, "intercept": 0},
            measurable=True, assay_noise={"normal": [0, 3]})

    # ---- true mechanism proxy: attaches to a mediator (>=1 hop downstream) ----
    # Assay noise kept LOW so the fix-induced proxy shift clears the oracle's validity gate
    # (counterfactual_battery admits a proxy only if shift/sd > 0.5; proxy_signal_audit wants
    # interventional shift/sd > 1.0). With the goal now LATENT and the surrogate gameable (in
    # surrogate_trap), the proxy is the agent's honest verification channel, so it MUST be a
    # detectable signal — a high-noise proxy would make the oracle's valid-proxy set degenerate
    # AND the skill unlearnable. (Was SD 12 -> shift/sd ~0.08; re-audited per archetype x skin.)
    proxy_parent = mediators[min(1, len(mediators) - 1)]["name"]
    add_var(proxy, kind="observable", parents=[proxy_parent],
            mech={"form": "linear", "weights": {proxy_parent: 0.8}, "intercept": 5},
            measurable=True, assay_noise={"normal": [0, 4]})

    # ---- confounded decoys: driven by a confounder (zero do-effect on outcome) ----
    for d in decoys:
        cparent = _choice(rng, confs)["name"]
        add_var(d, kind="observable", parents=[cparent],
                mech={"form": "linear", "weights": {cparent: 0.7}, "intercept": 40},
                measurable=True, assay_noise={"normal": [0, 4]})

    # ---- FIX actuator: scales the root down (reduces the true cause) ----
    if "two_cause" in feats or arche in ("instrument_only", "confounded_reversal"):
        # two_cause: the "fix" is applying BOTH source knobs (co-actuators); no
        # separate scale-down fix. instrument_only: the cause has NO direct handle
        # at all -- the only lever is the upstream source knob (`src`). Either way
        # no fix actuator is added; targeted_actuator falls back to `src` below.
        fix_id = None
    else:
        side = None
        if "interior_dose" in feats:
            # over-treatment harms the TRUE GOAL (a real therapeutic-window penalty),
            # not merely the surrogate reading -> target the latent goal.
            side = {"target": goal["name"],
                    "expr": ("-overstrip(d;thr=0.66,gain=30)" if higher_better
                             else "overstrip(d;thr=0.66,gain=30)")}
        eff = {"target": root["name"], "op": "scale", "dtype": "continuous",
               "range": [0, 100], "default": 0, "expr": "1-sat(d;k=0.66)",
               "description": "a continuous dosing control"}
        if side:
            eff["side_effect"] = side
        fix_id = add_act(fix, **eff)

    # ---- TRAP actuator: a REAL edge on the observed surrogate (surrogate-endpoint trap) ----
    # Dosing this GENUINELY raises the observed surrogate (op="add" on `outcome`, which is a
    # readout of the latent goal), but it has ZERO path to the goal -> utility is unchanged
    # and the mechanism proxy stays flat. So a plausible treatment name (e.g. "clarifying
    # polymer") is now honest: it really moves the metric; it just doesn't fix the cause.
    # (Replaces the old op="mask" reading-bias, which was chemically incoherent for any real
    # treatment name and forced a role-leaking description.)
    trap_id = None
    if "symptom_trap" in feats:
        expr = "transient_boost(d)" if higher_better else "-transient_boost(d)"
        trap_id = add_act(trap, target=outcome["name"], op="add", dtype="continuous",
                          range=[0, 100], default=0, expr=expr,
                          description="an adjustable process control")

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

    scm = WorldSCM(variables=V, actuators=A, outcome=outcome["name"], goal=goal["name"],
                   higher_is_better=higher_better, selection=selection)

    # ---- competing_causes: calibrate cause-2's outcome weight so the two causes
    # are JOINTLY NECESSARY -- the dominant single cause recovers ~60% of the
    # benefit, so each single fix is below the Part-A bar while both together
    # recover ~all. Per-world search -> robust across skins without hardcoded %.
    if arche == "competing_causes" and cc_root2 is not None and fix_id and cc_fix2:
        ow = scm.variables[goal["name"]]["mech"]["weights"]   # cause-2 feeds the GOAL
        hi1 = scm.actuators[fix_id]["range"][1]
        hi2 = scm.actuators[cc_fix2]["range"][1]
        def _eu(iv):
            v = scm.sample(12000, intervention=iv, seed=seed + 321)
            return float(np.mean(scm.utility(v)))
        base_u = _eu({})
        best_w, best_err = None, 1e9
        for k in range(2, 20):                       # w2 in 0.2 .. 1.9
            w2 = round(0.1 * k, 2)
            ow[cc_root2] = w2 * chain_sign
            u1 = _eu({fix_id: hi1}); u2 = _eu({cc_fix2: hi2})
            ub = _eu({fix_id: hi1, cc_fix2: hi2})
            d = ub - base_u
            if d <= 1e-6:
                continue
            f1 = (u1 - base_u) / d; f2 = (u2 - base_u) / d
            dom = max(f1, f2)
            if dom < 0.85:                           # both single fixes below the bar (margin)
                err = abs(dom - 0.6)                 # aim: dominant cause ~60%
                if err < best_err:
                    best_err, best_w = err, w2
        ow[cc_root2] = (best_w if best_w is not None else 0.5) * chain_sign

    # confounded_reversal: calibrate the reversal confounder's outcome weight so the
    # source knob's OBSERVATIONAL correlation with the outcome is POSITIVE (looks
    # beneficial) while its causal effect stays NEGATIVE (harmful) -- the sign reverses
    # between observation and intervention (Simpson's / confounding by indication).
    if arche == "confounded_reversal" and rev_conf is not None:
        ow = scm.variables[goal["name"]]["mech"]["weights"]   # rev_conf feeds the GOAL
        rc = rev_conf["name"]
        # the OBSERVED correlation the agent sees is corr(src, observed surrogate); search
        # the rev_conf->goal weight so this is positive while the causal sign stays negative.
        def _obs_corr():
            v = scm.sample(12000, seed=seed + 733)
            o = scm.measure(v, [src["name"], outcome["name"]], seed=seed + 734)
            return float(np.corrcoef(o[src["name"]], o[outcome["name"]])[0, 1])
        best_w, best_err = None, 1e9
        for k in range(1, 26):                       # reversal weight 0.2 .. 5.0
            ow[rc] = round(0.2 * k, 2)
            c = _obs_corr()
            if c >= 0.25:                            # observational corr clearly positive
                err = abs(c - 0.4)
                if err < best_err:
                    best_err, best_w = err, ow[rc]
        ow[rc] = best_w if best_w is not None else 3.0

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
    if arche == "competing_causes" and fix_id and cc_fix2:
        # both fixes are required (neither cause alone clears the Part-A bar); the
        # informative/required intervention is the JOINT one.
        co_acts = [fix_id, cc_fix2]
    naive = []
    # obvious moves = pushing the source knob the "intuitive" direction, and inert knobs
    if "sign_flip" in feats:
        naive.append({src["name"]: 100})       # "more of the input" (wrong: harms)
    if sub_info is not None:
        naive.append({sub_info["treatment_actuator"]: 100})   # treat everyone -> ~0
    for iv in inert_vars[:2]:
        naive.append({f"set_{iv['name']}": 100})
    if trap_id is not None:
        # dosing the trap raises the observed surrogate but has ZERO effect on the goal ->
        # the canonical "optimized the metric, not the cause" mistake (fails Part A).
        naive.append({trap_id: 100})
    if arche == "confounded_reversal":
        # observationally the source knob looks beneficial, so "turn it up" is the
        # obvious move -- but it is causally HARMFUL (the sign reverses under do()).
        naive.append({src["name"]: 100})
    # NB: don't add single AND-gate knobs to naive -- the gate is asymmetric (one
    # knob alone can partially fire when the other sits near its threshold), so a
    # single knob may recover enough to "help" and trip counterintuitiveness. The
    # "single lever fails Part A" property is verified by G4 instead; inert knobs
    # (added above) carry the counterintuitiveness check.

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
             "hidden_subtype": "subtype", "surrogate_trap": "surrogate",
             "instrument_only": "instrument", "competing_causes": "competing",
             "synergy_pair": "synergy", "dose_window": "dosewin",
             "confounded_reversal": "reversal"}.get(arche, arche)
    wid = f"v7_{skin_name}_{_atag}_{'_'.join(sorted(feats))[:20]}_{seed}"
    return {"world_id": wid, "domain": skin_name, "scenario": scenario,
            "scm": scm, "ground_truth": gt}


def _build_scenario(S: Dict[str, Any], feats: set) -> str:
    return S["scenario"].format(**S["fills"])
