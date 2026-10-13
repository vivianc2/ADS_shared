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

import os
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


def _draw_v91_values(seed: int) -> Dict[str, Any]:
    """v9.1 value randomisation (sample_world(vary_values=True)). Own stream, fixed draw
    order (every key drawn for every world, used or not) so a key's value depends only on
    the seed. Ranges: personal_docs results/week6_ygg/M7_V91_SAMPLER_2026-10-08.md."""
    r = random.Random(f"rpg_v91_values:{seed}")
    return {
        # True -> raising the source knob REDUCES the cause (root driven by M - knob)
        "src_up": r.random() < 0.5,
        # therapeutic window: the fix removes the cause fully at dose k and harms past thr
        # (k = thr -> a single interior optimum). Overdose gain 60 (v9: 30) narrows the
        # part-A window to ~10 dose points (v9 gain: ~16) so no constant covers much of the
        # range. Bounds: at 0.85 "max the knob" still costs 60*0.15 = 9 utility (> 10% of the
        # ~30-40 benefit); at 0.15 dose 50 still helps, so the oracle's 0/50/100 screen finds it.
        "window": round(r.uniform(0.15, 0.85), 2),
        # plain (no window) fixes: True -> "turn it down" control (engine lin), gold 0
        "fix_down": r.random() < 0.5,
        "fix2_down": r.random() < 0.5,
        # AND-gate co-actuators: gate direction (+1 rises with the knob, -1 falls) and the
        # threshold's distance from the common baseline 50
        "gate_dir_a": 1 if r.random() < 0.5 else -1,
        "gate_dir_b": 1 if r.random() < 0.5 else -1,
        "gate_gap_a": round(r.uniform(12.0, 25.0), 1),
        "gate_gap_b": round(r.uniform(12.0, 25.0), 1),
        # hidden_subtype: True -> the treatment helps the LOW-marker subgroup
        "subtype_flip": r.random() < 0.5,
    }


WORLD_VERSIONS = ("v9", "v9.1", "v10")
# v10: archetypes whose source knob is a lever and gets a per-world SETPOINT (interior optimum)
_SETPOINT_ARCHES = {"confounded_chain", "collider_selection", "surrogate_trap", "hidden_subtype",
                    "instrument_only"}
# v10: archetypes whose root fix gets a (visible) therapeutic window with probability 1/2
_WINDOW_ARCHES = {"confounded_chain", "collider_selection", "surrogate_trap", "hidden_subtype"}


def _resolve_version(version: Optional[str], vary_values: Optional[bool]) -> str:
    """Explicit `version` > explicit `vary_values` (True -> v9.1, False -> v9) > env RPG_WORLD_VERSION >
    env RPG_V91_VARY_VALUES (v9.1) > v9."""
    if version is None and vary_values is not None:
        version = "v9.1" if vary_values else "v9"
    if version is None:
        version = os.environ.get("RPG_WORLD_VERSION", "") or ("v9.1" if _vary_values_default() else "v9")
    if version not in WORLD_VERSIONS:
        raise ValueError(f"unknown world version {version!r}; expected one of {WORLD_VERSIONS}")
    return version


def _draw_v10_values(seed: int) -> Dict[str, Any]:
    """v10 draws (personal_docs plans/V10_WORLDGEN_PLAN_2026-10-12.md). Own stream, fixed draw order, so every
    value depends only on the seed and the v9.1 base world of that seed is unchanged by these draws."""
    r = random.Random(f"rpg_v10_values:{seed}")
    mu = round(r.uniform(25.0, 75.0), 1)                 # source knob's observational mean
    gap, side = r.uniform(25.0, 45.0), (1 if r.random() < 0.5 else -1)
    sp = mu + side * gap                                 # setpoint: 25-45 from the baseline level
    if not 5.0 <= sp <= 95.0:
        sp = mu - side * gap
    sym_p = [round(r.uniform(0.20, 0.45), 3) for _ in range(3)]
    d = {
        "src_mu": mu, "setpoint": round(sp, 1),
        "extra_window": r.random() < 0.5,
        "n_controls": r.randint(22, 28),                 # total controls; budget is 15
        "n_sym": r.choice([2, 3]), "sym_p": sym_p,       # symptomatic relief levers and their max relief
        "n_harm": r.choice([2, 3]),
        "harm_knee_set": [round(r.uniform(62.0, 78.0), 1) for _ in range(3)],
        "harm_knee_dose": [round(r.uniform(30.0, 60.0), 1) for _ in range(3)],
        # goal harm at the control's max, as a FRACTION of the world's achievable benefit (calibrated after the SCM
        # is built), so harm controls are "not nothing" (>= 10% of the gold benefit) in every world
        "harm_frac": [round(r.uniform(0.12, 0.30), 3) for _ in range(3)],
        "harm_set_first": r.random() < 0.5,
        "names_seed": r.randrange(1 << 30),
    }
    # interior golds everywhere (2026-10-13), drawn AFTER every earlier value so those are unchanged:
    #   AND-gate co-levers: own usual level + a setpoint 25-45 away (the gate opens near the setpoint);
    #   reversal lever: usual level 50, setpoint 25-45 to one side; competing fixes: one dose window each.
    def _sp(m, g, side):
        x = m + side * g
        return round(x if 5.0 <= x <= 95.0 else m - side * g, 1)
    gmu = [round(r.uniform(25.0, 75.0), 1) for _ in range(2)]
    ggap = [r.uniform(25.0, 45.0) for _ in range(2)]
    gside = [1 if r.random() < 0.5 else -1 for _ in range(2)]
    rgap, rside = r.uniform(25.0, 45.0), (1 if r.random() < 0.5 else -1)
    d.update({
        "gate_mu": gmu, "gate_sp": [_sp(gmu[i], ggap[i], gside[i]) for i in range(2)],
        "rev_sp": _sp(50.0, rgap, rside),
        "cc_window": [round(r.uniform(0.15, 0.85), 2) for _ in range(2)],
    })
    return d


def _take(rng, pool: List[Dict[str, Any]], k: int, used: set) -> List[Dict[str, Any]]:
    """Draw k distinct entries whose names are not already used."""
    avail = [x for x in pool if x["name"] not in used]
    rng.shuffle(avail)
    picked = avail[:k]
    for x in picked:
        used.add(x["name"])
    return picked


def _vary_values_default() -> bool:
    return os.environ.get("RPG_V91_VARY_VALUES", "0") not in ("", "0", "false", "False")


def world_kwargs(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """sample_world kwargs for a dataset row's extra_info: explicit world_version, else the v9.1 vary_values flag,
    else nothing (-> env RPG_WORLD_VERSION / RPG_V91_VARY_VALUES / v9). Every world-building call site uses this
    so the version recorded in the dataset is the version that is rebuilt."""
    info = info or {}
    if info.get("world_version"):
        return {"version": str(info["world_version"])}
    if info.get("vary_values"):
        return {"vary_values": True}
    return {}


def version_extra_info(version: str) -> Dict[str, Any]:
    """What to record in extra_info for a world built at `version` (v9 rows stay byte-identical: no key)."""
    if version == "v10":
        return {"world_version": "v10"}
    if version == "v9.1":
        return {"vary_values": True}
    return {}


def sample_world(seed: int, skin: Optional[str] = None,
                 features: Optional[List[str]] = None,
                 archetype: Optional[str] = None,
                 vary_values: Optional[bool] = None,
                 version: Optional[str] = None) -> Dict[str, Any]:
    """Sample one structurally-randomized world. Deterministic in `seed`.

    ``archetype`` selects the role-wiring family (see ARCHETYPES). Defaults to
    a random draw. All archetypes share the same engine/oracle/audits.

    ``vary_values`` (v9.1, OPT-IN; None -> env RPG_V91_VARY_VALUES, default off): randomise
    per world the VALUE the gold answer needs, so it must be inferred from experiments
    instead of learned as a benchmark constant (v9: dose_window always (0,66),
    instrument_only / confounded_reversal always 0, synergy_pair / competing_causes always
    (100,100); personal_docs results/week5_pando/BENCHMARK_REGULARITY_A4_2026-10-02.md).
    Drawn from a SEPARATE per-world stream ``random.Random(f"rpg_v91_values:{seed}")`` so the
    main ``rng`` sequence -- and hence names, structure, features and the id catalog -- is
    the same as the v9 world of that seed, and with the option off nothing is drawn
    (v9 worlds byte-identical). What varies (see ``_draw_v91_values``):
      - source-knob direction: half the worlds the cause falls as the knob RISES
        (instrument_only, confounded_reversal, chain-type worlds); the reversal confounder's
        sign follows so the observational sign still reverses;
      - therapeutic-window location (interior_dose / dose_window): k = thr ~ U[0.15, 0.85],
        overdose gain 60 (v9: 30);
        dose_window additionally drops the source-knob lever (it bypassed the window);
      - plain fix direction (competing_causes' two fixes, chain-type fixes): half are
        "turn it down" controls (engine "lin": untouched = 50, gold 0, raising worsens);
      - synergy / two_cause AND-gate: each co-actuator's gate direction and threshold;
      - hidden_subtype: which subgroup the treatment helps.
    NB the first observation (catalog) is IDENTICAL on/off for every archetype except
    dose_window, so the env's prompt drift guard cannot tell them apart -- carry the flag
    explicitly (extra_info["vary_values"]; skyrl_rpg env / rebuild_v9_sets / rpg_dataset do).

    ``version`` ("v9" | "v9.1" | "v10"; see _resolve_version). "v10" = the v9.1 values PLUS the v10 layers
    (personal_docs plans/V10_WORLDGEN_PLAN_2026-10-12.md): 22-28 controls against the 15-experiment budget
    (decoy handles, small-harm levers, symptomatic-relief levers, inert fill, both control types in every role),
    a per-world setpoint for the source lever, visible therapeutic windows (engine propagate_side_effects), and
    one label format for every control (v9 labels inert knobs "set <x>" and real levers plainly -- a tell).
    v9 / v9.1 worlds are unchanged (byte-identical)."""
    version = _resolve_version(version, vary_values)
    vary_values = version in ("v9.1", "v10")
    rng = random.Random(seed)
    vv = _draw_v91_values(seed) if vary_values else None
    v10 = _draw_v10_values(seed) if version == "v10" else None
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
        if vary_values:
            # v9.1: sign_flip only reshapes the (now lever-less) source arm and shrinks the
            # achievable benefit to ~6, so a window below ~0.3 is missed by the oracle's
            # 0/50/100 screen (empty gold). Plain source -> benefit ~30-40, window always found.
            feats.discard("sign_flip")
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
    # v10: per-world setpoint on the source lever (replaces sign_flip / the v9.1 direction coin on that lever),
    # and a therapeutic window on half of the chain-type root fixes.
    _setpoint = bool(v10 is not None and arche in _SETPOINT_ARCHES and "two_cause" not in feats)
    if _setpoint:
        feats.discard("sign_flip")
    if (v10 is not None and arche in _WINDOW_ARCHES and "two_cause" not in feats
            and "interior_dose" not in feats):
        # (2026-10-13: every v10 fix is windowed -- a plain fix had gold "anything >= 66", i.e. the max)
        feats.add("interior_dose")

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

    # v10 (rebuild 2026-10-12): every CONTROL name comes from one per-skin pool per control type -- level settings
    # (source knobs + extra process settings) and doses (fixes + traps + extra additives) -- and only names with a
    # physical unit (skins_v10.LEVEL_UNITS / DOSE_UNITS) qualify. Roles are assigned by draw order, so a name is the
    # true lever in one world and inert in another (v9/early v10: role-specific pools, a name -> role lookup learned
    # on train found the exact levers in 270/270 in-domain test worlds). The mechanism marker is drawn to match the
    # direction it moves (skins_v10.GOOD_MARKERS). Main-rng draws above are unchanged; the redrawn names are released.
    rno = None
    if v10 is not None:
        from skins_v10 import EXTRA as _XV, LEVEL_UNITS as _LU, DOSE_UNITS as _DU, GOOD_MARKERS as _GOOD, \
            EXTRA_BAD_MARKERS as _XBAD, EXTRA_GOOD_MARKERS as _XGOOD
        rno = random.Random(v10["names_seed"] + 11)
        LEVEL_POOL = [e for e in S["source_knob_pool"] + _XV[skin_name]["extra_process_pool"] if e["name"] in _LU[skin_name]]
        DOSE_POOL = [e for e in S["fix_actuator_pool"] + S["trap_actuator_pool"] + _XV[skin_name]["extra_additive_pool"]
                     if e["name"] in _DU[skin_name]]
        BAD_MARKERS = [p for p in S["proxy_pool"] + _XBAD[skin_name] if p["name"] not in _GOOD]
        for e in [src, fix, trap, proxy] + ([src2] if src2 is not None else []):
            used.discard(e["name"])
        src = _take(rno, LEVEL_POOL, 1, used)[0]
        fix = _take(rno, DOSE_POOL, 1, used)[0]
        trap = _take(rno, DOSE_POOL, 1, used)[0]
        if src2 is not None:
            src2 = _take(rno, LEVEL_POOL, 1, used)[0]
        # the marker tracks a mediator: it rises with the problem (bad marker) unless the chain is an AND-gate, where
        # the root is the GOOD state and the marker rises as the outcome recovers
        _mpool = ([p for p in S["proxy_pool"] + _XGOOD[skin_name] if p["name"] in _GOOD] if "two_cause" in feats
                  else BAD_MARKERS)
        proxy = _take(rno, _mpool, 1, used)[0]

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
                # v10: centred on 50 (setpoint 25-45 to one side, see _rev_sp)
                mech={"form": "linear", "weights": {rev_conf["name"]: 1.2},
                      "intercept": (-10.0 if v10 is not None else 0)},
                measurable=True, assay_noise={"normal": [0, 3]})
    else:
        # v9.1 AND-gate: both co-actuators sit at a common baseline 50 with the gate threshold
        # 50 +/- gap on the side of its direction, so the baseline level does not cue direction.
        _gate = vv is not None and "two_cause" in feats
        _mu = v10["src_mu"] if _setpoint else (50 if _gate else 40)
        if v10 is not None and "two_cause" in feats:
            _mu = v10["gate_mu"][0]
        add_var(src, kind="observable", dist={"normal": [_mu, 12]},
                measurable=True, assay_noise={"normal": [0, 3]})
    # v9.1 dose_window: no source-knob lever either. In v9 the source knob at its extreme
    # removes the cause on its own (src->0 with the fix untouched passes part A in ~60% of
    # v9 dose worlds, ~87% once the window moves), so the window was optional; with the fix
    # as the ONLY lever the interior dose is the task.
    _dose_only_fix = vv is not None and arche == "dose_window"
    if arche != "competing_causes" and not _dose_only_fix:
        add_act(src, target=src["name"], op="set", dtype="continuous",
                range=[0, 100], default=(int(round(v10["src_mu"])) if _setpoint else
                                         int(round(v10["gate_mu"][0])) if (v10 is not None and "two_cause" in feats) else
                                         50 if (v10 is not None and arche == "confounded_reversal") else 40),
                description=f"controller for {src['name']}")
    # (competing_causes: no source-knob lever -- cause 1's ONLY lever is its fix, so
    #  the two causes have exactly one clean lever each.)
    if src2 is not None:
        _mu2 = v10["gate_mu"][1] if v10 is not None else (50 if vv is not None else 20)
        add_var(src2, kind="observable", dist={"normal": [_mu2, 8]},
                measurable=True, assay_noise={"normal": [0, 3]})
        add_act(src2, target=src2["name"], op="set", dtype="continuous",
                range=[0, 100], default=(int(round(_mu2)) if v10 is not None else 20),
                description=f"controller for {src2['name']}")
    # v9.1 source direction: when src_up, the root is driven by the knob's SHORTFALL
    # (M - knob), so RAISING the knob removes the cause (mirror of v9, where lowering it
    # does). The knob's own baseline is unchanged (no level cue). M = 80 mirrors the v9
    # baseline N(40,12) exactly (same root distribution as v9 -> same audit behaviour; with
    # M = 100 the larger root saturated the chain and ~40% of collider worlds failed the decoy
    # audit); the cause is then gone for knob >= ~80. Reversal knob ~N(60,19): M = 100 (cap).
    # Not for the AND-gate (direction drawn per co-actuator below) nor competing_causes.
    src_drive = src["name"]
    _rev_sp = bool(v10 is not None and arche == "confounded_reversal" and rev_conf is not None)
    src_up = bool(vv is not None and vv["src_up"] and "two_cause" not in feats
                  and arche != "competing_causes" and not _setpoint and not _rev_sp)
    # direction in which RAISING the source lever helps (v9.1 src_up, or a v10 reversal setpoint above its level 50)
    _up_dir = bool(src_up or (_rev_sp and v10["rev_sp"] > 50.0))
    if src_up:
        src_drive = f"{src['name']}Shortfall"
        add_var({"name": src_drive, "aliases": ["shortfall"]}, kind="latent", parents=[src["name"]],
                mech={"form": "linear", "weights": {src["name"]: -1.0},
                      "intercept": (100.0 if arche == "confounded_reversal" else 80.0)})
    if _setpoint or _rev_sp:
        # v10: the cause grows with the knob's DISTANCE from a per-world setpoint (abs form), so the best setting
        # is interior and random (v9: 0; v9.1: 0 or 100). Baseline distance 25-45, like the v9 baseline drive.
        # Reversal: the setpoint lies 25-45 to one side of the lever's level (~N(50,19)), so over the observed range
        # moving AWAY from it hurts and the confounder can still reverse the observational sign.
        src_drive = f"{src['name']}Deviation"
        add_var({"name": src_drive, "aliases": ["setpoint deviation"]}, kind="latent", parents=[src["name"]],
                mech={"form": "abs", "of": src["name"], "center": (v10["rev_sp"] if _rev_sp else v10["setpoint"]),
                      "gain": 1.0, "intercept": 0.0})

    # ---- ROOT node ----
    if "two_cause" in feats:
        # AND-gate: root high only if BOTH source knobs clear thresholds.
        # RPG_SYNERGY_SOFT=<ma>: super-additive redesign — each lever gives a modest
        # individual effect (findable, partial credit, below accept bar) + dominant joint
        # synergy. NOW DEFAULT 20 (the fix): the original hard AND (ma=0) was unsolvable
        # within the 15-experiment budget (1-of-28 pair) and, combined with the inverted
        # proxy polarity, gave Opus part_a ~0.13. Set RPG_SYNERGY_SOFT=0 to reproduce the
        # old hard-AND design for provenance (synergy redesign).
        import os as _os
        _ms = float(_os.environ.get("RPG_SYNERGY_SOFT", "20"))
        _ta, _tb, _wa, _wb = 55, 55, 9, 9
        if vv is not None:
            # v9.1: per co-actuator gate direction; a negative width makes the sigmoid FALL
            # with the knob (engine _gated_and), so that lever's gold is 0, not 100.
            _ta = round(50 + vv["gate_dir_a"] * vv["gate_gap_a"], 1)
            _tb = round(50 + vv["gate_dir_b"] * vv["gate_gap_b"], 1)
            _wa, _wb = 9 * vv["gate_dir_a"], 9 * vv["gate_dir_b"]
        _ga, _gb = src["name"], src2["name"]
        if v10 is not None:
            # v10 (2026-10-13): each gate opens near its co-lever's own SETPOINT (distance < 15, edge width 3), so both
            # golds are interior and random; at the usual level (25-45 away) the gate is ~closed. Gate >= 0.95 within
            # +-6 of the setpoint, so part A needs each value within ~6% of its range.
            _ga, _gb = f"{src['name']}Deviation", f"{src2['name']}Deviation"
            for _nm, _of, _c in ((_ga, src["name"], v10["gate_sp"][0]), (_gb, src2["name"], v10["gate_sp"][1])):
                add_var({"name": _nm, "aliases": ["setpoint deviation"]}, kind="latent", parents=[_of],
                        mech={"form": "abs", "of": _of, "center": _c, "gain": 1.0, "intercept": 0.0})
            _ta, _tb, _wa, _wb = 15.0, 15.0, -3.0, -3.0
        add_var(root, kind="latent", parents=[_ga, _gb],
                mech={"form": "gated_and", "a": _ga, "b": _gb,
                      "ta": _ta, "tb": _tb, "wa": _wa, "wb": _wb,
                      "vmax": (55.0 if _ms else 95.0), "intercept": 3.0, "ma": _ms, "mb": _ms})
        # for two_cause, "high root" is GOOD (it's the required-uptake style),
        # so the chain to outcome is positive; handled below via chain_sign.
        chain_sign = +1.0
    elif "sign_flip" in feats:
        # source helps then hurts: derived "harm axis" rises past a knee, times severity
        harm = f"{src['name']}HarmAxis"
        add_var({"name": harm, "aliases": ["harm axis"]},
                kind="latent", parents=[src_drive],
                mech={"form": "sign_flip", "of": src_drive, "knee": 45,
                      "lo_gain": 0.05, "hi_gain": 0.9, "intercept": 2.0})
        add_var(root, kind="latent", parents=[harm, sev_name],
                mech={"form": "interaction", "a": harm, "b": sev_name,
                      "gain": 260.0, "scale": 100.0, "intercept": 3.0},
                noise={"normal": [0, 1.5]})
        chain_sign = -1.0   # more root -> worse outcome
    else:
        # plain: root rises with source × severity
        add_var(root, kind="latent", parents=[src_drive, sev_name],
                mech={"form": "interaction", "a": src_drive, "b": sev_name,
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
            if v10 is not None:
                used.discard(treat["name"])
                treat = _take(rno, DOSE_POOL, 1, used)[0]
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
                          "center": center, "scale": 100.0,
                          # v9.1: which subgroup is helped (negative gain -> the LOW-marker one)
                          "gain": (-46.0 if (vv is not None and vv["subtype_flip"]) else 46.0)})
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
            if v10 is not None:
                if _f2:
                    used.discard(_f2[0]["name"])
                _f2 = _take(rno, DOSE_POOL, 1, used)
            _f2e = _f2[0] if _f2 else {"name": f"Fix{cc_root2}", "aliases": ["adjustable process control"]}
            _f2down = vv is not None and vv["fix2_down"] and v10 is None
            _f2kw = {}
            if v10 is not None:
                # v10 (2026-10-13): cause 2's fix also has its own therapeutic window (random interior gold)
                _w2 = v10["cc_window"][1]
                _f2kw = {"side_effect": {"target": goal["name"], "expr": f"-overstrip(d;thr={_w2},gain=60)"}}
            cc_fix2 = add_act(_f2e,
                              target=cc_root2, op="scale", dtype="continuous",
                              # v9.1 "turn it down" fix (engine "lin"): untouched = dose 50,
                              # lowering removes the cause, raising worsens it
                              range=[0, 100], default=(50 if _f2down else 0),
                              expr=("lin(d;k=0.5)" if _f2down else
                                    f"1-sat(d;k={v10['cc_window'][1]})" if v10 is not None else "1-sat(d;k=0.66)"),
                              description="an adjustable process control", **_f2kw)
            # placeholder; calibrated after the SCM is built (see the weight search).
            competing_extra = {cc_root2: 0.5 * chain_sign}
            if v10 is not None:
                # v10: cause 2 gets a measurable marker. In v9 it has none, so its fix only lifts the outcome and is
                # indistinguishable from symptomatic relief (unidentifiable within the budget, not merely hard).
                XP2 = {k: list(v) for k, v in __import__("skins_v10").extra_pools(skin_name).items()}
                rn2 = random.Random(v10["names_seed"] + 7)
                mk = _take(rn2, BAD_MARKERS, 1, used)          # cause 2 hurts the outcome: a "higher = worse" marker
                if mk:
                    add_var(mk[0], kind="observable", parents=[cc_root2],
                            mech={"form": "linear", "weights": {cc_root2: 0.8}, "intercept": 5},
                            measurable=True, assay_noise={"normal": [0, 4]})

    # ---- GOAL (latent true objective) driven by last mediator (chain_sign) + small
    # confounder path. This is the TRUE causal chain terminus; the oracle scores utility
    # HERE. It is LATENT (no assay) — the trap cannot touch it and the agent cannot read it
    # directly; the agent must infer goal-recovery from the mechanism proxy.
    # v10: the harmful chain feeds the goal through a DAMAGE node, which symptomatic treatments reduce by a
    # fraction (engine "damped"). Damage is measured above the chain's healthy level (root at its floor value
    # 3), so relief is ~0 once the root cause is fixed and the gold never needs a symptomatic lever.
    XP = None
    if v10 is not None:
        from skins_v10 import extra_pools
        XP = {"extra_process_pool": LEVEL_POOL, "extra_additive_pool": DOSE_POOL}
        rn = random.Random(v10["names_seed"])
    damage_node, sym_actuators, _goal_shift = None, [], 0.0
    if v10 is not None and chain_sign < 0:
        healthy = 70.0 * 3.0 ** 2 / (35.0 ** 2 + 3.0 ** 2 + 1e-9)        # hill of the root floor (intercept 3)
        for _ in range(1, len(mediators)):
            healthy = 0.9 * healthy + 5.0
        ps = list(v10["sym_p"][: v10["n_sym"]])
        while 1.0 - float(np.prod([1.0 - p for p in ps])) > 0.70:          # all relief together < 70% of damage
            ps = [round(p * 0.9, 3) for p in ps]
        factors = []
        for e, p in zip(_take(rn, XP["extra_additive_pool"], len(ps), used), ps):
            dose = f"{e['name']}Dose"
            add_var({"name": dose, "aliases": [f"{e['aliases'][0]} dose"]}, kind="latent", dist={"normal": [0, 0]})
            add_act(e, target=dose, op="set", dtype="continuous", range=[0, 100], default=0,
                    description="an adjustable treatment")
            sym_actuators.append(e["name"])
            factors.append({"dose": dose, "p": p})
        damage_node = f"{root['name']}Damage"
        add_var({"name": damage_node, "aliases": ["damage pathway"]}, kind="latent",
                parents=[last_mediator] + [f["dose"] for f in factors],
                mech={"form": "damped", "of": last_mediator, "offset": round(healthy, 4), "factors": factors})
        _goal_shift = 0.9 * chain_sign * healthy                           # keeps every goal value unchanged
    weights = {(damage_node or last_mediator): 0.9 * chain_sign}
    for c in confs:
        weights[c["name"]] = 0.1 * (1 if higher_better else -1)
    weights.update(subtype_extra)
    weights.update(competing_extra)
    # confounded_reversal: the reversal confounder drives the GOAL POSITIVELY and strongly
    # (weight calibrated after the SCM is built) so the source knob's observational
    # correlation with the observed surrogate is OPPOSITE its causal sign. rev_conf is a
    # common cause of src (observed) and the goal -> confounding by indication.
    reversal_extra = ({rev_conf["name"]: (-1.0 if _up_dir else 1.0)}
                      if (arche == "confounded_reversal" and rev_conf is not None) else {})
    weights.update(reversal_extra)
    intercept = 15 if chain_sign < 0 else 8
    if _goal_shift:                       # v10 only (keeps the v9 int intercept byte-identical)
        intercept = intercept + _goal_shift
    add_var(goal, kind="latent",
            parents=[(damage_node or last_mediator)] + [c["name"] for c in confs]
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
        fix_expr, fix_default = "1-sat(d;k=0.66)", 0
        if v10 is not None and arche == "competing_causes":
            # v10 (2026-10-13): cause 1's fix has its own therapeutic window (gold = a random interior dose)
            _w = v10["cc_window"][0]
            fix_expr = f"1-sat(d;k={_w})"
            side = {"target": goal["name"], "expr": f"-overstrip(d;thr={_w},gain=60)"}
        elif "interior_dose" in feats:
            # over-treatment harms the TRUE GOAL (a real therapeutic-window penalty),
            # not merely the surrogate reading -> target the latent goal.
            if vv is None:
                side = {"target": goal["name"],
                        "expr": ("-overstrip(d;thr=0.66,gain=30)" if higher_better
                                 else "overstrip(d;thr=0.66,gain=30)")}
            else:
                # v9.1: the window sits at a per-world dose w (cause fully removed at w,
                # over-treatment harm starts at w) instead of always 0.66.
                _w = vv["window"]
                fix_expr = f"1-sat(d;k={_w})"
                side = {"target": goal["name"],
                        "expr": (f"-overstrip(d;thr={_w},gain=60)" if higher_better
                                 else f"overstrip(d;thr={_w},gain=60)")}
        elif vv is not None and vv["fix_down"] and v10 is None:
            # v9.1 "turn it down" plain fix (not in v10: a remedy-named control "currently given, stop it") (engine "lin"): the untouched process sits at dose
            # 50 (factor 1); lowering removes the cause (gold 0), raising past 50 worsens it --
            # so the battery's sign-at-max is "-", not a misleading "0" (a mirrored plateau fix
            # with untouched = 100 gave a real lever sign "0" and broke the gold answer's part B).
            fix_expr, fix_default = "lin(d;k=0.5)", 50
        eff = {"target": root["name"], "op": "scale", "dtype": "continuous",
               "range": [0, 100], "default": fix_default, "expr": fix_expr,
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
    # inert control actuators target a subset of inert vars (real handles, ~0 effect). v10: none -- these are
    # measurement-only quantities (heart rate, pain score, ...), which cannot be set directly.
    for iv in (inert_vars[:n_inert_knobs] if v10 is None else []):
        aid = f"set_{iv['name']}"
        A[aid] = {"aliases": [f"set {a}" for a in iv["aliases"][:2]] + [f"adjust {iv['name']}"],
                  "target": iv["name"], "op": "set", "dtype": "continuous",
                  "range": [0, 100], "default": 50, "description": f"control for {iv['name']}"}

    v10_roles: Dict[str, str] = {}
    harm_nodes: List[Tuple[str, float, float]] = []
    if v10 is not None:
        # One label format for every control: v9 labels inert knobs "set <x>" and the real levers plainly, which
        # reveals the answer in 100% of v9 worlds. v10 gives inert knobs their plain aliases too.
        for a in sym_actuators:
            v10_roles[a] = "symptomatic"

        def _set_control(entry, target, default):
            aid = f"set_{entry['name']}"
            A[aid] = {"aliases": list(entry["aliases"][:2]), "target": target, "op": "set",
                      "dtype": "continuous", "range": [0, 100], "default": default,
                      "description": f"control for {entry['name']}"}
            return aid

        def _level_dist(max_mean=80.0):
            """Observational distribution of a level variable. Drawn per variable: with every inert level at
            N(50,12) (v9) a lever whose level sits elsewhere stands out from the baseline measurement alone."""
            return {"normal": [round(rn.uniform(20.0, max_mean), 1), round(rn.uniform(6.0, 16.0), 1)]}

        def _new_level(pool_key, max_mean=80.0):
            """A new measurable process variable with a set-control (unit-bearing settings only)."""
            got = _take(rn, XP[pool_key], 1, used)
            if not got:
                return None, None
            e = got[0]
            dist = _level_dist(max_mean)
            add_var(e, kind="observable", dist=dist, measurable=True, assay_noise={"normal": [0, 3]})
            return e, _set_control(e, e["name"], int(round(dist["normal"][0])))

        def _new_dose(pool_key):
            """A new additive / treatment: a latent dose node (0 at baseline) with a set-control."""
            got = _take(rn, XP[pool_key], 1, used)
            if not got:
                return None, None, None
            e = got[0]
            dose = f"{e['name']}Dose"
            add_var({"name": dose, "aliases": [f"{e['aliases'][0]} dose"]}, kind="latent", dist={"normal": [0, 0]})
            add_act(e, target=dose, op="set", dtype="continuous", range=[0, 100], default=0,
                    description="an adjustable treatment")
            return e, e["name"], dose

        harm_nodes: List[Tuple[str, float, float]] = []          # (node, knee, fraction of benefit)

        def _harm_on_goal(of, knee, frac):
            """Hinge harm on the goal: 0 below `knee`, linear above it; its size at the control's max (100) is set
            to `frac` x the world's benefit after the SCM is built (placeholder slope 0.1 until then)."""
            h = f"{of}Strain"
            add_var({"name": h, "aliases": ["strain"]}, kind="latent", parents=[of],
                    mech={"form": "sign_flip", "of": of, "knee": knee, "lo_gain": 0.0,
                          "hi_gain": 0.1, "intercept": 0.0})
            V[goal["name"]]["parents"].append(h)
            V[goal["name"]]["mech"]["weights"][h] = -1.0
            harm_nodes.append((h, knee, frac))

        for iv in inert_vars:                      # all distractor levels (knob or not) get a drawn distribution
            V[iv["name"]]["dist"] = _level_dist()
        # (rebuild 2026-10-12) no handles on inert measurements or on confounded decoys: they are observations
        # (heart rate, body temperature, ...), not settings
        # small-harm levers, alternating the two control types
        for i in range(v10["n_harm"]):
            if (i % 2 == 0) == v10["harm_set_first"]:
                # level kept >= 20 below the harm knee, so lowering it is never a hidden benefit
                e, aid = _new_level("extra_process_pool", max_mean=v10["harm_knee_set"][i] - 20.0)
                if e is None:
                    continue
                _harm_on_goal(e["name"], v10["harm_knee_set"][i], v10["harm_frac"][i])
            else:
                e, aid, dose = _new_dose("extra_additive_pool")
                if e is None:
                    continue
                _harm_on_goal(dose, v10["harm_knee_dose"][i], v10["harm_frac"][i])
            v10_roles[aid] = "harm"

        # inert fill up to the drawn control count, keeping the two control types balanced
        def _is_level(aid):
            return V.get(A[aid]["target"], {}).get("measurable", False)
        while len(A) < v10["n_controls"]:
            n_level = sum(_is_level(a) for a in A)
            if n_level <= len(A) - n_level:
                e, aid = _new_level("extra_process_pool")
                if e is None:
                    e, aid, _ = _new_dose("extra_additive_pool")
            else:
                e, aid, _ = _new_dose("extra_additive_pool")
                if e is None:
                    e, aid = _new_level("extra_process_pool")
            if e is None:
                break
            v10_roles[aid] = "inert"

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
                   higher_is_better=higher_better, selection=selection,
                   propagate_side_effects=(v10 is not None))

    # ---- competing_causes: calibrate cause-2's outcome weight so the two causes
    # are JOINTLY NECESSARY -- the dominant single cause recovers ~60% of the
    # benefit, so each single fix is below the Part-A bar while both together
    # recover ~all. Per-world search -> robust across skins without hardcoded %.
    if arche == "competing_causes" and cc_root2 is not None and fix_id and cc_fix2:
        ow = scm.variables[goal["name"]]["mech"]["weights"]   # cause-2 feeds the GOAL
        hi1 = scm.actuators[fix_id]["range"][1]
        hi2 = scm.actuators[cc_fix2]["range"][1]
        if v10 is not None:
            # v10 windowed fixes are applied at their window (the best dose)
            hi1, hi2 = 100.0 * v10["cc_window"][0], 100.0 * v10["cc_window"][1]
        elif vv is not None:
            # v9.1 "turn it down" fixes are applied at 0, not at the max
            hi1 = 0 if vv["fix_down"] else hi1
            hi2 = 0 if vv["fix2_down"] else hi2
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
        # v9.1 src_up: the knob's causal effect is POSITIVE, so the reversal needs a NEGATIVE
        # observational correlation -> mirror the search (negative weight, corr <= -0.25).
        _rs = -1.0 if _up_dir else 1.0
        for k in range(1, 26):                       # reversal weight 0.2 .. 5.0
            ow[rc] = round(_rs * 0.2 * k, 2)
            c = _rs * _obs_corr()
            if c >= 0.25:                            # observational corr clearly positive (mirrored if src_up)
                err = abs(c - 0.4)
                if err < best_err:
                    best_err, best_w = err, ow[rc]
        ow[rc] = best_w if best_w is not None else _rs * 3.0

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

    # v10: size each harm lever relative to the world's benefit. Benefit estimate = the best of: each primary
    # lever alone on an 11-point grid, and all primary levers together at their single-best values.
    bnf = story_bnf = None
    if v10 is not None:
        prim = [a for a in A if v10_roles.get(a) is None and a not in (trap_id,)
                and a in {fix_id, src["name"], src2["name"] if src2 is not None else None, cc_fix2,
                          sub_info["treatment_actuator"] if sub_info is not None else None}]
        def _eu10(iv):
            return float(np.mean(scm.utility(scm.sample(6000, intervention=iv, seed=seed + 1717))))
        b0 = _eu10({})
        best, best_v = 0.0, {}
        for a in prim:
            us = [(_eu10({a: x}), x) for x in range(0, 101, 10)]
            u, x = max(us)
            best_v[a] = x
            best = max(best, u - b0)
        if len(best_v) > 1:
            best = max(best, _eu10(best_v) - b0)
        bnf = max(best, 5.0)
        story_bnf = bnf
        if sub_info is not None and best_v:
            # the story states the full recoverable loss, which in a subtype world includes the conditional policy
            # (treat only the subgroup the treatment helps); without it the stated loss was ~40% too small
            iv = dict(best_v)
            helps_high = not (vv is not None and vv["subtype_flip"])
            iv[sub_info["treatment_actuator"]] = {"policy": {
                "stratifier": sub_info["marker"], "threshold": sub_info["center"],
                "dose_if_ge": 100.0 if helps_high else 0.0, "dose_if_lt": 0.0 if helps_high else 100.0}}
            story_bnf = max(bnf, _eu10(iv) - b0)
        for h, knee, frac in harm_nodes:          # (empty list -> nothing to size)
            scm.variables[h]["mech"]["hi_gain"] = round(frac * bnf / max(1e-6, 100.0 - knee), 4)

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
    # (v9.1 src_up: the knob's harmful direction is LOWERING it, so the naive moves mirror to 0)
    _src_naive = 0 if _up_dir else 100
    if "sign_flip" in feats and src["name"] in A:
        naive.append({src["name"]: _src_naive})   # "more of the input" (wrong: harms)
    if sub_info is not None:
        naive.append({sub_info["treatment_actuator"]: 100})   # treat everyone -> ~0
    if v10 is not None:
        for a_ in [a for a, r in v10_roles.items() if r == "inert"][:2]:
            naive.append({a_: 100})
    else:
        for iv in inert_vars[:2]:
            naive.append({f"set_{iv['name']}": 100})
    if trap_id is not None:
        # dosing the trap raises the observed surrogate but has ZERO effect on the goal ->
        # the canonical "optimized the metric, not the cause" mistake (fails Part A).
        naive.append({trap_id: 100})
    if arche == "confounded_reversal":
        # observationally the source knob looks beneficial, so "turn it up" is the
        # obvious move -- but it is causally HARMFUL (the sign reverses under do()).
        naive.append({src["name"]: _src_naive})
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
    _atag = {"confounded_chain": "chain", "collider_selection": "collider",
             "hidden_subtype": "subtype", "surrogate_trap": "surrogate",
             "instrument_only": "instrument", "competing_causes": "competing",
             "synergy_pair": "synergy", "dose_window": "dosewin",
             "confounded_reversal": "reversal"}.get(arche, arche)
    if vv is not None:
        gt["_v91_values"] = dict(vv, src_up_applied=src_up)
    if v10 is not None:
        for aid in (fix_id, src2["name"] if src2 is not None else None, cc_fix2,
                    sub_info["treatment_actuator"] if sub_info is not None else None):
            if aid:
                v10_roles[aid] = "primary"
        if src["name"] in A:
            v10_roles[src["name"]] = "primary"
        if trap_id:
            v10_roles[trap_id] = "trap"
        missing = sorted(set(A) - set(v10_roles))
        assert not missing, f"v10: actuators without a role: {missing}"
        gt["_world_version"] = "v10"
        gt["_v10_roles"] = v10_roles
        # units: every control (and the measured level of a level control) gets a physical unit and range; the SCM
        # stays on 0-100 internally and the agent-facing layers convert (sim_v6 / catalog)
        for aid, act in scm.actuators.items():
            nm = aid[4:] if aid.startswith("set_") else aid
            if nm in _LU[skin_name]:
                u, lo, hi = _LU[skin_name][nm]
            else:
                assert nm in _DU[skin_name], f"v10: control {aid} has no unit"
                (u, hi), lo = _DU[skin_name][nm], 0
            act["units"] = {"unit": u, "lo": lo, "hi": hi}
            tgt = scm.variables.get(act["target"], {})
            # only a LEVEL control ("set" on a measured variable) shares its unit with its target; the surrogate trap
            # ("add" on the outcome) must not hand its dose unit to the outcome (bug found 2026-10-12)
            if act.get("op") == "set" and tgt.get("kind") == "observable":
                tgt["units"] = dict(act["units"])
        from skins_v10 import MEASURE_UNITS as _MU
        for n_, v_ in scm.variables.items():          # every measured signal gets a physical unit too
            if "units" in v_ or not (v_.get("measurable") or v_.get("kind") in ("observable", "outcome")):
                continue
            key = "__outcome__" if n_ == scm.outcome else n_
            src_u = _LU[skin_name].get(n_) or _MU[skin_name].get(key)
            assert src_u is not None, f"v10: measured signal {n_} has no unit"
            u, lo, hi = src_u
            v_["units"] = {"unit": u, "lo": lo, "hi": hi}
        gt["_v10_values"] = dict(v10, setpoint_applied=_setpoint)
    scenario = (_build_story_v10(S, skin_name, scm, v10["names_seed"], benefit=story_bnf) if v10 is not None
                else _build_scenario(S, feats))
    _tag = "v10" if v10 is not None else ("v91" if vv is not None else "v7")
    wid = f"{_tag}_{skin_name}_{_atag}_{'_'.join(sorted(feats))[:20]}_{seed}"
    return {"world_id": wid, "domain": skin_name, "scenario": scenario,
            "scm": scm, "ground_truth": gt}


def _build_scenario(S: Dict[str, Any], feats: set) -> str:
    return S["scenario"].format(**S["fills"])


def fmt_num(x: float) -> str:
    """A physical value at a sensible precision (3 significant figures, no exponent)."""
    x = float(x)
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.2f}".rstrip("0").rstrip(".") if x != 0 else "0"


def to_display(units: Dict[str, Any], internal: float) -> float:
    return units["lo"] + float(internal) / 100.0 * (units["hi"] - units["lo"])


def _build_story_v10(S: Dict[str, Any], skin: str, scm, seed: int, benefit: Optional[float] = None) -> str:
    """v10 situation report, generated per world from the world itself (rebuild 2026-10-12): the facility, the size and
    duration of the decline, and the trigger are drawn per world, and the report lists EVERY measurable signal and
    EVERY control with its unit, range and current state, in shuffled order, with uniform wording -- so an agent with
    no id catalog (the free-text harness) can name what exists, and nothing in the text depends on a control's role.
    The optional "suspicions" sentence names two controls drawn uniformly, so it is uninformative on average."""
    from skins_v10 import STORY
    r = random.Random(f"rpg_v10_story:{seed}")
    St = STORY[skin]
    facility, trigger, unitw = r.choice(St["facility"]), r.choice(St["trigger"]), r.choice(St["unit"])
    outcome = (S["outcome"].get("aliases") or [S["outcome"]["name"]])[0]
    # The size of the decline is the world's REAL recoverable loss (the sampler's estimate of the best achievable
    # improvement, in the outcome's display units), not a random %: a random "down 16%" told an agent that a
    # symptomatic +8-point lift had recovered the whole loss when the true gap was ~26 points (Opus trace, 2026-10-12).
    ou = scm.variables[scm.outcome].get("units")
    if benefit is not None and ou:
        drop = f"{fmt_num(float('%.2g' % (benefit / 100.0 * (ou['hi'] - ou['lo']))))} {ou['unit']}"
    else:
        drop = "a substantial amount"
    period = r.choice(["two weeks", "three weeks", "a month", "six weeks", "two months", "the past quarter"])
    opener = r.choice([
        f"At {facility}, {outcome} has fallen by about {drop} over {period}, and the drop has persisted across {unitw}.",
        f"{outcome[0].upper() + outcome[1:]} at {facility} is down roughly {drop} over {period}, and the loss has held across {unitw}.",
        f"Over {period}, {facility} has lost about {drop} of its usual {outcome}; the shortfall has been consistent across {unitw}.",
        f"Staff at {facility} report that {outcome} has slipped by around {drop} in {period}, a decline repeated across {unitw}.",
    ])
    when = r.choice([f"The decline began shortly after {trigger}.", f"It started around the time of {trigger}.",
                     f"The first affected {unitw} followed {trigger}."])
    import re as _re
    _ACR = {"ph": "pH", "co2": "CO2", "uv": "UV", "ldh": "LDH", "nic": "NIC", "pdu": "PDU", "ups": "UPS", "gpu": "GPU",
            "cpu": "CPU", "io": "I/O", "rf": "RF", "esc": "ESC", "hf": "HF", "cmp": "CMP", "di": "DI", "fec": "FEC",
            "tcp": "TCP", "mtu": "MTU", "numa": "NUMA", "bmc": "BMC", "ecc": "ECC", "dmds": "DMDS", "cip": "CIP"}

    def lab(spec, nm):
        t = (spec.get("aliases") or [nm])[0]
        return _re.sub(r"[A-Za-z0-9]+", lambda m: _ACR.get(m.group(0).lower(), m.group(0)), t)

    rng_txt = lambda u: f"from {fmt_num(u['lo'])} to {fmt_num(u['hi'])} {u['unit']}"
    acts = list(scm.actuators.items())
    r.shuffle(acts)
    opinions = r.choice(["none", "two", "open"])
    if opinions == "two" and len(acts) >= 2:
        a1, a2 = r.sample([lab(a, n) for n, a in acts], 2)
        view = r.choice([f"Opinions differ: some of the team suspect the {a1}, others the {a2}, and there is no consensus.",
                         f"Two theories are circulating, one about the {a1} and one about the {a2}; neither has been tested."])
    elif opinions == "open":
        view = r.choice(["Nobody has a convincing explanation yet.", "The team has no consensus on the cause.",
                         "Routine checks have not singled out an obvious culprit."])
    else:
        view = ""
    meas = [(n, v) for n, v in scm.variables.items()
            if (v["kind"] in ("observable",) or v.get("measurable")) and n != scm.outcome]
    r.shuffle(meas)
    def mdesc(n, v):
        u = v.get("units")
        return f"{lab(v, n)} ({u['unit']})" if u and u["unit"].lower() != lab(v, n).lower() else lab(v, n)
    levels, doses = [], []
    for n, a in acts:
        u = a["units"]
        tgt = scm.variables.get(a["target"], {})
        if tgt.get("kind") == "observable" and "dist" in tgt and "normal" in tgt["dist"]:
            usual = fmt_num(to_display(u, tgt["dist"]["normal"][0]))
            levels.append(f"{lab(a, n)} (usually about {usual} {u['unit']}; adjustable {rng_txt(u)})")
        elif tgt.get("kind") == "observable":
            levels.append(f"{lab(a, n)} (adjustable {rng_txt(u)})")
        else:
            cur = float(a.get("default", 0) or 0)
            if cur > 0:
                doses.append(f"{lab(a, n)} (currently {fmt_num(to_display(u, cur))} {u['unit']}; adjustable {rng_txt(u)})")
            else:
                doses.append(f"{lab(a, n)} (not currently used; can be applied at up to {fmt_num(u['hi'])} {u['unit']})")
    h_meas = r.choice(["Measurements available", "Signals the team can measure", "Instrumented readings available"])
    h_lev = r.choice(["Settings that can be adjusted", "Adjustable operating settings", "Settings under the team's control"])
    h_dose = r.choice(["Additions and treatments that can be applied", "Available additives and treatments",
                       "Things that can be added or applied"])
    close = r.choice([
        "You have a limited number of experiments; you may measure quantities and apply interventions, alone or in "
        "combination. Determine what is really driving the decline and what to do about it.",
        "Experiments are limited. Measure what you need and try interventions, singly or together, to find what is "
        "really causing the loss and how to fix it.",
        "Only a limited number of experiments can be run. Use measurements and interventions (alone or combined) to work "
        "out the true cause of the decline and the right corrective action.",
    ])
    parts = [" ".join(x for x in (opener, when, view) if x)]
    parts.append(f"{h_meas}: " + "; ".join(mdesc(n, v) for n, v in meas) + ".")
    if levels:
        parts.append(f"{h_lev}: " + "; ".join(levels) + ".")
    if doses:
        parts.append(f"{h_dose}: " + "; ".join(doses) + ".")
    parts.append(close)
    return "\n\n".join(parts)
