#!/usr/bin/env python3
"""RL reward for RPG — a PURE, DETERMINISTIC function of the policy's id-answer.

Design contract (reward-contract decisions):

- The reward is computed by the SAME oracle_v6.grade() the eval uses, so training and
  evaluation optimize/measure the same thing. (No separate, drift-prone reward logic.)
- The policy answers in OPAQUE IDS (m*/a*, via catalog.py), which we map to canonical
  names here. => NO free-text resolver and NO LLM anywhere in the reward path. The
  reward is a deterministic function of (chosen ids, chosen doses) and the world's
  precomputed gold+battery. This is the RLVR integrity requirement: the same answer
  always earns the same reward, and phrasing cannot change it.
- Dense, continuous shaping so GRPO groups have variance (V3):
      r = w_A * benefit_recovered(partA, clipped to [0,1])
        + w_B * battery_fraction(partB)
        - c_invalid * (fraction of answer ids that were invalid)
      (optionally minus a small over-budget / no-evidence term; off by default)
- Part-B uses strict=True (V5): the exact sampled proxy / valid-equivalent, not a
  lenient downstream set (measured no-op and it weakens the signal).
- LEVER-IDENTIFICATION GATE (opt-in, 2026-09-21): with RPG_LEVER_GATE=1 the whole reward is
  zeroed unless the answer names the correct variable(s) to intervene on (lever_sets());
  with RPG_LEVER_ONLY=1 the reward IS that binary check. See RewardConfig.

The answer the policy must emit (all ids from the world's catalog):
    {
      "actions":  [{"actuator": "a3", "value": 66}, ...],         # recommended fix
      "policy":   {"treatment":"a3","stratifier":"m2","threshold":50,   # optional (subtype)
                   "dose_if_ge":100,"dose_if_lt":0},
      "proxy":    "m5",                                            # true_mechanism_proxy
      "decoys":   ["m1","m7"],                                    # confounded_decoys
      "signs":    {"a3": "+", "a0": "0"}                          # actuator sign predictions
    }
Unknown ids are dropped (and counted for the invalid-id penalty), NEVER resolved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from oracle_v6 import grade
from catalog import Catalog


@dataclass
class RewardConfig:
    # env-var gated (default 0.5/0.5 = r1, unchanged) so a GRPO run can select r2=part_a-only
    # via RPG_W_A=1.0 RPG_W_B=0.0 without touching any other caller. (box2 2026-09-03 experiment)
    w_a: float = field(default_factory=lambda: float(os.environ.get("RPG_W_A", "0.5")))  # weight on part A (found the fix)
    w_b: float = field(default_factory=lambda: float(os.environ.get("RPG_W_B", "0.5")))  # weight on part B (mechanism)
    c_invalid: float = 0.25      # penalty per unit fraction of invalid ids in the answer
    strict_part_b: bool = True   # V5: strict proxy credit
    # FAITHFULNESS (v9): "observation/code alone CANNOT establish causation" (system prompt).
    # An answer produced with ZERO applied interventions did not do interventional science, so it
    # cannot have EARNED either the fix (part A) or the mechanism (part B) — it can only have guessed
    # from priors/labels. Gate ALL positive credit on >=1 applied intervention. This closes the
    # "answer from priors, no experiment" hack that c_no_evidence=0 left open.
    require_evidence: bool = True   # zero part_a/part_b if n_interventions == 0
    # optional shaping (off by default; enable via trainer if needed):
    c_no_evidence: float = 0.0   # ADDITIONAL flat penalty if the episode ran 0 interventions
    # LEVER-IDENTIFICATION GATE (2026-09-21). The deficit the results deck diagnoses is a
    # *stochastic identification error*: the policy is seduced by the wrong (confounded) knob.
    # But the r1 reward does not isolate identification — measured on the v8 Opus/27B dumps,
    # a wrong-lever answer still earns ~0.3/0.12 of part B (proxy/decoys/signs are graded
    # independently of the recommended knob), so 82%/28% of wrong-lever episodes score >0.1.
    # These knobs make identification a PREREQUISITE for any credit (all env-var gated,
    # defaults OFF = r1 unchanged):
    #   RPG_LEVER_GATE=1  -> part_a and part_b are zeroed unless the answer names the correct
    #                        lever(s); the dense A/B shaping applies on top of a correct id.
    #   RPG_LEVER_ONLY=1  -> reward is the BINARY lever check alone (no dose, no battery):
    #                        "can it find the variable to intervene on?" (implies the gate).
    #   RPG_LEVER_MODE    -> "full" (default): every must-have lever named (both co-causes on
    #                        competing/synergy worlds, the treatment on conditional-policy
    #                        subtype worlds) AND at least one causal lever;
    #                        "any": at least one causal lever (any knob that truly moves the
    #                        goal: the fix, its upstream source knob, either co-cause).
    #   RPG_LEVER_BONUS   -> flat credit added on a correct id in gate mode (keeps a reward step
    #                        for "right knob, poor dose" so GRPO groups don't all collapse to 0).
    # See _lever_check() for the exact sets.
    #   RPG_LEVER_EXACT=1 -> identification additionally requires NO spurious lever: the answer
    #                        must name only causal knobs (chosen <= causal). Closes the SHOTGUN
    #                        hack described below.
    #   RPG_LEVER_PRECISION=1 -> scale the gated credit by precision = |chosen & causal|/|chosen|;
    #                        the smooth, gradient-preserving form of the same fix.
    #
    # THE SHOTGUN HACK (measured 2026-09-22; personal_docs/results/lever_gate_preflight/ and
    # personal_docs/rl_logs/reward_lever_gate_2026-09-21.md §9). `chosen` is just the key set of
    # the recommended intervention and NOTHING caps its size, so an answer that names EVERY
    # actuator satisfies `any` mode on every world -- and satisfies `full` mode too, since a
    # superset always contains `must`. The v9 audits make non-causal knobs inert on the latent
    # goal, so the extras are nearly free in part A as well. Mean reward for "name all actuators
    # at range-max" (zero causal knowledge) over 12 hard-archetype worlds, perfect battery:
    #     r1 0.923 | gate_any 0.923 | gate_full 0.923 | lever_only 1.000     (gold = 1.000)
    # and E[reward | k knobs named] rises monotonically in k under every gated mode (gate_full
    # 0.006 -> 0.811 for k = 1 -> 8). So the gate as first written does not isolate
    # identification; it multiplies the incentive to name more knobs, which is far easier to
    # learn than finding the cause. Turn RPG_LEVER_EXACT or RPG_LEVER_PRECISION on for ANY run
    # whose read is the `lever_ok` rate.
    lever_gate: bool = field(default_factory=lambda: os.environ.get("RPG_LEVER_GATE", "0") not in ("", "0", "false", "False"))
    lever_only: bool = field(default_factory=lambda: os.environ.get("RPG_LEVER_ONLY", "0") not in ("", "0", "false", "False"))
    lever_mode: str = field(default_factory=lambda: os.environ.get("RPG_LEVER_MODE", "full"))
    lever_bonus: float = field(default_factory=lambda: float(os.environ.get("RPG_LEVER_BONUS", "0.0")))
    lever_exact: bool = field(default_factory=lambda: os.environ.get("RPG_LEVER_EXACT", "0") not in ("", "0", "false", "False"))
    lever_precision: bool = field(default_factory=lambda: os.environ.get("RPG_LEVER_PRECISION", "0") not in ("", "0", "false", "False"))
    # RPG_LEVER_MAX_EXTRA=<int>: tolerance form of lever_exact -- identification additionally
    # requires |chosen - causal| <= max_extra. -1 (default) = unbounded = pre-2026-09-22 behaviour;
    # 0 == lever_exact. Measured trade-off (E[reward | k knobs named], hard archetypes):
    #   gate_any                 0.21 -> 0.84 over k=1..8   ramp UP    (shotgun is optimal)
    #   + precision              0.21 -> 0.31               ramp flat  (density kept, weak pull)
    #   + exact (max_extra=0)    0.21 -> 0.00               ramp DOWN  (but near-zero density
    #                                                        from base, i.e. all-zero groups)
    #   + precision, max_extra=1 peaks at k = |causal|       correct shape: density at k<=|causal|+1
    #                                                        and nothing to gain past the truth
    # For a from-base run on the hard archetypes use RPG_LEVER_PRECISION=1 RPG_LEVER_MAX_EXTRA=1.
    lever_max_extra: int = field(default_factory=lambda: int(os.environ.get("RPG_LEVER_MAX_EXTRA", "-1")))
    # RPG_LEVER_SCALE = none | precision | jaccard | recall. What the gated credit is multiplied by.
    #   recall    = (|chosen & must| / |must|) * precision  -- RECOMMENDED for the hard archetypes.
    #   precision = |chosen & causal| / |chosen|            (punishes spurious levers only)
    #   jaccard   = |chosen & causal| / |chosen | causal|   (punishes spurious AND missing levers)
    # RPG_LEVER_PRECISION=1 is sugar for scale=precision. Measured on the base 9B over the 120
    # held-out hard-archetype worlds (n=8, 2026-09-22 P0, results/lever_gate_preflight/):
    # the model names 0.82 levers against a causal set of 2.33 -- 74% of answers name exactly ONE
    # knob, and when it names one it is causal ~76% of the time. So `any` is already satisfied
    # 57.7% of the time while `full` sits at 0.9%: on these archetypes the deficit is
    # COMPLETENESS (naming every co-cause), not picking a non-causal knob. A binary any-gate
    # therefore gates on something the policy mostly already does, and gate_full is an all-zero
    # reward. `jaccard` grades the thing that actually has headroom -- mean 0.269, with 59.4% of
    # episodes strictly between 0 and 1, i.e. a dense gradient -- while still scoring a
    # name-everything shotgun at only |causal|/n_actuators (~0.29).
    lever_scale: str = field(default_factory=lambda: os.environ.get("RPG_LEVER_SCALE", "none"))
    # RPG_W_ID=<float> (default 0 = off). When > 0 AND gating, identification becomes an ADDITIVE
    # reward term instead of a multiplier on A/B:
    #     reward = w_id*(recall*precision) + w_a*part_a*s + w_b*part_b*s      (s = the scale above)
    # WHY ADDITIVE (measured 2026-09-22 on the 960 real base-9B rollouts, results/lever_gate_preflight/):
    # a multiplicative recall term multiplies two small quantities and CRUSHES the GRPO signal --
    # mean within-group std over the 120 val worlds falls to 0.076 with 18% of groups degenerate
    # (all 8 identical), against 0.171 / 0% for r1. Additive keeps it: 0.162 / 0%.
    # It also fixes an ordering bug. Under a multiplier, "both co-causes, bad dose" can score BELOW
    # "one co-cause, good dose" -- backwards for a run whose target is completeness. And on
    # hidden_subtype part_a is NOT monotone in recall (measured part_a: 0.128 at recall 0, 0.049 at
    # recall 0.33), so multiplying the two is incoherent there.
    # The deeper reason identification needs its OWN term: on synergy_pair the synergy is
    # super-additive (RPG_SYNERGY_SOFT=20), so part_a is near-binary -- measured part_a 0.000 /
    # 0.118 / 0.477 at recall 0 / 0.5 / 1.0. Naming one of two co-causes buys almost no part_a, so
    # part_a alone gives GRPO no gradient toward the second one. The additive term supplies it.
    # Recommended: RPG_W_ID=0.5 RPG_W_A=0.3 RPG_W_B=0.2 (gold still = 1.0).
    w_id: float = field(default_factory=lambda: float(os.environ.get("RPG_W_ID", "0.0")))


def _num(x, default=None):
    """Coerce a model-supplied value to float, else return default. Never raises.
    Booleans are rejected (a policy dose of True/False is meaningless)."""
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return default
    return default


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else []


def _to_canonical_answer(struct: Dict[str, Any], cat: Catalog):
    """Map an id-space answer to the canonical-name answer grade() expects. Returns
    (answer_dict, invalid_fraction).

    ROBUSTNESS CONTRACT: an RL policy emits arbitrary/malformed answers (dict-valued
    doses, non-string ids, wrong container types). This function MUST be total — it
    never raises on any input; it sanitizes and drops what it can't use (dropped ids
    count toward the invalid-id penalty). A crash here would kill a training rollout."""
    if not isinstance(struct, dict):
        struct = {}
    n_ids = 0
    n_bad = 0

    def m(mid):                       # measurable id -> canonical name (str ids only)
        nonlocal n_ids, n_bad
        n_ids += 1
        nm = cat.measurable_name(mid) if isinstance(mid, str) else None
        if nm is None:
            n_bad += 1
        return nm

    def a(aid):                       # actuator id -> canonical name (str ids only)
        nonlocal n_ids, n_bad
        n_ids += 1
        nm = cat.actuator_name(aid) if isinstance(aid, str) else None
        if nm is None:
            n_bad += 1
        return nm

    # recommended scalar actions -> {actuator_name: value}; drop malformed items/values
    rec: Dict[str, Any] = {}
    for item in _as_list(struct.get("actions")):
        if not isinstance(item, dict):
            continue
        nm = a(item.get("actuator"))
        val = _num(item.get("value"))
        if nm is not None and val is not None:
            rec[nm] = val

    answer: Dict[str, Any] = {"recommended_intervention": rec, "structured": {}}

    # conditional policy (subtype worlds)
    pol = struct.get("policy")
    if isinstance(pol, dict):
        tname = a(pol.get("treatment"))
        sname = m(pol.get("stratifier"))
        if tname is not None and sname is not None:
            answer["recommended_policy"] = {
                "treatment": tname, "stratifier": sname,
                "threshold": _num(pol.get("threshold"), 50.0),
                "dose_if_ge": _num(pol.get("dose_if_ge"), 0.0),
                "dose_if_lt": _num(pol.get("dose_if_lt"), 0.0)}

    st: Dict[str, Any] = {}
    if struct.get("proxy") is not None:
        pn = m(struct.get("proxy"))
        if pn is not None:
            st["true_mechanism_proxy"] = pn
    decoys = []
    for d in _as_list(struct.get("decoys")):
        dn = m(d)
        if dn is not None:
            decoys.append(dn)
    st["confounded_decoys"] = decoys
    signs = {}
    signs_in = struct.get("signs")
    if isinstance(signs_in, dict):
        for aid, s in signs_in.items():
            an = a(aid)
            if an is not None:
                signs[an] = s
    st["actuator_sign_predictions"] = signs
    answer["structured"] = st

    invalid_fraction = (n_bad / n_ids) if n_ids else 0.0
    return answer, invalid_fraction


def lever_sets(world: Dict[str, Any], gold: Dict[str, Any]):
    """The (causal, must) lever sets for a world, from stored gold + ground truth only.

    causal = every actuator that genuinely moves the TRUE goal: the oracle's screened
             ``active_actuators`` (fix + its upstream source knob; both co-causes; synergy-
             rescued pairs) ∪ the gold intervention's keys (adds the conditional-policy
             treatment, which screens ~inactive on population average). The surrogate trap
             (zero true-goal effect) and inert knobs are never in it.
    must   = levers the archetype's skill REQUIRES all of: the co-actuators of a two-cause
             world (competing_causes / synergy_pair / the two_cause feature) and the
             treatment of a conditional-policy subtype world. Empty for single-cause worlds,
             where the fix and its source knob are interchangeable identifications.
    Measured over the local v8 world set: |causal| = 1 (reversal, instrument), 2 (chain,
    collider, surrogate, dose_window, competing, synergy), 3 (subtype); |must| = 0 / 2 / 3."""
    gt = world.get("ground_truth", {}) or {}
    scm_acts = set(getattr(world.get("scm"), "actuators", {}) or {})
    causal = set(gold.get("active_actuators") or []) | set((gold.get("intervention") or {}).keys())
    must = set(gt.get("co_actuators") or [])
    sp = gt.get("subtype_policy")
    if sp and gold.get("is_conditional_policy") and sp.get("treatment_actuator"):
        must.add(sp["treatment_actuator"])
    if scm_acts:                       # never require a lever the world cannot execute
        causal &= scm_acts
        must &= scm_acts
    return causal, must


def _lever_check(answer: Dict[str, Any], world: Dict[str, Any], gold: Dict[str, Any],
                 mode: str = "full", exact: bool = False,
                 max_extra: int = -1) -> Dict[str, Any]:
    """Did the canonical answer name the correct variable(s) to intervene on? Total —
    never raises. Levers are read from the answer itself (scalar actions ∪ the policy's
    treatment), independent of which grading variant later wins, so a spurious/stripped
    policy cannot change the identification verdict.

    ``exact=True`` (== ``max_extra=0``) additionally requires that NO non-causal knob was named
    (chosen <= causal); ``max_extra=n`` allows n spurious knobs before the verdict flips.
    Without it, naming every actuator passes both `any` and `full` -- see the SHOTGUN HACK note
    on RewardConfig. ``lever_precision`` / ``lever_jaccard`` are always returned as diagnostics:
    a run whose `lever_ok` climbs while precision falls is learning to shotgun, not to identify."""
    causal, must = lever_sets(world, gold)
    scm_acts = set(getattr(world.get("scm"), "actuators", {}) or {})
    chosen = set((answer.get("recommended_intervention") or {}).keys())
    pol = answer.get("recommended_policy")
    if isinstance(pol, dict) and pol.get("treatment"):
        chosen.add(pol["treatment"])
    if scm_acts:
        chosen &= scm_acts
    hit = chosen & causal
    extra = chosen - causal
    if mode == "any":
        ok = bool(hit)
    else:                              # "full" (default)
        ok = bool(hit) and must <= chosen
    # anti-shotgun: bound the number of spurious levers (exact == max_extra 0)
    cap = 0 if exact else max_extra
    if cap >= 0:
        ok = ok and len(extra) <= cap
    union = chosen | causal
    return {"lever_ok": ok, "lever_mode": mode, "lever_exact": bool(exact),
            "lever_max_extra": int(cap),
            "lever_precision": (len(hit) / len(chosen)) if chosen else 0.0,
            "lever_jaccard": (len(hit) / len(union)) if union else 0.0,
            # Recall is graded against `must` -- the levers the archetype's skill REQUIRES all of --
            # NOT against `causal`. `causal` is a superset that even the gold answer does not name
            # in full (subtype: |causal|=3, gold names 2), so a causal-recall or Jaccard reward
            # scores the ORACLE below 1.0 (measured: 0.944). `must` is satisfied by gold by
            # construction. Single-cause worlds have must == {} -> fall back to the binary hit.
            "lever_recall": (len(chosen & must) / len(must)) if must else (1.0 if hit else 0.0),
            "chosen_levers": sorted(chosen), "causal_levers": sorted(causal),
            "must_levers": sorted(must), "extra_levers": sorted(extra)}


def compute_reward(struct: Dict[str, Any], world: Dict[str, Any], cat: Catalog,
                   gold: Dict[str, Any], battery: Dict[str, Any],
                   cfg: RewardConfig = RewardConfig(),
                   n_interventions: Optional[int] = None) -> Dict[str, Any]:
    """Pure reward for one episode's final id-answer. Returns a dict with the scalar
    ``reward`` plus its components and the full grade (for logging/debugging)."""
    answer, invalid_frac = _to_canonical_answer(struct, cat)
    # Lever identification is decided from the answer alone (total, no simulation), BEFORE
    # grading, so it is available on the error path too and is independent of which grading
    # variant wins below.
    gating = bool(cfg.lever_gate or cfg.lever_only)
    lever = _lever_check(answer, world, gold, mode=cfg.lever_mode,
                         exact=gating and cfg.lever_exact,
                         max_extra=cfg.lever_max_extra if gating else -1)
    lever_ok = bool(lever["lever_ok"])
    evidence_gated = bool(cfg.require_evidence and n_interventions == 0)

    def _pa(g):
        """The recovered-utility scalar compute_reward derives part_a from (kept in sync
        with the block below), used to pick the better of two gradings."""
        b = g.get("benefit_recovered")
        return max(0.0, min(1.0, b)) if b is not None else (1.0 if g.get("part_a_utility_ok") else 0.0)

    def _grade_answer(ans):
        """Grade an answer, but don't let an OPTIONAL, spurious `recommended_policy` cost an
        otherwise-valid answer its credit. The policy field is advertised in the schema, so the
        model emits one on almost every world (measured: 83/83 at step-0) — including non-subtype
        worlds where it is spurious. A spurious policy hurts in TWO ways:
          (a) scale/add treatment actuators -> a dict-valued dose raises (TypeError float()... 'dict');
          (b) `set` treatment actuators -> the policy is EXECUTED (engine.py:344) and
              `_normalize_policy_answer` OVERRIDES the agent's good scalar dose with the stratified
              policy, silently DEFLATING part-A (no exception, so an except-only retry never fires).
        So: whenever a policy is present, grade BOTH the as-given and the policy-stripped answer and
        credit whichever recovers more utility. This fixes (a) and (b) symmetrically. A LEGITIMATE
        conditional-subtype policy still wins, because stripping it LOWERS benefit on those worlds.
        Only if EVERY variant fails to grade do we propagate the exception (-> reward 0, reward_error)."""
        variants = [ans]
        if "recommended_policy" in ans:
            variants.append({k: v for k, v in ans.items() if k != "recommended_policy"})
        grades, last_err = [], None
        for v in variants:
            try:
                grades.append(grade(world, v, gold, battery, strict=cfg.strict_part_b))
            except Exception as e:  # noqa: BLE001
                last_err = e
        if not grades:
            raise last_err
        return max(grades, key=_pa)

    try:
        g = _grade_answer(answer)
        benefit = g.get("benefit_recovered")
        part_a = max(0.0, min(1.0, benefit)) if benefit is not None else (1.0 if g["part_a_utility_ok"] else 0.0)
        part_b = float(g["battery_fraction"])
    except Exception as e:  # noqa: BLE001
        # ROBUSTNESS CONTRACT (see module docstring): the reward MUST be total — never
        # crash the trainer on a pathological answer. If even the policy-stripped answer
        # can't be graded, it did not solve the world -> reward 0 (minus any invalid-id
        # penalty). Flagged + surfaced so we can count how often it fires.
        # In lever-only mode the identification verdict needs no simulation, so it is still
        # paid out (subject to the evidence gate) even when the oracle could not grade.
        reward = -cfg.c_invalid * invalid_frac
        if cfg.lever_only and lever_ok and not evidence_gated:
            _sc = (cfg.lever_scale or "none").lower()
            if cfg.lever_precision and _sc == "none":
                _sc = "precision"
            reward += (float(lever["lever_recall"]) * float(lever["lever_precision"])
                       if _sc == "recall" else
                       float(lever["lever_jaccard"]) if _sc == "jaccard" else
                       float(lever["lever_precision"]) if _sc == "precision" else 1.0)
        return {
            "reward": float(reward),
            "part_a": 0.0, "part_b": 0.0,
            "invalid_id_fraction": invalid_frac, "accepted": False,
            "grade": {"error": f"{type(e).__name__}: {e}"}, "reward_error": True,
            "evidence_gated": evidence_gated, "lever_gated": gating and not lever_ok, **lever,
        }

    # FAITHFULNESS GATE (v9): no interventional evidence -> no discovery credit. Zero both parts
    # so an answer read off priors/labels without experimenting cannot score. (n_interventions is
    # the count of APPLIED interventions, passed by the env; None = caller didn't track -> no gate.)
    if evidence_gated:
        part_a, part_b = 0.0, 0.0

    # LEVER GATE: no credit of any kind unless the correct variable(s) were named. The dense
    # A/B shaping (and the optional flat bonus) apply only ON TOP of a correct identification.
    lever_gated = gating and not lever_ok
    if lever_gated:
        part_a, part_b = 0.0, 0.0

    # PRECISION SCALING (opt-in, anti-shotgun): the smooth alternative to lever_exact. Gated
    # credit is multiplied by |chosen & causal| / |chosen|, so every spurious knob dilutes the
    # payout instead of riding along free. Only active while gating, so r1 stays bit-identical.
    scale = (cfg.lever_scale or "none").lower()
    if cfg.lever_precision and scale == "none":      # back-compat sugar
        scale = "precision"
    if not gating or scale == "none":
        prec = 1.0
    elif scale == "recall":
        # completeness x precision: gold -> 1.0, one co-cause of two -> 0.5, shotgun -> |causal|/|chosen|
        prec = float(lever["lever_recall"]) * float(lever["lever_precision"])
    elif scale == "jaccard":
        prec = float(lever["lever_jaccard"])
    else:
        prec = float(lever["lever_precision"])
    if prec != 1.0:
        part_a *= prec
        part_b *= prec

    if cfg.lever_only:
        # binary "found the lever" reward; dose quality and mechanism battery are ignored
        reward = prec if (lever_ok and not evidence_gated) else 0.0
    elif gating and cfg.w_id > 0.0:
        # ADDITIVE identification term (see RewardConfig.w_id). part_a/part_b were already
        # multiplied by `prec` above; the identification term carries its own recall*precision.
        if lever_ok and not evidence_gated:
            id_term = cfg.w_id * float(lever["lever_recall"]) * float(lever["lever_precision"])
        else:
            id_term = 0.0
        reward = id_term + cfg.w_a * part_a + cfg.w_b * part_b
        if cfg.lever_gate and lever_ok and not evidence_gated:
            reward += cfg.lever_bonus * prec
    else:
        reward = cfg.w_a * part_a + cfg.w_b * part_b
        if cfg.lever_gate and lever_ok and not evidence_gated:
            reward += cfg.lever_bonus * prec
    reward -= cfg.c_invalid * invalid_frac
    if cfg.c_no_evidence and n_interventions == 0:
        reward -= cfg.c_no_evidence

    return {
        "reward": float(reward),
        "part_a": part_a, "part_b": part_b,
        "invalid_id_fraction": invalid_frac,
        "accepted": bool(g["accepted"]) and not evidence_gated and not lever_gated,
        "grade": g, "reward_error": False, "evidence_gated": evidence_gated,
        "lever_gated": lever_gated, **lever,
    }
