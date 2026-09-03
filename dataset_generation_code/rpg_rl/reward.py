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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from oracle_v6 import grade
from catalog import Catalog


@dataclass
class RewardConfig:
    w_a: float = 0.5              # weight on part A (found the fix)
    w_b: float = 0.5             # weight on part B (understood the mechanism)
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


def compute_reward(struct: Dict[str, Any], world: Dict[str, Any], cat: Catalog,
                   gold: Dict[str, Any], battery: Dict[str, Any],
                   cfg: RewardConfig = RewardConfig(),
                   n_interventions: Optional[int] = None) -> Dict[str, Any]:
    """Pure reward for one episode's final id-answer. Returns a dict with the scalar
    ``reward`` plus its components and the full grade (for logging/debugging)."""
    answer, invalid_frac = _to_canonical_answer(struct, cat)

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
        return {
            "reward": float(-cfg.c_invalid * invalid_frac),
            "part_a": 0.0, "part_b": 0.0,
            "invalid_id_fraction": invalid_frac, "accepted": False,
            "grade": {"error": f"{type(e).__name__}: {e}"}, "reward_error": True,
        }

    # FAITHFULNESS GATE (v9): no interventional evidence -> no discovery credit. Zero both parts
    # so an answer read off priors/labels without experimenting cannot score. (n_interventions is
    # the count of APPLIED interventions, passed by the env; None = caller didn't track -> no gate.)
    evidence_gated = bool(cfg.require_evidence and n_interventions == 0)
    if evidence_gated:
        part_a, part_b = 0.0, 0.0

    reward = cfg.w_a * part_a + cfg.w_b * part_b
    reward -= cfg.c_invalid * invalid_frac
    if cfg.c_no_evidence and n_interventions == 0:
        reward -= cfg.c_no_evidence

    return {
        "reward": float(reward),
        "part_a": part_a, "part_b": part_b,
        "invalid_id_fraction": invalid_frac,
        "accepted": bool(g["accepted"]) and not evidence_gated,
        "grade": g, "reward_error": False, "evidence_gated": evidence_gated,
    }
