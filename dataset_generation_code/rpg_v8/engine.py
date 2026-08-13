#!/usr/bin/env python3
"""RPG v6 SCM engine — large open-scenario worlds, actuators-only.

Generalizes the v5 evaluator. A world is:

- ``variables``: a DAG of typed nodes. Each carries meaningful ``aliases`` (for
  the free-text resolver), whether it is ``measurable`` (has an assay) and its
  ``assay_noise``, and its mechanism (exogenous ``dist`` or ``parents`` + ``mech``).
  Kinds: ``latent`` (no assay), ``observable`` (assay), ``outcome``.
- ``actuators``: first-class intervention handles. The agent can ONLY act
  through these (never set a causal variable by fiat). Each:
      {id, aliases, target, op in {set, scale, add}, dtype, range/values,
       default, expr (for scale/add doses), side_effect, description}
  ``set`` unifies v5's "knob" and "clamp": it forces ``target`` to a value.

Mechanism library and dose helpers are the v5 set (linear, saturating, hill,
soft_threshold, interaction, sign_flip; sat/overstrip/transient_boost).

Nothing about the causal math is agent-facing: the agent sees prose + resolver
replies only. This module is the ground truth the oracle and grader read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------
# dose + mechanism library (identical semantics to v5)
# --------------------------------------------------------------------------

def actuator_dose(act: Dict[str, Any], value: Any) -> float:
    dtype = act.get("dtype", "continuous")
    if dtype == "continuous":
        lo, hi = act.get("range", [0.0, 100.0])
        if hi == lo:
            return 0.0
        return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))
    values = act.get("values", ["off", "on"])
    if value in (None, "", "none", "off", "baseline"):
        return 0.0
    if value not in values:
        raise ValueError(f"value {value!r} not in {values}")
    return float(values.index(value) / max(1, len(values) - 1))


def _linear(m, v, n):
    out = np.full(n, float(m.get("intercept", 0.0)))
    for p, w in m.get("weights", {}).items():
        out = out + float(w) * v[p]
    return out


def _saturating(m, v, n):
    x = v[m["of"]]
    return float(m.get("gain", 1.0)) * (x / (x + float(m.get("k", 1.0)) + 1e-9))


def _hill(m, v, n):
    x = np.clip(v[m["of"]], 0.0, None)
    vmax, k, h = float(m.get("vmax", 100)), float(m.get("k", 50)), float(m.get("n", 2))
    xh = np.power(x, h)
    return vmax * xh / (np.power(k, h) + xh + 1e-9)


def _soft_threshold(m, v, n):
    x = v[m["of"]]
    return float(m.get("gain", 1.0)) / (1.0 + np.exp(-(x - float(m.get("threshold", 50))) / float(m.get("width", 5))))


def _interaction(m, v, n):
    a, b = v[m["a"]], v[m["b"]]
    scale = float(m.get("scale", 100.0))
    return float(m.get("intercept", 0.0)) + float(m.get("gain", 1.0)) * (a / scale) * (b / scale)


def _sign_flip(m, v, n):
    x = v[m["of"]]
    knee = float(m.get("knee", 50))
    below = np.minimum(x, knee)
    above = np.maximum(x - knee, 0.0)
    return float(m.get("intercept", 0.0)) + float(m.get("lo_gain", -0.3)) * below + float(m.get("hi_gain", 0.9)) * above


def _abs(m, v, n):
    """gain * |x - center| + intercept. Models 'distance from an optimum' (e.g.
    readmissions rise with |fluid balance - euvolemia|)."""
    x = v[m["of"]]
    return float(m.get("intercept", 0.0)) + float(m.get("gain", 1.0)) * np.abs(x - float(m.get("center", 0.0)))


def _gated_and(m, v, n):
    """AND-gate: output is high only when BOTH inputs clear their thresholds.
    output = vmax * sigmoid((a-ta)/wa) * sigmoid((b-tb)/wb) + intercept.
    Either input low -> its sigmoid ~0 -> output ~intercept. This models a true
    two-required-causes mechanism where neither lever helps alone."""
    a, b = v[m["a"]], v[m["b"]]
    ta, tb = float(m.get("ta", 50)), float(m.get("tb", 50))
    wa, wb = float(m.get("wa", 8)), float(m.get("wb", 8))
    ga = 1.0 / (1.0 + np.exp(-(a - ta) / wa))
    gb = 1.0 / (1.0 + np.exp(-(b - tb) / wb))
    return float(m.get("intercept", 0.0)) + float(m.get("vmax", 80.0)) * ga * gb


def _subtype_effect(m, v, n):
    """Treatment effect whose SIGN depends on a hidden subtype. Models effect
    heterogeneity: the same dose helps one subgroup and harms the other, so the
    population-average effect is ~0 and the ideal answer is a CONDITIONAL POLICY
    (treat only the subgroup it helps), not a single dose.

        out = gain * (dose/scale) * sign,  sign = +1 if subtype>=center else -1

    ``dose`` is the treatment-dose variable (set by an actuator, 0 at baseline);
    ``subtype`` is the hidden subtype latent. With subtype split ~50/50 around
    ``center`` and dose applied uniformly, the +1 and -1 halves cancel."""
    dose = v[m["dose"]]
    sub = v[m["subtype"]]
    sign = np.where(sub >= float(m.get("center", 50.0)), 1.0, -1.0)
    return float(m.get("gain", 1.0)) * (dose / float(m.get("scale", 100.0))) * sign


_MECH = {"linear": _linear, "saturating": _saturating, "hill": _hill,
         "soft_threshold": _soft_threshold, "interaction": _interaction,
         "gated_and": _gated_and, "abs": _abs,
         "subtype_effect": _subtype_effect,
         "sign_flip": _sign_flip}


def _sat(d, k=0.66):
    return float(min(1.0, d / max(k, 1e-9)))


def _overstrip(d, thr=0.66, gain=25.0):
    return float(gain * max(0.0, d - thr))


def _transient_boost(d, gain=20.0, decay=0.6):
    return float(gain * d * (1.0 - decay))


def _parse_kw(expr, key, default):
    m = re.search(rf"{key}\s*=\s*(-?\d+(?:\.\d+)?)", expr)
    return float(m.group(1)) if m else default


def eval_expr(expr: str, d: float) -> float:
    e = expr.strip()
    if e.startswith("1-sat"):
        return 1.0 - _sat(d, _parse_kw(e, "k", 0.66))
    if e.startswith("sat"):
        return _sat(d, _parse_kw(e, "k", 0.66))
    if e.startswith("-overstrip"):
        return -_overstrip(d, _parse_kw(e, "thr", 0.66), _parse_kw(e, "gain", 25.0))
    if e.startswith("overstrip"):
        return _overstrip(d, _parse_kw(e, "thr", 0.66), _parse_kw(e, "gain", 25.0))
    if e.startswith("-transient_boost"):
        return -_transient_boost(d)
    if e.startswith("transient_boost"):
        return _transient_boost(d)
    raise ValueError(f"unknown dose expr {expr!r}")


# --------------------------------------------------------------------------
# World container + evaluator
# --------------------------------------------------------------------------

@dataclass
class WorldSCM:
    variables: Dict[str, Dict[str, Any]]
    actuators: Dict[str, Dict[str, Any]]
    outcome: str
    higher_is_better: bool = True
    # Optional SELECTION spec. Models how the OBSERVATIONAL record was formed:
    # historical units are retained into the dataset only when a collider node
    # clears a threshold (e.g. only cases that got escalated/logged are on file).
    # Conditioning on such a collider opens a spurious path between its parents,
    # so two causally-independent variables look correlated in the observational
    # data yet the correlation VANISHES under do() (the record is no longer
    # selection-filtered once you run a controlled experiment). This is the
    # canonical collider/selection-bias trap.
    #   {"node": <collider var>, "op": ">="|"<=", "thresh": float,
    #    "soft": float (optional; logistic width for probabilistic selection)}
    # Selection is applied ONLY to observational sampling (select=True) and is
    # deliberately ignored under any actuator intervention, so the oracle's
    # interventional ground truth is never selection-distorted.
    selection: Optional[Dict[str, Any]] = None

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        d = {"variables": self.variables, "actuators": self.actuators,
             "outcome": self.outcome, "higher_is_better": self.higher_is_better}
        if self.selection is not None:
            d["selection"] = self.selection
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldSCM":
        return cls(variables=d["variables"], actuators=d["actuators"],
                   outcome=d["outcome"], higher_is_better=bool(d.get("higher_is_better", True)),
                   selection=d.get("selection"))

    # ---- catalogs ----
    def measurable_vars(self) -> List[str]:
        return [n for n, s in self.variables.items()
                if s["kind"] in ("observable", "outcome") or s.get("measurable")]

    def outcome_ancestors(self) -> set:
        """Names that can causally affect the outcome (transitive parents)."""
        anc: set = set()
        stack = [self.outcome]
        while stack:
            cur = stack.pop()
            for p in self.variables.get(cur, {}).get("parents", []):
                if p not in anc:
                    anc.add(p)
                    stack.append(p)
        return anc

    def _topo(self) -> List[str]:
        order, visiting, done = [], set(), set()

        def visit(nm):
            if nm in done:
                return
            if nm in visiting:
                raise ValueError(f"cycle at {nm}")
            visiting.add(nm)
            for p in self.variables[nm].get("parents", []):
                visit(p)
            visiting.discard(nm)
            done.add(nm)
            order.append(nm)

        for nm in self.variables:
            visit(nm)
        return order

    def _descendants(self, roots: set) -> set:
        desc, cur, changed = set(), set(roots), True
        while changed:
            changed = False
            for nm, s in self.variables.items():
                if nm in cur or nm in desc:
                    continue
                if any(p in cur or p in desc for p in s.get("parents", [])):
                    desc.add(nm)
                    changed = True
        return desc

    # ---- sampling ----
    def _selection_active(self, intervention: Dict[str, Any]) -> bool:
        """Selection describes how the OBSERVATIONAL record was formed. Under any
        real (non-mask) actuator intervention the agent is running a controlled
        experiment, so the historical selection filter no longer applies."""
        if self.selection is None:
            return False
        for aid, act in self.actuators.items():
            if aid in intervention and act.get("op") != "mask":
                return False
        return True

    def _selection_mask(self, vals: Dict[str, np.ndarray], rng) -> np.ndarray:
        """Boolean keep-mask implementing the selection spec on a collider node.
        Hard threshold by default; ``soft`` gives logistic (probabilistic)
        selection so the retained sample is not a razor-sharp truncation."""
        sel = self.selection
        x = vals[sel["node"]]
        thr = float(sel["thresh"])
        op = sel.get("op", ">=")
        soft = sel.get("soft")
        if soft:
            z = (x - thr) / float(soft)
            p = 1.0 / (1.0 + np.exp(-z))
            if op == "<=":
                p = 1.0 - p
            return rng.random(x.shape[0]) < p
        return (x >= thr) if op == ">=" else (x <= thr)

    def sample(self, n: int, intervention: Optional[Dict[str, Any]] = None, *,
               seed: int, select: bool = False) -> Dict[str, np.ndarray]:
        """Return structural values of every variable, optionally under an actuator
        intervention (maps actuator_id -> value). If ``select`` is True and the
        world carries a selection spec (and no real intervention is active),
        return a SELECTED observational sample (units passing the collider filter)
        via oversample-and-filter, holding the returned count at exactly ``n``."""
        intervention = dict(intervention or {})
        if not (select and self._selection_active(intervention)):
            return self._sample_raw(n, intervention, seed=seed)
        rng = np.random.default_rng(seed ^ 0x5E1EC7)
        kept: Dict[str, List[np.ndarray]] = {}
        got, attempt = 0, 0
        while got < n and attempt < 200:
            attempt += 1
            m = int((n - got) * 3) + 32
            raw = self._sample_raw(m, intervention, seed=seed + 100003 * attempt)
            mask = self._selection_mask(raw, rng)
            idx = np.nonzero(mask)[0]
            if idx.size:
                for k, v in raw.items():
                    kept.setdefault(k, []).append(v[idx])
                got += int(idx.size)
        if got == 0:
            # selection filtered everything (mis-specified spec) -> fall back to
            # the unselected sample rather than returning an empty frame.
            return self._sample_raw(n, intervention, seed=seed)
        return {k: np.concatenate(v)[:n] for k, v in kept.items()}

    def _sample_raw(self, n: int, intervention: Optional[Dict[str, Any]] = None, *, seed: int) -> Dict[str, np.ndarray]:
        """Return true structural values of every variable under an actuator
        intervention. ``intervention`` maps actuator_id -> value."""
        rng = np.random.default_rng(seed)
        intervention = dict(intervention or {})
        vals: Dict[str, np.ndarray] = {}

        # 'set' actuators force their target to a constant (overrides mechanism).
        set_targets: Dict[str, float] = {}
        policy_sets: List[Dict[str, Any]] = []   # deferred conditional (per-unit) sets
        scale_add: List[Dict[str, Any]] = []
        side_adds: List[Dict[str, Any]] = []
        for aid, act in self.actuators.items():
            if aid not in intervention:
                continue
            if act["op"] == "mask":
                # symptom-masking actuators bias the READING only (applied in
                # measure()); they never touch the structural/utility layer.
                continue
            v = intervention[aid]
            if act["op"] == "set":
                # A CONDITIONAL POLICY sets the target per-unit as a function of an
                # observed stratifier variable: {"policy": {"stratifier","threshold",
                # "dose_if_ge","dose_if_lt"}}. Applied AFTER the base topo pass (the
                # stratifier must be computed first), then descendants re-propagate.
                if isinstance(v, dict) and "policy" in v:
                    policy_sets.append({"target": act["target"], "policy": v["policy"]})
                    continue
                # continuous set: numeric value. Guard against a non-numeric value
                # slipping through (should be normalized upstream by the resolver);
                # fall back to the actuator default rather than crashing the run.
                try:
                    set_targets[act["target"]] = float(v)
                except (TypeError, ValueError):
                    set_targets[act["target"]] = float(act.get("default", 0.0))
            else:
                d = actuator_dose(act, v)
                scale_add.append({"target": act["target"], "op": act["op"],
                                  "expr": act.get("expr"), "d": d})
                if act.get("side_effect"):
                    se = act["side_effect"]
                    side_adds.append({"target": se["target"], "expr": se["expr"], "d": d})

        for nm in self._topo():
            spec = self.variables[nm]
            if nm in set_targets:
                vals[nm] = np.full(n, set_targets[nm])
                continue
            if "dist" in spec:
                dist = spec["dist"]
                if "normal" in dist:
                    vals[nm] = rng.normal(dist["normal"][0], dist["normal"][1], n)
                elif "uniform" in dist:
                    vals[nm] = rng.uniform(dist["uniform"][0], dist["uniform"][1], n)
                else:
                    raise ValueError(f"bad dist {dist}")
                continue
            if "mech" in spec:
                base = _MECH[spec["mech"]["form"]](spec["mech"], vals, n)
                if "noise" in spec:
                    nd = spec["noise"]["normal"]
                    base = base + rng.normal(nd[0], nd[1], n)
                vals[nm] = base
            else:
                # a bare exogenous variable with no dist/mech -> constant 0 baseline
                vals[nm] = np.zeros(n)

        # apply scale/add actuator effects to their targets, then re-propagate
        edited = set()
        for op in scale_add:
            t, d = op["target"], op["d"]
            factor = eval_expr(op["expr"], d) if op["expr"] else 1.0
            if op["op"] == "scale":
                vals[t] = vals[t] * factor
            elif op["op"] == "add":
                vals[t] = vals[t] + factor * np.ones(n)
            edited.add(t)
        edited |= set(set_targets)
        # conditional-policy sets: dose each unit per an observed stratifier. The
        # stratifier is read from the already-computed base pass, so the policy
        # acts on what the agent could actually observe about each unit.
        for ps in policy_sets:
            pol = ps["policy"]
            strat = vals[pol["stratifier"]]
            dose = np.where(strat >= float(pol["threshold"]),
                            float(pol.get("dose_if_ge", 0.0)),
                            float(pol.get("dose_if_lt", 0.0)))
            vals[ps["target"]] = dose.astype(float)
            edited.add(ps["target"])

        if edited:
            desc = self._descendants(edited)
            for nm in self._topo():
                if nm in edited or nm not in desc or "mech" not in self.variables[nm]:
                    continue
                vals[nm] = _MECH[self.variables[nm]["mech"]["form"]](self.variables[nm]["mech"], vals, n)

        for sa in side_adds:
            vals[sa["target"]] = vals[sa["target"]] + eval_expr(sa["expr"], sa["d"]) * np.ones(n)

        return vals

    def measure(self, vals: Dict[str, np.ndarray], names: List[str], *, seed: int,
                intervention: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Noisy assay readings for measurable variables. Symptom-masking
        actuators (op='mask') bias the reading without changing structure."""
        rng = np.random.default_rng(seed)
        intervention = dict(intervention or {})
        n = next(iter(vals.values())).shape[0] if vals else 0
        bias: Dict[str, np.ndarray] = {}
        for aid, act in self.actuators.items():
            if act.get("op") == "mask" and aid in intervention:
                d = actuator_dose(act, intervention[aid])
                bias.setdefault(act["target"], np.zeros(n))
                bias[act["target"]] += eval_expr(act["expr"], d)
        obs = {}
        for nm in names:
            spec = self.variables.get(nm)
            if spec is None or not (spec["kind"] in ("observable", "outcome") or spec.get("measurable")):
                raise ValueError(f"{nm!r} has no assay (not measurable)")
            base = vals[nm].astype(float).copy()
            if nm in bias:
                base = base + bias[nm]
            an = spec.get("assay_noise")
            if an and "normal" in an:
                base = base + rng.normal(an["normal"][0], an["normal"][1], base.shape[0])
            obs[nm] = base
        return obs

    def utility(self, vals: Dict[str, np.ndarray]) -> np.ndarray:
        y = vals[self.outcome]
        return y if self.higher_is_better else -y
