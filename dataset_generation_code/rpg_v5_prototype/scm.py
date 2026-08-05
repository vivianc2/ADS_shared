#!/usr/bin/env python3
"""Generic structural-causal-model evaluator for RPG v5 worlds.

A world is a DAG of typed nodes plus knob-effects. The engine here is the
*single* interpreter for the whole world family: it samples exogenous latents,
applies interventions (knob settings and clamps) as structural operations, and
evaluates every node in topological order. Because the mechanisms are data, the
evaluator is faithful to the world definition by construction.

Node kinds
----------
- ``knob``       : agent-settable. ``dtype`` in {continuous, dose, binary}.
- ``latent``     : hidden state; either exogenous (``dist``) or computed
                   (``parents`` + ``mech``).
- ``observable`` : measurable = mechanism value + ``obs_noise``.
- ``outcome``    : the target the oracle optimizes.

Mechanism library (closed, pure, vectorized)
--------------------------------------------
linear, saturating, hill, soft_threshold, interaction, sign_flip.

This module has no dependency on the shared RPG engine; it is a clean slice to
validate the v5 design before porting into ``world_gen_rpg_old.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dose helpers (match v4 semantics: d in [0, 1])
# ---------------------------------------------------------------------------

def knob_dose(spec: Dict[str, Any], value: Any) -> float:
    """Map a submitted knob value to a dose fraction d in [0, 1]."""
    dtype = spec.get("dtype", "continuous")
    if dtype == "continuous":
        lo, hi = spec.get("range", [0.0, 100.0])
        if hi == lo:
            return 0.0
        return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))
    # categorical / binary / dose: index / (len - 1)
    values = spec.get("values", ["off", "on"])
    if value not in values:
        # tolerate baseline aliases
        if value in (None, "", "none", "off", "baseline"):
            return 0.0
        raise ValueError(f"value {value!r} not in {values}")
    idx = values.index(value)
    denom = max(1, len(values) - 1)
    return float(idx / denom)


# ---------------------------------------------------------------------------
# Mechanism library
# ---------------------------------------------------------------------------

def _mech_linear(mech: Dict[str, Any], vals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    out = np.full(n, float(mech.get("intercept", 0.0)))
    for parent, w in mech.get("weights", {}).items():
        out = out + float(w) * vals[parent]
    return out


def _mech_saturating(mech: Dict[str, Any], vals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    # gain * (x / (x + k)); x assumed >= 0
    x = vals[mech["of"]]
    gain = float(mech.get("gain", 1.0))
    k = float(mech.get("k", 1.0))
    return gain * (x / (x + k + 1e-9))


def _mech_hill(mech: Dict[str, Any], vals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    # vmax * x^h / (k^h + x^h)
    x = np.clip(vals[mech["of"]], 0.0, None)
    vmax = float(mech.get("vmax", 100.0))
    k = float(mech.get("k", 50.0))
    h = float(mech.get("n", 2.0))
    xh = np.power(x, h)
    return vmax * xh / (np.power(k, h) + xh + 1e-9)


def _mech_soft_threshold(mech: Dict[str, Any], vals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    # gain * sigmoid((x - thr) / width)
    x = vals[mech["of"]]
    gain = float(mech.get("gain", 1.0))
    thr = float(mech.get("threshold", 50.0))
    width = float(mech.get("width", 5.0))
    return gain / (1.0 + np.exp(-(x - thr) / width))


def _mech_interaction(mech: Dict[str, Any], vals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    # gain * (a/scale) * (b/scale) + intercept
    a = vals[mech["a"]]
    b = vals[mech["b"]]
    gain = float(mech.get("gain", 1.0))
    scale = float(mech.get("scale", 100.0))
    return float(mech.get("intercept", 0.0)) + gain * (a / scale) * (b / scale)


def _mech_sign_flip(mech: Dict[str, Any], vals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """Monotone-then-reversing effect of one parent.

    Below ``knee`` the slope is ``lo_gain`` (e.g. helpful); above it, ``hi_gain``
    dominates (e.g. harmful). Implemented as two ramps joined at the knee so the
    marginal effect literally changes sign.
    """
    x = vals[mech["of"]]
    knee = float(mech.get("knee", 50.0))
    lo_gain = float(mech.get("lo_gain", -0.3))
    hi_gain = float(mech.get("hi_gain", 0.9))
    intercept = float(mech.get("intercept", 0.0))
    below = np.minimum(x, knee)
    above = np.maximum(x - knee, 0.0)
    return intercept + lo_gain * below + hi_gain * above


_MECH = {
    "linear": _mech_linear,
    "saturating": _mech_saturating,
    "hill": _mech_hill,
    "soft_threshold": _mech_soft_threshold,
    "interaction": _mech_interaction,
    "sign_flip": _mech_sign_flip,
}


# ---------------------------------------------------------------------------
# Knob-effect expression helpers (dose -> scalar), closed set
# ---------------------------------------------------------------------------

def _sat(d: float, k: float = 0.66) -> float:
    """Saturating benefit fraction: full effect by dose ``k``, capped at 1."""
    return float(min(1.0, d / max(k, 1e-9)))


def _overstrip(d: float, thr: float = 0.66, gain: float = 25.0) -> float:
    """Over-treatment penalty: zero until ``thr``, linear beyond."""
    return float(gain * max(0.0, d - thr))


def _transient_boost(d: float, gain: float = 20.0, decay: float = 0.6) -> float:
    """Symptom-masking boost that has already partly decayed by measurement."""
    return float(gain * d * (1.0 - decay))


# ---------------------------------------------------------------------------
# SCM container + evaluator
# ---------------------------------------------------------------------------

@dataclass
class SCM:
    nodes: Dict[str, Dict[str, Any]]
    knob_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # obs_effects: knob -> {target observable, expr}. These bias the *measured*
    # value of an observable WITHOUT changing the true structural state or the
    # graded utility. This is how symptom-masking traps are modeled faithfully:
    # the agent sees the reading move, but the true outcome (and its utility) do
    # not. Applied in observe(), never in sample().
    obs_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outcome: str = "ProductYield"
    higher_is_better: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "knob_effects": self.knob_effects,
            "obs_effects": self.obs_effects,
            "outcome": self.outcome,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SCM":
        return cls(
            nodes=d["nodes"],
            knob_effects=d.get("knob_effects", {}),
            obs_effects=d.get("obs_effects", {}),
            outcome=d["outcome"],
            higher_is_better=bool(d.get("higher_is_better", True)),
        )

    def node_kind(self, name: str) -> str:
        return self.nodes[name]["kind"]

    def knobs(self) -> List[str]:
        return [n for n, s in self.nodes.items() if s["kind"] == "knob"]

    def observables(self) -> List[str]:
        return [n for n, s in self.nodes.items() if s["kind"] in ("observable", "outcome")]

    def latents(self) -> List[str]:
        return [n for n, s in self.nodes.items() if s["kind"] == "latent"]

    # --- topological order over computed (non-knob, non-exogenous) nodes ---
    def _topo_order(self) -> List[str]:
        order: List[str] = []
        visiting: set = set()
        done: set = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"cycle at {name}")
            visiting.add(name)
            for p in self.nodes[name].get("parents", []):
                visit(p)
            visiting.discard(name)
            done.add(name)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return order

    # --- sampling / evaluation ---
    def sample(
        self,
        n: int,
        intervention: Optional[Dict[str, Any]] = None,
        clamp: Optional[Dict[str, float]] = None,
        *,
        seed: int,
    ) -> Dict[str, np.ndarray]:
        """Return the *true* (noiseless-except-structural) value of every node
        under ``do(intervention)`` and optional ``clamp`` on observables.

        Observation noise is added later in :meth:`observe`, so this function is
        the ground-truth structural layer used by the oracle.
        """
        rng = np.random.default_rng(seed)
        intervention = dict(intervention or {})
        clamp = dict(clamp or {})
        vals: Dict[str, np.ndarray] = {}

        # doses for each knob-effect target
        dose_by_knob = {
            k: knob_dose(self.nodes[k], intervention.get(k, self.nodes[k].get("default")))
            for k in self.knobs()
        }

        for name in self._topo_order():
            spec = self.nodes[name]
            kind = spec["kind"]

            if name in clamp:
                vals[name] = np.full(n, float(clamp[name]))
                continue

            if kind == "knob":
                v = intervention.get(name, spec.get("default"))
                if spec.get("dtype", "continuous") == "continuous":
                    vals[name] = np.full(n, float(v))
                else:
                    # represent categorical knob by its dose for any downstream mech
                    vals[name] = np.full(n, knob_dose(spec, v))
                continue

            if kind == "latent" and "dist" in spec:
                dist = spec["dist"]
                if "normal" in dist:
                    mu, sd = dist["normal"]
                    vals[name] = rng.normal(mu, sd, n)
                elif "uniform" in dist:
                    lo, hi = dist["uniform"]
                    vals[name] = rng.uniform(lo, hi, n)
                else:
                    raise ValueError(f"unknown dist for {name}: {dist}")
                continue

            # computed node (latent/observable/outcome with parents + mech)
            mech = spec["mech"]
            base = _MECH[mech["form"]](mech, vals, n)
            if "noise" in spec:
                nd = spec["noise"]["normal"]
                base = base + rng.normal(nd[0], nd[1], n)
            vals[name] = base

        # apply knob_effects as structural ops on their target nodes
        vals = self._apply_knob_effects(vals, dose_by_knob, n)
        return vals

    def _apply_knob_effects(
        self, vals: Dict[str, np.ndarray], dose_by_knob: Dict[str, float], n: int
    ) -> Dict[str, np.ndarray]:
        """Apply scale/add/set knob-effects, then re-evaluate descendants.

        A knob-effect edits a *target* node (e.g. scales DissolvedCopper); every
        node downstream of that target must be recomputed so the effect
        propagates through the chain. We do this by editing the target then
        re-running the topo evaluation for descendants, reusing already-drawn
        upstream values so common-random-numbers hold across interventions.
        """
        if not self.knob_effects:
            return vals

        # Collect edits per target.
        edits: Dict[str, List[Dict[str, Any]]] = {}
        side_adds: List[Dict[str, Any]] = []
        for knob, eff in self.knob_effects.items():
            d = dose_by_knob.get(knob, 0.0)
            target = eff["target"]
            edits.setdefault(target, []).append({"op": eff["op"], "by": eff.get("by"), "d": d})
            se = eff.get("side_effect")
            if se:
                side_adds.append({"target": se["target"], "expr": se["expr"], "d": d})

        # Apply target edits.
        for target, ops in edits.items():
            for op in ops:
                d = op["d"]
                if op["op"] == "scale":
                    factor = self._eval_by(op["by"], d)
                    vals[target] = vals[target] * factor
                elif op["op"] == "add":
                    vals[target] = vals[target] + self._eval_by(op["by"], d) * np.ones(n)
                elif op["op"] == "set":
                    vals[target] = np.full(n, self._eval_by(op["by"], d))

        # Re-evaluate descendants of any edited target (propagate through chain).
        edited = set(edits.keys())
        for name in self._topo_order():
            spec = self.nodes[name]
            if name in edited or "mech" not in spec:
                continue
            parents = spec.get("parents", [])
            if any(p in edited or p in self._descendants_of(edited) for p in parents):
                mech = spec["mech"]
                vals[name] = _MECH[mech["form"]](mech, vals, n)
                edited.add(name)  # its own descendants must also refresh

        # Apply side-effect adds (e.g. over-strip penalty on the outcome).
        for sa in side_adds:
            vals[sa["target"]] = vals[sa["target"]] + self._eval_expr(sa["expr"], sa["d"]) * np.ones(n)

        return vals

    def _descendants_of(self, roots: set) -> set:
        desc: set = set()
        changed = True
        current = set(roots)
        while changed:
            changed = False
            for name, spec in self.nodes.items():
                if name in current or name in desc:
                    continue
                if any(p in current or p in desc for p in spec.get("parents", [])):
                    desc.add(name)
                    changed = True
        return desc

    def _eval_by(self, by: Any, d: float) -> float:
        if by is None:
            return 1.0
        if isinstance(by, (int, float)):
            return float(by)
        return self._eval_expr(by, d)

    def _eval_expr(self, expr: str, d: float) -> float:
        """Evaluate a tiny closed set of dose expressions. No eval()."""
        e = expr.strip()
        if e.startswith("1-sat"):
            k = _parse_kw(e, "k", 0.66)
            return 1.0 - _sat(d, k)
        if e.startswith("sat"):
            k = _parse_kw(e, "k", 0.66)
            return _sat(d, k)
        if e.startswith("-overstrip") or e.startswith("overstrip"):
            thr = _parse_kw(e, "thr", 0.66)
            gain = _parse_kw(e, "gain", 25.0)
            v = _overstrip(d, thr, gain)
            return -v if e.startswith("-") else v
        if e.startswith("-transient_boost"):
            return -_transient_boost(d)
        if e.startswith("transient_boost"):
            return _transient_boost(d)
        raise ValueError(f"unknown dose expr {expr!r}")

    def observe(
        self, vals: Dict[str, np.ndarray], measurements: List[str], *, seed: int,
        intervention: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Project true node values to noisy observable readings.

        ``obs_effects`` (symptom-masking) bias the *reading* here as a function
        of the intervention dose, without touching the true structural value or
        the utility. That is what makes a palliative knob a genuine trap: the
        measured outcome improves while the real state does not.
        """
        rng = np.random.default_rng(seed)
        intervention = dict(intervention or {})
        # per-observable additive bias from obs_effects
        bias: Dict[str, np.ndarray] = {}
        n = next(iter(vals.values())).shape[0] if vals else 0
        for knob, eff in self.obs_effects.items():
            d = knob_dose(self.nodes[knob], intervention.get(knob, self.nodes[knob].get("default")))
            target = eff["target"]
            bias.setdefault(target, np.zeros(n))
            bias[target] = bias[target] + self._eval_expr(eff["expr"], d)
        obs: Dict[str, np.ndarray] = {}
        for name in measurements:
            spec = self.nodes.get(name)
            if spec is None or spec["kind"] not in ("observable", "outcome"):
                raise ValueError(f"{name!r} is not an observable measurement")
            base = vals[name].astype(float).copy()
            if name in bias:
                base = base + bias[name]
            on = spec.get("obs_noise")
            if on and "normal" in on:
                base = base + rng.normal(on["normal"][0], on["normal"][1], base.shape[0])
            obs[name] = base
        return obs

    def utility(self, vals: Dict[str, np.ndarray]) -> np.ndarray:
        y = vals[self.outcome]
        return y if self.higher_is_better else -y


def _parse_kw(expr: str, key: str, default: float) -> float:
    """Extract ``key=<num>`` from an expr like ``1-sat(d;k=0.66)``."""
    import re

    m = re.search(rf"{key}\s*=\s*(-?\d+(?:\.\d+)?)", expr)
    return float(m.group(1)) if m else default
