#!/usr/bin/env python3
"""Cue-free variant of a v10 world (evaluation split for RL, 2026-10-23).

THE CUE. In v10 every level control (a setting whose actual level is also measured) drifts naturally from unit to
unit. For a true lever that drift runs through the mechanism into the outcome, so the first observational batch shows
|corr(level, outcome)| of 0.3-0.65 (probe: cuefree_probe.py, 180 worlds); for inert settings the drift is independent
noise and |r| is ~0.01. Ranking settings by |r| therefore finds the lever without any experimental reasoning (the
no-graph |r| script: 52.8% part A on the 540 test worlds).

THE CHANGE. Operators in real plants react to the problem: some settings get turned up or down when things go badly.
`make_cuefree(world)` rewires every INERT level setting so that its observational level follows a hidden exogenous
driver of the problem (the latent with the largest |corr| with the outcome: the root's susceptibility or a
confounder), while keeping the setting's natural mean and SD (so the situation report's "usually about X" stays
true):  X = mu + s * a * sd * (L - mean_L) / sd_L + e,  e ~ N(0, sd * sqrt(1 - a^2)),  a = target_r / |rho_L|.
The target |r| of each such setting is drawn around the world's own lever |r| values (x U(0.6, 1.4), capped at
0.9 |rho_L|; worlds without a primary level lever draw from U(0.3, 0.65), the probe's range); the sign is random.
So passive correlation no longer singles out the lever: the lever's |r| rank among the level settings is at chance.
Nothing changes under intervention: X has no children, and setting a control overrides X, so every gold value,
mechanism, budget and battery item is the same as in the paired standard world (checked by build_cuefree.py).
Draws use their own stream (numpy default_rng([seed, 0xC0FFEE])), so the standard world of the seed is untouched.

V3 (`make_confounded`, 2026-10-24, camera-ready session; replaces v2 for new builds). v2 left a passive shortcut: every
rewired inert setting follows the SAME driver, so the inert settings correlate with each other and the lever (whose
drift stays independent) is the odd one out. One observational batch then finds the lever in most cue-free worlds:
mean |corr with the other settings| (lowest first) puts it first in 79% of cue-free test worlds, |corr with the
outcome| after removing the other settings' first principal component in 89% (personal_docs
week7/CORE108_AND_CODE_STATE_2026-10-23.md section 3). v3 gives every level setting (lever, inert and harm roles
alike; settings whose level has parents are left alone) one of three observational regimes, drawn independently:
  held        the setting is kept at its usual level by procedure: N(mu, 0.05 sd), so passive data carry almost no
              information about it (common for real setpoints);
  free        unchanged natural drift N(mu, sd) (a natural experiment for that setting);
  responsive  operators adjust it as conditions change: X = mu + s * a * sd * Z + e, e ~ N(0, sd * sqrt(1 - a^2)),
              Z = c * u_D + sqrt(1 - c^2) * u_W, where u_D is the standardised best linear predictor of the outcome
              from the exogenous latents (v2's driver) and u_W a standardised random mix of 1-3 other latents,
              orthogonal to u_D. Each responsive setting has its own c, W, a and sign, so responsive settings do
              not share one driver.
Regime probabilities: lever (primary) held 0.50 / free 0.20 / responsive 0.30; other settings 0.30 / 0.30 / 0.40.
The target |corr(X, outcome)| of a responsive setting is drawn around the world's lever |r| (x U(0.6, 1.4)); a is
raised (up to 0.97) when the target needs it.
Why these numbers (set before the validation build): a lever with its own natural variation (free, or the private
part of a responsive lever) is the only setting whose own variation moves the outcome, so adjusting for the other
settings identifies it from passive data. That is real observational causal inference, not a generator artefact, so
v3 does not try to remove it; it limits how often the lever varies on its own (held in half the worlds, as many real
setpoints are) and makes plain |r| and the odd-one-out pattern uninformative. Targets: plain |r| and odd-one-out put
the lever first in <= 30% of level-lever worlds; full linear adjustment in <= 50%; experiments unaffected.
Interventions override X, so every gold intervention still works; but the lever's (and harm settings')
observational distribution can change, so the baseline (no-intervention) utility and draws of experiments that leave
those settings alone can differ from the paired standard world. The oracle is recomputed for the new world
(build_cuefree.py) and every audit is re-run. Goal: no passive statistic reliably names the lever (validated by
cuefree_shortcut_probe.py: |r|, odd-one-out, partial-after-PC1 and full-regression partial |r|), while experiments
still identify it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

N_PROBE = 4000


def _level_controls(world):
    scm = world["scm"]
    roles = world["ground_truth"].get("_v10_roles") or {}
    out = []
    for aid, a in scm.actuators.items():
        t = scm.variables.get(a.get("target"), {})
        if a.get("op") == "set" and t.get("kind") == "observable" and t.get("measurable"):
            out.append((aid, a["target"], roles.get(aid)))
    return out


def make_cuefree(world: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Mutates world["scm"] in place (before audit / oracle); returns a record of what changed."""
    scm = world["scm"]
    rng = np.random.default_rng([int(seed), 0xC0FFEE])
    vals = scm.sample(N_PROBE, seed=int(seed) + 77, select=True)
    o = np.asarray(vals[scm.outcome], float)

    def corr(x):
        x = np.asarray(x, float)
        return float(np.corrcoef(x, o)[0, 1]) if x.std() > 1e-9 and o.std() > 1e-9 else 0.0
    lat = [n for n, v in scm.variables.items()
           if v.get("kind") == "latent" and "dist" in v and "normal" in v["dist"] and not v.get("parents")
           and float(v["dist"]["normal"][1]) > 0]
    rho = {n: corr(vals[n]) for n in lat}
    # v2 (2026-10-23): the driver is the best linear predictor of the outcome from ALL exogenous latents (OLS on the
    # probe), not the single most correlated one: its correlation R with the outcome is higher, so the rewired settings
    # can match levers whose |r| exceeds every single latent's (v1 left the lever-vs-inert |r| AUC at 0.65).
    Lm = np.column_stack([np.asarray(vals[n], float) for n in lat])
    mu_L = np.array([float(scm.variables[n]["dist"]["normal"][0]) for n in lat])
    sd_L = np.array([float(scm.variables[n]["dist"]["normal"][1]) for n in lat])
    Zs = (Lm - mu_L) / sd_L
    beta, *_ = np.linalg.lstsq(np.column_stack([Zs, np.ones(len(o))]), o, rcond=None)
    beta = beta[:-1]
    sd_Z = float(np.sqrt(np.sum(beta ** 2)))          # latents are independent N(mu, sd): var(sum b_i z_i) = sum b_i^2
    R = corr(Zs @ beta)
    L, rho_L = "+".join(lat), R
    levels = _level_controls(world)
    lever_r = [abs(corr(vals[t])) for _, t, role in levels if role == "primary"]
    changed: List[Dict[str, Any]] = []
    for aid, t, role in levels:
        spec = scm.variables[t]
        if role != "inert" or "dist" not in spec or "normal" not in spec["dist"]:
            continue
        mu, sd = (float(x) for x in spec["dist"]["normal"])
        base = float(rng.choice(lever_r)) if lever_r else float(rng.uniform(0.3, 0.65))
        target = float(np.clip(base * rng.uniform(0.6, 1.4), 0.05, 0.97 * abs(rho_L)))
        a = target / abs(rho_L)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        # X = mu + sign*a*sd * Z/sd_Z + e, Z = sum_i beta_i (L_i - mu_i)/sd_i
        ws = {n: sign * a * sd * float(b) / (sd_Z * float(si)) for n, b, si in zip(lat, beta, sd_L)}
        spec.pop("dist")
        spec["parents"] = list(lat)
        spec["mech"] = {"form": "linear", "weights": ws,
                        "intercept": mu - sum(ws[n] * float(m) for n, m in zip(lat, mu_L))}
        spec["noise"] = {"normal": [0.0, sd * math.sqrt(max(0.0, 1.0 - a * a))]}
        spec["_cuefree"] = {"mu": mu, "sd": sd, "driver": L, "target_r": target, "sign": sign}
        changed.append({"control": aid, "level": t, "target_r": round(target, 3), "sign": sign})
    # realised correlations after the change (same probe)
    vals2 = scm.sample(N_PROBE, seed=int(seed) + 77, select=True)
    o2 = np.asarray(vals2[scm.outcome], float)
    real = {t: float(np.corrcoef(np.asarray(vals2[t], float), o2)[0, 1]) for _, t, _ in levels
            if np.asarray(vals2[t]).std() > 1e-9}
    rec = {"version": 2, "driver": L, "rho_driver": round(rho_L, 3),
           "rho_single": {n: round(v, 3) for n, v in rho.items()}, "lever_r": [round(x, 3) for x in lever_r],
           "changed": changed, "realised_r": {k: round(v, 3) for k, v in real.items()},
           "roles": {t: role for _, t, role in levels}}
    world["ground_truth"]["_cuefree"] = rec
    return rec


# ------------------------------------------------------------------------------------------------------------------
# v3: heterogeneous observational regimes (see module docstring)
P_LEVER = {"held": 0.50, "free": 0.20, "responsive": 0.30}
P_OTHER = {"held": 0.30, "free": 0.30, "responsive": 0.40}
HELD_SD_FRAC = 0.05


def _exogenous_latents(scm):
    return [n for n, v in scm.variables.items()
            if v.get("kind") == "latent" and "dist" in v and "normal" in v["dist"] and not v.get("parents")
            and float(v["dist"]["normal"][1]) > 0]


def make_confounded(world: Dict[str, Any], seed: int, attempt: int = 0) -> Dict[str, Any]:
    """v3. Mutates world["scm"] in place (after calibrate, before the oracle); returns a record of what changed.
    `attempt` selects an independent regime draw (build_cuefree.py redraws when the new world fails an audit)."""
    scm = world["scm"]
    rng = np.random.default_rng([int(seed), 0xC0FFEE, 3, int(attempt)])
    vals = scm.sample(N_PROBE, seed=int(seed) + 77, select=True)
    o = np.asarray(vals[scm.outcome], float)

    def corr(x):
        x = np.asarray(x, float)
        return float(np.corrcoef(x, o)[0, 1]) if x.std() > 1e-9 and o.std() > 1e-9 else 0.0
    lat = _exogenous_latents(scm)
    mu_L = np.array([float(scm.variables[n]["dist"]["normal"][0]) for n in lat])
    sd_L = np.array([float(scm.variables[n]["dist"]["normal"][1]) for n in lat])
    Zs = (np.column_stack([np.asarray(vals[n], float) for n in lat]) - mu_L) / sd_L
    beta, *_ = np.linalg.lstsq(np.column_stack([Zs, np.ones(len(o))]), o, rcond=None)
    beta = beta[:-1]
    u_d = beta / max(float(np.linalg.norm(beta)), 1e-12)       # unit direction in standardised-latent space
    R = abs(corr(Zs @ u_d))                                       # corr(driver, outcome)
    levels = _level_controls(world)
    lever_r = [abs(corr(vals[t])) for _, t, role in levels if role == "primary"]
    changed: List[Dict[str, Any]] = []
    for aid, t, role in levels:
        spec = scm.variables[t]
        if role not in ("primary", "inert", "harm") or "dist" not in spec or "normal" not in spec["dist"]:
            continue
        mu, sd = (float(x) for x in spec["dist"]["normal"])
        p = P_LEVER if role == "primary" else P_OTHER
        regime = str(rng.choice(list(p), p=list(p.values())))
        if regime == "free":
            changed.append({"control": aid, "level": t, "role": role, "regime": regime})
            continue
        if regime == "held":
            spec["dist"] = {"normal": [mu, HELD_SD_FRAC * sd]}
            spec["_cuefree"] = {"version": 3, "regime": regime, "mu": mu, "sd": sd}
            changed.append({"control": aid, "level": t, "role": role, "regime": regime})
            continue
        # responsive: own mix of latents
        k = int(rng.integers(1, 4))
        pick = rng.choice(len(lat), size=min(k, len(lat)), replace=False)
        w = np.zeros(len(lat))
        w[pick] = rng.normal(size=len(pick))
        w = w - (w @ u_d) * u_d                                   # orthogonal to the driver
        nw = float(np.linalg.norm(w))
        u_w = w / nw if nw > 1e-9 else np.zeros(len(lat))
        base = float(rng.choice(lever_r)) if lever_r else float(rng.uniform(0.3, 0.65))
        target = float(np.clip(base * rng.uniform(0.6, 1.4), 0.02, 0.9))
        a = float(rng.uniform(0.6, 0.95))
        a = float(min(0.97, max(a, target / max(R, 1e-9))))
        c = float(np.clip(target / max(a * R, 1e-9), 0.0, 1.0))
        z = c * u_d + np.sqrt(max(0.0, 1.0 - c * c)) * u_w
        z = z / max(float(np.linalg.norm(z)), 1e-12)               # unit variance: latents are independent N(mu, sd)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        ws = {n: sign * a * sd * float(zi) / float(si) for n, zi, si in zip(lat, z, sd_L) if abs(zi) > 1e-12}
        spec.pop("dist")
        spec["parents"] = list(ws)
        spec["mech"] = {"form": "linear", "weights": ws,
                        "intercept": mu - sum(ws[n] * float(m) for n, m in zip(lat, mu_L) if n in ws)}
        spec["noise"] = {"normal": [0.0, sd * math.sqrt(max(0.0, 1.0 - a * a))]}
        spec["_cuefree"] = {"version": 3, "regime": regime, "mu": mu, "sd": sd, "a": a, "c": c, "sign": sign,
                            "target_r": target, "latents": sorted(ws)}
        changed.append({"control": aid, "level": t, "role": role, "regime": regime, "target_r": round(target, 3)})
    vals2 = scm.sample(N_PROBE, seed=int(seed) + 77, select=True)
    o2 = np.asarray(vals2[scm.outcome], float)
    real = {t: float(np.corrcoef(np.asarray(vals2[t], float), o2)[0, 1]) for _, t, _ in levels
            if np.asarray(vals2[t]).std() > 1e-9}
    rec = {"version": 3, "attempt": int(attempt), "rho_driver": round(R, 3), "lever_r": [round(x, 3) for x in lever_r], "changed": changed,
           "realised_r": {k: round(v, 3) for k, v in real.items()}, "roles": {t: role for _, t, role in levels}}
    world["ground_truth"]["_cuefree"] = rec
    return rec
