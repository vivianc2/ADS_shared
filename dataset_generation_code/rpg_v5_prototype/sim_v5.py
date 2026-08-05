#!/usr/bin/env python3
"""Runtime simulator for RPG v5 SCM worlds.

Loads a ``rpg_scm_v5`` world JSON and serves typed queries against the hidden
SCM while exposing only the agent-facing catalog. Four query modes:

- ``observational_sample``  : draw n units under the default (no intervention).
- ``interventional_sample`` : draw n units under do(intervention).
- ``sweep``                 : per-level means +/- SE for one knob over a grid.
- ``clamp``                 : force one clampable observable to fixed levels
                              (breaks a confound), report the outcome per level.

Returns compact statistical summaries (means, SDs, correlations) rather than raw
rows, which is what the scientist reasons over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scm import SCM


class SimV5:
    def __init__(self, record: Dict[str, Any]):
        self.record = record
        self.visible = record["visible"]
        self.hidden = record["hidden"]
        self.oracle = record["oracle"]
        self.scm = SCM.from_dict(self.hidden["scm"])
        self.seed = int(record.get("meta", {}).get("seed", 0))
        self.allowed_obs = [o["name"] for o in self.visible["observed_variables"]]
        self.knobs = {k["name"]: k for k in self.visible["action_variables"]}
        self.clampable = set(self.visible.get("clampable_measurements", []))
        self.max_queries = int(self.visible["experiment_budget"]["max_queries"])
        self.max_units = int(self.visible["experiment_budget"]["max_units_per_query"])

    @classmethod
    def from_json(cls, path: str) -> "SimV5":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    # --- validation ---
    def _check_measurements(self, ms: Optional[List[str]]) -> List[str]:
        ms = ms or self.allowed_obs
        bad = [m for m in ms if m not in self.allowed_obs]
        if bad:
            raise ValueError(f"unknown measurements: {bad}; allowed={self.allowed_obs}")
        return ms

    def _check_intervention(self, iv: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (iv or {}).items():
            if k not in self.knobs:
                raise ValueError(f"unknown knob {k!r}; allowed={list(self.knobs)}")
            spec = self.knobs[k]
            if spec["value_type"] == "continuous":
                lo, hi = spec["range"]
                fv = float(v)
                if not (lo - 1e-9 <= fv <= hi + 1e-9):
                    raise ValueError(f"{k}={v} out of range {spec['range']}")
                out[k] = fv
            else:
                if v not in spec["values"]:
                    raise ValueError(f"{k}={v!r} not in {spec['values']}")
                out[k] = v
        return out

    def _n(self, n_units: Any) -> int:
        n = int(n_units or 200)
        return max(1, min(n, self.max_units))

    def _summ(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        summ = {}
        for name, arr in obs.items():
            summ[name] = {"mean": round(float(np.mean(arr)), 3),
                          "sd": round(float(np.std(arr)), 3),
                          "se": round(float(np.std(arr) / np.sqrt(len(arr))), 3)}
        # pairwise correlations among returned measurements
        names = list(obs)
        corr = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = obs[names[i]], obs[names[j]]
                c = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
                corr[f"{names[i]}~{names[j]}"] = round(c, 3)
        return {"stats": summ, "correlations": corr}

    # --- query modes ---
    def run(self, query: Dict[str, Any], *, call_idx: int) -> Dict[str, Any]:
        mode = query.get("mode")
        base_seed = self.seed + 10007 * (call_idx + 1)
        if mode == "observational_sample":
            n = self._n(query.get("n_units"))
            ms = self._check_measurements(query.get("measurements"))
            vals = self.scm.sample(n, seed=base_seed)
            obs = self.scm.observe(vals, ms, seed=base_seed + 3)
            return {"mode": mode, "n_units": n, **self._summ(obs)}

        if mode == "interventional_sample":
            n = self._n(query.get("n_units"))
            ms = self._check_measurements(query.get("measurements"))
            iv = self._check_intervention(query.get("intervention", {}))
            vals = self.scm.sample(n, intervention=iv, seed=base_seed)
            obs = self.scm.observe(vals, ms, seed=base_seed + 3, intervention=iv)
            return {"mode": mode, "n_units": n, "intervention": iv, **self._summ(obs)}

        if mode == "sweep":
            knob = query.get("knob")
            if knob not in self.knobs:
                raise ValueError(f"unknown knob {knob!r}")
            ms = self._check_measurements(query.get("measurements"))
            n = self._n(query.get("n_units", 200))
            grid = query.get("grid")
            if not grid:
                spec = self.knobs[knob]
                if spec["value_type"] == "continuous":
                    lo, hi = spec["range"]
                    grid = [round(x, 2) for x in np.linspace(lo, hi, 5)]
                else:
                    grid = spec["values"]
            rows = []
            for i, lvl in enumerate(grid):
                iv = self._check_intervention({knob: lvl})
                vals = self.scm.sample(n, intervention=iv, seed=base_seed + 17 * (i + 1))
                obs = self.scm.observe(vals, ms, seed=base_seed + 3 + i, intervention=iv)
                rows.append({knob: lvl,
                             **{m: round(float(np.mean(obs[m])), 3) for m in ms}})
            return {"mode": mode, "knob": knob, "n_units": n, "grid": grid, "rows": rows}

        if mode == "clamp":
            node = query.get("node")
            if node not in self.clampable:
                raise ValueError(f"{node!r} not clampable; allowed={sorted(self.clampable)}")
            ms = self._check_measurements(query.get("measurements"))
            n = self._n(query.get("n_units", 200))
            levels = query.get("levels") or [20.0, 80.0]
            rows = []
            for i, lvl in enumerate(levels):
                vals = self.scm.sample(n, clamp={node: float(lvl)}, seed=base_seed + 29 * (i + 1))
                obs = self.scm.observe(vals, ms, seed=base_seed + 3 + i)
                rows.append({f"clamp_{node}": lvl,
                             **{m: round(float(np.mean(obs[m])), 3) for m in ms}})
            return {"mode": mode, "node": node, "n_units": n, "levels": levels, "rows": rows}

        raise ValueError(f"unknown query mode {mode!r}")

    def public_world(self) -> Dict[str, Any]:
        return {"world_id": self.record["world_id"], "domain": self.record.get("domain"),
                **self.visible}
