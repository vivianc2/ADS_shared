#!/usr/bin/env python3
"""RPG v6 runtime simulator.

Holds one calibrated world (SCM + precomputed gold/battery) and serves the
agent's free-text requests through the resolver:

- ``measure(requests, intervention)``: resolve each request to a measurable
  variable, return noisy assay readings (optionally under an active
  intervention held in place for the reading).
- ``intervene(requests)``: resolve each proposed action to an actuator + value,
  apply jointly, and return the resulting readings for whatever the agent asked
  to observe.
- ``grade(answer)``: score against the computed gold + counterfactual battery.

Every resolution (request -> interpretation | rejection) is returned to the
agent and logged, so a resolution miss is always visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from engine import WorldSCM
from resolver import Resolver
from oracle_v6 import (calibrate, optimal_gold, counterfactual_battery,
                       grade as _grade)


class SimV6:
    def __init__(self, world: Dict[str, Any], *, resolver_llm: Any = None,
                 n_sample: int = 400, precomputed: Optional[Dict[str, Any]] = None,
                 data_dir: Optional[str] = None):
        self.world = world
        self.scm: WorldSCM = world["scm"]
        self.gt = world["ground_truth"]
        self.n_sample = n_sample
        self.resolver = Resolver(self.scm, llm=resolver_llm)
        if precomputed:
            self.gold = precomputed["gold"]
            self.battery = precomputed["battery"]
        else:
            calibrate(world)
            self.gold = optimal_gold(world)
            self.battery = counterfactual_battery(world)
        self._seed0 = 1000
        self._q = 0
        # where per-experiment raw rows (CSV) are written for the code tool
        self.data_dir = Path(data_dir) if data_dir else None
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        # experiment_id -> {"path", "columns", "mode", "intervention"}
        self.experiments: Dict[int, Dict[str, Any]] = {}

    def _write_rows(self, exp_id: int, obs: Dict[str, np.ndarray],
                    intervention: Dict[str, Any], mode: str) -> Optional[str]:
        """Persist raw per-unit rows to a CSV the code tool can load."""
        if not self.data_dir:
            return None
        import pandas as pd
        df = pd.DataFrame({k: np.asarray(v) for k, v in obs.items()})
        df.insert(0, "unit_id", range(len(df)))
        # record the applied intervention as constant columns (so code can group)
        for aid, val in (intervention or {}).items():
            df[f"do_{aid}"] = val
        path = str(self.data_dir / f"experiment_{exp_id}.csv")
        df.to_csv(path, index=False)
        self.experiments[exp_id] = {"path": path, "columns": list(df.columns),
                                    "mode": mode, "intervention": dict(intervention or {})}
        return path

    # ---- measurement ----
    def measure(self, requests: List[str], intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._q += 1
        resolutions = [self.resolver.resolve_measure(r) for r in requests]
        var_ids, echoes = [], []
        for req, res in zip(requests, resolutions):
            echoes.append({"request": req, **res.to_dict()})
            if res.ok and res.target_id:
                var_ids.append(res.target_id)
        readings = {}
        csv_path = None
        if var_ids:
            iv = self._resolve_intervention_dict(intervention)
            # observational reads (no intervention) reflect the SELECTED historical
            # record when the world has a selection spec; select=True is a no-op
            # otherwise and auto-disables under any intervention (controlled expt).
            vals = self.scm.sample(self.n_sample, intervention=iv, seed=self._seed0 + self._q, select=True)
            obs = self.scm.measure(vals, var_ids, seed=self._seed0 + 5000 + self._q, intervention=iv)
            readings = {k: {"mean": round(float(np.mean(v)), 3),
                            "sd": round(float(np.std(v)), 3),
                            "se": round(float(np.std(v) / np.sqrt(len(v))), 3)}
                        for k, v in obs.items()}
            # pairwise correlations among returned readings
            names = list(obs)
            corr = {}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = obs[names[i]], obs[names[j]]
                    c = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
                    corr[f"{names[i]}~{names[j]}"] = round(c, 3)
            readings["_correlations"] = corr
            csv_path = self._write_rows(self._q, obs, iv, "measure")
        return {"mode": "measure", "experiment_id": self._q, "n_units": self.n_sample,
                "resolutions": echoes, "readings": readings, "raw_csv": csv_path}

    # ---- intervention ----
    def intervene(self, actions: List[Dict[str, Any]], measure_requests: List[str]) -> Dict[str, Any]:
        """actions: list of {"request": str, "value": optional}. Applied jointly."""
        self._q += 1
        resolutions, iv = [], {}
        for a in actions:
            res = self.resolver.resolve_intervene(a.get("request", ""), a.get("value"))
            resolutions.append({"request": a.get("request", ""), **res.to_dict()})
            if res.ok and res.target_id:
                iv[res.target_id] = res.value
        # resolve measurement targets (default: outcome + true proxy names? no —
        # only what the agent asks; but always include the outcome so it sees effect)
        m_res = [self.resolver.resolve_measure(r) for r in measure_requests]
        m_ids = [r.target_id for r in m_res if r.ok and r.target_id]
        m_echo = [{"request": rq, **rr.to_dict()} for rq, rr in zip(measure_requests, m_res)]
        if self.scm.outcome not in m_ids:
            m_ids.append(self.scm.outcome)
        readings = {}
        csv_path = None
        if iv and m_ids:
            vals = self.scm.sample(self.n_sample, intervention=iv, seed=self._seed0 + self._q)
            obs = self.scm.measure(vals, m_ids, seed=self._seed0 + 5000 + self._q, intervention=iv)
            readings = {k: round(float(np.mean(v)), 3) for k, v in obs.items()}
            csv_path = self._write_rows(self._q, obs, iv, "intervene")
        return {"mode": "intervene", "experiment_id": self._q, "n_units": self.n_sample,
                "applied_intervention": iv, "action_resolutions": resolutions,
                "measurement_resolutions": m_echo, "readings": readings, "raw_csv": csv_path}

    def _resolve_intervention_dict(self, intervention: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """intervention may be given as {actuator_id: value} already, or None."""
        if not intervention:
            return {}
        # accept only known actuator ids (this path is used internally / by tests)
        return {k: v for k, v in intervention.items() if k in self.scm.actuators}

    # ---- grading ----
    def grade(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        return _grade(self.world, answer, self.gold, self.battery)

    # ---- agent-facing scenario ----
    def public(self) -> Dict[str, Any]:
        return {
            "world_id": self.world["world_id"],
            "domain": self.world["domain"],
            "scenario": self.world["scenario"],
            "outcome_name": self.scm.outcome,
            "outcome_direction": "higher_is_better" if self.scm.higher_is_better else "lower_is_better",
        }
