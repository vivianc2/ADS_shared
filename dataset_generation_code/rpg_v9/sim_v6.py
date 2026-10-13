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
                 data_dir: Optional[str] = None, display_units: bool = True):
        self.world = world
        self.scm: WorldSCM = world["scm"]
        self.gt = world["ground_truth"]
        self.n_sample = n_sample
        # v10 units (2026-10-12): controls carry {"unit","lo","hi"}; the SCM runs on 0-100 internally and everything
        # the agent sends or sees is in physical units (display = lo + internal/100*(hi-lo)). display_units=False keeps
        # the internal scale (scripted reference policies). Worlds without units (v9) are unaffected either way.
        self.display = bool(display_units) and any("units" in a for a in self.scm.actuators.values())
        self.resolver = Resolver(self.scm, llm=resolver_llm, display=self.display)
        if precomputed:
            self.gold = precomputed["gold"]
            self.battery = precomputed["battery"]
        else:
            calibrate(world)
            self.gold = optimal_gold(world)
            self.battery = counterfactual_battery(world)
        self.csv_columns: Dict[str, str] = {}      # canonical -> agent id for the code tool's CSVs (set by RPGEnv, v10)
        # v10 free-text harness: the agent names controls in its own words, so it must never see a canonical actuator
        # id or description -- they encode the role ("set_X" / "control for X" = added level control, "controller
        # for X" = the source lever). Controls are shown by their first alias; see agent_view(). RPGEnv overrides
        # csv_columns with its own id-space map.
        self.v10 = self.gt.get("_world_version") == "v10"
        self.act_label: Dict[str, str] = {aid: (act.get("aliases") or [aid])[0]
                                          for aid, act in self.scm.actuators.items()}
        if self.v10:
            self.csv_columns = {f"do_{aid}": "do_" + "_".join(lab.split())
                                for aid, lab in self.act_label.items()}
        self._seed0 = 1000
        self._q = 0
        # where per-experiment raw rows (CSV) are written for the code tool
        # absolute: the code sandbox runs inside this dir, so a relative path (e.g. --outdir out) would not resolve
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        # experiment_id -> {"path", "columns", "mode", "intervention"}
        self.experiments: Dict[int, Dict[str, Any]] = {}

    # ---- unit conversion (identity unless self.display) ----
    def act_to_internal(self, aid: str, v: Any) -> Any:
        u = self.scm.actuators.get(aid, {}).get("units") if self.display else None
        if u is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return round(min(100.0, max(0.0, (float(v) - u["lo"]) / (u["hi"] - u["lo"]) * 100.0)), 4)

    def act_to_display(self, aid: str, x: Any) -> Any:
        u = self.scm.actuators.get(aid, {}).get("units") if self.display else None
        if u is None or isinstance(x, bool) or not isinstance(x, (int, float)):
            return x
        return round(u["lo"] + float(x) / 100.0 * (u["hi"] - u["lo"]), 4)

    def var_to_internal(self, var: str, v: Any) -> Any:
        u = self.scm.variables.get(var, {}).get("units") if self.display else None
        if u is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return (float(v) - u["lo"]) / (u["hi"] - u["lo"]) * 100.0

    def _obs_display(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.display:
            return obs
        out = {}
        for k, v in obs.items():
            u = self.scm.variables.get(k, {}).get("units")
            out[k] = (u["lo"] + np.asarray(v, dtype=float) / 100.0 * (u["hi"] - u["lo"])) if u else v
        return out

    def answer_to_internal(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """A graded answer whose values are in display units -> internal (recommended doses and policy doses /
        threshold). Identity unless self.display."""
        if not self.display:
            return answer
        a = dict(answer)
        a["recommended_intervention"] = {k: self.act_to_internal(k, v)
                                         for k, v in (answer.get("recommended_intervention") or {}).items()}
        pol = answer.get("recommended_policy")
        if isinstance(pol, dict) and pol.get("treatment") in self.scm.actuators:
            p = dict(pol)
            for key in ("dose_if_ge", "dose_if_lt"):
                if key in p:
                    p[key] = self.act_to_internal(pol["treatment"], p[key])
            if "threshold" in p and pol.get("stratifier") in self.scm.variables:
                p["threshold"] = self.var_to_internal(pol["stratifier"], p["threshold"])
            a["recommended_policy"] = p
        return a

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
        if self.csv_columns:
            # v10: the agent works in id space; canonical names would leak structure (e.g. "do_set_X" for added
            # level controls vs a plain id for the source lever -- the v9 label tell, via the code tool)
            df = df.rename(columns=self.csv_columns)
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
            obs = self._obs_display(self.scm.measure(vals, var_ids, seed=self._seed0 + 5000 + self._q, intervention=iv))
            readings = {k: {"mean": round(float(np.mean(v)), 3),
                            "sd": round(float(np.std(v)), 3),
                            "se": round(float(np.std(v) / np.sqrt(len(v))), 3)}
                        for k, v in obs.items()}
            # pairwise correlations among returned readings
            names = list(obs)
            corr = {}
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            if self.gt.get("_world_version") == "v10":
                # v10 worlds have ~20-30 signals: all pairs would be 200-400 lines per measurement. Report each
                # signal's correlation with the OUTCOME (or all pairs when <= 8 signals and no outcome); the full
                # per-unit data stays available to the code tool (experiment_<n>.csv).
                if self.scm.outcome in names:
                    k = names.index(self.scm.outcome)
                    pairs = [(i, j) for i, j in pairs if k in (i, j)]
                elif len(names) > 8:
                    pairs = []
            for i, j in pairs:
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
                iv[res.target_id] = res.value          # display units when self.display
        # resolve measurement targets (default: outcome + true proxy names? no —
        # only what the agent asks; but always include the outcome so it sees effect)
        m_res = [self.resolver.resolve_measure(r) for r in measure_requests]
        m_ids = [r.target_id for r in m_res if r.ok and r.target_id]
        m_echo = [{"request": rq, **rr.to_dict()} for rq, rr in zip(measure_requests, m_res)]
        if self.scm.outcome not in m_ids:
            m_ids.append(self.scm.outcome)
        readings = {}
        csv_path = None
        iv_int = {k: self.act_to_internal(k, v) for k, v in iv.items()}
        if iv and m_ids:
            vals = self.scm.sample(self.n_sample, intervention=iv_int, seed=self._seed0 + self._q)
            obs = self._obs_display(self.scm.measure(vals, m_ids, seed=self._seed0 + 5000 + self._q, intervention=iv_int))
            readings = {k: round(float(np.mean(v)), 3) for k, v in obs.items()}
            csv_path = self._write_rows(self._q, obs, iv, "intervene")
        out = {"mode": "intervene", "experiment_id": self._q, "n_units": self.n_sample,
                "applied_intervention": iv, "action_resolutions": resolutions,
                "measurement_resolutions": m_echo, "readings": readings, "raw_csv": csv_path}
        if self.display:
            out["applied_internal"] = iv_int
        return out

    def _resolve_intervention_dict(self, intervention: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """intervention may be given as {actuator_id: value} already, or None."""
        if not intervention:
            return {}
        # accept only known actuator ids (this path is used internally / by tests)
        return {k: v for k, v in intervention.items() if k in self.scm.actuators}

    # ---- what the free-text agent is shown ----
    _REJECT_V10 = "Not available on this system: no assay or control matches that request."

    def agent_view(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """The measure/intervene result as shown to a free-text agent. v9: unchanged. v10: controls appear only by
        their first alias (never the canonical id or description), every rejection has one wording (v9's "no assay"
        vs "not part of this system" tells the agent which hidden states exist, e.g. the true root), and the CSV is
        named by its code-tool variable, not its path (the path contains the world id, i.e. the archetype)."""
        if not self.v10:
            return result
        lab = lambda aid: self.act_label.get(aid, aid)

        def echo(e: Dict[str, Any]) -> Dict[str, Any]:
            out = {"request": e.get("request", "")}
            if not e.get("ok"):
                if e.get("candidates"):
                    names = [lab(c) if c in self.scm.actuators else c for c in e["candidates"]]
                    out.update(ok=False, interpretation="ambiguous request; did you mean one of: " + ", ".join(names))
                else:
                    out.update(ok=False, interpretation=self._REJECT_V10)
            elif e.get("kind") == "intervene":
                out.update(ok=True, interpretation=f"apply '{lab(e['target_id'])}' at {e.get('value')}")
            else:
                out.update(ok=True, interpretation=f"measure '{e['target_id']}'")
            return out

        v = {k: result[k] for k in ("mode", "experiment_id", "n_units", "readings") if k in result}
        for key in ("resolutions", "action_resolutions", "measurement_resolutions"):
            if key in result:
                v[key] = [echo(e) for e in result[key]]
        if "applied_intervention" in result:
            v["applied_intervention"] = {lab(a): val for a, val in result["applied_intervention"].items()}
        if result.get("raw_csv"):
            v["raw_csv"] = f"experiment_{result['experiment_id']}_csv"
        return v

    # ---- grading ----
    def grade(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """answer values are in display units when self.display (free-text harness)."""
        return _grade(self.world, self.answer_to_internal(answer), self.gold, self.battery)

    # ---- agent-facing scenario ----
    def public(self) -> Dict[str, Any]:
        return {
            "world_id": self.world["world_id"],
            "domain": self.world["domain"],
            "scenario": self.world["scenario"],
            "outcome_name": self.scm.outcome,
            "outcome_direction": "higher_is_better" if self.scm.higher_is_better else "lower_is_better",
        }
