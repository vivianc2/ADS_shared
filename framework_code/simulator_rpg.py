#!/usr/bin/env python3
"""Runtime simulator for ACED RPG worlds.

RPG worlds are dynamic simulator worlds, not Bayesian networks.  This module
loads a ``schema_version: rpg_v1`` JSON file and executes typed trajectory
queries against ``hidden.simulator_config`` while exposing only the public
measurement/action catalog to callers.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

from dataset_generation_code import world_gen_rpg as rpg  # noqa: E402
from schemas_rpg import RPGParsedQuery as RPGQuery  # noqa: E402
from schemas_rpg import RPGQueryResult  # noqa: E402
from schemas_rpg import StaticRPGParsedQuery, StaticRPGQueryResult  # noqa: E402


RPG_QUERY_MODES = {
    "observational_trajectory",
    "policy_rollout",
    "policy_comparison",
}

STATIC_RPG_QUERY_MODES = {
    "observational_sample",
    "interventional_sample",
    "inspect_unit",
}

class RPGSimulator:
    """Execute policy rollout experiments for one RPG world."""

    def __init__(self, world: Dict[str, Any], world_path: Optional[str] = None):
        if world.get("schema_version") != rpg.SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {rpg.SCHEMA_VERSION!r}")
        self.world = world
        self.world_path = world_path
        self.meta = world["meta"]
        self.visible = world["visible"]
        self.hidden = world["hidden"]
        self.cfg = world["hidden"]["simulator_config"]
        self.allowed_measurements = list(self.visible["allowed_measurements"])
        self.allowed_measurement_set = set(self.allowed_measurements)
        self.allowed_policies = list(self.visible["allowed_policies"])
        self.policy_by_id = {p["policy_id"]: p for p in self.cfg["policies"]}
        self.default_horizon = int(self.visible.get("default_horizon", self.cfg["horizon"]))
        self.default_policy_id = (
            self.visible.get("default_observational_policy_id")
            or self.cfg.get("default_policy_id")
            or self.allowed_policies[0]["policy_id"]
        )

    @classmethod
    def from_json(cls, path: str) -> "RPGSimulator":
        with open(path, "r", encoding="utf-8") as f:
            world = json.load(f)
        return cls(world, world_path=path)

    def public_world(self) -> Dict[str, Any]:
        """Return agent-facing world information with hidden/oracle fields removed."""
        return {
            "schema_version": self.world["schema_version"],
            "world_id": self.meta["world_id"],
            "story": self.visible["story"],
            "observed_variables": copy.deepcopy(self.visible["observed_variables"]),
            "action_variables": copy.deepcopy(self.visible["action_variables"]),
            "allowed_policies": copy.deepcopy(self.visible["allowed_policies"]),
            "allowed_measurements": list(self.allowed_measurements),
            "allowed_query_modes": list(self.visible["allowed_query_modes"]),
            "discovery_protocol": copy.deepcopy(self.visible.get("discovery_protocol", {})),
            "experiment_budget": copy.deepcopy(self.visible.get("experiment_budget", {})),
            "default_observational_policy_id": self.default_policy_id,
            "default_horizon": self.default_horizon,
            "sample_unit": self.visible.get("sample_unit", "unit_period_row"),
            "question": self.visible["question"],
        }

    def validate_query(self, query: RPGQuery) -> None:
        if query.mode not in RPG_QUERY_MODES:
            raise ValueError(f"unknown RPG query mode {query.mode!r}")
        if not isinstance(query.n_units, int) or query.n_units <= 0:
            raise ValueError("n_units must be a positive integer")
        horizon = query.horizon if query.horizon is not None else self.default_horizon
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if horizon > self.default_horizon:
            raise ValueError(
                f"horizon {horizon} exceeds world default horizon {self.default_horizon}; "
                "RPG v1 only exposes validated horizons up to the world horizon"
            )
        measurements = query.measurements or self.allowed_measurements
        unknown_measurements = sorted(set(measurements) - self.allowed_measurement_set)
        if unknown_measurements:
            raise ValueError(f"unknown or hidden measurements requested: {unknown_measurements}")

        if query.mode == "observational_trajectory":
            if query.policy_ids:
                raise ValueError("observational_trajectory does not accept explicit policy_ids")
            return

        if query.mode == "policy_rollout" and len(query.policy_ids) != 1:
            raise ValueError("policy_rollout requires exactly one policy_id")
        if query.mode == "policy_comparison" and len(query.policy_ids) < 2:
            raise ValueError("policy_comparison requires at least two policy_ids")
        if len(set(query.policy_ids)) != len(query.policy_ids):
            raise ValueError("policy_ids must be distinct")

        unknown_policies = sorted(set(query.policy_ids) - set(self.policy_by_id))
        if unknown_policies:
            raise ValueError(f"unknown policy_ids: {unknown_policies}")

    def estimate_sample_usage(self, query: RPGQuery, *, accounting: str = "unit_period_rows") -> Dict[str, int]:
        self.validate_query(query)
        horizon = query.horizon or self.default_horizon
        n_policies = 1 if query.mode == "observational_trajectory" else len(query.policy_ids)
        rows = query.n_units * horizon * n_policies
        measurements = query.measurements or self.allowed_measurements
        # condition_id, unit_id, time, policy_id, public actions, measurements.
        columns = 4 + len(self.visible.get("action_variables", [])) + len(measurements)
        return {
            "unit_period_rows": rows,
            "cells": rows * columns,
            "n_units": query.n_units,
            "horizon": horizon,
            "n_policies": n_policies,
        }

    def run_query(self, query: RPGQuery) -> RPGQueryResult:
        try:
            self.validate_query(query)
            df = self._execute_query(query)
            rows = int(len(df))
            cells = int(rows * len(df.columns))
            return RPGQueryResult(
                success=True,
                query=query,
                dataframe=df,
                sample_rows=rows,
                sample_cells=cells,
                sample_units=rows,
                columns=list(df.columns),
            )
        except Exception as exc:
            return RPGQueryResult(success=False, query=query, error_message=str(exc))

    def run_query_to_csv(self, query: RPGQuery, output_dir: str, prefix: str = "rpg_query") -> RPGQueryResult:
        result = self.run_query(query)
        if not result.success or result.dataframe is None:
            return result
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        existing = len(list(out.glob(f"{prefix}_*.csv")))
        path = out / f"{prefix}_{existing + 1:04d}.csv"
        result.dataframe.to_csv(path, index=False)
        result.data_file = str(path)
        return result

    def rollout(
        self,
        policy_id: str,
        *,
        n_units: int,
        measurements: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        query = RPGQuery(
            mode="policy_rollout",
            policy_ids=[policy_id],
            n_units=n_units,
            measurements=list(measurements) if measurements is not None else None,
            horizon=horizon,
            seed=seed,
        )
        result = self.run_query(query)
        if not result.success or result.dataframe is None:
            raise ValueError(result.error_message)
        return result.dataframe

    def compare_policies(
        self,
        policy_ids: List[str],
        *,
        n_units: int,
        measurements: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        query = RPGQuery(
            mode="policy_comparison",
            policy_ids=policy_ids,
            n_units=n_units,
            measurements=list(measurements) if measurements is not None else None,
            horizon=horizon,
            seed=seed,
        )
        result = self.run_query(query)
        if not result.success or result.dataframe is None:
            raise ValueError(result.error_message)
        return result.dataframe

    def observational_trajectory(
        self,
        *,
        n_units: int,
        measurements: Optional[Iterable[str]] = None,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        query = RPGQuery(
            mode="observational_trajectory",
            n_units=n_units,
            measurements=list(measurements) if measurements is not None else None,
            horizon=horizon,
            seed=seed,
        )
        result = self.run_query(query)
        if not result.success or result.dataframe is None:
            raise ValueError(result.error_message)
        return result.dataframe

    def _execute_query(self, query: RPGQuery) -> pd.DataFrame:
        horizon = query.horizon or self.default_horizon
        measurements = query.measurements or self.allowed_measurements
        policy_ids = [self.default_policy_id] if query.mode == "observational_trajectory" else query.policy_ids
        base_seed = query.seed if query.seed is not None else int(self.meta.get("seed", 0)) + 500000

        frames = []
        for condition_idx, policy_id in enumerate(policy_ids):
            cfg = copy.deepcopy(self.cfg)
            cfg["horizon"] = horizon
            policy = self.policy_by_id[policy_id]
            rollout = rpg.rollout(
                cfg,
                policy,
                query.n_units,
                base_seed + condition_idx * 104729,
                return_rows=True,
                measurements=measurements,
            )
            df = pd.DataFrame(rollout["rows"])
            df.insert(0, "condition_id", condition_idx)
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


class StaticRPGSimulator:
    """Execute static partially observed RPG sampling queries."""

    def __init__(self, world: Dict[str, Any], world_path: Optional[str] = None):
        allowed_static_versions = {rpg.SCHEMA_VERSION_STATIC, "rpg_static_v2"}
        if world.get("schema_version") not in allowed_static_versions:
            raise ValueError(f"expected schema_version in {sorted(allowed_static_versions)!r}")
        self.world = world
        self.world_path = world_path
        self.meta = world["meta"]
        self.visible = world["visible"]
        self.hidden = world["hidden"]
        self.cfg = copy.deepcopy(world["hidden"]["simulator_config"])
        self.cfg.setdefault("seed", int(self.meta.get("seed", self.cfg.get("world_seed", 0))))
        self.allowed_measurements = list(self.visible["allowed_measurements"])
        self.allowed_measurement_set = set(self.allowed_measurements)
        self.intervenable_variables = list(self.visible["intervenable_variables"])
        self.knob_specs = {v["name"]: v for v in self.intervenable_variables}
        self.max_intervention_knobs = int(self.visible.get("max_intervention_knobs", len(self.knob_specs)))
        self._seen_case_seeds: set[int] = set()

    @classmethod
    def from_json(cls, path: str) -> "StaticRPGSimulator":
        with open(path, "r", encoding="utf-8") as f:
            world = json.load(f)
        return cls(world, world_path=path)

    @staticmethod
    def _has_text(value: Any) -> bool:
        return value is not None and bool(str(value).strip())

    def public_world(self) -> Dict[str, Any]:
        """Return only agent-facing static RPG fields."""
        return {
            "schema_version": self.world["schema_version"],
            "world_id": self.meta["world_id"],
            "story": self.visible["story"],
            "observed_variables": copy.deepcopy(self.visible["observed_variables"]),
            "intervenable_variables": copy.deepcopy(self.visible["intervenable_variables"]),
            "allowed_measurements": list(self.allowed_measurements),
            "allowed_query_modes": list(self.visible["allowed_query_modes"]),
            "discovery_protocol": copy.deepcopy(self.visible.get("discovery_protocol", {})),
            "experiment_budget": copy.deepcopy(self.visible.get("experiment_budget", {})),
            "question": self.visible["question"],
            "answer_schema": self.visible["answer_schema"],
            "max_intervention_knobs": self.max_intervention_knobs,
        }

    def validate_measurements(self, measurements: Optional[List[str]]) -> List[str]:
        if measurements is None:
            measurements = list(self.allowed_measurements)
        unknown = sorted(set(measurements) - self.allowed_measurement_set)
        if unknown:
            raise ValueError(f"unknown or hidden measurements requested: {unknown}")
        return list(measurements)

    def validate_intervention(
        self,
        intervention: Optional[Dict[str, Any]],
        *,
        enforce_answer_knob_cap: bool = False,
    ) -> Dict[str, Any]:
        intervention = dict(intervention or {})
        unknown = sorted(set(intervention) - set(self.knob_specs))
        if unknown:
            raise ValueError(f"unknown or non-intervenable knobs: {unknown}")
        if enforce_answer_knob_cap and len(intervention) > self.max_intervention_knobs:
            raise ValueError(
                f"intervention sets {len(intervention)} knobs, above max_intervention_knobs="
                f"{self.max_intervention_knobs}"
            )
        for name, value in intervention.items():
            spec = self.knob_specs[name]
            if str(spec.get("value_type", "")).lower() == "continuous":
                lo = float(spec.get("min", 0.0))
                hi = float(spec.get("max", 100.0))
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"invalid value for {name}: {value!r}; expected a number in [{lo}, {hi}]"
                    )
                if not (lo <= fv <= hi):
                    raise ValueError(
                        f"invalid value for {name}: {value!r}; must be within [{lo}, {hi}]"
                    )
                intervention[name] = fv
            else:
                allowed = [str(v) for v in spec.get("values", [])]
                if str(value) not in allowed:
                    raise ValueError(f"invalid value for {name}: {value!r}; allowed values are {allowed}")
                intervention[name] = str(value)
        return intervention

    def validate_query(self, query: StaticRPGParsedQuery) -> None:
        if query.mode not in STATIC_RPG_QUERY_MODES:
            raise ValueError(f"unknown static RPG query mode {query.mode!r}")
        self.validate_measurements(query.measurements)
        if query.mode == "interventional_sample":
            self.validate_intervention(query.intervention, enforce_answer_knob_cap=True)
            if not isinstance(query.n_units, int) or query.n_units <= 0:
                raise ValueError("n_units must be a positive integer")
        elif query.mode == "observational_sample":
            if query.intervention:
                raise ValueError("observational_sample does not accept an intervention")
            if not isinstance(query.n_units, int) or query.n_units <= 0:
                raise ValueError("n_units must be a positive integer")
        elif query.mode == "inspect_unit":
            if query.intervention:
                raise ValueError("inspect_unit uses observational assignment and does not accept intervention")
            if query.case_seed is None:
                raise ValueError("inspect_unit requires case_seed")

    def estimate_sample_usage(self, query: StaticRPGParsedQuery) -> Dict[str, int]:
        self.validate_query(query)
        measurements = self.validate_measurements(query.measurements)
        rows = 1 if query.mode == "inspect_unit" else int(query.n_units)
        # unit_id + query_mode + requested measurements + do columns.
        columns = 2 + len(measurements) + len(query.intervention or {})
        if query.mode == "inspect_unit":
            columns += 1  # case_seed
        return {
            "rows": rows,
            "cells": rows * columns,
            "n_units": rows,
            "n_measurements": len(measurements),
            "n_intervention_knobs": len(query.intervention or {}),
        }

    def run_query(self, query: StaticRPGParsedQuery) -> StaticRPGQueryResult:
        try:
            self.validate_query(query)
            df = self._execute_query(query)
            rows = int(len(df))
            cells = int(rows * len(df.columns))
            return StaticRPGQueryResult(
                success=True,
                query=query,
                dataframe=df,
                n_rows=rows,
                columns=list(df.columns),
                preview=df.head(10).to_csv(index=False),
                sample_rows=rows,
                sample_cells=cells,
                sample_units=cells,
            )
        except Exception as exc:
            return StaticRPGQueryResult(success=False, query=query, error_message=str(exc))

    def score_answer(self, answer: Any, *, n_score: Optional[int] = None) -> Dict[str, Any]:
        """Score an answer against the world's declared static-RPG answer schema."""
        schema = str(
            self.visible.get("answer_schema")
            or self.world.get("oracle", {}).get("gold_answer", {}).get("answer_schema")
            or "intervention_with_hypothesis"
        )
        if schema == "conditional_policy":
            return self._score_conditional_policy(answer, n_score=n_score)
        if schema == "latent_regime_policy":
            return self._score_latent_regime_policy(answer, n_score=n_score)
        if schema == "latent_cause_hypothesis":
            return self._score_latent_cause_hypothesis(answer)
        if schema == "anomaly_identification":
            return self._score_anomaly_identification(answer)
        return self._score_intervention_answer(answer, n_score=n_score)

    def _flatten_answer_text(self, value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(f"{k} {self._flatten_answer_text(v)}" for k, v in value.items())
        if isinstance(value, list):
            return " ".join(self._flatten_answer_text(v) for v in value)
        return "" if value is None else str(value)

    def _score_latent_cause_hypothesis(self, answer: Any) -> Dict[str, Any]:
        parsed = self._parse_json_answer(answer)
        oracle = self.world["oracle"]["gold_answer"]
        latent = parsed.get("latent_hypothesis") or {}
        action_plan = parsed.get("action_plan") or {}
        evidence = parsed.get("evidence") or []
        alternatives = parsed.get("alternatives_ruled_out") or []
        decisive_test = parsed.get("decisive_test") or ""

        if not isinstance(latent, dict):
            raise ValueError("latent_cause_hypothesis answer requires a latent_hypothesis object")
        if not isinstance(action_plan, dict):
            raise ValueError("latent_cause_hypothesis answer requires an action_plan object")
        if not isinstance(evidence, list):
            evidence = [evidence]
        if not isinstance(alternatives, list):
            alternatives = [alternatives]

        latent_text = self._flatten_answer_text(latent).lower()
        all_text = self._flatten_answer_text(parsed).lower()
        evidence_text = self._flatten_answer_text(evidence).lower()
        alternative_text = self._flatten_answer_text(alternatives).lower()
        test_text = self._flatten_answer_text(decisive_test).lower()
        action_text = self._flatten_answer_text(action_plan).lower()

        aliases = [
            str(a).lower()
            for a in (oracle.get("latent_hypothesis") or {}).get("accepted_aliases", [])
        ]
        scoring_terms = oracle.get("scoring_terms") or {}
        cause_match = any(alias in all_text for alias in aliases)
        # Guard against surface-correlation answers that mention only the
        # visible trigger/context without identifying the hidden mechanism.
        mechanism_terms = tuple(
            str(t).lower()
            for t in scoring_terms.get(
                "mechanism_terms",
                ("gutter", "downspout", "roof drainage", "roof runoff", "drainage path"),
            )
        )
        obstruction_terms = tuple(
            str(t).lower()
            for t in scoring_terms.get(
                "hidden_state_terms",
                ("clog", "block", "obstruct", "blocked", "obstruction", "backflow", "overflow"),
            )
        )
        mechanism_match = any(t in all_text for t in mechanism_terms) and any(t in all_text for t in obstruction_terms)

        evidence_families = scoring_terms.get("evidence_groups") or {
            "leaf": ("leaf", "leaves", "leaf-fall", "leaf fall"),
            "rain": ("rain", "rainfall", "storm"),
            "downspout": ("downspout", "discharge", "delay"),
            "roof": ("roof", "overflow", "backflow", "flow test", "water-flow"),
            "flood": ("flood", "pool", "pooling", "yard", "patio"),
            "action_verification": ("clear", "flush", "flow test", "clearing", "flushing"),
        }
        evidence_hits = {
            name: any(str(term).lower() in evidence_text or str(term).lower() in test_text or str(term).lower() in action_text for term in terms)
            for name, terms in evidence_families.items()
        }
        evidence_count = sum(1 for ok in evidence_hits.values() if ok)

        alternative_families = scoring_terms.get("alternative_groups") or {
            "soil": ("soil", "compaction", "aerat"),
            "slope": ("slope", "grade", "regrade"),
            "neighbor": ("neighbor", "driveway"),
            "rain_only": ("rain alone", "rainfall alone", "rain only"),
        }
        alternative_hits = {
            name: any(str(term).lower() in alternative_text for term in terms)
            for name, terms in alternative_families.items()
        }
        alternatives_count = sum(1 for ok in alternative_hits.values() if ok)

        verification_terms = tuple(
            str(t).lower()
            for t in scoring_terms.get(
                "verification_terms",
                ("flow test", "water-flow", "clear", "flush", "clearing", "flushing", "backflow"),
            )
        )
        verification_match = any(
            term in test_text or term in evidence_text or term in action_text
            for term in verification_terms
        )
        action_terms = tuple(
            str(t).lower()
            for t in scoring_terms.get(
                "action_terms",
                ("cleargutters", "clearreargutters", "clear rear gutters", "clear gutters", "flushdownspout", "flush downspout"),
            )
        )
        action_match = any(
            term in action_text
            for term in action_terms
        ) or (
            ("gutter" in action_text or "downspout" in action_text)
            and ("clear" in action_text or "flush" in action_text)
        )

        accepted = (
            bool(cause_match or mechanism_match)
            and evidence_count >= 3
            and alternatives_count >= 1
            and verification_match
            and action_match
        )
        return {
            "success": True,
            "accepted": bool(accepted),
            "answer_schema": "latent_cause_hypothesis",
            "answer": parsed,
            "cause_match": bool(cause_match or mechanism_match),
            "semantic_alias_match": bool(cause_match),
            "mechanism_match": bool(mechanism_match),
            "evidence_count": evidence_count,
            "evidence_hits": evidence_hits,
            "alternatives_count": alternatives_count,
            "alternative_hits": alternative_hits,
            "verification_match": bool(verification_match),
            "action_match": bool(action_match),
            "gold_latent_hypothesis": oracle.get("latent_hypothesis"),
            "gold_action_plan": oracle.get("action_plan"),
            "trajectory_requirements": oracle.get("trajectory_requirements"),
            "hypothesis_present": self._has_text(latent_text),
        }

    def _score_intervention_answer(self, answer: Any, *, n_score: Optional[int] = None) -> Dict[str, Any]:
        parsed = self._parse_intervention_answer(answer)
        intervention = self.validate_intervention(
            parsed.get("intervention") or {},
            enforce_answer_knob_cap=True,
        )
        intervention = self._canonicalize_default_intervention(intervention)
        key = rpg._static_intervention_key(intervention)
        oracle = self.world["oracle"]
        gold_utility = float(oracle["gold_answer"]["expected_utility"])
        tolerance = float(oracle.get("oracle_tolerance", 0.0))

        stored_scores = {
            s["intervention_key"]: float(s["expected_utility"])
            for s in oracle.get("action_scores", [])
        }
        if key in stored_scores:
            expected_utility = stored_scores[key]
            score_source = "stored_oracle_action_scores"
        else:
            n = int(n_score or oracle.get("oracle_n_units", 50000))
            hidden = rpg._static_sample_hidden(self.cfg, n, seed=int(self.meta["seed"]) + 910001)
            outcomes = rpg._static_apply(self.cfg, hidden, intervention, seed=int(self.meta["seed"]) + 920001)
            utility = rpg._static_utility_from_outcomes(self.cfg, outcomes)
            expected_utility = float(np.mean(utility))
            score_source = f"fresh_monte_carlo_n={n}"

        accepted = expected_utility >= gold_utility - tolerance
        return {
            "success": True,
            "accepted": bool(accepted),
            "answer": parsed,
            "intervention": intervention,
            "intervention_key": key,
            "expected_utility": expected_utility,
            "gold_intervention": oracle["gold_answer"]["intervention"],
            "gold_intervention_key": oracle["gold_answer"]["intervention_key"],
            "gold_expected_utility": gold_utility,
            "oracle_tolerance": tolerance,
            "utility_gap_from_gold": gold_utility - expected_utility,
            "score_source": score_source,
            "hypothesis_present": self._has_text(parsed.get("hypothesis")),
        }

    def _score_conditional_policy(self, answer: Any, *, n_score: Optional[int] = None) -> Dict[str, Any]:
        parsed = self._parse_json_answer(answer)
        policy = parsed.get("policy") or {}
        if not isinstance(policy, dict):
            raise ValueError("conditional_policy answer requires a 'policy' object")
        branch_var = str(policy.get("branch_variable", ""))
        if branch_var not in self.allowed_measurement_set:
            raise ValueError(f"branch_variable {branch_var!r} not in allowed_measurements")
        if "branch_threshold" not in policy:
            raise ValueError("conditional_policy.policy requires branch_threshold")
        branch_threshold = float(policy["branch_threshold"])
        if_above = self.validate_intervention(
            policy.get("if_above") or {},
            enforce_answer_knob_cap=True,
        )
        if_above = self._canonicalize_default_intervention(if_above)
        if_below = self.validate_intervention(
            policy.get("if_below") or {},
            enforce_answer_knob_cap=True,
        )
        if_below = self._canonicalize_default_intervention(if_below)

        oracle = self.world["oracle"]
        gold = oracle["gold_answer"]
        n = int(n_score or oracle.get("oracle_n_units", 20000))
        seed0 = int(self.meta["seed"])
        hidden = rpg._static_sample_hidden(self.cfg, n, seed=seed0 + 930001)
        baseline_outcomes = rpg._static_apply(self.cfg, hidden, {}, seed=seed0 + 940001)
        proxy_obs = rpg._static_observe(
            self.cfg, hidden, baseline_outcomes, [branch_var], seed=seed0 + 950001
        )[branch_var]

        utility = np.zeros_like(proxy_obs, dtype=float)
        for offset, (mask, intervention) in enumerate(
            ((proxy_obs >= branch_threshold, if_above), (proxy_obs < branch_threshold, if_below))
        ):
            if int(mask.sum()) == 0:
                continue
            sub_hidden = {
                k: v[mask] if isinstance(v, np.ndarray) else v
                for k, v in hidden.items()
            }
            sub_out = rpg._static_apply(
                self.cfg,
                sub_hidden,
                dict(intervention),
                seed=seed0 + 960001 + offset * 1009,
            )
            utility[mask] = rpg._static_utility_from_outcomes(self.cfg, sub_out)

        expected_utility = float(np.mean(utility))
        gold_utility = float(gold["expected_utility"])
        tolerance = float(oracle.get("oracle_tolerance", 0.0))
        accepted = expected_utility >= gold_utility - tolerance
        normalized_policy = {
            "branch_variable": branch_var,
            "branch_threshold": branch_threshold,
            "if_above": if_above,
            "if_below": if_below,
        }
        return {
            "success": True,
            "accepted": bool(accepted),
            "answer_schema": "conditional_policy",
            "answer": parsed,
            "policy": normalized_policy,
            "expected_utility": expected_utility,
            "gold_policy": gold.get("policy"),
            "gold_expected_utility": gold_utility,
            "oracle_tolerance": tolerance,
            "utility_gap_from_gold": gold_utility - expected_utility,
            "score_source": f"fresh_monte_carlo_n={n}",
            "hypothesis_present": self._has_text(parsed.get("hypothesis")),
        }

    def _score_latent_regime_policy(self, answer: Any, *, n_score: Optional[int] = None) -> Dict[str, Any]:
        parsed = self._parse_json_answer(answer)
        latent = parsed.get("latent_structure") or {}
        if not isinstance(latent, dict):
            raise ValueError("latent_regime_policy answer requires a 'latent_structure' object")
        try:
            n_regimes = int(latent.get("n_regimes"))
        except Exception as exc:
            raise ValueError("latent_structure.n_regimes must be an integer") from exc
        policy = parsed.get("policy") or {}
        if not isinstance(policy, dict):
            raise ValueError("latent_regime_policy answer requires a 'policy' object")
        branch_var = str(policy.get("branch_variable", ""))
        if branch_var not in self.allowed_measurement_set:
            raise ValueError(f"branch_variable {branch_var!r} not in allowed_measurements")
        if "branch_threshold" not in policy:
            raise ValueError("latent_regime_policy.policy requires branch_threshold")
        branch_threshold = float(policy["branch_threshold"])
        if_above = self.validate_intervention(
            policy.get("if_above") or {},
            enforce_answer_knob_cap=True,
        )
        if_above = self._canonicalize_default_intervention(if_above)
        if_below = self.validate_intervention(
            policy.get("if_below") or {},
            enforce_answer_knob_cap=True,
        )
        if_below = self._canonicalize_default_intervention(if_below)

        oracle = self.world["oracle"]
        gold = oracle["gold_answer"]
        n = int(n_score or oracle.get("oracle_n_units", 20000))
        seed0 = int(self.meta["seed"])
        hidden = rpg._static_sample_hidden(self.cfg, n, seed=seed0 + 1030001)
        baseline_outcomes = rpg._static_apply(self.cfg, hidden, {}, seed=seed0 + 1040001)
        proxy_obs = rpg._static_observe(
            self.cfg, hidden, baseline_outcomes, [branch_var], seed=seed0 + 1050001
        )[branch_var]

        utility = np.zeros_like(proxy_obs, dtype=float)
        for offset, (mask, intervention) in enumerate(
            ((proxy_obs >= branch_threshold, if_above), (proxy_obs < branch_threshold, if_below))
        ):
            if int(mask.sum()) == 0:
                continue
            sub_hidden = {
                k: v[mask] if isinstance(v, np.ndarray) else v
                for k, v in hidden.items()
            }
            sub_out = rpg._static_apply(
                self.cfg,
                sub_hidden,
                dict(intervention),
                seed=seed0 + 1060001 + offset * 1009,
            )
            utility[mask] = rpg._static_utility_from_outcomes(self.cfg, sub_out)

        expected_utility = float(np.mean(utility))
        gold_utility = float(gold["expected_utility"])
        tolerance = float(oracle.get("oracle_tolerance", 0.0))
        regime_ok = n_regimes == int((gold.get("latent_structure") or {}).get("n_regimes", 2))
        utility_ok = expected_utility >= gold_utility - tolerance
        normalized_policy = {
            "branch_variable": branch_var,
            "branch_threshold": branch_threshold,
            "if_above": if_above,
            "if_below": if_below,
        }
        return {
            "success": True,
            "accepted": bool(regime_ok and utility_ok),
            "answer_schema": "latent_regime_policy",
            "answer": parsed,
            "n_regimes": n_regimes,
            "regime_count_correct": bool(regime_ok),
            "policy": normalized_policy,
            "expected_utility": expected_utility,
            "gold_latent_structure": gold.get("latent_structure"),
            "gold_policy": gold.get("policy"),
            "gold_expected_utility": gold_utility,
            "oracle_tolerance": tolerance,
            "utility_gap_from_gold": gold_utility - expected_utility,
            "score_source": f"fresh_monte_carlo_n={n}",
            "evidence_present": self._has_text(latent.get("evidence")),
            "hypothesis_present": self._has_text(parsed.get("hypothesis")),
        }

    def _score_anomaly_identification(self, answer: Any) -> Dict[str, Any]:
        parsed = self._parse_anomaly_answer(answer)
        rule_text = str(parsed.get("anomaly_rule") or "").strip()
        oracle = self.world["oracle"]
        thresholds = oracle.get("gold_answer", {}).get(
            "precision_recall_threshold",
            {"precision": 0.7, "recall": 0.6},
        )
        n_audit = 2000
        seed0 = int(self.meta["seed"])
        hidden = rpg._static_sample_hidden(self.cfg, n_audit, seed=seed0 + 980001)
        outcomes = rpg._static_apply(self.cfg, hidden, {}, seed=seed0 + 990001)
        obs = rpg._static_observe(
            self.cfg,
            hidden,
            outcomes,
            list(self.allowed_measurements),
            seed=seed0 + 990111,
        )
        truth = hidden["IsAnomaly"].astype(bool)
        try:
            flag_mask = self._eval_simple_rule(rule_text, obs)
        except Exception as exc:
            return {
                "success": True,
                "accepted": False,
                "answer_schema": "anomaly_identification",
                "answer": parsed,
                "error": f"rule parse failed: {exc}",
                "rule_text": rule_text,
                "hypothesis_present": self._has_text(parsed.get("hypothesis")),
            }

        n_flagged = int(flag_mask.sum())
        n_anomalies = int(truth.sum())
        true_positive = int(np.sum(flag_mask & truth))
        precision = float(true_positive / max(n_flagged, 1))
        recall = float(true_positive / max(n_anomalies, 1))
        precision_threshold = float(thresholds.get("precision", 0.7))
        recall_threshold = float(thresholds.get("recall", 0.6))
        accepted = precision >= precision_threshold and recall >= recall_threshold
        return {
            "success": True,
            "accepted": bool(accepted),
            "answer_schema": "anomaly_identification",
            "answer": parsed,
            "rule_text": rule_text,
            "precision": precision,
            "recall": recall,
            "precision_threshold": precision_threshold,
            "recall_threshold": recall_threshold,
            "n_flagged_in_audit": n_flagged,
            "n_anomalies_in_audit": n_anomalies,
            "true_positive_in_audit": true_positive,
            "score_source": f"fresh_audit_batch_n={n_audit}",
            "hypothesis_present": self._has_text(parsed.get("hypothesis")),
        }

    def _parse_json_answer(self, answer: Any) -> Dict[str, Any]:
        if isinstance(answer, dict):
            parsed = answer
        else:
            text = str(answer).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("answer must contain a JSON object")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("answer must be a JSON object")
        return parsed

    def _canonicalize_default_intervention(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Remove explicit default settings, e.g. {"Knob": "off"} == {}."""
        canonical: Dict[str, Any] = {}
        for name, value in intervention.items():
            default = str(self.knob_specs.get(name, {}).get("default", "off"))
            if str(value) != default:
                canonical[name] = value
        return canonical

    def _parse_intervention_answer(self, answer: Any) -> Dict[str, Any]:
        parsed = self._parse_json_answer(answer)
        if "intervention" not in parsed:
            raise ValueError("answer must include an 'intervention' object")
        if not isinstance(parsed.get("intervention"), dict):
            raise ValueError("answer.intervention must be a JSON object")
        return parsed

    def _parse_anomaly_answer(self, answer: Any) -> Dict[str, Any]:
        parsed = self._parse_json_answer(answer)
        if "anomaly_rule" not in parsed:
            raise ValueError("anomaly_identification answer requires 'anomaly_rule'")
        if parsed.get("flagged_unit_ids") is not None and not isinstance(parsed.get("flagged_unit_ids"), list):
            raise ValueError("anomaly_identification.flagged_unit_ids must be a list when present")
        return parsed

    def _eval_simple_rule(self, rule_text: str, obs: Dict[str, np.ndarray]) -> np.ndarray:
        tokens = re.split(r"\s+(AND|OR)\s+", rule_text.strip())
        if not tokens or not tokens[0].strip():
            raise ValueError("empty anomaly_rule")

        def clause_mask(clause: str) -> np.ndarray:
            match = re.match(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|>|<)\s*([-+]?\d+(?:\.\d+)?)\s*$",
                clause,
            )
            if not match:
                raise ValueError(f"unparseable clause: {clause!r}")
            var, op, raw_value = match.group(1), match.group(2), match.group(3)
            if var not in obs:
                raise ValueError(f"unknown variable in rule: {var!r}")
            arr = obs[var].astype(float)
            value = float(raw_value)
            if op == ">":
                return arr > value
            if op == "<":
                return arr < value
            if op == ">=":
                return arr >= value
            if op == "<=":
                return arr <= value
            if op == "==":
                return arr == value
            raise ValueError(f"unsupported operator: {op}")

        mask = clause_mask(tokens[0])
        idx = 1
        while idx < len(tokens):
            if idx + 1 >= len(tokens):
                raise ValueError("dangling boolean operator in rule")
            joiner = tokens[idx].upper()
            rhs = clause_mask(tokens[idx + 1])
            if joiner == "AND":
                mask = mask & rhs
            elif joiner == "OR":
                mask = mask | rhs
            else:
                raise ValueError(f"unsupported boolean operator: {joiner}")
            idx += 2
        return mask

    def _execute_query(self, query: StaticRPGParsedQuery) -> pd.DataFrame:
        if query.mode == "observational_sample":
            return self._observational_sample(query)
        if query.mode == "interventional_sample":
            return self._interventional_sample(query)
        if query.mode == "inspect_unit":
            return self._inspect_unit(query)
        raise ValueError(f"unknown query mode {query.mode!r}")

    def _observational_sample(self, query: StaticRPGParsedQuery) -> pd.DataFrame:
        measurements = self.validate_measurements(query.measurements)
        seed = self._query_seed(query, offset=1000)
        hidden = rpg._static_sample_hidden(self.cfg, query.n_units, seed=seed)
        assignments = rpg._static_assignment(self.cfg, hidden, seed=seed + 17)
        obs = self._observe_under_unit_assignments(hidden, assignments, measurements, seed=seed + 31)
        rows = self._rows_from_obs(obs, query_mode="observational_sample")
        return pd.DataFrame(rows)

    def _interventional_sample(self, query: StaticRPGParsedQuery) -> pd.DataFrame:
        measurements = self.validate_measurements(query.measurements)
        intervention = self.validate_intervention(query.intervention)
        seed = self._query_seed(query, offset=2000)
        hidden = rpg._static_sample_hidden(self.cfg, query.n_units, seed=seed)
        outcomes = rpg._static_apply(self.cfg, hidden, intervention, seed=seed + 17)
        obs = rpg._static_observe(self.cfg, hidden, outcomes, measurements, seed=seed + 31)
        rows = self._rows_from_obs(obs, query_mode="interventional_sample")
        for row in rows:
            for knob, value in intervention.items():
                row[f"do_{knob}"] = value
        return pd.DataFrame(rows)

    def _inspect_unit(self, query: StaticRPGParsedQuery) -> pd.DataFrame:
        measurements = self.validate_measurements(query.measurements)
        case_seed = int(query.case_seed or 0)
        self._seen_case_seeds.add(case_seed)
        seed = int(self.meta.get("seed", 0)) + 300000 + case_seed * 104729
        hidden = rpg._static_sample_hidden(self.cfg, 1, seed=seed)
        assignments = rpg._static_assignment(self.cfg, hidden, seed=seed + 17)
        obs = self._observe_under_unit_assignments(hidden, assignments, measurements, seed=seed + 31)
        rows = self._rows_from_obs(obs, query_mode="inspect_unit")
        rows[0]["case_seed"] = case_seed
        return pd.DataFrame(rows)

    def _observe_under_unit_assignments(
        self,
        hidden: Dict[str, Any],
        assignments: Dict[str, np.ndarray],
        measurements: List[str],
        *,
        seed: int,
    ) -> Dict[str, np.ndarray]:
        n_units = self._hidden_n(hidden)
        per_unit_key: List[Tuple[Tuple[str, str], ...]] = []
        for i in range(n_units):
            per_unit_key.append(tuple(sorted((k, str(assignments[k][i])) for k in assignments)))
        unique_keys = list(dict.fromkeys(per_unit_key))
        merged: Dict[str, np.ndarray] = {}
        for idx, key in enumerate(unique_keys):
            mask = np.array([k == key for k in per_unit_key], dtype=bool)
            sub_hidden = {
                name: values[mask] if isinstance(values, np.ndarray) else values
                for name, values in hidden.items()
            }
            intervention = dict(key)
            outcomes = rpg._static_apply(self.cfg, sub_hidden, intervention, seed=seed + 31 * (idx + 1))
            obs = rpg._static_observe(self.cfg, sub_hidden, outcomes, measurements, seed=seed + 41 * (idx + 1))
            for name, values in obs.items():
                if name not in merged:
                    merged[name] = np.empty(n_units, dtype=values.dtype)
                merged[name][mask] = values
        return merged

    def _rows_from_obs(self, obs: Dict[str, np.ndarray], *, query_mode: str) -> List[Dict[str, Any]]:
        n_units = len(next(iter(obs.values()))) if obs else 0
        rows: List[Dict[str, Any]] = []
        for i in range(n_units):
            row: Dict[str, Any] = {"unit_id": i, "query_mode": query_mode}
            for name in obs:
                value = obs[name][i]
                if hasattr(value, "item"):
                    value = value.item()
                row[name] = value
            rows.append(row)
        return rows

    def _hidden_n(self, hidden: Dict[str, Any]) -> int:
        for value in hidden.values():
            if isinstance(value, np.ndarray):
                return int(value.shape[0])
        raise ValueError("hidden state contains no arrays")

    def _query_seed(self, query: StaticRPGParsedQuery, *, offset: int) -> int:
        raw = json.dumps(query.to_dict(), sort_keys=True)
        stable = sum((i + 1) * ord(ch) for i, ch in enumerate(raw)) % 1_000_000
        return int(self.meta.get("seed", 0)) + offset + stable


def _demo() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run a typed RPG simulator query.")
    ap.add_argument("world_json")
    ap.add_argument("--mode", choices=sorted(RPG_QUERY_MODES), default="observational_trajectory")
    ap.add_argument("--policy-id", action="append", dest="policy_ids", default=[])
    ap.add_argument("--n-units", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--measurements", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    sim = RPGSimulator.from_json(args.world_json)
    query = RPGQuery(
        mode=args.mode,
        n_units=args.n_units,
        policy_ids=args.policy_ids,
        measurements=args.measurements,
        horizon=args.horizon,
        seed=args.seed,
    )
    if args.outdir:
        result = sim.run_query_to_csv(query, args.outdir)
    else:
        result = sim.run_query(query)
    print(json.dumps(result.to_dict(), indent=2))
    if result.success:
        print(result.preview_csv())


if __name__ == "__main__":
    _demo()
