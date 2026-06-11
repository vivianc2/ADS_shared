"""Schemas for RPG simulator worlds.

These are parallel to ``schemas.py`` but do not use BN terminology.  RPG
queries are longitudinal policy experiments, not static observational or
do-interventional samples.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class RPGQueryMode(Enum):
    OBSERVATIONAL_TRAJECTORY = "observational_trajectory"
    POLICY_ROLLOUT = "policy_rollout"
    POLICY_COMPARISON = "policy_comparison"


@dataclass
class RPGParsedQuery:
    """Structured representation of a typed RPG trajectory experiment."""

    mode: str
    n_units: int
    policy_ids: List[str] = field(default_factory=list)
    measurements: Optional[List[str]] = None
    horizon: Optional[int] = None
    seed: Optional[int] = None
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "n_units": self.n_units,
            "policy_ids": list(self.policy_ids),
            "measurements": None if self.measurements is None else list(self.measurements),
            "horizon": self.horizon,
            "seed": self.seed,
            "raw_query": self.raw_query,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_query: str = "") -> "RPGParsedQuery":
        policy_ids = data.get("policy_ids") or []
        if isinstance(policy_ids, str):
            policy_ids = [policy_ids]
        policy_ids = [
            f"policy_{m.group(1).upper()}" if (m := re.fullmatch(r"policy_([A-Za-z])", str(policy_id))) else str(policy_id)
            for policy_id in policy_ids
        ]
        measurements = data.get("measurements")
        if isinstance(measurements, str):
            measurements = [m.strip() for m in measurements.split(",") if m.strip()]
        horizon = data.get("horizon")
        if horizon is not None:
            horizon = int(horizon)
        seed = data.get("seed")
        if seed is not None:
            seed = int(seed)
        return cls(
            mode=str(data.get("mode", data.get("query_mode", ""))),
            n_units=int(data.get("n_units", data.get("n_samples", 100))),
            policy_ids=list(policy_ids),
            measurements=measurements,
            horizon=horizon,
            seed=seed,
            raw_query=raw_query or data.get("raw_query", ""),
        )


@dataclass
class RPGQueryResult:
    """Result returned by an RPG world model or simulator query."""

    success: bool
    query: RPGParsedQuery
    dataframe: Optional[pd.DataFrame] = None
    data_file: Optional[str] = None
    n_rows: int = 0
    columns: List[str] = field(default_factory=list)
    preview: str = ""
    error_message: Optional[str] = None
    sample_rows: int = 0
    sample_cells: int = 0
    sample_units: int = 0
    sample_accounting: str = "unit_period_rows"
    sample_usage_after: Optional[Dict[str, Any]] = None

    def to_xml(self) -> str:
        lines = ["<rpg_query_result>"]
        lines.append(f"  <success>{str(self.success).lower()}</success>")
        lines.append(f"  <mode>{self.query.mode}</mode>")
        lines.append(f"  <n_units>{self.query.n_units}</n_units>")
        if self.query.horizon is not None:
            lines.append(f"  <horizon>{self.query.horizon}</horizon>")
        if self.query.policy_ids:
            lines.append(f"  <policy_ids>{', '.join(self.query.policy_ids)}</policy_ids>")
        if self.query.measurements:
            lines.append(f"  <measurements>{', '.join(self.query.measurements)}</measurements>")

        if self.success:
            lines.append(f"  <n_rows>{self.n_rows}</n_rows>")
            lines.append(f"  <columns>{', '.join(self.columns)}</columns>")
            if self.sample_usage_after:
                remaining = self.sample_usage_after.get("sample_units_remaining")
                remaining_text = "unlimited" if remaining is None else str(remaining)
                lines.append(
                    "  <sample_budget>"
                    f"{self.sample_usage_after.get('sample_units_used', 0)} "
                    f"{self.sample_usage_after.get('sample_accounting', self.sample_accounting)} used; "
                    f"{remaining_text} remaining"
                    "</sample_budget>"
                )
            if self.data_file:
                lines.append(f"  <data_file>{self.data_file}</data_file>")
            lines.append(f"  <preview>\n{self.preview}\n  </preview>")
        else:
            lines.append(f"  <error>{self.error_message}</error>")

        lines.append("</rpg_query_result>")
        return "\n".join(lines)

    def preview_csv(self, n: int = 10) -> str:
        if self.dataframe is None:
            return ""
        return self.dataframe.head(n).to_csv(index=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "query": self.query.to_dict(),
            "data_file": self.data_file,
            "n_rows": self.n_rows,
            "columns": self.columns,
            "preview": self.preview,
            "error_message": self.error_message,
            "sample_rows": self.sample_rows,
            "sample_cells": self.sample_cells,
            "sample_units": self.sample_units,
            "sample_accounting": self.sample_accounting,
            "sample_usage_after": self.sample_usage_after,
        }


@dataclass
class RPGWorldInfo:
    """Public RPG world information shown to scientist agents."""

    story: str
    observed_variables: List[Dict[str, Any]]
    action_variables: List[Dict[str, Any]]
    allowed_policies: List[Dict[str, Any]]
    allowed_measurements: List[str]
    allowed_query_modes: List[str]
    default_horizon: int
    default_observational_policy_id: str
    question: str = ""
    discovery_protocol: Dict[str, Any] = field(default_factory=dict)
    experiment_budget: Dict[str, Any] = field(default_factory=dict)

    def to_xml(self) -> str:
        lines = ["<rpg_world_info>"]
        lines.append(f"  <story>{self.story}</story>")
        lines.append(f"  <default_horizon>{self.default_horizon}</default_horizon>")
        lines.append(
            f"  <default_observational_policy_id>{self.default_observational_policy_id}</default_observational_policy_id>"
        )
        if self.experiment_budget:
            lines.append("  <experiment_budget>")
            for key in [
                "sample_accounting",
                "max_total_samples",
                "max_samples_per_query",
                "default_units",
                "max_units",
                "max_queries",
            ]:
                if key in self.experiment_budget:
                    lines.append(f"    <{key}>{self.experiment_budget.get(key)}</{key}>")
            if self.experiment_budget.get("counted_unit"):
                lines.append(f"    <counted_unit>{self.experiment_budget.get('counted_unit')}</counted_unit>")
            lines.append("  </experiment_budget>")
        lines.append("  <allowed_query_modes>")
        for mode in self.allowed_query_modes:
            lines.append(f"    <mode>{mode}</mode>")
        lines.append("  </allowed_query_modes>")
        lines.append("  <observed_variables>")
        for var in self.observed_variables:
            lines.append("    <variable>")
            lines.append(f"      <name>{var.get('name')}</name>")
            lines.append(f"      <role>{var.get('role', '')}</role>")
            lines.append(f"      <description>{var.get('description', var.get('desc', ''))}</description>")
            lines.append(f"      <scale>{json.dumps(var.get('scale', {}))}</scale>")
            lines.append("    </variable>")
        lines.append("  </observed_variables>")
        lines.append("  <action_variables>")
        for var in self.action_variables:
            lines.append("    <action>")
            lines.append(f"      <name>{var.get('name')}</name>")
            lines.append(f"      <values>{', '.join(var.get('values', []))}</values>")
            lines.append(f"      <description>{var.get('description', var.get('desc', ''))}</description>")
            lines.append("    </action>")
        lines.append("  </action_variables>")
        lines.append("  <allowed_policies>")
        for policy in self.allowed_policies:
            lines.append("    <policy>")
            lines.append(f"      <policy_id>{policy.get('policy_id')}</policy_id>")
            lines.append(f"      <display_name>{policy.get('display_name', '')}</display_name>")
            lines.append(f"      <description>{policy.get('description', '')}</description>")
            lines.append("    </policy>")
        lines.append("  </allowed_policies>")
        if self.question:
            lines.append(f"  <question>{self.question}</question>")
        lines.append("</rpg_world_info>")
        return "\n".join(lines)

    def get_measurement_catalog(self) -> str:
        lines = []
        for var in self.observed_variables:
            lines.append(
                f"- {var.get('name')}: {var.get('description', var.get('desc', ''))} "
                f"(role: {var.get('role', '')})"
            )
        return "\n".join(lines)

    def get_policy_catalog(self) -> str:
        lines = []
        for policy in self.allowed_policies:
            lines.append(
                f"- {policy.get('policy_id')}: {policy.get('display_name', '')}; "
                f"{policy.get('description', '')}"
            )
        return "\n".join(lines)


class RPGWorldModelError(Exception):
    pass


class RPGQueryParseError(RPGWorldModelError):
    pass


class RPGQueryValidationError(RPGWorldModelError):
    pass


class RPGQueryExecutionError(RPGWorldModelError):
    pass


# ---------------------------------------------------------------------------
# Static RPG v2 schemas
# ---------------------------------------------------------------------------


class StaticRPGQueryMode(Enum):
    OBSERVATIONAL_SAMPLE = "observational_sample"
    INTERVENTIONAL_SAMPLE = "interventional_sample"
    INSPECT_UNIT = "inspect_unit"


@dataclass
class StaticRPGParsedQuery:
    """Structured representation of one static RPG sampling request."""

    mode: str
    n_units: int = 100
    measurements: Optional[List[str]] = None
    intervention: Dict[str, Any] = field(default_factory=dict)
    case_seed: Optional[int] = None
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "n_units": self.n_units,
            "measurements": None if self.measurements is None else list(self.measurements),
            "intervention": dict(self.intervention),
            "case_seed": self.case_seed,
            "raw_query": self.raw_query,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_query: str = "") -> "StaticRPGParsedQuery":
        measurements = data.get("measurements")
        if isinstance(measurements, str):
            measurements = [m.strip() for m in measurements.split(",") if m.strip()]
        intervention = data.get("intervention") or data.get("do") or {}
        if not isinstance(intervention, dict):
            raise ValueError("intervention must be a JSON object mapping knob names to values")
        mode = str(data.get("mode", data.get("query_mode", ""))).strip()
        n_units = int(data.get("n_units", data.get("n_samples", 1 if mode == "inspect_unit" else 100)))
        case_seed = data.get("case_seed")
        if case_seed is not None:
            case_seed = int(case_seed)
        return cls(
            mode=mode,
            n_units=n_units,
            measurements=measurements,
            intervention={str(k): v for k, v in intervention.items()},
            case_seed=case_seed,
            raw_query=raw_query or data.get("raw_query", ""),
        )


@dataclass
class StaticRPGQueryResult:
    """Result returned by the static RPG simulator/world model."""

    success: bool
    query: StaticRPGParsedQuery
    dataframe: Optional[pd.DataFrame] = None
    data_file: Optional[str] = None
    n_rows: int = 0
    columns: List[str] = field(default_factory=list)
    preview: str = ""
    error_message: Optional[str] = None
    sample_rows: int = 0
    sample_cells: int = 0
    sample_units: int = 0
    sample_accounting: str = "cells"
    sample_usage_after: Optional[Dict[str, Any]] = None

    def preview_csv(self, n: int = 10) -> str:
        if self.dataframe is None:
            return ""
        return self.dataframe.head(n).to_csv(index=False)

    def to_xml(self) -> str:
        lines = ["<static_rpg_query_result>"]
        lines.append(f"  <success>{str(self.success).lower()}</success>")
        lines.append(f"  <mode>{self.query.mode}</mode>")
        if self.query.mode != StaticRPGQueryMode.INSPECT_UNIT.value:
            lines.append(f"  <n_units>{self.query.n_units}</n_units>")
        if self.query.case_seed is not None:
            lines.append(f"  <case_seed>{self.query.case_seed}</case_seed>")
        if self.query.intervention:
            lines.append(f"  <intervention>{json.dumps(self.query.intervention, sort_keys=True)}</intervention>")
        if self.query.measurements:
            lines.append(f"  <measurements>{', '.join(self.query.measurements)}</measurements>")
        if self.success:
            lines.append(f"  <n_rows>{self.n_rows}</n_rows>")
            lines.append(f"  <columns>{', '.join(self.columns)}</columns>")
            if self.sample_usage_after:
                remaining = self.sample_usage_after.get("sample_units_remaining")
                remaining_text = "unlimited" if remaining is None else str(remaining)
                lines.append(
                    "  <sample_budget>"
                    f"{self.sample_usage_after.get('sample_units_used', 0)} "
                    f"{self.sample_usage_after.get('sample_accounting', self.sample_accounting)} used; "
                    f"{remaining_text} remaining"
                    "</sample_budget>"
                )
            if self.data_file:
                lines.append(f"  <data_file>{self.data_file}</data_file>")
            if self.preview:
                lines.append(f"  <preview>\n{self.preview}\n  </preview>")
        else:
            lines.append(f"  <error>{self.error_message}</error>")
        lines.append("</static_rpg_query_result>")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "query": self.query.to_dict(),
            "data_file": self.data_file,
            "n_rows": self.n_rows,
            "columns": self.columns,
            "preview": self.preview,
            "error_message": self.error_message,
            "sample_rows": self.sample_rows,
            "sample_cells": self.sample_cells,
            "sample_units": self.sample_units,
            "sample_accounting": self.sample_accounting,
            "sample_usage_after": self.sample_usage_after,
        }


@dataclass
class StaticRPGQuestion:
    question_type: str
    question_text: str
    ground_truth: Dict[str, Any]
    answer_schema: str = "intervention_with_hypothesis"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticRPGWorldInfo:
    """Public static RPG world information shown to scientist agents."""

    world_id: str
    story: str
    observed_variables: List[Dict[str, Any]]
    intervenable_variables: List[Dict[str, Any]]
    allowed_measurements: List[str]
    allowed_query_modes: List[str]
    experiment_budget: Dict[str, Any]
    question: str
    answer_schema: str
    max_intervention_knobs: int
    discovery_protocol: Dict[str, Any] = field(default_factory=dict)

    def get_measurement_catalog(self) -> str:
        lines = []
        for var in self.observed_variables:
            scale = var.get("scale", {})
            scale_text = json.dumps(scale, sort_keys=True)
            lines.append(f"- {var.get('name')}: {var.get('description', '')} scale={scale_text}")
        return "\n".join(lines)

    def get_intervention_catalog(self) -> str:
        lines = []
        for var in self.intervenable_variables:
            default = var.get("default", "")
            value_type = str(var.get("value_type", "")).lower()
            if value_type == "continuous":
                lo = var.get("min", 0)
                hi = var.get("max", 100)
                spec = f"type=continuous, setpoint in [{lo}, {hi}]"
            else:
                values = ", ".join(str(v) for v in var.get("values", []))
                type_tag = f"type={value_type}, " if value_type else ""
                spec = f"{type_tag}values=[{values}]"
            lines.append(
                f"- {var.get('name')}: {spec}, default={default}. "
                f"{var.get('description', '')}"
            )
        return "\n".join(lines)

    def to_xml(self) -> str:
        lines = ["<static_rpg_world_info>"]
        lines.append(f"  <world_id>{self.world_id}</world_id>")
        lines.append(f"  <story>{self.story}</story>")
        lines.append(f"  <question>{self.question}</question>")
        lines.append(f"  <answer_schema>{self.answer_schema}</answer_schema>")
        lines.append(f"  <max_intervention_knobs>{self.max_intervention_knobs}</max_intervention_knobs>")
        lines.append("  <experiment_budget>")
        for key, value in self.experiment_budget.items():
            lines.append(f"    <{key}>{value}</{key}>")
        lines.append("  </experiment_budget>")
        lines.append("  <allowed_query_modes>")
        for mode in self.allowed_query_modes:
            lines.append(f"    <mode>{mode}</mode>")
        lines.append("  </allowed_query_modes>")
        lines.append("  <observed_variables>")
        for var in self.observed_variables:
            lines.append("    <variable>")
            lines.append(f"      <name>{var.get('name')}</name>")
            lines.append(f"      <description>{var.get('description', '')}</description>")
            lines.append(f"      <scale>{json.dumps(var.get('scale', {}), sort_keys=True)}</scale>")
            lines.append("    </variable>")
        lines.append("  </observed_variables>")
        lines.append("  <intervenable_variables>")
        for var in self.intervenable_variables:
            lines.append("    <knob>")
            lines.append(f"      <name>{var.get('name')}</name>")
            lines.append(f"      <values>{', '.join(str(v) for v in var.get('values', []))}</values>")
            lines.append(f"      <default>{var.get('default', '')}</default>")
            lines.append(f"      <description>{var.get('description', '')}</description>")
            lines.append("    </knob>")
        lines.append("  </intervenable_variables>")
        lines.append("</static_rpg_world_info>")
        return "\n".join(lines)
