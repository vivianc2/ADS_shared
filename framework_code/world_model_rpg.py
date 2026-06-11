"""World model for RPG simulator worlds.

This is the RPG analogue of ``world_model_causal.py``.  It parses scientist
requests into typed longitudinal policy experiments, validates them against the
public RPG catalog, executes them through ``RPGSimulator``, and returns CSV
previews.  It deliberately avoids BN/do-calculus terminology.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

from schemas_rpg import (
    RPGParsedQuery,
    RPGQueryExecutionError,
    RPGQueryParseError,
    RPGQueryResult,
    RPGQueryValidationError,
    RPGWorldInfo,
    StaticRPGParsedQuery,
    StaticRPGQueryResult,
    StaticRPGWorldInfo,
)
from simulator_rpg import RPGSimulator, StaticRPGSimulator


logger = logging.getLogger(__name__)


@dataclass
class WorldModelRPG:
    """Agent-facing query layer for one RPG world."""

    simulator: RPGSimulator
    llm: Optional[Any] = None
    output_dir: str = "./query_results_rpg"

    default_units: int = 100
    max_units: int = 10000
    preview_rows: int = 10
    max_total_samples: Optional[int] = None
    max_samples_per_query: Optional[int] = None
    max_queries: Optional[int] = None
    sample_accounting: str = "unit_period_rows"
    use_world_budget: bool = True

    _query_counter: int = field(default=0, init=False)
    _successful_query_count: int = field(default=0, init=False)
    _sample_rows_used: int = field(default=0, init=False)
    _sample_cells_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        Path(self.output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        if self.use_world_budget:
            self._apply_world_budget_defaults()
        if self.sample_accounting == "rows":
            self.sample_accounting = "unit_period_rows"
        if self.sample_accounting not in ("unit_period_rows", "cells"):
            raise ValueError("sample_accounting must be 'unit_period_rows', 'rows', or 'cells'")

    def _apply_world_budget_defaults(self) -> None:
        budget = self.simulator.visible.get("experiment_budget", {})
        if not budget:
            return
        self.sample_accounting = str(budget.get("sample_accounting", self.sample_accounting))
        if self.default_units == 100 and budget.get("default_units") is not None:
            self.default_units = int(budget["default_units"])
        if self.max_units == 10000 and budget.get("max_units") is not None:
            self.max_units = int(budget["max_units"])
        if self.max_total_samples is None and budget.get("max_total_samples") is not None:
            self.max_total_samples = int(budget["max_total_samples"])
        if self.max_samples_per_query is None and budget.get("max_samples_per_query") is not None:
            self.max_samples_per_query = int(budget["max_samples_per_query"])
        if self.max_queries is None and budget.get("max_queries") is not None:
            self.max_queries = int(budget["max_queries"])

    @classmethod
    def from_world_json(
        cls,
        path: str,
        llm: Optional[Any] = None,
        output_dir: str = "./query_results_rpg",
        use_world_budget: bool = True,
    ) -> "WorldModelRPG":
        return cls(
            simulator=RPGSimulator.from_json(path),
            llm=llm,
            output_dir=output_dir,
            use_world_budget=use_world_budget,
        )

    def get_world_info(self) -> RPGWorldInfo:
        public = self.simulator.public_world()
        return RPGWorldInfo(
            story=public["story"],
            observed_variables=public["observed_variables"],
            action_variables=public["action_variables"],
            allowed_policies=public["allowed_policies"],
            allowed_measurements=public["allowed_measurements"],
            allowed_query_modes=public["allowed_query_modes"],
            default_horizon=public["default_horizon"],
            default_observational_policy_id=public["default_observational_policy_id"],
            question=public["question"],
            discovery_protocol=public.get("discovery_protocol", {}),
            experiment_budget=public.get("experiment_budget", {}),
        )

    def reset_sample_usage(self) -> None:
        self._sample_rows_used = 0
        self._sample_cells_used = 0

    def get_sample_usage(self) -> Dict[str, Any]:
        units = self._sample_cells_used if self.sample_accounting == "cells" else self._sample_rows_used
        remaining = None
        if self.max_total_samples is not None:
            remaining = max(0, self.max_total_samples - units)
        return {
            "sample_rows_used": self._sample_rows_used,
            "sample_cells_used": self._sample_cells_used,
            "sample_units_used": units,
            "sample_accounting": self.sample_accounting,
            "max_total_samples": self.max_total_samples,
            "sample_units_remaining": remaining,
            "successful_queries": self._successful_query_count,
            "max_queries": self.max_queries,
        }

    def process_query(self, query: str, seed: Optional[int] = None) -> RPGQueryResult:
        """Parse, validate, execute, and save one RPG trajectory query."""
        self._query_counter += 1
        try:
            parsed = self._parse_query(query)
            if seed is not None and parsed.seed is None:
                parsed.seed = seed
            self._validate_query(parsed)
            self._validate_sample_budget(parsed)
            df = self.simulator._execute_query(parsed)
            self._record_sample_usage(df)
            self._successful_query_count += 1
            return self._create_success_result(parsed, df)
        except RPGQueryParseError as exc:
            return self._create_error_result(query, str(exc))
        except RPGQueryValidationError as exc:
            return self._create_error_result(query, str(exc))
        except Exception as exc:
            logger.exception("RPG query failed")
            return self._create_error_result(query, f"{type(exc).__name__}: {exc}")

    def _parse_query(self, query: str) -> RPGParsedQuery:
        explicit = self._extract_explicit_json(query)
        if explicit is not None:
            return RPGParsedQuery.from_dict(explicit, raw_query=query)

        parsed = self._parse_simple_text(query)
        if parsed is not None:
            return parsed

        if self.llm is not None:
            return self._parse_with_llm(query)

        raise RPGQueryParseError(
            "Could not parse RPG query. Use JSON with mode/n_units/policy_ids/measurements/horizon, "
            "or a simple request such as 'compare policy_A and policy_B for 100 units over 8 periods measuring X, Y'."
        )

    def _extract_explicit_json(self, query: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"<json>\s*(\{.*?\})\s*</json>", query, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            stripped = query.strip()
            text = stripped if stripped.startswith("{") and stripped.endswith("}") else ""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RPGQueryParseError(f"Invalid JSON query: {exc}") from exc

    def _parse_simple_text(self, query: str) -> Optional[RPGParsedQuery]:
        q = query.strip()
        lower = q.lower()
        policy_ids = []
        for policy_id in re.findall(r"policy_[a-z]", q, flags=re.IGNORECASE):
            normalized = f"policy_{policy_id.split('_', 1)[1].upper()}"
            if normalized not in policy_ids:
                policy_ids.append(normalized)
        all_policies_requested = bool(
            re.search(r"\b(?:all|every|each)\s+(?:candidate\s+)?polic(?:y|ies)\b", lower)
            or ("candidate policies" in lower and "compare" in lower)
        )

        if any(word in lower for word in ["observational", "default", "current practice", "passive"]):
            mode = "observational_trajectory"
            policy_ids = []
        elif all_policies_requested:
            mode = "policy_comparison"
            policy_ids = [policy["policy_id"] for policy in self.simulator.allowed_policies]
        elif "compare" in lower or len(policy_ids) > 1:
            mode = "policy_comparison"
        elif policy_ids:
            mode = "policy_rollout"
        else:
            return None

        n_units = self.default_units
        n_match = re.search(r"(\d+)\s*(?:units?|people|patients|students|users|clients|participants|samples?|trajectories)", lower)
        if n_match:
            n_units = int(n_match.group(1))

        horizon = None
        h_match = re.search(r"(?:over|for)\s+(\d+)\s*(?:periods?|steps?|weeks?|months?|time steps?)", lower)
        if h_match:
            horizon = int(h_match.group(1))

        seed = None
        seed_match = re.search(r"seed\s*=?\s*(\d+)", lower)
        if seed_match:
            seed = int(seed_match.group(1))

        measurements = self._parse_measurements_from_text(q)
        return RPGParsedQuery(
            mode=mode,
            n_units=n_units,
            policy_ids=policy_ids,
            measurements=measurements,
            horizon=horizon,
            seed=seed,
            raw_query=query,
        )

    def _parse_measurements_from_text(self, query: str) -> Optional[List[str]]:
        lower = query.lower()
        if (
            "measure all" in lower
            or "measuring all" in lower
            or "all measurements" in lower
            or re.search(r"\bmeasure(?:ments?|ing)?\s+all\b", lower)
        ):
            return None

        allowed = self.simulator.allowed_measurements
        found = [name for name in allowed if name.lower() in lower]
        if found:
            return found

        measure_match = re.search(r"(?:measuring|measure|measurements?)\s+(.+)$", query, re.IGNORECASE)
        if not measure_match:
            return None
        raw = measure_match.group(1)
        raw = re.split(r"\b(?:over|for|with seed|seed)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        pieces = [p.strip(" .;") for p in re.split(r",|\band\b", raw) if p.strip(" .;")]
        # Keep exact pieces here; validation will give a clear error if any are unknown.
        return pieces or None

    def _parse_with_llm(self, query: str) -> RPGParsedQuery:
        system = (
            "You parse requests for an RPG longitudinal simulator. Output only "
            "<json>{...}</json>. Valid modes are observational_trajectory, "
            "policy_rollout, and policy_comparison. Include n_units, policy_ids, "
            "measurements, horizon, and seed when specified."
        )
        info = self.get_world_info()
        user = (
            "Allowed measurements:\n"
            f"{', '.join(info.allowed_measurements)}\n\n"
            "Allowed policies:\n"
            f"{', '.join(p['policy_id'] for p in info.allowed_policies)}\n\n"
            f"User query:\n{query}"
        )
        output = self.llm.generate(system, user)
        explicit = self._extract_explicit_json(output)
        if explicit is None:
            raise RPGQueryParseError(f"LLM did not return JSON: {output[:300]}")
        return RPGParsedQuery.from_dict(explicit, raw_query=query)

    def _validate_query(self, parsed: RPGParsedQuery) -> None:
        if parsed.n_units > self.max_units:
            raise RPGQueryValidationError(f"n_units {parsed.n_units} exceeds max_units {self.max_units}")
        try:
            self.simulator.validate_query(parsed)
        except Exception as exc:
            raise RPGQueryValidationError(str(exc)) from exc

    def _validate_sample_budget(self, parsed: RPGParsedQuery) -> None:
        if self.max_queries is not None and self._successful_query_count >= self.max_queries:
            raise RPGQueryValidationError(
                f"Query budget exhausted: {self._successful_query_count}/{self.max_queries} successful queries used"
            )
        usage = self.simulator.estimate_sample_usage(parsed, accounting=self.sample_accounting)
        units = usage["cells"] if self.sample_accounting == "cells" else usage["unit_period_rows"]
        if self.max_samples_per_query is not None and units > self.max_samples_per_query:
            raise RPGQueryValidationError(
                f"Query requests {units} {self.sample_accounting}, above per-query limit {self.max_samples_per_query}"
            )
        if self.max_total_samples is not None:
            used = self._sample_cells_used if self.sample_accounting == "cells" else self._sample_rows_used
            if used + units > self.max_total_samples:
                raise RPGQueryValidationError(
                    f"Query would use {used + units}/{self.max_total_samples} {self.sample_accounting}"
                )

    def _record_sample_usage(self, df: pd.DataFrame) -> None:
        self._sample_rows_used += int(len(df))
        self._sample_cells_used += int(len(df) * len(df.columns))

    def _create_success_result(self, parsed: RPGParsedQuery, df: pd.DataFrame) -> RPGQueryResult:
        out = Path(self.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        world_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.simulator.meta.get("world_id", "rpg_world")))
        path = out / f"{world_id}_query_{self._query_counter:04d}.csv"
        if path.exists():
            suffix = 2
            while True:
                candidate = out / f"{world_id}_query_{self._query_counter:04d}_{suffix}.csv"
                if not candidate.exists():
                    path = candidate
                    break
                suffix += 1
        df.to_csv(path, index=False)
        rows = int(len(df))
        cells = int(rows * len(df.columns))
        return RPGQueryResult(
            success=True,
            query=parsed,
            dataframe=df,
            data_file=str(path),
            n_rows=rows,
            columns=list(df.columns),
            preview=df.head(self.preview_rows).to_csv(index=False),
            sample_rows=rows,
            sample_cells=cells,
            sample_units=cells if self.sample_accounting == "cells" else rows,
            sample_accounting=self.sample_accounting,
            sample_usage_after=self.get_sample_usage(),
        )

    def _create_error_result(self, raw_query: str, message: str) -> RPGQueryResult:
        parsed = RPGParsedQuery(
            mode="parse_error",
            n_units=0,
            policy_ids=[],
            measurements=[],
            raw_query=raw_query,
        )
        return RPGQueryResult(
            success=False,
            query=parsed,
            error_message=message,
            sample_accounting=self.sample_accounting,
            sample_usage_after=self.get_sample_usage(),
        )


def _demo() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run a natural-language RPG world-model query.")
    ap.add_argument("world_json")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--outdir", default="./query_results_rpg")
    args = ap.parse_args()

    wm = WorldModelRPG.from_world_json(args.world_json, output_dir=args.outdir)
    result = wm.process_query(" ".join(args.query))
    print(result.to_xml())


if __name__ == "__main__":
    _demo()


# ---------------------------------------------------------------------------
# Static RPG v2 world model
# ---------------------------------------------------------------------------


@dataclass
class StaticRPGWorldModel:
    """Agent-facing query layer for static partially observed RPG worlds."""

    simulator: StaticRPGSimulator
    output_dir: str = "./query_results_rpg"
    preview_rows: int = 10
    use_world_budget: bool = True

    max_total_samples: Optional[int] = None
    max_samples_per_query: Optional[int] = None
    max_units_per_query: Optional[int] = None
    max_measurements_per_query: Optional[int] = None
    max_queries: Optional[int] = None
    sample_accounting: str = "cells"

    _query_counter: int = field(default=0, init=False)
    _successful_query_count: int = field(default=0, init=False)
    _sample_cells_used: int = field(default=0, init=False)
    _sample_rows_used: int = field(default=0, init=False)
    _successful_query_records: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        Path(self.output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        if self.use_world_budget:
            self._apply_world_budget_defaults()

    @classmethod
    def from_world_json(
        cls,
        path: str,
        output_dir: str = "./query_results_rpg",
        use_world_budget: bool = True,
    ) -> "StaticRPGWorldModel":
        return cls(
            simulator=StaticRPGSimulator.from_json(path),
            output_dir=output_dir,
            use_world_budget=use_world_budget,
        )

    def _apply_world_budget_defaults(self) -> None:
        budget = self.simulator.visible.get("experiment_budget", {})
        self.sample_accounting = str(budget.get("sample_accounting", self.sample_accounting))
        if self.max_total_samples is None and budget.get("max_total_samples") is not None:
            self.max_total_samples = int(budget["max_total_samples"])
        if self.max_samples_per_query is None and budget.get("max_samples_per_query") is not None:
            self.max_samples_per_query = int(budget["max_samples_per_query"])
        if self.max_units_per_query is None and budget.get("max_units_per_query") is not None:
            self.max_units_per_query = int(budget["max_units_per_query"])
        if self.max_measurements_per_query is None and budget.get("max_measurements_per_query") is not None:
            self.max_measurements_per_query = int(budget["max_measurements_per_query"])
        if self.max_queries is None and budget.get("max_queries") is not None:
            self.max_queries = int(budget["max_queries"])

    def get_world_info(self) -> StaticRPGWorldInfo:
        public = self.simulator.public_world()
        return StaticRPGWorldInfo(
            world_id=public["world_id"],
            story=public["story"],
            observed_variables=public["observed_variables"],
            intervenable_variables=public["intervenable_variables"],
            allowed_measurements=public["allowed_measurements"],
            allowed_query_modes=public["allowed_query_modes"],
            experiment_budget=public["experiment_budget"],
            discovery_protocol=public.get("discovery_protocol", {}),
            question=public["question"],
            answer_schema=public["answer_schema"],
            max_intervention_knobs=int(public["max_intervention_knobs"]),
        )

    def reset_sample_usage(self) -> None:
        self._query_counter = 0
        self._successful_query_count = 0
        self._sample_cells_used = 0
        self._sample_rows_used = 0
        self._successful_query_records = []

    def get_sample_usage(self) -> Dict[str, Any]:
        used = self._sample_cells_used if self.sample_accounting == "cells" else self._sample_rows_used
        remaining = None
        if self.max_total_samples is not None:
            remaining = max(0, self.max_total_samples - used)
        return {
            "sample_rows_used": self._sample_rows_used,
            "sample_cells_used": self._sample_cells_used,
            "sample_units_used": used,
            "sample_accounting": self.sample_accounting,
            "max_total_samples": self.max_total_samples,
            "sample_units_remaining": remaining,
            "successful_queries": self._successful_query_count,
            "max_queries": self.max_queries,
        }

    def process_query(self, query: str) -> StaticRPGQueryResult:
        self._query_counter += 1
        try:
            parsed = self._parse_query(query)
            self._validate_query(parsed)
            self._check_not_duplicate(parsed)
            self._validate_budget(parsed)
            result = self.simulator.run_query(parsed)
            if not result.success:
                result.sample_usage_after = self.get_sample_usage()
                return result
            self._record_usage(result)
            self._successful_query_count += 1
            self._successful_query_records.append(result.query.to_dict())
            self._save_result_csv(result)
            result.sample_accounting = self.sample_accounting
            result.sample_usage_after = self.get_sample_usage()
            return result
        except RPGQueryParseError as exc:
            return self._error_result(query, f"parse_error: {exc}", "parse_error")
        except RPGQueryValidationError as exc:
            # Covers schema/measurement validation, budget exhaustion, and
            # duplicate-query rejections. Labelling these as "parse_error" (the
            # old behaviour) made trace analysis misreport budget overspends as
            # malformed JSON, so keep the cause distinct.
            return self._error_result(query, f"invalid_query: {exc}", "validation_error")
        except Exception as exc:
            logger.exception("static RPG query failed")
            return self._error_result(query, f"{type(exc).__name__}: {exc}", "execution_error")

    def _error_result(self, query: str, message: str, mode: str) -> StaticRPGQueryResult:
        parsed = StaticRPGParsedQuery(mode=mode, n_units=0, raw_query=query)
        return StaticRPGQueryResult(
            success=False,
            query=parsed,
            error_message=message,
            sample_accounting=self.sample_accounting,
            sample_usage_after=self.get_sample_usage(),
        )

    @staticmethod
    def _query_signature(
        mode: str,
        measurements: Optional[List[str]],
        intervention: Dict[str, Any],
        case_seed: Optional[int] = None,
    ):
        meas = tuple(sorted(measurements or []))
        intv = tuple(sorted((str(k), str(v)) for k, v in (intervention or {}).items()))
        # case_seed distinguishes individual inspect_unit lookups, which are not
        # duplicates of each other even with identical measurements.
        return (mode, meas, intv, case_seed)

    def _check_not_duplicate(self, parsed: StaticRPGParsedQuery) -> None:
        """Reject a query identical to one already run successfully.

        Re-issuing the same (mode, measurements, intervention) returns no new
        information and was a major driver of the degeneration loop (the agent
        re-querying to "re-see" data it had lost from context). n_units is
        ignored on purpose: resampling the same shape is the wasteful pattern we
        want to stop. This raises before budget/execution, so it costs neither a
        successful query nor cells -- only the turn that issued it.
        """
        sig = self._query_signature(
            parsed.mode, parsed.measurements, parsed.intervention, parsed.case_seed
        )
        for i, rec in enumerate(self._successful_query_records, 1):
            rec_sig = self._query_signature(
                rec.get("mode", ""), rec.get("measurements"), rec.get("intervention") or {},
                rec.get("case_seed"),
            )
            if rec_sig == sig:
                raise RPGQueryValidationError(
                    f"duplicate_query: identical to successful query #{i} "
                    f"(mode={parsed.mode}, same measurements and intervention). You already "
                    "have this data in RECENT QUERY RESULTS or your memory. Change the "
                    "measurements or the intervention, or submit your answer now."
                )

    def score_answer(self, answer: Any) -> Dict[str, Any]:
        try:
            score = self.simulator.score_answer(answer)
            if score.get("answer_schema") == "latent_cause_hypothesis":
                trajectory = self._latent_cause_trajectory_evidence()
                score["trajectory_evidence"] = trajectory
                if not trajectory["mechanism_verification_query"]:
                    score["accepted_without_trajectory"] = bool(score.get("accepted"))
                    score["accepted"] = False
            return score
        except Exception as exc:
            return {"success": False, "accepted": False, "error": str(exc), "answer": answer}

    def _latent_cause_trajectory_evidence(self) -> Dict[str, Any]:
        requirements = (
            self.simulator.world.get("oracle", {})
            .get("gold_answer", {})
            .get("trajectory_requirements", {})
        )
        mechanism_measurements = set(requirements.get("mechanism_measurements") or {
            "DownspoutDischargeDelay",
            "RoofEdgeOverflowScore",
            "FlowTestBackflowScore",
            "YardFloodArea",
        })
        context_measurements = set(requirements.get("context_measurements") or {"RainfallAmount", "RecentLeafFallIntensity"})
        outcome_measurement = str(requirements.get("outcome_measurement") or "YardFloodArea")
        targeted_actions = set(requirements.get("targeted_actions") or {"RunRoofEdgeFlowTest", "ClearRearGutters", "FlushDownspout"})
        saw_mechanism_measurement = False
        saw_context_and_outcome = False
        mechanism_verification_query = False
        for record in self._successful_query_records:
            measurements = set(record.get("measurements") or [])
            intervention = dict(record.get("intervention") or {})
            if measurements & mechanism_measurements:
                saw_mechanism_measurement = True
            if (measurements & context_measurements) and outcome_measurement in measurements:
                saw_context_and_outcome = True
            if (
                record.get("mode") == "interventional_sample"
                and (set(intervention) & targeted_actions)
                and (measurements & mechanism_measurements)
            ):
                mechanism_verification_query = True
        return {
            "saw_mechanism_measurement": saw_mechanism_measurement,
            "saw_context_and_outcome": saw_context_and_outcome,
            "mechanism_verification_query": mechanism_verification_query,
            "n_successful_queries": len(self._successful_query_records),
        }

    def _parse_query(self, query: str) -> StaticRPGParsedQuery:
        explicit = self._extract_json(query)
        if explicit is None:
            raise RPGQueryParseError(
                "Static RPG queries must be JSON. Example: "
                '{"mode":"observational_sample","n_units":200,'
                '"measurements":["OutcomeProxy","MechanismProxy"]}'
            )
        parsed = StaticRPGParsedQuery.from_dict(explicit, raw_query=query)
        return parsed

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"<json>\s*(\{.*?\})\s*</json>", text, re.DOTALL)
        if match:
            raw = match.group(1)
        else:
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                raw = stripped
            else:
                match = re.search(r"\{.*\}", stripped, re.DOTALL)
                raw = match.group(0) if match else ""
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RPGQueryParseError(f"Invalid JSON query: {exc}") from exc
        if not isinstance(data, dict):
            raise RPGQueryParseError("Query JSON must be an object")
        return data

    def _validate_query(self, parsed: StaticRPGParsedQuery) -> None:
        measurements = self.simulator.validate_measurements(parsed.measurements)
        if self.max_measurements_per_query is not None and len(measurements) > self.max_measurements_per_query:
            raise RPGQueryValidationError(
                f"requested {len(measurements)} measurements, above max_measurements_per_query="
                f"{self.max_measurements_per_query}"
            )
        n_units = 1 if parsed.mode == "inspect_unit" else parsed.n_units
        if self.max_units_per_query is not None and n_units > self.max_units_per_query:
            raise RPGQueryValidationError(
                f"requested {n_units} units, above max_units_per_query={self.max_units_per_query}"
            )
        try:
            self.simulator.validate_query(parsed)
        except Exception as exc:
            raise RPGQueryValidationError(str(exc)) from exc

    def _validate_budget(self, parsed: StaticRPGParsedQuery) -> None:
        if self.max_queries is not None and self._successful_query_count >= self.max_queries:
            raise RPGQueryValidationError(
                f"query budget exhausted: {self._successful_query_count}/{self.max_queries} successful queries used"
            )
        usage = self.simulator.estimate_sample_usage(parsed)
        units = usage["cells"] if self.sample_accounting == "cells" else usage["rows"]
        query_units = 1 if parsed.mode == "inspect_unit" else max(1, int(parsed.n_units))
        units_per_requested_unit = max(1, int((units + query_units - 1) // query_units))
        if self.max_samples_per_query is not None and units > self.max_samples_per_query:
            max_units = max(0, int(self.max_samples_per_query // units_per_requested_unit))
            raise RPGQueryValidationError(
                f"query requests {units} {self.sample_accounting}, above per-query limit "
                f"{self.max_samples_per_query}. With this query shape, max affordable n_units "
                f"for the per-query limit is {max_units}."
            )
        if self.max_total_samples is not None:
            used = self._sample_cells_used if self.sample_accounting == "cells" else self._sample_rows_used
            if used + units > self.max_total_samples:
                remaining = max(0, self.max_total_samples - used)
                max_units = max(0, int(remaining // units_per_requested_unit))
                raise RPGQueryValidationError(
                    f"query would use {used + units}/{self.max_total_samples} {self.sample_accounting}. "
                    f"Currently used {used}; remaining {remaining}; this query would cost {units}. "
                    f"With this query shape, max affordable n_units is {max_units}."
                )

    def _record_usage(self, result: StaticRPGQueryResult) -> None:
        self._sample_rows_used += int(result.sample_rows)
        self._sample_cells_used += int(result.sample_cells)
        result.sample_units = result.sample_cells if self.sample_accounting == "cells" else result.sample_rows

    def _save_result_csv(self, result: StaticRPGQueryResult) -> None:
        if result.dataframe is None:
            return
        out = Path(self.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        world_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.simulator.meta.get("world_id", "static_rpg")))
        path = out / f"{world_id}_query_{self._query_counter:04d}.csv"
        if path.exists():
            suffix = 2
            while True:
                candidate = out / f"{world_id}_query_{self._query_counter:04d}_{suffix}.csv"
                if not candidate.exists():
                    path = candidate
                    break
                suffix += 1
        result.dataframe.to_csv(path, index=False)
        result.data_file = str(path)
        result.preview = result.dataframe.head(self.preview_rows).to_csv(index=False)
