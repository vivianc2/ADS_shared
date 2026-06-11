#!/usr/bin/env python3
"""Evaluate completed static RPG agent runs.

This evaluator is intentionally lightweight: RPG agent runs already store a
structured final answer and an oracle score.  The script can optionally rescore
each answer against the snapshotted world JSON, then reports aggregate accuracy,
query usage, invalid-query patterns, and per-world details.

Example:
    python3 framework_code/evaluate_rpg.py \
      results/rpg_static_all6_opus/rpg_static_all6_opus.json \
      -o framework_code/evaluations/rpg/eval_rpg_static_all6_opus.json \
      --rescore --details
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_text: Optional[str], *, run_path: Path) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        Path.cwd() / path,
        run_path.parent / path,
        ROOT / path,
    ]
    return next((p for p in candidates if p.exists()), path)


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return statistics.mean(vals)


def median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return statistics.median(vals)


def accuracy(n_accepted: int, n_total: int) -> Optional[float]:
    if n_total == 0:
        return None
    return n_accepted / n_total


def read_log_summary(row: Dict[str, Any], *, run_path: Path, include_trace: bool) -> Dict[str, Any]:
    log_path = resolve_path(row.get("log_path"), run_path=run_path)
    if not log_path or not log_path.exists():
        return {
            "log_found": False,
            "query_count": int(row.get("total_queries") or 0),
            "invalid_query_count": None,
            "invalid_query_errors": [],
            "query_trace_for_eval": [],
            "query_trace": [] if include_trace else None,
        }

    log = load_json(log_path)
    turns = log.get("turns") or []
    query_turns = [t for t in turns if t.get("action_type") == "query"]
    invalid_errors: List[str] = []
    query_trace: List[Dict[str, Any]] = []
    for turn in query_turns:
        result = turn.get("query_result") or {}
        query = result.get("query") or {}
        success = bool(result.get("success"))
        if not success:
            invalid_errors.append(str(result.get("error_message") or "unknown query failure"))
        query_trace.append(
            {
                "turn_index": turn.get("turn_index"),
                "success": success,
                "mode": query.get("mode"),
                "n_units": query.get("n_units"),
                "measurements": query.get("measurements"),
                "intervention": query.get("intervention") or {},
                "error": result.get("error_message"),
            }
        )

    return {
        "log_found": True,
        "query_count": len(query_turns),
        "invalid_query_count": len(invalid_errors),
        "invalid_query_errors": invalid_errors,
        "query_trace_for_eval": query_trace,
        "query_trace": query_trace if include_trace else None,
    }


def score_with_world(row: Dict[str, Any], *, run_path: Path, n_score: Optional[int] = None) -> Dict[str, Any]:
    world_path = resolve_path(row.get("world_file"), run_path=run_path)
    if not world_path or not world_path.exists():
        return {
            "success": False,
            "accepted": False,
            "error": f"missing world_file: {row.get('world_file')}",
        }
    try:
        from simulator_rpg import StaticRPGSimulator

        simulator = StaticRPGSimulator.from_json(str(world_path))
        score = simulator.score_answer(row.get("extracted_answer"), n_score=n_score)
        score.setdefault("answer_schema", simulator.visible.get("answer_schema"))
        return score
    except Exception as exc:
        return {"success": False, "accepted": False, "error": str(exc)}


def answer_schema(row: Dict[str, Any], score: Dict[str, Any]) -> str:
    if score.get("answer_schema"):
        return str(score["answer_schema"])
    ground_truth = row.get("ground_truth") or {}
    if isinstance(ground_truth, dict) and ground_truth.get("answer_schema"):
        return str(ground_truth["answer_schema"])
    question_type = str(row.get("question_type") or "")
    if "latent_regime" in question_type or row.get("archetype") == "latent_regime_discovery":
        return "latent_regime_policy"
    if "story_hidden_cause" in question_type or row.get("archetype") == "story_hidden_cause_discovery":
        return "latent_cause_hypothesis"
    if "conditional" in question_type or row.get("archetype") == "hidden_subtype":
        return "conditional_policy"
    if "anomaly" in question_type or row.get("archetype") == "anomaly_discovery":
        return "anomaly_identification"
    return "intervention_with_hypothesis"


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _branch_matches_gold(score: Dict[str, Any]) -> Optional[bool]:
    policy = score.get("policy") or {}
    gold = score.get("gold_policy") or {}
    if not isinstance(policy, dict) or not isinstance(gold, dict):
        return None
    if not policy.get("branch_variable") or not gold.get("branch_variable"):
        return None
    return str(policy.get("branch_variable")) == str(gold.get("branch_variable"))


def _actions_match_gold(score: Dict[str, Any]) -> Optional[bool]:
    policy = score.get("policy") or {}
    gold = score.get("gold_policy") or {}
    if not isinstance(policy, dict) or not isinstance(gold, dict):
        return None
    keys = ("if_above", "if_below")
    if not all(isinstance(policy.get(k), dict) and isinstance(gold.get(k), dict) for k in keys):
        return None
    return all(policy.get(k) == gold.get(k) for k in keys)


def _utility_ok(score: Dict[str, Any]) -> Optional[bool]:
    expected = _as_float(score.get("expected_utility"))
    gold = _as_float(score.get("gold_expected_utility"))
    tolerance = _as_float(score.get("oracle_tolerance")) or 0.0
    if expected is None or gold is None:
        return None
    return expected >= gold - tolerance


def failure_bucket(row: Dict[str, Any], score: Dict[str, Any], log_summary: Dict[str, Any]) -> str:
    if bool(score.get("accepted")):
        return "accepted"
    schema = answer_schema(row, score)
    if not bool(score.get("success", True)):
        if schema == "latent_cause_hypothesis":
            return "latent_cause_parse_or_schema_error"
        if schema == "latent_regime_policy":
            return "latent_parse_or_schema_error"
        if schema == "conditional_policy":
            return "conditional_parse_or_schema_error"
        if schema == "anomaly_identification":
            return "anomaly_parse_or_schema_error"
        return "score_error"

    if schema == "anomaly_identification":
        precision = score.get("precision")
        recall = score.get("recall")
        p_thr = score.get("precision_threshold", 0.7)
        r_thr = score.get("recall_threshold", 0.6)
        if precision is None or recall is None:
            return "anomaly_rule_parse_or_missing_metric"
        if float(precision) < float(p_thr) and float(recall) < float(r_thr):
            return "anomaly_low_precision_and_recall"
        if float(precision) < float(p_thr):
            return "anomaly_low_precision"
        if float(recall) < float(r_thr):
            return "anomaly_low_recall"
        return "anomaly_rejected_other"

    if schema == "conditional_policy":
        return "conditional_policy_suboptimal"

    if schema == "latent_regime_policy":
        regime_ok = _as_bool(score.get("regime_count_correct"))
        if regime_ok is False:
            return "latent_wrong_regime_count"
        if log_summary.get("invalid_query_count"):
            return "latent_policy_suboptimal_with_invalid_queries"
        utility_ok = _utility_ok(score)
        if utility_ok is False:
            branch_match = _branch_matches_gold(score)
            action_match = _actions_match_gold(score)
            if branch_match is False:
                return "latent_wrong_or_weak_branch_proxy"
            if action_match is False:
                return "latent_wrong_or_weak_actions"
            return "latent_threshold_or_policy_suboptimal"
        return "latent_rejected_other"

    if schema == "latent_cause_hypothesis":
        trajectory = score.get("trajectory_evidence") or {}
        if score.get("accepted_without_trajectory") or trajectory.get("mechanism_verification_query") is False:
            return "latent_cause_no_mechanism_verification_query"
        if not bool(score.get("cause_match")):
            return "latent_cause_missing_or_surface_proxy"
        if int(score.get("evidence_count") or 0) < 3:
            return "latent_cause_weak_evidence"
        if int(score.get("alternatives_count") or 0) < 1:
            return "latent_cause_no_alternative_ruled_out"
        if not bool(score.get("verification_match")):
            return "latent_cause_no_decisive_test"
        if not bool(score.get("action_match")):
            return "latent_cause_wrong_action"
        return "latent_cause_rejected_other"

    if log_summary.get("invalid_query_count"):
        return "suboptimal_intervention_with_invalid_queries"
    return "suboptimal_intervention"


def row_eval_diagnostics(score: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    for key in (
        "expected_utility",
        "gold_expected_utility",
        "utility_gap_from_gold",
        "oracle_tolerance",
        "precision",
        "recall",
        "n_regimes",
    ):
        if key in score:
            diagnostics[key] = score.get(key)
    for key in (
        "hypothesis_present",
        "evidence_present",
        "regime_count_correct",
        "cause_match",
        "semantic_alias_match",
        "mechanism_match",
        "verification_match",
        "action_match",
    ):
        if key in score:
            diagnostics[key] = score.get(key)
    for key in ("evidence_count", "alternatives_count", "evidence_hits", "alternative_hits"):
        if key in score:
            diagnostics[key] = score.get(key)
    if "trajectory_evidence" in score:
        diagnostics["trajectory_evidence"] = score.get("trajectory_evidence")
    branch_match = _branch_matches_gold(score)
    if branch_match is not None:
        diagnostics["branch_variable_matches_gold"] = branch_match
    actions_match = _actions_match_gold(score)
    if actions_match is not None:
        diagnostics["branch_actions_match_gold"] = actions_match
    utility_ok = _utility_ok(score)
    if utility_ok is not None:
        diagnostics["utility_within_oracle_tolerance"] = utility_ok
    return diagnostics


def latent_cause_trace_evidence(log_summary: Dict[str, Any], score: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    requirements = (score or {}).get("trajectory_requirements") or {}
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
    for turn in log_summary.get("query_trace_for_eval") or log_summary.get("query_trace") or []:
        if not turn.get("success"):
            continue
        measurements = set(turn.get("measurements") or [])
        intervention = dict(turn.get("intervention") or {})
        if measurements & mechanism_measurements:
            saw_mechanism_measurement = True
        if (measurements & context_measurements) and outcome_measurement in measurements:
            saw_context_and_outcome = True
        if (
            turn.get("mode") == "interventional_sample"
            and (set(intervention) & targeted_actions)
            and (measurements & mechanism_measurements)
        ):
            mechanism_verification_query = True
    return {
        "saw_mechanism_measurement": saw_mechanism_measurement,
        "saw_context_and_outcome": saw_context_and_outcome,
        "mechanism_verification_query": mechanism_verification_query,
    }


def apply_trajectory_requirements(
    row: Dict[str, Any],
    score: Dict[str, Any],
    log_summary: Dict[str, Any],
) -> Dict[str, Any]:
    schema = answer_schema(row, score)
    if schema != "latent_cause_hypothesis":
        return score
    adjusted = dict(score)
    trace_evidence = latent_cause_trace_evidence(log_summary, adjusted)
    adjusted["trajectory_evidence"] = trace_evidence
    if bool(adjusted.get("accepted")) and not trace_evidence["mechanism_verification_query"]:
        adjusted["accepted_without_trajectory"] = True
        adjusted["accepted"] = False
    return adjusted


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_accepted = sum(1 for r in rows if r["accepted"])
    query_counts = [r["total_queries"] for r in rows if r.get("total_queries") is not None]
    cell_counts = [
        r["resource_usage"].get("sample_cells_used")
        for r in rows
        if isinstance(r.get("resource_usage"), dict)
        and r["resource_usage"].get("sample_cells_used") is not None
    ]
    invalid_counts = [
        r["log_summary"].get("invalid_query_count")
        for r in rows
        if r.get("log_summary", {}).get("invalid_query_count") is not None
    ]
    utility_gaps = [
        gap
        for r in rows
        for gap in [_as_float((r.get("score") or {}).get("utility_gap_from_gold"))]
        if gap is not None
    ]
    expected_utils = [
        val
        for r in rows
        for val in [_as_float((r.get("score") or {}).get("expected_utility"))]
        if val is not None
    ]
    gold_utils = [
        val
        for r in rows
        for val in [_as_float((r.get("score") or {}).get("gold_expected_utility"))]
        if val is not None
    ]
    utility_ok_values = [
        val
        for r in rows
        for val in [_utility_ok(r.get("score") or {})]
        if val is not None
    ]
    regime_ok_values = [
        val
        for r in rows
        for val in [_as_bool((r.get("score") or {}).get("regime_count_correct"))]
        if val is not None
    ]
    evidence_values = [
        val
        for r in rows
        for val in [_as_bool((r.get("score") or {}).get("evidence_present"))]
        if val is not None
    ]
    hypothesis_values = [
        val
        for r in rows
        for val in [_as_bool((r.get("score") or {}).get("hypothesis_present"))]
        if val is not None
    ]
    branch_match_values = [
        val
        for r in rows
        for val in [_branch_matches_gold(r.get("score") or {})]
        if val is not None
    ]
    action_match_values = [
        val
        for r in rows
        for val in [_actions_match_gold(r.get("score") or {})]
        if val is not None
    ]
    buckets = Counter(r["failure_bucket"] for r in rows)
    return {
        "n": n,
        "accepted": n_accepted,
        "accuracy": accuracy(n_accepted, n),
        "avg_queries": mean(query_counts),
        "median_queries": median(query_counts),
        "avg_sample_cells_used": mean(cell_counts),
        "avg_invalid_queries": mean(invalid_counts),
        "avg_expected_utility": mean(expected_utils),
        "avg_gold_expected_utility": mean(gold_utils),
        "avg_utility_gap_from_gold": mean(utility_gaps),
        "utility_within_oracle_tolerance_rate": mean(1.0 if v else 0.0 for v in utility_ok_values),
        "regime_count_correct_rate": mean(1.0 if v else 0.0 for v in regime_ok_values),
        "evidence_present_rate": mean(1.0 if v else 0.0 for v in evidence_values),
        "hypothesis_present_rate": mean(1.0 if v else 0.0 for v in hypothesis_values),
        "branch_variable_matches_gold_rate": mean(1.0 if v else 0.0 for v in branch_match_values),
        "branch_actions_match_gold_rate": mean(1.0 if v else 0.0 for v in action_match_values),
        "failure_buckets": dict(sorted(buckets.items())),
    }


def grouped_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "")].append(row)
    return {name: summarize_group(group_rows) for name, group_rows in sorted(groups.items())}


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    run_path = Path(args.run_json).expanduser().resolve()
    payload = load_json(run_path)
    raw_rows = list(payload.get("results") or [])

    rows: List[Dict[str, Any]] = []
    for raw in raw_rows:
        score = (
            score_with_world(raw, run_path=run_path, n_score=args.rescore_n)
            if args.rescore
            else dict(raw.get("score") or {})
        )
        log_summary = read_log_summary(raw, run_path=run_path, include_trace=args.include_traces)
        score = apply_trajectory_requirements(raw, score, log_summary)
        schema = answer_schema(raw, score)
        accepted = bool(score.get("accepted"))
        rows.append(
            {
                "world_name": raw.get("world_name"),
                "world_file": raw.get("world_file"),
                "source_world_file": raw.get("source_world_file"),
                "topic": raw.get("topic"),
                "archetype": raw.get("archetype"),
                "question_type": raw.get("question_type"),
                "answer_schema": schema,
                "accepted": accepted,
                "total_queries": raw.get("total_queries"),
                "max_queries": raw.get("max_queries"),
                "resource_usage": raw.get("resource_usage") or {},
                "score": score,
                "failure_bucket": failure_bucket(raw, score, log_summary),
                "eval_diagnostics": row_eval_diagnostics(score),
                "log_summary": log_summary,
                "extracted_answer": raw.get("extracted_answer"),
                "ground_truth": raw.get("ground_truth"),
            }
        )

    n = len(rows)
    n_accepted = sum(1 for r in rows if r["accepted"])
    eval_payload: Dict[str, Any] = {
        "run_json": str(run_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": payload.get("run_type"),
        "scientist_backend": payload.get("scientist_backend"),
        "scientist_model": payload.get("scientist_model"),
        "rescore": bool(args.rescore),
        "rescore_n": args.rescore_n if args.rescore else None,
        "n_worlds": n,
        "accepted": n_accepted,
        "accuracy": accuracy(n_accepted, n),
        "overall": summarize_group(rows),
        "by_archetype": grouped_summary(rows, "archetype"),
        "by_answer_schema": grouped_summary(rows, "answer_schema"),
        "by_topic": grouped_summary(rows, "topic"),
        "failure_buckets": dict(sorted(Counter(r["failure_bucket"] for r in rows).items())),
    }
    if args.details:
        eval_payload["details"] = rows
    return eval_payload


def print_console_summary(payload: Dict[str, Any]) -> None:
    print(
        f"RPG evaluation: {payload['accepted']}/{payload['n_worlds']} "
        f"accepted ({payload['accuracy']:.3f})"
    )
    print("\nBy archetype:")
    for archetype, summary in payload["by_archetype"].items():
        print(
            f"  {archetype}: {summary['accepted']}/{summary['n']} "
            f"({summary['accuracy']:.3f}), avg_queries={summary['avg_queries']:.2f}, "
            f"failures={summary['failure_buckets']}"
        )
    print("\nFailure buckets:")
    for name, count in payload["failure_buckets"].items():
        print(f"  {name}: {count}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a static RPG run JSON.")
    parser.add_argument("run_json", help="Path to run_agent_batch_rpg output JSON.")
    parser.add_argument("-o", "--output", default=None, help="Where to write evaluation JSON.")
    parser.add_argument("--rescore", action="store_true", help="Rescore final answers using snapshotted worlds.")
    parser.add_argument(
        "--rescore-n",
        type=int,
        default=None,
        help="Monte Carlo units per world for --rescore. Default uses each world's stored oracle_n_units.",
    )
    parser.add_argument("--details", action="store_true", help="Include per-world details in the evaluation JSON.")
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include compact per-query traces; implies --details.",
    )
    args = parser.parse_args(argv)
    if args.include_traces:
        args.details = True
    if args.rescore_n is not None and args.rescore_n <= 0:
        parser.error("--rescore-n must be a positive integer")

    payload = evaluate(args)
    print_console_summary(payload)

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
