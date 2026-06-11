"""Audit advanced-benchmark worlds by re-deriving gold answers from scratch.

Mirrors check_faithfulness.py's pattern: reads every world in a directory,
rebuilds the pgmpy BN, and checks that each stored advanced question's gold
answer matches what our generators would compute today.

Usage:
    python audit_advanced.py --indir ./out_bn_advanced
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from advanced_utils import (
    classify_mediator,
    compute_effect_table,
    expected_value_observational,
    expected_value_under_do,
    get_minimal_backdoor_adjustment_sets,
    get_non_intervenable_names,
    intervenable_var_names,
    is_valid_backdoor_set,
    rebuild_from_world,
    scoring_for_target,
)
from pgmpy.inference import CausalInference, VariableElimination


_FLOAT_EPS = 1e-5


def _find_var(world: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for v in world["variables"]:
        if v["name"] == name:
            return v
    return None


# ---------------------------------------------------------------------
# Per-type audits
# ---------------------------------------------------------------------

def _audit_budget_argmin(q, model, g, world, ci, ve) -> List[str]:
    errs: List[str] = []
    meta = q.get("metadata", {})
    target = meta.get("target")
    var = _find_var(world, target)
    if var is None:
        errs.append(f"target {target!r} not found")
        return errs
    scoring = scoring_for_target(var)
    baseline = expected_value_observational(ve, target, scoring)
    if abs(baseline - meta["baseline_expected"]) > _FLOAT_EPS:
        errs.append(
            f"baseline drift: stored={meta['baseline_expected']} "
            f"recomputed={baseline}"
        )
    effects = compute_effect_table(model, world, target, scoring)
    best = min(effects.items(), key=lambda kv: kv[1])
    (X, v), best_val = best
    ans = q["answer"]
    if abs(best_val - meta["best_expected"]) > _FLOAT_EPS:
        errs.append(
            f"best-expected drift: stored={meta['best_expected']} recomputed={best_val}"
        )
    # Verify stored action achieves the stored best_expected.
    stored_val = effects.get((ans["variable"], ans["value"]))
    if stored_val is None:
        errs.append(f"stored action {(ans['variable'], ans['value'])} not in effect table")
    elif abs(stored_val - meta["best_expected"]) > _FLOAT_EPS:
        errs.append(
            f"stored action effect drift: stored_ans={ans} "
            f"stored_exp={meta['best_expected']} recomputed={stored_val}"
        )
    return errs


def _audit_budget_satisfy(q, model, g, world, ci, ve) -> List[str]:
    errs: List[str] = []
    meta = q.get("metadata", {})
    target = meta.get("target")
    var = _find_var(world, target)
    if var is None:
        errs.append(f"target {target!r} not found")
        return errs
    scoring = scoring_for_target(var)
    baseline = expected_value_observational(ve, target, scoring)
    if abs(baseline - meta["baseline_expected"]) > _FLOAT_EPS:
        errs.append(f"baseline drift {baseline} vs {meta['baseline_expected']}")
    effects = compute_effect_table(model, world, target, scoring)
    improvements = {k: baseline - v for k, v in effects.items()}
    threshold = meta["threshold"]
    recomputed = {
        (X, v) for (X, v), imp in improvements.items() if imp >= threshold - _FLOAT_EPS
    }
    stored = {(item["variable"], item["value"]) for item in q["answer"]}
    if recomputed != stored:
        missing = recomputed - stored
        extra = stored - recomputed
        errs.append(f"feasible-set mismatch: missing={missing} extra={extra}")
    return errs


def _audit_side_effect(q, model, g, world, ci, ve) -> List[str]:
    errs: List[str] = []
    meta = q.get("metadata", {})
    target = meta.get("target")
    side = meta.get("side_effect")
    tvar = _find_var(world, target)
    svar = _find_var(world, side)
    if tvar is None or svar is None:
        errs.append("target or side-effect variable not found")
        return errs
    tscore = scoring_for_target(tvar)
    sscore = scoring_for_target(svar)
    t_base = expected_value_observational(ve, target, tscore)
    s_base = expected_value_observational(ve, side, sscore)
    if abs(t_base - meta["baseline_target"]) > _FLOAT_EPS:
        errs.append("baseline_target drift")
    if abs(s_base - meta["baseline_side"]) > _FLOAT_EPS:
        errs.append("baseline_side drift")
    t_effects = compute_effect_table(model, world, target, tscore)
    t_thresh = meta["target_threshold"]
    s_thresh = meta["side_threshold"]
    recomputed = set()
    for (X, v), te in t_effects.items():
        t_imp = t_base - te
        s_eff = expected_value_under_do(model, side, {X: v}, sscore)
        s_change = s_eff - s_base
        if t_imp >= t_thresh - _FLOAT_EPS and s_change <= s_thresh + _FLOAT_EPS:
            recomputed.add((X, v))
    stored = {(item["variable"], item["value"]) for item in q["answer"]}
    if recomputed != stored:
        errs.append(
            f"side-effect feasible-set mismatch: "
            f"missing={recomputed - stored} extra={stored - recomputed}"
        )
    return errs


def _audit_adjustment_set(q, model, g, world, ci, ve) -> List[str]:
    errs: List[str] = []
    meta = q.get("metadata", {})
    T = meta.get("treatment")
    O = meta.get("outcome")
    if T is None or O is None:
        errs.append("treatment or outcome missing from metadata")
        return errs
    recomputed = get_minimal_backdoor_adjustment_sets(ci, T, O)
    bucket = meta.get("answer_bucket")
    if bucket == "adj_unidentifiable":
        if recomputed != []:
            errs.append(
                f"stored unidentifiable but recomputed sets={[list(s) for s in recomputed]}"
            )
    elif bucket == "adj_none":
        if recomputed != [frozenset()]:
            errs.append(
                f"stored 'none' but recomputed sets={[list(s) for s in recomputed]}"
            )
    else:
        stored_sets = {frozenset(s) for s in meta.get("all_minimal_sets", [])}
        rec_sets = set(recomputed)
        if stored_sets != rec_sets:
            errs.append(
                f"minimal-set mismatch: missing={rec_sets - stored_sets} "
                f"extra={stored_sets - rec_sets}"
            )
    # Extra: every stored set must actually be valid.
    for s in meta.get("all_minimal_sets", []):
        if s and not is_valid_backdoor_set(ci, T, O, set(s)):
            errs.append(f"stored set {s} fails is_valid_backdoor_set")
    return errs


def _audit_mediator_class(q, model, g, world, ci, ve) -> List[str]:
    meta = q.get("metadata", {})
    T = meta.get("treatment"); M = meta.get("mediator"); O = meta.get("outcome")
    if None in (T, M, O):
        return ["mediator metadata incomplete"]
    recomputed = classify_mediator(g, T, M, O)
    if recomputed != q["answer"]:
        return [f"mediator class mismatch: stored={q['answer']} recomputed={recomputed}"]
    return []


def _audit_rank_topk(q, model, g, world, ci, ve) -> List[str]:
    errs: List[str] = []
    meta = q.get("metadata", {})
    target = meta.get("target")
    var = _find_var(world, target)
    if var is None:
        return [f"target {target!r} not found"]
    scoring = scoring_for_target(var)
    baseline = expected_value_observational(ve, target, scoring)
    if abs(baseline - meta["baseline_expected"]) > _FLOAT_EPS:
        errs.append("baseline drift")
    effects = compute_effect_table(model, world, target, scoring)
    sorted_actions = sorted(effects.items(), key=lambda kv: kv[1])
    K = meta.get("K", len(q["answer"]))
    topk = sorted_actions[: min(K, len(sorted_actions))]
    stored_order = [(item["variable"], item["value"]) for item in q["answer"]]
    recomputed_order = [(X, v) for (X, v), _ in topk]
    if stored_order != recomputed_order:
        errs.append(
            f"top-K order mismatch: stored={stored_order} recomputed={recomputed_order}"
        )
    return errs


_AUDITORS = {
    "advanced_budget_argmin":  _audit_budget_argmin,
    "advanced_budget_satisfy": _audit_budget_satisfy,
    "advanced_side_effect":    _audit_side_effect,
    "advanced_adjustment_set": _audit_adjustment_set,
    "advanced_mediator_class": _audit_mediator_class,
    "advanced_rank_topK":      _audit_rank_topk,
}


# ---------------------------------------------------------------------
# Intervenable check
# ---------------------------------------------------------------------

def _check_intervenable_respected(q, world) -> List[str]:
    non_int = get_non_intervenable_names(world)
    if not non_int:
        return []
    errs: List[str] = []
    qt = q["question_type"]
    meta = q.get("metadata", {})
    if qt == "advanced_budget_argmin":
        if q["answer"].get("variable") in non_int:
            errs.append(f"argmin action uses non-intervenable var: {q['answer']}")
    elif qt in ("advanced_budget_satisfy", "advanced_side_effect"):
        for item in q["answer"]:
            if item.get("variable") in non_int:
                errs.append(f"feasible action uses non-intervenable var: {item}")
    elif qt == "advanced_rank_topK":
        for item in q["answer"]:
            if item.get("variable") in non_int:
                errs.append(f"topK action uses non-intervenable var: {item}")
    return errs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def audit_world(world: Dict[str, Any]) -> Dict[str, Any]:
    model, g = rebuild_from_world(world)
    ci = CausalInference(model)
    ve = VariableElimination(model)
    issues: List[Dict[str, Any]] = []
    bucket_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}

    for q in world.get("questions", []):
        qt = q.get("question_type", "?")
        type_counts[qt] = type_counts.get(qt, 0) + 1
        bucket = q.get("metadata", {}).get("answer_bucket")
        if bucket:
            key = f"{qt}:{bucket}"
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
        auditor = _AUDITORS.get(qt)
        if auditor is None:
            continue
        errs = auditor(q, model, g, world, ci, ve)
        errs += _check_intervenable_respected(q, world)
        # preferred_low sanity
        meta = q.get("metadata", {})
        target_name = meta.get("target")
        if target_name is not None:
            tv = _find_var(world, target_name)
            if tv is None or tv.get("preferred_low") not in (True, False):
                errs.append(f"target {target_name!r} missing preferred_low")
        if errs:
            issues.append({
                "id": q.get("id"),
                "question_type": qt,
                "errors": errs,
            })

    return {
        "path": world.get("meta", {}).get("json_path"),
        "n_questions": len(world.get("questions", [])),
        "type_counts": type_counts,
        "bucket_counts": bucket_counts,
        "issues": issues,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True,
                    help="Directory containing world_*.json files")
    ap.add_argument("--output", type=str, default=None,
                    help="Optional path to write full JSON report")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, "world_*.json")))
    if not paths:
        print(f"No world_*.json files found in {args.indir}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    all_reports: List[Dict[str, Any]] = []
    total_issues = 0
    total_type: Dict[str, int] = {}
    total_bucket: Dict[str, int] = {}
    for i, p in enumerate(paths):
        with open(p, "r", encoding="utf-8") as f:
            world = json.load(f)
        world.setdefault("meta", {})["json_path"] = p
        r = audit_world(world)
        all_reports.append(r)
        total_issues += len(r["issues"])
        for k, v in r["type_counts"].items():
            total_type[k] = total_type.get(k, 0) + v
        for k, v in r["bucket_counts"].items():
            total_bucket[k] = total_bucket.get(k, 0) + v
        if (i + 1) % 5 == 0 or (i + 1) == len(paths):
            print(f"  [{i+1}/{len(paths)}] {os.path.basename(p)}  issues={len(r['issues'])}")

    dt = time.time() - t0
    print(f"\nAudit complete in {dt:.1f}s. "
          f"{len(paths)} worlds, {total_issues} issues.")
    print("\nQuestion type counts:")
    for k in sorted(total_type):
        print(f"  {k:28s} {total_type[k]}")
    print("\nBucket counts:")
    for k in sorted(total_bucket):
        print(f"  {k:60s} {total_bucket[k]}")
    if total_issues:
        print("\nIssue summary (first 20):")
        shown = 0
        for r in all_reports:
            for iss in r["issues"]:
                if shown >= 20:
                    break
                print(f"  {os.path.basename(r['path'] or '?')} "
                      f"qid={iss['id']} type={iss['question_type']}")
                for e in iss["errors"]:
                    print(f"    - {e}")
                shown += 1
            if shown >= 20:
                break

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "n_worlds": len(paths),
                "total_issues": total_issues,
                "type_counts": total_type,
                "bucket_counts": total_bucket,
                "worlds": all_reports,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nWrote report to {args.output}")

    sys.exit(0 if total_issues == 0 else 2)


if __name__ == "__main__":
    main()
