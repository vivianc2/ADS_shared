#!/usr/bin/env python3
"""Audit generated ACED RPG worlds.

This script is intentionally stricter than the generator's basic
``--validate-only`` mode.  It checks dataset representation, public/private
schema separation, archetype-specific difficulty signatures, sample generation,
and optionally recomputes gold answers with fresh high-budget rollouts.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import world_gen_rpg as rpg  # noqa: E402


REQUIRED_SIGNATURES = {
    "delayed_harm": {
        "aggressive_looks_better_early",
        "aggressive_violates_safety",
        "safety_changes_choice",
        "gold_margin",
    },
    "dose_response": {
        "curve_shape_matches_subvariant",
        "best_dose_margin",
        "low_not_enough",
    },
    "proxy_metric_hacking": {
        "metric_hack_tempts_on_observed_metric",
        "metric_hack_not_latent_improvement",
        "metric_and_gold_differ",
        "gold_margin",
    },
    "latent_mediator": {
        "program_moves_true_mediator",
        "program_also_moves_decoy",
        "decoy_focus_not_pathway",
        "gold_margin",
    },
    "heterogeneous_response": {
        "adaptive_policy_best",
        "adaptive_beats_static",
        "subtype_a_prefers_a",
        "subtype_b_prefers_c",
        "proxy_informative_not_perfect",
        "gold_margin",
    },
}


def _load_worlds(outdir: str) -> List[Tuple[str, Dict[str, Any]]]:
    paths = sorted(glob.glob(os.path.join(outdir, "world_*.json")))
    worlds = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            worlds.append((path, json.load(f)))
    return worlds


def _check_manifest(outdir: str, paths: List[str]) -> Dict[str, Any]:
    manifest_path = os.path.join(outdir, "manifest_rpg_v1.json")
    if not os.path.exists(manifest_path):
        return {"present": False, "issues": ["missing manifest_rpg_v1.json"]}
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    actual = {os.path.basename(p) for p in paths}
    listed = {os.path.basename(w["path"]) for w in manifest.get("worlds", [])}
    issues = []
    if actual - listed:
        issues.append(f"extra world files: {sorted(actual - listed)[:5]}")
    if listed - actual:
        issues.append(f"missing world files: {sorted(listed - actual)[:5]}")
    if manifest.get("generated") != len(paths):
        issues.append(f"manifest generated={manifest.get('generated')} but actual={len(paths)}")
    return {
        "present": True,
        "generated": manifest.get("generated"),
        "world_count": len(manifest.get("worlds", [])),
        "extra_world_files": sorted(actual - listed),
        "missing_world_files": sorted(listed - actual),
        "issues": issues,
    }


def _answer_from_fresh_scores(world: Dict[str, Any], scores: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    arch = world["meta"]["archetype"]
    cfg = world["hidden"]["simulator_config"]
    policies = cfg["policies"]
    stored = world["oracle"]["gold_answer"]

    if arch == "delayed_harm":
        allowed = stored["safety_constraint"]["allowed_violation_rate"]
        feasible = sorted(
            [s for s in scores if s["metrics"]["safety_violation_rate"] <= allowed],
            key=lambda s: s["target_mean"],
            reverse=True,
        )
        if not feasible:
            return None, None
        margin = None
        if len(feasible) > 1:
            margin = feasible[0]["target_mean"] - feasible[1]["target_mean"]
        return feasible[0]["policy_id"], margin

    if arch == "dose_response":
        ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
        return ranked[0]["policy_id"], ranked[0]["expected_utility"] - ranked[1]["expected_utility"]

    if arch == "proxy_metric_hacking":
        ranked = sorted(scores, key=lambda s: s["target_mean"], reverse=True)
        return ranked[0]["policy_id"], ranked[0]["target_mean"] - ranked[1]["target_mean"]

    if arch == "latent_mediator":
        def kind(name: str) -> Dict[str, Any]:
            return next(
                s for s in scores
                if next(p for p in policies if p["policy_id"] == s["policy_id"])["kind"] == name
            )

        base = kind("no_program")
        program = kind("program")
        decoy = kind("decoy_focus")
        program_effect = program["target_mean"] - base["target_mean"]
        decoy_effect = decoy["target_mean"] - base["target_mean"]
        answer = stored["variable_name"] if program_effect > decoy_effect else stored["decoy_variable_name"]
        return answer, program_effect - decoy_effect

    if arch == "heterogeneous_response":
        ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
        return ranked[0]["policy_id"], ranked[0]["expected_utility"] - ranked[1]["expected_utility"]

    return None, None


def _policy_by_id(cfg: Dict[str, Any], policy_id: str) -> Dict[str, Any]:
    for policy in cfg["policies"]:
        if policy["policy_id"] == policy_id:
            return policy
    raise KeyError(policy_id)


def audit_world(path: str, world: Dict[str, Any], sample_units: int) -> Dict[str, Any]:
    issues: List[str] = []
    meta = world.get("meta", {})
    arch = meta.get("archetype")
    cfg = world.get("hidden", {}).get("simulator_config", {})
    visibility = world.get("variable_visibility", {})
    visible = world.get("visible", {})

    public_observed = set(visibility.get("agent_visible_observed", []))
    public_actions = set(visibility.get("agent_visible_actions", []))
    public_names = public_observed | public_actions
    hidden_names = set(visibility.get("hidden_latent", []))
    top_vars = {v.get("name") for v in world.get("variables", [])}
    allowed_measurements = set(visible.get("allowed_measurements", []))
    allowed_policies = {p.get("policy_id") for p in visible.get("allowed_policies", [])}
    cfg_policy_ids = {p.get("policy_id") for p in cfg.get("policies", [])}

    if world.get("schema_version") != rpg.SCHEMA_VERSION:
        issues.append("wrong schema_version")
    if not world.get("validators", {}).get("accepted"):
        issues.append("validators.accepted is false")
    if top_vars != public_names:
        issues.append("top-level variables do not exactly match visible observed/actions")
    if hidden_names & public_names:
        issues.append("hidden names overlap public names")
    if hidden_names & top_vars:
        issues.append("hidden names leak into top-level variables")
    if world.get("edges"):
        issues.append("top-level edges should be empty for RPG worlds")
    if allowed_measurements != public_observed:
        issues.append("allowed_measurements does not match public observed variables")
    if allowed_policies != cfg_policy_ids:
        issues.append("visible allowed policies do not match simulator policies")
    if visible.get("default_observational_policy_id") not in cfg_policy_ids:
        issues.append("default_observational_policy_id missing or invalid")
    budget = visible.get("experiment_budget", {})
    if budget.get("sample_accounting") != "cells":
        issues.append("experiment_budget should use cell accounting")
    for key in ["max_total_samples", "max_samples_per_query", "default_units", "max_units", "max_queries"]:
        if not isinstance(budget.get(key), int) or budget.get(key) <= 0:
            issues.append(f"experiment_budget.{key} missing or nonpositive")
    if budget.get("max_samples_per_query", 0) > budget.get("max_total_samples", 0):
        issues.append("per-query budget exceeds total budget")
    protocol = visible.get("discovery_protocol", {})
    if protocol.get("task_style") != "budgeted_iterative_scientific_discovery":
        issues.append("missing budgeted discovery protocol")
    if len(public_observed) < 9:
        issues.append("too few visible observed measurements")
    if len(hidden_names) < 2:
        issues.append("too few hidden latent/state variables")
    if len(cfg_policy_ids) < 3:
        issues.append("too few candidate policies")

    question = (world.get("questions") or [{}])[0]
    answer = question.get("answer")
    gold = world.get("oracle", {}).get("gold_answer", {})
    if answer not in {gold.get("policy_id"), gold.get("variable_name")}:
        issues.append("question answer does not match oracle gold")
    if "Answer with" not in question.get("question", ""):
        issues.append("question does not state answer format")

    checks = {c.get("name"): c for c in world.get("validators", {}).get("signature_checks", [])}
    missing = REQUIRED_SIGNATURES.get(arch, set()) - set(checks)
    if missing:
        issues.append(f"missing required signature checks: {sorted(missing)}")
    failed = [name for name, check in checks.items() if not check.get("passed")]
    if failed:
        issues.append(f"failed signature checks: {failed}")

    if sample_units > 0:
        measurements = list(visible.get("allowed_measurements", []))
        horizon = int(cfg.get("horizon", meta.get("horizon", 0)))
        for i, policy_id in enumerate(sorted(cfg_policy_ids)):
            policy = _policy_by_id(cfg, policy_id)
            rollout = rpg.rollout(
                cfg,
                policy,
                sample_units,
                901000 + meta.get("seed", 0) + i * 1009,
                return_rows=True,
                measurements=measurements,
            )
            rows = rollout.get("rows", [])
            if len(rows) != sample_units * horizon:
                issues.append(f"sample row count mismatch for {policy_id}")
                continue
            for row in rows[:3] + rows[-3:]:
                row_names = set(row)
                if hidden_names & row_names:
                    issues.append(f"hidden names leaked into sample rows for {policy_id}")
                if not public_actions <= row_names:
                    issues.append(f"action columns missing from sample rows for {policy_id}")
                missing_measurements = allowed_measurements - row_names
                if missing_measurements:
                    issues.append(f"measurement columns missing from sample rows for {policy_id}")
                for key, value in row.items():
                    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                        issues.append(f"bad float in sample row {key}")

    return {
        "path": path,
        "world_id": meta.get("world_id"),
        "archetype": arch,
        "sub_variant": meta.get("sub_variant"),
        "subdomain": meta.get("subdomain"),
        "answer": answer,
        "gold_margin": world.get("oracle", {}).get("gold_margin"),
        "n_visible_observed": len(public_observed),
        "n_visible_actions": len(public_actions),
        "n_hidden": len(hidden_names),
        "n_policies": len(cfg_policy_ids),
        "issues": sorted(set(issues)),
    }


def audit_dataset(outdir: str, sample_units: int, recheck_rollouts: int) -> Dict[str, Any]:
    worlds = _load_worlds(outdir)
    paths = [p for p, _ in worlds]
    reports = [audit_world(path, world, sample_units) for path, world in worlds]

    by_arch = collections.Counter(r["archetype"] for r in reports)
    by_subvariant = collections.Counter((r["archetype"], r["sub_variant"]) for r in reports)
    by_template = collections.Counter((r["archetype"], r["subdomain"]) for r in reports)
    answers = collections.defaultdict(collections.Counter)
    margins = collections.defaultdict(list)
    for report in reports:
        answers[report["archetype"]][report["answer"]] += 1
        margin = report["gold_margin"]
        if isinstance(margin, (int, float)):
            margins[report["archetype"]].append(float(margin))

    representation_issues = []
    if dict(by_arch) != rpg.DEFAULT_DISTRIBUTION:
        representation_issues.append(f"archetype distribution mismatch: {dict(by_arch)}")
    expected_subs = {
        ("delayed_harm", "safety_constrained_long_horizon"): 12,
        ("dose_response", "inverted_u"): 4,
        ("dose_response", "minimum_effective"): 4,
        ("dose_response", "saturation"): 4,
        ("proxy_metric_hacking", "metric_hacking"): 12,
        ("latent_mediator", "mediated_only"): 6,
        ("latent_mediator", "direct_and_mediated"): 6,
        ("heterogeneous_response", "observed_proxy_policy"): 12,
    }
    if dict(by_subvariant) != expected_subs:
        representation_issues.append(f"subvariant distribution mismatch: {dict(by_subvariant)}")
    for arch in rpg.ARCHETYPES:
        counts = [count for (a, _), count in by_template.items() if a == arch]
        if sorted(counts) != [3, 3, 3, 3]:
            representation_issues.append(f"template imbalance for {arch}: {counts}")

    recheck_issues: List[Dict[str, Any]] = []
    recheck_summary: Dict[str, Any] = {"enabled": recheck_rollouts > 0}
    if recheck_rollouts > 0:
        t0 = time.time()
        fresh_margins = collections.defaultdict(list)
        for idx, (path, world) in enumerate(worlds, 1):
            cfg = world["hidden"]["simulator_config"]
            scores = rpg.evaluate_policies(
                cfg,
                cfg["policies"],
                recheck_rollouts,
                1700000 + idx * 1009,
            )
            recomputed, margin = _answer_from_fresh_scores(world, scores)
            answer = world["questions"][0]["answer"]
            if recomputed != answer or (margin is not None and margin < 1.0):
                recheck_issues.append({
                    "file": os.path.basename(path),
                    "archetype": world["meta"]["archetype"],
                    "stored": answer,
                    "recomputed": recomputed,
                    "fresh_margin": margin,
                })
            if margin is not None:
                fresh_margins[world["meta"]["archetype"]].append(float(margin))
            if idx % 20 == 0:
                print(f"fresh oracle recheck {idx}/{len(worlds)} elapsed={time.time() - t0:.1f}s")
        recheck_summary = {
            "enabled": True,
            "rollouts": recheck_rollouts,
            "issues": recheck_issues,
            "margin_by_archetype": {
                arch: {
                    "min": min(vals),
                    "median": statistics.median(vals),
                    "max": max(vals),
                }
                for arch, vals in sorted(fresh_margins.items())
            },
        }

    failed_reports = [r for r in reports if r["issues"]]
    manifest = _check_manifest(outdir, paths)
    all_issues = (
        representation_issues
        + manifest.get("issues", [])
        + [f"{os.path.basename(r['path'])}: {r['issues']}" for r in failed_reports]
        + [f"fresh oracle issue: {i}" for i in recheck_issues]
    )
    return {
        "outdir": outdir,
        "n_worlds": len(worlds),
        "n_ok": len(worlds) - len(failed_reports),
        "manifest": manifest,
        "representation": {
            "by_archetype": dict(sorted(by_arch.items())),
            "by_subvariant": {str(k): v for k, v in sorted(by_subvariant.items())},
            "by_template": {str(k): v for k, v in sorted(by_template.items())},
            "issues": representation_issues,
        },
        "difficulty": {
            "margin_by_archetype": {
                arch: {
                    "min": min(vals),
                    "median": statistics.median(vals),
                    "max": max(vals),
                }
                for arch, vals in sorted(margins.items())
            },
            "answer_counts": {arch: dict(counter) for arch, counter in sorted(answers.items())},
            "required_signatures": {
                arch: sorted(names) for arch, names in sorted(REQUIRED_SIGNATURES.items())
            },
        },
        "sample_smoke_units_per_policy": sample_units,
        "fresh_oracle_recheck": recheck_summary,
        "reports": reports,
        "issues": all_issues,
        "ok": not all_issues,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit generated RPG worlds.")
    ap.add_argument(
        "outdir_positional",
        nargs="?",
        help="Optional dataset directory. Equivalent to --outdir.",
    )
    ap.add_argument("--outdir", default="dataset_generation_code/all_out_rpg/out_rpg_v1")
    ap.add_argument("--sample-units", type=int, default=7)
    ap.add_argument("--recheck-rollouts", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument(
        "--static",
        action="store_true",
        help="Run the static-rpg v2 audit instead of the v1 dynamic audit. "
             "Selected automatically if the first world in outdir has "
             "schema_version == 'rpg_static_v2'.",
    )
    ap.add_argument(
        "--recheck-oracle-n",
        type=int,
        default=20000,
        help="Static-only: units used for the fresh oracle recheck (0 to skip).",
    )
    args = ap.parse_args()
    if args.outdir_positional:
        args.outdir = args.outdir_positional

    # Auto-detect static if not explicitly set.
    if not args.static:
        try:
            paths = sorted(glob.glob(os.path.join(args.outdir, "world_*.json")))
            if paths:
                with open(paths[0], "r", encoding="utf-8") as f:
                    first = json.load(f)
                if first.get("schema_version") == "rpg_static_v2":
                    args.static = True
        except Exception:
            pass

    if args.static:
        report = audit_static_dataset(
            args.outdir,
            recheck_oracle_n=args.recheck_oracle_n,
        )
    else:
        report = audit_dataset(args.outdir, args.sample_units, args.recheck_rollouts)
    if args.summary_only:
        report = dict(report)
        report.pop("reports", None)
    print(json.dumps(report, indent=2, default=str))
    if not report["ok"]:
        raise SystemExit(1)


# ============================================================================
# Static (rpg_static_v2) audit
# ============================================================================

STATIC_LEAKAGE_TERMS = [
    "LatentBurden", "LatentDriver",
    "BurdenSubstrate", "BackgroundStrain",
    "HealthSeekingTrait", "HealthSeeking",
    "BaselineSeverity",
    "DecoyState",
    "LatentSeverity",
    "LatentSubtype",
    "BaselineRisk",
    "OperationQuality",
    "decoy",
    "red herring",
    "true lever",
]


def _static_visible_text(world: Dict[str, Any]) -> str:
    """Concatenate every piece of agent-visible text into one lowercased blob
    so substring leakage checks are simple."""
    visible = world.get("visible", {})
    parts = [
        world.get("story", ""),
        visible.get("story", ""),
        visible.get("question", ""),
    ]
    for v in visible.get("observed_variables", []) or []:
        parts.append(v.get("description", ""))
    for v in visible.get("intervenable_variables", []) or []:
        parts.append(v.get("description", ""))
    for q in world.get("questions", []) or []:
        parts.append(q.get("question", ""))
    return "\n".join(p for p in parts if p).lower()


def audit_static_world(path: str, world: Dict[str, Any], *, recheck_oracle_n: int) -> Dict[str, Any]:
    """Run the static-rpg-v2 audit on a single world."""
    issues: List[str] = []
    checks: Dict[str, Any] = {}

    # 1. Schema check.
    schema = world.get("schema_version")
    checks["schema_version"] = schema
    if schema != "rpg_static_v2":
        issues.append(f"schema_version={schema!r} (expected 'rpg_static_v2')")

    # 2. Required top-level blocks.
    for key in ("meta", "visible", "hidden", "oracle", "validators", "questions"):
        if key not in world:
            issues.append(f"missing top-level block {key!r}")
    if issues:
        return {"path": path, "ok": False, "issues": issues, "checks": checks}

    arch = world["meta"].get("archetype")
    answer_schema = world.get("visible", {}).get("answer_schema", "intervention_with_hypothesis")
    checks["archetype"] = arch
    checks["answer_schema"] = answer_schema

    # 3. Hidden-name leakage in visible text.
    visible_text = _static_visible_text(world)
    leaks_found = [t for t in STATIC_LEAKAGE_TERMS if t.lower() in visible_text]
    checks["leakage_terms_found"] = leaks_found
    if leaks_found:
        issues.append(f"leakage terms in visible text: {leaks_found}")

    # 4. Public/hidden split: variable names in `visible.observed_variables`
    #    must NOT include any hidden-state name.
    obs_var_names = [v.get("name") for v in world["visible"].get("observed_variables", [])]
    intv_var_names = [v.get("name") for v in world["visible"].get("intervenable_variables", [])]
    checks["n_observed_variables"] = len(obs_var_names)
    checks["n_intervenable_variables"] = len(intv_var_names)
    forbidden_hidden_names = {
        "LatentBurden", "BurdenSubstrate", "BaselineSeverity",
        "HealthSeekingTrait", "DecoyState_A", "DecoyState_B",
        "LatentSeverity", "LatentSubtype", "BaselineRisk", "OperationQuality",
        "LatentRegime", "LatentRegimeAxis", "ExposureHistory", "SitePractice",
    }
    leaked_names = set(obs_var_names + intv_var_names) & forbidden_hidden_names
    if leaked_names:
        issues.append(f"hidden variable names appear in visible block: {sorted(leaked_names)}")

    # 5. Validators block: every signature check passed.
    sig_checks = world["validators"].get("signature_checks", []) or []
    failed_sig = [c for c in sig_checks if not c.get("passed")]
    checks["signature_check_count"] = len(sig_checks)
    checks["failed_signature_checks"] = [c.get("name") for c in failed_sig]
    if failed_sig:
        issues.append(f"signature checks failed: {[c.get('name') for c in failed_sig]}")

    # 6. Recoverability band on-spec.
    band = world["hidden"].get("diagnostics", {}).get("recoverability_band", {})
    small = band.get("small_budget_hit_rate")
    medium = band.get("medium_budget_hit_rate")
    checks["recoverability_small"] = small
    checks["recoverability_medium"] = medium
    skip_recoverability_thresholds = {
        "hidden_subtype",
        "negative_control",
        "anomaly_discovery",
        "latent_regime_discovery",
    }
    if arch not in skip_recoverability_thresholds and small is not None and small > 0.40:
        issues.append(f"recoverability_small={small} > 0.40")
    if arch not in skip_recoverability_thresholds and medium is not None and medium < 0.70:
        issues.append(f"recoverability_medium={medium} < 0.70")

    # 7. Proxy-correlation calibration for hidden_cause.
    if arch == "hidden_cause":
        pc = world["hidden"].get("diagnostics", {}).get("observational", {}).get("proxy_target_corr_obs", {})
        checks["proxy_corr_true"] = pc.get("latent_driver_proxy")
        checks["proxy_corr_decoy_a"] = pc.get("decoy_proxy_a")
        checks["proxy_corr_decoy_b"] = pc.get("decoy_proxy_b")
        true_pc = pc.get("latent_driver_proxy")
        if true_pc is not None and abs(true_pc) > 0.80:
            issues.append(f"true-proxy obs correlation {true_pc} > 0.80 — proxy is too clean")
        max_decoy = max(
            abs(pc.get("decoy_proxy_a", 0.0) or 0.0),
            abs(pc.get("decoy_proxy_b", 0.0) or 0.0),
        )
        if max_decoy < 0.15:
            issues.append(f"max decoy proxy obs correlation {max_decoy} < 0.15 — decoys not tempting enough")

    if arch == "latent_regime_discovery":
        gold_answer = world.get("oracle", {}).get("gold_answer", {})
        latent_structure = gold_answer.get("latent_structure", {})
        checks["latent_regime_answer_schema"] = gold_answer.get("answer_schema")
        checks["latent_regime_gold_n_regimes"] = latent_structure.get("n_regimes")
        checks["latent_regime_gain_over_static"] = gold_answer.get("conditional_gain_over_static")
        if gold_answer.get("answer_schema") != "latent_regime_policy":
            issues.append("latent_regime_discovery gold answer_schema is not latent_regime_policy")
        if int(latent_structure.get("n_regimes", -1)) != 2:
            issues.append("latent_regime_discovery gold n_regimes is not 2")
        gain = gold_answer.get("conditional_gain_over_static")
        if gain is not None and float(gain) < 4.0:
            issues.append(f"latent_regime conditional gain {gain} < 4.0")

    # 8. Fresh oracle recheck (optional).
    if recheck_oracle_n > 0:
        try:
            cfg = world["hidden"]["simulator_config"]
            cfg2 = {
                "archetype": cfg["archetype"],
                "template": cfg["template"],
                "parameters": cfg["parameters"],
                "mixture_weight": cfg["mixture_weight"],
                "seed": int(cfg["world_seed"]),
            }
            if answer_schema == "intervention_with_hypothesis":
                fresh_scores = rpg._static_oracle_score(  # noqa: SLF001
                    cfg2, n_oracle=recheck_oracle_n, seed=int(cfg["world_seed"]) + 991919,
                )
                fresh_ranked = sorted(fresh_scores, key=lambda s: s["expected_utility"], reverse=True)
                fresh_gold_key = fresh_ranked[0]["intervention_key"]
                stored_gold_key = world["oracle"]["gold_answer"]["intervention_key"]
                checks["fresh_gold_key"] = fresh_gold_key
                checks["stored_gold_key"] = stored_gold_key
                if fresh_gold_key != stored_gold_key:
                    issues.append(
                        f"fresh oracle gold {fresh_gold_key!r} != stored gold {stored_gold_key!r}"
                    )
            else:
                checks["fresh_oracle_recheck_skipped"] = answer_schema
        except Exception as e:
            issues.append(f"fresh oracle recheck failed: {e}")

    # 9. Gold-position record (for dataset-level balance reporting).
    gold_iv = world["oracle"]["gold_answer"].get("intervention", {})
    if gold_iv:
        gold_knob = sorted(gold_iv.keys())[0]
        intv_names_in_order = [v.get("name") for v in world["visible"]["intervenable_variables"]]
        try:
            checks["gold_knob_position"] = intv_names_in_order.index(gold_knob)
        except ValueError:
            issues.append(f"gold knob {gold_knob!r} not in visible.intervenable_variables")
            checks["gold_knob_position"] = -1
    else:
        checks["gold_knob_position"] = -1  # NoIntervention

    return {"path": path, "ok": not issues, "issues": issues, "checks": checks}


def audit_static_dataset(outdir: str, *, recheck_oracle_n: int) -> Dict[str, Any]:
    """Run the static-rpg-v2 audit across every world in `outdir`."""
    paths = sorted(glob.glob(os.path.join(outdir, "world_*.json")))
    if not paths:
        return {"ok": False, "issues": [f"no worlds in {outdir}"], "n_worlds": 0}
    reports: List[Dict[str, Any]] = []
    pos_distribution: Dict[Tuple[str, int], int] = collections.defaultdict(int)
    arch_count: Dict[str, int] = collections.defaultdict(int)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            world = json.load(f)
        r = audit_static_world(path, world, recheck_oracle_n=recheck_oracle_n)
        reports.append(r)
        arch = world["meta"].get("archetype")
        arch_count[arch] += 1
        pos = r["checks"].get("gold_knob_position")
        if isinstance(pos, int) and pos >= 0:
            pos_distribution[(arch, pos)] += 1

    n_ok = sum(1 for r in reports if r["ok"])
    summary = {
        "n_worlds": len(paths),
        "n_ok": n_ok,
        "n_failed": len(paths) - n_ok,
        "archetype_count": dict(arch_count),
        "gold_position_distribution": {
            f"{a}@pos{p}": c for (a, p), c in sorted(pos_distribution.items())
        },
    }
    return {"ok": n_ok == len(paths), "summary": summary, "reports": reports}


if __name__ == "__main__":
    main()
