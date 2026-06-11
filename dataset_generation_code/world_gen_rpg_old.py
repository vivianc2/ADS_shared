#!/usr/bin/env python3
"""Generate ACED RPG simulator worlds.

RPG worlds are dynamic structural-causal simulator worlds.  They are not
static Bayesian networks and should not be passed through json_converter.py.

The design mirrors world_gen_advanced.py where it matters most:
semantic names and stories are separated from formal ground truth, while code
owns mechanisms, oracle rollouts, validators, and gold answers.

RPG v1 implements five archetypes:

  1. delayed_harm
  2. dose_response
  3. proxy_metric_hacking
  4. latent_mediator
  5. heterogeneous_response

Each output world is self-contained.  The future agent pipeline should expose
only the "visible" fields to scientists and use hidden.simulator_config for
trajectory queries.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Make bedrock_llm.py (in framework_code/) importable for the optional LLM
# polish / template-proposal hooks at the bottom of this file.
_RPG_FRAMEWORK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "framework_code"
)
if os.path.isdir(_RPG_FRAMEWORK_DIR) and _RPG_FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _RPG_FRAMEWORK_DIR)


SCHEMA_VERSION = "rpg_v1"
BENCHMARK_NAME = "aced_rpg_v1"

ARCHETYPES = [
    "delayed_harm",
    "dose_response",
    "proxy_metric_hacking",
    "latent_mediator",
    "heterogeneous_response",
]

DEFAULT_DISTRIBUTION: Dict[str, int] = {
    "delayed_harm": 12,
    "dose_response": 12,
    "proxy_metric_hacking": 12,
    "latent_mediator": 12,
    "heterogeneous_response": 12,
}

DOSE_SUB_VARIANTS = ["inverted_u", "minimum_effective", "saturation"]
LATENT_MEDIATOR_SUB_VARIANTS = ["mediated_only", "direct_and_mediated"]

DEFAULT_HORIZON = 8
DEFAULT_ORACLE_ROLLOUTS = 20000

# Agent-facing discovery budget. Oracle rollouts remain high-budget and
# private; scientist agents should operate under this visible budget.
DISCOVERY_PROTOCOL_VERSION = "rpg_discovery_v1"
DISCOVERY_SAMPLE_ACCOUNTING = "cells"
DISCOVERY_DEFAULT_UNITS = 40
DISCOVERY_MAX_UNITS = 400
DISCOVERY_MAX_SAMPLES_PER_QUERY = 8000
DISCOVERY_MAX_TOTAL_SAMPLES = 24000
DISCOVERY_MAX_QUERIES = 8

# Validator margins are in 0-100 score units unless otherwise noted.
MIN_GOLD_MARGIN = 1.5
MIN_CLEAR_EFFECT = 4.0
MIN_TEMPTATION_GAP = 3.0
MAX_ORACLE_SE_FRACTION = 0.50


# ---------------------------------------------------------------------------
# Scenario templates
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, List[Dict[str, Any]]] = {
    "delayed_harm": [
        {
            "topic": "Treatment effectiveness",
            "subdomain": "oncology treatment intensity",
            "setting": "a regional oncology network",
            "unit": "patients",
            "period": "week",
            "roles": {
                "burden_state": "DiseaseBurden",
                "harm_state": "TreatmentToxicity",
                "adherence_state": "MedicationAdherence",
                "target_state": "RecoveryLikelihood",
                "dose_action": "TreatmentDose",
                "support_action": "SupportiveCare",
                "burden_obs": "ImagingScore",
                "harm_obs": "AdverseEventReport",
                "target_obs": "RecoverySurvey",
                "adherence_obs": "AdherenceLog",
            },
        },
        {
            "topic": "Education",
            "subdomain": "accelerated exam preparation",
            "setting": "a university learning lab",
            "unit": "students",
            "period": "study block",
            "roles": {
                "burden_state": "KnowledgeGap",
                "harm_state": "BurnoutLoad",
                "adherence_state": "StudyPersistence",
                "target_state": "MasteryReadiness",
                "dose_action": "PracticeIntensity",
                "support_action": "MentorSupport",
                "burden_obs": "DiagnosticQuizGap",
                "harm_obs": "StressReport",
                "target_obs": "MasteryAssessment",
                "adherence_obs": "StudyLogCompletion",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "retention campaign pressure",
            "setting": "a subscription platform analytics team",
            "unit": "users",
            "period": "campaign week",
            "roles": {
                "burden_state": "ChurnPressure",
                "harm_state": "NotificationFatigue",
                "adherence_state": "AppEngagement",
                "target_state": "RetentionHealth",
                "dose_action": "MessageIntensity",
                "support_action": "QuietModeSupport",
                "burden_obs": "ChurnRiskScore",
                "harm_obs": "ComplaintSignal",
                "target_obs": "RetentionIndex",
                "adherence_obs": "SessionLogRate",
            },
        },
        {
            "topic": "Hospital data",
            "subdomain": "post-discharge follow-up intensity",
            "setting": "a hospital discharge planning group",
            "unit": "patients",
            "period": "post-discharge week",
            "roles": {
                "burden_state": "ReadmissionRisk",
                "harm_state": "CareBurden",
                "adherence_state": "FollowupParticipation",
                "target_state": "HomeRecoveryStatus",
                "dose_action": "FollowupIntensity",
                "support_action": "NavigationSupport",
                "burden_obs": "ReadmissionRiskScore",
                "harm_obs": "CareBurdenSurvey",
                "target_obs": "RecoveryStatusReport",
                "adherence_obs": "VisitCompletionLog",
            },
        },
    ],
    "dose_response": [
        {
            "topic": "Education",
            "subdomain": "weekly practice load",
            "setting": "a school district instructional research office",
            "unit": "students",
            "period": "week",
            "roles": {
                "capacity_state": "SkillGrowth",
                "strain_state": "PracticeFatigue",
                "target_state": "FinalPerformance",
                "dose_action": "PracticeDose",
                "capacity_obs": "QuizScore",
                "strain_obs": "StressSurvey",
                "target_obs": "PerformanceAssessment",
                "effort_obs": "HomeworkCompletion",
            },
        },
        {
            "topic": "Treatment effectiveness",
            "subdomain": "physical therapy session frequency",
            "setting": "a rehabilitation clinic network",
            "unit": "patients",
            "period": "therapy week",
            "roles": {
                "capacity_state": "MobilityGain",
                "strain_state": "SorenessLoad",
                "target_state": "FunctionalRecovery",
                "dose_action": "TherapyDose",
                "capacity_obs": "MobilityTestScore",
                "strain_obs": "SorenessReport",
                "target_obs": "FunctionAssessment",
                "effort_obs": "ExerciseCompletion",
            },
        },
        {
            "topic": "Labor & Policy",
            "subdomain": "job training hours",
            "setting": "a workforce development agency",
            "unit": "participants",
            "period": "training week",
            "roles": {
                "capacity_state": "JobSkillGrowth",
                "strain_state": "ScheduleStrain",
                "target_state": "PlacementReadiness",
                "dose_action": "TrainingDose",
                "capacity_obs": "SkillCheckScore",
                "strain_obs": "ScheduleStrainSurvey",
                "target_obs": "ReadinessAssessment",
                "effort_obs": "WorkshopAttendance",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "product reminder frequency",
            "setting": "a product experimentation team",
            "unit": "users",
            "period": "product week",
            "roles": {
                "capacity_state": "HabitFormation",
                "strain_state": "ReminderAnnoyance",
                "target_state": "LongTermEngagement",
                "dose_action": "ReminderDose",
                "capacity_obs": "HabitScore",
                "strain_obs": "AnnoyanceSurvey",
                "target_obs": "EngagementIndex",
                "effort_obs": "FeatureUseCount",
            },
        },
    ],
    "proxy_metric_hacking": [
        {
            "topic": "Education",
            "subdomain": "test-score accountability",
            "setting": "an education measurement consortium",
            "unit": "students",
            "period": "instruction cycle",
            "roles": {
                "latent_state": "TrueLearning",
                "proxy_state": "TestFamiliarity",
                "action_var": "InstructionPolicy",
                "target_obs": "RetentionTest",
                "metric_obs": "ReportedTestScore",
                "proxy_obs": "PracticeItemSimilarity",
                "engagement_obs": "ClassEngagementLog",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "engagement dashboard optimization",
            "setting": "a platform experimentation group",
            "unit": "users",
            "period": "product cycle",
            "roles": {
                "latent_state": "UserValue",
                "proxy_state": "ClickPrompting",
                "action_var": "ProductPolicy",
                "target_obs": "RetentionQuality",
                "metric_obs": "DashboardEngagement",
                "proxy_obs": "PromptExposure",
                "engagement_obs": "SessionDepthLog",
            },
        },
        {
            "topic": "Hospital data",
            "subdomain": "documentation quality scoring",
            "setting": "a hospital quality improvement office",
            "unit": "cases",
            "period": "review cycle",
            "roles": {
                "latent_state": "CareQuality",
                "proxy_state": "DocumentationPolish",
                "action_var": "QualityProgram",
                "target_obs": "PatientOutcomeAudit",
                "metric_obs": "DocumentationScore",
                "proxy_obs": "TemplateCompletionRate",
                "engagement_obs": "CareTeamReviewLog",
            },
        },
        {
            "topic": "Labor & Policy",
            "subdomain": "employment program reporting",
            "setting": "a public employment program evaluator",
            "unit": "participants",
            "period": "service cycle",
            "roles": {
                "latent_state": "JobStability",
                "proxy_state": "ReportingIntensity",
                "action_var": "ProgramPolicy",
                "target_obs": "SustainedEmploymentAudit",
                "metric_obs": "PlacementCountMetric",
                "proxy_obs": "ReportCompletionRate",
                "engagement_obs": "CounselorContactLog",
            },
        },
    ],
    "latent_mediator": [
        {
            "topic": "Treatment effectiveness",
            "subdomain": "chronic care coaching",
            "setting": "a chronic care management program",
            "unit": "patients",
            "period": "care month",
            "roles": {
                "mediator_state": "SelfManagementSkill",
                "decoy_state": "ClinicMessageRecall",
                "outcome_state": "HealthStability",
                "action_var": "CoachingProgram",
                "mediator_obs": "SelfManagementSurvey",
                "decoy_obs": "MessageRecallQuiz",
                "outcome_obs": "StabilityAssessment",
            },
        },
        {
            "topic": "Education",
            "subdomain": "attendance support program",
            "setting": "a district student support office",
            "unit": "students",
            "period": "school month",
            "roles": {
                "mediator_state": "AttendanceRoutine",
                "decoy_state": "PortalLoginHabit",
                "outcome_state": "CourseProgress",
                "action_var": "SupportProgram",
                "mediator_obs": "AttendanceConsistency",
                "decoy_obs": "PortalLoginCount",
                "outcome_obs": "ProgressAssessment",
            },
        },
        {
            "topic": "Criminal Justice",
            "subdomain": "reentry support services",
            "setting": "a county reentry services office",
            "unit": "clients",
            "period": "service month",
            "roles": {
                "mediator_state": "HousingStability",
                "decoy_state": "AppointmentFamiliarity",
                "outcome_state": "CommunityStability",
                "action_var": "ReentryProgram",
                "mediator_obs": "HousingStabilityCheck",
                "decoy_obs": "AppointmentRecallScore",
                "outcome_obs": "StabilityReview",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "new-user onboarding",
            "setting": "a product onboarding research team",
            "unit": "users",
            "period": "onboarding week",
            "roles": {
                "mediator_state": "WorkflowFluency",
                "decoy_state": "TooltipRecognition",
                "outcome_state": "SustainedUse",
                "action_var": "OnboardingProgram",
                "mediator_obs": "WorkflowTaskScore",
                "decoy_obs": "TooltipRecallQuiz",
                "outcome_obs": "SustainedUseIndex",
            },
        },
    ],
    "heterogeneous_response": [
        {
            "topic": "Treatment effectiveness",
            "subdomain": "personalized therapy selection",
            "setting": "a multisite clinical trial group",
            "unit": "patients",
            "period": "treatment month",
            "roles": {
                "subtype_param": "ResponseSubtype",
                "subtype_obs": "SubtypeScreen",
                "outcome_state": "ClinicalImprovement",
                "harm_state": "SideEffectBurden",
                "action_var": "TherapyChoice",
                "outcome_obs": "ImprovementAssessment",
                "harm_obs": "SideEffectReport",
            },
        },
        {
            "topic": "Education",
            "subdomain": "instructional format matching",
            "setting": "a learning sciences lab",
            "unit": "students",
            "period": "module week",
            "roles": {
                "subtype_param": "LearningPreference",
                "subtype_obs": "PreferenceScreen",
                "outcome_state": "ConceptMastery",
                "harm_state": "FrustrationLoad",
                "action_var": "InstructionFormat",
                "outcome_obs": "MasteryCheck",
                "harm_obs": "FrustrationSurvey",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "personalized onboarding path",
            "setting": "a product growth research team",
            "unit": "users",
            "period": "onboarding week",
            "roles": {
                "subtype_param": "UserIntentType",
                "subtype_obs": "IntentScreen",
                "outcome_state": "ActivationSuccess",
                "harm_state": "FrictionLoad",
                "action_var": "OnboardingPath",
                "outcome_obs": "ActivationIndex",
                "harm_obs": "FrictionReport",
            },
        },
        {
            "topic": "Labor & Policy",
            "subdomain": "job support pathway matching",
            "setting": "a workforce policy evaluation team",
            "unit": "participants",
            "period": "service month",
            "roles": {
                "subtype_param": "BarrierProfile",
                "subtype_obs": "BarrierScreen",
                "outcome_state": "EmploymentProgress",
                "harm_state": "ProgramBurden",
                "action_var": "SupportPathway",
                "outcome_obs": "EmploymentProgressReview",
                "harm_obs": "ProgramBurdenSurvey",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Dataclasses and simple helpers
# ---------------------------------------------------------------------------

@dataclass
class WorldBuildResult:
    world: Dict[str, Any]
    json_path: str
    archetype: str
    sub_variant: Optional[str]
    topic: str
    seed: int


@dataclass(frozen=True)
class DatasetSlot:
    archetype: str
    sub_variant: Optional[str]
    template_index: int


def _clip100(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 100.0)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def _se(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


def _safe_id(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _jsonify(obj: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-serializable Python values."""
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _continuous_var(
    name: str,
    role: str,
    desc: str,
    *,
    higher_is_better: Optional[bool],
    observed: bool = True,
    intervenable: bool = False,
) -> Dict[str, Any]:
    return {
        "name": name,
        "values": ["continuous_0_100"],
        "desc": desc,
        "description": desc,
        "role": role,
        "observed": observed,
        "intervenable": intervenable,
        "scale": {
            "type": "continuous",
            "min": 0,
            "max": 100,
            "higher_is_better": higher_is_better,
        },
    }


def _categorical_var(
    name: str,
    values: List[str],
    role: str,
    desc: str,
    *,
    observed: bool = True,
    intervenable: bool = False,
) -> Dict[str, Any]:
    return {
        "name": name,
        "values": list(values),
        "desc": desc,
        "description": desc,
        "role": role,
        "observed": observed,
        "intervenable": intervenable,
        "scale": {"type": "categorical"},
    }


def _policy(
    policy_id: str,
    display_name: str,
    description: str,
    kind: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "policy_id": policy_id,
        "display_name": display_name,
        "description": description,
        "kind": kind,
        "params": dict(params or {}),
    }


def _edge(source: str, target: str, *, lag: int = 1, kind: str = "dynamic") -> Dict[str, Any]:
    return {"source": source, "target": target, "lag": lag, "kind": kind}


def _story(template: Dict[str, Any], horizon: int) -> str:
    return (
        f"{template['setting'].capitalize()} studies longitudinal policy "
        f"experiments over {horizon} {template['period']}s. Researchers can "
        f"request simulated trajectories for {template['unit']} under named "
        f"candidate policies."
    )


def _non_intervenable(variables: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"name": v["name"], "reason": "measurement or baseline property, not a directly assigned action"}
        for v in variables
        if not v.get("intervenable", False)
    ]


def _discovery_protocol(horizon: int) -> Dict[str, Any]:
    return {
        "protocol_version": DISCOVERY_PROTOCOL_VERSION,
        "task_style": "budgeted_iterative_scientific_discovery",
        "objective": (
            "Run a sequence of small trajectory experiments, analyze returned "
            "CSV data, update hypotheses, and answer the question under the "
            "world's visible objective."
        ),
        "experiment_budget": {
            "sample_accounting": DISCOVERY_SAMPLE_ACCOUNTING,
            "max_total_samples": DISCOVERY_MAX_TOTAL_SAMPLES,
            "max_samples_per_query": DISCOVERY_MAX_SAMPLES_PER_QUERY,
            "default_units": DISCOVERY_DEFAULT_UNITS,
            "max_units": DISCOVERY_MAX_UNITS,
            "max_queries": DISCOVERY_MAX_QUERIES,
            "counted_unit": (
                "returned dataframe cells = rows * columns; rows are "
                "unit-period-policy observations"
            ),
        },
        "recommended_workflow": [
            "Start with a small exploratory rollout or focused policy comparison.",
            "Measure only variables needed for the current hypothesis; measuring all costs more.",
            "Use shorter horizons only to diagnose early signals, not for final answers.",
            "Run full-horizon follow-up experiments on plausible policies or mechanisms.",
            "Compute final-period summaries, safety rates, subgroup summaries, or mechanism contrasts from the CSV.",
            "Revise the next experiment based on the previous result rather than brute-forcing huge samples.",
        ],
        "current_limitations": [
            "Agents may compare predefined candidate policies.",
            "Agents may not yet submit arbitrary new week-by-week adaptive policies.",
        ],
        "anti_bruteforce_note": (
            "Large full-horizon comparisons across every policy and every "
            "measurement will usually exceed the per-query or total budget."
        ),
        "validated_horizon": horizon,
    }


def _visible_block(
    story: str,
    variables: List[Dict[str, Any]],
    action_variables: List[Dict[str, Any]],
    policies: List[Dict[str, Any]],
    question: str,
    horizon: int,
    default_policy_id: Optional[str] = None,
) -> Dict[str, Any]:
    observed = [v for v in variables if v.get("observed", True) and not v.get("intervenable", False)]
    protocol = _discovery_protocol(horizon)
    return {
        "story": story,
        "observed_variables": observed,
        "action_variables": action_variables,
        "agent_visible_variables": [v["name"] for v in observed + action_variables],
        "visibility_note": (
            "Only variables listed in this visible block should be shown to "
            "scientist agents. Hidden state variables and oracle metadata are "
            "not agent-visible."
        ),
        "hidden_variable_policy": (
            "Simulator state variables, mechanism parameters, oracle scores, "
            "validator details, and gold answers must be stripped from any "
            "scientist-agent prompt or query response."
        ),
        "allowed_policies": [
            {
                "policy_id": p["policy_id"],
                "display_name": p["display_name"],
                "description": p["description"],
            }
            for p in policies
        ],
        "allowed_measurements": [v["name"] for v in observed],
        "allowed_query_modes": [
            "observational_trajectory",
            "policy_rollout",
            "policy_comparison",
        ],
        "discovery_protocol": protocol,
        "experiment_budget": protocol["experiment_budget"],
        "default_observational_policy_id": default_policy_id,
        "sample_unit": "unit_period_row",
        "default_horizon": horizon,
        "question": question,
    }


def _visible_names(variables: List[Dict[str, Any]]) -> List[str]:
    return [v["name"] for v in variables if v.get("observed", True)]


def _add_if_wanted(row: Dict[str, Any], wanted: set, name: str, values: np.ndarray, i: int) -> None:
    if name in wanted:
        value = values[i]
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        row[name] = value


def _aux_continuous(name: str, role: str, desc: str, higher_is_better: Optional[bool] = None) -> Dict[str, Any]:
    return _continuous_var(name, role, desc, higher_is_better=higher_is_better)


def _replace_policy_id_strings(obj: Any, mapping: Dict[str, str]) -> Any:
    """Recursively replace exposed policy ids with per-world neutral aliases."""
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    if isinstance(obj, list):
        return [_replace_policy_id_strings(v, mapping) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_policy_id_strings(v, mapping) for v in obj)
    if isinstance(obj, dict):
        return {k: _replace_policy_id_strings(v, mapping) for k, v in obj.items()}
    return obj


def _alias_policies_for_output(
    policies: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Assign neutral per-world policy ids.

    The simulator still reads the `kind` field, so policy ids can be made
    non-semantic without changing the mechanism.  This avoids benchmark items
    where the same human-readable policy id is always the answer.
    """
    labels = [f"policy_{chr(ord('A') + i)}" for i in range(len(policies))]
    rng = random.Random(seed * 104729 + 991)
    rng.shuffle(labels)
    mapping = {p["policy_id"]: labels[i] for i, p in enumerate(policies)}
    aliased: List[Dict[str, Any]] = []
    for p in policies:
        q = dict(p)
        q["original_policy_id"] = p["policy_id"]
        q["policy_id"] = mapping[p["policy_id"]]
        letter = q["policy_id"].split("_", 1)[1]
        q["display_name"] = f"Policy {letter}"
        aliased.append(q)
    return aliased, mapping


def _score_policy(policy: Dict[str, Any], rollout: Dict[str, Any]) -> Dict[str, Any]:
    utility = rollout["per_unit"]["utility"]
    out = {
        "policy_id": policy["policy_id"],
        "display_name": policy["display_name"],
        "expected_utility": _mean(utility),
        "utility_standard_error": _se(utility),
        "target_mean": _mean(rollout["per_unit"]["target"]),
        "target_standard_error": _se(rollout["per_unit"]["target"]),
        "metrics": rollout["summary"],
        "trajectory_means": rollout["trajectory_means"],
    }
    if "harm" in rollout["per_unit"]:
        out["harm_mean"] = _mean(rollout["per_unit"]["harm"])
        out["harm_standard_error"] = _se(rollout["per_unit"]["harm"])
    if "subgroup" in rollout:
        out["subgroup_metrics"] = rollout["subgroup"]
    return _jsonify(out)


def _find_policy(policies: List[Dict[str, Any]], policy_id: str) -> Dict[str, Any]:
    for p in policies:
        if p["policy_id"] == policy_id:
            return p
    raise KeyError(policy_id)


def _policy_by_id(scores: List[Dict[str, Any]], policy_id: str) -> Dict[str, Any]:
    for s in scores:
        if s["policy_id"] == policy_id:
            return s
    raise KeyError(policy_id)


def _top_two(scores: List[Dict[str, Any]], key: str = "expected_utility") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ranked = sorted(scores, key=lambda s: s[key], reverse=True)
    if len(ranked) < 2:
        raise ValueError("need at least two scores")
    return ranked[0], ranked[1]


def _check(
    name: str,
    passed: bool,
    value: Any,
    threshold: Any,
    description: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": _jsonify(value),
        "threshold": threshold,
        "description": description,
    }


def _all_pass(checks: Iterable[Dict[str, Any]]) -> bool:
    return all(bool(c.get("passed")) for c in checks)


# ---------------------------------------------------------------------------
# Rollout dispatch
# ---------------------------------------------------------------------------

def rollout(
    simulator_config: Dict[str, Any],
    policy: Dict[str, Any],
    n_units: int,
    seed: int,
    *,
    return_rows: bool = False,
    measurements: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run one candidate policy in one simulator world."""
    arch = simulator_config["archetype"]
    if arch == "delayed_harm":
        return _rollout_delayed_harm(simulator_config, policy, n_units, seed, return_rows, measurements)
    if arch == "dose_response":
        return _rollout_dose_response(simulator_config, policy, n_units, seed, return_rows, measurements)
    if arch == "proxy_metric_hacking":
        return _rollout_proxy_metric(simulator_config, policy, n_units, seed, return_rows, measurements)
    if arch == "latent_mediator":
        return _rollout_latent_mediator(simulator_config, policy, n_units, seed, return_rows, measurements)
    if arch == "heterogeneous_response":
        return _rollout_heterogeneous(simulator_config, policy, n_units, seed, return_rows, measurements)
    raise KeyError(f"unknown archetype {arch!r}")


def evaluate_policies(
    simulator_config: Dict[str, Any],
    policies: List[Dict[str, Any]],
    n_rollouts: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Oracle-evaluate every candidate policy with high-budget simulation."""
    scores: List[Dict[str, Any]] = []
    for i, policy in enumerate(policies):
        # Common seed base keeps comparisons stable while policy index avoids
        # accidental identical row order in saved previews.
        r = rollout(simulator_config, policy, n_rollouts, seed + i * 104729)
        scores.append(_score_policy(policy, r))
    return scores


# ---------------------------------------------------------------------------
# Archetype 1: delayed harm
# ---------------------------------------------------------------------------

def _delayed_action_arrays(
    policy: Dict[str, Any],
    t: int,
    harm_proxy_prev: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    kind = policy["kind"]
    dose = np.zeros(n, dtype=np.int8)
    support = np.zeros(n, dtype=bool)

    if kind == "always_standard":
        dose[:] = 1
    elif kind == "always_high":
        dose[:] = 2
    elif kind == "high_then_standard":
        dose[:] = 2 if t < int(policy["params"].get("high_periods", 2)) else 1
    elif kind == "standard_plus_support":
        dose[:] = 1
        support[:] = True
    elif kind == "adaptive_toxicity_switch":
        threshold = float(policy["params"].get("switch_threshold", 42.0))
        high_until = int(policy["params"].get("latest_high_period", 5))
        use_high = (harm_proxy_prev < threshold) & (t < high_until)
        dose[use_high] = 2
        dose[~use_high] = 1
        support[~use_high] = True
    else:
        raise KeyError(kind)
    return dose, support


def _rollout_delayed_harm(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    n_units: int,
    seed: int,
    return_rows: bool,
    measurements: Optional[List[str]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    roles = cfg["roles"]
    horizon = int(cfg["horizon"])

    burden = _clip100(rng.normal(p["init_burden_mean"], p["init_burden_sd"], n_units))
    harm = _clip100(rng.normal(p["init_harm_mean"], p["init_harm_sd"], n_units))
    adherence = _clip01(rng.normal(p["init_adherence_mean"], p["init_adherence_sd"], n_units))
    recovery = _clip100(100.0 - p["recovery_burden_weight"] * burden - p["recovery_harm_weight"] * harm)
    baseline_profile = _clip100(0.72 * burden + 0.28 * harm + rng.normal(0, 5.0, n_units))
    resilience_index = _clip100(100.0 * adherence - 0.25 * harm + rng.normal(0, 5.0, n_units))

    traj = {
        roles["burden_state"]: [_mean(burden)],
        roles["harm_state"]: [_mean(harm)],
        roles["target_state"]: [_mean(recovery)],
        roles["adherence_state"]: [_mean(100.0 * adherence)],
    }
    rows: List[Dict[str, Any]] = []
    harm_proxy_prev = _clip100(harm + rng.normal(0, p["harm_obs_sd"], n_units))

    for t in range(horizon):
        dose, support = _delayed_action_arrays(policy, t, harm_proxy_prev, n_units)
        treatment_effect = np.where(dose == 2, p["high_effect"], np.where(dose == 1, p["standard_effect"], 0.0))
        harm_gain = np.where(dose == 2, p["high_harm_gain"], np.where(dose == 1, p["standard_harm_gain"], 0.0))
        support_reduction = support.astype(float) * p["support_harm_reduction"]

        burden = _clip100(
            burden
            + p["natural_burden_growth"]
            - treatment_effect * adherence
            + p["harm_burden_penalty"] * harm
            + rng.normal(0, p["process_sd"], n_units)
        )
        harm = _clip100(
            harm
            + harm_gain
            - support_reduction
            - p["harm_recovery_rate"]
            + rng.normal(0, p["harm_process_sd"], n_units)
        )
        adherence = _clip01(
            p["base_adherence"]
            - p["adherence_harm_slope"] * harm
            + rng.normal(0, p["adherence_noise_sd"], n_units)
        )
        recovery = _clip100(
            100.0
            - p["recovery_burden_weight"] * burden
            - p["recovery_harm_weight"] * harm
            + rng.normal(0, p["target_noise_sd"], n_units)
        )

        burden_obs = _clip100(burden + rng.normal(0, p["burden_obs_sd"], n_units))
        harm_obs = _clip100(harm + rng.normal(0, p["harm_obs_sd"], n_units))
        target_obs = _clip100(recovery + rng.normal(0, p["target_obs_sd"], n_units))
        adherence_obs = _clip100(100.0 * adherence + rng.normal(0, p["adherence_obs_sd"], n_units))
        baseline_obs = _clip100(baseline_profile + rng.normal(0, 2.5, n_units))
        resilience_obs = _clip100(resilience_index + rng.normal(0, 4.0, n_units))
        resource_use_obs = _clip100(12.0 + 28.0 * dose + 12.0 * support.astype(float) + rng.normal(0, 3.5, n_units))
        contact_load_obs = _clip100(8.0 + 33.0 * dose - 9.0 * support.astype(float) + rng.normal(0, 4.0, n_units))
        short_term_response_obs = _clip100(100.0 - burden + 0.20 * recovery + rng.normal(0, 5.0, n_units))
        burden_trend_obs = _clip100(52.0 - treatment_effect * adherence + p["natural_burden_growth"] + rng.normal(0, 4.0, n_units))
        harm_proxy_prev = harm_obs

        traj[roles["burden_state"]].append(_mean(burden))
        traj[roles["harm_state"]].append(_mean(harm))
        traj[roles["target_state"]].append(_mean(recovery))
        traj[roles["adherence_state"]].append(_mean(100.0 * adherence))

        if return_rows:
            wanted = set(measurements or cfg["observed_variable_names"])
            for i in range(n_units):
                row = {
                    "unit_id": i,
                    "time": t + 1,
                    "policy_id": policy["policy_id"],
                    roles["dose_action"]: ["None", "Standard", "High"][int(dose[i])],
                    roles["support_action"]: "On" if bool(support[i]) else "Off",
                }
                _add_if_wanted(row, wanted, roles["burden_obs"], burden_obs, i)
                _add_if_wanted(row, wanted, roles["harm_obs"], harm_obs, i)
                _add_if_wanted(row, wanted, roles["target_obs"], target_obs, i)
                _add_if_wanted(row, wanted, roles["adherence_obs"], adherence_obs, i)
                _add_if_wanted(row, wanted, "BaselineProfileScore", baseline_obs, i)
                _add_if_wanted(row, wanted, "ResilienceIndex", resilience_obs, i)
                _add_if_wanted(row, wanted, "ResourceUseIndex", resource_use_obs, i)
                _add_if_wanted(row, wanted, "ContactLoadIndex", contact_load_obs, i)
                _add_if_wanted(row, wanted, "ShortTermResponseSignal", short_term_response_obs, i)
                _add_if_wanted(row, wanted, "BurdenTrendSignal", burden_trend_obs, i)
                rows.append(row)

    safety_threshold = float(cfg["safety"]["harm_threshold"])
    severe = harm > safety_threshold
    utility = recovery.copy()
    return {
        "per_unit": {
            "utility": utility,
            "target": recovery,
            "harm": harm,
            "safety_violation": severe.astype(float),
        },
        "summary": {
            "final_target_mean": _mean(recovery),
            "final_harm_mean": _mean(harm),
            "safety_violation_rate": _mean(severe.astype(float)),
            "early_target_mean_t2": traj[roles["target_state"]][min(2, horizon)],
            "early_burden_mean_t2": traj[roles["burden_state"]][min(2, horizon)],
        },
        "trajectory_means": traj,
        "rows": rows,
    }


def _build_delayed_harm(
    template: Dict[str, Any],
    seed: int,
    horizon: int,
    oracle_rollouts: int,
    forced_sub_variant: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    roles = template["roles"]
    params = {
        "init_burden_mean": rng.uniform(54, 62),
        "init_burden_sd": rng.uniform(7.0, 10.0),
        "init_harm_mean": rng.uniform(9, 14),
        "init_harm_sd": rng.uniform(3.5, 5.5),
        "init_adherence_mean": rng.uniform(0.82, 0.90),
        "init_adherence_sd": rng.uniform(0.04, 0.07),
        "natural_burden_growth": rng.uniform(2.1, 3.0),
        "standard_effect": rng.uniform(6.0, 7.4),
        "high_effect": rng.uniform(10.5, 12.2),
        "standard_harm_gain": rng.uniform(2.6, 3.7),
        "high_harm_gain": rng.uniform(9.5, 11.8),
        "support_harm_reduction": rng.uniform(4.8, 6.5),
        "harm_recovery_rate": rng.uniform(1.2, 1.8),
        "harm_burden_penalty": rng.uniform(0.004, 0.010),
        "base_adherence": rng.uniform(0.91, 0.96),
        "adherence_harm_slope": rng.uniform(0.0035, 0.0050),
        "recovery_burden_weight": rng.uniform(0.70, 0.80),
        "recovery_harm_weight": rng.uniform(0.12, 0.22),
        "process_sd": rng.uniform(1.2, 2.2),
        "harm_process_sd": rng.uniform(1.1, 2.0),
        "adherence_noise_sd": rng.uniform(0.018, 0.030),
        "target_noise_sd": rng.uniform(1.0, 2.0),
        "burden_obs_sd": rng.uniform(4.0, 6.0),
        "harm_obs_sd": rng.uniform(4.0, 6.5),
        "target_obs_sd": rng.uniform(3.0, 5.5),
        "adherence_obs_sd": rng.uniform(4.0, 6.0),
    }
    safety = {"harm_threshold": rng.uniform(50, 57), "allowed_violation_rate": 0.15}
    policies = [
        _policy("always_standard", "Always standard", "Use the standard action level every period.", "always_standard"),
        _policy("always_high", "Always high", "Use the high action level every period.", "always_high"),
        _policy("high_then_standard", "High then standard", "Use high level for two periods, then standard.", "high_then_standard", {"high_periods": 2}),
        _policy("standard_plus_support", "Standard plus support", "Use standard level with support enabled every period.", "standard_plus_support"),
        _policy(
            "adaptive_toxicity_switch",
            "Adaptive toxicity switch",
            f"Use high level while the latest {roles['harm_obs']} is below a threshold, otherwise standard with support.",
            "adaptive_toxicity_switch",
            {"switch_threshold": 42.0, "latest_high_period": 5},
        ),
    ]
    cfg = {
        "archetype": "delayed_harm",
        "sub_variant": "safety_constrained_long_horizon",
        "horizon": horizon,
        "roles": roles,
        "unit": template["unit"],
        "period": template["period"],
        "parameters": params,
        "safety": safety,
        "policies": policies,
        "update_equations_id": "delayed_harm_v1",
        "observation_equations_id": "delayed_harm_noisy_proxy_v1",
    }
    scores = evaluate_policies(cfg, policies, oracle_rollouts, seed + 5000)
    allowed = safety["allowed_violation_rate"]
    feasible = [s for s in scores if s["metrics"]["safety_violation_rate"] <= allowed]
    if feasible:
        best = sorted(feasible, key=lambda s: s["target_mean"], reverse=True)[0]
        runner_pool = [s for s in feasible if s["policy_id"] != best["policy_id"]]
        runner = sorted(runner_pool, key=lambda s: s["target_mean"], reverse=True)[0] if runner_pool else best
    else:
        best, runner = _top_two(scores, "expected_utility")
    target_only_best = sorted(scores, key=lambda s: s["target_mean"], reverse=True)[0]
    margin = best["target_mean"] - runner["target_mean"] if best is not runner else 0.0
    high = _policy_by_id(scores, "always_high")
    standard = _policy_by_id(scores, "always_standard")
    early_benefit = standard["metrics"]["early_burden_mean_t2"] - high["metrics"]["early_burden_mean_t2"]

    checks = [
        _check("has_safe_candidate", len(feasible) >= 1, len(feasible), ">= 1", "At least one policy satisfies the safety constraint."),
        _check("aggressive_looks_better_early", early_benefit >= MIN_TEMPTATION_GAP, early_benefit, f">= {MIN_TEMPTATION_GAP}", "Always-high lowers early burden more than standard."),
        _check("aggressive_violates_safety", high["metrics"]["safety_violation_rate"] > allowed + 0.10, high["metrics"]["safety_violation_rate"], f"> {allowed + 0.10:.2f}", "Always-high is tempting but unsafe."),
        _check("safety_changes_choice", target_only_best["policy_id"] != best["policy_id"], {"target_only_best": target_only_best["policy_id"], "safe_best": best["policy_id"]}, "different", "Safety constraint changes the recommendation."),
        _check("gold_margin", margin >= MIN_GOLD_MARGIN, margin, f">= {MIN_GOLD_MARGIN}", "Best safe policy beats runner-up safe policy."),
        _check("oracle_se_small", best["target_standard_error"] <= max(0.2, abs(margin) * MAX_ORACLE_SE_FRACTION), best["target_standard_error"], f"<= max(0.2, {MAX_ORACLE_SE_FRACTION}*margin)", "Oracle standard error is small relative to margin."),
    ]
    question = (
        f"Which candidate policy should be deployed for {horizon} {template['period']}s to maximize "
        f"final {roles['target_obs']} while keeping the risk of severe {roles['harm_obs']} below "
        f"{allowed:.0%}? Answer with one policy_id."
    )
    return _assemble_world(
        template=template,
        cfg=cfg,
        variables=_delayed_variables(roles, policies),
        action_variables=[
            _categorical_var(roles["dose_action"], ["None", "Standard", "High"], "action", "Assigned intensity level for the period.", intervenable=True),
            _categorical_var(roles["support_action"], ["Off", "On"], "action", "Whether additional support is assigned for the period.", intervenable=True),
        ],
        policies=policies,
        scores=scores,
        checks=checks,
        question=question,
        answer=best["policy_id"],
        question_type="rpg_delayed_harm",
        objective=f"Maximize {roles['target_obs']} subject to severe {roles['harm_obs']} risk <= {allowed:.0%}.",
        gold_answer={
            "answer_type": "policy_id",
            "policy_id": best["policy_id"],
            "safety_constraint": safety,
            "target_only_best_policy_id": target_only_best["policy_id"],
        },
        runner_up={"policy_id": runner["policy_id"], "target_mean": runner["target_mean"]},
        gold_margin=margin,
        causal_edges=[
            _edge(roles["dose_action"], roles["burden_state"]),
            _edge(roles["dose_action"], roles["harm_state"]),
            _edge(roles["support_action"], roles["harm_state"]),
            _edge(roles["harm_state"], roles["adherence_state"]),
            _edge(roles["burden_state"], roles["target_state"]),
            _edge(roles["harm_state"], roles["target_state"]),
        ],
        seed=seed,
        oracle_rollouts=oracle_rollouts,
    )


def _delayed_variables(roles: Dict[str, str], policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _continuous_var(roles["burden_obs"], "burden_proxy", "Noisy 0-100 measurement of the current burden level.", higher_is_better=False),
        _continuous_var(roles["harm_obs"], "harm_proxy", "Noisy 0-100 measurement of accumulated adverse burden.", higher_is_better=False),
        _continuous_var(roles["target_obs"], "target_proxy", "Noisy 0-100 summary of final status.", higher_is_better=True),
        _continuous_var(roles["adherence_obs"], "adherence_proxy", "Noisy 0-100 record of participation or adherence.", higher_is_better=True),
        _aux_continuous("BaselineProfileScore", "baseline_proxy", "Noisy 0-100 pre-policy baseline profile summary.", higher_is_better=None),
        _aux_continuous("ResilienceIndex", "baseline_proxy", "Noisy 0-100 pre-policy resilience or capacity summary.", higher_is_better=True),
        _aux_continuous("ResourceUseIndex", "process_proxy", "Noisy 0-100 trace of resources used during the period.", higher_is_better=None),
        _aux_continuous("ContactLoadIndex", "process_proxy", "Noisy 0-100 trace of assigned contact or workload intensity.", higher_is_better=None),
        _aux_continuous("ShortTermResponseSignal", "short_term_proxy", "Noisy 0-100 short-horizon response signal.", higher_is_better=True),
        _aux_continuous("BurdenTrendSignal", "trend_proxy", "Noisy 0-100 one-period burden trend signal.", higher_is_better=False),
    ]


# ---------------------------------------------------------------------------
# Archetype 2: dose response
# ---------------------------------------------------------------------------

def _dose_curve(sub_variant: str, rng: random.Random) -> Tuple[Dict[int, float], Dict[int, float]]:
    if sub_variant == "inverted_u":
        gain = {0: 0.0, 1: rng.uniform(3.0, 4.3), 2: rng.uniform(7.5, 9.0), 3: rng.uniform(6.0, 7.0)}
        strain = {0: 0.0, 1: rng.uniform(0.8, 1.5), 2: rng.uniform(2.8, 3.8), 3: rng.uniform(8.0, 10.5)}
    elif sub_variant == "minimum_effective":
        gain = {0: 0.0, 1: rng.uniform(0.8, 1.8), 2: rng.uniform(4.0, 5.6), 3: rng.uniform(8.5, 10.5)}
        strain = {0: 0.0, 1: rng.uniform(0.5, 1.1), 2: rng.uniform(1.6, 2.4), 3: rng.uniform(3.0, 4.2)}
    else:  # saturation
        gain = {0: 0.0, 1: rng.uniform(2.2, 3.4), 2: rng.uniform(7.4, 8.8), 3: rng.uniform(7.7, 8.9)}
        strain = {0: 0.0, 1: rng.uniform(0.8, 1.4), 2: rng.uniform(2.4, 3.4), 3: rng.uniform(5.4, 7.0)}
    return gain, strain


def _rollout_dose_response(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    n_units: int,
    seed: int,
    return_rows: bool,
    measurements: Optional[List[str]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    roles = cfg["roles"]
    horizon = int(cfg["horizon"])
    dose = int(policy["params"]["dose_code"])

    capacity = _clip100(rng.normal(p["init_capacity_mean"], p["init_capacity_sd"], n_units))
    strain = _clip100(rng.normal(p["init_strain_mean"], p["init_strain_sd"], n_units))
    target = _clip100(p["target_intercept"] + p["target_capacity_weight"] * capacity - p["target_strain_weight"] * strain)
    baseline_readiness = _clip100(capacity - 0.25 * strain + rng.normal(0, 5.0, n_units))
    schedule_load = _clip100(strain + rng.normal(6.0, 5.0, n_units))

    traj = {
        roles["capacity_state"]: [_mean(capacity)],
        roles["strain_state"]: [_mean(strain)],
        roles["target_state"]: [_mean(target)],
    }
    rows: List[Dict[str, Any]] = []

    gain = float(p["dose_gain"][str(dose)])
    strain_gain = float(p["dose_strain"][str(dose)])
    for t in range(horizon):
        capacity = _clip100(
            capacity
            + gain
            - p["strain_learning_penalty"] * strain
            + rng.normal(0, p["capacity_process_sd"], n_units)
        )
        strain = _clip100(
            strain
            + strain_gain
            - p["strain_recovery_rate"]
            + rng.normal(0, p["strain_process_sd"], n_units)
        )
        target = _clip100(
            p["target_intercept"]
            + p["target_capacity_weight"] * capacity
            - p["target_strain_weight"] * strain
            + rng.normal(0, p["target_noise_sd"], n_units)
        )
        capacity_obs = _clip100(capacity + rng.normal(0, p["capacity_obs_sd"], n_units))
        strain_obs = _clip100(strain + rng.normal(0, p["strain_obs_sd"], n_units))
        target_obs = _clip100(target + rng.normal(0, p["target_obs_sd"], n_units))
        effort_obs = _clip100(20.0 + 18.0 * dose + rng.normal(0, p["effort_obs_sd"], n_units))
        baseline_obs = _clip100(baseline_readiness + rng.normal(0, 3.0, n_units))
        load_obs = _clip100(schedule_load + 4.0 * dose + rng.normal(0, 3.0, n_units))
        resource_obs = _clip100(10.0 + 22.0 * dose + rng.normal(0, 4.0, n_units))
        short_gain_obs = _clip100(50.0 + gain * 4.5 - strain_gain * 2.0 + rng.normal(0, 5.0, n_units))
        recovery_obs = _clip100(100.0 - strain + rng.normal(0, 5.0, n_units))
        attendance_obs = _clip100(72.0 + 4.5 * dose - 0.25 * strain + rng.normal(0, 6.0, n_units))

        traj[roles["capacity_state"]].append(_mean(capacity))
        traj[roles["strain_state"]].append(_mean(strain))
        traj[roles["target_state"]].append(_mean(target))

        if return_rows:
            wanted = set(measurements or cfg["observed_variable_names"])
            for i in range(n_units):
                row = {
                    "unit_id": i,
                    "time": t + 1,
                    "policy_id": policy["policy_id"],
                    roles["dose_action"]: ["None", "Low", "Medium", "High"][dose],
                }
                _add_if_wanted(row, wanted, roles["capacity_obs"], capacity_obs, i)
                _add_if_wanted(row, wanted, roles["strain_obs"], strain_obs, i)
                _add_if_wanted(row, wanted, roles["target_obs"], target_obs, i)
                _add_if_wanted(row, wanted, roles["effort_obs"], effort_obs, i)
                _add_if_wanted(row, wanted, "BaselineReadinessScore", baseline_obs, i)
                _add_if_wanted(row, wanted, "ScheduleLoadIndex", load_obs, i)
                _add_if_wanted(row, wanted, "ResourceUseIndex", resource_obs, i)
                _add_if_wanted(row, wanted, "ShortTermGainSignal", short_gain_obs, i)
                _add_if_wanted(row, wanted, "RecoveryWindowIndex", recovery_obs, i)
                _add_if_wanted(row, wanted, "AttendanceTrace", attendance_obs, i)
                rows.append(row)

    return {
        "per_unit": {"utility": target, "target": target, "harm": strain},
        "summary": {
            "final_target_mean": _mean(target),
            "final_strain_mean": _mean(strain),
            "dose_code": dose,
        },
        "trajectory_means": traj,
        "rows": rows,
    }


def _build_dose_response(
    template: Dict[str, Any],
    seed: int,
    horizon: int,
    oracle_rollouts: int,
    forced_sub_variant: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    roles = template["roles"]
    sub_variant = forced_sub_variant or DOSE_SUB_VARIANTS[seed % len(DOSE_SUB_VARIANTS)]
    if sub_variant not in DOSE_SUB_VARIANTS:
        raise ValueError(f"unknown dose_response sub_variant {sub_variant!r}")
    gain, strain = _dose_curve(sub_variant, rng)
    params = {
        "init_capacity_mean": rng.uniform(31, 39),
        "init_capacity_sd": rng.uniform(6.0, 9.0),
        "init_strain_mean": rng.uniform(10, 16),
        "init_strain_sd": rng.uniform(3.0, 5.0),
        "dose_gain": {str(k): v for k, v in gain.items()},
        "dose_strain": {str(k): v for k, v in strain.items()},
        "strain_learning_penalty": rng.uniform(0.030, 0.055),
        "strain_recovery_rate": rng.uniform(0.8, 1.5),
        "target_intercept": rng.uniform(17, 23),
        "target_capacity_weight": rng.uniform(0.78, 0.92),
        "target_strain_weight": rng.uniform(0.34, 0.48),
        "capacity_process_sd": rng.uniform(1.4, 2.4),
        "strain_process_sd": rng.uniform(1.0, 1.8),
        "target_noise_sd": rng.uniform(1.0, 2.0),
        "capacity_obs_sd": rng.uniform(3.0, 5.0),
        "strain_obs_sd": rng.uniform(3.0, 5.5),
        "target_obs_sd": rng.uniform(3.0, 5.0),
        "effort_obs_sd": rng.uniform(4.0, 6.5),
    }
    dose_names = {0: "none", 1: "low", 2: "medium", 3: "high"}
    policies = [
        _policy(f"dose_{dose_names[d]}", f"{dose_names[d].title()} dose", f"Assign the {dose_names[d]} dose every period.", "static_dose", {"dose_code": d})
        for d in [0, 1, 2, 3]
    ]
    cfg = {
        "archetype": "dose_response",
        "sub_variant": sub_variant,
        "horizon": horizon,
        "roles": roles,
        "unit": template["unit"],
        "period": template["period"],
        "parameters": params,
        "policies": policies,
        "update_equations_id": "dose_response_v1",
        "observation_equations_id": "dose_response_noisy_proxy_v1",
    }
    scores = evaluate_policies(cfg, policies, oracle_rollouts, seed + 6000)
    best, runner = _top_two(scores, "expected_utility")
    margin = best["expected_utility"] - runner["expected_utility"]
    by_dose = {int(_find_policy(policies, s["policy_id"])["params"]["dose_code"]): s for s in scores}
    target_curve = {str(d): by_dose[d]["target_mean"] for d in sorted(by_dose)}
    if sub_variant == "inverted_u":
        shape_ok = by_dose[2]["target_mean"] > by_dose[3]["target_mean"] + MIN_CLEAR_EFFECT
    elif sub_variant == "minimum_effective":
        shape_ok = by_dose[3]["target_mean"] > by_dose[2]["target_mean"] + MIN_CLEAR_EFFECT
    else:
        shape_ok = by_dose[2]["target_mean"] >= by_dose[3]["target_mean"] + 1.0 and by_dose[2]["target_mean"] > by_dose[1]["target_mean"] + MIN_CLEAR_EFFECT
    checks = [
        _check("curve_shape_matches_subvariant", shape_ok, {"sub_variant": sub_variant, "target_curve": target_curve}, "shape-specific margin", "Oracle target curve matches intended dose-response shape."),
        _check("best_dose_margin", margin >= MIN_GOLD_MARGIN, margin, f">= {MIN_GOLD_MARGIN}", "Best dose beats runner-up."),
        _check("low_not_enough", by_dose[1]["target_mean"] < max(by_dose[2]["target_mean"], by_dose[3]["target_mean"]) - MIN_CLEAR_EFFECT, target_curve, f"gap >= {MIN_CLEAR_EFFECT}", "Low dose is not enough to solve the task."),
        _check("oracle_se_small", best["target_standard_error"] <= max(0.2, margin * MAX_ORACLE_SE_FRACTION), best["target_standard_error"], f"<= max(0.2, {MAX_ORACLE_SE_FRACTION}*margin)", "Oracle standard error is small relative to margin."),
    ]
    question = (
        f"Which dose policy should be used for {horizon} {template['period']}s to maximize final "
        f"{roles['target_obs']}? Explore the dose levels rather than assuming more is always better. "
        f"Answer with one policy_id."
    )
    return _assemble_world(
        template=template,
        cfg=cfg,
        variables=_dose_variables(roles, policies),
        action_variables=[
            _categorical_var(roles["dose_action"], ["None", "Low", "Medium", "High"], "action", "Assigned dose level for the period.", intervenable=True),
        ],
        policies=policies,
        scores=scores,
        checks=checks,
        question=question,
        answer=best["policy_id"],
        question_type="rpg_dose_response",
        objective=f"Maximize final {roles['target_obs']} over candidate dose policies.",
        gold_answer={
            "answer_type": "policy_id",
            "policy_id": best["policy_id"],
            "curve_shape": sub_variant,
            "target_curve": target_curve,
        },
        runner_up={"policy_id": runner["policy_id"], "expected_utility": runner["expected_utility"]},
        gold_margin=margin,
        causal_edges=[
            _edge(roles["dose_action"], roles["capacity_state"]),
            _edge(roles["dose_action"], roles["strain_state"]),
            _edge(roles["strain_state"], roles["capacity_state"]),
            _edge(roles["capacity_state"], roles["target_state"]),
            _edge(roles["strain_state"], roles["target_state"]),
        ],
        seed=seed,
        oracle_rollouts=oracle_rollouts,
    )


def _dose_variables(roles: Dict[str, str], policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _continuous_var(roles["capacity_obs"], "capacity_proxy", "Noisy 0-100 measurement of accumulated skill or capacity.", higher_is_better=True),
        _continuous_var(roles["strain_obs"], "strain_proxy", "Noisy 0-100 measurement of load or strain.", higher_is_better=False),
        _continuous_var(roles["target_obs"], "target_proxy", "Noisy 0-100 final performance measurement.", higher_is_better=True),
        _continuous_var(roles["effort_obs"], "effort_proxy", "Noisy 0-100 record of dose-related participation.", higher_is_better=None),
        _aux_continuous("BaselineReadinessScore", "baseline_proxy", "Noisy 0-100 pre-policy readiness summary.", higher_is_better=True),
        _aux_continuous("ScheduleLoadIndex", "baseline_proxy", "Noisy 0-100 measure of concurrent load.", higher_is_better=False),
        _aux_continuous("ResourceUseIndex", "process_proxy", "Noisy 0-100 trace of resources used by the dose.", higher_is_better=None),
        _aux_continuous("ShortTermGainSignal", "short_term_proxy", "Noisy 0-100 short-horizon gain signal.", higher_is_better=True),
        _aux_continuous("RecoveryWindowIndex", "process_proxy", "Noisy 0-100 recovery-room or rest-window signal.", higher_is_better=True),
        _aux_continuous("AttendanceTrace", "process_proxy", "Noisy 0-100 participation trace recorded during the period.", higher_is_better=True),
    ]


# ---------------------------------------------------------------------------
# Archetype 3: proxy metric hacking
# ---------------------------------------------------------------------------

def _rollout_proxy_metric(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    n_units: int,
    seed: int,
    return_rows: bool,
    measurements: Optional[List[str]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    roles = cfg["roles"]
    horizon = int(cfg["horizon"])
    action = policy["kind"]

    latent = _clip100(rng.normal(p["init_latent_mean"], p["init_latent_sd"], n_units))
    proxy = _clip100(rng.normal(p["init_proxy_mean"], p["init_proxy_sd"], n_units))
    metric = _clip100(p["metric_latent_weight"] * latent + p["metric_proxy_weight"] * proxy)
    baseline_profile = _clip100(0.85 * latent + rng.normal(0, 5.0, n_units))
    audit_propensity = _clip100(40.0 + 0.35 * latent + 0.25 * proxy + rng.normal(0, 6.0, n_units))
    traj = {
        roles["latent_state"]: [_mean(latent)],
        roles["proxy_state"]: [_mean(proxy)],
        roles["metric_obs"]: [_mean(metric)],
    }
    rows: List[Dict[str, Any]] = []

    true_gain = p["true_gain"][action]
    proxy_gain = p["proxy_gain"][action]
    for t in range(horizon):
        latent = _clip100(
            latent
            + true_gain
            - p["proxy_crowding_penalty"] * proxy
            + rng.normal(0, p["latent_process_sd"], n_units)
        )
        proxy = _clip100(proxy + proxy_gain + rng.normal(0, p["proxy_process_sd"], n_units))
        metric = _clip100(
            p["metric_intercept"]
            + p["metric_latent_weight"] * latent
            + p["metric_proxy_weight"] * proxy
            + rng.normal(0, p["metric_obs_sd"], n_units)
        )
        target_obs = _clip100(latent + rng.normal(0, p["target_obs_sd"], n_units))
        proxy_obs = _clip100(proxy + rng.normal(0, p["proxy_obs_sd"], n_units))
        engagement_obs = _clip100(0.55 * latent + 0.25 * proxy + rng.normal(0, p["engagement_obs_sd"], n_units))
        baseline_obs = _clip100(baseline_profile + rng.normal(0, 3.0, n_units))
        audit_obs = _clip100(audit_propensity + 0.25 * latent + rng.normal(0, 5.0, n_units))
        compliance_obs = _clip100(0.18 * latent + 0.82 * proxy + rng.normal(0, 4.5, n_units))
        resource_obs = _clip100(18.0 + p["true_gain"][action] * 4.0 + p["proxy_gain"][action] * 2.0 + rng.normal(0, 4.0, n_units))
        rater_attention_obs = _clip100(35.0 + 0.55 * proxy + rng.normal(0, 6.0, n_units))
        delayed_trace_obs = _clip100(0.75 * latent + 0.10 * proxy + rng.normal(0, 6.0, n_units))

        traj[roles["latent_state"]].append(_mean(latent))
        traj[roles["proxy_state"]].append(_mean(proxy))
        traj[roles["metric_obs"]].append(_mean(metric))

        if return_rows:
            wanted = set(measurements or cfg["observed_variable_names"])
            for i in range(n_units):
                row = {
                    "unit_id": i,
                    "time": t + 1,
                    "policy_id": policy["policy_id"],
                    roles["action_var"]: policy["policy_id"],
                }
                _add_if_wanted(row, wanted, roles["target_obs"], target_obs, i)
                _add_if_wanted(row, wanted, roles["metric_obs"], metric, i)
                _add_if_wanted(row, wanted, roles["proxy_obs"], proxy_obs, i)
                _add_if_wanted(row, wanted, roles["engagement_obs"], engagement_obs, i)
                _add_if_wanted(row, wanted, "BaselineProfileScore", baseline_obs, i)
                _add_if_wanted(row, wanted, "IndependentAuditScore", audit_obs, i)
                _add_if_wanted(row, wanted, "SurfaceComplianceSignal", compliance_obs, i)
                _add_if_wanted(row, wanted, "ResourceUseIndex", resource_obs, i)
                _add_if_wanted(row, wanted, "RaterAttentionIndex", rater_attention_obs, i)
                _add_if_wanted(row, wanted, "DelayedOutcomeTrace", delayed_trace_obs, i)
                rows.append(row)

    return {
        "per_unit": {"utility": latent, "target": latent, "harm": proxy},
        "summary": {
            "latent_target_mean": _mean(latent),
            "observed_metric_mean": _mean(metric),
            "proxy_lever_mean": _mean(proxy),
        },
        "trajectory_means": traj,
        "rows": rows,
    }


def _build_proxy_metric(
    template: Dict[str, Any],
    seed: int,
    horizon: int,
    oracle_rollouts: int,
    forced_sub_variant: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    roles = template["roles"]
    params = {
        "init_latent_mean": rng.uniform(37, 45),
        "init_latent_sd": rng.uniform(6.0, 8.5),
        "init_proxy_mean": rng.uniform(16, 24),
        "init_proxy_sd": rng.uniform(4.0, 7.0),
        "true_gain": {
            "no_change": 0.0,
            "substantive_improvement": rng.uniform(5.8, 7.5),
            "metric_optimization": rng.uniform(-0.4, 0.5),
            "balanced": rng.uniform(3.8, 5.0),
        },
        "proxy_gain": {
            "no_change": 0.0,
            "substantive_improvement": rng.uniform(1.2, 2.4),
            "metric_optimization": rng.uniform(8.5, 11.0),
            "balanced": rng.uniform(3.5, 5.0),
        },
        "proxy_crowding_penalty": rng.uniform(0.006, 0.014),
        "metric_intercept": rng.uniform(5.0, 9.0),
        "metric_latent_weight": rng.uniform(0.52, 0.64),
        "metric_proxy_weight": rng.uniform(0.58, 0.72),
        "latent_process_sd": rng.uniform(1.2, 2.2),
        "proxy_process_sd": rng.uniform(1.2, 2.2),
        "metric_obs_sd": rng.uniform(2.5, 4.5),
        "target_obs_sd": rng.uniform(3.5, 5.5),
        "proxy_obs_sd": rng.uniform(3.0, 5.0),
        "engagement_obs_sd": rng.uniform(4.0, 6.0),
    }
    policies = [
        _policy("no_change", "No change", "Keep the current process unchanged.", "no_change"),
        _policy("substantive_improvement", "Core-practice variant", "Assign a variant that changes frontline practices throughout the cycle.", "substantive_improvement"),
        _policy("metric_optimization", "Reporting-practice variant", "Assign a variant that changes reporting and display practices throughout the cycle.", "metric_optimization"),
        _policy("balanced", "Combined-practice variant", "Assign a variant combining frontline-practice and reporting-practice changes.", "balanced"),
    ]
    cfg = {
        "archetype": "proxy_metric_hacking",
        "sub_variant": "metric_hacking",
        "horizon": horizon,
        "roles": roles,
        "unit": template["unit"],
        "period": template["period"],
        "parameters": params,
        "policies": policies,
        "update_equations_id": "proxy_metric_hacking_v1",
        "observation_equations_id": "proxy_metric_hacking_noisy_proxy_v1",
    }
    scores = evaluate_policies(cfg, policies, oracle_rollouts, seed + 7000)
    best, runner = _top_two(scores, "target_mean")
    margin = best["target_mean"] - runner["target_mean"]
    metric_best = sorted(scores, key=lambda s: s["metrics"]["observed_metric_mean"], reverse=True)[0]
    hack = _policy_by_id(scores, "metric_optimization")
    base = _policy_by_id(scores, "no_change")
    substantive = _policy_by_id(scores, "substantive_improvement")
    checks = [
        _check("metric_hack_tempts_on_observed_metric", metric_best["policy_id"] == "metric_optimization", metric_best["policy_id"], "metric_optimization", "Metric-focused policy is best on the visible metric."),
        _check("metric_hack_not_latent_improvement", hack["target_mean"] <= base["target_mean"] + 2.0, hack["target_mean"] - base["target_mean"], "<= 2.0", "Metric-focused policy has near-zero latent target effect."),
        _check("substantive_improves_latent_target", substantive["target_mean"] >= base["target_mean"] + MIN_CLEAR_EFFECT, substantive["target_mean"] - base["target_mean"], f">= {MIN_CLEAR_EFFECT}", "Substantive policy improves the latent target."),
        _check("gold_margin", margin >= MIN_GOLD_MARGIN, margin, f">= {MIN_GOLD_MARGIN}", "Latent-target best policy beats runner-up."),
        _check("metric_and_gold_differ", metric_best["policy_id"] != best["policy_id"], {"metric_best": metric_best["policy_id"], "latent_best": best["policy_id"]}, "different", "Optimizing the visible metric gives the wrong answer."),
    ]
    question = (
        f"Which candidate policy genuinely improves the underlying outcome measured by {roles['target_obs']}, "
        f"rather than only improving {roles['metric_obs']}? Answer with one policy_id."
    )
    return _assemble_world(
        template=template,
        cfg=cfg,
        variables=_proxy_variables(roles, policies),
        action_variables=[
            _categorical_var(roles["action_var"], [p["policy_id"] for p in policies], "action", "Assigned program or policy variant for the period.", intervenable=True),
        ],
        policies=policies,
        scores=scores,
        checks=checks,
        question=question,
        answer=best["policy_id"],
        question_type="rpg_proxy_metric_hacking",
        objective=f"Choose the policy that maximizes latent target as measured by {roles['target_obs']}, not the proxy dashboard metric.",
        gold_answer={
            "answer_type": "policy_id",
            "policy_id": best["policy_id"],
            "metric_best_policy_id": metric_best["policy_id"],
        },
        runner_up={"policy_id": runner["policy_id"], "target_mean": runner["target_mean"]},
        gold_margin=margin,
        causal_edges=[
            _edge(roles["action_var"], roles["latent_state"]),
            _edge(roles["action_var"], roles["proxy_state"]),
            _edge(roles["latent_state"], roles["target_obs"], kind="observation"),
            _edge(roles["latent_state"], roles["metric_obs"], kind="observation"),
            _edge(roles["proxy_state"], roles["metric_obs"], kind="observation"),
        ],
        seed=seed,
        oracle_rollouts=oracle_rollouts,
    )


def _proxy_variables(roles: Dict[str, str], policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _continuous_var(roles["target_obs"], "latent_target_proxy", "Noisy 0-100 audit of the underlying target construct.", higher_is_better=True),
        _continuous_var(roles["metric_obs"], "visible_metric", "Noisy 0-100 operational metric visible in routine dashboards.", higher_is_better=True),
        _continuous_var(roles["proxy_obs"], "proxy_lever_measure", "Noisy 0-100 measurement of metric-facing activity.", higher_is_better=None),
        _continuous_var(roles["engagement_obs"], "secondary_proxy", "Noisy 0-100 secondary trace measurement.", higher_is_better=True),
        _aux_continuous("BaselineProfileScore", "baseline_proxy", "Noisy 0-100 pre-policy baseline profile summary.", higher_is_better=None),
        _aux_continuous("IndependentAuditScore", "alternate_target_proxy", "Noisy 0-100 independent audit measurement.", higher_is_better=True),
        _aux_continuous("SurfaceComplianceSignal", "metric_proxy", "Noisy 0-100 surface compliance trace.", higher_is_better=None),
        _aux_continuous("ResourceUseIndex", "process_proxy", "Noisy 0-100 trace of process resources used.", higher_is_better=None),
        _aux_continuous("RaterAttentionIndex", "process_proxy", "Noisy 0-100 trace of attention from rating or review systems.", higher_is_better=None),
        _aux_continuous("DelayedOutcomeTrace", "alternate_target_proxy", "Noisy 0-100 delayed trace related to downstream outcomes.", higher_is_better=True),
    ]


# ---------------------------------------------------------------------------
# Archetype 4: latent mediator
# ---------------------------------------------------------------------------

def _rollout_latent_mediator(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    n_units: int,
    seed: int,
    return_rows: bool,
    measurements: Optional[List[str]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    roles = cfg["roles"]
    horizon = int(cfg["horizon"])
    kind = policy["kind"]

    mediator = _clip100(rng.normal(p["init_mediator_mean"], p["init_mediator_sd"], n_units))
    decoy = _clip100(rng.normal(p["init_decoy_mean"], p["init_decoy_sd"], n_units))
    outcome = _clip100(rng.normal(p["init_outcome_mean"], p["init_outcome_sd"], n_units))
    baseline_status = _clip100(0.45 * mediator + 0.45 * outcome + rng.normal(0, 6.0, n_units))
    traj = {
        roles["mediator_state"]: [_mean(mediator)],
        roles["decoy_state"]: [_mean(decoy)],
        roles["outcome_state"]: [_mean(outcome)],
    }
    rows: List[Dict[str, Any]] = []

    for t in range(horizon):
        prev_mediator = mediator.copy()
        program_on = 1.0 if kind == "program" else 0.0
        decoy_focus = 1.0 if kind == "decoy_focus" else 0.0
        mediator = _clip100(
            mediator
            + program_on * p["program_to_mediator"]
            + decoy_focus * p["decoy_focus_to_mediator"]
            - p["mediator_decay"]
            + rng.normal(0, p["mediator_process_sd"], n_units)
        )
        decoy = _clip100(
            decoy
            + program_on * p["program_to_decoy"]
            + decoy_focus * p["decoy_focus_to_decoy"]
            - p["decoy_decay"]
            + rng.normal(0, p["decoy_process_sd"], n_units)
        )
        outcome = _clip100(
            outcome
            + p["mediator_to_outcome"] * ((prev_mediator - p["mediator_reference"]) / 10.0)
            + program_on * p["direct_effect"]
            + rng.normal(0, p["outcome_process_sd"], n_units)
        )
        mediator_obs = _clip100(mediator + rng.normal(0, p["mediator_obs_sd"], n_units))
        decoy_obs = _clip100(decoy + rng.normal(0, p["decoy_obs_sd"], n_units))
        outcome_obs = _clip100(outcome + rng.normal(0, p["outcome_obs_sd"], n_units))
        baseline_obs = _clip100(baseline_status + rng.normal(0, 3.0, n_units))
        mediator_obs_2 = _clip100(0.82 * mediator + 0.12 * decoy + rng.normal(0, p["mediator_obs_sd"] + 1.0, n_units))
        decoy_obs_2 = _clip100(0.15 * mediator + 0.80 * decoy + rng.normal(0, p["decoy_obs_sd"] + 1.0, n_units))
        exposure_obs = _clip100(15.0 + program_on * 65.0 + decoy_focus * 58.0 + rng.normal(0, 5.0, n_units))
        short_outcome_obs = _clip100(0.45 * outcome + 0.35 * mediator + rng.normal(0, 5.5, n_units))
        admin_trace_obs = _clip100(0.35 * decoy + 0.20 * mediator + 20.0 * program_on + rng.normal(0, 6.0, n_units))

        traj[roles["mediator_state"]].append(_mean(mediator))
        traj[roles["decoy_state"]].append(_mean(decoy))
        traj[roles["outcome_state"]].append(_mean(outcome))

        if return_rows:
            wanted = set(measurements or cfg["observed_variable_names"])
            for i in range(n_units):
                row = {
                    "unit_id": i,
                    "time": t + 1,
                    "policy_id": policy["policy_id"],
                    roles["action_var"]: policy["policy_id"],
                }
                _add_if_wanted(row, wanted, roles["mediator_obs"], mediator_obs, i)
                _add_if_wanted(row, wanted, roles["decoy_obs"], decoy_obs, i)
                _add_if_wanted(row, wanted, roles["outcome_obs"], outcome_obs, i)
                _add_if_wanted(row, wanted, "BaselineStatusScore", baseline_obs, i)
                _add_if_wanted(row, wanted, "SecondaryPathwaySurvey", mediator_obs_2, i)
                _add_if_wanted(row, wanted, "AdministrativeFamiliarityScore", decoy_obs_2, i)
                _add_if_wanted(row, wanted, "ProgramExposureLog", exposure_obs, i)
                _add_if_wanted(row, wanted, "ShortTermOutcomeSignal", short_outcome_obs, i)
                _add_if_wanted(row, wanted, "AdministrativeTraceIndex", admin_trace_obs, i)
                rows.append(row)

    return {
        "per_unit": {"utility": outcome, "target": outcome, "harm": decoy},
        "summary": {
            "final_outcome_mean": _mean(outcome),
            "final_mediator_mean": _mean(mediator),
            "final_decoy_mean": _mean(decoy),
        },
        "trajectory_means": traj,
        "rows": rows,
    }


def _build_latent_mediator(
    template: Dict[str, Any],
    seed: int,
    horizon: int,
    oracle_rollouts: int,
    forced_sub_variant: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    roles = template["roles"]
    sub_variant = forced_sub_variant or LATENT_MEDIATOR_SUB_VARIANTS[seed % len(LATENT_MEDIATOR_SUB_VARIANTS)]
    if sub_variant == "mediated_only":
        direct_effect = rng.uniform(0.0, 0.4)
    elif sub_variant == "direct_and_mediated":
        direct_effect = rng.uniform(0.8, 1.4)
    else:
        raise ValueError(f"unknown latent_mediator sub_variant {sub_variant!r}")
    params = {
        "init_mediator_mean": rng.uniform(31, 39),
        "init_mediator_sd": rng.uniform(5.0, 8.0),
        "init_decoy_mean": rng.uniform(31, 39),
        "init_decoy_sd": rng.uniform(5.0, 8.0),
        "init_outcome_mean": rng.uniform(35, 43),
        "init_outcome_sd": rng.uniform(5.0, 8.0),
        "program_to_mediator": rng.uniform(6.2, 8.2),
        "program_to_decoy": rng.uniform(5.5, 7.5),
        "decoy_focus_to_mediator": rng.uniform(-0.2, 0.5),
        "decoy_focus_to_decoy": rng.uniform(7.0, 9.0),
        "mediator_to_outcome": rng.uniform(1.6, 2.2),
        "direct_effect": direct_effect,
        "mediator_reference": rng.uniform(35, 42),
        "mediator_decay": rng.uniform(0.5, 1.1),
        "decoy_decay": rng.uniform(0.4, 1.0),
        "mediator_process_sd": rng.uniform(1.1, 2.0),
        "decoy_process_sd": rng.uniform(1.1, 2.0),
        "outcome_process_sd": rng.uniform(1.2, 2.1),
        "mediator_obs_sd": rng.uniform(3.0, 5.0),
        "decoy_obs_sd": rng.uniform(3.0, 5.0),
        "outcome_obs_sd": rng.uniform(3.5, 5.5),
    }
    policies = [
        _policy("no_program", "No program", "Keep the current process unchanged.", "no_program"),
        _policy("primary_program", "Program X", "Assign program variant X every period.", "program"),
        _policy("decoy_focused_program", "Program Y", "Assign program variant Y every period.", "decoy_focus"),
    ]
    cfg = {
        "archetype": "latent_mediator",
        "sub_variant": sub_variant,
        "horizon": horizon,
        "roles": roles,
        "unit": template["unit"],
        "period": template["period"],
        "parameters": params,
        "policies": policies,
        "update_equations_id": "latent_mediator_v1",
        "observation_equations_id": "latent_mediator_noisy_proxy_v1",
    }
    scores = evaluate_policies(cfg, policies, oracle_rollouts, seed + 8000)
    base = _policy_by_id(scores, "no_program")
    program = _policy_by_id(scores, "primary_program")
    decoy_focus = _policy_by_id(scores, "decoy_focused_program")
    program_outcome_effect = program["target_mean"] - base["target_mean"]
    program_mediator_effect = program["metrics"]["final_mediator_mean"] - base["metrics"]["final_mediator_mean"]
    program_decoy_effect = program["metrics"]["final_decoy_mean"] - base["metrics"]["final_decoy_mean"]
    decoy_focus_outcome_effect = decoy_focus["target_mean"] - base["target_mean"]
    # Gold is a variable name, not a policy.  Margin is the outcome effect gap
    # between the true-mediator intervention and the decoy-focused intervention.
    margin = program_outcome_effect - decoy_focus_outcome_effect
    checks = [
        _check("program_moves_true_mediator", program_mediator_effect >= MIN_CLEAR_EFFECT, program_mediator_effect, f">= {MIN_CLEAR_EFFECT}", "Primary program changes the true mediator."),
        _check("program_also_moves_decoy", program_decoy_effect >= MIN_CLEAR_EFFECT, program_decoy_effect, f">= {MIN_CLEAR_EFFECT}", "Decoy also moves, making the task nontrivial."),
        _check("program_changes_outcome", program_outcome_effect >= MIN_CLEAR_EFFECT, program_outcome_effect, f">= {MIN_CLEAR_EFFECT}", "Primary program changes the final outcome."),
        _check("decoy_focus_not_pathway", decoy_focus_outcome_effect <= program_outcome_effect - MIN_GOLD_MARGIN, {"decoy_effect": decoy_focus_outcome_effect, "program_effect": program_outcome_effect}, f"gap >= {MIN_GOLD_MARGIN}", "Moving the decoy alone does not explain the outcome effect."),
        _check("gold_margin", margin >= MIN_GOLD_MARGIN, margin, f">= {MIN_GOLD_MARGIN}", "True pathway evidence beats decoy pathway evidence."),
    ]
    question = (
        f"Which intermediate measurement is on the actual pathway from {roles['action_var']} to final "
        f"{roles['outcome_obs']}: {roles['mediator_obs']} or {roles['decoy_obs']}? "
        f"Answer with the variable name."
    )
    return _assemble_world(
        template=template,
        cfg=cfg,
        variables=_mediator_variables(roles, policies),
        action_variables=[
            _categorical_var(roles["action_var"], [p["policy_id"] for p in policies], "action", "Assigned program variant for the period.", intervenable=True),
        ],
        policies=policies,
        scores=scores,
        checks=checks,
        question=question,
        answer=roles["mediator_obs"],
        question_type="rpg_latent_mediator",
        objective=f"Identify which intermediate measurement lies on the pathway to {roles['outcome_obs']}.",
        gold_answer={
            "answer_type": "variable_name",
            "variable_name": roles["mediator_obs"],
            "decoy_variable_name": roles["decoy_obs"],
            "mechanism_label": sub_variant,
            "program_outcome_effect": program_outcome_effect,
            "decoy_focus_outcome_effect": decoy_focus_outcome_effect,
        },
        runner_up={"variable_name": roles["decoy_obs"], "supporting_effect": decoy_focus_outcome_effect},
        gold_margin=margin,
        causal_edges=[
            _edge(roles["action_var"], roles["mediator_state"]),
            _edge(roles["action_var"], roles["decoy_state"]),
            _edge(roles["mediator_state"], roles["outcome_state"]),
            _edge(roles["mediator_state"], roles["mediator_obs"], kind="observation"),
            _edge(roles["decoy_state"], roles["decoy_obs"], kind="observation"),
            _edge(roles["outcome_state"], roles["outcome_obs"], kind="observation"),
        ],
        seed=seed,
        oracle_rollouts=oracle_rollouts,
    )


def _mediator_variables(roles: Dict[str, str], policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _continuous_var(roles["mediator_obs"], "candidate_mediator_proxy", "Noisy 0-100 measurement of one candidate intermediate construct.", higher_is_better=True),
        _continuous_var(roles["decoy_obs"], "candidate_mediator_proxy", "Noisy 0-100 measurement of another candidate intermediate construct.", higher_is_better=True),
        _continuous_var(roles["outcome_obs"], "target_proxy", "Noisy 0-100 final outcome measurement.", higher_is_better=True),
        _aux_continuous("BaselineStatusScore", "baseline_proxy", "Noisy 0-100 pre-policy baseline status summary.", higher_is_better=None),
        _aux_continuous("SecondaryPathwaySurvey", "candidate_mediator_proxy", "Noisy 0-100 secondary survey for a candidate pathway.", higher_is_better=True),
        _aux_continuous("AdministrativeFamiliarityScore", "candidate_mediator_proxy", "Noisy 0-100 administrative familiarity measurement.", higher_is_better=True),
        _aux_continuous("ProgramExposureLog", "process_proxy", "Noisy 0-100 trace of program exposure.", higher_is_better=None),
        _aux_continuous("ShortTermOutcomeSignal", "short_term_proxy", "Noisy 0-100 short-term outcome signal.", higher_is_better=True),
        _aux_continuous("AdministrativeTraceIndex", "process_proxy", "Noisy 0-100 administrative trace measurement.", higher_is_better=None),
    ]


# ---------------------------------------------------------------------------
# Archetype 5: heterogeneous response
# ---------------------------------------------------------------------------

def _heterogeneous_actions(policy: Dict[str, Any], subtype_proxy: np.ndarray, t: int, early_response: np.ndarray) -> np.ndarray:
    kind = policy["kind"]
    n = len(subtype_proxy)
    if kind == "always_a":
        return np.zeros(n, dtype=np.int8)
    if kind == "always_b":
        return np.ones(n, dtype=np.int8)
    if kind == "always_c":
        return np.full(n, 2, dtype=np.int8)
    if kind == "proxy_matched":
        return np.where(subtype_proxy == 0, 0, 2).astype(np.int8)
    if kind == "proxy_opposite":
        return np.where(subtype_proxy == 0, 2, 0).astype(np.int8)
    if kind == "early_switch":
        if t < 2:
            return np.ones(n, dtype=np.int8)
        return np.where(early_response < 46.0, 2, 0).astype(np.int8)
    raise KeyError(kind)


def _rollout_heterogeneous(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    n_units: int,
    seed: int,
    return_rows: bool,
    measurements: Optional[List[str]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    roles = cfg["roles"]
    horizon = int(cfg["horizon"])

    subtype = rng.binomial(1, p["subtype1_probability"], n_units).astype(np.int8)
    proxy_correct = rng.random(n_units) < p["subtype_proxy_accuracy"]
    subtype_proxy = np.where(proxy_correct, subtype, 1 - subtype).astype(np.int8)
    severity = _clip100(rng.normal(p["severity_mean"], p["severity_sd"], n_units))
    outcome = _clip100(rng.normal(p["init_outcome_mean"], p["init_outcome_sd"], n_units) - p["severity_outcome_penalty"] * severity)
    harm = _clip100(rng.normal(p["init_harm_mean"], p["init_harm_sd"], n_units))
    early_response = np.full(n_units, 50.0)
    secondary_proxy_correct = rng.random(n_units) < max(0.60, p["subtype_proxy_accuracy"] - 0.12)
    secondary_subtype_proxy = np.where(secondary_proxy_correct, subtype, 1 - subtype).astype(np.int8)
    baseline_severity_obs0 = _clip100(severity + rng.normal(0, 4.0, n_units))

    traj = {
        roles["outcome_state"]: [_mean(outcome)],
        roles["harm_state"]: [_mean(harm)],
    }
    rows: List[Dict[str, Any]] = []

    effect = np.array(p["action_effect"], dtype=float)  # shape [subtype, action]
    harm_gain = np.array(p["action_harm"], dtype=float)

    for t in range(horizon):
        action = _heterogeneous_actions(policy, subtype_proxy, t, early_response)
        unit_effect = effect[subtype, action]
        unit_harm_gain = harm_gain[subtype, action]
        outcome = _clip100(
            outcome
            + unit_effect
            - p["harm_outcome_penalty"] * harm
            - p["severity_period_penalty"] * severity
            + rng.normal(0, p["outcome_process_sd"], n_units)
        )
        harm = _clip100(
            harm
            + unit_harm_gain
            - p["harm_recovery_rate"]
            + rng.normal(0, p["harm_process_sd"], n_units)
        )
        outcome_obs = _clip100(outcome + rng.normal(0, p["outcome_obs_sd"], n_units))
        harm_obs = _clip100(harm + rng.normal(0, p["harm_obs_sd"], n_units))
        baseline_obs = _clip100(baseline_severity_obs0 + rng.normal(0, 2.0, n_units))
        early_response_obs = _clip100(early_response + rng.normal(0, 4.0, n_units))
        resource_obs = _clip100(18.0 + 12.0 * action + unit_effect * 3.0 + rng.normal(0, 4.5, n_units))
        fit_obs = _clip100(55.0 + unit_effect * 5.0 - unit_harm_gain * 3.0 + rng.normal(0, 6.0, n_units))
        engagement_obs = _clip100(0.55 * outcome - 0.25 * harm + rng.normal(0, 6.0, n_units))
        followup_obs = _clip100(62.0 + 2.5 * action - 0.20 * harm + rng.normal(0, 6.0, n_units))
        early_response = outcome_obs

        traj[roles["outcome_state"]].append(_mean(outcome))
        traj[roles["harm_state"]].append(_mean(harm))

        if return_rows:
            wanted = set(measurements or cfg["observed_variable_names"])
            for i in range(n_units):
                row = {
                    "unit_id": i,
                    "time": t + 1,
                    "policy_id": policy["policy_id"],
                    roles["action_var"]: ["A", "B", "C"][int(action[i])],
                }
                if roles["subtype_obs"] in wanted:
                    row[roles["subtype_obs"]] = "TypeA" if subtype_proxy[i] == 0 else "TypeB"
                if "SecondarySubtypeScreen" in wanted:
                    row["SecondarySubtypeScreen"] = "TypeA" if secondary_subtype_proxy[i] == 0 else "TypeB"
                _add_if_wanted(row, wanted, roles["outcome_obs"], outcome_obs, i)
                _add_if_wanted(row, wanted, roles["harm_obs"], harm_obs, i)
                _add_if_wanted(row, wanted, "BaselineSeverityScore", baseline_obs, i)
                _add_if_wanted(row, wanted, "EarlyResponseSignal", early_response_obs, i)
                _add_if_wanted(row, wanted, "ResourceUseIndex", resource_obs, i)
                _add_if_wanted(row, wanted, "ProgramFitSurvey", fit_obs, i)
                _add_if_wanted(row, wanted, "EngagementTrace", engagement_obs, i)
                _add_if_wanted(row, wanted, "FollowupIntensityLog", followup_obs, i)
                rows.append(row)

    utility = outcome - p["utility_harm_weight"] * harm
    subgroup: Dict[str, Dict[str, float]] = {}
    for g, label in [(0, "SubtypeA"), (1, "SubtypeB")]:
        mask = subtype == g
        subgroup[label] = {
            "n": int(np.sum(mask)),
            "utility_mean": _mean(utility[mask]) if np.any(mask) else float("nan"),
            "target_mean": _mean(outcome[mask]) if np.any(mask) else float("nan"),
            "harm_mean": _mean(harm[mask]) if np.any(mask) else float("nan"),
        }

    return {
        "per_unit": {
            "utility": utility,
            "target": outcome,
            "harm": harm,
        },
        "summary": {
            "final_target_mean": _mean(outcome),
            "final_harm_mean": _mean(harm),
            "subtype_proxy_accuracy": p["subtype_proxy_accuracy"],
        },
        "subgroup": subgroup,
        "trajectory_means": traj,
        "rows": rows,
    }


def _build_heterogeneous(
    template: Dict[str, Any],
    seed: int,
    horizon: int,
    oracle_rollouts: int,
    forced_sub_variant: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    roles = template["roles"]
    a0 = rng.uniform(6.8, 8.2)
    c1 = rng.uniform(6.8, 8.2)
    b = rng.uniform(3.8, 5.0)
    params = {
        "subtype1_probability": rng.uniform(0.42, 0.58),
        "subtype_proxy_accuracy": rng.uniform(0.78, 0.88),
        "severity_mean": rng.uniform(35, 45),
        "severity_sd": rng.uniform(7.0, 10.0),
        "init_outcome_mean": rng.uniform(47, 55),
        "init_outcome_sd": rng.uniform(5.0, 8.0),
        "init_harm_mean": rng.uniform(8, 14),
        "init_harm_sd": rng.uniform(3.0, 5.0),
        "severity_outcome_penalty": rng.uniform(0.08, 0.14),
        "severity_period_penalty": rng.uniform(0.010, 0.025),
        "harm_outcome_penalty": rng.uniform(0.055, 0.085),
        "harm_recovery_rate": rng.uniform(0.8, 1.3),
        "utility_harm_weight": rng.uniform(0.35, 0.50),
        # rows: true subtype A/B; columns: action A/B/C
        "action_effect": [
            [a0, b, rng.uniform(0.4, 1.6)],
            [rng.uniform(0.4, 1.6), b, c1],
        ],
        "action_harm": [
            [rng.uniform(1.1, 1.8), rng.uniform(1.0, 1.6), rng.uniform(4.8, 6.5)],
            [rng.uniform(4.8, 6.5), rng.uniform(1.0, 1.6), rng.uniform(1.1, 1.8)],
        ],
        "outcome_process_sd": rng.uniform(1.4, 2.4),
        "harm_process_sd": rng.uniform(1.0, 1.8),
        "outcome_obs_sd": rng.uniform(3.0, 5.0),
        "harm_obs_sd": rng.uniform(3.0, 5.0),
    }
    policies = [
        _policy("always_a", "Always A", "Assign option A to every unit.", "always_a"),
        _policy("always_b", "Always B", "Assign option B to every unit.", "always_b"),
        _policy("always_c", "Always C", "Assign option C to every unit.", "always_c"),
        _policy("matched_to_screen", "Matched to screen", f"Assign A when {roles['subtype_obs']} is TypeA and C when it is TypeB.", "proxy_matched"),
        _policy("opposite_screen_rule", "Alternative screen rule", f"Assign C when {roles['subtype_obs']} is TypeA and A when it is TypeB.", "proxy_opposite"),
        _policy("early_response_switch", "Early response switch", "Start with B, then switch based on early observed response.", "early_switch"),
    ]
    cfg = {
        "archetype": "heterogeneous_response",
        "sub_variant": "observed_proxy_policy",
        "horizon": horizon,
        "roles": roles,
        "unit": template["unit"],
        "period": template["period"],
        "parameters": params,
        "policies": policies,
        "update_equations_id": "heterogeneous_response_v1",
        "observation_equations_id": "heterogeneous_response_noisy_proxy_v1",
    }
    scores = evaluate_policies(cfg, policies, oracle_rollouts, seed + 9000)
    best, runner = _top_two(scores, "expected_utility")
    margin = best["expected_utility"] - runner["expected_utility"]
    best_static = sorted(
        [s for s in scores if s["policy_id"] in {"always_a", "always_b", "always_c"}],
        key=lambda s: s["expected_utility"],
        reverse=True,
    )[0]
    matched = _policy_by_id(scores, "matched_to_screen")
    always_a = _policy_by_id(scores, "always_a")
    always_c = _policy_by_id(scores, "always_c")
    subtype_a_gap = always_a["subgroup_metrics"]["SubtypeA"]["utility_mean"] - always_c["subgroup_metrics"]["SubtypeA"]["utility_mean"]
    subtype_b_gap = always_c["subgroup_metrics"]["SubtypeB"]["utility_mean"] - always_a["subgroup_metrics"]["SubtypeB"]["utility_mean"]
    checks = [
        _check("adaptive_policy_best", best["policy_id"] == "matched_to_screen", best["policy_id"], "matched_to_screen", "Screen-matched policy is the best policy."),
        _check("adaptive_beats_static", matched["expected_utility"] >= best_static["expected_utility"] + MIN_CLEAR_EFFECT, matched["expected_utility"] - best_static["expected_utility"], f">= {MIN_CLEAR_EFFECT}", "Adaptive policy beats the best static policy."),
        _check("subtype_a_prefers_a", subtype_a_gap >= MIN_CLEAR_EFFECT, subtype_a_gap, f">= {MIN_CLEAR_EFFECT}", "Subtype A benefits more from action A than action C."),
        _check("subtype_b_prefers_c", subtype_b_gap >= MIN_CLEAR_EFFECT, subtype_b_gap, f">= {MIN_CLEAR_EFFECT}", "Subtype B benefits more from action C than action A."),
        _check("proxy_informative_not_perfect", 0.70 <= params["subtype_proxy_accuracy"] <= 0.92, params["subtype_proxy_accuracy"], "0.70..0.92", "Observed subtype proxy is informative but imperfect."),
        _check("gold_margin", margin >= MIN_GOLD_MARGIN, margin, f">= {MIN_GOLD_MARGIN}", "Best policy beats runner-up."),
    ]
    question = (
        f"Which candidate policy gives the best expected final utility for heterogeneous {template['unit']} "
        f"when utility rewards {roles['outcome_obs']} and penalizes {roles['harm_obs']}? "
        f"Answer with one policy_id."
    )
    return _assemble_world(
        template=template,
        cfg=cfg,
        variables=_hetero_variables(roles, policies),
        action_variables=[
            _categorical_var(roles["action_var"], ["A", "B", "C"], "action", "Assigned treatment or pathway option for the period.", intervenable=True),
        ],
        policies=policies,
        scores=scores,
        checks=checks,
        question=question,
        answer=best["policy_id"],
        question_type="rpg_heterogeneous_response",
        objective=f"Maximize final utility: {roles['outcome_obs']} minus harm penalty for {roles['harm_obs']}.",
        gold_answer={
            "answer_type": "policy_id",
            "policy_id": best["policy_id"],
            "best_static_policy_id": best_static["policy_id"],
            "subtype_proxy_accuracy": params["subtype_proxy_accuracy"],
        },
        runner_up={"policy_id": runner["policy_id"], "expected_utility": runner["expected_utility"]},
        gold_margin=margin,
        causal_edges=[
            _edge(roles["subtype_param"], roles["outcome_state"]),
            _edge(roles["subtype_param"], roles["harm_state"]),
            _edge(roles["subtype_param"], roles["subtype_obs"], kind="observation"),
            _edge(roles["action_var"], roles["outcome_state"]),
            _edge(roles["action_var"], roles["harm_state"]),
            _edge(roles["outcome_state"], roles["outcome_obs"], kind="observation"),
            _edge(roles["harm_state"], roles["harm_obs"], kind="observation"),
        ],
        seed=seed,
        oracle_rollouts=oracle_rollouts,
    )


def _hetero_variables(roles: Dict[str, str], policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _categorical_var(roles["subtype_obs"], ["TypeA", "TypeB"], "baseline_proxy", "Noisy baseline screen for response subtype.", observed=True, intervenable=False),
        _continuous_var(roles["outcome_obs"], "target_proxy", "Noisy 0-100 final outcome measurement.", higher_is_better=True),
        _continuous_var(roles["harm_obs"], "harm_proxy", "Noisy 0-100 burden or side-effect measurement.", higher_is_better=False),
        _categorical_var("SecondarySubtypeScreen", ["TypeA", "TypeB"], "baseline_proxy", "Second noisy baseline screen for response subtype.", observed=True, intervenable=False),
        _aux_continuous("BaselineSeverityScore", "baseline_proxy", "Noisy 0-100 pre-policy severity or barrier score.", higher_is_better=False),
        _aux_continuous("EarlyResponseSignal", "short_term_proxy", "Noisy 0-100 early response signal from the current path.", higher_is_better=True),
        _aux_continuous("ResourceUseIndex", "process_proxy", "Noisy 0-100 trace of resources used by the assigned option.", higher_is_better=None),
        _aux_continuous("ProgramFitSurvey", "process_proxy", "Noisy 0-100 survey of fit between unit and assigned option.", higher_is_better=True),
        _aux_continuous("EngagementTrace", "process_proxy", "Noisy 0-100 participation trace during the period.", higher_is_better=True),
        _aux_continuous("FollowupIntensityLog", "process_proxy", "Noisy 0-100 trace of follow-up intensity.", higher_is_better=None),
    ]


# ---------------------------------------------------------------------------
# World assembly
# ---------------------------------------------------------------------------

def _state_variables_for(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    roles = cfg["roles"]
    arch = cfg["archetype"]
    if arch == "delayed_harm":
        return [
            _continuous_var(roles["burden_state"], "hidden_state", "Hidden burden state in the simulator.", higher_is_better=False, observed=False),
            _continuous_var(roles["harm_state"], "hidden_state", "Hidden accumulated harm state in the simulator.", higher_is_better=False, observed=False),
            _continuous_var(roles["adherence_state"], "hidden_state", "Hidden participation or adherence state in the simulator.", higher_is_better=True, observed=False),
            _continuous_var(roles["target_state"], "hidden_state", "Hidden final target state in the simulator.", higher_is_better=True, observed=False),
        ]
    if arch == "dose_response":
        return [
            _continuous_var(roles["capacity_state"], "hidden_state", "Hidden capacity state in the simulator.", higher_is_better=True, observed=False),
            _continuous_var(roles["strain_state"], "hidden_state", "Hidden strain state in the simulator.", higher_is_better=False, observed=False),
            _continuous_var(roles["target_state"], "hidden_state", "Hidden final target state in the simulator.", higher_is_better=True, observed=False),
        ]
    if arch == "proxy_metric_hacking":
        return [
            _continuous_var(roles["latent_state"], "hidden_state", "Hidden latent target state in the simulator.", higher_is_better=True, observed=False),
            _continuous_var(roles["proxy_state"], "hidden_state", "Hidden metric-facing state in the simulator.", higher_is_better=None, observed=False),
        ]
    if arch == "latent_mediator":
        return [
            _continuous_var(roles["mediator_state"], "hidden_state", "Hidden true mediator state in the simulator.", higher_is_better=True, observed=False),
            _continuous_var(roles["decoy_state"], "hidden_state", "Hidden decoy intermediate state in the simulator.", higher_is_better=True, observed=False),
            _continuous_var(roles["outcome_state"], "hidden_state", "Hidden final outcome state in the simulator.", higher_is_better=True, observed=False),
        ]
    if arch == "heterogeneous_response":
        return [
            _categorical_var(roles["subtype_param"], ["SubtypeA", "SubtypeB"], "unit_parameter", "Hidden response subtype.", observed=False),
            _continuous_var(roles["outcome_state"], "hidden_state", "Hidden final target state in the simulator.", higher_is_better=True, observed=False),
            _continuous_var(roles["harm_state"], "hidden_state", "Hidden harm or burden state in the simulator.", higher_is_better=False, observed=False),
        ]
    raise KeyError(arch)


def _default_policy_id_for(cfg: Dict[str, Any], policies: List[Dict[str, Any]]) -> Optional[str]:
    """Choose the reference policy for observational trajectory queries."""
    arch = cfg["archetype"]
    preferred_kind = {
        "delayed_harm": "always_standard",
        "dose_response": "static_dose",
        "proxy_metric_hacking": "no_change",
        "latent_mediator": "no_program",
        "heterogeneous_response": "always_b",
    }.get(arch)
    for p in policies:
        if arch == "dose_response" and int(p.get("params", {}).get("dose_code", -1)) == 0:
            return p["policy_id"]
        if preferred_kind is not None and p.get("kind") == preferred_kind:
            return p["policy_id"]
    return policies[0]["policy_id"] if policies else None


def _assemble_world(
    *,
    template: Dict[str, Any],
    cfg: Dict[str, Any],
    variables: List[Dict[str, Any]],
    action_variables: List[Dict[str, Any]],
    policies: List[Dict[str, Any]],
    scores: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
    question: str,
    answer: str,
    question_type: str,
    objective: str,
    gold_answer: Dict[str, Any],
    runner_up: Dict[str, Any],
    gold_margin: float,
    causal_edges: List[Dict[str, Any]],
    seed: int,
    oracle_rollouts: int,
) -> Dict[str, Any]:
    accepted = _all_pass(checks)
    horizon = int(cfg["horizon"])
    world_id = f"rpg_{cfg['archetype']}_{_safe_id(template['subdomain'])}_seed{seed}"
    story = _story(template, horizon)
    hidden_state_vars = _state_variables_for(cfg)
    default_policy_id = _default_policy_id_for(cfg, policies)
    policies, policy_aliases = _alias_policies_for_output(policies, seed)
    if default_policy_id is not None:
        default_policy_id = policy_aliases.get(default_policy_id, default_policy_id)
    scores = _replace_policy_id_strings(scores, policy_aliases)
    variables = _replace_policy_id_strings(variables, policy_aliases)
    action_variables = _replace_policy_id_strings(action_variables, policy_aliases)
    gold_answer = _replace_policy_id_strings(gold_answer, policy_aliases)
    runner_up = _replace_policy_id_strings(runner_up, policy_aliases)
    answer = _replace_policy_id_strings(answer, policy_aliases)
    all_variables_by_name = {v["name"]: v for v in variables}
    for action_var in action_variables:
        all_variables_by_name.setdefault(action_var["name"], action_var)
    all_variables = list(all_variables_by_name.values())
    observed_names = [v["name"] for v in variables if v.get("observed", True) and not v.get("intervenable", False)]
    hidden_cfg = dict(cfg)
    hidden_cfg["policies"] = policies
    hidden_cfg["policy_id_aliases"] = policy_aliases
    hidden_cfg["observed_variable_names"] = observed_names
    hidden_cfg["default_policy_id"] = default_policy_id
    hidden_cfg["schema_version"] = SCHEMA_VERSION
    hidden_cfg["world_seed"] = seed

    world = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "benchmark": BENCHMARK_NAME,
            "world_id": world_id,
            "archetype": cfg["archetype"],
            "sub_variant": cfg.get("sub_variant"),
            "topic": template["topic"],
            "subdomain": template["subdomain"],
            "seed": seed,
            "world_seed": seed,
            "parameter_seed": seed,
            "unit_seed": seed + 111,
            "noise_seed": seed + 222,
            "oracle_seed": seed + 333,
            "horizon": horizon,
            "oracle_n_rollouts": oracle_rollouts,
            "n_candidate_policies": len(policies),
            "n_questions": 1,
            "n_visible_observed_variables": len(observed_names),
            "n_visible_action_variables": len(action_variables),
            "n_hidden_state_variables": len(hidden_state_vars),
            "generator": "world_gen_rpg.py",
        },
        "story": story,
        "variables": all_variables,
        "non_intervenable_variables": _non_intervenable(all_variables),
        # Kept for continuity with old inspectors. RPG worlds do not expose a
        # public static BN graph; hidden dynamic edges live under hidden.
        "edges": [],
        "edge_visibility_note": (
            "RPG causal edges involve hidden simulator state and are stored in "
            "hidden.causal_edges_unrolled_template. They are not agent-visible."
        ),
        "cpds": [],
        "visible": _visible_block(story, all_variables, action_variables, policies, question, horizon, default_policy_id),
        "hidden": {
            "state_variables": hidden_state_vars,
            "latent_variables": hidden_state_vars,
            "agent_visible": False,
            "visibility_note": (
                "These variables are simulator internals. They are excluded "
                "from the agent-facing world information and should only be "
                "used by the simulator/oracle."
            ),
            "unit_parameters": {
                "unit": template["unit"],
                "population": "synthetic units sampled independently from hidden unit distributions",
            },
            "mechanism_parameters": cfg["parameters"],
            "simulator_config": hidden_cfg,
            "update_equations_id": cfg["update_equations_id"],
            "observation_equations_id": cfg["observation_equations_id"],
            "causal_edges_unrolled_template": causal_edges,
        },
        "variable_visibility": {
            "agent_visible_observed": observed_names,
            "agent_visible_actions": [v["name"] for v in action_variables],
            "hidden_latent": [v["name"] for v in hidden_state_vars],
        },
        "oracle": {
            "objective": objective,
            "policy_scores": scores,
            "gold_answer": _jsonify(gold_answer),
            "runner_up": _jsonify(runner_up),
            "gold_margin": float(gold_margin),
            "oracle_n_rollouts": oracle_rollouts,
        },
        "validators": {
            "accepted": accepted,
            "signature_checks": checks,
        },
        "questions": [
            {
                "id": 0,
                "question_type": question_type,
                "difficulty": "hard",
                "question": question,
                "answer": answer,
                "metadata": {
                    "archetype": cfg["archetype"],
                    "sub_variant": cfg.get("sub_variant"),
                    "roles": cfg["roles"],
                    "gold": _jsonify(gold_answer),
                    "oracle_policy_scores": scores,
                },
            }
        ],
    }
    return world


# ---------------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------------

BUILDERS = {
    "delayed_harm": _build_delayed_harm,
    "dose_response": _build_dose_response,
    "proxy_metric_hacking": _build_proxy_metric,
    "latent_mediator": _build_latent_mediator,
    "heterogeneous_response": _build_heterogeneous,
}


def _balanced_subvariant(arch: str, i: int, count: int) -> Optional[str]:
    if arch == "dose_response":
        variants = DOSE_SUB_VARIANTS
    elif arch == "latent_mediator":
        variants = LATENT_MEDIATOR_SUB_VARIANTS
    elif arch == "delayed_harm":
        variants = ["safety_constrained_long_horizon"]
    elif arch == "proxy_metric_hacking":
        variants = ["metric_hacking"]
    elif arch == "heterogeneous_response":
        variants = ["observed_proxy_policy"]
    else:
        variants = [None]
    if variants == [None]:
        return None
    # Exact balance when count is divisible; otherwise extras rotate through
    # variants rather than being controlled by seed/rejection luck.
    return variants[i % len(variants)]


def _slot_distribution(distribution: Dict[str, int], seed_base: int) -> List[DatasetSlot]:
    slots: List[DatasetSlot] = []
    for arch in ARCHETYPES:
        count = int(distribution.get(arch, 0))
        n_templates = len(SCENARIOS[arch])
        for i in range(count):
            slots.append(DatasetSlot(
                archetype=arch,
                sub_variant=_balanced_subvariant(arch, i, count),
                template_index=i % n_templates,
            ))
    rng = random.Random(seed_base * 7919 + 17)
    rng.shuffle(slots)
    return slots


def generate_world(
    archetype: str,
    seed: int,
    outdir: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    oracle_rollouts: int = DEFAULT_ORACLE_ROLLOUTS,
    max_attempts: int = 12,
    template_index: Optional[int] = None,
    sub_variant: Optional[str] = None,
) -> WorldBuildResult:
    if archetype not in BUILDERS:
        raise KeyError(f"unknown archetype {archetype!r}")
    os.makedirs(outdir, exist_ok=True)
    templates = SCENARIOS[archetype]
    last_reason = "no attempts"
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 9973
        idx = template_index if template_index is not None else (seed + attempt) % len(templates)
        template = templates[idx % len(templates)]
        world = BUILDERS[archetype](template, attempt_seed, horizon, oracle_rollouts, sub_variant)
        if world["validators"]["accepted"]:
            world_id = world["meta"]["world_id"]
            json_path = os.path.join(outdir, f"world_{world_id}.json")
            world["meta"]["json_path"] = json_path
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(world, f, ensure_ascii=False, indent=2)
            return WorldBuildResult(
                world=world,
                json_path=json_path,
                archetype=archetype,
                sub_variant=world["meta"].get("sub_variant"),
                topic=world["meta"]["topic"],
                seed=attempt_seed,
            )
        failed = [c for c in world["validators"]["signature_checks"] if not c.get("passed")]
        last_reason = "; ".join(f"{c['name']}={c['value']}" for c in failed[:3]) or "validator rejected"
    raise RuntimeError(f"{archetype} failed validation after {max_attempts} attempts: {last_reason}")


def generate_dataset(
    *,
    outdir: str,
    distribution: Dict[str, int],
    seed_base: int = 4000,
    horizon: int = DEFAULT_HORIZON,
    oracle_rollouts: int = DEFAULT_ORACLE_ROLLOUTS,
    max_attempts_per_world: int = 12,
    only_archetype: Optional[str] = None,
) -> List[WorldBuildResult]:
    slots = _slot_distribution(distribution, seed_base)
    if only_archetype:
        slots = [s for s in slots if s.archetype == only_archetype]
    os.makedirs(outdir, exist_ok=True)
    results: List[WorldBuildResult] = []
    skipped: List[Tuple[int, str, str]] = []
    t0 = time.time()
    total = len(slots)
    for i, slot in enumerate(slots):
        arch = slot.archetype
        seed = seed_base + i * 101
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (total - i) if i else 0.0
        print(f"\n[{i + 1}/{total}] {arch} seed={seed} elapsed={elapsed:.0f}s eta={eta:.0f}s")
        try:
            result = generate_world(
                arch,
                seed,
                outdir,
                horizon=horizon,
                oracle_rollouts=oracle_rollouts,
                max_attempts=max_attempts_per_world,
                template_index=slot.template_index,
                sub_variant=slot.sub_variant,
            )
            print(
                f"  [ok] {result.archetype}/{result.sub_variant or '-'} "
                f"topic={result.topic!r} path={os.path.basename(result.json_path)}"
            )
            results.append(result)
        except Exception as e:
            print(f"  [skip] {arch}: {e}")
            skipped.append((i, arch, str(e)))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "outdir": outdir,
        "seed_base": seed_base,
        "horizon": horizon,
        "oracle_rollouts": oracle_rollouts,
        "requested_distribution": distribution,
        "generated": len(results),
        "skipped": [{"slot": i, "archetype": a, "reason": r} for i, a, r in skipped],
        "worlds": [
            {
                "path": r.json_path,
                "archetype": r.archetype,
                "sub_variant": r.sub_variant,
                "topic": r.topic,
                "seed": r.seed,
            }
            for r in results
        ],
    }
    manifest_path = os.path.join(outdir, "manifest_rpg_v1.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    generated_paths = {os.path.abspath(r.json_path) for r in results}
    stale_removed: List[str] = []
    for name in os.listdir(outdir):
        if not (name.startswith("world_") and name.endswith(".json")):
            continue
        path = os.path.abspath(os.path.join(outdir, name))
        if path not in generated_paths:
            os.remove(path)
            stale_removed.append(name)
    if stale_removed:
        manifest["stale_removed"] = sorted(stale_removed)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Removed {len(stale_removed)} stale world file(s) from {outdir}")
    if skipped:
        print(f"\nWARNING: skipped {len(skipped)} worlds. See {manifest_path}")
    print(f"\nGenerated {len(results)}/{total} RPG worlds in {outdir} ({time.time() - t0:.0f}s)")
    return results


def validate_world_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        world = json.load(f)
    issues: List[str] = []
    if world.get("schema_version") != SCHEMA_VERSION:
        issues.append("wrong schema_version")
    for key in ["meta", "story", "variables", "visible", "hidden", "oracle", "validators", "questions"]:
        if key not in world:
            issues.append(f"missing {key}")
    if not world.get("validators", {}).get("accepted", False):
        issues.append("validators.accepted is false")
    if not world.get("questions"):
        issues.append("missing question")
    if world.get("questions") and world["questions"][0].get("answer") != world.get("oracle", {}).get("gold_answer", {}).get("policy_id", world["questions"][0].get("answer")):
        # Mediator worlds use variable_name instead of policy_id, so this is
        # intentionally a soft consistency check below.
        gold = world.get("oracle", {}).get("gold_answer", {})
        answer = world["questions"][0].get("answer")
        if answer not in {gold.get("policy_id"), gold.get("variable_name")}:
            issues.append("question answer does not match oracle gold")
    visibility = world.get("variable_visibility", {})
    public_names = set(visibility.get("agent_visible_observed", [])) | set(visibility.get("agent_visible_actions", []))
    hidden_names = set(visibility.get("hidden_latent", []))
    if public_names & hidden_names:
        issues.append("hidden latent listed as agent-visible")
    top_level_var_names = {v.get("name") for v in world.get("variables", [])}
    if hidden_names & top_level_var_names:
        issues.append("hidden latent present in top-level variables")
    if top_level_var_names != public_names:
        issues.append("top-level variables do not exactly match agent-visible variables")
    top_level_edges = world.get("edges", [])
    leaked_edges = [
        edge for edge in top_level_edges
        if isinstance(edge, list) and len(edge) >= 2 and (edge[0] in hidden_names or edge[1] in hidden_names)
    ]
    if leaked_edges:
        issues.append("hidden latent present in top-level edges")
    visible = world.get("visible", {})
    allowed_measurements = set(visible.get("allowed_measurements", []))
    if allowed_measurements != set(visibility.get("agent_visible_observed", [])):
        issues.append("visible.allowed_measurements does not match variable_visibility.agent_visible_observed")
    visible_var_names = {v.get("name") for v in visible.get("observed_variables", [])}
    if visible_var_names != set(visibility.get("agent_visible_observed", [])):
        issues.append("visible.observed_variables does not match variable_visibility.agent_visible_observed")
    budget = visible.get("experiment_budget", {})
    if budget.get("sample_accounting") != DISCOVERY_SAMPLE_ACCOUNTING:
        issues.append("visible.experiment_budget sample_accounting mismatch")
    for key in ["max_total_samples", "max_samples_per_query", "default_units", "max_units", "max_queries"]:
        if not isinstance(budget.get(key), int) or budget.get(key) <= 0:
            issues.append(f"visible.experiment_budget.{key} missing or nonpositive")
    if visible.get("discovery_protocol", {}).get("task_style") != "budgeted_iterative_scientific_discovery":
        issues.append("visible.discovery_protocol missing budgeted discovery task style")
    return {"path": path, "ok": not issues, "issues": issues}


def validate_dataset_dir(outdir: str) -> Dict[str, Any]:
    paths = sorted(
        os.path.join(outdir, p)
        for p in os.listdir(outdir)
        if p.startswith("world_") and p.endswith(".json")
    )
    reports = [validate_world_file(p) for p in paths]
    manifest_report: Dict[str, Any] = {"present": False}
    manifest_path = os.path.join(outdir, "manifest_rpg_v1.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        actual = {os.path.basename(p) for p in paths}
        listed = {os.path.basename(w["path"]) for w in manifest.get("worlds", [])}
        manifest_report = {
            "present": True,
            "generated": manifest.get("generated"),
            "world_count": len(manifest.get("worlds", [])),
            "extra_world_files": sorted(actual - listed),
            "missing_world_files": sorted(listed - actual),
        }
    return {
        "n_worlds": len(paths),
        "n_ok": sum(1 for r in reports if r["ok"]),
        "manifest": manifest_report,
        "reports": reports,
    }


def _parse_distribution(raw: Optional[str]) -> Dict[str, int]:
    if not raw:
        return dict(DEFAULT_DISTRIBUTION)
    data = json.loads(raw)
    unknown = set(data) - set(ARCHETYPES)
    if unknown:
        raise ValueError(f"unknown archetypes in distribution: {sorted(unknown)}")
    return {a: int(data.get(a, 0)) for a in ARCHETYPES}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate ACED RPG v1 dynamic simulator worlds.",
    )
    ap.add_argument("--outdir", type=str, default="./out_rpg_v1")
    ap.add_argument("--seed-base", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--oracle-rollouts", type=int, default=DEFAULT_ORACLE_ROLLOUTS)
    ap.add_argument("--max-attempts-per-world", type=int, default=12)
    ap.add_argument("--only-archetype", type=str, choices=ARCHETYPES, default=None)
    ap.add_argument(
        "--distribution",
        type=str,
        default=None,
        help='JSON counts, e.g. \'{"delayed_harm":2,"dose_response":2}\'',
    )
    ap.add_argument("--validate-only", action="store_true", help="Validate an existing outdir instead of generating.")
    args = ap.parse_args()

    if args.validate_only:
        report = validate_dataset_dir(args.outdir)
        print(json.dumps(report, indent=2))
        manifest = report.get("manifest", {})
        manifest_bad = bool(
            manifest.get("extra_world_files") or manifest.get("missing_world_files")
        )
        if report["n_ok"] != report["n_worlds"] or manifest_bad:
            raise SystemExit(1)
        return

    distribution = _parse_distribution(args.distribution)
    generate_dataset(
        outdir=args.outdir,
        distribution=distribution,
        seed_base=args.seed_base,
        horizon=args.horizon,
        oracle_rollouts=args.oracle_rollouts,
        max_attempts_per_world=args.max_attempts_per_world,
        only_archetype=args.only_archetype,
    )


# ==============================================================================
# STATIC RPG v2: partially observed simulator worlds, no fixed action menu
# ==============================================================================
#
# Everything below here implements the design in
# `worldgen_rpg_plan_static_partial_observation.md`. It does not share state
# with the dynamic (time-based) generator above; it lives in the same file
# so we can keep one CLI (`python world_gen_rpg.py --static ...`) and one
# `_rpg` naming convention.
#
# Two optional LLM hooks (off by default, gated by CLI flags):
#
#   --llm-polish               After each world is built, Opus 4.8 rewrites the
#                              story, question, and variable descriptions into
#                              smooth natural language. Mechanism, oracle,
#                              validators, and variable NAMES are not touched.
#                              A leakage check rejects any polish that mentions
#                              hidden variable names or roles (`decoy`,
#                              `true lever`, latent constructs, etc.).
#
#   --llm-extra-templates N    Before generation, Opus 4.8 proposes N fresh
#                              domain templates per archetype. Mechanism code
#                              still owns the role -> variable mapping; the
#                              LLM only invents names/scenarios. Schema-
#                              validated before use.
#
# Both hooks are *off* by default so the deterministic code path keeps
# working when AWS credentials are unavailable.

SCHEMA_VERSION_STATIC = "rpg_static_v2"
BENCHMARK_NAME_STATIC = "aced_rpg_static_v2"

STATIC_ARCHETYPES = [
    "hidden_cause",
    "confounded_action",
    "mechanism_chain",
    "negative_control",
    "hidden_subtype",
    "anomaly_discovery",
    "latent_regime_discovery",
]
STATIC_DEFAULT_DISTRIBUTION: Dict[str, int] = {
    "hidden_cause": 6,
    "confounded_action": 6,
    "mechanism_chain": 6,
    "negative_control": 6,
    "hidden_subtype": 6,
    "anomaly_discovery": 6,
    # Experimental v3-style archetype. Kept at 0 so existing all-six
    # generation commands remain stable unless the distribution requests it.
    "latent_regime_discovery": 0,
}

STATIC_DEFAULT_ORACLE_N = 50000
STATIC_MAX_TOTAL_SAMPLES = 12000
STATIC_MAX_SAMPLES_PER_QUERY = 4000
STATIC_MAX_UNITS_PER_QUERY = 400
STATIC_MAX_MEASUREMENTS_PER_QUERY = 3
STATIC_MAX_QUERIES = 8
STATIC_MIN_GOLD_MARGIN = 4.0
STATIC_MIXTURE_WEIGHTS = [0.0, 0.10, 0.20]
STATIC_ORACLE_TOLERANCE_FRACTION = 0.5

# Recoverability band thresholds: small-budget naive must miss often,
# medium-budget naive must hit often.
STATIC_RECOVER_SMALL_N = 80
STATIC_RECOVER_MEDIUM_N = 400
STATIC_RECOVER_N_SEEDS = 30
STATIC_RECOVER_SMALL_MAX = 0.40
STATIC_RECOVER_MEDIUM_MIN = 0.70


# ---------------------------------------------------------------------------
# Templates (neutral domain names; mechanism lives in code).
# ---------------------------------------------------------------------------

STATIC_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "hidden_cause": [
        {
            "topic": "Hospital data",
            "subdomain": "chronic upper-GI symptoms",
            "setting": (
                "A regional clinic network tracks a stable population of "
                "patients with persistent upper-GI complaints. Current "
                "guidelines focus on stress, diet, and acid suppression. The "
                "clinic team can run small intake studies on freshly enrolled "
                "patients to test other approaches."
            ),
            "unit": "patient",
            "names": {
                "primary_target_obs": "SymptomReport",
                "secondary_target_obs": "EndoscopyFindingScore",
                # Neutral lab panel: NO semantic pair with true_lever_knob.
                "latent_driver_proxy": "SerumPanelBReading",
                # Decoy proxies intentionally semantically MATCH the decoy
                # knobs so a name-pattern-matching LLM is *actively misled*.
                "decoy_proxy_a": "StressInventoryScore",
                "decoy_proxy_b": "DietProfileScore",
                "tertiary_obs": "QualityOfLifeIndex",
                "decoy_knob_a": "StressReductionProgram",
                "decoy_knob_b": "DietModification",
                "weak_knob": "AntacidDose",
                # Neutral name for the true lever: no semantic link to
                # SerumPanelBReading.
                "true_lever_knob": "OralRegimenM",
            },
            "knob_descriptions": {
                "StressReductionProgram": "Eight-week structured stress-reduction program.",
                "DietModification": "Switch to a bland low-acid diet for the study window.",
                "AntacidDose": "Daily proton-pump-inhibitor dose: off, low (20mg), or high (40mg).",
                "OralRegimenM": "A 14-day course of a daily oral medication regimen (Regimen M) per clinic protocol.",
            },
            "measurement_descriptions": {
                "SymptomReport": "Patient-reported 0-100 symptom-burden score (lower is better).",
                "EndoscopyFindingScore": "Clinician-rated 0-100 endoscopy severity (lower is better).",
                "SerumPanelBReading": "Routine 0-100 serum-panel B reading collected during the study window (higher = more reactive on the assay).",
                "StressInventoryScore": "Self-reported 0-100 stress index (higher = more stress).",
                "DietProfileScore": "Dietitian-rated 0-100 acidic-food exposure (higher = more exposure).",
                "QualityOfLifeIndex": "Independent 0-100 quality-of-life summary (higher is better).",
            },
        },
        {
            "topic": "Agriculture",
            "subdomain": "stagnant orchard yield",
            "setting": (
                "An orchard cooperative tracks blocks of trees with chronically "
                "low fruit yield. Local advice focuses on irrigation, soil "
                "amendment, and pruning. Researchers can run small treated-plot "
                "studies on freshly surveyed blocks."
            ),
            "unit": "tree block",
            "names": {
                "primary_target_obs": "YieldShortfall",
                "secondary_target_obs": "LeafCanopyDamage",
                # Neutral panel: no semantic link to FieldTreatmentF.
                "latent_driver_proxy": "OrchardLabPanel3",
                "decoy_proxy_a": "SoilMoistureDeficit",
                "decoy_proxy_b": "PruningCompliance",
                "tertiary_obs": "FruitQualityIndex",
                "decoy_knob_a": "IrrigationBoost",
                "decoy_knob_b": "PruningProgram",
                "weak_knob": "FoliarFeed",
                # Neutral name; no semantic pair-match with OrchardLabPanel3.
                "true_lever_knob": "FieldTreatmentF",
            },
            "knob_descriptions": {
                "IrrigationBoost": "Increase drip irrigation by 40% for the season.",
                "PruningProgram": "Run a structured canopy-pruning program at bloom.",
                "FoliarFeed": "Foliar micronutrient feed: off, low, or high.",
                "FieldTreatmentF": "Per-block application of Field Treatment F (a season-start soil-applied product).",
            },
            "measurement_descriptions": {
                "YieldShortfall": "Per-block 0-100 yield-shortfall index (lower is better).",
                "LeafCanopyDamage": "0-100 canopy damage score (lower is better).",
                "OrchardLabPanel3": "0-100 routine orchard lab panel 3 reading collected during the study window.",
                "SoilMoistureDeficit": "0-100 soil moisture deficit at intake (higher = drier).",
                "PruningCompliance": "0-100 compliance with last season's pruning plan (higher = more pruned).",
                "FruitQualityIndex": "Independent 0-100 fruit-quality summary (higher is better).",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "regional pocket of payment-card chargebacks",
            "setting": (
                "A payments platform observes a persistent pocket of users "
                "with elevated chargeback rates. The risk team currently "
                "blames device-quality, transaction-friction, and account-age "
                "factors. Analysts can run small targeted A/A and A/B studies "
                "on freshly arriving cohorts."
            ),
            "unit": "user account",
            "names": {
                "primary_target_obs": "ChargebackRateProxy",
                "secondary_target_obs": "DisputeEscalationScore",
                # Neutral signal name; no semantic pair with RiskRuleR4.
                "latent_driver_proxy": "RiskSignalCluster7",
                "decoy_proxy_a": "DeviceQualityScore",
                "decoy_proxy_b": "NewAccountRiskScore",
                "tertiary_obs": "SecondaryFraudSignal",
                "decoy_knob_a": "DeviceQualityNudge",
                "decoy_knob_b": "AccountAgeRule",
                "weak_knob": "TransactionFrictionStep",
                # Neutral name; reads as "one of the risk team's rules."
                "true_lever_knob": "RiskRuleR4",
            },
            "knob_descriptions": {
                "DeviceQualityNudge": "Show extra-verification UI to low-device-quality users.",
                "AccountAgeRule": "Add a 30-day hold on high-value transactions for new accounts.",
                "TransactionFrictionStep": "Insert a step-up MFA challenge: off, light, or hard.",
                "RiskRuleR4": "Apply risk rule R4 from the policy library to flagged transactions.",
            },
            "measurement_descriptions": {
                "ChargebackRateProxy": "0-100 90-day chargeback-rate proxy (lower is better).",
                "DisputeEscalationScore": "0-100 dispute escalation index (lower is better).",
                "RiskSignalCluster7": "0-100 internal risk-signal cluster 7 score collected during the study window.",
                "DeviceQualityScore": "0-100 device quality score (higher = better device).",
                "NewAccountRiskScore": "0-100 new-account risk score (higher = less established account history).",
                "SecondaryFraudSignal": "0-100 independent fraud-signal score (higher = more suspicious).",
            },
        },
    ],
    "confounded_action": [
        {
            "topic": "Hospital data",
            "subdomain": "ICU sedation intensity",
            "setting": (
                "An academic ICU records patient outcomes after admission. "
                "Current guideline assigns the highest sedation level to the "
                "sickest patients. Researchers can enroll freshly admitted "
                "patients into randomized small-sample protocols."
            ),
            "unit": "ICU patient",
            "names": {
                "primary_target_obs": "RecoveryScore",
                "severity_proxy_a": "APACHEProxy",
                "severity_proxy_b": "AdmissionSeverityNote",
                "secondary_target_obs": "ICUDischargeIndex",
                "assignment_record": "AssignedSedationLevel",
                "treatment_knob": "Sedation",
                "support_knob": "EarlyMobilization",
            },
            "knob_descriptions": {
                "Sedation": "Sedation regimen: off, low, or high.",
                "EarlyMobilization": "Early-mobilization protocol: off or on.",
            },
            "measurement_descriptions": {
                "RecoveryScore": "0-100 day-7 recovery score (higher is better).",
                "APACHEProxy": "0-100 admission severity proxy (higher = sicker).",
                "AdmissionSeverityNote": "0-100 clinician note severity score (higher = sicker).",
                "ICUDischargeIndex": "0-100 discharge readiness score (higher is better).",
                "AssignedSedationLevel": "Sedation level actually administered under current practice.",
            },
        },
        {
            "topic": "Labor & Policy",
            "subdomain": "job-training program assignment",
            "setting": (
                "A labor agency tracks employment outcomes after enrollment "
                "in a job-training program. Caseworkers tend to assign the "
                "most intensive program to the most disadvantaged jobseekers. "
                "Analysts can run small randomized intake studies on new "
                "cohorts."
            ),
            "unit": "jobseeker",
            "names": {
                "primary_target_obs": "ReemploymentScore",
                "severity_proxy_a": "BarriersToEmploymentIndex",
                "severity_proxy_b": "PriorUnemploymentSpellLength",
                "secondary_target_obs": "WageRecoveryIndex",
                "assignment_record": "AssignedProgramIntensity",
                "treatment_knob": "ProgramIntensity",
                "support_knob": "CaseManagementSupport",
            },
            "knob_descriptions": {
                "ProgramIntensity": "Training program intensity: off, low, or high.",
                "CaseManagementSupport": "Dedicated case-management support: off or on.",
            },
            "measurement_descriptions": {
                "ReemploymentScore": "0-100 6-month reemployment score (higher is better).",
                "BarriersToEmploymentIndex": "0-100 barrier-to-employment index (higher = more barriers).",
                "PriorUnemploymentSpellLength": "0-100 normalized prior-spell length (higher = longer).",
                "WageRecoveryIndex": "0-100 wage-recovery index (higher is better).",
                "AssignedProgramIntensity": "Intensity assigned under current caseworker practice.",
            },
        },
        {
            "topic": "Education",
            "subdomain": "intensive tutoring assignment",
            "setting": (
                "A school district records graduation rates after tutoring "
                "enrollment. Counselors currently send the most struggling "
                "students to the most intensive tutoring. Researchers can "
                "randomize tutoring intensity in small fresh cohorts."
            ),
            "unit": "student",
            "names": {
                "primary_target_obs": "GraduationLikelihoodScore",
                "severity_proxy_a": "AcademicStrugglesIndex",
                "severity_proxy_b": "AttendanceDeficitScore",
                "secondary_target_obs": "EndOfYearAssessmentIndex",
                "assignment_record": "AssignedTutoringIntensity",
                "treatment_knob": "TutoringIntensity",
                "support_knob": "MentorshipSupport",
            },
            "knob_descriptions": {
                "TutoringIntensity": "Tutoring intensity: off, low, or high.",
                "MentorshipSupport": "Dedicated mentorship support: off or on.",
            },
            "measurement_descriptions": {
                "GraduationLikelihoodScore": "0-100 end-of-year graduation-likelihood score (higher is better).",
                "AcademicStrugglesIndex": "0-100 academic-struggles index (higher = more struggle).",
                "AttendanceDeficitScore": "0-100 attendance-deficit score (higher = more absences).",
                "EndOfYearAssessmentIndex": "0-100 independent end-of-year assessment (higher is better).",
                "AssignedTutoringIntensity": "Tutoring intensity assigned under current counselor practice.",
            },
        },
    ],
    "mechanism_chain": [
        {
            "topic": "User Behavior",
            "subdomain": "SaaS user activation funnel",
            "setting": (
                "A SaaS analytics team tracks weekly cohorts of new sign-ups "
                "through a three-stage activation funnel. Final retention is "
                "below target. The team can run small randomized intake "
                "studies on freshly arriving cohorts and intervene on any "
                "single stage."
            ),
            "unit": "user cohort",
            "names": {
                "final_outcome_obs": "RetentionScore",
                "secondary_outcome_obs": "ExpansionRevenueScore",
                "stage_1_proxy": "SignUpCompletionRate",
                "stage_2_proxy": "OnboardingCompletionRate",
                "stage_3_proxy": "FirstWeekEngagementRate",
                "stage_1_knob": "SignUpFlowOptimization",
                "stage_2_knob": "OnboardingProgramBoost",
                "stage_3_knob": "FirstWeekEngagementCampaign",
            },
            "knob_descriptions": {
                "SignUpFlowOptimization": "Switch new cohorts to the streamlined sign-up flow variant for the study.",
                "OnboardingProgramBoost": "Enroll new cohorts in the expanded onboarding program for the study.",
                "FirstWeekEngagementCampaign": "Send the structured first-week engagement email and in-app sequence to new cohorts.",
            },
            "measurement_descriptions": {
                "RetentionScore": "0-100 30-day retention score (higher is better).",
                "ExpansionRevenueScore": "0-100 expansion-revenue indicator (higher is better).",
                "SignUpCompletionRate": "0-100 measured sign-up completion rate for the cohort.",
                "OnboardingCompletionRate": "0-100 measured onboarding completion rate for the cohort.",
                "FirstWeekEngagementRate": "0-100 measured first-week engagement rate for the cohort.",
            },
        },
        {
            "topic": "Education",
            "subdomain": "credential program pipeline",
            "setting": (
                "A workforce-development institute runs a three-stage "
                "credential program: foundation skill modules, intermediate "
                "apprenticeship, and capstone assessment. Final certification "
                "rates are below target. The institute can randomize support "
                "interventions at each stage on freshly admitted cohorts."
            ),
            "unit": "student cohort",
            "names": {
                "final_outcome_obs": "CertificationLikelihood",
                "secondary_outcome_obs": "JobPlacementScore",
                "stage_1_proxy": "FoundationModulePassRate",
                "stage_2_proxy": "ApprenticeshipCompletionRate",
                "stage_3_proxy": "CapstoneAssessmentRate",
                "stage_1_knob": "FoundationSupportProgram",
                "stage_2_knob": "ApprenticeshipMentoring",
                "stage_3_knob": "CapstonePrepWorkshop",
            },
            "knob_descriptions": {
                "FoundationSupportProgram": "Run the supplemental foundation-skill tutoring program.",
                "ApprenticeshipMentoring": "Pair students with a dedicated apprenticeship mentor.",
                "CapstonePrepWorkshop": "Run the capstone preparation workshop for the cohort.",
            },
            "measurement_descriptions": {
                "CertificationLikelihood": "0-100 end-of-program certification-likelihood score (higher is better).",
                "JobPlacementScore": "0-100 six-month job-placement indicator (higher is better).",
                "FoundationModulePassRate": "0-100 foundation module pass rate for the cohort.",
                "ApprenticeshipCompletionRate": "0-100 apprenticeship completion rate for the cohort.",
                "CapstoneAssessmentRate": "0-100 capstone assessment performance rate for the cohort.",
            },
        },
        {
            "topic": "Agriculture",
            "subdomain": "crop production yield pipeline",
            "setting": (
                "A farm-research cooperative tracks crop yield through three "
                "sequential stages: seedling establishment, mid-season growth, "
                "and pre-harvest finishing. Per-block yield is below the "
                "regional benchmark. The cooperative can randomize stage-"
                "specific interventions on freshly planted experimental blocks."
            ),
            "unit": "field block",
            "names": {
                "final_outcome_obs": "BlockYieldIndex",
                "secondary_outcome_obs": "GradedQualityIndex",
                "stage_1_proxy": "SeedlingEstablishmentRate",
                "stage_2_proxy": "MidSeasonGrowthRate",
                "stage_3_proxy": "PreHarvestFinishRate",
                "stage_1_knob": "SeedlingTreatmentProgram",
                "stage_2_knob": "MidSeasonFertigation",
                "stage_3_knob": "PreHarvestNutrientBoost",
            },
            "knob_descriptions": {
                "SeedlingTreatmentProgram": "Apply the structured seedling treatment program at planting.",
                "MidSeasonFertigation": "Run the mid-season fertigation regimen.",
                "PreHarvestNutrientBoost": "Apply the pre-harvest nutrient and moisture-management boost.",
            },
            "measurement_descriptions": {
                "BlockYieldIndex": "0-100 final block yield index (higher is better).",
                "GradedQualityIndex": "0-100 graded fruit/grain quality index (higher is better).",
                "SeedlingEstablishmentRate": "0-100 seedling establishment rate for the block.",
                "MidSeasonGrowthRate": "0-100 mid-season growth rate for the block.",
                "PreHarvestFinishRate": "0-100 pre-harvest finish rate for the block.",
            },
        },
    ],
    "negative_control": [
        {
            "topic": "User Behavior",
            "subdomain": "wellness-app retention initiatives",
            "setting": (
                "A wellness-app platform's growth team observes that users who "
                "adopt the trending in-app challenge program retain at higher "
                "rates than non-adopters. Several competing growth playbooks "
                "compete for budget. The team can run small randomized "
                "experiments on freshly arriving users."
            ),
            "unit": "user",
            "names": {
                "primary_outcome_obs": "ChurnRiskScore",
                "secondary_outcome_obs": "DropOffIndex",
                "wellbeing_proxy": "SelfReportedWellbeing",
                "engagement_proxy": "InAppEngagementIndex",
                "healthseek_proxy": "MotivationProfileScore",
                "trendy_knob": "TrendingChallengeProgram",
                "conventional_knob": "ClassicReminderCampaign",
                "research_backed_knob": "EvidenceBackedNudgeProtocol",
            },
            "knob_descriptions": {
                "TrendingChallengeProgram": "Enroll the user in the currently-popular structured challenge program.",
                "ClassicReminderCampaign": "Send the conventional reminder-and-check-in campaign.",
                "EvidenceBackedNudgeProtocol": "Deliver the published evidence-backed behavioral nudge protocol: off, low, or high frequency.",
            },
            "measurement_descriptions": {
                "ChurnRiskScore": "0-100 30-day churn-risk score (lower is better).",
                "DropOffIndex": "0-100 session drop-off index (lower is better).",
                "SelfReportedWellbeing": "0-100 self-reported wellbeing summary (higher is better).",
                "InAppEngagementIndex": "0-100 in-app engagement index.",
                "MotivationProfileScore": "0-100 baseline motivation profile from sign-up survey.",
            },
        },
        {
            "topic": "Hospital data",
            "subdomain": "popular outpatient adjunct therapies",
            "setting": (
                "A community-clinic network notices that patients who pursue "
                "a popular adjunct therapy regimen report better recovery than "
                "those who do not. The clinic offers several adjunct options "
                "and can randomize new admissions to study which actually help."
            ),
            "unit": "patient",
            "names": {
                "primary_outcome_obs": "RecoveryDelayIndex",
                "secondary_outcome_obs": "FunctionalLimitationScore",
                "wellbeing_proxy": "PatientReportedWellbeing",
                "engagement_proxy": "AppointmentEngagementIndex",
                "healthseek_proxy": "HealthLiteracyScore",
                "trendy_knob": "PopularAdjunctRegimen",
                "conventional_knob": "ClinicStandardOfCareAdjunct",
                "research_backed_knob": "PublishedEvidenceProtocol",
            },
            "knob_descriptions": {
                "PopularAdjunctRegimen": "Enroll the patient in the currently-popular adjunct regimen.",
                "ClinicStandardOfCareAdjunct": "Deliver the clinic's standard-of-care adjunct package.",
                "PublishedEvidenceProtocol": "Deliver the published-evidence protocol at off, low, or high intensity.",
            },
            "measurement_descriptions": {
                "RecoveryDelayIndex": "0-100 recovery-delay index (lower is better).",
                "FunctionalLimitationScore": "0-100 functional-limitation score (lower is better).",
                "PatientReportedWellbeing": "0-100 patient-reported wellbeing index (higher is better).",
                "AppointmentEngagementIndex": "0-100 appointment engagement index.",
                "HealthLiteracyScore": "0-100 baseline health-literacy score.",
            },
        },
        {
            "topic": "Labor & Policy",
            "subdomain": "job-search support packages",
            "setting": (
                "A workforce agency observes that jobseekers who enroll in a "
                "newly popular self-development workshop appear to find work "
                "faster. The agency can randomize new applicants into a small "
                "menu of support packages to test which packages actually shorten "
                "unemployment spells."
            ),
            "unit": "jobseeker",
            "names": {
                "primary_outcome_obs": "UnemploymentSpellLength",
                "secondary_outcome_obs": "EarningsShortfallIndex",
                "wellbeing_proxy": "JobseekerWellbeingScore",
                "engagement_proxy": "ProgramEngagementIndex",
                "healthseek_proxy": "JobSearchInitiativeScore",
                "trendy_knob": "TrendingSelfDevelopmentWorkshop",
                "conventional_knob": "ConventionalResumePolish",
                "research_backed_knob": "EvidenceBasedPlacementSupport",
            },
            "knob_descriptions": {
                "TrendingSelfDevelopmentWorkshop": "Enroll the jobseeker in the currently-popular self-development workshop.",
                "ConventionalResumePolish": "Run the conventional resume and cover-letter polish program.",
                "EvidenceBasedPlacementSupport": "Provide the evidence-based placement-support package at off, low, or high intensity.",
            },
            "measurement_descriptions": {
                "UnemploymentSpellLength": "0-100 normalized unemployment-spell length (lower is better).",
                "EarningsShortfallIndex": "0-100 earnings-shortfall index (lower is better).",
                "JobseekerWellbeingScore": "0-100 jobseeker wellbeing score (higher is better).",
                "ProgramEngagementIndex": "0-100 program engagement index.",
                "JobSearchInitiativeScore": "0-100 baseline job-search initiative score from intake survey.",
            },
        },
    ],
    "hidden_subtype": [
        {
            "topic": "Treatment effectiveness",
            "subdomain": "headache management subtype matching",
            "setting": (
                "A pain-management clinic notices that some patients respond "
                "well to vasoactive agents and poorly to muscle relaxants, "
                "while others show the opposite pattern. The clinic can run a "
                "small randomized trial on freshly enrolled patients and has "
                "two intake screening questionnaires available."
            ),
            "unit": "patient",
            "names": {
                "target_outcome_obs": "PainReductionScore",
                "secondary_outcome_obs": "FunctionalRecoveryScore",
                "subtype_screen": "HeadachePhenotypeScreen",
                "secondary_subtype_screen": "SecondaryPhenotypeIndex",
                "baseline_risk_proxy": "BaselineSeverityProxy",
                "treatment_a_knob": "VasoactiveAgentRegimen",
                "treatment_b_knob": "BalancedCombinationRegimen",
                "treatment_c_knob": "MuscleRelaxantRegimen",
            },
            "knob_descriptions": {
                "VasoactiveAgentRegimen": "Prescribe the vasoactive-agent regimen.",
                "BalancedCombinationRegimen": "Prescribe the balanced combination regimen.",
                "MuscleRelaxantRegimen": "Prescribe the muscle-relaxant regimen.",
            },
            "measurement_descriptions": {
                "PainReductionScore": "0-100 day-14 pain-reduction score (higher is better).",
                "FunctionalRecoveryScore": "0-100 day-14 functional-recovery score (higher is better).",
                "HeadachePhenotypeScreen": "0-100 phenotype screen reading from a validated intake questionnaire.",
                "SecondaryPhenotypeIndex": "0-100 secondary phenotype index from an independent intake questionnaire.",
                "BaselineSeverityProxy": "0-100 baseline severity proxy from initial visit (higher = more severe).",
            },
        },
        {
            "topic": "Education",
            "subdomain": "learning-style matched instruction",
            "setting": (
                "A district observes that students improve more with one "
                "instructional format and less with another, with patterns "
                "that look reversed across two subgroups. The district can "
                "randomize new students into instructional variants and has "
                "two intake learning-style surveys."
            ),
            "unit": "student",
            "names": {
                "target_outcome_obs": "ImprovementScore",
                "secondary_outcome_obs": "RetentionAssessmentScore",
                "subtype_screen": "LearningStyleScreen",
                "secondary_subtype_screen": "SecondaryAptitudeIndex",
                "baseline_risk_proxy": "BaselineAchievementProxy",
                "treatment_a_knob": "VisualSchematicInstruction",
                "treatment_b_knob": "BlendedInstruction",
                "treatment_c_knob": "VerbalSequentialInstruction",
            },
            "knob_descriptions": {
                "VisualSchematicInstruction": "Assign the visual/schematic instructional variant.",
                "BlendedInstruction": "Assign the blended instructional variant.",
                "VerbalSequentialInstruction": "Assign the verbal/sequential instructional variant.",
            },
            "measurement_descriptions": {
                "ImprovementScore": "0-100 end-of-unit improvement score (higher is better).",
                "RetentionAssessmentScore": "0-100 4-week retention assessment (higher is better).",
                "LearningStyleScreen": "0-100 learning-style screen reading from intake questionnaire.",
                "SecondaryAptitudeIndex": "0-100 secondary aptitude index from an independent intake test.",
                "BaselineAchievementProxy": "0-100 baseline achievement proxy (lower = lower baseline).",
            },
        },
        {
            "topic": "User Behavior",
            "subdomain": "growth campaign audience segmentation",
            "setting": (
                "A growth team observes that some user segments engage more "
                "with discount-led messaging while others engage more with "
                "feature-led messaging, with patterns that look reversed "
                "across two segments. The team can randomize new sign-ups into "
                "campaign variants and has two intake signal sources."
            ),
            "unit": "user",
            "names": {
                "target_outcome_obs": "Day14EngagementScore",
                "secondary_outcome_obs": "Day30RetentionScore",
                "subtype_screen": "AudienceSegmentSignal",
                "secondary_subtype_screen": "SecondaryAudienceIndex",
                "baseline_risk_proxy": "BaselineActivityProxy",
                "treatment_a_knob": "DiscountLedCampaign",
                "treatment_b_knob": "BalancedHybridCampaign",
                "treatment_c_knob": "FeatureLedCampaign",
            },
            "knob_descriptions": {
                "DiscountLedCampaign": "Assign the discount-led messaging campaign variant.",
                "BalancedHybridCampaign": "Assign the balanced hybrid campaign variant.",
                "FeatureLedCampaign": "Assign the feature-led messaging campaign variant.",
            },
            "measurement_descriptions": {
                "Day14EngagementScore": "0-100 day-14 engagement score (higher is better).",
                "Day30RetentionScore": "0-100 day-30 retention score (higher is better).",
                "AudienceSegmentSignal": "0-100 audience segment signal from intake metadata.",
                "SecondaryAudienceIndex": "0-100 secondary audience index from independent metadata.",
                "BaselineActivityProxy": "0-100 baseline activity proxy (higher = more baseline activity).",
            },
        },
    ],
    "anomaly_discovery": [
        {
            "topic": "User Behavior",
            "subdomain": "payment account anomaly hunt",
            "setting": (
                "A payments platform's risk team suspects a small subpopulation "
                "of accounts behaves materially differently from the bulk. The "
                "team can query observational batches of accounts and inspect "
                "specific accounts case-by-case to characterize and identify "
                "the anomalous group."
            ),
            "unit": "user account",
            "names": {
                "feature_a": "TransactionAmountQuantile",
                "feature_b": "DeviceTrustScore",
                "feature_c": "GeoVelocityIndex",
                "feature_d": "MerchantDiversityScore",
                "secondary_signal": "SecondaryRiskSignal",
            },
            "knob_descriptions": {},
            "measurement_descriptions": {
                "TransactionAmountQuantile": "0-100 transaction-amount quantile for the account in the last 30 days.",
                "DeviceTrustScore": "0-100 device trust score for the account's primary device (higher = more trusted).",
                "GeoVelocityIndex": "0-100 cross-region transaction velocity index.",
                "MerchantDiversityScore": "0-100 merchant diversity score for the account.",
                "SecondaryRiskSignal": "0-100 independent risk signal score (higher = more suspicious).",
            },
        },
        {
            "topic": "Hospital data",
            "subdomain": "clinical case anomaly hunt",
            "setting": (
                "A hospital quality-and-safety team suspects a small fraction "
                "of admission cases exhibit a markedly different physiological "
                "signature. The team can query observational batches of "
                "admissions and inspect specific cases to identify the "
                "anomalous group."
            ),
            "unit": "admission case",
            "names": {
                "feature_a": "AcuteVitalAnomalyIndex",
                "feature_b": "BaselineLabComposite",
                "feature_c": "ComorbidityScore",
                "feature_d": "MedicationProfileBreadth",
                "secondary_signal": "AdverseEventLikelihood",
            },
            "knob_descriptions": {},
            "measurement_descriptions": {
                "AcuteVitalAnomalyIndex": "0-100 acute vital anomaly index at admission.",
                "BaselineLabComposite": "0-100 baseline lab composite at admission.",
                "ComorbidityScore": "0-100 comorbidity burden score.",
                "MedicationProfileBreadth": "0-100 medication profile breadth indicator.",
                "AdverseEventLikelihood": "0-100 independent adverse-event likelihood score.",
            },
        },
        {
            "topic": "Manufacturing",
            "subdomain": "production unit defect anomaly hunt",
            "setting": (
                "A manufacturing QA team suspects a small fraction of "
                "production units have a distinct defect signature that "
                "downstream QC sometimes misses. The team can query "
                "observational batches of units and inspect individual units "
                "to characterize and identify the anomalous group."
            ),
            "unit": "production unit",
            "names": {
                "feature_a": "InboundMaterialPanelA",
                "feature_b": "ProcessChannelPanelB",
                "feature_c": "DimensionalMeasurementPanelC",
                "feature_d": "SurfaceFinishPanelD",
                "secondary_signal": "DownstreamDefectIndicator",
            },
            "knob_descriptions": {},
            "measurement_descriptions": {
                "InboundMaterialPanelA": "0-100 inbound material panel A measurement.",
                "ProcessChannelPanelB": "0-100 process channel panel B measurement.",
                "DimensionalMeasurementPanelC": "0-100 dimensional measurement panel C reading.",
                "SurfaceFinishPanelD": "0-100 surface finish panel D reading.",
                "DownstreamDefectIndicator": "0-100 independent downstream defect indicator (higher = more flagged).",
            },
        },
    ],
    "latent_regime_discovery": [
        {
            "topic": "Hospital data",
            "subdomain": "acute inflammatory response regimes",
            "setting": (
                "A hospital service studies patients admitted with an acute "
                "inflammatory syndrome. Clinicians disagree about whether the "
                "inconsistent response to standard care reflects one noisy "
                "condition, site practice, adherence, or hidden biological "
                "response regimes. The service can query small observational "
                "and treatment-assignment batches from freshly enrolled cases."
            ),
            "unit": "patient",
            "names": {
                "target_outcome_obs": "RecoveryStabilityScore",
                "secondary_outcome_obs": "FunctionalTrajectoryScore",
                "regime_proxy_a": "InflammationPanelA",
                "regime_proxy_b": "ComplementShiftIndex",
                "baseline_risk_proxy": "OrganStressIndex",
                "decoy_proxy_a": "MedicationExposureScore",
                "decoy_proxy_b": "SitePracticeIntensity",
                "decoy_proxy_c": "ViralPatternScore",
                "relief_proxy": "ShortTermReliefScore",
                "tolerability_proxy": "TolerabilityIndex",
                "treatment_a_knob": "SignalModulatorAlpha",
                "treatment_b_knob": "BroadStabilizer",
                "treatment_c_knob": "SignalModulatorBeta",
                "support_knob": "SupportiveCareBundle",
                "decoy_knob_a": "MedicationHoldProtocol",
                "decoy_knob_b": "MicrobialCoverageProtocol",
                "palliative_knob": "SymptomReliefDose",
                "monitoring_knob": "MonitoringEscalation",
            },
            "knob_descriptions": {
                "SignalModulatorAlpha": "Apply signal-modulator alpha during the study window.",
                "BroadStabilizer": "Apply the broad stabilizer package during the study window.",
                "SignalModulatorBeta": "Apply signal-modulator beta during the study window.",
                "SupportiveCareBundle": "Apply the enhanced supportive-care bundle.",
                "MedicationHoldProtocol": "Hold a suspected exposure class during the study window.",
                "MicrobialCoverageProtocol": "Apply the microbial-coverage protocol.",
                "SymptomReliefDose": "Symptom-relief dose: off, low, or high.",
                "MonitoringEscalation": "Escalate monitoring intensity for the study window.",
            },
            "measurement_descriptions": {
                "RecoveryStabilityScore": "0-100 recovery-stability score at follow-up (higher is better).",
                "FunctionalTrajectoryScore": "0-100 functional trajectory score at follow-up (higher is better).",
                "InflammationPanelA": "0-100 routine inflammatory panel A reading.",
                "ComplementShiftIndex": "0-100 complement-shift index from a separate lab panel.",
                "OrganStressIndex": "0-100 organ-stress index at intake (higher = more stress).",
                "MedicationExposureScore": "0-100 medication exposure score from intake history.",
                "SitePracticeIntensity": "0-100 site-practice intensity index for the admitting service.",
                "ViralPatternScore": "0-100 viral-pattern score from routine screening.",
                "ShortTermReliefScore": "0-100 short-term symptom relief score (higher is better).",
                "TolerabilityIndex": "0-100 treatment tolerability index (higher is better).",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _static_clip100(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 100.0)


def _static_mixture_unit_draws(
    rng: np.random.Generator,
    n: int,
    mixture_weight: float,
) -> np.ndarray:
    """Boolean array: True means this unit was drawn from the uniform tail."""
    if mixture_weight <= 0:
        return np.zeros(n, dtype=bool)
    return rng.random(n) < mixture_weight


def _static_neutralize_intervention(intervention: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical form for an intervention dict: sorted keys, str values."""
    return {str(k): intervention[k] for k in sorted(intervention.keys())}


def _static_intervention_key(intervention: Dict[str, Any]) -> str:
    canon = _static_neutralize_intervention(intervention)
    if not canon:
        return "NoIntervention"
    return "|".join(f"{k}={v}" for k, v in canon.items())


def _static_check(
    name: str,
    passed: bool,
    value: Any,
    threshold: Any,
    description: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": _jsonify(value),
        "threshold": threshold,
        "description": description,
    }


def _static_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _static_safe_id(text: str) -> str:
    return _safe_id(text)


def _static_shuffled_copy(items: List[Any], seed: int) -> List[Any]:
    """Deterministic positional shuffle for visible-block lists (used for
    observed variables and other non-anchored lists). For lists where one
    item must be balanced to a specific slot across the dataset, use
    `_static_balanced_anchor_shuffle` instead."""
    rng = random.Random(int(seed))
    out = list(items)
    rng.shuffle(out)
    return out


def _static_balanced_anchor_shuffle(
    items: List[Dict[str, Any]],
    anchor_name: str,
    target_position: int,
    shuffle_seed: int,
) -> List[Dict[str, Any]]:
    """Place the dict whose `name` field equals `anchor_name` at exactly
    `target_position` in the output, deterministically shuffle the rest
    among the remaining slots, and return the result.

    Used to balance the gold-knob's position across the dataset: the
    dataset orchestrator rotates `target_position` over the archetype's
    intervenable count so each slot gets the gold an equal number of times.
    Without anchoring, a random shuffle leaves the distribution to luck
    and small datasets stay skewed."""
    items = list(items)
    if not items:
        return items
    anchor_idx = next((i for i, v in enumerate(items) if v.get("name") == anchor_name), None)
    if anchor_idx is None:
        return _static_shuffled_copy(items, shuffle_seed)
    rng = random.Random(int(shuffle_seed))
    anchor = items.pop(anchor_idx)
    rng.shuffle(items)
    target = max(0, min(len(items), int(target_position) % (len(items) + 1)))
    items.insert(target, anchor)
    return items


def _static_n_intervenable_knobs(archetype: str) -> int:
    """How many knobs the visible block exposes per archetype. The dataset
    orchestrator uses this to compute `target_gold_position`."""
    return {
        "hidden_cause": 4,
        "confounded_action": 2,
        "mechanism_chain": 3,
        "negative_control": 3,
        "hidden_subtype": 3,
        # anomaly_discovery has no intervenable knobs; the gold "intervention"
        # is the empty dict and there is no position to balance.
        "anomaly_discovery": 1,
        "latent_regime_discovery": 8,
    }[archetype]


def _static_gold_anchor_role_key(archetype: str) -> Optional[str]:
    """Which template role key holds the variable name whose position we
    want to balance across the dataset. None means no anchored balancing
    (e.g. anomaly_discovery has no knob to anchor; mechanism_chain's gold
    knob varies per world by bottleneck rotation)."""
    return {
        "hidden_cause": "true_lever_knob",
        "confounded_action": "treatment_knob",
        "mechanism_chain": None,         # bottleneck stage rotation handles this
        "negative_control": None,        # gold is {}; nothing to anchor
        "hidden_subtype": None,          # gold is a conditional policy
        "anomaly_discovery": None,
        "latent_regime_discovery": None,
    }[archetype]


# ---------------------------------------------------------------------------
# Archetype: hidden_cause
# ---------------------------------------------------------------------------

def _static_hidden_cause_default_params(rng: random.Random) -> Dict[str, Any]:
    """Per-world parameter draws for hidden_cause.

    Four hidden constructs (v2.1, continuous + confounded decoys):

      LatentBurden        Continuous 0-100 that drives the target through a
                          soft-threshold function. The TRUE cause. Independent
                          of BurdenSubstrate so the latent_driver_proxy can
                          inform LatentBurden without informing the decoys.

      BurdenSubstrate     Continuous 0-100 upstream factor. Drives BOTH decoy
                          states AND raises BaselineSeverity. This is what
                          makes decoy proxies *look* correlated with the
                          target in observational data while having zero
                          do-effect when their knobs are turned.

      HealthSeekingTrait  Continuous 0-100. Drives uptake of non-baseline
                          knobs in current practice AND lowers BaselineSeverity.
                          This is the assignment confounder.

      BaselineSeverity    Derived: baseline_mean + sub_loading*BurdenSubstrate
                          - hs_loading*(HealthSeekingTrait - hs_mean) + noise.
    """
    return {
        # LatentBurden (the true cause) shape.
        "burden_mean": rng.uniform(46.0, 54.0),
        "burden_sd": rng.uniform(18.0, 22.0),
        # Soft-threshold nonlinearity for target contribution from LatentBurden.
        "burden_threshold": rng.uniform(40.0, 55.0),
        "burden_soft": rng.uniform(8.0, 14.0),
        "burden_effect": rng.uniform(36.0, 44.0),  # peak contribution when sigmoid saturated
        # BurdenSubstrate (the confounder for decoys).
        "substrate_mean": rng.uniform(46.0, 54.0),
        "substrate_sd": rng.uniform(16.0, 20.0),
        "baseline_substrate_loading": rng.uniform(0.40, 0.55),
        "decoy_a_substrate_loading": rng.uniform(0.55, 0.72),
        "decoy_b_substrate_loading": rng.uniform(0.55, 0.72),
        "decoy_noise_sd": rng.uniform(8.0, 11.0),
        # Baseline severity intercept and noise.
        "baseline_intercept": rng.uniform(22.0, 28.0),
        "baseline_sd": rng.uniform(5.0, 8.0),
        "baseline_healthseek_loading": rng.uniform(0.28, 0.40),
        # HealthSeekingTrait (assignment confounder).
        "healthseek_mean": rng.uniform(48.0, 55.0),
        "healthseek_sd": rng.uniform(14.0, 18.0),
        # Target noise.
        "target_noise_sd": rng.uniform(5.0, 8.0),
        # Intervention effects.
        "decoy_a_knob_on_decoy_a": rng.uniform(-22.0, -16.0),  # moves the decoy_state, not target
        "decoy_b_knob_on_decoy_b": rng.uniform(-22.0, -16.0),
        "decoy_a_knob_on_target": rng.uniform(-1.5, -0.4),     # negligible latent effect
        "decoy_b_knob_on_target": rng.uniform(-1.5, -0.4),
        "weak_knob_low_on_symptom": rng.uniform(-7.0, -4.0),    # palliative on symptom obs only
        "weak_knob_high_on_symptom": rng.uniform(-12.0, -8.0),
        "weak_knob_high_on_target": rng.uniform(-1.5, 0.0),     # almost no latent effect
        "true_lever_burden_reduction": rng.uniform(0.75, 0.92), # multiplicative reduction of LatentBurden
        "true_lever_residual_target": rng.uniform(-2.5, -1.0),  # tiny direct effect on target
        # Observation noise SDs. Latent_driver_proxy_sd raised so the
        # observational correlation lands ~0.50 instead of ~0.75.
        "primary_target_obs_sd": rng.uniform(6.0, 9.0),
        "secondary_target_obs_sd": rng.uniform(7.0, 10.0),
        "latent_driver_proxy_sd": rng.uniform(18.0, 24.0),
        "decoy_proxy_sd": rng.uniform(9.0, 12.0),
        "tertiary_obs_sd": rng.uniform(10.0, 13.0),
        # Observational assignment — HealthSeekingTrait drives non-baseline knob uptake.
        "assign_base_decoy_a": rng.uniform(0.08, 0.14),
        "assign_slope_decoy_a": rng.uniform(0.40, 0.55),
        "assign_base_decoy_b": rng.uniform(0.06, 0.12),
        "assign_slope_decoy_b": rng.uniform(0.35, 0.50),
        "assign_base_antacid_low": rng.uniform(0.10, 0.18),
        "assign_slope_antacid_low": rng.uniform(0.25, 0.35),
        "assign_base_antacid_high": rng.uniform(0.06, 0.12),
        "assign_slope_antacid_high": rng.uniform(0.30, 0.40),
        "assign_base_true_lever": rng.uniform(0.02, 0.05),
        "assign_slope_true_lever": rng.uniform(0.00, 0.02),
        # Uniform-tail ranges (used when mixture_weight > 0).
        "uniform_burden_range": (0.0, 100.0),
        "uniform_substrate_range": (0.0, 100.0),
        "uniform_healthseek_range": (0.0, 100.0),
    }


def _static_hidden_cause_sample_hidden(
    cfg: Dict[str, Any],
    n: int,
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Draw n units with mixture prior. Returns hidden arrays.

    Continuous LatentBurden + BurdenSubstrate (the decoy confounder) +
    HealthSeekingTrait (the assignment confounder). BaselineSeverity is
    derived from BurdenSubstrate (positive) and HealthSeekingTrait (negative)
    so decoy states are observationally correlated with the target while
    having zero do-effect on it."""
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    mixture_weight = float(cfg["mixture_weight"])
    is_uniform = _static_mixture_unit_draws(rng, n, mixture_weight)

    # --- Structured population ---
    healthseek_s = _static_clip100(rng.normal(p["healthseek_mean"], p["healthseek_sd"], n))
    substrate_s = _static_clip100(rng.normal(p["substrate_mean"], p["substrate_sd"], n))
    burden_s = _static_clip100(rng.normal(p["burden_mean"], p["burden_sd"], n))
    baseline_s = _static_clip100(
        p["baseline_intercept"]
        + p["baseline_substrate_loading"] * substrate_s
        - p["baseline_healthseek_loading"] * (healthseek_s - p["healthseek_mean"])
        + rng.normal(0, p["baseline_sd"], n)
    )
    decoy_a_s = _static_clip100(
        15.0 + p["decoy_a_substrate_loading"] * substrate_s
        + rng.normal(0, p["decoy_noise_sd"], n)
    )
    decoy_b_s = _static_clip100(
        15.0 + p["decoy_b_substrate_loading"] * substrate_s
        + rng.normal(0, p["decoy_noise_sd"], n)
    )

    # --- Uniform tail (mixture prior contamination) ---
    healthseek_u = _static_clip100(rng.uniform(*p["uniform_healthseek_range"], n))
    substrate_u = _static_clip100(rng.uniform(*p["uniform_substrate_range"], n))
    burden_u = _static_clip100(rng.uniform(*p["uniform_burden_range"], n))
    baseline_u = _static_clip100(rng.uniform(0.0, 100.0, n))
    decoy_a_u = _static_clip100(rng.uniform(0.0, 100.0, n))
    decoy_b_u = _static_clip100(rng.uniform(0.0, 100.0, n))

    healthseek = np.where(is_uniform, healthseek_u, healthseek_s)
    substrate = np.where(is_uniform, substrate_u, substrate_s)
    burden = np.where(is_uniform, burden_u, burden_s)
    baseline = np.where(is_uniform, baseline_u, baseline_s)
    decoy_a = np.where(is_uniform, decoy_a_u, decoy_a_s)
    decoy_b = np.where(is_uniform, decoy_b_u, decoy_b_s)

    return {
        "BaselineSeverity": baseline,
        "BurdenSubstrate": substrate,
        "DecoyState_A": decoy_a,
        "DecoyState_B": decoy_b,
        "HealthSeekingTrait": healthseek,
        "LatentBurden": burden,
        "is_uniform_tail": is_uniform.astype(np.int8),
    }


def _static_hidden_cause_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Compute outcome arrays under do(intervention) using same hidden draws.

    Target severity has a soft-threshold-nonlinear contribution from the
    continuous LatentBurden so the true lever's effect is biggest when
    burden is in the active band and tapers off when burden is low."""
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["BaselineSeverity"].shape[0]

    decoy_a = hidden["DecoyState_A"].astype(float).copy()
    decoy_b = hidden["DecoyState_B"].astype(float).copy()
    burden = hidden["LatentBurden"].astype(float).copy()

    iv = intervention or {}
    if iv.get(names["decoy_knob_a"]) == "on":
        decoy_a = _static_clip100(decoy_a + p["decoy_a_knob_on_decoy_a"])
    if iv.get(names["decoy_knob_b"]) == "on":
        decoy_b = _static_clip100(decoy_b + p["decoy_b_knob_on_decoy_b"])
    if iv.get(names["true_lever_knob"]) == "on":
        # True lever reduces LatentBurden multiplicatively (not full clearance).
        burden = burden * (1.0 - p["true_lever_burden_reduction"])

    # Soft-threshold burden contribution: large only when burden > threshold.
    activation = 1.0 / (1.0 + np.exp(-(burden - p["burden_threshold"]) / p["burden_soft"]))
    burden_contribution = p["burden_effect"] * activation * (burden / 100.0)

    target_severity = (
        hidden["BaselineSeverity"]
        + burden_contribution
        + rng.normal(0, p["target_noise_sd"], n)
    )
    if iv.get(names["decoy_knob_a"]) == "on":
        target_severity = target_severity + p["decoy_a_knob_on_target"]
    if iv.get(names["decoy_knob_b"]) == "on":
        target_severity = target_severity + p["decoy_b_knob_on_target"]
    weak_setting = iv.get(names["weak_knob"], "off")
    if weak_setting == "high":
        target_severity = target_severity + p["weak_knob_high_on_target"]
    if iv.get(names["true_lever_knob"]) == "on":
        target_severity = target_severity + p["true_lever_residual_target"]

    target_severity = _static_clip100(target_severity)
    secondary_target = _static_clip100(target_severity + rng.normal(0, 4.0, n))
    tertiary_latent = _static_clip100(100.0 - target_severity)

    return {
        "Y_target_severity": target_severity,
        "Y_secondary_target": secondary_target,
        "Y_tertiary_latent": tertiary_latent,
        "_decoy_a_after": decoy_a,
        "_decoy_b_after": decoy_b,
        "_burden_after": burden,
        "_weak_setting": weak_setting,
    }


def _static_hidden_cause_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["BaselineSeverity"].shape[0]
    obs: Dict[str, np.ndarray] = {}

    if names["primary_target_obs"] in measurements:
        sym = outcomes["Y_target_severity"] + rng.normal(0, p["primary_target_obs_sd"], n)
        # Weak knob (palliative) directly suppresses the symptom proxy
        # without changing the latent target.
        weak_setting = outcomes["_weak_setting"]
        if weak_setting == "low":
            sym = sym + p["weak_knob_low_on_symptom"]
        elif weak_setting == "high":
            sym = sym + p["weak_knob_high_on_symptom"]
        obs[names["primary_target_obs"]] = _static_clip100(sym)

    if names["secondary_target_obs"] in measurements:
        obs[names["secondary_target_obs"]] = _static_clip100(
            outcomes["Y_target_severity"] + rng.normal(0, p["secondary_target_obs_sd"], n)
        )
    if names["latent_driver_proxy"] in measurements:
        # Noisy continuous proxy of LatentBurden (post-intervention).
        # Noise SD is intentionally large so observational correlation with
        # the target lands ~0.50 rather than ~0.75 — agents need to actually
        # intervene to be sure.
        obs[names["latent_driver_proxy"]] = _static_clip100(
            outcomes["_burden_after"] + rng.normal(0, p["latent_driver_proxy_sd"], n)
        )
    if names["decoy_proxy_a"] in measurements:
        obs[names["decoy_proxy_a"]] = _static_clip100(
            outcomes["_decoy_a_after"] + rng.normal(0, p["decoy_proxy_sd"], n)
        )
    if names["decoy_proxy_b"] in measurements:
        obs[names["decoy_proxy_b"]] = _static_clip100(
            outcomes["_decoy_b_after"] + rng.normal(0, p["decoy_proxy_sd"], n)
        )
    if names["tertiary_obs"] in measurements:
        obs[names["tertiary_obs"]] = _static_clip100(
            outcomes["Y_tertiary_latent"] + rng.normal(0, p["tertiary_obs_sd"], n)
        )

    return obs


def _static_hidden_cause_candidate_interventions(
    template: Dict[str, Any],
) -> List[Dict[str, Any]]:
    names = template["names"]
    return [
        {},  # NoIntervention
        {names["decoy_knob_a"]: "on"},
        {names["decoy_knob_b"]: "on"},
        {names["weak_knob"]: "low"},
        {names["weak_knob"]: "high"},
        {names["true_lever_knob"]: "on"},
    ]


def _static_hidden_cause_assignment(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Per-unit intervention assignment for observational sampling.

    Current-practice uptake of every non-baseline knob scales with the hidden
    `HealthSeekingTrait`. Since HealthSeekingTrait also lowers BaselineSeverity,
    every common-practice intervention will look beneficial in observational
    data even though only `true_lever_knob` actually changes the latent driver.
    True-lever uptake stays low (not in guidelines).
    """
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    n = hidden["BaselineSeverity"].shape[0]
    names = cfg["template"]["names"]
    hs = hidden["HealthSeekingTrait"] / 100.0

    p_decoy_a = np.clip(p["assign_base_decoy_a"] + p["assign_slope_decoy_a"] * hs, 0.0, 0.95)
    p_decoy_b = np.clip(p["assign_base_decoy_b"] + p["assign_slope_decoy_b"] * hs, 0.0, 0.95)
    p_low = np.clip(p["assign_base_antacid_low"] + p["assign_slope_antacid_low"] * hs, 0.0, 0.95)
    p_high = np.clip(p["assign_base_antacid_high"] + p["assign_slope_antacid_high"] * hs, 0.0, 0.95)
    p_true = np.clip(p["assign_base_true_lever"] + p["assign_slope_true_lever"] * hs, 0.0, 0.95)

    u_weak = rng.random(n)
    weak_assign = np.where(
        u_weak < p_high,
        "high",
        np.where(u_weak < p_high + p_low, "low", "off"),
    )
    return {
        names["decoy_knob_a"]: np.where(rng.random(n) < p_decoy_a, "on", "off"),
        names["decoy_knob_b"]: np.where(rng.random(n) < p_decoy_b, "on", "off"),
        names["weak_knob"]: weak_assign,
        names["true_lever_knob"]: np.where(rng.random(n) < p_true, "on", "off"),
    }


# ---------------------------------------------------------------------------
# Archetype: confounded_action
# ---------------------------------------------------------------------------

def _static_confounded_default_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "severity_mean": rng.uniform(45.0, 55.0),
        "severity_sd": rng.uniform(15.0, 22.0),
        "baseline_outcome": rng.uniform(78.0, 84.0),
        "severity_penalty_on_outcome": rng.uniform(0.50, 0.65),
        "treatment_effects": {
            "off": 0.0,
            "low": rng.uniform(4.0, 6.5),
            "high": rng.uniform(10.0, 14.0),
        },
        "support_effect": rng.uniform(3.5, 5.5),
        "outcome_noise_sd": rng.uniform(5.0, 7.0),
        # Assignment propensity under current practice.
        "assignment_severity_intercept": -4.0,
        "assignment_severity_slope": 0.10,  # logit slope per severity unit
        "assignment_noise_sd": rng.uniform(0.4, 0.7),
        # Observation noise.
        "severity_proxy_sd": rng.uniform(9.0, 14.0),
        "outcome_proxy_sd": rng.uniform(7.0, 10.0),
        "secondary_outcome_sd": rng.uniform(8.0, 11.0),
        # Uniform-tail ranges.
        "uniform_severity_range": (0.0, 100.0),
    }


def _static_confounded_sample_hidden(
    cfg: Dict[str, Any],
    n: int,
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    mixture_weight = float(cfg["mixture_weight"])
    is_uniform = _static_mixture_unit_draws(rng, n, mixture_weight)

    sev_struct = _static_clip100(rng.normal(p["severity_mean"], p["severity_sd"], n))
    sev_uni = _static_clip100(rng.uniform(*p["uniform_severity_range"], n))
    severity = np.where(is_uniform, sev_uni, sev_struct)

    return {
        "LatentSeverity": severity,
        "is_uniform_tail": is_uniform.astype(np.int8),
    }


def _static_confounded_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentSeverity"].shape[0]

    treatment_setting = (intervention or {}).get(names["treatment_knob"], "off")
    support_setting = (intervention or {}).get(names["support_knob"], "off")

    treat_effect = float(p["treatment_effects"].get(treatment_setting, 0.0))
    support_effect = float(p["support_effect"] if support_setting == "on" else 0.0)

    target_outcome = (
        p["baseline_outcome"]
        - p["severity_penalty_on_outcome"] * hidden["LatentSeverity"]
        + treat_effect
        + support_effect
        + rng.normal(0, p["outcome_noise_sd"], n)
    )
    target_outcome = _static_clip100(target_outcome)
    secondary_outcome = _static_clip100(target_outcome + rng.normal(0, 3.0, n))

    return {
        "Y_target_outcome": target_outcome,
        "Y_secondary_outcome": secondary_outcome,
        "_treatment_setting": treatment_setting,
        "_support_setting": support_setting,
    }


def _static_confounded_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentSeverity"].shape[0]
    obs: Dict[str, np.ndarray] = {}

    if names["severity_proxy_a"] in measurements:
        obs[names["severity_proxy_a"]] = _static_clip100(
            hidden["LatentSeverity"] + rng.normal(0, p["severity_proxy_sd"], n)
        )
    if names["severity_proxy_b"] in measurements:
        obs[names["severity_proxy_b"]] = _static_clip100(
            hidden["LatentSeverity"] + rng.normal(0, p["severity_proxy_sd"], n)
        )
    if names["primary_target_obs"] in measurements:
        obs[names["primary_target_obs"]] = _static_clip100(
            outcomes["Y_target_outcome"] + rng.normal(0, p["outcome_proxy_sd"], n)
        )
    if names["secondary_target_obs"] in measurements:
        obs[names["secondary_target_obs"]] = _static_clip100(
            outcomes["Y_secondary_outcome"] + rng.normal(0, p["secondary_outcome_sd"], n)
        )
    if names["assignment_record"] in measurements:
        obs[names["assignment_record"]] = np.array(
            [outcomes["_treatment_setting"]] * n, dtype=object
        )
    return obs


def _static_confounded_candidate_interventions(
    template: Dict[str, Any],
) -> List[Dict[str, Any]]:
    names = template["names"]
    interventions: List[Dict[str, Any]] = []
    for t in ["off", "low", "high"]:
        for s in ["off", "on"]:
            interventions.append({names["treatment_knob"]: t, names["support_knob"]: s})
    return interventions


def _static_confounded_assignment(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentSeverity"].shape[0]
    # Sicker units more likely to receive high treatment under current practice.
    logits = (
        p["assignment_severity_intercept"]
        + p["assignment_severity_slope"] * hidden["LatentSeverity"]
        + rng.normal(0, p["assignment_noise_sd"], n)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    u = rng.random(n)
    treatment = np.where(probs > 0.66, "high", np.where(probs > 0.33, "low", "off"))
    # SupportiveCare assigned roughly randomly under current practice.
    support = np.where(rng.random(n) < 0.35, "on", "off")
    return {names["treatment_knob"]: treatment, names["support_knob"]: support}


# ---------------------------------------------------------------------------
# Archetype: mechanism_chain
# ---------------------------------------------------------------------------
#
# Three sequential stages produce a final outcome. Each stage has a hidden
# "yield rate." The bottleneck stage has low yield; the others have high
# yield. Intervening on the bottleneck moves the outcome a lot; intervening
# on a non-bottleneck stage barely moves it (Liebig-style multiplicative
# chain). The bottleneck is selected per world for balance.

def _static_mechanism_chain_default_params(rng: random.Random, *, bottleneck_stage: int) -> Dict[str, Any]:
    return {
        "bottleneck_stage": int(bottleneck_stage),
        "bottleneck_yield_mean": rng.uniform(28.0, 38.0),
        "bottleneck_yield_sd": rng.uniform(8.0, 11.0),
        "high_yield_mean": rng.uniform(68.0, 78.0),
        "high_yield_sd": rng.uniform(7.0, 10.0),
        "op_quality_mean": rng.uniform(48.0, 54.0),
        "op_quality_sd": rng.uniform(20.0, 24.0),
        "op_quality_loading_on_yield_high": rng.uniform(0.65, 0.90),
        "bottleneck_intervention_boost": rng.uniform(28.0, 36.0),
        "non_bottleneck_intervention_boost": rng.uniform(2.0, 5.0),
        "target_multiplier": rng.uniform(180.0, 220.0),
        "target_noise_sd": rng.uniform(3.0, 5.0),
        "primary_target_obs_sd": rng.uniform(5.0, 8.0),
        "secondary_target_obs_sd": rng.uniform(6.0, 9.0),
        "yield_proxy_sd": rng.uniform(10.0, 14.0),
        "assign_intervention_rate": rng.uniform(0.20, 0.32),
        "assign_bottleneck_op_penalty": rng.uniform(0.75, 1.00),
        "assign_non_bottleneck_op_bonus": rng.uniform(0.75, 1.00),
        "uniform_yield_range": (0.0, 100.0),
        "uniform_op_quality_range": (0.0, 100.0),
    }


def _static_mechanism_chain_sample_hidden(
    cfg: Dict[str, Any], n: int, *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    bottleneck = int(p["bottleneck_stage"])
    is_uniform = _static_mixture_unit_draws(rng, n, float(cfg["mixture_weight"]))
    op_quality_s = _static_clip100(rng.normal(p["op_quality_mean"], p["op_quality_sd"], n))
    op_quality_u = _static_clip100(rng.uniform(*p["uniform_op_quality_range"], n))
    op_quality = np.where(is_uniform, op_quality_u, op_quality_s)
    yields: Dict[str, np.ndarray] = {}
    for stage in (1, 2, 3):
        if stage == bottleneck:
            mean, sd = p["bottleneck_yield_mean"], p["bottleneck_yield_sd"]
        else:
            mean, sd = p["high_yield_mean"], p["high_yield_sd"]
        y_s = _static_clip100(rng.normal(mean, sd, n))
        y_u = _static_clip100(rng.uniform(*p["uniform_yield_range"], n))
        y = np.where(is_uniform, y_u, y_s)
        if stage != bottleneck:
            y = _static_clip100(
                y + p["op_quality_loading_on_yield_high"] * (op_quality - p["op_quality_mean"])
            )
        yields[f"Yield_{stage}"] = y
    return {
        **yields,
        "OperationQuality": op_quality,
        "is_uniform_tail": is_uniform.astype(np.int8),
    }


def _static_mechanism_chain_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["Yield_1"].shape[0]
    iv = intervention or {}
    eff = {}
    for stage in (1, 2, 3):
        y = hidden[f"Yield_{stage}"].astype(float).copy()
        knob = names[f"stage_{stage}_knob"]
        if iv.get(knob) == "on":
            boost = (
                p["bottleneck_intervention_boost"]
                if stage == int(p["bottleneck_stage"])
                else p["non_bottleneck_intervention_boost"]
            )
            y = _static_clip100(y + boost)
        eff[f"y{stage}"] = y
    # Multiplicative chain (Liebig-style)
    target = (
        p["target_multiplier"]
        * (eff["y1"] / 100.0)
        * (eff["y2"] / 100.0)
        * (eff["y3"] / 100.0)
        + rng.normal(0, p["target_noise_sd"], n)
    )
    target = _static_clip100(target)
    secondary = _static_clip100(target + rng.normal(0, 4.0, n))
    return {
        "Y_target_throughput": target,
        "Y_secondary_throughput": secondary,
        "_yield_1_after": eff["y1"],
        "_yield_2_after": eff["y2"],
        "_yield_3_after": eff["y3"],
    }


def _static_mechanism_chain_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["Yield_1"].shape[0]
    obs: Dict[str, np.ndarray] = {}
    if names["final_outcome_obs"] in measurements:
        obs[names["final_outcome_obs"]] = _static_clip100(
            outcomes["Y_target_throughput"] + rng.normal(0, p["primary_target_obs_sd"], n)
        )
    if names["secondary_outcome_obs"] in measurements:
        obs[names["secondary_outcome_obs"]] = _static_clip100(
            outcomes["Y_secondary_throughput"] + rng.normal(0, p["secondary_target_obs_sd"], n)
        )
    for stage in (1, 2, 3):
        proxy = names[f"stage_{stage}_proxy"]
        if proxy in measurements:
            obs[proxy] = _static_clip100(
                outcomes[f"_yield_{stage}_after"] + rng.normal(0, p["yield_proxy_sd"], n)
            )
    return obs


def _static_mechanism_chain_candidate_interventions(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = template["names"]
    return [
        {},
        {names["stage_1_knob"]: "on"},
        {names["stage_2_knob"]: "on"},
        {names["stage_3_knob"]: "on"},
    ]


def _static_mechanism_chain_assignment(
    cfg: Dict[str, Any], hidden: Dict[str, np.ndarray], *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = cfg["template"]["names"]
    p = cfg["parameters"]
    n = hidden["Yield_1"].shape[0]
    base_rate = float(p["assign_intervention_rate"])
    bottleneck = int(p["bottleneck_stage"])
    op_scaled = (hidden["OperationQuality"] - p["op_quality_mean"]) / 50.0
    assignments: Dict[str, np.ndarray] = {}
    for stage in (1, 2, 3):
        if stage == bottleneck:
            prob = base_rate - p["assign_bottleneck_op_penalty"] * op_scaled
        else:
            prob = base_rate + p["assign_non_bottleneck_op_bonus"] * op_scaled
        prob = np.clip(prob, 0.01, 0.90)
        assignments[names[f"stage_{stage}_knob"]] = np.where(rng.random(n) < prob, "on", "off")
    return assignments


# ---------------------------------------------------------------------------
# Archetype: negative_control
# ---------------------------------------------------------------------------
#
# Every offered intervention has near-zero do-effect on the target. Under
# observational data, current-practice uptake is confounded by
# HealthSeekingTrait (which also lowers BaselineSeverity), so non-baseline
# interventions LOOK beneficial. Gold answer = NoIntervention.

def _static_negative_control_default_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "substrate_mean": rng.uniform(48.0, 56.0),
        "substrate_sd": rng.uniform(16.0, 20.0),
        "healthseek_mean": rng.uniform(48.0, 55.0),
        "healthseek_sd": rng.uniform(14.0, 18.0),
        "baseline_intercept": rng.uniform(50.0, 56.0),
        "baseline_sd": rng.uniform(6.0, 9.0),
        "baseline_substrate_loading": rng.uniform(0.32, 0.42),
        "baseline_healthseek_loading": rng.uniform(0.62, 0.82),
        "target_noise_sd": rng.uniform(5.0, 8.0),
        # All knob effects are intentionally tiny.
        "trendy_knob_on_target": rng.uniform(0.2, 1.0),
        "conventional_knob_on_target": rng.uniform(0.1, 0.9),
        "research_low_on_target": rng.uniform(0.1, 0.8),
        "research_high_on_target": rng.uniform(0.4, 1.2),
        # Observation noise.
        "primary_target_obs_sd": rng.uniform(7.0, 10.0),
        "secondary_target_obs_sd": rng.uniform(8.0, 11.0),
        "wellbeing_proxy_sd": rng.uniform(10.0, 14.0),
        "engagement_proxy_sd": rng.uniform(11.0, 14.0),
        "healthseek_proxy_sd": rng.uniform(10.0, 14.0),
        # Assignment driven by HealthSeekingTrait.
        "assign_base_trendy": rng.uniform(0.08, 0.13),
        "assign_slope_trendy": rng.uniform(0.62, 0.78),
        "assign_base_conventional": rng.uniform(0.10, 0.15),
        "assign_slope_conventional": rng.uniform(0.52, 0.68),
        "assign_base_research_low": rng.uniform(0.10, 0.15),
        "assign_slope_research_low": rng.uniform(0.42, 0.56),
        "assign_base_research_high": rng.uniform(0.06, 0.10),
        "assign_slope_research_high": rng.uniform(0.48, 0.62),
        "uniform_substrate_range": (0.0, 100.0),
        "uniform_healthseek_range": (0.0, 100.0),
    }


def _static_negative_control_sample_hidden(
    cfg: Dict[str, Any], n: int, *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    mix = float(cfg["mixture_weight"])
    is_u = _static_mixture_unit_draws(rng, n, mix)
    healthseek_s = _static_clip100(rng.normal(p["healthseek_mean"], p["healthseek_sd"], n))
    substrate_s = _static_clip100(rng.normal(p["substrate_mean"], p["substrate_sd"], n))
    baseline_s = _static_clip100(
        p["baseline_intercept"]
        + p["baseline_substrate_loading"] * substrate_s
        - p["baseline_healthseek_loading"] * (healthseek_s - p["healthseek_mean"])
        + rng.normal(0, p["baseline_sd"], n)
    )
    healthseek_u = _static_clip100(rng.uniform(*p["uniform_healthseek_range"], n))
    substrate_u = _static_clip100(rng.uniform(*p["uniform_substrate_range"], n))
    baseline_u = _static_clip100(rng.uniform(0.0, 100.0, n))
    return {
        "BaselineSeverity": np.where(is_u, baseline_u, baseline_s),
        "BurdenSubstrate": np.where(is_u, substrate_u, substrate_s),
        "HealthSeekingTrait": np.where(is_u, healthseek_u, healthseek_s),
        "is_uniform_tail": is_u.astype(np.int8),
    }


def _static_negative_control_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["BaselineSeverity"].shape[0]
    iv = intervention or {}
    target = hidden["BaselineSeverity"].astype(float).copy() + rng.normal(0, p["target_noise_sd"], n)
    if iv.get(names["trendy_knob"]) == "on":
        target = target + p["trendy_knob_on_target"]
    if iv.get(names["conventional_knob"]) == "on":
        target = target + p["conventional_knob_on_target"]
    rk_setting = iv.get(names["research_backed_knob"], "off")
    if rk_setting == "low":
        target = target + p["research_low_on_target"]
    elif rk_setting == "high":
        target = target + p["research_high_on_target"]
    target = _static_clip100(target)
    secondary = _static_clip100(target + rng.normal(0, 4.0, n))
    return {
        "Y_target_severity": target,
        "Y_secondary_target": secondary,
        "_healthseek_after": hidden["HealthSeekingTrait"],
        "_substrate_after": hidden["BurdenSubstrate"],
    }


def _static_negative_control_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["BaselineSeverity"].shape[0]
    obs: Dict[str, np.ndarray] = {}
    if names["primary_outcome_obs"] in measurements:
        obs[names["primary_outcome_obs"]] = _static_clip100(
            outcomes["Y_target_severity"] + rng.normal(0, p["primary_target_obs_sd"], n)
        )
    if names["secondary_outcome_obs"] in measurements:
        obs[names["secondary_outcome_obs"]] = _static_clip100(
            outcomes["Y_secondary_target"] + rng.normal(0, p["secondary_target_obs_sd"], n)
        )
    if names["wellbeing_proxy"] in measurements:
        obs[names["wellbeing_proxy"]] = _static_clip100(
            100.0 - outcomes["Y_target_severity"] + rng.normal(0, p["wellbeing_proxy_sd"], n)
        )
    if names["engagement_proxy"] in measurements:
        obs[names["engagement_proxy"]] = _static_clip100(
            outcomes["_healthseek_after"] + rng.normal(0, p["engagement_proxy_sd"], n)
        )
    if names["healthseek_proxy"] in measurements:
        obs[names["healthseek_proxy"]] = _static_clip100(
            outcomes["_healthseek_after"] + rng.normal(0, p["healthseek_proxy_sd"], n)
        )
    return obs


def _static_negative_control_candidate_interventions(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = template["names"]
    return [
        {},
        {names["trendy_knob"]: "on"},
        {names["conventional_knob"]: "on"},
        {names["research_backed_knob"]: "low"},
        {names["research_backed_knob"]: "high"},
    ]


def _static_negative_control_assignment(
    cfg: Dict[str, Any], hidden: Dict[str, np.ndarray], *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["BaselineSeverity"].shape[0]
    hs = hidden["HealthSeekingTrait"] / 100.0
    p_trendy = np.clip(p["assign_base_trendy"] + p["assign_slope_trendy"] * hs, 0.0, 0.95)
    p_conv = np.clip(p["assign_base_conventional"] + p["assign_slope_conventional"] * hs, 0.0, 0.95)
    p_low = np.clip(p["assign_base_research_low"] + p["assign_slope_research_low"] * hs, 0.0, 0.95)
    p_high = np.clip(p["assign_base_research_high"] + p["assign_slope_research_high"] * hs, 0.0, 0.95)
    u = rng.random(n)
    research = np.where(u < p_high, "high", np.where(u < p_high + p_low, "low", "off"))
    return {
        names["trendy_knob"]: np.where(rng.random(n) < p_trendy, "on", "off"),
        names["conventional_knob"]: np.where(rng.random(n) < p_conv, "on", "off"),
        names["research_backed_knob"]: research,
    }


# ---------------------------------------------------------------------------
# Archetype: hidden_subtype
# ---------------------------------------------------------------------------
#
# Two latent subtypes with opposite-best treatments. A noisy SubtypeScreen
# proxy. A conditional policy keyed on the screen beats every static
# intervention. Answer schema: conditional_policy.

def _static_hidden_subtype_default_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "p_subtype_1": rng.uniform(0.42, 0.58),
        "screen_reliability": rng.uniform(0.72, 0.85),
        "baseline_risk_mean": rng.uniform(40.0, 50.0),
        "baseline_risk_sd": rng.uniform(15.0, 20.0),
        "baseline_outcome_intercept": rng.uniform(60.0, 66.0),
        "baseline_risk_penalty": rng.uniform(0.20, 0.30),
        # Treatment effects: (subtype_0, subtype_1) — A favors 0, C favors 1, B middle.
        "treatment_a_effect_subtype_0": rng.uniform(14.0, 18.0),
        "treatment_a_effect_subtype_1": rng.uniform(2.0, 5.0),
        "treatment_b_effect_subtype_0": rng.uniform(7.0, 10.0),
        "treatment_b_effect_subtype_1": rng.uniform(7.0, 10.0),
        "treatment_c_effect_subtype_0": rng.uniform(2.0, 5.0),
        "treatment_c_effect_subtype_1": rng.uniform(14.0, 18.0),
        "target_noise_sd": rng.uniform(5.0, 7.0),
        "subtype_screen_sd": rng.uniform(8.0, 13.0),
        "secondary_screen_sd": rng.uniform(12.0, 18.0),
        "baseline_risk_proxy_sd": rng.uniform(10.0, 14.0),
        "primary_target_obs_sd": rng.uniform(6.0, 9.0),
        "secondary_target_obs_sd": rng.uniform(7.0, 10.0),
        # Assignment: roughly even across treatments under current practice.
        "assign_treatment_rate": rng.uniform(0.32, 0.42),
    }


def _static_hidden_subtype_sample_hidden(
    cfg: Dict[str, Any], n: int, *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    mix = float(cfg["mixture_weight"])
    is_u = _static_mixture_unit_draws(rng, n, mix)
    subtype_s = (rng.random(n) < p["p_subtype_1"]).astype(np.int8)
    subtype_u = (rng.random(n) < 0.5).astype(np.int8)
    baseline_risk_s = _static_clip100(rng.normal(p["baseline_risk_mean"], p["baseline_risk_sd"], n))
    baseline_risk_u = _static_clip100(rng.uniform(0.0, 100.0, n))
    return {
        "LatentSubtype": np.where(is_u, subtype_u, subtype_s).astype(np.int8),
        "BaselineRisk": np.where(is_u, baseline_risk_u, baseline_risk_s),
        "is_uniform_tail": is_u.astype(np.int8),
    }


def _static_hidden_subtype_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentSubtype"].shape[0]
    iv = intervention or {}
    sub = hidden["LatentSubtype"].astype(int)
    target = (
        p["baseline_outcome_intercept"]
        - p["baseline_risk_penalty"] * hidden["BaselineRisk"]
        + rng.normal(0, p["target_noise_sd"], n)
    )
    if iv.get(names["treatment_a_knob"]) == "on":
        target = target + np.where(
            sub == 0, p["treatment_a_effect_subtype_0"], p["treatment_a_effect_subtype_1"]
        )
    if iv.get(names["treatment_b_knob"]) == "on":
        target = target + np.where(
            sub == 0, p["treatment_b_effect_subtype_0"], p["treatment_b_effect_subtype_1"]
        )
    if iv.get(names["treatment_c_knob"]) == "on":
        target = target + np.where(
            sub == 0, p["treatment_c_effect_subtype_0"], p["treatment_c_effect_subtype_1"]
        )
    target = _static_clip100(target)
    secondary = _static_clip100(target + rng.normal(0, 4.0, n))
    return {
        "Y_target_outcome": target,
        "Y_secondary_outcome": secondary,
        "_subtype_after": sub,
    }


def _static_hidden_subtype_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentSubtype"].shape[0]
    obs: Dict[str, np.ndarray] = {}
    sub = hidden["LatentSubtype"].astype(int)
    # Noisy 0-100 screen: subtype 0 has low mean, subtype 1 has high mean.
    screen_signal = np.where(sub == 1, 70.0, 30.0)
    if names["subtype_screen"] in measurements:
        obs[names["subtype_screen"]] = _static_clip100(
            screen_signal + rng.normal(0, p["subtype_screen_sd"], n)
        )
    if names["secondary_subtype_screen"] in measurements:
        obs[names["secondary_subtype_screen"]] = _static_clip100(
            screen_signal + rng.normal(0, p["secondary_screen_sd"], n)
        )
    if names["baseline_risk_proxy"] in measurements:
        obs[names["baseline_risk_proxy"]] = _static_clip100(
            hidden["BaselineRisk"] + rng.normal(0, p["baseline_risk_proxy_sd"], n)
        )
    if names["target_outcome_obs"] in measurements:
        obs[names["target_outcome_obs"]] = _static_clip100(
            outcomes["Y_target_outcome"] + rng.normal(0, p["primary_target_obs_sd"], n)
        )
    if names["secondary_outcome_obs"] in measurements:
        obs[names["secondary_outcome_obs"]] = _static_clip100(
            outcomes["Y_secondary_outcome"] + rng.normal(0, p["secondary_target_obs_sd"], n)
        )
    return obs


def _static_hidden_subtype_candidate_interventions(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = template["names"]
    return [
        {},
        {names["treatment_a_knob"]: "on"},
        {names["treatment_b_knob"]: "on"},
        {names["treatment_c_knob"]: "on"},
    ]


def _static_hidden_subtype_assignment(
    cfg: Dict[str, Any], hidden: Dict[str, np.ndarray], *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentSubtype"].shape[0]
    r = float(p["assign_treatment_rate"])
    # Three treatments, roughly even, with the rest being NoIntervention.
    return {
        names["treatment_a_knob"]: np.where(rng.random(n) < r, "on", "off"),
        names["treatment_b_knob"]: np.where(rng.random(n) < r, "on", "off"),
        names["treatment_c_knob"]: np.where(rng.random(n) < r, "on", "off"),
    }


def _static_hidden_subtype_score_conditional_policy(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    *,
    n_oracle: int,
    seed: int,
) -> float:
    """Expected utility under a conditional policy keyed on a noisy proxy.
    Used by the oracle to pick the best conditional gold policy."""
    names = cfg["template"]["names"]
    hidden = _static_sample_hidden(cfg, n_oracle, seed=seed)
    branch_var = policy["branch_variable"]
    threshold = float(policy["branch_threshold"])
    k_above = policy["if_above"]
    k_below = policy["if_below"]
    # Get the proxy reading per unit (must be one of the measurements).
    proxy_obs = _static_observe(
        cfg, hidden, _static_apply(cfg, hidden, {}, seed=seed + 11), [branch_var], seed=seed + 23
    )[branch_var]
    masks = {"above": proxy_obs >= threshold, "below": proxy_obs < threshold}
    util = np.zeros_like(proxy_obs)
    for label, mask, sub_iv in (("above", masks["above"], k_above), ("below", masks["below"], k_below)):
        if int(mask.sum()) == 0:
            continue
        sub_hidden = {k: v[mask] if isinstance(v, np.ndarray) else v for k, v in hidden.items()}
        sub_out = _static_apply(cfg, sub_hidden, dict(sub_iv), seed=seed + 31 * (hash(label) & 0xFFFF))
        util[mask] = sub_out["Y_target_outcome"]
    return float(np.mean(util))


# ---------------------------------------------------------------------------
# Archetype: latent_regime_discovery
# ---------------------------------------------------------------------------
# Two hidden response regimes create a bimodal latent axis. Several observed
# panels are plausible, but only two noisy panels reveal the split. Alpha helps
# the low-axis regime; Beta helps the high-axis regime; global single-action
# treatment is inferior to a regime-aware policy.

def _static_latent_regime_default_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "p_regime_high": rng.uniform(0.45, 0.55),
        "axis_low_mean": rng.uniform(27.0, 34.0),
        "axis_high_mean": rng.uniform(66.0, 73.0),
        "axis_sd": rng.uniform(6.0, 9.0),
        "baseline_risk_mean": rng.uniform(44.0, 54.0),
        "baseline_risk_sd": rng.uniform(14.0, 19.0),
        "baseline_outcome_intercept": rng.uniform(63.0, 69.0),
        "baseline_risk_penalty": rng.uniform(0.18, 0.28),
        "treatment_a_effect_low": rng.uniform(16.0, 21.0),
        "treatment_a_effect_high": rng.uniform(-6.0, -2.0),
        "treatment_c_effect_low": rng.uniform(-6.0, -2.0),
        "treatment_c_effect_high": rng.uniform(16.0, 21.0),
        "broad_effect_low": rng.uniform(5.0, 8.0),
        "broad_effect_high": rng.uniform(5.0, 8.0),
        "support_effect": rng.uniform(2.0, 4.0),
        "decoy_a_effect": rng.uniform(-1.0, 1.5),
        "decoy_b_effect": rng.uniform(-1.0, 2.0),
        "palliative_low_on_relief": rng.uniform(5.0, 7.0),
        "palliative_high_on_relief": rng.uniform(8.0, 11.0),
        "palliative_target_effect": rng.uniform(-0.5, 1.0),
        "monitoring_effect": rng.uniform(-0.5, 1.0),
        "alpha_beta_combo_penalty": rng.uniform(10.0, 14.0),
        "multi_action_toxicity": rng.uniform(1.0, 2.5),
        "target_noise_sd": rng.uniform(4.5, 6.5),
        "regime_proxy_a_sd": rng.uniform(7.0, 10.0),
        "regime_proxy_b_sd": rng.uniform(10.0, 14.0),
        "baseline_risk_proxy_sd": rng.uniform(8.0, 12.0),
        "decoy_proxy_sd": rng.uniform(10.0, 15.0),
        "primary_target_obs_sd": rng.uniform(5.0, 7.5),
        "secondary_target_obs_sd": rng.uniform(6.0, 8.0),
        "relief_obs_sd": rng.uniform(6.0, 8.0),
        "tolerability_obs_sd": rng.uniform(5.0, 7.0),
        "assign_action_rate": rng.uniform(0.18, 0.26),
    }


def _static_latent_regime_sample_hidden(
    cfg: Dict[str, Any], n: int, *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    mix = float(cfg["mixture_weight"])
    is_u = _static_mixture_unit_draws(rng, n, mix)
    regime_s = (rng.random(n) < p["p_regime_high"]).astype(np.int8)
    regime_u = (rng.random(n) < 0.5).astype(np.int8)
    regime = np.where(is_u, regime_u, regime_s).astype(np.int8)
    axis_s = rng.normal(
        np.where(regime == 1, p["axis_high_mean"], p["axis_low_mean"]),
        p["axis_sd"],
        n,
    )
    axis_u = rng.uniform(0.0, 100.0, n)
    baseline_risk_s = _static_clip100(rng.normal(p["baseline_risk_mean"], p["baseline_risk_sd"], n))
    baseline_risk_u = _static_clip100(rng.uniform(0.0, 100.0, n))
    # Plausible-but-not-causal nuisance structure.
    exposure = _static_clip100(35.0 + 0.35 * baseline_risk_s + rng.normal(0, 13.0, n))
    site_practice = _static_clip100(42.0 + 0.25 * baseline_risk_s + rng.normal(0, 14.0, n))
    viral_pattern = _static_clip100(50.0 + rng.normal(0, 16.0, n))
    return {
        "LatentRegime": regime,
        "LatentRegimeAxis": _static_clip100(np.where(is_u, axis_u, axis_s)),
        "BaselineRisk": np.where(is_u, baseline_risk_u, baseline_risk_s),
        "ExposureHistory": np.where(is_u, rng.uniform(0.0, 100.0, n), exposure),
        "SitePractice": np.where(is_u, rng.uniform(0.0, 100.0, n), site_practice),
        "ViralPattern": np.where(is_u, rng.uniform(0.0, 100.0, n), viral_pattern),
        "is_uniform_tail": is_u.astype(np.int8),
    }


def _static_latent_regime_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentRegime"].shape[0]
    iv = intervention or {}
    regime = hidden["LatentRegime"].astype(int)
    target = (
        p["baseline_outcome_intercept"]
        - p["baseline_risk_penalty"] * hidden["BaselineRisk"]
        + rng.normal(0, p["target_noise_sd"], n)
    )
    active_primary = 0
    if iv.get(names["treatment_a_knob"]) == "on":
        active_primary += 1
        target = target + np.where(regime == 0, p["treatment_a_effect_low"], p["treatment_a_effect_high"])
    if iv.get(names["treatment_c_knob"]) == "on":
        active_primary += 1
        target = target + np.where(regime == 0, p["treatment_c_effect_low"], p["treatment_c_effect_high"])
    if iv.get(names["treatment_b_knob"]) == "on":
        target = target + np.where(regime == 0, p["broad_effect_low"], p["broad_effect_high"])
    if iv.get(names["support_knob"]) == "on":
        target = target + p["support_effect"]
    if iv.get(names["decoy_knob_a"]) == "on":
        target = target + p["decoy_a_effect"]
    if iv.get(names["decoy_knob_b"]) == "on":
        target = target + p["decoy_b_effect"]
    if iv.get(names["monitoring_knob"]) == "on":
        target = target + p["monitoring_effect"]
    if iv.get(names["palliative_knob"]) in {"low", "high"}:
        target = target + p["palliative_target_effect"]
    if active_primary >= 2:
        target = target - p["alpha_beta_combo_penalty"]
    n_active = sum(1 for value in iv.values() if str(value) != "off")
    if n_active > 1:
        target = target - p["multi_action_toxicity"] * (n_active - 1)
    target = _static_clip100(target)
    secondary = _static_clip100(target + rng.normal(0, 3.5, n))
    relief = _static_clip100(
        target
        + (p["palliative_low_on_relief"] if iv.get(names["palliative_knob"]) == "low" else 0.0)
        + (p["palliative_high_on_relief"] if iv.get(names["palliative_knob"]) == "high" else 0.0)
        + rng.normal(0, 5.0, n)
    )
    tolerability = _static_clip100(100.0 - 0.45 * hidden["BaselineRisk"] - 5.0 * max(n_active - 1, 0) + rng.normal(0, 5.0, n))
    return {
        "Y_target_outcome": target,
        "Y_secondary_outcome": secondary,
        "Y_relief_outcome": relief,
        "Y_tolerability": tolerability,
        "_regime": regime,
        "_axis": hidden["LatentRegimeAxis"],
    }


def _static_latent_regime_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentRegime"].shape[0]
    obs: Dict[str, np.ndarray] = {}
    if names["target_outcome_obs"] in measurements:
        obs[names["target_outcome_obs"]] = _static_clip100(
            outcomes["Y_target_outcome"] + rng.normal(0, p["primary_target_obs_sd"], n)
        )
    if names["secondary_outcome_obs"] in measurements:
        obs[names["secondary_outcome_obs"]] = _static_clip100(
            outcomes["Y_secondary_outcome"] + rng.normal(0, p["secondary_target_obs_sd"], n)
        )
    if names["regime_proxy_a"] in measurements:
        obs[names["regime_proxy_a"]] = _static_clip100(
            hidden["LatentRegimeAxis"] + rng.normal(0, p["regime_proxy_a_sd"], n)
        )
    if names["regime_proxy_b"] in measurements:
        obs[names["regime_proxy_b"]] = _static_clip100(
            hidden["LatentRegimeAxis"] + rng.normal(0, p["regime_proxy_b_sd"], n)
        )
    if names["baseline_risk_proxy"] in measurements:
        obs[names["baseline_risk_proxy"]] = _static_clip100(
            hidden["BaselineRisk"] + rng.normal(0, p["baseline_risk_proxy_sd"], n)
        )
    if names["decoy_proxy_a"] in measurements:
        obs[names["decoy_proxy_a"]] = _static_clip100(
            hidden["ExposureHistory"] + rng.normal(0, p["decoy_proxy_sd"], n)
        )
    if names["decoy_proxy_b"] in measurements:
        obs[names["decoy_proxy_b"]] = _static_clip100(
            hidden["SitePractice"] + rng.normal(0, p["decoy_proxy_sd"], n)
        )
    if names["decoy_proxy_c"] in measurements:
        obs[names["decoy_proxy_c"]] = _static_clip100(
            hidden["ViralPattern"] + rng.normal(0, p["decoy_proxy_sd"], n)
        )
    if names["relief_proxy"] in measurements:
        obs[names["relief_proxy"]] = _static_clip100(
            outcomes["Y_relief_outcome"] + rng.normal(0, p["relief_obs_sd"], n)
        )
    if names["tolerability_proxy"] in measurements:
        obs[names["tolerability_proxy"]] = _static_clip100(
            outcomes["Y_tolerability"] + rng.normal(0, p["tolerability_obs_sd"], n)
        )
    return obs


def _static_latent_regime_candidate_interventions(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = template["names"]
    return [
        {},
        {names["treatment_a_knob"]: "on"},
        {names["treatment_b_knob"]: "on"},
        {names["treatment_c_knob"]: "on"},
        {names["support_knob"]: "on"},
        {names["decoy_knob_a"]: "on"},
        {names["decoy_knob_b"]: "on"},
        {names["palliative_knob"]: "low"},
        {names["palliative_knob"]: "high"},
        {names["monitoring_knob"]: "on"},
    ]


def _static_latent_regime_assignment(
    cfg: Dict[str, Any], hidden: Dict[str, np.ndarray], *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["LatentRegime"].shape[0]
    r = float(p["assign_action_rate"])
    weak = rng.random(n)
    return {
        names["treatment_a_knob"]: np.where(rng.random(n) < r, "on", "off"),
        names["treatment_b_knob"]: np.where(rng.random(n) < r, "on", "off"),
        names["treatment_c_knob"]: np.where(rng.random(n) < r, "on", "off"),
        names["support_knob"]: np.where(rng.random(n) < r, "on", "off"),
        names["decoy_knob_a"]: np.where(rng.random(n) < r, "on", "off"),
        names["decoy_knob_b"]: np.where(rng.random(n) < r, "on", "off"),
        names["palliative_knob"]: np.where(weak < r * 0.6, "high", np.where(weak < r * 1.2, "low", "off")),
        names["monitoring_knob"]: np.where(rng.random(n) < r, "on", "off"),
    }


def _static_score_conditional_policy_utility(
    cfg: Dict[str, Any],
    policy: Dict[str, Any],
    *,
    n_oracle: int,
    seed: int,
) -> float:
    """Expected utility under a two-branch policy keyed on a public measurement."""
    hidden = _static_sample_hidden(cfg, n_oracle, seed=seed)
    branch_var = policy["branch_variable"]
    threshold = float(policy["branch_threshold"])
    k_above = policy["if_above"]
    k_below = policy["if_below"]
    baseline = _static_apply(cfg, hidden, {}, seed=seed + 11)
    proxy_obs = _static_observe(cfg, hidden, baseline, [branch_var], seed=seed + 23)[branch_var]
    util = np.zeros_like(proxy_obs, dtype=float)
    for offset, (mask, sub_iv) in enumerate(((proxy_obs >= threshold, k_above), (proxy_obs < threshold, k_below))):
        if int(mask.sum()) == 0:
            continue
        sub_hidden = {k: v[mask] if isinstance(v, np.ndarray) else v for k, v in hidden.items()}
        sub_out = _static_apply(cfg, sub_hidden, dict(sub_iv), seed=seed + 31 * (offset + 1))
        util[mask] = _static_utility_from_outcomes(cfg, sub_out)
    return float(np.mean(util))


# ---------------------------------------------------------------------------
# Archetype: anomaly_discovery
# ---------------------------------------------------------------------------
#
# (1-π) of units are sampled from a structured "normal" feature distribution.
# π ∈ [0.05, 0.10] are sampled from an "anomaly" distribution that shifts on
# two features simultaneously. Agent must identify anomalous units by their
# feature signature. Answer schema: anomaly_identification (flagged_unit_ids
# + anomaly_rule).

def _static_anomaly_default_params(rng: random.Random) -> Dict[str, Any]:
    # Two of the four features are the anomaly-discriminating pair; chosen
    # per world. We rotate this choice via the builder for diversity.
    return {
        "p_anomaly": rng.uniform(0.05, 0.10),
        "anomaly_shift_a": rng.uniform(22.0, 30.0),  # mean shift on feature_a
        "anomaly_shift_b": rng.uniform(-30.0, -22.0),  # mean shift on feature_b (opposite sign)
        "feature_mean": rng.uniform(45.0, 55.0),
        "feature_sd": rng.uniform(11.0, 15.0),
        "feature_obs_sd": rng.uniform(2.0, 4.0),
        "secondary_signal_anomaly_loading": rng.uniform(0.30, 0.45),
        "uniform_feature_range": (0.0, 100.0),
        "primary_observe_field": "target_obs",
        # Pseudo "intervention" knob: a no-op audit toggle so the framework's
        # candidate-intervention code still produces something. Gold is {}.
        "audit_knob_no_op_target": 0.0,
    }


def _static_anomaly_sample_hidden(
    cfg: Dict[str, Any], n: int, *, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    is_anomaly = (rng.random(n) < p["p_anomaly"]).astype(np.int8)
    feat_a = _static_clip100(
        rng.normal(p["feature_mean"], p["feature_sd"], n)
        + is_anomaly.astype(float) * p["anomaly_shift_a"]
    )
    feat_b = _static_clip100(
        rng.normal(p["feature_mean"], p["feature_sd"], n)
        + is_anomaly.astype(float) * p["anomaly_shift_b"]
    )
    feat_c = _static_clip100(rng.normal(p["feature_mean"], p["feature_sd"], n))
    feat_d = _static_clip100(rng.normal(p["feature_mean"], p["feature_sd"], n))
    # Secondary signal: a weaker independent anomaly indicator.
    secondary = _static_clip100(
        50.0
        + p["secondary_signal_anomaly_loading"] * (is_anomaly.astype(float) * 40.0 - 5.0)
        + rng.normal(0, 12.0, n)
    )
    # The "target" for this archetype is just the anomaly indicator on a 0-100 scale.
    return {
        "IsAnomaly": is_anomaly,
        "Feature_A_state": feat_a,
        "Feature_B_state": feat_b,
        "Feature_C_state": feat_c,
        "Feature_D_state": feat_d,
        "SecondarySignal_state": secondary,
    }


def _static_anomaly_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    # No-op apply: there are no causal interventions in this archetype.
    return {
        "Y_anomaly_indicator": hidden["IsAnomaly"].astype(float) * 100.0,
        "_feat_a": hidden["Feature_A_state"],
        "_feat_b": hidden["Feature_B_state"],
        "_feat_c": hidden["Feature_C_state"],
        "_feat_d": hidden["Feature_D_state"],
        "_secondary": hidden["SecondarySignal_state"],
    }


def _static_anomaly_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    names = cfg["template"]["names"]
    n = hidden["IsAnomaly"].shape[0]
    obs: Dict[str, np.ndarray] = {}
    if names["feature_a"] in measurements:
        obs[names["feature_a"]] = _static_clip100(outcomes["_feat_a"] + rng.normal(0, p["feature_obs_sd"], n))
    if names["feature_b"] in measurements:
        obs[names["feature_b"]] = _static_clip100(outcomes["_feat_b"] + rng.normal(0, p["feature_obs_sd"], n))
    if names["feature_c"] in measurements:
        obs[names["feature_c"]] = _static_clip100(outcomes["_feat_c"] + rng.normal(0, p["feature_obs_sd"], n))
    if names["feature_d"] in measurements:
        obs[names["feature_d"]] = _static_clip100(outcomes["_feat_d"] + rng.normal(0, p["feature_obs_sd"], n))
    if names["secondary_signal"] in measurements:
        obs[names["secondary_signal"]] = _static_clip100(outcomes["_secondary"] + rng.normal(0, 6.0, n))
    return obs


def _static_anomaly_candidate_interventions(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Only NoIntervention is meaningful in this archetype.
    return [{}]


def _static_anomaly_assignment(
    cfg: Dict[str, Any], hidden: Dict[str, np.ndarray], *, seed: int
) -> Dict[str, np.ndarray]:
    # No knobs => empty dict (means assignment-tuple key for stratification is ()).
    return {}


# ---------------------------------------------------------------------------
# Archetype dispatch
# ---------------------------------------------------------------------------

def _static_default_params(archetype: str, rng: random.Random, *, bottleneck_stage: int = 1) -> Dict[str, Any]:
    if archetype == "hidden_cause":
        return _static_hidden_cause_default_params(rng)
    if archetype == "confounded_action":
        return _static_confounded_default_params(rng)
    if archetype == "mechanism_chain":
        return _static_mechanism_chain_default_params(rng, bottleneck_stage=bottleneck_stage)
    if archetype == "negative_control":
        return _static_negative_control_default_params(rng)
    if archetype == "hidden_subtype":
        return _static_hidden_subtype_default_params(rng)
    if archetype == "anomaly_discovery":
        return _static_anomaly_default_params(rng)
    if archetype == "latent_regime_discovery":
        return _static_latent_regime_default_params(rng)
    raise KeyError(archetype)


def _static_sample_hidden(cfg: Dict[str, Any], n: int, *, seed: int) -> Dict[str, np.ndarray]:
    arch = cfg["archetype"]
    if arch == "hidden_cause":
        return _static_hidden_cause_sample_hidden(cfg, n, seed=seed)
    if arch == "confounded_action":
        return _static_confounded_sample_hidden(cfg, n, seed=seed)
    if arch == "mechanism_chain":
        return _static_mechanism_chain_sample_hidden(cfg, n, seed=seed)
    if arch == "negative_control":
        return _static_negative_control_sample_hidden(cfg, n, seed=seed)
    if arch == "hidden_subtype":
        return _static_hidden_subtype_sample_hidden(cfg, n, seed=seed)
    if arch == "anomaly_discovery":
        return _static_anomaly_sample_hidden(cfg, n, seed=seed)
    if arch == "latent_regime_discovery":
        return _static_latent_regime_sample_hidden(cfg, n, seed=seed)
    raise KeyError(arch)


def _static_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    arch = cfg["archetype"]
    if arch == "hidden_cause":
        return _static_hidden_cause_apply(cfg, hidden, intervention, seed=seed)
    if arch == "confounded_action":
        return _static_confounded_apply(cfg, hidden, intervention, seed=seed)
    if arch == "mechanism_chain":
        return _static_mechanism_chain_apply(cfg, hidden, intervention, seed=seed)
    if arch == "negative_control":
        return _static_negative_control_apply(cfg, hidden, intervention, seed=seed)
    if arch == "hidden_subtype":
        return _static_hidden_subtype_apply(cfg, hidden, intervention, seed=seed)
    if arch == "anomaly_discovery":
        return _static_anomaly_apply(cfg, hidden, intervention, seed=seed)
    if arch == "latent_regime_discovery":
        return _static_latent_regime_apply(cfg, hidden, intervention, seed=seed)
    raise KeyError(arch)


def _static_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    arch = cfg["archetype"]
    if arch == "hidden_cause":
        return _static_hidden_cause_observe(cfg, hidden, outcomes, measurements, seed=seed)
    if arch == "confounded_action":
        return _static_confounded_observe(cfg, hidden, outcomes, measurements, seed=seed)
    if arch == "mechanism_chain":
        return _static_mechanism_chain_observe(cfg, hidden, outcomes, measurements, seed=seed)
    if arch == "negative_control":
        return _static_negative_control_observe(cfg, hidden, outcomes, measurements, seed=seed)
    if arch == "hidden_subtype":
        return _static_hidden_subtype_observe(cfg, hidden, outcomes, measurements, seed=seed)
    if arch == "anomaly_discovery":
        return _static_anomaly_observe(cfg, hidden, outcomes, measurements, seed=seed)
    if arch == "latent_regime_discovery":
        return _static_latent_regime_observe(cfg, hidden, outcomes, measurements, seed=seed)
    raise KeyError(arch)


def _static_candidate_interventions(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    arch = cfg["archetype"]
    if arch == "hidden_cause":
        return _static_hidden_cause_candidate_interventions(cfg["template"])
    if arch == "confounded_action":
        return _static_confounded_candidate_interventions(cfg["template"])
    if arch == "mechanism_chain":
        return _static_mechanism_chain_candidate_interventions(cfg["template"])
    if arch == "negative_control":
        return _static_negative_control_candidate_interventions(cfg["template"])
    if arch == "hidden_subtype":
        return _static_hidden_subtype_candidate_interventions(cfg["template"])
    if arch == "anomaly_discovery":
        return _static_anomaly_candidate_interventions(cfg["template"])
    if arch == "latent_regime_discovery":
        return _static_latent_regime_candidate_interventions(cfg["template"])
    raise KeyError(arch)


def _static_assignment(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    arch = cfg["archetype"]
    if arch == "hidden_cause":
        return _static_hidden_cause_assignment(cfg, hidden, seed=seed)
    if arch == "confounded_action":
        return _static_confounded_assignment(cfg, hidden, seed=seed)
    if arch == "mechanism_chain":
        return _static_mechanism_chain_assignment(cfg, hidden, seed=seed)
    if arch == "negative_control":
        return _static_negative_control_assignment(cfg, hidden, seed=seed)
    if arch == "hidden_subtype":
        return _static_hidden_subtype_assignment(cfg, hidden, seed=seed)
    if arch == "anomaly_discovery":
        return _static_anomaly_assignment(cfg, hidden, seed=seed)
    if arch == "latent_regime_discovery":
        return _static_latent_regime_assignment(cfg, hidden, seed=seed)
    raise KeyError(arch)


_STATIC_HIGHER_IS_BETTER = {
    "hidden_cause": False,       # severity, lower is better
    "confounded_action": True,
    "mechanism_chain": True,     # throughput, higher is better
    "negative_control": False,   # severity, lower is better
    "hidden_subtype": True,      # recovery/outcome, higher is better
    "anomaly_discovery": True,   # not used as utility; agent submits anomaly id
    "latent_regime_discovery": True,
}


def _static_primary_target_higher_is_better(cfg: Dict[str, Any]) -> bool:
    return _STATIC_HIGHER_IS_BETTER[cfg["archetype"]]


def _static_utility_from_outcomes(
    cfg: Dict[str, Any],
    outcomes: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return per-unit utility (higher = better) for oracle ranking."""
    arch = cfg["archetype"]
    if arch == "hidden_cause":
        return -outcomes["Y_target_severity"]
    if arch == "confounded_action":
        return outcomes["Y_target_outcome"]
    if arch == "mechanism_chain":
        return outcomes["Y_target_throughput"]
    if arch == "negative_control":
        return -outcomes["Y_target_severity"]
    if arch == "hidden_subtype":
        return outcomes["Y_target_outcome"]
    if arch == "latent_regime_discovery":
        return outcomes["Y_target_outcome"]
    if arch == "anomaly_discovery":
        # Anomaly discovery has no intervention utility; return zeros so oracle
        # candidate ranking yields a single "NoIntervention" winner trivially.
        return np.zeros(outcomes["Y_anomaly_indicator"].shape[0])
    raise KeyError(arch)
    raise KeyError(cfg["archetype"])


# ---------------------------------------------------------------------------
# Oracle and validators
# ---------------------------------------------------------------------------

def _static_oracle_score(
    cfg: Dict[str, Any],
    *,
    n_oracle: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Score every candidate intervention with CRN: share hidden draws."""
    candidates = _static_candidate_interventions(cfg)
    hidden = _static_sample_hidden(cfg, n_oracle, seed=seed)
    scores: List[Dict[str, Any]] = []
    for i, iv in enumerate(candidates):
        outcomes = _static_apply(cfg, hidden, iv, seed=seed + 7919 * (i + 1))
        utility = _static_utility_from_outcomes(cfg, outcomes)
        scores.append({
            "intervention": _static_neutralize_intervention(iv),
            "intervention_key": _static_intervention_key(iv),
            "expected_utility": _mean(utility),
            "utility_standard_error": _se(utility),
        })
    return scores


def _static_oracle_observational_correlations(
    cfg: Dict[str, Any],
    *,
    n: int,
    seed: int,
) -> Dict[str, Any]:
    """Diagnostics on the observational distribution for validators."""
    hidden = _static_sample_hidden(cfg, n, seed=seed)
    assignments = _static_assignment(cfg, hidden, seed=seed + 17)
    template = cfg["template"]
    names = template["names"]

    n_units = int(next(v.shape[0] for v in hidden.values() if isinstance(v, np.ndarray)))
    arch = cfg["archetype"]
    # Mapping from each archetype to its latent target field returned by _static_apply.
    _LATENT_TARGET_KEY = {
        "hidden_cause": "Y_target_severity",
        "confounded_action": "Y_target_outcome",
        "mechanism_chain": "Y_target_throughput",
        "negative_control": "Y_target_severity",
        "hidden_subtype": "Y_target_outcome",
        "anomaly_discovery": "Y_anomaly_indicator",
        "latent_regime_discovery": "Y_target_outcome",
    }
    primary_target_key = _LATENT_TARGET_KEY[arch]

    # Build per-unit outcomes and per-unit primary-obs by stratification on
    # assignment tuple. Each stratum shares the same do(.) so a vectorized
    # apply gives consistent results.
    keys = []
    for i in range(n_units):
        unit_iv = {var: assignments[var][i] for var in assignments}
        keys.append(tuple(sorted(unit_iv.items())))
    unique_keys = list(dict.fromkeys(keys))
    primary_array = np.zeros(n_units, dtype=float)
    # For obs correlations of proxies vs target we also need the *observed*
    # primary measurement (noisy), not the latent outcome.
    primary_obs_array = np.zeros(n_units, dtype=float)
    proxy_obs: Dict[str, np.ndarray] = {}
    if cfg["archetype"] == "hidden_cause":
        proxy_names = [
            ("latent_driver_proxy", names["latent_driver_proxy"]),
            ("decoy_proxy_a", names["decoy_proxy_a"]),
            ("decoy_proxy_b", names["decoy_proxy_b"]),
        ]
        for slot, _ in proxy_names:
            proxy_obs[slot] = np.zeros(n_units, dtype=float)

    # Pick the visible primary-observation field per archetype.
    _PRIMARY_OBS_KEY = {
        "hidden_cause": names.get("primary_target_obs"),
        "confounded_action": names.get("primary_target_obs"),
        "mechanism_chain": names.get("final_outcome_obs"),
        "negative_control": names.get("primary_outcome_obs"),
        "hidden_subtype": names.get("target_outcome_obs"),
        "anomaly_discovery": None,  # no causal outcome
        "latent_regime_discovery": names.get("target_outcome_obs"),
    }
    primary_obs_name = _PRIMARY_OBS_KEY[arch]

    for idx_key, k in enumerate(unique_keys):
        mask = np.array([kk == k for kk in keys], dtype=bool)
        iv_dict = dict(k)
        sub_hidden = {kk: vv[mask] if isinstance(vv, np.ndarray) else vv for kk, vv in hidden.items()}
        sub_out = _static_apply(cfg, sub_hidden, iv_dict, seed=seed + 31 * (idx_key + 1))
        primary_array[mask] = sub_out[primary_target_key]
        if primary_obs_name is None:
            continue
        if arch == "hidden_cause":
            wanted = [primary_obs_name] + [v for _, v in proxy_names]
            sub_obs = _static_observe(cfg, sub_hidden, sub_out, wanted, seed=seed + 41 * (idx_key + 1))
            primary_obs_array[mask] = sub_obs[primary_obs_name]
            for slot, var in proxy_names:
                proxy_obs[slot][mask] = sub_obs[var]
        else:
            sub_obs = _static_observe(cfg, sub_hidden, sub_out, [primary_obs_name], seed=seed + 41 * (idx_key + 1))
            primary_obs_array[mask] = sub_obs[primary_obs_name]

    diagnostics: Dict[str, Any] = {
        "observational_n": int(n_units),
        "primary_outcome_mean": _mean(primary_array),
        "primary_obs_mean": _mean(primary_obs_array),
    }

    if cfg["archetype"] == "hidden_cause":
        # Proxy <-> primary observed correlations (this is what an analyst
        # would actually see; validators rely on this).
        diagnostics["proxy_target_corr_obs"] = {
            slot: float(_static_corr(arr, primary_obs_array))
            for slot, arr in proxy_obs.items()
        }
        # Hidden-state checks (not agent-visible, just for audit).
        diagnostics["obs_corr_decoy_state_a_target_latent"] = float(
            _static_corr(hidden["DecoyState_A"], primary_array)
        )
        diagnostics["obs_corr_burden_target_latent"] = float(
            _static_corr(hidden["LatentBurden"], primary_array)
        )
        # Apparent intervention effects under observational data (naive).
        apparent_effects = {}
        for knob_key in [names["decoy_knob_a"], names["decoy_knob_b"], names["weak_knob"], names["true_lever_knob"]]:
            assigns = assignments[knob_key]
            mask_on = assigns != "off"
            if mask_on.sum() > 30 and (~mask_on).sum() > 30:
                apparent_effects[knob_key] = float(
                    _mean(primary_array[mask_on]) - _mean(primary_array[~mask_on])
                )
        diagnostics["observational_apparent_effects_on_primary"] = apparent_effects
    elif cfg["archetype"] == "confounded_action":
        assigns = assignments[names["treatment_knob"]]
        mask_high = assigns == "high"
        mask_off = assigns == "off"
        diagnostics["obs_mean_outcome_high"] = float(_mean(primary_array[mask_high])) if mask_high.sum() > 30 else None
        diagnostics["obs_mean_outcome_off"] = float(_mean(primary_array[mask_off])) if mask_off.sum() > 30 else None
        if diagnostics["obs_mean_outcome_high"] is not None and diagnostics["obs_mean_outcome_off"] is not None:
            diagnostics["obs_sign_high_minus_off"] = float(
                diagnostics["obs_mean_outcome_high"] - diagnostics["obs_mean_outcome_off"]
            )

    elif cfg["archetype"] == "negative_control":
        apparent_effects = {}
        for knob_key in [names["trendy_knob"], names["conventional_knob"], names["research_backed_knob"]]:
            assigns = assignments[knob_key]
            mask_on = assigns != "off"
            if mask_on.sum() > 30 and (~mask_on).sum() > 30:
                apparent_effects[knob_key] = float(
                    _mean(primary_array[mask_on]) - _mean(primary_array[~mask_on])
                )
        diagnostics["observational_apparent_effects_on_primary"] = apparent_effects

    elif cfg["archetype"] == "latent_regime_discovery":
        h = hidden
        diagnostics["latent_regime_rate_high"] = float(np.mean(h["LatentRegime"]))
        diagnostics["latent_axis_mean_low"] = float(np.mean(h["LatentRegimeAxis"][h["LatentRegime"] == 0]))
        diagnostics["latent_axis_mean_high"] = float(np.mean(h["LatentRegimeAxis"][h["LatentRegime"] == 1]))
        diagnostics["latent_axis_gap"] = (
            diagnostics["latent_axis_mean_high"] - diagnostics["latent_axis_mean_low"]
        )

    return diagnostics


def _static_knob_default(cfg: Dict[str, Any], knob_name: str) -> str:
    """Default ("baseline") value for a knob. All current archetypes use 'off'."""
    return "off"


def _static_primary_obs_name(cfg: Dict[str, Any]) -> Optional[str]:
    """Per-archetype primary observation field name used by the recoverability
    band's naive baselines."""
    arch = cfg["archetype"]
    names = cfg["template"]["names"]
    return {
        "hidden_cause": names.get("primary_target_obs"),
        "confounded_action": names.get("primary_target_obs"),
        "mechanism_chain": names.get("final_outcome_obs"),
        "negative_control": names.get("primary_outcome_obs"),
        "hidden_subtype": names.get("target_outcome_obs"),
        # anomaly_discovery has no causal primary outcome; recoverability is skipped.
        "anomaly_discovery": None,
        "latent_regime_discovery": names.get("target_outcome_obs"),
    }.get(arch)


def _static_observational_outcomes(
    cfg: Dict[str, Any],
    n_obs: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Sample n_obs units under the current-practice assignment policy and
    return (primary_obs_array, assignments_per_knob).
    Builds the primary target observation per-unit by stratifying on
    assignment tuple (each stratum shares the same do(.)).
    """
    primary_obs = _static_primary_obs_name(cfg)
    hidden = _static_sample_hidden(cfg, n_obs, seed=seed)
    assigns = _static_assignment(cfg, hidden, seed=seed + 17)
    per_unit_key = []
    for i in range(n_obs):
        per_unit_key.append(tuple(sorted({k: str(assigns[k][i]) for k in assigns}.items())))
    primary_array = np.zeros(n_obs, dtype=float)
    if primary_obs is None:
        return primary_array, assigns
    seen = list(dict.fromkeys(per_unit_key))
    for idx, k in enumerate(seen):
        mask = np.array([pk == k for pk in per_unit_key], dtype=bool)
        sub_hidden = {kk: vv[mask] if isinstance(vv, np.ndarray) else vv for kk, vv in hidden.items()}
        sub_out = _static_apply(cfg, sub_hidden, dict(k), seed=seed + 31 * (idx + 1))
        sub_obs = _static_observe(cfg, sub_hidden, sub_out, [primary_obs], seed=seed + 41 * (idx + 1))
        primary_array[mask] = sub_obs[primary_obs]
    return primary_array, assigns


def _static_obs_naive_pick(
    cfg: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    n_obs: int,
    seed: int,
    min_matched_units: int = 3,
) -> Optional[Dict[str, Any]]:
    """Naive analyst with `n_obs` observational units. For each candidate
    intervention, look at units whose assignment matches the candidate
    (knobs in the candidate match; knobs not in the candidate match the
    archetype default). Pick the intervention with the best apparent
    primary-obs mean. Returns None if no candidate had enough matched units.
    """
    higher_better = _static_primary_target_higher_is_better(cfg)
    primary_array, assigns = _static_observational_outcomes(cfg, n_obs, seed=seed)
    best_iv: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for iv in candidates:
        mask = np.ones(n_obs, dtype=bool)
        for k in assigns:
            target_val = iv.get(k, _static_knob_default(cfg, k))
            mask &= (assigns[k] == target_val)
        if int(mask.sum()) < min_matched_units:
            continue
        score = float(np.mean(primary_array[mask]))
        if best_score is None:
            best_score = score
            best_iv = iv
        else:
            improvement = (score > best_score) if higher_better else (score < best_score)
            if improvement:
                best_score = score
                best_iv = iv
    return best_iv


def _static_intv_naive_pick(
    cfg: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    n_per_intervention: int,
    seed: int,
) -> Dict[str, Any]:
    """Naive analyst with `n_per_intervention` units under uniform do(iv) for
    each candidate. Measures the primary observation only. Picks the
    intervention with the best mean primary obs."""
    primary_obs = _static_primary_obs_name(cfg)
    if primary_obs is None:
        # Anomaly archetype: there is no causal primary; return NoIntervention.
        return {}
    higher_better = _static_primary_target_higher_is_better(cfg)
    best_iv: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for i, iv in enumerate(candidates):
        hidden = _static_sample_hidden(cfg, n_per_intervention, seed=seed + 31 * (i + 1))
        outcomes = _static_apply(cfg, hidden, iv, seed=seed + 53 * (i + 1))
        obs = _static_observe(cfg, hidden, outcomes, [primary_obs], seed=seed + 73 * (i + 1))
        score = float(np.mean(obs[primary_obs]))
        if best_score is None:
            best_score = score
            best_iv = iv
        else:
            improvement = (score > best_score) if higher_better else (score < best_score)
            if improvement:
                best_score = score
                best_iv = iv
    assert best_iv is not None
    return best_iv


def _static_recoverability_band(
    cfg: Dict[str, Any],
    *,
    gold_intervention: Dict[str, Any],
    seed_base: int,
) -> Dict[str, Any]:
    """Two naive baselines define the difficulty band.

    Small budget: an analyst with `STATIC_RECOVER_SMALL_N` observational units
    (current-practice assignment) computes apparent group-mean outcomes per
    candidate intervention. This should fail because observational data is
    confounded.

    Medium budget: an analyst who knows to intervene runs uniform do(iv) with
    `STATIC_RECOVER_MEDIUM_N` units per candidate and measures the primary
    proxy. This should succeed.
    """
    candidates = _static_candidate_interventions(cfg)
    gold_key = _static_intervention_key(gold_intervention)

    hits_small = 0
    obs_picks: List[str] = []
    for s in range(STATIC_RECOVER_N_SEEDS):
        pick = _static_obs_naive_pick(
            cfg,
            candidates,
            n_obs=STATIC_RECOVER_SMALL_N,
            seed=seed_base + 101 * (s + 1),
        )
        pick_key = _static_intervention_key(pick) if pick is not None else "no_match"
        obs_picks.append(pick_key)
        if pick is not None and pick_key == gold_key:
            hits_small += 1
    small_rate = hits_small / STATIC_RECOVER_N_SEEDS

    hits_med = 0
    intv_picks: List[str] = []
    for s in range(STATIC_RECOVER_N_SEEDS):
        pick = _static_intv_naive_pick(
            cfg,
            candidates,
            n_per_intervention=STATIC_RECOVER_MEDIUM_N,
            seed=seed_base + 211 * (s + 1),
        )
        pick_key = _static_intervention_key(pick)
        intv_picks.append(pick_key)
        if pick_key == gold_key:
            hits_med += 1
    medium_rate = hits_med / STATIC_RECOVER_N_SEEDS

    # Most common wrong pick under small-budget obs (for diagnostics).
    pick_counter: Dict[str, int] = {}
    for k in obs_picks:
        pick_counter[k] = pick_counter.get(k, 0) + 1
    sorted_picks = sorted(pick_counter.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "small_budget_mode": "observational",
        "small_budget_n_obs": STATIC_RECOVER_SMALL_N,
        "medium_budget_mode": "interventional_per_candidate",
        "medium_budget_n_per_intervention": STATIC_RECOVER_MEDIUM_N,
        "n_seeds": STATIC_RECOVER_N_SEEDS,
        "small_budget_hit_rate": small_rate,
        "medium_budget_hit_rate": medium_rate,
        "small_budget_top_picks": sorted_picks[:3],
        "gold_intervention_key": gold_key,
    }


def _static_validate(
    cfg: Dict[str, Any],
    scores: List[Dict[str, Any]],
    observational_diag: Dict[str, Any],
    recoverability: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], float]:
    """Run archetype-specific validators. Returns (checks, gold, runner_up, margin)."""
    arch = cfg["archetype"]
    names = cfg["template"]["names"]

    ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
    gold = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
    margin = float(gold["expected_utility"] - runner_up["expected_utility"])

    checks: List[Dict[str, Any]] = []

    # Universal margin check applies to archetypes whose answer is a single
    # static intervention. negative_control gold = {} with intentionally
    # near-zero margin; anomaly_discovery has no causal intervention; and
    # hidden_subtype is judged by conditional_beats_static below.
    if arch not in ("negative_control", "anomaly_discovery", "hidden_subtype", "latent_regime_discovery"):
        checks.append(_static_check(
            "gold_margin",
            margin >= STATIC_MIN_GOLD_MARGIN,
            margin,
            f">= {STATIC_MIN_GOLD_MARGIN}",
            "Gold intervention beats runner-up by margin under uniform do().",
        ))

    # Recoverability band applies to archetypes with a meaningful do()
    # comparison. Skipped for anomaly_discovery, negative_control, and
    # hidden_subtype: hidden_subtype's gold is a conditional policy, so a
    # static-intervention recoverability band is not the right difficulty test.
    if arch not in ("anomaly_discovery", "negative_control", "hidden_subtype", "latent_regime_discovery"):
        checks.append(_static_check(
            "recoverability_small_budget",
            recoverability["small_budget_hit_rate"] <= STATIC_RECOVER_SMALL_MAX,
            recoverability["small_budget_hit_rate"],
            f"<= {STATIC_RECOVER_SMALL_MAX}",
            "Naive small-budget baseline rarely recovers gold.",
        ))
        checks.append(_static_check(
            "recoverability_medium_budget",
            recoverability["medium_budget_hit_rate"] >= STATIC_RECOVER_MEDIUM_MIN,
            recoverability["medium_budget_hit_rate"],
            f">= {STATIC_RECOVER_MEDIUM_MIN}",
            "Naive medium-budget baseline reliably recovers gold.",
        ))

    if arch == "hidden_cause":
        # Gold must be the true_lever intervention.
        true_lever_key = _static_intervention_key({names["true_lever_knob"]: "on"})
        checks.append(_static_check(
            "true_lever_is_gold",
            gold["intervention_key"] == true_lever_key,
            gold["intervention_key"],
            true_lever_key,
            "True lever is the oracle-best intervention under do(.).",
        ))
        # Continuous LatentBurden: enough of the population must sit in the
        # active band of the soft threshold or the lever has nothing to do.
        n_hidden_check = 5000
        h = _static_sample_hidden(cfg, n_hidden_check, seed=cfg["seed"] + 333)
        active_band_rate = float(
            np.mean(h["LatentBurden"] >= cfg["parameters"]["burden_threshold"])
        )
        checks.append(_static_check(
            "latent_burden_active_band_rate",
            0.30 <= active_band_rate <= 0.85,
            active_band_rate,
            "0.30 <= rate <= 0.85",
            "Share of units above the burden activation threshold (room for "
            "the true lever to act on).",
        ))
        # Decoy proxies must be observationally correlated with the target
        # (i.e. look tempting) but not too strongly.
        diag_corrs = observational_diag.get("proxy_target_corr_obs", {})
        true_pc = diag_corrs.get("latent_driver_proxy")
        decoy_a_pc = diag_corrs.get("decoy_proxy_a")
        decoy_b_pc = diag_corrs.get("decoy_proxy_b")
        if true_pc is not None and decoy_a_pc is not None and decoy_b_pc is not None:
            checks.append(_static_check(
                "true_proxy_calibrated",
                0.30 <= abs(true_pc) <= 0.75,
                true_pc,
                "0.30 <= |corr| <= 0.75",
                "Latent-driver proxy is informative but noisy: needs to be "
                "neither perfect nor useless.",
            ))
            max_decoy = max(abs(decoy_a_pc), abs(decoy_b_pc))
            checks.append(_static_check(
                "decoys_tempting_obs",
                max_decoy >= 0.20,
                {"decoy_a": decoy_a_pc, "decoy_b": decoy_b_pc},
                "max |decoy corr| >= 0.20",
                "At least one decoy proxy correlates with the target in obs "
                "data, so naive analysis is genuinely tempted.",
            ))

    elif arch == "confounded_action":
        # Sign reversal: under obs, treatment=high looks worse than off.
        sign = observational_diag.get("obs_sign_high_minus_off")
        if sign is not None:
            checks.append(_static_check(
                "observational_sign_negative",
                sign < -2.0,
                sign,
                "< -2.0",
                "Under observational data, high treatment is associated with worse outcomes.",
            ))
        # Gold should set treatment_knob = "high".
        gold_iv = gold["intervention"]
        checks.append(_static_check(
            "gold_uses_high_treatment",
            gold_iv.get(names["treatment_knob"]) == "high",
            gold_iv.get(names["treatment_knob"]),
            "high",
            "Oracle-best intervention uses high treatment level.",
        ))

    elif arch == "mechanism_chain":
        bottleneck = int(cfg["parameters"]["bottleneck_stage"])
        bottleneck_key = _static_intervention_key({names[f"stage_{bottleneck}_knob"]: "on"})
        checks.append(_static_check(
            "bottleneck_intervention_is_gold",
            gold["intervention_key"] == bottleneck_key,
            gold["intervention_key"],
            bottleneck_key,
            "Oracle-best intervention is on the bottleneck stage.",
        ))
        no_iv_score = next(
            (s["expected_utility"] for s in scores if s["intervention_key"] == "NoIntervention"),
            None,
        )
        if no_iv_score is not None:
            non_bottleneck_effects = [
                s["expected_utility"] - no_iv_score
                for s in scores
                if s["intervention_key"] not in (bottleneck_key, "NoIntervention")
            ]
            max_nb = max(non_bottleneck_effects) if non_bottleneck_effects else 0.0
            checks.append(_static_check(
                "non_bottleneck_interventions_weak",
                max_nb <= STATIC_MIN_GOLD_MARGIN,
                max_nb,
                f"<= {STATIC_MIN_GOLD_MARGIN}",
                "Non-bottleneck stage interventions yield small target gains.",
            ))

    elif arch == "negative_control":
        checks.append(_static_check(
            "gold_is_no_intervention",
            gold["intervention_key"] == "NoIntervention",
            gold["intervention_key"],
            "NoIntervention",
            "Oracle-best is NoIntervention (no offered knob actually works).",
        ))
        no_iv_score = next(
            (s["expected_utility"] for s in scores if s["intervention_key"] == "NoIntervention"),
            None,
        )
        if no_iv_score is not None:
            magnitudes = [
                abs(s["expected_utility"] - no_iv_score)
                for s in scores
                if s["intervention_key"] != "NoIntervention"
            ]
            max_mag = max(magnitudes) if magnitudes else 0.0
            checks.append(_static_check(
                "no_intervention_has_meaningful_effect",
                max_mag < STATIC_MIN_GOLD_MARGIN,
                max_mag,
                f"< {STATIC_MIN_GOLD_MARGIN}",
                "Every non-baseline intervention has |do-effect| below the gold-margin threshold.",
            ))
        apparent = observational_diag.get("observational_apparent_effects_on_primary", {}) or {}
        if apparent:
            max_app = max(abs(v) for v in apparent.values())
            checks.append(_static_check(
                "observational_effect_strong",
                max_app >= 3.0,
                max_app,
                ">= 3.0",
                "At least one intervention LOOKS materially beneficial in obs data.",
            ))
        checks.append(_static_check(
            "small_naive_picks_non_baseline",
            recoverability["small_budget_hit_rate"] <= 0.50,
            recoverability["small_budget_hit_rate"],
            "<= 0.50",
            "Small-budget observational naive rarely picks NoIntervention.",
        ))

    elif arch == "hidden_subtype":
        p = cfg["parameters"]
        a0, a1 = p["treatment_a_effect_subtype_0"], p["treatment_a_effect_subtype_1"]
        c0, c1 = p["treatment_c_effect_subtype_0"], p["treatment_c_effect_subtype_1"]
        checks.append(_static_check(
            "subtype_effects_differ_A",
            abs(a0 - a1) >= 8.0,
            {"A_sub0": a0, "A_sub1": a1},
            ">= 8 unit gap",
            "Treatment A's effect differs by subtype enough to warrant a conditional policy.",
        ))
        checks.append(_static_check(
            "subtype_effects_differ_C",
            abs(c0 - c1) >= 8.0,
            {"C_sub0": c0, "C_sub1": c1},
            ">= 8 unit gap",
            "Treatment C's effect differs by subtype enough to warrant a conditional policy.",
        ))
        treatment_keys = (
            _static_intervention_key({names["treatment_a_knob"]: "on"}),
            _static_intervention_key({names["treatment_b_knob"]: "on"}),
            _static_intervention_key({names["treatment_c_knob"]: "on"}),
        )
        checks.append(_static_check(
            "gold_is_a_treatment",
            gold["intervention_key"] in treatment_keys,
            gold["intervention_key"],
            "one of treatment_{a,b,c}=on",
            "Static-best intervention is one of the three treatments.",
        ))

    elif arch == "anomaly_discovery":
        h = _static_sample_hidden(cfg, 5000, seed=cfg["seed"] + 333)
        p_anom = float(np.mean(h["IsAnomaly"]))
        checks.append(_static_check(
            "anomaly_prevalence_in_band",
            0.03 <= p_anom <= 0.13,
            p_anom,
            "0.03 <= p_anom <= 0.13",
            "Observed anomaly prevalence within design range.",
        ))
        anom_mask = h["IsAnomaly"] == 1
        norm_mask = h["IsAnomaly"] == 0
        if int(anom_mask.sum()) > 0 and int(norm_mask.sum()) > 0:
            shift_a = float(_mean(h["Feature_A_state"][anom_mask]) - _mean(h["Feature_A_state"][norm_mask]))
            shift_b = float(_mean(h["Feature_B_state"][anom_mask]) - _mean(h["Feature_B_state"][norm_mask]))
        else:
            shift_a = shift_b = 0.0
        checks.append(_static_check(
            "feature_separation_a",
            abs(shift_a) >= 14.0,
            shift_a,
            "|shift| >= 14",
            "Feature A separates anomalous from normal units.",
        ))
        checks.append(_static_check(
            "feature_separation_b",
            abs(shift_b) >= 14.0,
            shift_b,
            "|shift| >= 14",
            "Feature B separates anomalous from normal units.",
        ))

    return checks, gold, runner_up, margin


def _static_all_pass(checks: Iterable[Dict[str, Any]]) -> bool:
    return all(bool(c.get("passed")) for c in checks)


# ---------------------------------------------------------------------------
# World assembly
# ---------------------------------------------------------------------------

def _static_visible_block_hidden_cause(
    cfg: Dict[str, Any],
    template: Dict[str, Any],
    question: str,
    answer_schema: str,
    max_knobs: int,
    *,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    knob_descs = template["knob_descriptions"]
    observed_variables = [
        {
            "name": names["primary_target_obs"],
            "description": descs[names["primary_target_obs"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False},
        },
        {
            "name": names["secondary_target_obs"],
            "description": descs[names["secondary_target_obs"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False},
        },
        {
            "name": names["latent_driver_proxy"],
            "description": descs[names["latent_driver_proxy"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None},
        },
        {
            "name": names["decoy_proxy_a"],
            "description": descs[names["decoy_proxy_a"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None},
        },
        {
            "name": names["decoy_proxy_b"],
            "description": descs[names["decoy_proxy_b"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None},
        },
        {
            "name": names["tertiary_obs"],
            "description": descs[names["tertiary_obs"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True},
        },
    ]
    intervenable_variables = [
        {
            "name": names["decoy_knob_a"],
            "values": ["off", "on"],
            "default": "off",
            "description": knob_descs[names["decoy_knob_a"]],
        },
        {
            "name": names["decoy_knob_b"],
            "values": ["off", "on"],
            "default": "off",
            "description": knob_descs[names["decoy_knob_b"]],
        },
        {
            "name": names["weak_knob"],
            "values": ["off", "low", "high"],
            "default": "off",
            "description": knob_descs[names["weak_knob"]],
        },
        {
            "name": names["true_lever_knob"],
            "values": ["off", "on"],
            "default": "off",
            "description": knob_descs[names["true_lever_knob"]],
        },
    ]
    obs_seed = int(cfg["seed"]) * 31 + 1
    intv_seed = int(cfg["seed"]) * 31 + 2
    observed_variables = _static_shuffled_copy(observed_variables, obs_seed)
    intervenable_variables = _static_balanced_anchor_shuffle(
        intervenable_variables,
        anchor_name=names["true_lever_knob"],
        target_position=target_gold_position,
        shuffle_seed=intv_seed,
    )
    return _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs
    )


def _static_visible_block_confounded(
    cfg: Dict[str, Any],
    template: Dict[str, Any],
    question: str,
    answer_schema: str,
    max_knobs: int,
    *,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    knob_descs = template["knob_descriptions"]
    observed_variables = [
        {
            "name": names["primary_target_obs"],
            "description": descs[names["primary_target_obs"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True},
        },
        {
            "name": names["severity_proxy_a"],
            "description": descs[names["severity_proxy_a"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False},
        },
        {
            "name": names["severity_proxy_b"],
            "description": descs[names["severity_proxy_b"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False},
        },
        {
            "name": names["secondary_target_obs"],
            "description": descs[names["secondary_target_obs"]],
            "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True},
        },
        {
            "name": names["assignment_record"],
            "description": descs[names["assignment_record"]],
            "scale": {"type": "categorical", "values": ["off", "low", "high"]},
        },
    ]
    intervenable_variables = [
        {
            "name": names["treatment_knob"],
            "values": ["off", "low", "high"],
            "default": "off",
            "description": knob_descs[names["treatment_knob"]],
        },
        {
            "name": names["support_knob"],
            "values": ["off", "on"],
            "default": "off",
            "description": knob_descs[names["support_knob"]],
        },
    ]
    obs_seed = int(cfg["seed"]) * 31 + 1
    intv_seed = int(cfg["seed"]) * 31 + 2
    observed_variables = _static_shuffled_copy(observed_variables, obs_seed)
    intervenable_variables = _static_balanced_anchor_shuffle(
        intervenable_variables,
        anchor_name=names["treatment_knob"],
        target_position=target_gold_position,
        shuffle_seed=intv_seed,
    )
    return _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs
    )


def _static_visible_block(
    cfg: Dict[str, Any],
    template: Dict[str, Any],
    observed_variables: List[Dict[str, Any]],
    intervenable_variables: List[Dict[str, Any]],
    question: str,
    answer_schema: str,
    max_knobs: int,
) -> Dict[str, Any]:
    return {
        "story": template["setting"],
        "observed_variables": observed_variables,
        "intervenable_variables": intervenable_variables,
        "allowed_query_modes": [
            "observational_sample",
            "interventional_sample",
            "inspect_unit",
        ],
        "allowed_measurements": [v["name"] for v in observed_variables],
        "experiment_budget": {
            "sample_accounting": "cells",
            "max_total_samples": STATIC_MAX_TOTAL_SAMPLES,
            "max_samples_per_query": STATIC_MAX_SAMPLES_PER_QUERY,
            "max_units_per_query": STATIC_MAX_UNITS_PER_QUERY,
            "max_measurements_per_query": STATIC_MAX_MEASUREMENTS_PER_QUERY,
            "max_queries": STATIC_MAX_QUERIES,
            "counted_unit": "returned dataframe cells = rows * columns",
        },
        "discovery_protocol": {
            "task_style": "budgeted_iterative_scientific_discovery",
            "objective": (
                "Decide what to measure and what to intervene on, request small "
                "samples under the budget, infer the latent mechanism, and "
                "submit a structured answer."
            ),
            "recommended_workflow": [
                "Start with a small observational sample to inspect the natural distribution.",
                "Form a hypothesis about what may drive the outcome and which proxy might inform it.",
                "Run interventional samples on knobs your hypothesis predicts will move the target.",
                "Cross-check on a secondary outcome proxy before committing.",
            ],
            "anti_bruteforce_note": (
                "A naive sweep over every intervention with every measurement at "
                "max units will exceed the per-query column cap or the total cell "
                "budget. Choose what to measure."
            ),
        },
        "question": question,
        "answer_schema": answer_schema,
        "max_intervention_knobs": max_knobs,
    }


def _static_assemble_world(
    *,
    cfg: Dict[str, Any],
    template: Dict[str, Any],
    scores: List[Dict[str, Any]],
    gold: Dict[str, Any],
    runner_up: Dict[str, Any],
    margin: float,
    checks: List[Dict[str, Any]],
    observational_diag: Dict[str, Any],
    recoverability: Dict[str, Any],
    seed: int,
    oracle_n: int,
    question: str,
    answer_schema: str,
    max_knobs: int,
    objective: str,
    visible_block: Dict[str, Any],
) -> Dict[str, Any]:
    accepted = _static_all_pass(checks)
    world_id = f"rpg_static_{cfg['archetype']}_{_static_safe_id(template['subdomain'])}_seed{seed}"
    answer_value = {
        "intervention": gold["intervention"],
        "hypothesis": None,
    }
    world = {
        "schema_version": SCHEMA_VERSION_STATIC,
        "meta": {
            "benchmark": BENCHMARK_NAME_STATIC,
            "world_id": world_id,
            "archetype": cfg["archetype"],
            "topic": template["topic"],
            "subdomain": template["subdomain"],
            "seed": seed,
            "mixture_weight": cfg["mixture_weight"],
            "oracle_n_units": oracle_n,
            "n_candidate_interventions": len(scores),
            "n_questions": 1,
            "generator": "world_gen_rpg.py (static section)",
        },
        "story": template["setting"],
        "variables": [],
        "non_intervenable_variables": [],
        "edges": [],
        "cpds": [],
        "visible": visible_block,
        "hidden": {
            "latent_variables_by_archetype": cfg["archetype"],
            "unit_prior": {
                "type": "structured_uniform_mixture",
                "uniform_weight": cfg["mixture_weight"],
                "structured_weight": 1.0 - cfg["mixture_weight"],
            },
            "simulator_config": {
                "archetype": cfg["archetype"],
                "schema_version": SCHEMA_VERSION_STATIC,
                "world_seed": seed,
                "parameters": _jsonify(cfg["parameters"]),
                "template": template,
                "mixture_weight": cfg["mixture_weight"],
            },
            "diagnostics": {
                "observational": _jsonify(observational_diag),
                "recoverability_band": _jsonify(recoverability),
            },
            "visibility_note": (
                "These fields define the simulator and oracle. They are not "
                "agent-visible. Only the `visible` block should be shown to "
                "scientist agents."
            ),
        },
        "oracle": {
            "objective": objective,
            "action_scores": _jsonify(scores),
            "gold_answer": {
                "intervention": gold["intervention"],
                "expected_utility": gold["expected_utility"],
                "intervention_key": gold["intervention_key"],
            },
            "runner_up": {
                "intervention": runner_up["intervention"],
                "expected_utility": runner_up["expected_utility"],
            },
            "gold_margin": float(margin),
            "oracle_tolerance": float(STATIC_ORACLE_TOLERANCE_FRACTION * margin),
            "oracle_n_units": oracle_n,
            "oracle_standard_error": float(gold["utility_standard_error"]),
        },
        "validators": {
            "accepted": accepted,
            "signature_checks": _jsonify(checks),
        },
        "questions": [
            {
                "id": 0,
                "question_type": f"rpg_{cfg['archetype']}",
                "difficulty": "hard",
                "answer_schema": answer_schema,
                "question": question,
                "answer": _jsonify(answer_value),
                "metadata": {
                    "archetype": cfg["archetype"],
                    "max_intervention_knobs": max_knobs,
                    "gold": _jsonify({
                        "intervention": gold["intervention"],
                        "expected_utility": gold["expected_utility"],
                        "runner_up": runner_up["intervention"],
                        "runner_up_expected_utility": runner_up["expected_utility"],
                        "gold_margin": margin,
                        "oracle_tolerance": STATIC_ORACLE_TOLERANCE_FRACTION * margin,
                    }),
                },
            }
        ],
    }
    return world


# ---------------------------------------------------------------------------
# Per-archetype builders
# ---------------------------------------------------------------------------

def _static_build_hidden_cause(
    template: Dict[str, Any],
    *,
    seed: int,
    mixture_weight: float,
    oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    parameters = _static_default_params("hidden_cause", rng)
    cfg = {
        "archetype": "hidden_cause",
        "template": template,
        "parameters": parameters,
        "mixture_weight": mixture_weight,
        "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
    gold_intervention = ranked[0]["intervention"]
    recoverability = _static_recoverability_band(cfg, gold_intervention=gold_intervention, seed_base=seed + 9001)
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)

    names = template["names"]
    question = (
        f"A population shows persistently high {names['primary_target_obs']} "
        f"under current practice. Using the available query modes and budget, "
        f"determine which single intervention, applied uniformly to a freshly "
        f"sampled population, most improves the underlying condition reflected "
        f"by {names['primary_target_obs']} and the related target measurements. "
        f"Submit one intervention as a `do(.)` dict over the intervenable "
        f"variables, plus a one-paragraph hypothesis explaining the underlying "
        f"mechanism."
    )
    objective = (
        f"Minimize the expected underlying target severity reflected by "
        f"{names['primary_target_obs']} and related target measurements on "
        f"freshly drawn units under the chosen do(.) intervention."
    )
    visible_block = _static_visible_block_hidden_cause(
        cfg, template, question, "intervention_with_hypothesis", max_knobs=1,
        target_gold_position=target_gold_position,
    )
    return _static_assemble_world(
        cfg=cfg,
        template=template,
        scores=scores,
        gold=gold,
        runner_up=runner_up,
        margin=margin,
        checks=checks,
        observational_diag=observational_diag,
        recoverability=recoverability,
        seed=seed,
        oracle_n=oracle_n,
        question=question,
        answer_schema="intervention_with_hypothesis",
        max_knobs=1,
        objective=objective,
        visible_block=visible_block,
    )


def _static_build_confounded_action(
    template: Dict[str, Any],
    *,
    seed: int,
    mixture_weight: float,
    oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    parameters = _static_default_params("confounded_action", rng)
    cfg = {
        "archetype": "confounded_action",
        "template": template,
        "parameters": parameters,
        "mixture_weight": mixture_weight,
        "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
    gold_intervention = ranked[0]["intervention"]
    recoverability = _static_recoverability_band(cfg, gold_intervention=gold_intervention, seed_base=seed + 9001)
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)

    names = template["names"]
    question = (
        f"In this population, conventional wisdom is that aggressive use of "
        f"{names['treatment_knob']} leads to worse {names['primary_target_obs']}, "
        f"because passive records show worse outcomes among units who received "
        f"high {names['treatment_knob']}. Using the available query modes and "
        f"budget, determine the intervention over `{names['treatment_knob']}` "
        f"and `{names['support_knob']}` that, applied uniformly to a freshly "
        f"sampled population, maximizes {names['primary_target_obs']}. Submit "
        f"the intervention as a `do(.)` dict and a one-paragraph hypothesis "
        f"explaining whether the conventional belief is mistaken."
    )
    objective = (
        f"Maximize expected {names['primary_target_obs']} on freshly drawn "
        f"units under do(.)."
    )
    visible_block = _static_visible_block_confounded(
        cfg, template, question, "intervention_with_hypothesis", max_knobs=2,
        target_gold_position=target_gold_position,
    )
    return _static_assemble_world(
        cfg=cfg,
        template=template,
        scores=scores,
        gold=gold,
        runner_up=runner_up,
        margin=margin,
        checks=checks,
        observational_diag=observational_diag,
        recoverability=recoverability,
        seed=seed,
        oracle_n=oracle_n,
        question=question,
        answer_schema="intervention_with_hypothesis",
        max_knobs=2,
        objective=objective,
        visible_block=visible_block,
    )


# ---------------------------------------------------------------------------
# Visible-block builders + builders for new archetypes
# ---------------------------------------------------------------------------

def _static_visible_block_mechanism_chain(
    cfg: Dict[str, Any], template: Dict[str, Any], question: str,
    answer_schema: str, max_knobs: int, *, target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    knob_descs = template["knob_descriptions"]
    observed_variables = [
        {"name": names["final_outcome_obs"], "description": descs[names["final_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["secondary_outcome_obs"], "description": descs[names["secondary_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["stage_1_proxy"], "description": descs[names["stage_1_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["stage_2_proxy"], "description": descs[names["stage_2_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["stage_3_proxy"], "description": descs[names["stage_3_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
    ]
    intervenable_variables = [
        {"name": names["stage_1_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["stage_1_knob"]]},
        {"name": names["stage_2_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["stage_2_knob"]]},
        {"name": names["stage_3_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["stage_3_knob"]]},
    ]
    observed_variables = _static_shuffled_copy(observed_variables, int(cfg["seed"]) * 31 + 1)
    # No anchor: the bottleneck stage rotates across worlds in the dataset
    # orchestrator (via template/bottleneck rotation), so a pure shuffle is fine.
    intervenable_variables = _static_shuffled_copy(intervenable_variables, int(cfg["seed"]) * 31 + 2)
    return _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs,
    )


def _static_visible_block_negative_control(
    cfg: Dict[str, Any], template: Dict[str, Any], question: str,
    answer_schema: str, max_knobs: int, *, target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    knob_descs = template["knob_descriptions"]
    observed_variables = [
        {"name": names["primary_outcome_obs"], "description": descs[names["primary_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False}},
        {"name": names["secondary_outcome_obs"], "description": descs[names["secondary_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False}},
        {"name": names["wellbeing_proxy"], "description": descs[names["wellbeing_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["engagement_proxy"], "description": descs[names["engagement_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["healthseek_proxy"], "description": descs[names["healthseek_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
    ]
    intervenable_variables = [
        {"name": names["trendy_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["trendy_knob"]]},
        {"name": names["conventional_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["conventional_knob"]]},
        {"name": names["research_backed_knob"], "values": ["off", "low", "high"], "default": "off",
         "description": knob_descs[names["research_backed_knob"]]},
    ]
    observed_variables = _static_shuffled_copy(observed_variables, int(cfg["seed"]) * 31 + 1)
    intervenable_variables = _static_shuffled_copy(intervenable_variables, int(cfg["seed"]) * 31 + 2)
    return _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs,
    )


def _static_visible_block_hidden_subtype(
    cfg: Dict[str, Any], template: Dict[str, Any], question: str,
    answer_schema: str, max_knobs: int, *, target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    knob_descs = template["knob_descriptions"]
    observed_variables = [
        {"name": names["target_outcome_obs"], "description": descs[names["target_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["secondary_outcome_obs"], "description": descs[names["secondary_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["subtype_screen"], "description": descs[names["subtype_screen"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["secondary_subtype_screen"], "description": descs[names["secondary_subtype_screen"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["baseline_risk_proxy"], "description": descs[names["baseline_risk_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False}},
    ]
    intervenable_variables = [
        {"name": names["treatment_a_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["treatment_a_knob"]]},
        {"name": names["treatment_b_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["treatment_b_knob"]]},
        {"name": names["treatment_c_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["treatment_c_knob"]]},
    ]
    observed_variables = _static_shuffled_copy(observed_variables, int(cfg["seed"]) * 31 + 1)
    intervenable_variables = _static_shuffled_copy(intervenable_variables, int(cfg["seed"]) * 31 + 2)
    block = _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs,
    )
    # For conditional_policy answers, advertise candidate branch variables so
    # the agent and runtime know which proxies may be used to split units.
    block["conditional_policy_branch_candidates"] = [names["subtype_screen"], names["secondary_subtype_screen"]]
    block["conditional_policy_branch_threshold_default"] = 50.0
    return block


def _static_visible_block_anomaly_discovery(
    cfg: Dict[str, Any], template: Dict[str, Any], question: str,
    answer_schema: str, max_knobs: int, *, target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    observed_variables = [
        {"name": names["feature_a"], "description": descs[names["feature_a"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["feature_b"], "description": descs[names["feature_b"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["feature_c"], "description": descs[names["feature_c"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["feature_d"], "description": descs[names["feature_d"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["secondary_signal"], "description": descs[names["secondary_signal"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
    ]
    observed_variables = _static_shuffled_copy(observed_variables, int(cfg["seed"]) * 31 + 1)
    intervenable_variables: List[Dict[str, Any]] = []  # no intervenable knobs
    return _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs,
    )


def _static_visible_block_latent_regime(
    cfg: Dict[str, Any], template: Dict[str, Any], question: str,
    answer_schema: str, max_knobs: int, *, target_gold_position: int = 0,
) -> Dict[str, Any]:
    names = template["names"]
    descs = template["measurement_descriptions"]
    knob_descs = template["knob_descriptions"]
    observed_variables = [
        {"name": names["target_outcome_obs"], "description": descs[names["target_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["secondary_outcome_obs"], "description": descs[names["secondary_outcome_obs"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["regime_proxy_a"], "description": descs[names["regime_proxy_a"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["regime_proxy_b"], "description": descs[names["regime_proxy_b"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["baseline_risk_proxy"], "description": descs[names["baseline_risk_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": False}},
        {"name": names["decoy_proxy_a"], "description": descs[names["decoy_proxy_a"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["decoy_proxy_b"], "description": descs[names["decoy_proxy_b"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["decoy_proxy_c"], "description": descs[names["decoy_proxy_c"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": None}},
        {"name": names["relief_proxy"], "description": descs[names["relief_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
        {"name": names["tolerability_proxy"], "description": descs[names["tolerability_proxy"]],
         "scale": {"type": "continuous", "min": 0, "max": 100, "higher_is_better": True}},
    ]
    intervenable_variables = [
        {"name": names["treatment_a_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["treatment_a_knob"]]},
        {"name": names["treatment_b_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["treatment_b_knob"]]},
        {"name": names["treatment_c_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["treatment_c_knob"]]},
        {"name": names["support_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["support_knob"]]},
        {"name": names["decoy_knob_a"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["decoy_knob_a"]]},
        {"name": names["decoy_knob_b"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["decoy_knob_b"]]},
        {"name": names["palliative_knob"], "values": ["off", "low", "high"], "default": "off",
         "description": knob_descs[names["palliative_knob"]]},
        {"name": names["monitoring_knob"], "values": ["off", "on"], "default": "off",
         "description": knob_descs[names["monitoring_knob"]]},
    ]
    observed_variables = _static_shuffled_copy(observed_variables, int(cfg["seed"]) * 31 + 1)
    intervenable_variables = _static_shuffled_copy(intervenable_variables, int(cfg["seed"]) * 31 + 2)
    block = _static_visible_block(
        cfg, template, observed_variables, intervenable_variables, question, answer_schema, max_knobs,
    )
    block["latent_structure_prompt"] = {
        "allowed_n_regimes_range": [1, 3],
        "policy_branch_candidates": [names["regime_proxy_a"], names["regime_proxy_b"]],
        "branch_threshold_default": 50.0,
    }
    return block


# --- per-archetype builders for the new ones ---

def _static_build_mechanism_chain(
    template: Dict[str, Any], *, seed: int, mixture_weight: float, oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    # Rotate bottleneck stage by seed so the dataset is balanced across stages.
    bottleneck = (seed // 101) % 3 + 1
    rng = random.Random(seed)
    parameters = _static_default_params("mechanism_chain", rng, bottleneck_stage=bottleneck)
    cfg = {
        "archetype": "mechanism_chain", "template": template, "parameters": parameters,
        "mixture_weight": mixture_weight, "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
    gold_intervention = ranked[0]["intervention"]
    recoverability = _static_recoverability_band(
        cfg, gold_intervention=gold_intervention, seed_base=seed + 9001,
    )
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)
    names = template["names"]
    question = (
        f"This system has three sequential stages, each measurable as "
        f"{names['stage_1_proxy']}, {names['stage_2_proxy']}, and "
        f"{names['stage_3_proxy']}, with final outcome {names['final_outcome_obs']}. "
        f"Identify which single stage is the bottleneck and propose the "
        f"intervention there. Submit a `do(.)` dict with exactly one of "
        f"`{names['stage_1_knob']}`, `{names['stage_2_knob']}`, "
        f"`{names['stage_3_knob']}` set to 'on', plus a one-paragraph hypothesis "
        f"explaining why the other stages are NOT the right place to intervene."
    )
    objective = f"Maximize expected {names['final_outcome_obs']} on freshly drawn units under do(.)."
    visible_block = _static_visible_block_mechanism_chain(
        cfg, template, question, "intervention_with_hypothesis", max_knobs=1,
        target_gold_position=target_gold_position,
    )
    return _static_assemble_world(
        cfg=cfg, template=template, scores=scores, gold=gold, runner_up=runner_up,
        margin=margin, checks=checks, observational_diag=observational_diag,
        recoverability=recoverability, seed=seed, oracle_n=oracle_n, question=question,
        answer_schema="intervention_with_hypothesis", max_knobs=1, objective=objective,
        visible_block=visible_block,
    )


def _static_build_negative_control(
    template: Dict[str, Any], *, seed: int, mixture_weight: float, oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    parameters = _static_default_params("negative_control", rng)
    cfg = {
        "archetype": "negative_control", "template": template, "parameters": parameters,
        "mixture_weight": mixture_weight, "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    ranked = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)
    gold_intervention = ranked[0]["intervention"]
    recoverability = _static_recoverability_band(
        cfg, gold_intervention=gold_intervention, seed_base=seed + 9001,
    )
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)
    names = template["names"]
    question = (
        f"This population has access to several heavily discussed interventions. "
        f"Determine which intervention should be applied uniformly to a freshly "
        f"sampled population to improve {names['primary_outcome_obs']}. If no "
        f"intervention reliably improves the outcome under controlled assignment, "
        f"return an empty intervention dict `{{}}` and explain in one paragraph "
        f"why the observational pattern is misleading."
    )
    objective = (
        f"Minimize expected {names['primary_outcome_obs']} on freshly drawn units under do(.), "
        f"accepting the empty intervention as the answer when no offered knob "
        f"actually moves the latent target."
    )
    visible_block = _static_visible_block_negative_control(
        cfg, template, question, "intervention_with_hypothesis", max_knobs=1,
        target_gold_position=target_gold_position,
    )
    return _static_assemble_world(
        cfg=cfg, template=template, scores=scores, gold=gold, runner_up=runner_up,
        margin=margin, checks=checks, observational_diag=observational_diag,
        recoverability=recoverability, seed=seed, oracle_n=oracle_n, question=question,
        answer_schema="intervention_with_hypothesis", max_knobs=1, objective=objective,
        visible_block=visible_block,
    )


def _static_build_hidden_subtype(
    template: Dict[str, Any], *, seed: int, mixture_weight: float, oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    parameters = _static_default_params("hidden_subtype", rng)
    cfg = {
        "archetype": "hidden_subtype", "template": template, "parameters": parameters,
        "mixture_weight": mixture_weight, "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    # For the *conditional* gold we score a fixed-threshold policy keyed on the
    # primary SubtypeScreen, comparing (A | screen<50, C | screen>=50) vs the
    # reverse. We pick whichever conditional has higher expected utility.
    names = template["names"]
    branch_var = names["subtype_screen"]
    branch_threshold = 50.0
    policy_AC = {
        "branch_variable": branch_var, "branch_threshold": branch_threshold,
        "if_above": {names["treatment_c_knob"]: "on"},
        "if_below": {names["treatment_a_knob"]: "on"},
    }
    policy_CA = {
        "branch_variable": branch_var, "branch_threshold": branch_threshold,
        "if_above": {names["treatment_a_knob"]: "on"},
        "if_below": {names["treatment_c_knob"]: "on"},
    }
    util_AC = _static_hidden_subtype_score_conditional_policy(
        cfg, policy_AC, n_oracle=min(oracle_n, 15000), seed=seed + 12321,
    )
    util_CA = _static_hidden_subtype_score_conditional_policy(
        cfg, policy_CA, n_oracle=min(oracle_n, 15000), seed=seed + 12321,
    )
    if util_AC >= util_CA:
        conditional_gold_policy = policy_AC
        conditional_gold_utility = util_AC
    else:
        conditional_gold_policy = policy_CA
        conditional_gold_utility = util_CA
    gold_intervention = sorted(scores, key=lambda s: s["expected_utility"], reverse=True)[0]["intervention"]
    recoverability = _static_recoverability_band(
        cfg, gold_intervention=gold_intervention, seed_base=seed + 9001,
    )
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)
    # Extra validator: the conditional policy must beat the best static.
    static_best = max(s["expected_utility"] for s in scores)
    conditional_gain = float(conditional_gold_utility - static_best)
    checks.append(_static_check(
        "conditional_beats_static",
        conditional_gain >= 3.0,
        conditional_gain,
        ">= 3.0",
        "Best conditional policy beats the best static intervention by margin.",
    ))
    question = (
        f"Different units in this population respond differently to the same "
        f"intervention. Propose a *conditional policy* keyed on the observable "
        f"`{names['subtype_screen']}` (or `{names['secondary_subtype_screen']}`): "
        f"map a screen reading above a chosen threshold to one of "
        f"`{names['treatment_a_knob']}` / `{names['treatment_b_knob']}` / "
        f"`{names['treatment_c_knob']}`, and below the threshold to another. "
        f"Submit a conditional_policy JSON with `branch_variable`, "
        f"`branch_threshold`, `if_above`, `if_below`, plus a one-paragraph "
        f"hypothesis about the heterogeneity."
    )
    objective = f"Maximize expected {names['target_outcome_obs']} via a conditional policy on the SubtypeScreen."
    visible_block = _static_visible_block_hidden_subtype(
        cfg, template, question, "conditional_policy", max_knobs=2,
        target_gold_position=target_gold_position,
    )
    world = _static_assemble_world(
        cfg=cfg, template=template, scores=scores, gold=gold, runner_up=runner_up,
        margin=margin, checks=checks, observational_diag=observational_diag,
        recoverability=recoverability, seed=seed, oracle_n=oracle_n, question=question,
        answer_schema="conditional_policy", max_knobs=2, objective=objective,
        visible_block=visible_block,
    )
    # Override gold_answer with the conditional policy.
    world["oracle"]["gold_answer"] = {
        "answer_schema": "conditional_policy",
        "policy": conditional_gold_policy,
        "expected_utility": conditional_gold_utility,
        "static_best_intervention": gold["intervention"],
        "static_best_expected_utility": static_best,
        "conditional_gain_over_static": conditional_gain,
    }
    world["oracle"]["oracle_tolerance"] = max(2.0, 0.5 * conditional_gain)
    world["questions"][0]["answer"] = {
        "answer_schema": "conditional_policy",
        "policy": conditional_gold_policy,
        "hypothesis": None,
    }
    return world


def _static_build_latent_regime_discovery(
    template: Dict[str, Any], *, seed: int, mixture_weight: float, oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    parameters = _static_default_params("latent_regime_discovery", rng)
    cfg = {
        "archetype": "latent_regime_discovery", "template": template, "parameters": parameters,
        "mixture_weight": mixture_weight, "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    recoverability = {
        "small_budget_mode": "n/a_latent_structure",
        "small_budget_n_obs": 0,
        "medium_budget_mode": "n/a_latent_structure",
        "medium_budget_n_per_intervention": 0,
        "n_seeds": 0,
        "small_budget_hit_rate": 0.0,
        "medium_budget_hit_rate": 1.0,
        "small_budget_top_picks": [],
        "gold_intervention_key": "conditional_latent_regime_policy",
    }
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)
    names = template["names"]
    branch_var = names["regime_proxy_a"]
    branch_threshold = 50.0
    policy_low_high = {
        "branch_variable": branch_var,
        "branch_threshold": branch_threshold,
        "if_above": {names["treatment_c_knob"]: "on"},
        "if_below": {names["treatment_a_knob"]: "on"},
    }
    policy_high_low = {
        "branch_variable": branch_var,
        "branch_threshold": branch_threshold,
        "if_above": {names["treatment_a_knob"]: "on"},
        "if_below": {names["treatment_c_knob"]: "on"},
    }
    policy_utility = _static_score_conditional_policy_utility(
        cfg, policy_low_high, n_oracle=min(oracle_n, 20000), seed=seed + 12321,
    )
    reverse_utility = _static_score_conditional_policy_utility(
        cfg, policy_high_low, n_oracle=min(oracle_n, 20000), seed=seed + 12321,
    )
    if reverse_utility > policy_utility:
        gold_policy = policy_high_low
        conditional_gold_utility = reverse_utility
    else:
        gold_policy = policy_low_high
        conditional_gold_utility = policy_utility
    static_best = max(s["expected_utility"] for s in scores)
    conditional_gain = float(conditional_gold_utility - static_best)
    checks.append(_static_check(
        "conditional_policy_beats_best_static",
        conditional_gain >= 4.0,
        conditional_gain,
        ">= 4.0",
        "Regime-aware conditional policy beats the best global single action.",
    ))
    checks.append(_static_check(
        "two_regimes_balanced",
        0.35 <= observational_diag.get("latent_regime_rate_high", 0.0) <= 0.65,
        observational_diag.get("latent_regime_rate_high"),
        "0.35 <= high-regime rate <= 0.65",
        "Latent regimes are both common enough to matter.",
    ))
    checks.append(_static_check(
        "latent_axis_separated",
        observational_diag.get("latent_axis_gap", 0.0) >= 28.0,
        observational_diag.get("latent_axis_gap"),
        ">= 28.0",
        "Hidden regimes create a meaningful distributional split.",
    ))
    question = (
        f"Clinicians disagree about whether this population is one noisy "
        f"syndrome or multiple hidden response regimes. Use the available "
        f"query modes and budget to determine the latent structure and propose "
        f"a regime-aware action policy for improving {names['target_outcome_obs']}. "
        f"Submit JSON with `latent_structure` (including `n_regimes` and "
        f"brief evidence), `policy` (with `branch_variable`, `branch_threshold`, "
        f"`if_above`, and `if_below`), and a one-paragraph `hypothesis`."
    )
    objective = (
        f"Discover the hidden response-regime structure and maximize expected "
        f"{names['target_outcome_obs']} using a regime-aware policy."
    )
    visible_block = _static_visible_block_latent_regime(
        cfg, template, question, "latent_regime_policy", max_knobs=2,
        target_gold_position=target_gold_position,
    )
    world = _static_assemble_world(
        cfg=cfg, template=template, scores=scores, gold=gold, runner_up=runner_up,
        margin=margin, checks=checks, observational_diag=observational_diag,
        recoverability=recoverability, seed=seed, oracle_n=oracle_n, question=question,
        answer_schema="latent_regime_policy", max_knobs=2, objective=objective,
        visible_block=visible_block,
    )
    latent_structure = {
        "n_regimes": 2,
        "evidence": (
            f"{names['regime_proxy_a']} and {names['regime_proxy_b']} separate "
            f"the population into low- and high-axis response regimes."
        ),
    }
    world["oracle"]["gold_answer"] = {
        "answer_schema": "latent_regime_policy",
        "latent_structure": latent_structure,
        "policy": gold_policy,
        "expected_utility": conditional_gold_utility,
        "static_best_intervention": gold["intervention"],
        "static_best_expected_utility": static_best,
        "conditional_gain_over_static": conditional_gain,
    }
    world["oracle"]["oracle_tolerance"] = max(2.0, 0.5 * conditional_gain)
    world["questions"][0]["answer"] = {
        "answer_schema": "latent_regime_policy",
        "latent_structure": latent_structure,
        "policy": gold_policy,
        "hypothesis": None,
    }
    world["questions"][0]["metadata"]["gold"] = _jsonify(world["oracle"]["gold_answer"])
    return world


def _static_build_anomaly_discovery(
    template: Dict[str, Any], *, seed: int, mixture_weight: float, oracle_n: int,
    target_gold_position: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    parameters = _static_default_params("anomaly_discovery", rng)
    cfg = {
        "archetype": "anomaly_discovery", "template": template, "parameters": parameters,
        "mixture_weight": mixture_weight, "seed": seed,
    }
    scores = _static_oracle_score(cfg, n_oracle=oracle_n, seed=seed + 7001)
    observational_diag = _static_oracle_observational_correlations(cfg, n=4000, seed=seed + 8001)
    gold_intervention: Dict[str, Any] = {}
    # For anomaly_discovery the recoverability band is meaningless against an
    # action gold. Synthesize a trivial diagnostic dict; validator branch
    # skips both recoverability checks anyway.
    recoverability = {
        "small_budget_mode": "n/a", "small_budget_n_obs": 0,
        "medium_budget_mode": "n/a", "medium_budget_n_per_intervention": 0,
        "n_seeds": 0, "small_budget_hit_rate": 0.0, "medium_budget_hit_rate": 1.0,
        "small_budget_top_picks": [], "gold_intervention_key": "NoIntervention",
    }
    checks, gold, runner_up, margin = _static_validate(cfg, scores, observational_diag, recoverability)
    names = template["names"]
    question = (
        f"Most units in this population follow a stable joint distribution over "
        f"`{names['feature_a']}`, `{names['feature_b']}`, `{names['feature_c']}`, "
        f"`{names['feature_d']}`, and `{names['secondary_signal']}`. A small "
        f"fraction do not. Sample observational data, identify the anomalous "
        f"units, and characterize how their feature signature differs. Submit a "
        f"JSON containing `flagged_unit_ids` (the unit ids you observed and "
        f"believe are anomalous) and `anomaly_rule` (a short conjunctive rule "
        f"like 'FeatureA > 70 AND FeatureB < 30')."
    )
    objective = "Identify the anomalous subpopulation by flagged unit IDs and a feature-signature rule."
    visible_block = _static_visible_block_anomaly_discovery(
        cfg, template, question, "anomaly_identification", max_knobs=0,
        target_gold_position=target_gold_position,
    )
    world = _static_assemble_world(
        cfg=cfg, template=template, scores=scores, gold=gold, runner_up=runner_up,
        margin=margin, checks=checks, observational_diag=observational_diag,
        recoverability=recoverability, seed=seed, oracle_n=oracle_n, question=question,
        answer_schema="anomaly_identification", max_knobs=0, objective=objective,
        visible_block=visible_block,
    )
    # Store the gold rule as a structured representation (the audit/score code
    # can apply it directly to feature arrays for fresh-batch precision/recall).
    p = cfg["parameters"]
    world["oracle"]["gold_answer"] = {
        "answer_schema": "anomaly_identification",
        "anomaly_rule_structured": {
            "all_of": [
                {"variable": names["feature_a"], "op": ">", "threshold": float(p["feature_mean"]) + 12.0},
                {"variable": names["feature_b"], "op": "<", "threshold": float(p["feature_mean"]) - 12.0},
            ],
            "logic": "both_conditions_indicate_anomaly",
        },
        "anomaly_prevalence": float(p["p_anomaly"]),
        "precision_recall_threshold": {"precision": 0.7, "recall": 0.6},
    }
    world["oracle"]["oracle_tolerance"] = 0.0
    world["questions"][0]["answer"] = {
        "answer_schema": "anomaly_identification",
        "flagged_unit_ids": [],
        "anomaly_rule": f"{names['feature_a']} > {float(p['feature_mean']) + 12.0:.0f} AND {names['feature_b']} < {float(p['feature_mean']) - 12.0:.0f}",
        "hypothesis": None,
    }
    return world


STATIC_BUILDERS = {
    "hidden_cause": _static_build_hidden_cause,
    "confounded_action": _static_build_confounded_action,
    "mechanism_chain": _static_build_mechanism_chain,
    "negative_control": _static_build_negative_control,
    "hidden_subtype": _static_build_hidden_subtype,
    "anomaly_discovery": _static_build_anomaly_discovery,
    "latent_regime_discovery": _static_build_latent_regime_discovery,
}


# ---------------------------------------------------------------------------
# Top-level static-rpg generation
# ---------------------------------------------------------------------------

def static_rpg_generate_world(
    archetype: str,
    seed: int,
    outdir: str,
    *,
    template_index: Optional[int] = None,
    mixture_weight: Optional[float] = None,
    oracle_n: int = STATIC_DEFAULT_ORACLE_N,
    max_attempts: int = 6,
    target_gold_position: int = 0,
    llm: Optional["_StaticRPGLLM"] = None,
) -> Dict[str, Any]:
    if archetype not in STATIC_BUILDERS:
        raise KeyError(f"unknown static archetype {archetype!r}")
    os.makedirs(outdir, exist_ok=True)
    templates = STATIC_TEMPLATES[archetype]
    last_reason = "no attempts"
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 9973
        idx = template_index if template_index is not None else (seed + attempt) % len(templates)
        template = templates[idx % len(templates)]
        rho = (
            mixture_weight
            if mixture_weight is not None
            else STATIC_MIXTURE_WEIGHTS[(seed + attempt) % len(STATIC_MIXTURE_WEIGHTS)]
        )
        world = STATIC_BUILDERS[archetype](
            template,
            seed=attempt_seed,
            mixture_weight=rho,
            oracle_n=oracle_n,
            target_gold_position=target_gold_position,
        )
        if world["validators"]["accepted"]:
            if llm is not None:
                world = _static_llm_polish_world(world, llm)
            world_id = world["meta"]["world_id"]
            json_path = os.path.join(outdir, f"world_{world_id}.json")
            world["meta"]["json_path"] = json_path
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(world, f, ensure_ascii=False, indent=2)
            return {
                "world": world,
                "path": json_path,
                "requested_seed": seed,
                "actual_seed": int(world["meta"]["seed"]),
                "attempt_index": attempt,
            }
        failed = [c for c in world["validators"]["signature_checks"] if not c.get("passed")]
        last_reason = "; ".join(f"{c['name']}={c['value']}" for c in failed[:4]) or "validator rejected"
    raise RuntimeError(f"{archetype} failed static validation after {max_attempts} attempts: {last_reason}")


def static_rpg_generate_dataset(
    *,
    outdir: str,
    distribution: Dict[str, int],
    seed_base: int = 5000,
    oracle_n: int = STATIC_DEFAULT_ORACLE_N,
    max_attempts_per_world: int = 6,
    only_archetype: Optional[str] = None,
    llm: Optional["_StaticRPGLLM"] = None,
) -> List[Dict[str, Any]]:
    os.makedirs(outdir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    skipped: List[Tuple[int, str, str]] = []
    i = 0
    for arch in STATIC_ARCHETYPES:
        if only_archetype and arch != only_archetype:
            continue
        count = int(distribution.get(arch, 0))
        if count <= 0:
            continue
        if arch not in STATIC_BUILDERS or arch not in STATIC_TEMPLATES:
            skipped.append((i, arch, "static archetype is listed but not implemented yet"))
            i += count
            continue
        n_templates = len(STATIC_TEMPLATES[arch])
        n_knobs = _static_n_intervenable_knobs(arch)
        for j in range(count):
            seed = seed_base + i * 101
            template_idx = j % n_templates
            rho = STATIC_MIXTURE_WEIGHTS[j % len(STATIC_MIXTURE_WEIGHTS)]
            # Rotate the gold-knob's slot across worlds so the answer is
            # uniformly distributed over knob positions.
            gold_pos = j % n_knobs
            print(
                f"\n[static {i + 1}] {arch} seed={seed} mix={rho} "
                f"template_idx={template_idx} gold_pos={gold_pos}"
            )
            try:
                res = static_rpg_generate_world(
                    arch,
                    seed,
                    outdir,
                    template_index=template_idx,
                    mixture_weight=rho,
                    oracle_n=oracle_n,
                    max_attempts=max_attempts_per_world,
                    target_gold_position=gold_pos,
                    llm=llm,
                )
                polish_meta = res["world"].get("meta", {}).get("llm_polish")
                results.append({
                    "path": res["path"],
                    "archetype": arch,
                    "seed": seed,
                    "requested_seed": res.get("requested_seed", seed),
                    "actual_seed": res["world"]["meta"].get("seed"),
                    "attempt_index": res.get("attempt_index", 0),
                    "mixture_weight": rho,
                    "template_index": template_idx,
                    "llm_polish": polish_meta,
                })
                print(f"  [ok] {os.path.basename(res['path'])}")
            except Exception as e:
                print(f"  [skip] {arch}: {e}")
                skipped.append((i, arch, str(e)))
            i += 1
    manifest = {
        "schema_version": SCHEMA_VERSION_STATIC,
        "benchmark": BENCHMARK_NAME_STATIC,
        "outdir": outdir,
        "seed_base": seed_base,
        "oracle_n": oracle_n,
        "requested_distribution": distribution,
        "generated": len(results),
        "skipped": [{"slot": s_i, "archetype": a, "reason": r} for s_i, a, r in skipped],
        "worlds": results,
    }
    manifest_path = os.path.join(outdir, "manifest_rpg_static_v2.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nGenerated {len(results)} static-rpg worlds in {outdir}")
    return results


# ---------------------------------------------------------------------------
# Optional LLM hooks (Opus 4.8 via Bedrock).
# ---------------------------------------------------------------------------
#
# The LLM is allowed to *polish narrative* and *propose new templates*. It is
# NEVER given access to oracle scores, validators, or mechanism parameters.
# A post-polish leakage check rejects any output that mentions hidden
# variable names or role labels.

STATIC_LLM_DEFAULT_MODEL = "us.anthropic.claude-opus-4-7"
STATIC_LLM_DEFAULT_TEMP = 0.6
STATIC_LLM_MAX_TOKENS = 2200


@dataclass
class _StaticRPGLLM:
    """Thin wrapper exposing .chat(system, user) -> str over Bedrock.
    Mirrors world_gen_advanced.py:_LLM."""
    model_id: str = STATIC_LLM_DEFAULT_MODEL
    region_name: Optional[str] = None
    temperature: float = STATIC_LLM_DEFAULT_TEMP
    max_new_tokens: int = STATIC_LLM_MAX_TOKENS
    _client: Any = field(default=None, init=False, repr=False)
    model_name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        from bedrock_llm import BedrockLLM  # type: ignore
        self._client = BedrockLLM(
            model_id=self.model_id,
            region_name=self.region_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        self.model_name = self.model_id

    def chat(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        return self._client.generate(system, user, max_new_tokens=max_new_tokens)


def _static_extract_first_json(text: str) -> Dict[str, Any]:
    """Extract the first balanced {...} object from `text`. Raises on failure."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return json.loads(text[start : i + 1])
    raise ValueError("No balanced JSON object found in LLM response")


def _static_llm_json(
    llm: _StaticRPGLLM,
    system: str,
    user: str,
    *,
    max_tries: int = 3,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Call LLM, expect JSON output, retry on parse failure."""
    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        try:
            raw = llm.chat(system, user, max_new_tokens=max_new_tokens)
            return _static_extract_first_json(raw)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"static-rpg LLM JSON parse failed after {max_tries} tries: {last_err}")


# Per-archetype guidance for the LLM. Describes the *shape* of the test
# without naming which knob is the gold answer.
_STATIC_ARCHETYPE_BRIEF: Dict[str, str] = {
    "hidden_cause": (
        "This world tests whether a scientist agent can discover a hidden "
        "construct behind a chronic problem and choose an unconventional "
        "intervention. There are multiple intervenable knobs. One acts on a "
        "construct that explains the problem. The others match a popular "
        "prevailing theory but do not actually fix the underlying issue. "
        "Your description must make every knob sound *equally plausible* — "
        "a domain expert who knows the real-world answer for a specific "
        "scenario (e.g. H. pylori for ulcers) must not be able to leak that "
        "answer into the descriptions. The story should explicitly name the "
        "prevailing theory because the entrenched prior is part of the test."
    ),
    "confounded_action": (
        "This world tests whether a scientist agent can recognize that "
        "observational data is misleading because the most-intensive "
        "intervention is currently given to the worst-off units. The "
        "intervention is actually beneficial under do(), but passive records "
        "make it look harmful. Your story must mention the observational "
        "pattern (high-intensity intervention paired with worse outcomes) "
        "and the conventional wisdom that has grown around it, without "
        "telling the reader the answer."
    ),
    "mechanism_chain": (
        "This world tests whether a scientist agent can identify the "
        "bottleneck step in a three-stage sequential process. Each stage has "
        "a measurable proxy and an intervention. Only the bottleneck stage's "
        "intervention meaningfully moves the final outcome; the others have "
        "small or no effect because their stages are already operating well. "
        "Your story should describe the stages as a normal pipeline without "
        "indicating which stage is the bottleneck. All stage interventions "
        "must sound equally credible."
    ),
    "negative_control": (
        "This world tests whether a scientist agent can recognize that "
        "*none* of the offered interventions actually improves the target "
        "under controlled assignment, even though observational data make "
        "them look beneficial because users who self-select into them have "
        "intrinsically better outcomes. The agent should ultimately submit "
        "the empty intervention. Your story should describe the popular "
        "intervention positively (as people genuinely talk about it) without "
        "stating that it does or does not work."
    ),
    "hidden_subtype": (
        "This world tests whether a scientist agent can recognize "
        "heterogeneity of response across hidden subtypes and propose a "
        "conditional policy keyed on a noisy proxy. There are three offered "
        "treatments, two SubtypeScreen-like proxies, and one baseline-risk "
        "proxy. No single global treatment is best for the whole population. "
        "Your story should describe the heterogeneity without naming a "
        "subtype variable, without saying which treatment is for whom, and "
        "without giving away the answer."
    ),
    "anomaly_discovery": (
        "This world tests whether a scientist agent can identify a small "
        "subpopulation generated by a different feature-distribution from "
        "the rest. There are no causal interventions in this archetype. The "
        "agent's job is to observe feature samples, characterize the "
        "anomalous signature, and identify anomalous units. Your story "
        "should set up a population with a 'small fraction behaves "
        "differently' framing, without naming the anomaly indicator or "
        "which features carry the signal."
    ),
}


def _static_leakage_terms() -> List[str]:
    """Substrings whose presence in polished text indicates internal leakage.

    Kept tight on purpose: structural names + a few role words that are
    only meaningful inside the simulator. Common causal-inference vocabulary
    (confounded, intervention, causal) is allowed."""
    return [
        "LatentBurden", "LatentDriver",
        "BurdenSubstrate", "BackgroundStrain",
        "HealthSeekingTrait", "HealthSeeking",
        "BaselineSeverity",
        "DecoyState",
        "LatentSeverity",
        "decoy",
        "red herring",
        "true lever",
        "is the gold",
        "is the right answer",
    ]


def _static_polish_validate(
    candidate: Dict[str, Any],
    obs_names: List[str],
    intv_names: List[str],
) -> Tuple[bool, str]:
    if not isinstance(candidate, dict):
        return False, "not a dict"
    if "story" not in candidate or "question" not in candidate:
        return False, "missing story/question"
    obs_pol = candidate.get("observed_variables") or []
    intv_pol = candidate.get("intervenable_variables") or []
    pol_obs_names = {v.get("name") for v in obs_pol if isinstance(v, dict)}
    pol_intv_names = {v.get("name") for v in intv_pol if isinstance(v, dict)}
    if pol_obs_names != set(obs_names):
        return False, f"observed names changed: {sorted(pol_obs_names - set(obs_names))}"
    if pol_intv_names != set(intv_names):
        return False, f"intervenable names changed: {sorted(pol_intv_names - set(intv_names))}"
    # Question must still mention the answer schema obligations.
    q = (candidate.get("question") or "").lower()
    if "do(" not in q and "intervention" not in q:
        return False, "question lost reference to intervention/do(.)"
    if "hypothesis" not in q:
        return False, "question lost the hypothesis requirement"
    return True, "ok"


def _static_leakage_check(blob: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    joined = json.dumps(blob).lower()
    for term in _static_leakage_terms():
        if term.lower() in joined:
            return False, term
    return True, None


def _static_llm_polish_world(
    world: Dict[str, Any],
    llm: _StaticRPGLLM,
    *,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Polish story / question / variable descriptions in place (on a copy).

    Mechanism, oracle, validators, and variable NAMES are not touched.
    Falls back to the un-polished world if the polished output fails the
    leakage check or schema check."""
    archetype = world["meta"]["archetype"]
    visible = world["visible"]
    obs_var_names = [v["name"] for v in visible["observed_variables"]]
    intv_var_names = [v["name"] for v in visible["intervenable_variables"]]
    brief = _STATIC_ARCHETYPE_BRIEF.get(archetype, "")

    system = (
        "You are an expert science writer producing items for a "
        "causal-discovery benchmark. Polish the story, question, and "
        "variable descriptions of the input world so they read like a "
        "natural human briefing. You may rewrite freely for tone and "
        "flow, but you must obey the hard rules below.\n\n"
        f"BENCHMARK BRIEF FOR THIS ARCHETYPE:\n{brief}\n\n"
        "HARD RULES:\n"
        "1. Every input variable name (observed and intervenable) must appear "
        "verbatim in your output. Do not rename, abbreviate, or expand them.\n"
        "2. Do not introduce new variables.\n"
        "3. Do not state or imply which intervention is the correct answer.\n"
        "4. Do not say 'decoy', 'red herring', or 'true lever'. Do not name "
        "internal simulator constructs like 'LatentDriver', "
        "'HealthSeekingTrait', 'BackgroundStrain', or 'BaselineSeverity'.\n"
        "5. Describe every intervenable knob equally plausibly. Even if your "
        "training data tells you the textbook answer for the specific "
        "scenario (e.g. antibiotics for H. pylori), do NOT lean into it in "
        "the descriptions.\n"
        "6. The polished question must still: (a) ask for an action expressed "
        "as a do(.) dict over the intervenable variables, and (b) require a "
        "one-paragraph hypothesis describing the underlying mechanism.\n"
        "7. Output JSON only. No markdown, no commentary, no fences. JSON "
        "must contain top-level keys: story, question, observed_variables, "
        "intervenable_variables. Each *_variables entry is a dict with "
        "exactly two keys: name, description."
    )
    user = json.dumps({
        "archetype": archetype,
        "domain": world["meta"]["topic"],
        "subdomain": world["meta"]["subdomain"],
        "answer_schema": visible["answer_schema"],
        "max_intervention_knobs": visible.get("max_intervention_knobs"),
        "current_story": visible["story"],
        "current_question": visible["question"],
        "observed_variables": [
            {"name": v["name"], "description": v["description"]}
            for v in visible["observed_variables"]
        ],
        "intervenable_variables": [
            {
                "name": v["name"],
                "values": v["values"],
                "default": v["default"],
                "description": v["description"],
            }
            for v in visible["intervenable_variables"]
        ],
    }, indent=2)

    polished: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            candidate = _static_llm_json(llm, system, user)
        except Exception as e:
            rejection_reason = f"json_parse:{e}"
            continue
        ok, why = _static_polish_validate(candidate, obs_var_names, intv_var_names)
        if not ok:
            rejection_reason = f"schema:{why}"
            continue
        leak_ok, leak_term = _static_leakage_check(candidate)
        if not leak_ok:
            rejection_reason = f"leakage:{leak_term}"
            continue
        polished = candidate
        break

    if polished is None:
        world.setdefault("meta", {})["llm_polish"] = {
            "applied": False,
            "rejected_reason": rejection_reason,
            "model": llm.model_name,
        }
        return world

    new_world = copy.deepcopy(world)
    new_world["visible"]["story"] = polished["story"]
    new_world["visible"]["question"] = polished["question"]
    obs_desc_map = {v["name"]: v["description"] for v in polished["observed_variables"]}
    for v in new_world["visible"]["observed_variables"]:
        if v["name"] in obs_desc_map:
            v["description"] = obs_desc_map[v["name"]]
    intv_desc_map = {v["name"]: v["description"] for v in polished["intervenable_variables"]}
    for v in new_world["visible"]["intervenable_variables"]:
        if v["name"] in intv_desc_map:
            v["description"] = intv_desc_map[v["name"]]
    new_world["story"] = polished["story"]
    if new_world.get("questions"):
        new_world["questions"][0]["question"] = polished["question"]
    new_world.setdefault("meta", {})["llm_polish"] = {
        "applied": True,
        "model": llm.model_name,
    }
    return new_world


# ---------------------------------------------------------------------------
# Optional: LLM proposes additional templates so diversity isn't limited to
# the three hand-written domains per archetype.
# ---------------------------------------------------------------------------

def _static_required_template_keys(archetype: str) -> Tuple[List[str], List[str]]:
    """Return (role_keys_required_in_names_dict, top_level_keys_required)."""
    _ROLE_KEYS: Dict[str, List[str]] = {
        "hidden_cause": [
            "primary_target_obs", "secondary_target_obs", "latent_driver_proxy",
            "decoy_proxy_a", "decoy_proxy_b", "tertiary_obs",
            "decoy_knob_a", "decoy_knob_b", "weak_knob", "true_lever_knob",
        ],
        "confounded_action": [
            "primary_target_obs", "severity_proxy_a", "severity_proxy_b",
            "secondary_target_obs", "assignment_record",
            "treatment_knob", "support_knob",
        ],
        "mechanism_chain": [
            "final_outcome_obs", "secondary_outcome_obs",
            "stage_1_proxy", "stage_2_proxy", "stage_3_proxy",
            "stage_1_knob", "stage_2_knob", "stage_3_knob",
        ],
        "negative_control": [
            "primary_outcome_obs", "secondary_outcome_obs",
            "wellbeing_proxy", "engagement_proxy", "healthseek_proxy",
            "trendy_knob", "conventional_knob", "research_backed_knob",
        ],
        "hidden_subtype": [
            "target_outcome_obs", "secondary_outcome_obs",
            "subtype_screen", "secondary_subtype_screen", "baseline_risk_proxy",
            "treatment_a_knob", "treatment_b_knob", "treatment_c_knob",
        ],
        "anomaly_discovery": [
            "feature_a", "feature_b", "feature_c", "feature_d", "secondary_signal",
        ],
    }
    if archetype not in _ROLE_KEYS:
        raise KeyError(archetype)
    top_keys = [
        "topic", "subdomain", "setting", "unit", "names",
        "knob_descriptions", "measurement_descriptions",
    ]
    return _ROLE_KEYS[archetype], top_keys


def _static_role_briefs(archetype: str) -> Dict[str, str]:
    """Per-role semantic guidance for the LLM template proposer.

    These briefs tell the LLM WHAT each role key represents in the underlying
    mechanism, so the names + descriptions it invents cohere with the
    mechanism the code will run. They are used ONLY at template-proposal
    time. The agent never sees them: only the final neutral
    `knob_descriptions` and `measurement_descriptions` reach the visible
    block.

    Rule of thumb when adding new archetypes: every role key needs a brief.
    A brief that says 'pick the gold answer' or 'pick the unsafe one' would
    be a leak; the briefs below describe ROLE not LABEL.
    """
    if archetype == "hidden_cause":
        return {
            "primary_target_obs": (
                "A 0-100 score measuring the headline symptom that has been "
                "chronically elevated in this population. Lower is better. "
                "This is the headline signal for the underlying condition "
                "the agent is asked to improve."
            ),
            "secondary_target_obs": (
                "An independent 0-100 measurement of the same underlying "
                "outcome, drawn from a different instrument. Useful for "
                "cross-checking the primary outcome. Lower is better."
            ),
            "latent_driver_proxy": (
                "A routine 0-100 lab/instrument reading available during "
                "the study window. "
                "Sounds mundane and operational. Higher values mean the "
                "instrument detected something, but the description must "
                "not say what. This is the only proxy that hints at the "
                "hidden construct, so its description should be neutral and "
                "boring — definitely not 'a marker of the underlying cause'."
            ),
            "decoy_proxy_a": (
                "A 0-100 measurement aligned with one pillar of the popular "
                "prevailing theory (e.g. stress for ulcers; device quality "
                "for chargebacks). Higher means more of that factor present."
            ),
            "decoy_proxy_b": (
                "A 0-100 measurement aligned with a second pillar of the "
                "popular prevailing theory (e.g. diet for ulcers; account "
                "age for chargebacks). Independent of decoy_proxy_a."
            ),
            "tertiary_obs": (
                "A 0-100 secondary outcome that moves in the opposite "
                "direction to primary_target_obs when the underlying issue "
                "improves (e.g. quality of life, satisfaction). Higher is "
                "better. Provides another cross-check channel."
            ),
            "decoy_knob_a": (
                "A binary intervention drawn from the prevailing wrong "
                "theory's first pillar (e.g. stress-reduction program; "
                "device-quality nudge). Plausible-sounding. Current "
                "practice prescribes it. Does NOT address the hidden "
                "construct."
            ),
            "decoy_knob_b": (
                "A binary intervention drawn from the prevailing wrong "
                "theory's second pillar (e.g. diet modification; "
                "account-age rule). Independent of decoy_knob_a but "
                "similarly prescribed by current practice. Does NOT "
                "address the hidden construct."
            ),
            "weak_knob": (
                "A discrete-level palliative intervention (values "
                "off / low / high). It suppresses the visible "
                "primary_target_obs reading (e.g. antacid suppressing "
                "symptoms, MFA friction reducing chargebacks at the "
                "transaction layer) without changing the underlying "
                "construct. Should sound symptomatic, not root-cause."
            ),
            "true_lever_knob": (
                "A binary unconventional intervention that the prevailing "
                "theory does NOT emphasize, but which actually acts on the "
                "hidden construct (e.g. antibiotics for ulcers, "
                "TLS-fingerprint gate for fraud, root-zone fungicide for "
                "orchard yield). Its name and description must NOT signal "
                "'this is the answer' — present it as one more thing on "
                "the menu."
            ),
        }
    if archetype == "confounded_action":
        return {
            "primary_target_obs": (
                "A 0-100 outcome score where higher is better (recovery, "
                "reemployment, graduation). The agent is asked to maximize "
                "this."
            ),
            "severity_proxy_a": (
                "A 0-100 noisy proxy of how badly-off a unit was at intake "
                "(higher = sicker / more disadvantaged / more at-risk). "
                "Informative but imperfect."
            ),
            "severity_proxy_b": (
                "A second 0-100 noisy proxy of pre-intervention severity, "
                "drawn from a different source than severity_proxy_a."
            ),
            "secondary_target_obs": (
                "A 0-100 independent measurement of the same final outcome, "
                "higher is better. Cross-check channel."
            ),
            "assignment_record": (
                "A categorical record of what level of the treatment_knob "
                "this unit actually received under current practice. "
                "Used by the agent for observational analysis."
            ),
            "treatment_knob": (
                "The main intervention, with discrete levels off / low / "
                "high. Under current practice it is preferentially "
                "assigned to the worst-off units. Under do() it actually "
                "helps. Name and description must sound conventional, NOT "
                "tip off that the high level is the right answer."
            ),
            "support_knob": (
                "A binary auxiliary intervention (off / on) that is "
                "moderately helpful regardless of severity (e.g. early "
                "mobilization, case management, mentorship). Currently "
                "assigned roughly at random."
            ),
        }
    if archetype == "mechanism_chain":
        return {
            "final_outcome_obs": (
                "A 0-100 score measuring the pipeline's final output that "
                "is below target. Higher is better."
            ),
            "secondary_outcome_obs": (
                "An independent 0-100 measurement of a downstream outcome "
                "(e.g. revenue, quality, placement). Higher is better."
            ),
            "stage_1_proxy": (
                "A 0-100 measurement of stage 1's throughput / pass-through "
                "rate. Higher means stage 1 is operating well."
            ),
            "stage_2_proxy": (
                "A 0-100 measurement of stage 2's throughput / pass-through "
                "rate. Higher means stage 2 is operating well."
            ),
            "stage_3_proxy": (
                "A 0-100 measurement of stage 3's throughput / pass-through "
                "rate. Higher means stage 3 is operating well."
            ),
            "stage_1_knob": (
                "A binary intervention that boosts stage 1's throughput. "
                "Sounds like a standard operational program targeting that "
                "stage."
            ),
            "stage_2_knob": (
                "A binary intervention that boosts stage 2's throughput. "
                "Sounds like a standard operational program targeting that "
                "stage."
            ),
            "stage_3_knob": (
                "A binary intervention that boosts stage 3's throughput. "
                "Sounds like a standard operational program targeting that "
                "stage."
            ),
        }
    if archetype == "negative_control":
        return {
            "primary_outcome_obs": (
                "A 0-100 score measuring the headline outcome the agent is "
                "asked to improve. Lower is better (e.g. churn risk, "
                "recovery delay)."
            ),
            "secondary_outcome_obs": (
                "An independent 0-100 measurement of the same outcome. "
                "Lower is better."
            ),
            "wellbeing_proxy": (
                "A 0-100 positive-direction wellbeing measure (higher = "
                "better). Generally moves inversely to primary_outcome_obs."
            ),
            "engagement_proxy": (
                "A 0-100 engagement / activity measure. Correlates with the "
                "hidden HealthSeeking-like trait that drives intervention "
                "uptake."
            ),
            "healthseek_proxy": (
                "A 0-100 baseline motivation / initiative measure from "
                "intake. Correlates strongly with HealthSeeking-like trait."
            ),
            "trendy_knob": (
                "A binary intervention that is currently culturally popular "
                "or socially endorsed (e.g. trending challenge, popular "
                "adjunct therapy, viral workshop). Heavily adopted under "
                "current practice."
            ),
            "conventional_knob": (
                "A binary intervention that is the conventional, established "
                "playbook (e.g. classic reminder campaign, standard-of-care "
                "adjunct)."
            ),
            "research_backed_knob": (
                "A 3-level (off / low / high) intervention framed as "
                "evidence-based or research-backed protocol. Plausibly "
                "credible-sounding."
            ),
        }
    if archetype == "hidden_subtype":
        return {
            "target_outcome_obs": (
                "A 0-100 score measuring the headline outcome (recovery, "
                "improvement, engagement). Higher is better."
            ),
            "secondary_outcome_obs": (
                "An independent 0-100 longer-term outcome (retention, "
                "follow-up assessment). Higher is better."
            ),
            "subtype_screen": (
                "A 0-100 noisy classifier reading from a validated intake "
                "screen / questionnaire. It correlates with the hidden "
                "subtype but is imperfect."
            ),
            "secondary_subtype_screen": (
                "A 0-100 secondary noisy classifier reading from an "
                "independent intake instrument. Less reliable than the "
                "primary screen."
            ),
            "baseline_risk_proxy": (
                "A 0-100 baseline-condition severity proxy. Lower means a "
                "less severe baseline."
            ),
            "treatment_a_knob": (
                "A binary intervention that is best for one subtype and "
                "weak for the other. Treat as one of three peer treatments."
            ),
            "treatment_b_knob": (
                "A binary intervention that is moderate for both subtypes "
                "— a 'balanced' option."
            ),
            "treatment_c_knob": (
                "A binary intervention that is best for the other subtype "
                "and weak for the first. Treat as one of three peer "
                "treatments."
            ),
        }
    if archetype == "anomaly_discovery":
        return {
            "feature_a": (
                "A 0-100 feature that is the primary anomaly-shifting feature "
                "(anomalous units show an upward shift on this feature). "
                "Should sound like a routine domain measurement."
            ),
            "feature_b": (
                "A 0-100 feature that is the secondary anomaly-shifting "
                "feature (anomalous units show a downward shift on this "
                "feature). Should sound like a routine domain measurement."
            ),
            "feature_c": (
                "A 0-100 feature that is NOT involved in the anomaly signature "
                "and looks the same for anomalous and normal units. Should "
                "sound like a routine domain measurement."
            ),
            "feature_d": (
                "A 0-100 feature that is NOT involved in the anomaly signature "
                "and looks the same for anomalous and normal units. Should "
                "sound like a routine domain measurement."
            ),
            "secondary_signal": (
                "A 0-100 independent (weaker) anomaly signal — useful for "
                "cross-checking once the agent has a candidate rule."
            ),
        }
    raise KeyError(archetype)


def _static_template_validate(
    candidate: Dict[str, Any],
    archetype: str,
) -> Tuple[bool, str]:
    role_keys, top_keys = _static_required_template_keys(archetype)
    for k in top_keys:
        if k not in candidate:
            return False, f"missing top-level key {k}"
    names = candidate.get("names") or {}
    for rk in role_keys:
        if rk not in names:
            return False, f"missing names.{rk}"
        v = names[rk]
        if not isinstance(v, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]+", v):
            return False, f"names.{rk} not a CamelCase identifier"
    name_values = list(names.values())
    if len(set(name_values)) != len(name_values):
        return False, "duplicate variable name across roles"
    knob_roles = [k for k in role_keys if k.endswith("_knob")]
    measurement_roles = [k for k in role_keys if k not in knob_roles]
    for rk in knob_roles:
        nm = names[rk]
        if nm not in candidate["knob_descriptions"]:
            return False, f"knob_descriptions missing entry for {nm}"
    for rk in measurement_roles:
        nm = names[rk]
        if nm not in candidate["measurement_descriptions"]:
            return False, f"measurement_descriptions missing entry for {nm}"
    leak_ok, leak_term = _static_leakage_check(candidate)
    if not leak_ok:
        return False, f"leakage:{leak_term}"
    return True, "ok"


def _static_llm_propose_template(
    archetype: str,
    llm: _StaticRPGLLM,
    *,
    avoid_subdomains: Optional[List[str]] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Ask the LLM for a fresh domain template. Mechanism / role assignments
    are still owned by code (the role keys are fixed). The LLM only invents
    a domain, a scenario, neutral variable names, and short descriptions.

    Coherence is enforced by passing per-role *semantic briefs* — the LLM
    is told what each role represents in the mechanism so the names it
    invents match what the code will do with them. The briefs themselves
    are never written into the world JSON (the agent never sees them); the
    only thing that reaches the visible block is the polished neutral
    descriptions."""
    role_keys, top_keys = _static_required_template_keys(archetype)
    role_briefs = _static_role_briefs(archetype)
    avoid = avoid_subdomains or []
    brief = _STATIC_ARCHETYPE_BRIEF.get(archetype, "")
    # Show the required schema with one of the existing templates as a model.
    example_template = STATIC_TEMPLATES[archetype][0]
    system = (
        "You are an expert benchmark designer proposing a fresh domain "
        "template for a causal-discovery item. You will receive the "
        "archetype brief, the required schema, an example template you may "
        "use ONLY for structural reference (not for content), the semantic "
        "role each variable plays in the underlying mechanism, and a list "
        "of subdomains to avoid. Invent a plausible new scenario in a "
        "different subdomain.\n\n"
        f"ARCHETYPE BRIEF:\n{brief}\n\n"
        "WHAT THE ROLE KEYS MEAN — the names + descriptions you invent for "
        "each role must be CONSISTENT with the role's mechanism semantics. "
        "These role meanings are NEVER shown to the agent; the agent only "
        "sees your polished, neutral `knob_descriptions` and "
        "`measurement_descriptions`. Use the meanings here to choose "
        "internally coherent names, but do NOT echo the meanings (or "
        "words like 'decoy', 'true lever', 'palliative') into your output.\n\n"
        "HARD RULES:\n"
        "1. Return JSON only. No markdown fences. No commentary.\n"
        "2. The JSON must include every top-level key and every entry under "
        "`names`, exactly as specified.\n"
        "3. Every variable name (under `names`) is CamelCase, alphanumeric, "
        "and unique across roles.\n"
        "4. `knob_descriptions` and `measurement_descriptions` map every "
        "invented variable name to a NEUTRAL one-sentence description that "
        "is faithful to the role's semantics but does NOT betray the role.\n"
        "5. Knob descriptions must NOT reveal which knob is the correct "
        "answer. The agent must read the four knob descriptions and not "
        "be able to guess which one is the unconventional / "
        "sign-reversed / bottleneck answer.\n"
        "6. Do not use the words 'decoy', 'red herring', 'true lever', "
        "'palliative', or name any latent / hidden / confounder construct.\n"
        "7. Pick a subdomain that is NOT in the avoid list and is distinct "
        "from the example. Aim for a domain a human reader would find "
        "plausible (medical, operational, agricultural, civic, etc.).\n"
        "8. `setting` is a 2-3 sentence neutral framing that names the "
        "prevailing (wrong) theory if applicable to the archetype, "
        "without saying it is wrong.\n"
        "9. Every role's invented name + description must be semantically "
        "plausible given its role brief (e.g. if a role is 'palliative '"
        "intervention with off/low/high levels', the invented name should "
        "be something that could plausibly have three dose levels in the "
        "chosen domain)."
    )
    user = json.dumps({
        "archetype": archetype,
        "required_top_level_keys": top_keys,
        "required_name_roles": role_keys,
        "role_semantic_briefs": role_briefs,
        "example_template_for_structure_only": example_template,
        "avoid_subdomains": avoid,
    }, indent=2)

    last_reason: Optional[str] = None
    for _ in range(max_retries + 1):
        try:
            candidate = _static_llm_json(llm, system, user)
        except Exception as e:
            last_reason = f"json_parse:{e}"
            continue
        ok, why = _static_template_validate(candidate, archetype)
        if not ok:
            last_reason = f"schema:{why}"
            continue
        return candidate
    raise RuntimeError(f"LLM template proposal for {archetype} failed: {last_reason}")


def _static_main_subcli(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate static-rpg v2 partially-observed worlds.")
    ap.add_argument("--outdir", type=str, default="./out_rpg_static_v2")
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--oracle-n", type=int, default=STATIC_DEFAULT_ORACLE_N)
    ap.add_argument("--max-attempts-per-world", type=int, default=6)
    ap.add_argument("--only-archetype", type=str, choices=STATIC_ARCHETYPES, default=None)
    ap.add_argument(
        "--distribution",
        type=str,
        default=None,
        help='JSON counts, e.g. \'{"hidden_cause":3,"confounded_action":3}\'',
    )
    ap.add_argument(
        "--llm-polish",
        action="store_true",
        help="After each world is built, ask Opus 4.8 (Bedrock) to rewrite "
             "story/question/variable descriptions in natural prose. Falls "
             "back to the un-polished world if leakage is detected.",
    )
    ap.add_argument(
        "--llm-extra-templates",
        type=int,
        default=0,
        help="Before generation, ask Opus 4.8 to invent this many additional "
             "domain templates per archetype. 0 = use only the hand-written "
             "templates (deterministic, no AWS dependency).",
    )
    ap.add_argument(
        "--llm-model",
        type=str,
        default=STATIC_LLM_DEFAULT_MODEL,
        help="Bedrock model id for LLM polish / template proposal.",
    )
    args = ap.parse_args(argv)
    if args.distribution:
        data = json.loads(args.distribution)
        unknown = set(data) - set(STATIC_ARCHETYPES)
        if unknown:
            raise ValueError(f"unknown static archetypes: {sorted(unknown)}")
        distribution = {a: int(data.get(a, 0)) for a in STATIC_ARCHETYPES}
    else:
        distribution = dict(STATIC_DEFAULT_DISTRIBUTION)

    llm = None
    if args.llm_polish or args.llm_extra_templates > 0:
        llm = _StaticRPGLLM(model_id=args.llm_model)
        if args.llm_extra_templates > 0:
            _static_extend_templates_via_llm(llm, args.llm_extra_templates)

    static_rpg_generate_dataset(
        outdir=args.outdir,
        distribution=distribution,
        seed_base=args.seed_base,
        oracle_n=args.oracle_n,
        max_attempts_per_world=args.max_attempts_per_world,
        only_archetype=args.only_archetype,
        llm=llm if args.llm_polish else None,
    )


def _static_extend_templates_via_llm(
    llm: _StaticRPGLLM,
    n_extra_per_archetype: int,
) -> None:
    """Mutates the module-level STATIC_TEMPLATES by appending LLM-proposed
    templates. Skips silently on failure so the caller can still proceed
    with the hand-written templates."""
    for arch in STATIC_ARCHETYPES:
        existing = [t.get("subdomain", "") for t in STATIC_TEMPLATES[arch]]
        added = 0
        attempts = 0
        while added < n_extra_per_archetype and attempts < n_extra_per_archetype * 3:
            attempts += 1
            try:
                t = _static_llm_propose_template(
                    arch, llm, avoid_subdomains=existing
                )
            except Exception as e:
                print(f"  [llm-template skip] {arch}: {e}")
                continue
            STATIC_TEMPLATES[arch].append(t)
            existing.append(t.get("subdomain", ""))
            added += 1
            print(f"  [llm-template ok] {arch}: '{t.get('subdomain')}'")
        if added < n_extra_per_archetype:
            print(f"  [llm-template] {arch}: only added {added}/{n_extra_per_archetype}")


# ---------------------------------------------------------------------------
# Top-level CLI dispatch (preserves the existing dynamic CLI when --static is
# not passed; otherwise routes to the static sub-CLI).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    if "--static" in _sys.argv:
        _sys.argv.remove("--static")
        _static_main_subcli()
    else:
        main()
