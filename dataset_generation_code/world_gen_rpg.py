#!/usr/bin/env python3
"""Generate RPG v3 story-hidden-cause discovery worlds.

This file intentionally starts the RPG generator over around the new research
target: the scientist should infer an unobserved, story-plausible cause from
indirect observations and targeted tests.  The old v1/v2 generator lives in
``world_gen_rpg_old.py``; this module keeps a narrow compatibility surface so
the existing simulator can still delegate old archetypes if needed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from dataset_generation_code import world_gen_rpg_old as old_rpg
except Exception:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import world_gen_rpg_old as old_rpg  # type: ignore


SCHEMA_VERSION = getattr(old_rpg, "SCHEMA_VERSION", "rpg_v1")
SCHEMA_VERSION_STATIC = "rpg_static_v3"
BENCHMARK_NAME_STATIC = "aced_rpg_static_v3_story_hidden_cause"

STATIC_ARCHETYPES = ["story_hidden_cause_discovery"]
STATIC_DEFAULT_DISTRIBUTION = {"story_hidden_cause_discovery": 1}
STATIC_DEFAULT_ORACLE_N = 50000

REQUIRED_OBSERVED_ROLES = [
    "context_intensity",
    "visible_trigger",
    "exposure_modifier",
    "maintenance_gap",
    "mechanism_proxy_primary",
    "mechanism_proxy_secondary",
    "location_effect",
    "secondary_outcome",
    "outcome",
    "alternative_proxy_primary",
    "alternative_proxy_secondary",
    "alternative_proxy_tertiary",
    "diagnostic_test_signal",
]

REQUIRED_ACTION_ROLES = [
    "targeted_fix_primary",
    "targeted_fix_secondary",
    "diagnostic_test",
    "alternative_fix_primary",
    "symptom_mitigation",
    "partial_reroute",
    "alternative_fix_secondary",
    "distractor_check",
    "weak_buffer",
    "cosmetic_action",
]

STORY_TEMPLATE_PROMPT = """\
You are writing one ACED RPG benchmark world template for scientific latent-cause discovery.

The scientist will see a rich story, observed continuous measurements, and a larger list of possible actions. The true cause must be a hidden, story-plausible object/process/state that is mentioned only casually in the story and is not directly listed as an observed variable or action. The task should feel like real science: the scientist has to hypothesize the hidden cause from semantics, correlations, mechanism proxies, alternative explanations, and targeted experiments.

Create a setting like this kind of problem, but do NOT reuse the gutter/yard-flooding example unless explicitly asked. Good domains include homes, clinics, labs, farms, factories, schools, transit systems, software operations, field ecology, or small organizations. The hidden cause should be ordinary-language understandable but unobserved.

Return exactly one JSON object and no markdown. It must match this schema:
{
  "topic": "short broad domain",
  "subdomain": "short concrete problem",
  "world_slug": "lower_snake_case_unique_slug",
  "hidden_cause": {
    "name": "CamelCaseHiddenContinuousState",
    "plain_name": "ordinary language hidden cause",
    "aliases": ["3-8 natural language aliases"]
  },
  "story": "180-260 words. Mention the relevant hidden object/process/state casually as part of the world, but do not say it is the cause.",
  "observed_variables": [
    {"role": "context_intensity", "name": "CamelCase", "description": "observed continuous clue", "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more ..."}},
    {"role": "visible_trigger", ...},
    {"role": "exposure_modifier", ...},
    {"role": "maintenance_gap", ...},
    {"role": "mechanism_proxy_primary", ...},
    {"role": "mechanism_proxy_secondary", ...},
    {"role": "location_effect", ...},
    {"role": "secondary_outcome", ...},
    {"role": "outcome", ...},
    {"role": "alternative_proxy_primary", ...},
    {"role": "alternative_proxy_secondary", ...},
    {"role": "alternative_proxy_tertiary", ...},
    {"role": "diagnostic_test_signal", ...}
  ],
  "actions": [
    {"role": "targeted_fix_primary", "name": "NeutralCamelCase", "value_type": "dose", "values": ["none", "low", "standard", "high"], "default": "none", "description": "an intervention that (unstated) acts on the hidden cause; name must NOT hint at this"},
    {"role": "targeted_fix_secondary", "value_type": "dose", ...},
    {"role": "diagnostic_test", "value_type": "binary", "values": ["off", "on"], ...},
    {"role": "alternative_fix_primary", "value_type": "dose", ...},
    {"role": "symptom_mitigation", "value_type": "continuous", "min": 0, "max": 100, "default": 0, ...},
    {"role": "partial_reroute", "value_type": "dose", ...},
    {"role": "alternative_fix_secondary", "value_type": "dose", ...},
    {"role": "distractor_check", "value_type": "binary", "values": ["off", "on"], ...},
    {"role": "weak_buffer", "value_type": "dose", ...},
    {"role": "cosmetic_action", "value_type": "dose", ...}
  ],
  "scoring_terms": {
    "mechanism_terms": ["words that indicate the hidden mechanism, not just the surface trigger"],
    "hidden_state_terms": ["words for the hidden state/process being high or impaired"],
    "evidence_groups": {
      "trigger": ["terms for visible trigger"],
      "context": ["terms for background intensity"],
      "mechanism_proxy": ["terms for mechanism proxy measurements"],
      "outcome": ["terms for outcome"],
      "verification": ["terms for decisive action/test"]
    },
    "alternative_groups": {
      "alternative_primary": ["terms for first plausible alternative"],
      "alternative_secondary": ["terms for second plausible alternative"],
      "alternative_tertiary": ["terms for third plausible alternative"]
    },
    "verification_terms": ["diagnostic-test and intervention terms"],
    "action_terms": ["terms for the targeted fixes"]
  },
  "gold_explanation": {
    "description": "one sentence describing the hidden mechanism",
    "evidence": ["3 evidence bullets"],
    "alternatives_ruled_out": ["2-3 alternative explanation bullets"],
    "decisive_test": "one sentence decisive experiment",
    "why_action": "one sentence why the targeted action is best"
  }
}

Hard constraints:
- Include exactly the required roles listed in the schema; do not add extra roles.
- Observed variable names and action names must be NEUTRAL: they must not reveal the
  hidden cause AND must not signal which action targets it. Do not use verbs like
  "Clear", "Fix", "Unblock", "Stop", or "Treat<Cause>"; prefer neutral labels such
  as RegimenA..RegimenF, ProtocolX, or a generic setpoint name. The agent must learn
  each action's effect from data, not from its label.
- Mechanism-proxy variable names must also be neutral (do not embed the mechanism word).
- All observed variables are continuous 0-100.
- Actions are NON-BINARY. Each action declares a "value_type":
  - most actions are "dose" with an ordered "values" list low->high whose first entry
    is the baseline (e.g. ["none","low","standard","high"]);
  - at least one action (use symptom_mitigation) is "continuous" with numeric "min"/"max"
    (a setpoint dial);
  - the diagnostic_test and distractor_check stay "binary" ("off"/"on").
  Choosing the right dose/level is part of the task, so do not make every effect a simple
  on/off; the best level should not trivially be the maximum.
- The core task is latent-variable discovery: the hidden cause is UNOBSERVED (never a
  column, never an action name) and only casually present in the story; the answer is a
  mechanism explanation, not an arm ranking.
- The visible_trigger should be a plausible clue but not itself the true hidden cause.
- The diagnostic_test action should make diagnostic_test_signal much more informative, not directly solve the problem.
- At least three non-targeted actions should be plausible decoys.
"""


def rollout(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for the old dynamic RPG simulator."""
    return old_rpg.rollout(*args, **kwargs)


def _clip100(x: np.ndarray | float) -> np.ndarray:
    return np.clip(x, 0.0, 100.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_id(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _static_intervention_key(intervention: Dict[str, Any]) -> str:
    canon = {
        str(k): str(v)
        for k, v in sorted((intervention or {}).items())
        if str(v) not in ("off", "none", "")
    }
    if not canon:
        return "NoAction"
    return "|".join(f"{k}={v}" for k, v in canon.items())


def _split_terms(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", str(text))
    out: List[str] = []
    for word in words:
        pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).lower().split()
        out.extend(pieces)
    return [w for w in out if len(w) >= 3]


def _template_role_map(template: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    observed = {str(v.get("role")): str(v.get("name")) for v in template.get("observed_variables", [])}
    actions = {str(a.get("role")): str(a.get("name")) for a in template.get("actions", [])}
    missing_obs = [r for r in REQUIRED_OBSERVED_ROLES if r not in observed or not observed[r]]
    missing_actions = [r for r in REQUIRED_ACTION_ROLES if r not in actions or not actions[r]]
    if missing_obs or missing_actions:
        raise ValueError(
            "story-hidden template is missing required role labels: "
            f"observed={missing_obs}, actions={missing_actions}"
        )
    return {"observed": observed, "actions": actions}


def _obs(cfg: Dict[str, Any], role: str) -> str:
    return cfg["role_map"]["observed"][role]


def _act(cfg: Dict[str, Any], role: str) -> str:
    return cfg["role_map"]["actions"][role]


def _hidden_cause_name(cfg: Dict[str, Any]) -> str:
    return str(cfg["template"]["hidden_cause"]["name"])


# ----------------------------------------------------------------------------
# v4: non-binary (dose / continuous) action support.
#
# Every action declares a ``value_type`` (``binary`` | ``dose`` | ``continuous``).
# The mechanism never reads raw values; it reads a *dose fraction* in [0, 1] so
# the same structural equations cover on/off, multi-level doses, and continuous
# setpoints.  Binary knobs map off->0.0 / on->1.0, so any pre-v4 binary world
# reproduces its old behaviour exactly.
# ----------------------------------------------------------------------------

def _action_value_type(spec: Dict[str, Any]) -> str:
    vt = str(spec.get("value_type") or "").strip().lower()
    if vt in ("binary", "dose", "continuous"):
        return vt
    # Infer from the declared values when value_type is omitted.
    values = [str(v) for v in spec.get("values", [])]
    if values == ["off", "on"] or len(values) == 2:
        return "binary"
    if values:
        return "dose"
    return "continuous"


def _dose_fraction(spec: Dict[str, Any], value: Any) -> float:
    """Map a submitted action value to a dose magnitude in [0, 1]."""
    vt = _action_value_type(spec)
    if vt == "continuous":
        lo = float(spec.get("min", 0.0))
        hi = float(spec.get("max", 100.0))
        if hi <= lo:
            return 0.0
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return 0.0
        return float(np.clip((fv - lo) / (hi - lo), 0.0, 1.0))
    values = [str(v) for v in spec.get("values", ["off", "on"])]
    if len(values) <= 1:
        return 0.0
    try:
        idx = values.index(str(value))
    except ValueError:
        idx = 0
    return idx / (len(values) - 1)


def _therapeutic_probe_value(spec: Dict[str, Any], *, dose_saturation: float = 1.0) -> Any:
    """A valid 'apply this knob therapeutically' value for validators/diagnostics.

    For dose knobs this is the level closest to the saturation dose (the level
    the mechanism treats as full therapeutic effect), not necessarily the max,
    so probes reflect the intended gold dose rather than an over-treated one.
    """
    vt = _action_value_type(spec)
    if vt == "continuous":
        lo = float(spec.get("min", 0.0))
        hi = float(spec.get("max", 100.0))
        return lo + dose_saturation * (hi - lo)
    values = [str(v) for v in spec.get("values", ["off", "on"])]
    if len(values) <= 1:
        return values[0] if values else "on"
    if vt == "binary":
        return values[-1]
    # Dose: pick the level whose fraction is closest to dose_saturation.
    target = float(np.clip(dose_saturation, 0.0, 1.0))
    best_idx = min(
        range(len(values)),
        key=lambda i: abs(i / (len(values) - 1) - target),
    )
    # Never probe with the baseline level.
    best_idx = max(best_idx, 1)
    return values[best_idx]


def _action_spec_map(template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """role -> full action spec (with value_type/values/min/max)."""
    out: Dict[str, Dict[str, Any]] = {}
    for action in template.get("actions", []):
        out[str(action.get("role"))] = action
    return out


def _act_dose(cfg: Dict[str, Any], intervention: Dict[str, Any], role: str) -> float:
    """Dose fraction in [0, 1] for the action playing ``role``."""
    specs = cfg.get("action_specs")
    if not specs:  # backward-compat: older worlds stored only the template.
        specs = _action_spec_map(cfg["template"])
    spec = specs[role]
    name = str(spec["name"])
    default = spec.get("default", "off")
    return _dose_fraction(spec, intervention.get(name, default))


def _normalise_story_template(template: Dict[str, Any]) -> Dict[str, Any]:
    template = json.loads(json.dumps(template))
    _template_role_map(template)
    if not re.match(r"^[a-z0-9_]+$", str(template.get("world_slug", ""))):
        template["world_slug"] = _safe_id(str(template.get("world_slug") or template.get("subdomain") or "story_hidden_world"))
    names = [v["name"] for v in template["observed_variables"]]
    if len(names) != len(set(names)):
        raise ValueError("observed variable names must be unique")
    action_names = [a["name"] for a in template["actions"]]
    if len(action_names) != len(set(action_names)):
        raise ValueError("action names must be unique")
    hidden_name = str(template["hidden_cause"]["name"]).lower()
    for item in template["observed_variables"] + template["actions"]:
        if hidden_name and hidden_name in str(item["name"]).lower():
            raise ValueError(f"public name leaks hidden cause name: {item['name']}")
    for action in template["actions"]:
        vt = _action_value_type(action)
        action["value_type"] = vt
        if vt == "continuous":
            action.setdefault("min", 0.0)
            action.setdefault("max", 100.0)
            action.setdefault("values", [])  # continuous has no enumerated levels
            action.setdefault("default", float(action["min"]))
            if float(action["max"]) <= float(action["min"]):
                raise ValueError(f"continuous action {action['name']} needs max > min")
        else:
            action.setdefault("values", ["off", "on"])
            if not isinstance(action["values"], list) or len(action["values"]) < 2:
                raise ValueError(f"action {action['name']} needs >= 2 ordered values")
            action.setdefault("default", str(action["values"][0]))
    template.setdefault("scoring_terms", {})
    template.setdefault("gold_explanation", {})
    return template


def _terms_for_template_item(template: Dict[str, Any], collection: str, role: str) -> List[str]:
    for item in template.get(collection, []):
        if item.get("role") == role:
            text = f"{item.get('name', '')} {item.get('description', '')}"
            terms = _split_terms(text)
            return terms[:8]
    return []


def _story_scoring_terms(template: Dict[str, Any]) -> Dict[str, Any]:
    scoring = dict(template.get("scoring_terms") or {})
    aliases = [str(x).lower() for x in template.get("hidden_cause", {}).get("aliases", [])]
    scoring.setdefault("mechanism_terms", aliases + _split_terms(template.get("hidden_cause", {}).get("plain_name", "")))
    scoring.setdefault("hidden_state_terms", ["hidden", "blocked", "impaired", "stuck", "delayed", "obstructed", "overloaded"])
    scoring.setdefault("evidence_groups", {
        "trigger": _terms_for_template_item(template, "observed_variables", "visible_trigger"),
        "context": _terms_for_template_item(template, "observed_variables", "context_intensity"),
        "mechanism_proxy": (
            _terms_for_template_item(template, "observed_variables", "mechanism_proxy_primary")
            + _terms_for_template_item(template, "observed_variables", "mechanism_proxy_secondary")
            + _terms_for_template_item(template, "observed_variables", "diagnostic_test_signal")
        ),
        "outcome": _terms_for_template_item(template, "observed_variables", "outcome"),
        "verification": (
            _terms_for_template_item(template, "actions", "diagnostic_test")
            + _terms_for_template_item(template, "actions", "targeted_fix_primary")
            + _terms_for_template_item(template, "actions", "targeted_fix_secondary")
        ),
    })
    scoring.setdefault("alternative_groups", {
        "alternative_primary": _terms_for_template_item(template, "observed_variables", "alternative_proxy_primary"),
        "alternative_secondary": _terms_for_template_item(template, "observed_variables", "alternative_proxy_secondary"),
        "alternative_tertiary": _terms_for_template_item(template, "observed_variables", "alternative_proxy_tertiary"),
    })
    scoring.setdefault("verification_terms", scoring["evidence_groups"].get("verification", []))
    scoring.setdefault("action_terms", (
        _terms_for_template_item(template, "actions", "targeted_fix_primary")
        + _terms_for_template_item(template, "actions", "targeted_fix_secondary")
    ))
    return scoring


def _template_yard_flooding() -> Dict[str, Any]:
    return {
        "topic": "Home environment",
        "subdomain": "yard flooding after storms",
        "world_slug": "yard_flooding_hidden_gutter_obstruction",
        "hidden_cause": {
            "name": "GutterObstructionLevel",
            "plain_name": "clogged gutter or blocked downspout",
            "aliases": [
                "clogged gutter",
                "blocked gutter",
                "gutter obstruction",
                "blocked downspout",
                "clogged downspout",
                "leaf-clogged gutter",
                "roof drainage blockage",
                "roof runoff obstruction",
            ],
        },
        "story": (
            "A homeowner is trying to understand why the low back corner of a yard floods after some storms. "
            "The house has a sloped shingle roof, gutters along the back edge, a downspout beside a garden bed, "
            "a stone patio, a compacted play path, and several large deciduous trees that drop leaves unevenly "
            "through the season. Neighbors sometimes worry about runoff from the adjacent driveway, and the yard "
            "does have a gentle slope toward the patio. The owner has storm logs, leaf-fall notes, moisture "
            "readings, and a few maintenance options, but nobody has a direct sensor for the hidden state of the "
            "roof drainage path."
        ),
        "observed_variables": [
            {
                "role": "context_intensity",
                "name": "RainfallAmount",
                "description": "Total rain during the storm window.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more rain"},
            },
            {
                "role": "visible_trigger",
                "name": "RecentLeafFallIntensity",
                "description": "How much fresh leaf litter was observed around the house before the storm.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more fresh leaves"},
            },
            {
                "role": "exposure_modifier",
                "name": "WindTowardBackRoofIndex",
                "description": "How strongly wind pushed leaves and rain toward the back roof edge.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more back-roof exposure"},
            },
            {
                "role": "maintenance_gap",
                "name": "DaysSinceExteriorMaintenance",
                "description": "Days since the homeowner last did exterior water-path maintenance.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "longer since maintenance"},
            },
            {
                "role": "mechanism_proxy_primary",
                "name": "DownspoutDischargeDelay",
                "description": "Delay before strong water discharge appears at the downspout during a storm.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more delayed or pulsed discharge"},
            },
            {
                "role": "mechanism_proxy_secondary",
                "name": "RoofEdgeOverflowScore",
                "description": "Noise/photo score for water spilling over the rear roof edge rather than exiting smoothly.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more overflow"},
            },
            {
                "role": "location_effect",
                "name": "PatioPoolingDepth",
                "description": "Depth of water pooling near the patio after the storm.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "deeper pooling"},
            },
            {
                "role": "secondary_outcome",
                "name": "GardenBedSoilMoisture",
                "description": "Moisture reading near the garden bed and downspout side of the yard.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "wetter soil"},
            },
            {
                "role": "outcome",
                "name": "YardFloodArea",
                "description": "Estimated percent of the low yard corner covered by standing water.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "larger flooded area"},
            },
            {
                "role": "alternative_proxy_primary",
                "name": "SoilCompactionScore",
                "description": "How compacted the surface soil is along the play path and low yard.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more compacted soil"},
            },
            {
                "role": "alternative_proxy_secondary",
                "name": "GroundSlopeTowardPatio",
                "description": "How strongly the local ground slope points water toward the patio.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more slope toward patio"},
            },
            {
                "role": "alternative_proxy_tertiary",
                "name": "NeighborDrivewayRunoffTrace",
                "description": "Trace evidence that water arrived from the neighboring driveway edge.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more neighbor-runoff evidence"},
            },
            {
                "role": "diagnostic_test_signal",
                "name": "FlowTestBackflowScore",
                "description": "Backflow/overflow seen during a roof-edge water-flow test; most informative when that test is run.",
                "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more backflow during test"},
            },
        ],
        "actions": [
            {
                "role": "targeted_fix_primary",
                "name": "ClearRearGutters",
                "values": ["off", "on"],
                "default": "off",
                "description": "Remove debris from the rear gutter run before the storm or flow test.",
            },
            {
                "role": "targeted_fix_secondary",
                "name": "FlushDownspout",
                "values": ["off", "on"],
                "default": "off",
                "description": "Flush the downspout line and check whether water exits freely.",
            },
            {
                "role": "diagnostic_test",
                "name": "RunRoofEdgeFlowTest",
                "values": ["off", "on"],
                "default": "off",
                "description": "Run controlled water along the rear roof edge and measure discharge/backflow behavior.",
            },
            {
                "role": "alternative_fix_primary",
                "name": "AerateCompactedSoil",
                "values": ["off", "on"],
                "default": "off",
                "description": "Aerate the compacted play path and low yard surface.",
            },
            {
                "role": "symptom_mitigation",
                "name": "TemporaryPatioBerm",
                "values": ["off", "on"],
                "default": "off",
                "description": "Place a temporary berm at the patio edge to redirect shallow surface water.",
            },
            {
                "role": "partial_reroute",
                "name": "ExtendDownspoutOutlet",
                "values": ["off", "on"],
                "default": "off",
                "description": "Extend the downspout outlet farther from the garden bed.",
            },
            {
                "role": "alternative_fix_secondary",
                "name": "RegradeLowCorner",
                "values": ["off", "on"],
                "default": "off",
                "description": "Regrade the low corner so water has a gentler path away from the patio.",
            },
            {
                "role": "distractor_check",
                "name": "InspectNeighborRunoff",
                "values": ["off", "on"],
                "default": "off",
                "description": "Inspect the neighboring driveway edge and look for cross-property inflow.",
            },
            {
                "role": "weak_buffer",
                "name": "InstallRainBarrel",
                "values": ["off", "on"],
                "default": "off",
                "description": "Capture some roof runoff near the downspout outlet.",
            },
            {
                "role": "cosmetic_action",
                "name": "ApplyMulchBedCover",
                "values": ["off", "on"],
                "default": "off",
                "description": "Cover the garden bed surface to reduce splash and local soil disturbance.",
            },
        ],
        "scoring_terms": {
            "mechanism_terms": ["gutter", "downspout", "roof drainage", "roof runoff", "drainage path"],
            "hidden_state_terms": ["clog", "block", "obstruct", "blocked", "obstruction", "backflow", "overflow"],
            "evidence_groups": {
                "trigger": ["leaf", "leaves", "leaf-fall", "leaf fall"],
                "context": ["rain", "rainfall", "storm"],
                "mechanism_proxy": ["downspout", "discharge", "delay", "roof", "overflow", "backflow"],
                "outcome": ["flood", "pool", "pooling", "yard", "patio"],
                "verification": ["clear", "flush", "flow test", "clearing", "flushing"],
            },
            "alternative_groups": {
                "alternative_primary": ["soil", "compaction", "aerat"],
                "alternative_secondary": ["slope", "grade", "regrade"],
                "alternative_tertiary": ["neighbor", "driveway"],
            },
            "verification_terms": ["flow test", "water-flow", "clear", "flush", "clearing", "flushing", "backflow"],
            "action_terms": ["clearreargutters", "clear rear gutters", "clear gutters", "flushdownspout", "flush downspout"],
        },
        "gold_explanation": {
            "description": "Fresh leaves and delayed maintenance obstruct the rear roof drainage path, redirecting roof runoff into the low yard.",
            "evidence": [
                "Recent leaf fall predicts flooding beyond rainfall because it increases the hidden drainage obstruction.",
                "Downspout discharge delay and roof-edge overflow move with the hidden obstruction and with flooding.",
                "Clearing gutters or flushing the downspout reduces flooding more directly than soil-only actions.",
            ],
            "alternatives_ruled_out": [
                "Soil compaction contributes but does not explain delayed downspout discharge or roof-edge overflow.",
                "Ground slope explains where water pools, not why flooding spikes after leaf fall.",
                "Neighbor runoff is a weaker contributor in this generated world.",
            ],
            "decisive_test": "Run a roof-edge flow test or clear/flush the rear gutter path and check whether discharge delay, overflow, and flooding drop together.",
            "why_action": "The upstream hidden cause is roof-drainage obstruction, so drainage-path maintenance is the targeted first action.",
        },
    }


def _template_medical_dose() -> Dict[str, Any]:
    """v4 hard example: neutral, non-binary (dose + continuous) medical world.

    A fatigue/anemia clinic where the prevailing theories (diet, thyroid, sleep)
    are decoys and the true cause is an unobserved slow internal blood loss that
    standard panels miss.  The action catalog uses neutral names (RegimenA..F, an
    infusion-rate setpoint, generic protocols) so the answer cannot be read off
    the labels; the agent must learn each knob's effect, and the right *dose*,
    from queried data.  The gold is the targeted regimen at an interior
    ("standard") dose: low is sub-therapeutic, high adds an over-treatment
    side-effect, and the continuous infusion is a palliative that lowers the
    visible outcome reading without touching the hidden cause or the mechanism
    proxies.
    """
    dose_levels = ["none", "low", "standard", "high"]
    three_levels = ["none", "low", "high"]
    return {
        "topic": "Clinical medicine",
        "subdomain": "persistent fatigue and anemia in an outpatient clinic",
        "world_slug": "outpatient_fatigue_occult_blood_loss",
        "hidden_cause": {
            "name": "OccultBloodLossBurden",
            "plain_name": "an unrecognized slow internal blood loss",
            "aliases": [
                "occult bleeding",
                "occult blood loss",
                "slow internal bleeding",
                "chronic blood loss",
                "gastrointestinal blood loss",
                "GI bleeding",
                "hidden internal bleed",
                "ongoing blood loss source",
            ],
        },
        "story": (
            "An outpatient clinic is trying to understand why a group of patients keeps returning with "
            "worsening fatigue and low blood counts despite routine care. The patients vary in age and "
            "activity, several take over-the-counter pain relievers and antiplatelet agents for joints or "
            "the heart, and many have gone a long time since their last gastrointestinal check-up. Some "
            "clinicians blame poor dietary iron, others suspect sluggish thyroid function, and a few point "
            "to chronic poor sleep. The clinic has lab panels, marrow-response and iron-store readings, "
            "questionnaires, and several treatment options, but no panel directly reports what is happening "
            "inside the gut between visits."
        ),
        "observed_variables": [
            {"role": "context_intensity", "name": "MetabolicDemandIndex",
             "description": "Estimated physiological demand load for the patient over the observation window.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "higher demand"}},
            {"role": "visible_trigger", "name": "AntiplateletExposureIndex",
             "description": "Cumulative exposure to antiplatelet and NSAID-type agents before the visit.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more exposure"}},
            {"role": "exposure_modifier", "name": "MucosalVulnerabilityIndex",
             "description": "How vulnerable the patient's GI lining appears to irritation.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more vulnerable"}},
            {"role": "maintenance_gap", "name": "MonthsSinceGiEvaluation",
             "description": "Time since the patient's last gastrointestinal evaluation.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "longer since evaluation"}},
            {"role": "mechanism_proxy_primary", "name": "MarrowCompensationIndex",
             "description": "Marrow erythropoietic response (reticulocyte-style compensation) reading.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "stronger compensation"}},
            {"role": "mechanism_proxy_secondary", "name": "IronStoreDepletionMarker",
             "description": "Composite marker of how depleted the patient's iron stores are.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more depleted"}},
            {"role": "location_effect", "name": "FunctionalCapacityDrop",
             "description": "Measured drop in functional exercise capacity at the visit.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "larger drop"}},
            {"role": "secondary_outcome", "name": "SecondaryFatigueScore",
             "description": "Patient-reported secondary fatigue/quality-of-life score.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "worse fatigue"}},
            {"role": "outcome", "name": "AnemiaSeverityIndex",
             "description": "Composite severity of anemia and fatigue at the visit (primary outcome).",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more severe"}},
            {"role": "alternative_proxy_primary", "name": "DietaryIronAdequacy",
             "description": "How adequate the patient's dietary iron intake appears.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more adequate diet"}},
            {"role": "alternative_proxy_secondary", "name": "ThyroidActivityIndex",
             "description": "Composite index of thyroid activity.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more active thyroid"}},
            {"role": "alternative_proxy_tertiary", "name": "SleepDebtIndex",
             "description": "Accumulated sleep-debt estimate from questionnaires.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more sleep debt"}},
            {"role": "diagnostic_test_signal", "name": "OccultSourceAssaySignal",
             "description": "Assay signal for an internal source; most informative when the source assay is ordered.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "stronger positive signal"}},
        ],
        "actions": [
            {"role": "targeted_fix_primary", "name": "RegimenB", "value_type": "dose",
             "values": dose_levels, "default": "none",
             "description": "A daily oral regimen used for upper-GI mucosal support and protection."},
            {"role": "targeted_fix_secondary", "name": "RegimenD", "value_type": "dose",
             "values": dose_levels, "default": "none",
             "description": "An adjunct oral agent that supports gastrointestinal lining integrity."},
            {"role": "diagnostic_test", "name": "OrderSourceAssay", "value_type": "binary",
             "values": ["off", "on"], "default": "off",
             "description": "Order an internal-source assay panel for this batch of patients."},
            {"role": "alternative_fix_primary", "name": "RegimenA", "value_type": "dose",
             "values": dose_levels, "default": "none",
             "description": "An oral nutritional repletion regimen aimed at dietary deficiency."},
            {"role": "symptom_mitigation", "name": "SupportiveInfusionRate", "value_type": "continuous",
             "min": 0, "max": 100, "default": 0, "oracle_grid": [33, 66, 100],
             "description": "Rate of supportive intravenous repletion infusion given at the visit (0-100)."},
            {"role": "partial_reroute", "name": "RegimenE", "value_type": "dose",
             "values": three_levels, "default": "none",
             "description": "A regimen intended to ease downstream physiological load."},
            {"role": "alternative_fix_secondary", "name": "RegimenF", "value_type": "dose",
             "values": three_levels, "default": "none",
             "description": "A regimen aimed at supporting thyroid-related metabolic activity."},
            {"role": "distractor_check", "name": "SleepProtocolAdjust", "value_type": "binary",
             "values": ["off", "on"], "default": "off",
             "description": "Apply a structured sleep-hygiene protocol adjustment."},
            {"role": "weak_buffer", "name": "MicronutrientAdjunct", "value_type": "dose",
             "values": three_levels, "default": "none",
             "description": "A general micronutrient adjunct supplement."},
            {"role": "cosmetic_action", "name": "WellnessCoachingTier", "value_type": "dose",
             "values": ["none", "basic", "premium"], "default": "none",
             "description": "Enroll patients in a general wellness-coaching tier."},
        ],
        # Interior-optimum dose-response + a strong palliative trap.
        "mechanism_params": {
            "dose_saturation": 0.66,    # "standard" dose captures full therapeutic benefit
            "overtreat_penalty": 1.0,   # dose beyond standard only buys side-effects
            "overtreat_outcome": 14.0,  # high dose worsens the primary outcome
            "overtreat_secondary": 24.0,  # and worsens the secondary outcome more
            "palliative_outcome": 8.0,  # infusion lowers the visible outcome (not the cause)
        },
        "max_intervention_knobs": 3,
        "oracle_max_joint": 2,  # gold is a single targeted regimen at the right dose
        "scoring_terms": {
            "mechanism_terms": [
                "occult", "bleed", "bleeding", "blood loss", "internal bleed",
                "gastrointestinal", "gi bleed", "hemorrhage",
            ],
            "hidden_state_terms": ["bleed", "bleeding", "blood loss", "occult", "ongoing loss", "depletion"],
            "evidence_groups": {
                "trigger": ["antiplatelet", "nsaid", "pain reliever", "anticoagulant"],
                "context": ["demand", "metabolic", "activity"],
                "mechanism_proxy": ["marrow", "reticulocyte", "compensation", "iron", "store", "ferritin", "depletion", "assay"],
                "outcome": ["anemia", "fatigue", "hemoglobin", "blood count"],
                "verification": ["assay", "source", "regimen", "endoscopy", "gi evaluation", "stool"],
            },
            "alternative_groups": {
                "alternative_primary": ["diet", "dietary", "nutrition", "intake"],
                "alternative_secondary": ["thyroid", "metabolic", "hypothyroid"],
                "alternative_tertiary": ["sleep", "sleep debt", "rest"],
            },
            "verification_terms": ["assay", "source assay", "gi evaluation", "endoscopy", "regimen", "stool"],
            "action_terms": ["regimenb", "regimen b", "regimend", "regimen d", "order source assay", "source assay"],
        },
        "gold_explanation": {
            "description": (
                "An unrecognized slow internal (gastrointestinal) blood loss, promoted by antiplatelet/NSAID "
                "exposure and missed because of long gaps since GI evaluation, continually depletes iron and "
                "drives the anemia."
            ),
            "evidence": [
                "Antiplatelet/NSAID exposure predicts anemia severity beyond metabolic demand, consistent with an exposure-driven internal bleed.",
                "Marrow compensation and iron-store depletion move together with the hidden loss and with the outcome.",
                "Ordering the source assay sharpens the occult-source signal, and the mucosal-support regimen lowers the mechanism proxies and outcome together.",
            ],
            "alternatives_ruled_out": [
                "Dietary iron inadequacy is a contributor but does not explain the marrow-compensation and assay signals or why repletion alone does not hold.",
                "Thyroid activity affects fatigue but not the iron-store depletion pattern tied to the bleed.",
                "Sleep debt tracks fatigue reports but not the anemia-severity mechanism.",
            ],
            "decisive_test": (
                "Order the source assay and apply the mucosal-support regimen at a standard dose, then check whether "
                "the occult-source signal, marrow compensation, iron-store depletion, and anemia severity all move together; "
                "note that pushing the supportive infusion lowers the anemia reading without changing the mechanism proxies."
            ),
            "why_action": (
                "The targeted mucosal-support regimen at a standard dose addresses the upstream blood-loss source; "
                "low doses are sub-therapeutic, the highest dose adds side-effects, and the supportive infusion only "
                "masks the reading."
            ),
        },
    }


def _template_factory_drift() -> Dict[str, Any]:
    """v4 hard example (non-medical): neutral, non-binary manufacturing world.

    A precision line shows intermittent defect spikes. The prevailing theories
    (tool wear, ambient humidity, operator variance) are decoys; the true cause
    is an unobserved slow thermal/calibration drift in a shared upstream stage
    that the per-station gauges never report directly. Action names are neutral
    (AdjustmentA..F, a continuous rework-buffer setpoint), so the agent must
    learn each knob's effect and the right correction level from data — and the
    rework buffer is a palliative that lowers the measured defect rate without
    correcting the drift.
    """
    dose_levels = ["none", "low", "standard", "high"]
    three_levels = ["none", "low", "high"]
    return {
        "topic": "Manufacturing operations",
        "subdomain": "intermittent defect spikes on a precision line",
        "world_slug": "precision_line_upstream_thermal_drift",
        "hidden_cause": {
            "name": "UpstreamThermalDrift",
            "plain_name": "an unrecognized upstream thermal/calibration drift",
            "aliases": [
                "thermal drift",
                "calibration drift",
                "upstream drift",
                "temperature creep",
                "heat soak",
                "baseline drift",
                "process setpoint drift",
                "slow calibration shift",
            ],
        },
        "story": (
            "A precision manufacturing line keeps producing intermittent spikes in defects that nobody can "
            "pin down. The line shares a preheat/conditioning stage upstream of several stations, has seen "
            "recent tooling swaps, sits in a bay whose ambient conditions wander across shifts, and runs "
            "different operator crews. Some engineers blame worn tooling, others blame humidity in the bay, "
            "and a few blame operator-to-operator variance. The team has throughput logs, per-station "
            "dimensional and finish readings, maintenance timestamps, and several adjustment options, but "
            "no gauge directly reports the condition of the shared upstream conditioning stage between checks."
        ),
        "observed_variables": [
            {"role": "context_intensity", "name": "LineThroughputLoad",
             "description": "Overall production load on the line during the window.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "higher load"}},
            {"role": "visible_trigger", "name": "AmbientHeatExposureIndex",
             "description": "Cumulative ambient heat exposure reaching the line before the run.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more heat exposure"}},
            {"role": "exposure_modifier", "name": "ThermalCouplingIndex",
             "description": "How strongly ambient conditions couple into the shared upstream stage.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "stronger coupling"}},
            {"role": "maintenance_gap", "name": "HoursSinceCalibrationCheck",
             "description": "Hours since the last calibration check on the line.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "longer since check"}},
            {"role": "mechanism_proxy_primary", "name": "DimensionalVarianceIndex",
             "description": "Spread of finished-part dimensions relative to spec.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more variance"}},
            {"role": "mechanism_proxy_secondary", "name": "SurfaceFinishDeviation",
             "description": "Deviation of surface finish from the reference standard.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more deviation"}},
            {"role": "location_effect", "name": "StationRejectClustering",
             "description": "How clustered rejects are at particular downstream stations.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more clustering"}},
            {"role": "secondary_outcome", "name": "ReworkRateScore",
             "description": "Fraction of units sent to rework at the visit (secondary outcome).",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more rework"}},
            {"role": "outcome", "name": "DefectRateIndex",
             "description": "Composite finished-unit defect rate (primary outcome).",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more defects"}},
            {"role": "alternative_proxy_primary", "name": "ToolWearIndex",
             "description": "Estimated wear level of the active tooling set.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more wear"}},
            {"role": "alternative_proxy_secondary", "name": "AmbientHumidityIndex",
             "description": "Ambient humidity level in the production bay.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more humid"}},
            {"role": "alternative_proxy_tertiary", "name": "OperatorVarianceIndex",
             "description": "Variability attributable to operator-crew differences.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "more variance"}},
            {"role": "diagnostic_test_signal", "name": "DriftProbeSignal",
             "description": "Signal from an upstream-stage probe; most informative when the probe is run.",
             "scale": {"type": "continuous", "min": 0, "max": 100, "higher": "stronger drift signal"}},
        ],
        "actions": [
            {"role": "targeted_fix_primary", "name": "AdjustmentC", "value_type": "dose",
             "values": dose_levels, "default": "none",
             "description": "A correction applied to the shared upstream conditioning stage."},
            {"role": "targeted_fix_secondary", "name": "AdjustmentE", "value_type": "dose",
             "values": dose_levels, "default": "none",
             "description": "A secondary correction to upstream stage settings."},
            {"role": "diagnostic_test", "name": "RunDriftProbe", "value_type": "binary",
             "values": ["off", "on"], "default": "off",
             "description": "Run an instrumented probe on the upstream conditioning stage."},
            {"role": "alternative_fix_primary", "name": "AdjustmentA", "value_type": "dose",
             "values": dose_levels, "default": "none",
             "description": "Replace or refresh the active tooling set."},
            {"role": "symptom_mitigation", "name": "ReworkBufferRate", "value_type": "continuous",
             "min": 0, "max": 100, "default": 0, "oracle_grid": [33, 66, 100],
             "description": "Rate of added downstream rework/sorting applied at the line (0-100)."},
            {"role": "partial_reroute", "name": "AdjustmentF", "value_type": "dose",
             "values": three_levels, "default": "none",
             "description": "Reroute part of the flow to ease downstream load."},
            {"role": "alternative_fix_secondary", "name": "AdjustmentB", "value_type": "dose",
             "values": three_levels, "default": "none",
             "description": "Tighten humidity control in the production bay."},
            {"role": "distractor_check", "name": "OperatorAudit", "value_type": "binary",
             "values": ["off", "on"], "default": "off",
             "description": "Audit operator-crew procedures for the line."},
            {"role": "weak_buffer", "name": "AdjustmentD", "value_type": "dose",
             "values": three_levels, "default": "none",
             "description": "A minor general adjustment to station settings."},
            {"role": "cosmetic_action", "name": "CosmeticPolishTier", "value_type": "dose",
             "values": ["none", "basic", "premium"], "default": "none",
             "description": "Apply a cosmetic finishing/polish tier to shipped units."},
        ],
        "mechanism_params": {
            "dose_saturation": 0.66,
            "overtreat_penalty": 1.0,
            "overtreat_outcome": 14.0,
            "overtreat_secondary": 24.0,
            "palliative_outcome": 8.0,
        },
        "max_intervention_knobs": 3,
        "oracle_max_joint": 2,
        "scoring_terms": {
            "mechanism_terms": [
                "thermal drift", "calibration drift", "upstream drift", "temperature creep",
                "heat soak", "setpoint drift", "process drift", "calibration shift",
            ],
            "hidden_state_terms": ["drift", "creep", "miscalibration", "shift", "uncalibrated", "out of calibration"],
            "evidence_groups": {
                "trigger": ["ambient heat", "heat exposure", "temperature", "preheat"],
                "context": ["throughput", "load", "volume"],
                "mechanism_proxy": ["dimensional", "variance", "finish", "deviation", "drift probe", "upstream"],
                "outcome": ["defect", "defect rate", "scrap", "reject"],
                "verification": ["probe", "calibrate", "recalibration", "adjustment", "upstream"],
            },
            "alternative_groups": {
                "alternative_primary": ["tool", "tooling", "wear"],
                "alternative_secondary": ["humidity", "ambient", "moisture"],
                "alternative_tertiary": ["operator", "crew", "shift"],
            },
            "verification_terms": ["drift probe", "probe", "recalibrate", "calibration", "upstream adjustment"],
            "action_terms": ["adjustmentc", "adjustment c", "adjustmente", "adjustment e", "run drift probe", "drift probe"],
        },
        "gold_explanation": {
            "description": (
                "A slow, unobserved thermal/calibration drift in the shared upstream conditioning stage, "
                "amplified by ambient heat exposure and long gaps since calibration, drives the intermittent "
                "defect spikes downstream."
            ),
            "evidence": [
                "Ambient heat exposure predicts the defect rate beyond throughput load, consistent with a heat-driven upstream drift.",
                "Dimensional variance and surface-finish deviation move together with the hidden drift and with the defect rate.",
                "Running the upstream drift probe sharpens the drift signal, and correcting the upstream stage lowers the mechanism proxies and defect rate together.",
            ],
            "alternatives_ruled_out": [
                "Tool wear contributes but does not explain the dimensional-variance pattern or why replacing tooling does not hold the defect rate down.",
                "Ambient humidity tracks the bay but not the upstream-stage drift signal.",
                "Operator variance shifts some readings but not the calibration-linked mechanism proxies.",
            ],
            "decisive_test": (
                "Run the drift probe and apply the upstream correction at a standard level, then check whether the drift signal, "
                "dimensional variance, surface-finish deviation, and defect rate all drop together; note that adding rework buffer "
                "lowers the measured defect rate without changing the mechanism proxies."
            ),
            "why_action": (
                "The upstream correction at a standard level addresses the drift at its source; low levels are insufficient, the "
                "highest level over-corrects and adds side-effects, and the rework buffer only masks the measured defect rate."
            ),
        },
    }


_BUILTIN_TEMPLATES = {
    "medical_dose": _template_medical_dose,
    "factory_drift": _template_factory_drift,
    "yard_flooding": _template_yard_flooding,
}

# Default offline (non-LLM) mix: two hard, neutral, non-binary worlds in
# different domains so a builtin dataset is not all one topic.
DEFAULT_BUILTIN_TEMPLATES = ["medical_dose", "factory_drift"]


def _story_hidden_default_params(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {
        "trigger_to_hidden": rng.uniform(0.58, 0.70),
        "exposure_to_hidden": rng.uniform(0.14, 0.22),
        "maintenance_to_hidden": rng.uniform(0.22, 0.32),
        "context_to_outcome": rng.uniform(0.12, 0.18),
        "hidden_to_outcome_gain": rng.uniform(0.70, 0.86),
        "alt_primary_to_outcome": rng.uniform(0.05, 0.09),
        "alt_secondary_to_location": rng.uniform(0.05, 0.09),
        "alt_tertiary_to_outcome": rng.uniform(0.03, 0.07),
        "noise_sd": rng.uniform(3.0, 4.5),
        # ---- v4 dose-response knobs (defaults reproduce binary v3 behaviour) ----
        # Targeted-fix dose-response: linear & unsaturated (saturation==1.0),
        # no over-treatment penalty.  Medical templates override these so the
        # gold dose is an interior level rather than "max everything".
        "dose_saturation": 1.0,
        "overtreat_penalty": 0.0,
        "overtreat_outcome": 0.0,
        "overtreat_secondary": 0.0,
        "targeted_primary_base": 62.0,      # v3: hidden_cause -= 62 + 0.18*trigger
        "targeted_primary_trigger": 0.18,
        "targeted_secondary_base": 48.0,    # v3: hidden_cause -= 48
        "palliative_outcome": 3.0,          # v3: symptom_mitigation -= 3 on outcome
        "cosmetic_outcome": 3.0,            # v3: cosmetic_action -= 3 on outcome
    }


def _sample_story_hidden(cfg: Dict[str, Any], n: int, *, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    context = _clip100(rng.gamma(shape=3.0, scale=15.0, size=n) + rng.normal(0, 4, n))
    trigger = _clip100(rng.beta(2.0, 2.2, size=n) * 100.0)
    exposure = _clip100(rng.normal(50, 22, n))
    maintenance = _clip100(rng.gamma(shape=2.4, scale=18.0, size=n))
    alt_primary = _clip100(rng.normal(48, 18, n))
    alt_secondary = _clip100(rng.normal(52, 16, n))
    alt_tertiary = _clip100(rng.beta(1.5, 5.0, size=n) * 100.0)

    latent_raw = (
        p["trigger_to_hidden"] * trigger
        + p["exposure_to_hidden"] * exposure
        + p["maintenance_to_hidden"] * maintenance
        + rng.normal(0, 8.0, n)
        - 18.0
    )
    hidden_cause = _clip100(latent_raw)
    upstream_load = _clip100(0.85 * context + 0.22 * exposure + rng.normal(0, 5, n))
    antecedent_saturation = _clip100(0.38 * context + 0.18 * alt_primary + rng.normal(0, 8, n))
    return {
        _obs(cfg, "context_intensity"): context,
        _obs(cfg, "visible_trigger"): trigger,
        _obs(cfg, "exposure_modifier"): exposure,
        _obs(cfg, "maintenance_gap"): maintenance,
        _obs(cfg, "alternative_proxy_primary"): alt_primary,
        _obs(cfg, "alternative_proxy_secondary"): alt_secondary,
        _obs(cfg, "alternative_proxy_tertiary"): alt_tertiary,
        _hidden_cause_name(cfg): hidden_cause,
        "_UpstreamLoad": upstream_load,
        "AntecedentSaturation": antecedent_saturation,
    }


def _apply_story_hidden(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = cfg["parameters"]
    n = len(hidden[_obs(cfg, "context_intensity")])
    hidden_cause = hidden[_hidden_cause_name(cfg)].astype(float).copy()
    alt_primary = hidden[_obs(cfg, "alternative_proxy_primary")].astype(float).copy()
    alt_secondary = hidden[_obs(cfg, "alternative_proxy_secondary")].astype(float).copy()
    context = hidden[_obs(cfg, "context_intensity")].astype(float)
    trigger = hidden[_obs(cfg, "visible_trigger")].astype(float)
    upstream_load = hidden["_UpstreamLoad"].astype(float)
    alt_tertiary = hidden[_obs(cfg, "alternative_proxy_tertiary")].astype(float)

    # Dose fractions in [0, 1] for every action (binary on/off -> 1.0/0.0).
    d_primary = _act_dose(cfg, intervention, "targeted_fix_primary")
    d_secondary = _act_dose(cfg, intervention, "targeted_fix_secondary")
    d_alt_primary = _act_dose(cfg, intervention, "alternative_fix_primary")
    d_alt_secondary = _act_dose(cfg, intervention, "alternative_fix_secondary")
    d_reroute = _act_dose(cfg, intervention, "partial_reroute")
    d_symptom = _act_dose(cfg, intervention, "symptom_mitigation")
    d_weak = _act_dose(cfg, intervention, "weak_buffer")
    d_diag = _act_dose(cfg, intervention, "diagnostic_test")
    d_distract = _act_dose(cfg, intervention, "distractor_check")
    d_cosmetic = _act_dose(cfg, intervention, "cosmetic_action")

    # Saturating dose-response on the two targeted fixes: benefit plateaus at the
    # saturation dose, and dose beyond it only buys an over-treatment penalty.
    # With dose_saturation == 1.0 and overtreat_penalty == 0.0 (the defaults)
    # this is the original linear, no-penalty binary behaviour.
    sat = max(1e-6, float(p.get("dose_saturation", 1.0)))
    benefit_primary = min(1.0, d_primary / sat)
    benefit_secondary = min(1.0, d_secondary / sat)
    overtreat = float(p.get("overtreat_penalty", 0.0)) * (
        max(0.0, d_primary - sat) + max(0.0, d_secondary - sat)
    )

    hidden_cause = _clip100(
        hidden_cause
        - (p.get("targeted_primary_base", 62.0) + p.get("targeted_primary_trigger", 0.18) * trigger)
        * benefit_primary
    )
    hidden_cause = _clip100(hidden_cause - p.get("targeted_secondary_base", 48.0) * benefit_secondary)
    alt_primary = _clip100(alt_primary - 16.0 * d_alt_primary)
    alt_secondary = _clip100(alt_secondary - 14.0 * d_alt_secondary)

    hidden_mediated_load = (
        p["hidden_to_outcome_gain"] * (hidden_cause / 100.0) * upstream_load
        - 10.0 * d_reroute
        - 6.0 * d_weak
    )
    hidden_mediated_load = _clip100(hidden_mediated_load)

    outcome = (
        p["context_to_outcome"] * context
        + hidden_mediated_load
        + p["alt_primary_to_outcome"] * alt_primary
        + p["alt_secondary_to_location"] * alt_secondary
        + p["alt_tertiary_to_outcome"] * alt_tertiary
        - p.get("palliative_outcome", 3.0) * d_symptom  # palliative: lowers reading, not the cause
        - p.get("cosmetic_outcome", 3.0) * d_cosmetic
        + p.get("overtreat_outcome", 0.0) * overtreat   # over-treatment hurts a little
        + rng.normal(0, p["noise_sd"], n)
    )
    outcome = _clip100(outcome)

    mechanism_primary = _clip100(
        0.72 * hidden_cause
        + 0.18 * context
        - 18.0 * d_reroute
        - 20.0 * benefit_secondary
        + rng.normal(0, 5.0, n)
    )
    mechanism_secondary = _clip100(
        0.62 * hidden_cause
        + 0.28 * upstream_load
        - 30.0 * benefit_primary
        + rng.normal(0, 5.5, n)
    )
    location_effect = _clip100(
        0.56 * outcome + 0.10 * alt_secondary + 0.22 * hidden_mediated_load
        - 5.0 * d_symptom + rng.normal(0, 4, n)
    )
    secondary_outcome = _clip100(
        0.45 * outcome + 0.28 * hidden_mediated_load + 0.20 * context
        + p.get("overtreat_secondary", 0.0) * overtreat  # over-treatment shows up here
        + rng.normal(0, 4, n)
    )
    diagnostic_signal = _clip100(
        (0.82 * hidden_cause + 0.22 * upstream_load + rng.normal(0, 4, n))
        if d_diag > 0.5
        else (0.20 * hidden_cause + 0.08 * upstream_load + rng.normal(0, 8, n))
    )
    alt_tertiary_trace = _clip100(alt_tertiary + 10.0 * d_distract + rng.normal(0, 3.0, n))
    return {
        _hidden_cause_name(cfg): hidden_cause,
        "_HiddenMediatedLoad": hidden_mediated_load,
        _obs(cfg, "outcome"): outcome,
        _obs(cfg, "mechanism_proxy_primary"): mechanism_primary,
        _obs(cfg, "mechanism_proxy_secondary"): mechanism_secondary,
        _obs(cfg, "location_effect"): location_effect,
        _obs(cfg, "secondary_outcome"): secondary_outcome,
        _obs(cfg, "diagnostic_test_signal"): diagnostic_signal,
        _obs(cfg, "alternative_proxy_tertiary"): alt_tertiary_trace,
        _obs(cfg, "alternative_proxy_primary"): alt_primary,
        _obs(cfg, "alternative_proxy_secondary"): alt_secondary,
    }


def _observe_story_hidden(
    cfg: Dict[str, Any],
    hidden: Dict[str, np.ndarray],
    outcomes: Dict[str, np.ndarray],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs: Dict[str, np.ndarray] = {}
    source = {**hidden, **outcomes}
    for name in measurements:
        if name not in source:
            raise ValueError(f"unknown story-hidden measurement {name!r}")
        arr = np.asarray(source[name], dtype=float)
        low_noise = {
            _obs(cfg, "context_intensity"),
            _obs(cfg, "visible_trigger"),
            _obs(cfg, "exposure_modifier"),
            _obs(cfg, "maintenance_gap"),
        }
        medium_noise = {
            _obs(cfg, "alternative_proxy_primary"),
            _obs(cfg, "alternative_proxy_secondary"),
            _obs(cfg, "alternative_proxy_tertiary"),
        }
        if name in low_noise:
            noise = 1.5
        elif name in medium_noise:
            noise = 2.5
        else:
            noise = 3.5
        obs[name] = _clip100(arr + rng.normal(0, noise, len(arr)))
    return obs


def _static_sample_hidden(cfg: Dict[str, Any], n: int, *, seed: int) -> Dict[str, Any]:
    if cfg.get("archetype") == "story_hidden_cause_discovery":
        return _sample_story_hidden(cfg, n, seed=seed)
    return old_rpg._static_sample_hidden(cfg, n, seed=seed)  # noqa: SLF001


def _static_apply(
    cfg: Dict[str, Any],
    hidden: Dict[str, Any],
    intervention: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, Any]:
    if cfg.get("archetype") == "story_hidden_cause_discovery":
        return _apply_story_hidden(cfg, hidden, intervention, seed=seed)
    return old_rpg._static_apply(cfg, hidden, intervention, seed=seed)  # noqa: SLF001


def _static_observe(
    cfg: Dict[str, Any],
    hidden: Dict[str, Any],
    outcomes: Dict[str, Any],
    measurements: List[str],
    *,
    seed: int,
) -> Dict[str, Any]:
    if cfg.get("archetype") == "story_hidden_cause_discovery":
        return _observe_story_hidden(cfg, hidden, outcomes, measurements, seed=seed)
    return old_rpg._static_observe(cfg, hidden, outcomes, measurements, seed=seed)  # noqa: SLF001


def _static_assignment(cfg: Dict[str, Any], hidden: Dict[str, Any], *, seed: int) -> Dict[str, np.ndarray]:
    if cfg.get("archetype") == "story_hidden_cause_discovery":
        return {}
    return old_rpg._static_assignment(cfg, hidden, seed=seed)  # noqa: SLF001


def _static_utility_from_outcomes(cfg: Dict[str, Any], outcomes: Dict[str, Any]) -> np.ndarray:
    if cfg.get("archetype") == "story_hidden_cause_discovery":
        return -np.asarray(outcomes[_obs(cfg, "outcome")], dtype=float)
    return old_rpg._static_utility_from_outcomes(cfg, outcomes)  # noqa: SLF001


def _action_oracle_levels(action: Dict[str, Any]) -> List[Any]:
    """Non-baseline levels the oracle sweeps for one action."""
    vt = _action_value_type(action)
    if vt == "continuous":
        grid = action.get("oracle_grid") or [33.0, 66.0, 100.0]
        lo = float(action.get("min", 0.0))
        return [float(v) for v in grid if float(v) > lo]
    values = [str(v) for v in action.get("values", ["off", "on"])]
    return values[1:]  # drop the baseline / off / none level


def _candidate_story_interventions(template: Dict[str, Any], *, max_knobs: int = 3) -> List[Dict[str, Any]]:
    """Enumerate baseline + every level-combination of up to ``max_knobs`` actions.

    For all-binary worlds each action contributes a single "on" level, so this
    reduces exactly to the v3 subset enumeration (e.g. 176 candidates for 10
    binary actions at max_knobs=3).  For dose / continuous actions it sweeps the
    declared value grids, which is why ``max_knobs`` is kept small for those
    worlds to bound oracle cost.
    """
    action_levels = {a["name"]: _action_oracle_levels(a) for a in template["actions"]}
    names = [a["name"] for a in template["actions"] if action_levels[a["name"]]]
    candidates: List[Dict[str, Any]] = [{}]
    for size in range(1, max(1, int(max_knobs)) + 1):
        for combo in itertools.combinations(names, size):
            level_lists = [action_levels[name] for name in combo]
            for level_combo in itertools.product(*level_lists):
                candidates.append({name: level for name, level in zip(combo, level_combo)})
    return candidates


def _score_intervention(cfg: Dict[str, Any], intervention: Dict[str, str], *, n: int, seed: int) -> float:
    hidden = _static_sample_hidden(cfg, n, seed=seed)
    outcomes = _static_apply(cfg, hidden, intervention, seed=seed + 11)
    return float(np.mean(_static_utility_from_outcomes(cfg, outcomes)))


def _build_story_hidden_world(seed: int, *, oracle_n: int, template: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    template = _normalise_story_template(template or _template_yard_flooding())
    role_map = _template_role_map(template)
    # Template-supplied mechanism overrides win over the seed-sampled defaults.
    params = {**_story_hidden_default_params(seed), **dict(template.get("mechanism_params") or {})}
    # Agent-facing joint-action cap (how many knobs an answer/query may set).
    max_knobs = int(template.get("max_intervention_knobs", 3))
    # Oracle joint cap — kept small for dose/continuous worlds to bound cost,
    # since the gold here is a single targeted regimen at the right dose.
    oracle_max_joint = int(template.get("oracle_max_joint", max_knobs))
    cfg = {
        "archetype": "story_hidden_cause_discovery",
        "template": template,
        "role_map": role_map,
        "action_specs": _action_spec_map(template),
        "parameters": params,
        "seed": seed,
        "world_seed": seed,
    }
    action_scores = []
    candidates = _candidate_story_interventions(template, max_knobs=oracle_max_joint)
    print(
        f"[oracle] seed={seed} slug={template['world_slug']} scoring "
        f"{len(candidates)} joint-action candidates with oracle_n={oracle_n}",
        flush=True,
    )
    for idx, intervention in enumerate(candidates):
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(candidates):
            print(
                f"[oracle] seed={seed} candidate {idx + 1}/{len(candidates)}",
                flush=True,
            )
        utility = _score_intervention(cfg, intervention, n=oracle_n, seed=seed + 7001 + idx * 101)
        action_scores.append({
            "intervention": intervention,
            "intervention_key": _static_intervention_key(intervention),
            "expected_utility": utility,
        })
    action_scores.sort(key=lambda x: x["expected_utility"], reverse=True)

    hidden_probe = _static_sample_hidden(cfg, 6000, seed=seed + 9001)
    base_out = _static_apply(cfg, hidden_probe, {}, seed=seed + 9011)
    sat = float(params.get("dose_saturation", 1.0))
    target_primary = {
        _act(cfg, "targeted_fix_primary"): _therapeutic_probe_value(
            cfg["action_specs"]["targeted_fix_primary"], dose_saturation=sat
        )
    }
    alt_primary_action = {
        _act(cfg, "alternative_fix_primary"): _therapeutic_probe_value(
            cfg["action_specs"]["alternative_fix_primary"], dose_saturation=sat
        )
    }
    target_out = _static_apply(cfg, hidden_probe, target_primary, seed=seed + 9021)
    alt_out = _static_apply(cfg, hidden_probe, alt_primary_action, seed=seed + 9031)
    corr_trigger_outcome = float(np.corrcoef(hidden_probe[_obs(cfg, "visible_trigger")], base_out[_obs(cfg, "outcome")])[0, 1])
    corr_context_outcome = float(np.corrcoef(hidden_probe[_obs(cfg, "context_intensity")], base_out[_obs(cfg, "outcome")])[0, 1])
    corr_hidden_mechanism = float(np.corrcoef(hidden_probe[_hidden_cause_name(cfg)], base_out[_obs(cfg, "mechanism_proxy_primary")])[0, 1])
    targeted_gain = float(np.mean(base_out[_obs(cfg, "outcome")] - target_out[_obs(cfg, "outcome")]))
    alternative_gain = float(np.mean(base_out[_obs(cfg, "outcome")] - alt_out[_obs(cfg, "outcome")]))

    question = (
        "Do not give a black-box prediction. Use the available records and targeted tests to explain what hidden "
        "story-plausible cause best accounts for the outcome pattern. Your answer should name the unobserved cause "
        "in ordinary language, give evidence from queried data, rule out plausible alternatives, "
        "state a decisive test, and recommend the next action (including its dose/level where applicable). "
        "The action names are intentionally neutral and do NOT reveal which one targets the cause; "
        "decide that from the data you collect, not from the labels."
    )
    # The action names are deliberately uninformative, so the agent must learn
    # the effect of each knob (and the right dose level) from queried data.
    value_types = sorted({_action_value_type(a) for a in template["actions"]})
    visible = {
        "story": template["story"],
        "question": question,
        "observed_variables": template["observed_variables"],
        "intervenable_variables": template["actions"],
        "allowed_measurements": [v["name"] for v in template["observed_variables"]],
        "allowed_query_modes": ["observational_sample", "interventional_sample", "inspect_unit"],
        "answer_schema": "latent_cause_hypothesis",
        "max_intervention_knobs": max_knobs,
        "public_action_space": {
            "n_actions": len(template["actions"]),
            "max_joint_actions_per_query": max_knobs,
            "action_value_types": value_types,
            "n_oracle_scored_candidates": len(candidates),
            "note": (
                "Each interventional query may set any subset of public actions up to the cap. "
                "Actions are non-binary: most take an ordered dose level and some take a continuous "
                "setpoint, so choosing the right level is part of the task. Action names are neutral "
                "and do not indicate which knob addresses the hidden cause."
            ),
        },
        "experiment_budget": {
            "max_queries": 8,
            "max_total_samples": 14000,
            "max_samples_per_query": 4500,
            "max_units_per_query": 450,
            "max_measurements_per_query": 4,
            "sample_accounting": "cells",
        },
        "discovery_protocol": {
            "task_goal": "Infer a story-plausible hidden cause, not just an observed predictor.",
            "important_warning": (
                "Observed measurements may be clues, downstream symptoms, or competing alternatives; "
                "none of them is automatically the hidden cause. Action names are neutral and do not "
                "tell you which knob targets the cause, and the best level is not always the maximum."
            ),
            "answer_must_include": [
                "latent_hypothesis",
                "evidence",
                "alternatives_ruled_out",
                "decisive_test",
                "action_plan",
            ],
        },
    }
    gold_explanation = template.get("gold_explanation") or {}
    scoring_terms = _story_scoring_terms(template)
    targeted_actions = [
        _act(cfg, "diagnostic_test"),
        _act(cfg, "targeted_fix_primary"),
        _act(cfg, "targeted_fix_secondary"),
    ]
    mechanism_measurements = [
        _obs(cfg, "mechanism_proxy_primary"),
        _obs(cfg, "mechanism_proxy_secondary"),
        _obs(cfg, "diagnostic_test_signal"),
        _obs(cfg, "outcome"),
    ]
    gold = {
        "answer_schema": "latent_cause_hypothesis",
        "latent_hypothesis": {
            "name": template["hidden_cause"]["plain_name"],
            "description": gold_explanation.get(
                "description",
                "A story-implied hidden state links the visible trigger to the mechanism proxies and the outcome.",
            ),
            "accepted_aliases": template["hidden_cause"]["aliases"],
        },
        "evidence": gold_explanation.get("evidence", [
            f"{_obs(cfg, 'visible_trigger')} predicts {_obs(cfg, 'outcome')} beyond {_obs(cfg, 'context_intensity')}.",
            f"{_obs(cfg, 'mechanism_proxy_primary')} and {_obs(cfg, 'mechanism_proxy_secondary')} move with the hidden cause and the outcome.",
            f"{_act(cfg, 'targeted_fix_primary')} or {_act(cfg, 'targeted_fix_secondary')} reduces the outcome more directly than alternative-only actions.",
        ]),
        "alternatives_ruled_out": gold_explanation.get("alternatives_ruled_out", [
            f"{_obs(cfg, 'alternative_proxy_primary')} contributes but does not explain the mechanism proxies.",
            f"{_obs(cfg, 'alternative_proxy_secondary')} affects where the outcome appears, not why the trigger matters.",
            f"{_obs(cfg, 'alternative_proxy_tertiary')} is a weaker contributor in this generated world.",
        ]),
        "decisive_test": gold_explanation.get(
            "decisive_test",
            f"Run {_act(cfg, 'diagnostic_test')} or intervene on {_act(cfg, 'targeted_fix_primary')} and check whether mechanism proxies and outcome drop together.",
        ),
        "action_plan": {
            "do_now": [_act(cfg, "targeted_fix_primary"), _act(cfg, "targeted_fix_secondary")],
            "avoid_first": [_act(cfg, "alternative_fix_secondary")],
            "why": gold_explanation.get(
                "why_action",
                "The targeted actions address the hidden upstream cause rather than only a surface alternative.",
            ),
        },
        "scoring_terms": scoring_terms,
        "trajectory_requirements": {
            "targeted_actions": targeted_actions,
            "mechanism_measurements": mechanism_measurements,
            "context_measurements": [_obs(cfg, "context_intensity"), _obs(cfg, "visible_trigger")],
            "outcome_measurement": _obs(cfg, "outcome"),
        },
        "oracle_best_intervention": action_scores[0]["intervention"],
        "oracle_best_expected_utility": action_scores[0]["expected_utility"],
    }
    validators = {
        "accepted": bool(
            corr_trigger_outcome > 0.35
            and corr_context_outcome > 0.25
            and corr_hidden_mechanism > 0.65
            and targeted_gain > alternative_gain + 4.0
        ),
        "checks": {
            "corr_trigger_outcome": corr_trigger_outcome,
            "corr_context_outcome": corr_context_outcome,
            "corr_hidden_cause_mechanism_proxy": corr_hidden_mechanism,
            "targeted_fix_gain_over_baseline": targeted_gain,
            "alternative_fix_gain_over_baseline": alternative_gain,
            "targeted_beats_alternative_margin": targeted_gain - alternative_gain,
            "n_oracle_joint_action_candidates": len(candidates),
        },
    }
    print(
        f"[validator] seed={seed} accepted={validators['accepted']} "
        f"checks={json.dumps(validators['checks'], sort_keys=True)}",
        flush=True,
    )
    world_id = f"rpg_static_story_hidden_cause_{template['world_slug']}_seed{seed}"
    return {
        "schema_version": SCHEMA_VERSION_STATIC,
        "benchmark": BENCHMARK_NAME_STATIC,
        "meta": {
            "world_id": world_id,
            "seed": seed,
            "archetype": "story_hidden_cause_discovery",
            "topic": template["topic"],
            "subdomain": template["subdomain"],
        },
        "visible": visible,
        "hidden": {
            "simulator_config": cfg,
            "latent_variables": {
                template["hidden_cause"]["name"]: (
                    f"Continuous hidden state for {template['hidden_cause']['plain_name']}; "
                    "never directly visible to the scientist."
                ),
            },
            "diagnostics": validators["checks"],
        },
        "oracle": {
            "gold_answer": gold,
            "action_scores": action_scores,
            "oracle_n_units": oracle_n,
            "oracle_tolerance": 0.0,
        },
        "validators": validators,
        "questions": [
            {
                "id": 0,
                "question_type": "rpg_story_hidden_cause_discovery",
                "question": question,
                "answer_schema": "latent_cause_hypothesis",
                "answer": gold,
            }
        ],
    }


def static_rpg_generate_world(
    *,
    archetype: str,
    seed: int,
    oracle_n: int = STATIC_DEFAULT_ORACLE_N,
    max_attempts: int = 1,
    template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if archetype != "story_hidden_cause_discovery":
        raise ValueError(f"unknown v3 RPG archetype {archetype!r}")
    last_world: Optional[Dict[str, Any]] = None
    for attempt in range(max_attempts):
        world = _build_story_hidden_world(seed + attempt * 101, oracle_n=oracle_n, template=template)
        last_world = world
        if world["validators"]["accepted"]:
            return world
    assert last_world is not None
    return last_world


def static_rpg_generate_dataset(
    *,
    outdir: str,
    distribution: Optional[Dict[str, int]] = None,
    start_seed: int = 5000,
    oracle_n: int = STATIC_DEFAULT_ORACLE_N,
    max_attempts_per_world: int = 5,
    templates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    distribution = distribution or dict(STATIC_DEFAULT_DISTRIBUTION)
    templates = templates or [_BUILTIN_TEMPLATES[name]() for name in DEFAULT_BUILTIN_TEMPLATES]
    os.makedirs(outdir, exist_ok=True)
    worlds = []
    idx = 0
    for archetype, count in distribution.items():
        for _ in range(int(count)):
            seed = start_seed + idx * 101
            template = templates[idx % len(templates)]
            print(
                f"[world] {idx + 1}/{sum(int(v) for v in distribution.values())} "
                f"archetype={archetype} seed={seed} template={template.get('world_slug', '(missing slug)')}",
                flush=True,
            )
            world = static_rpg_generate_world(
                archetype=archetype,
                seed=seed,
                oracle_n=oracle_n,
                max_attempts=max_attempts_per_world,
                template=template,
            )
            filename = f"world_{world['meta']['world_id']}.json"
            path = os.path.join(outdir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(world, f, indent=2)
            worlds.append({
                "path": path,
                "world_id": world["meta"]["world_id"],
                "archetype": world["meta"]["archetype"],
                "accepted": world["validators"]["accepted"],
                "checks": world["validators"]["checks"],
            })
            print(f"[ok] {filename}" if world["validators"]["accepted"] else f"[warn] validator not accepted: {filename}")
            idx += 1
    manifest = {
        "schema_version": SCHEMA_VERSION_STATIC,
        "benchmark": BENCHMARK_NAME_STATIC,
        "generated": len(worlds),
        "distribution": distribution,
        "worlds": worlds,
    }
    manifest_path = os.path.join(outdir, "manifest_rpg_static_v3.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("LLM template response did not contain a JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM template response must be a JSON object")
    return data


def _load_template_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "templates" in data:
        data = data["templates"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("--template-json must contain a template object, a list, or {'templates': [...]}")
    templates = [_normalise_story_template(t) for t in data]
    if not templates:
        raise ValueError("--template-json did not contain any templates")
    return templates


def _build_template_llm(args: argparse.Namespace) -> Any:
    if args.llm_backend != "bedrock":
        raise ValueError(f"unsupported template LLM backend {args.llm_backend!r}")
    framework_dir = Path(__file__).resolve().parents[1] / "framework_code"
    if str(framework_dir) not in sys.path:
        sys.path.insert(0, str(framework_dir))
    from bedrock_llm import BedrockLLM  # type: ignore

    return BedrockLLM(
        model_id=args.llm_model,
        region_name=args.llm_region,
        temperature=args.llm_temperature,
        max_new_tokens=args.llm_max_tokens,
    )


def _generate_llm_templates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print(
        f"[template] building {args.llm_template_count} LLM template(s) "
        f"with {args.llm_model}",
        flush=True,
    )
    llm = _build_template_llm(args)
    templates = []
    system = "You create rigorous JSON benchmark templates for latent scientific discovery tasks."
    for idx in range(int(args.llm_template_count)):
        print(f"[template] requesting template {idx + 1}/{args.llm_template_count}", flush=True)
        user = (
            STORY_TEMPLATE_PROMPT
            + "\n\n"
            + f"Template index: {idx}. Choose a fresh setting and avoid reusing previous examples."
        )
        raw = llm.generate(system, user, max_new_tokens=args.llm_max_tokens)
        print(f"[template] received template {idx + 1}/{args.llm_template_count}", flush=True)
        template = _normalise_story_template(_extract_json_object(raw))
        print(
            f"[template] accepted {idx + 1}/{args.llm_template_count}: "
            f"{template.get('world_slug', '(missing slug)')}",
            flush=True,
        )
        templates.append(template)
    return templates


def _write_templates(templates: List[Dict[str, Any]], outdir: str) -> str:
    path = os.path.join(outdir, "rpg_story_hidden_templates.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompt": STORY_TEMPLATE_PROMPT, "templates": templates}, f, indent=2)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate RPG v3 story-hidden-cause worlds.")
    parser.add_argument("--outdir", default="dataset_generation_code/all_out_rpg/out_rpg_v3_story_hidden")
    parser.add_argument("--distribution", default=None, help="JSON dict of archetype counts.")
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--oracle-n", type=int, default=STATIC_DEFAULT_ORACLE_N)
    parser.add_argument("--max-attempts-per-world", type=int, default=5)
    parser.add_argument("--template-json", default=None, help="Saved RPG story-hidden template JSON object/list.")
    parser.add_argument("--use-llm-templates", action="store_true", help="Ask an LLM to create concrete world templates.")
    parser.add_argument(
        "--builtin-template",
        default=",".join(DEFAULT_BUILTIN_TEMPLATES),
        help=(
            "Comma-separated built-in template(s) used when neither --template-json nor "
            "--use-llm-templates is given; templates are cycled across the worlds. "
            f"Choices: {', '.join(sorted(_BUILTIN_TEMPLATES))}, or 'all'. "
            f"Default: {','.join(DEFAULT_BUILTIN_TEMPLATES)}."
        ),
    )
    parser.add_argument("--llm-template-count", type=int, default=1)
    parser.add_argument("--llm-backend", default="bedrock", choices=["bedrock"])
    parser.add_argument("--llm-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--llm-region", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    distribution = json.loads(args.distribution) if args.distribution else None
    if args.template_json:
        templates = _load_template_json(args.template_json)
    elif args.use_llm_templates:
        templates = _generate_llm_templates(args)
    else:
        requested = (
            sorted(_BUILTIN_TEMPLATES)
            if args.builtin_template.strip().lower() == "all"
            else [t.strip() for t in args.builtin_template.split(",") if t.strip()]
        )
        unknown = [t for t in requested if t not in _BUILTIN_TEMPLATES]
        if unknown:
            parser.error(
                f"unknown --builtin-template {unknown}; choices: {sorted(_BUILTIN_TEMPLATES)} or 'all'"
            )
        templates = [_normalise_story_template(_BUILTIN_TEMPLATES[name]()) for name in requested]
    static_rpg_generate_dataset(
        outdir=args.outdir,
        distribution=distribution,
        start_seed=args.start_seed,
        oracle_n=args.oracle_n,
        max_attempts_per_world=args.max_attempts_per_world,
        templates=templates,
    )
    _write_templates(templates, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
