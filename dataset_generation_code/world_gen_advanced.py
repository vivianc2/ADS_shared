"""Advanced causal-reasoning benchmark: archetype-driven world generator.

ONE question per world. Six archetypes:
    1. safety_constrained    — pick intervention that improves target without
                                worsening a safety outcome.
    2. confounding_reversal  — observational sign of treatment effect on
                                outcome is opposite to interventional sign.
    3. mediator_structure    — does T affect O only through M, also directly,
                                or not through M? (also a 'which-mediator?'
                                sub-variant).
    4. satisficing            — find any intervention that improves target by
                                at least a hidden threshold.
    5. subgroup_robust        — find an intervention that helps every
                                subgroup, not just the population average.
    6. invalid_premise        — proposed intervention is on a non-intervenable
                                variable, or on the wrong side of an arrow.

Pipeline (per world):
    role_plan (LLM) → required+forbidden edges (code) → background graph (code)
        → controlled CPDs for central edges (code) → strong CPDs for background
        → archetype validator on exact-inference signature → story (LLM)
        → question template + lazy-gold metadata.

LLM is used ONLY for naming/realism (variable names, values, story). All
graph and probability decisions are deterministic code, so empirical
signatures are guaranteed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Make bedrock_llm.py (in framework_code/) importable for the LLM client.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "framework_code")
if os.path.isdir(_FRAMEWORK_DIR) and _FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _FRAMEWORK_DIR)


# ===================================================================
# Constants
# ===================================================================

TOPICS = [
    "Screening & diagnosis",
    "Treatment effectiveness",
    "Hospital data",
    "Education",
    "Social Science",
    "Labor & Policy",
    "User Behavior",
    "Criminal Justice",
]

# Default per-archetype counts for a 60-world dataset.  Combine with
# --n-nodes 10 15 to get 30 worlds at each size.
DEFAULT_DISTRIBUTION: Dict[str, int] = {
    "safety_constrained":  15,
    "confounding_reversal": 12,
    "mediator_structure":  12,
    "satisficing":         10,
    "subgroup_robust":      5,
    "invalid_premise":      6,
}

# Topics that play nicely with each archetype (used to avoid weird pairings).
ARCHETYPE_TOPICS: Dict[str, List[str]] = {
    "safety_constrained":   ["Screening & diagnosis", "Treatment effectiveness",
                             "Hospital data", "Criminal Justice"],
    "confounding_reversal": ["Treatment effectiveness", "Education",
                             "Criminal Justice", "Labor & Policy",
                             "Hospital data"],
    "mediator_structure":   list(TOPICS),
    "satisficing":          list(TOPICS),
    "subgroup_robust":      ["Education", "Labor & Policy", "User Behavior",
                             "Hospital data"],
    "invalid_premise":      list(TOPICS),
}

MEDIATOR_SUB_VARIANTS = [
    "mediated_only",
    "direct_and_mediated",
    "not_mediator",
    "which_mediator",
]

INVALID_PREMISE_SUB_VARIANTS = [
    "valid_proposed_intervention",
    "non_intervenable_proxy",
    "wrong_side_intervention",
]


# ===================================================================
# Subdomain salts — keep worlds within the same (archetype, topic) from
# converging on identical role plans.  Claude Opus 4.8 at moderate temp
# is highly deterministic; without a sub-context anchor it picks the
# same canonical study every time (e.g. "colorectal cancer screening"
# every safety_constrained × Screening world).
# ===================================================================

SUBDOMAINS: Dict[str, List[str]] = {
    "Screening & diagnosis": [
        "breast cancer mammography screening",
        "cardiovascular disease risk screening",
        "depression and anxiety mental-health screening",
        "diabetes glycemic screening",
        "neonatal genetic disorder screening",
        "tuberculosis pulmonary screening",
        "skin cancer dermoscopy screening",
        "prostate cancer PSA testing",
        "Alzheimer's cognitive assessment",
        "lung cancer low-dose CT screening",
        "HPV-related cervical cancer screening",
        "pediatric developmental delay screening",
    ],
    "Treatment effectiveness": [
        "hypertension medication management",
        "post-surgical pain control protocols",
        "antidepressant pharmacotherapy",
        "type-2 diabetes glycemic control",
        "asthma controller therapy",
        "chronic kidney disease management",
        "rheumatoid arthritis disease-modifying therapy",
        "stroke rehabilitation program",
        "HIV antiretroviral adherence",
        "weight-loss bariatric program",
        "chemotherapy regimen selection",
        "cardiac rehabilitation post-MI",
    ],
    "Hospital data": [
        "ICU sepsis management",
        "emergency department triage",
        "30-day readmission reduction",
        "hospital-acquired-infection prevention",
        "post-operative recovery pathway",
        "labor & delivery outcomes",
        "pediatric ward length of stay",
        "stroke unit door-to-needle time",
        "surgical site complication tracking",
        "discharge planning for elderly patients",
        "telemetry-monitored cardiac ward",
        "transplant unit graft survival",
    ],
    "Education": [
        "high-school dropout prevention",
        "early-childhood literacy program",
        "STEM achievement gap intervention",
        "college completion for first-gen students",
        "remedial math course redesign",
        "vocational training apprenticeship",
        "ESL bilingual classroom outcomes",
        "summer-bridge college transition",
        "after-school tutoring program",
        "graduate-school comprehensive exam pass rates",
        "online vs in-person undergraduate course",
        "K-12 special education IEP outcomes",
    ],
    "Social Science": [
        "neighborhood-level civic engagement",
        "household financial inclusion",
        "intergenerational social mobility",
        "online misinformation susceptibility",
        "voter turnout in local elections",
        "community-based mental-health support",
        "informal social-network influence",
        "religiosity & life satisfaction study",
        "immigrant integration outcomes",
        "rural-urban migration well-being",
        "youth volunteering longitudinal panel",
        "household division-of-labor study",
    ],
    "Labor & Policy": [
        "minimum-wage policy & employment",
        "active labor-market reemployment program",
        "unemployment-insurance generosity study",
        "remote-work-policy productivity",
        "workplace-safety regulation enforcement",
        "occupational-licensing reform",
        "parental-leave policy outcomes",
        "vocational-training subsidy evaluation",
        "EITC anti-poverty effectiveness",
        "right-to-work law impact",
        "gig-economy labor protections",
        "veterans hiring incentive program",
    ],
    "User Behavior": [
        "app-notification engagement design",
        "subscription-service churn prevention",
        "e-commerce checkout-funnel optimization",
        "social-media content moderation impact",
        "video-platform recommendation tuning",
        "fintech onboarding completion",
        "wearable-fitness habit formation",
        "online-banking security UX",
        "MOOC course completion levers",
        "ride-share demand-pricing experiment",
        "gaming retention through reward schedules",
        "streaming-service free-trial conversion",
    ],
    "Criminal Justice": [
        "drug-court diversion program",
        "juvenile-offender rehabilitation",
        "pretrial bail reform impact",
        "community-policing engagement",
        "probation supervision intensity",
        "reentry employment program for released individuals",
        "domestic-violence protective-order enforcement",
        "body-camera policy evaluation",
        "forensic-evidence backlog reduction",
        "specialty-court (mental-health) outcomes",
        "violence-interruption street-outreach program",
        "white-collar fraud prosecution program",
    ],
}

SYSTEM_JSON = (
    "You must output ONLY one JSON object and nothing else. "
    "No markdown, no commentary, no trailing text."
)

# Min separation between best and runner-up actions (expected-state-index
# units). Worlds with smaller margins are rejected so finite samples don't
# flip the gold answer.
MIN_TOP_GAP = 0.20

# Minimum effect size (absolute change in expected target index) for a
# "meaningful" intervention. Used by satisficing thresholds and validators.
MIN_EFFECT = 0.20

# Safety tolerance for safety-constrained: |Δsafety| ≤ this counts as "safe".
SAFETY_TOLERANCE = 0.08

# Reversal margin for confounding_reversal: each side must move ≥ this.
REVERSAL_MARGIN = 0.18

ROOT_MIN_PROB = 0.10


# ===================================================================
# LLM adapter
# ===================================================================

@dataclass
class _LLM:
    """Thin wrapper that exposes .chat(system, user) -> str over Bedrock."""
    model_id: str
    region_name: Optional[str] = None
    # Higher than typical for structured-JSON calls because Opus 4.8 at low
    # temp is highly deterministic and was producing identical role plans
    # across worlds with the same (topic, archetype) — even with seed
    # rotation.  0.7 gives meaningful diversity while still parsing reliably.
    temperature: float = 0.7
    max_new_tokens: int = 1800
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from bedrock_llm import BedrockLLM
        self._client = BedrockLLM(
            model_id=self.model_id,
            region_name=self.region_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        self.model_name = self.model_id

    def chat(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        return self._client.generate(system, user, max_new_tokens=max_new_tokens)


def build_llm(backend: str, model: str) -> _LLM:
    if backend != "bedrock":
        raise ValueError(
            f"Only Bedrock is supported in the new generator (backend={backend!r}). "
            f"Use the legacy world_gen_advanced_old.py for local Qwen."
        )
    return _LLM(model_id=model)


# ===================================================================
# JSON utilities
# ===================================================================

def _extract_first_json(text: str) -> Dict[str, Any]:
    """Extract the first balanced {…} object from `text`. Raises on failure."""
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


def _llm_json(llm: _LLM, system: str, user: str, max_tries: int = 3) -> Dict[str, Any]:
    """Call LLM, expect JSON output, retry on parse failure."""
    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        raw = llm.chat(system, user)
        try:
            return _extract_first_json(raw)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"LLM JSON parse failed after {max_tries} tries: {last_err}")


# ===================================================================
# Variable / role specs
# ===================================================================

@dataclass
class VarSpec:
    name: str
    values: List[str]
    desc: str
    role: str           # "target", "safety", "good_intv", ..., or "background"
    intervenable: bool
    preferred_low: Optional[bool] = None  # only meaningful for ordinal vars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "desc": self.desc,
            "role": self.role,
            "intervenable": self.intervenable,
            "preferred_low": self.preferred_low,
        }


@dataclass
class RolePlan:
    archetype: str
    topic: str
    sub_variant: Optional[str]   # e.g. "mediated_only" / "direct_and_mediated" / ...
    study_name: str
    variables: List[VarSpec]
    roles: Dict[str, str]        # role -> variable name

    @property
    def by_name(self) -> Dict[str, VarSpec]:
        return {v.name: v for v in self.variables}

    def role_var(self, role: str) -> VarSpec:
        return self.by_name[self.roles[role]]


# ===================================================================
# Role-plan prompts (one per archetype)
# ===================================================================

_VARIABLE_SPEC_RULES = (
    "RULES for every variable:\n"
    "  - 'name' is PascalCase, no spaces, alphanumeric only (e.g. FollowUpVisits).\n"
    "  - 'values' is a list of 2 or 3 short string states (no spaces inside;\n"
    "    use camelCase or kebab-form: 'No', 'Yes', 'Low', 'Medium', 'High',\n"
    "    'None', 'Weekly', 'Monthly', etc.).\n"
    "  - For ordinal OUTCOME variables (target, outcome, mediator,\n"
    "    safety_outcome): list states from BEST (preferred) to WORST.\n"
    "  - For INTERVENTION variables: list states from BASELINE/OFF to\n"
    "    STRONGEST/ON, e.g. ['No','Yes'] or ['None','Some','Heavy'] or\n"
    "    ['Off','Light','Full']. State[0] must be the inactive baseline.\n"
    "  - 'desc' must be DEFINITIONAL only — describe what the variable\n"
    "    MEASURES in 5-15 words.  STRICT BANS for desc:\n"
    "      * NO causal language: do not use 'causes', 'drives', 'affects',\n"
    "        'leads to', 'influences', 'results in', 'produces'.\n"
    "      * NO mention of confounding, mediation, moderation, spurious,\n"
    "        bias, adjustment, reversal, true effect, isolate, attenuate.\n"
    "      * NO comparison to other variables ('higher than X', 'similar\n"
    "        to Y', 'related to Z').\n"
    "      * NO statement of which value is good/bad for an outcome.\n"
    "    GOOD desc examples: 'Whether the patient was admitted overnight',\n"
    "    'Charge severity classification at booking', 'Hours of weekly\n"
    "    tutoring assigned'. \n"
    "    BAD desc examples (do NOT do this): 'Severity that drives both\n"
    "    detention and conviction', 'Treatment that may causally improve\n"
    "    outcomes after adjusting for severity'.\n"
    "  - 'intervenable': true if a researcher could randomize/assign this in a\n"
    "    real experiment; false for things like AgeGroup, BaselineSeverity,\n"
    "    PriorHistory, ComorbidityBurden.\n"
    "  - DO NOT name a variable after a probability or a calculation\n"
    "    (no 'EffectSize', 'CorrelationLevel', 'PValue').\n"
    "  - Avoid generic placeholders ('VariableX', 'NodeA').\n"
    "  - Each variable must be semantically distinct.\n"
)


def _subdomain_clause(subdomain: Optional[str]) -> str:
    """Diversification clause: pin the role plan to a SPECIFIC sub-context
    so worlds with the same (topic, archetype) don't collapse to the same
    canonical study. Caller passes a different subdomain per world.
    """
    if not subdomain:
        return ""
    return (
        f"\nSPECIFIC SUB-CONTEXT for THIS world: {subdomain}.\n"
        f"All variable names and descriptions MUST be specific to this "
        f"sub-context — do NOT default to a generic or canonical example "
        f"in the broader topic. Pick variables a researcher in this exact "
        f"sub-context would actually study.\n"
    )


def _prompt_safety_constrained(topic: str, n_background: int, subdomain: Optional[str] = None) -> str:
    return f"""{SYSTEM_JSON}

Design a CAUSAL STUDY PLAN for the topic "{topic}".
{_subdomain_clause(subdomain)}
Scenario archetype: a researcher must pick ONE intervention that improves a
target outcome WITHOUT worsening a safety outcome.

Output exactly this JSON structure:

{{
  "study_name": "<short descriptive name>",
  "roles": {{
    "target":            <variable name>,
    "safety_outcome":    <variable name>,
    "good_intervention": <variable name>,
    "risky_intervention":<variable name>,
    "weak_intervention": <variable name>,
    "good_mediator":     <variable name>,
    "confounder":        <variable name>
  }},
  "variables": [
    {{ "name": ..., "values": [...], "desc": ..., "role": "target",            "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "safety_outcome",    "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "good_intervention", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "risky_intervention","intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "weak_intervention", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "good_mediator",     "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "confounder",        "intervenable": false }},
    ... and {n_background} more variables with role "background"
        (some intervenable, some not — realistic context variables for this topic)
  ]
}}

The TARGET and SAFETY_OUTCOME must each have 2 or 3 ordered states (best→worst),
since they are outcomes the researcher wants to influence in a known direction.
Each INTERVENTION should have 2 or 3 states (e.g. None/Some/Heavy or Off/On).
The CONFOUNDER (e.g. BaselineSeverity, AgeGroup) is not intervenable.

{_VARIABLE_SPEC_RULES}
"""


def _prompt_confounding_reversal(topic: str, n_background: int, subdomain: Optional[str] = None) -> str:
    return f"""{SYSTEM_JSON}

Design a CAUSAL STUDY PLAN for the topic "{topic}".
{_subdomain_clause(subdomain)}
Scenario archetype: a strong CONFOUNDER drives both treatment assignment and
the outcome, so the OBSERVATIONAL association between treatment and outcome
points OPPOSITE to the true causal effect.

Output exactly this JSON structure:

{{
  "study_name": "<short descriptive name>",
  "roles": {{
    "treatment":  <variable name>,
    "outcome":    <variable name>,
    "mediator":   <variable name>,
    "confounder": <variable name>
  }},
  "variables": [
    {{ "name": ..., "values": [...], "desc": ..., "role": "treatment",  "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "outcome",    "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "mediator",   "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "confounder", "intervenable": false }},
    ... and {n_background} more "background" variables
  ]
}}

The TREATMENT should be binary (e.g. ["No", "Yes"]) so we can talk about
"treated vs untreated".  The OUTCOME should be ordinal with 2-3 states.
The CONFOUNDER (e.g. BaselineSeverity) drives BOTH assignment and outcome.

{_VARIABLE_SPEC_RULES}
"""


def _prompt_mediator_structure(topic: str, sub_variant: str, n_background: int, subdomain: Optional[str] = None) -> str:
    sub_explain = {
        "mediated_only": (
            "T's effect on O flows ONLY through the mediator M (no direct edge,\n"
            "no second pathway)."
        ),
        "direct_and_mediated": (
            "T affects O both through M AND through a direct/other pathway."
        ),
        "not_mediator": (
            "T affects O, but NOT through the variable proposed_M; the real\n"
            "pathway is via a different variable (true_mediator)."
        ),
        "which_mediator": (
            "Several candidate mediators exist — only ONE truly lies on a directed\n"
            "path from T to O; the others are decoys."
        ),
    }[sub_variant]

    role_block_map = {
        "mediated_only": """
  "roles": {
    "treatment":  <variable name>,
    "outcome":    <variable name>,
    "mediator":   <variable name>,
    "confounder": <variable name>
  },""",
        "direct_and_mediated": """
  "roles": {
    "treatment":  <variable name>,
    "outcome":    <variable name>,
    "mediator":   <variable name>,
    "confounder": <variable name>
  },""",
        "not_mediator": """
  "roles": {
    "treatment":     <variable name>,
    "outcome":       <variable name>,
    "proposed_M":    <variable name>,
    "true_mediator": <variable name>,
    "confounder":    <variable name>
  },""",
        "which_mediator": """
  "roles": {
    "treatment":     <variable name>,
    "outcome":       <variable name>,
    "true_mediator": <variable name>,
    "decoy_M1":      <variable name>,
    "decoy_M2":      <variable name>,
    "confounder":    <variable name>
  },""",
    }
    role_vars_map = {
        "mediated_only": [
            ("treatment",  True),
            ("outcome",    False),
            ("mediator",   False),
            ("confounder", False),
        ],
        "direct_and_mediated": [
            ("treatment",  True),
            ("outcome",    False),
            ("mediator",   False),
            ("confounder", False),
        ],
        "not_mediator": [
            ("treatment",     True),
            ("outcome",       False),
            ("proposed_M",    False),
            ("true_mediator", False),
            ("confounder",    False),
        ],
        "which_mediator": [
            ("treatment",     True),
            ("outcome",       False),
            ("true_mediator", False),
            ("decoy_M1",      False),
            ("decoy_M2",      False),
            ("confounder",    False),
        ],
    }
    role_lines = "\n    ".join(
        f'{{ "name": ..., "values": [...], "desc": ..., "role": "{r}", "intervenable": {str(it).lower()} }},'
        for r, it in role_vars_map[sub_variant]
    )

    return f"""{SYSTEM_JSON}

Design a CAUSAL STUDY PLAN for the topic "{topic}".
{_subdomain_clause(subdomain)}
Scenario archetype (mediator structure, sub-variant "{sub_variant}"):
{sub_explain}

Output exactly this JSON structure:
{{
  "study_name": "<short descriptive name>",{role_block_map[sub_variant]}
  "variables": [
    {role_lines}
    ... and {n_background} more "background" variables
  ]
}}

TREATMENT should be binary or ternary. OUTCOME should be ordinal (2-3 states).
Mediator-like variables should have 2-3 ordered states.

{_VARIABLE_SPEC_RULES}
"""


def _prompt_satisficing(topic: str, n_background: int, subdomain: Optional[str] = None) -> str:
    return f"""{SYSTEM_JSON}

Design a CAUSAL STUDY PLAN for the topic "{topic}".
{_subdomain_clause(subdomain)}
Scenario archetype: a researcher needs to find ANY intervention that
"meaningfully" improves a target outcome — multiple may qualify, none may.

Output exactly this JSON structure:

{{
  "study_name": "<short descriptive name>",
  "roles": {{
    "target":         <variable name>,
    "intervention_A": <variable name>,
    "intervention_B": <variable name>,
    "intervention_C": <variable name>,
    "intervention_D": <variable name>,
    "confounder":     <variable name>
  }},
  "variables": [
    {{ "name": ..., "values": [...], "desc": ..., "role": "target",         "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_A", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_B", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_C", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_D", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "confounder",     "intervenable": false }},
    ... and {n_background} more "background" variables
  ]
}}

TARGET is ordinal (2-3 states, BEST → WORST). Each INTERVENTION has 2-3
states. Distinct, plausible interventions for the topic.

{_VARIABLE_SPEC_RULES}
"""


def _prompt_subgroup_robust(topic: str, n_background: int, subdomain: Optional[str] = None) -> str:
    return f"""{SYSTEM_JSON}

Design a CAUSAL STUDY PLAN for the topic "{topic}".
{_subdomain_clause(subdomain)}
Scenario archetype: an intervention helps the AVERAGE patient/student/user
but harms or fails one subgroup; another intervention helps EVERY subgroup
moderately. The researcher needs the robust one.

Output exactly this JSON structure:

{{
  "study_name": "<short descriptive name>",
  "roles": {{
    "target":           <variable name>,
    "group":            <variable name>,
    "intervention_avg": <variable name>,
    "intervention_rob": <variable name>,
    "intervention_bad": <variable name>,
    "confounder":       <variable name>
  }},
  "variables": [
    {{ "name": ..., "values": [...], "desc": ..., "role": "target",           "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "group",            "intervenable": false }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_avg", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_rob", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "intervention_bad", "intervenable": true  }},
    {{ "name": ..., "values": [...], "desc": ..., "role": "confounder",       "intervenable": false }},
    ... and {n_background} more "background" variables
  ]
}}

GROUP is a non-intervenable subgroup variable with 2 states (e.g.
LowResource/HighResource, Younger/Older, FirstTime/Repeat).
TARGET is ordinal (2-3 states, BEST → WORST).
Each INTERVENTION has 2-3 states.

{_VARIABLE_SPEC_RULES}
"""


def _prompt_invalid_premise(
    topic: str, sub_variant: Optional[str], n_background: int,
    subdomain: Optional[str] = None,
) -> str:
    sub = sub_variant or "non_intervenable_proxy"
    if sub == "non_intervenable_proxy":
        scenario = (
            "someone proposes intervening on a NON-INTERVENABLE variable "
            "(like AgeGroup, PriorHistory, BaselineSeverity) to influence a "
            "target outcome. The valid alternative is to intervene on a "
            "downstream manipulable policy variable."
        )
        role_block = """
    "target":              <variable name>,
    "non_intervenable_x":  <variable name>,
    "intervenable_alt":    <variable name>,
    "alt_mediator":        <variable name>,
    "confounder":          <variable name>
"""
        role_lines = """
    { "name": ..., "values": [...], "desc": ..., "role": "target",              "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "non_intervenable_x", "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "intervenable_alt",   "intervenable": true  },
    { "name": ..., "values": [...], "desc": ..., "role": "alt_mediator",       "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "confounder",         "intervenable": false },"""
        extra_rules = (
            "`non_intervenable_x` MUST be a variable that genuinely cannot be "
            "intervened on in a real experiment - e.g. AgeGroup, "
            "BiologicalSex, PriorOffenseCount, BaselineSeverity. It should "
            "still PREDICT the target.\n"
            "`intervenable_alt` is a manipulable policy variable that can move "
            "the target."
        )
    elif sub == "wrong_side_intervention":
        scenario = (
            "someone proposes manipulating an INTERVENABLE downstream proxy, "
            "status, or administrative readout. It may be easy to set and "
            "strongly associated with the target, but it is on the wrong side "
            "of the causal arrows and should not move the target when set."
        )
        role_block = """
    "target":                 <variable name>,
    "proposed_intervention":  <variable name>,
    "intervenable_alt":       <variable name>,
    "alt_mediator":           <variable name>,
    "confounder":             <variable name>
"""
        role_lines = """
    { "name": ..., "values": [...], "desc": ..., "role": "target",                "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "proposed_intervention", "intervenable": true  },
    { "name": ..., "values": [...], "desc": ..., "role": "intervenable_alt",      "intervenable": true  },
    { "name": ..., "values": [...], "desc": ..., "role": "alt_mediator",          "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "confounder",            "intervenable": false },"""
        extra_rules = (
            "`proposed_intervention` should be a manipulable proxy/status/readout "
            "that a naive analyst might try to set (for example an alert flag, "
            "queue label, documentation status, assignment code, or display "
            "state). Do NOT make it semantically identical to the target.\n"
            "`intervenable_alt` is a different manipulable policy variable "
            "that can move the target through `alt_mediator`."
        )
    elif sub == "valid_proposed_intervention":
        scenario = (
            "someone proposes a genuinely sensible INTERVENABLE action. The "
            "proposal should move a downstream mechanism and thereby improve "
            "the target outcome, so the correct answer is that the proposal is "
            "valid rather than needing a replacement."
        )
        role_block = """
    "target":                 <variable name>,
    "proposed_intervention":  <variable name>,
    "intervenable_alt":       <variable name>,
    "alt_mediator":           <variable name>,
    "confounder":             <variable name>
"""
        role_lines = """
    { "name": ..., "values": [...], "desc": ..., "role": "target",                "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "proposed_intervention", "intervenable": true  },
    { "name": ..., "values": [...], "desc": ..., "role": "intervenable_alt",      "intervenable": true  },
    { "name": ..., "values": [...], "desc": ..., "role": "alt_mediator",          "intervenable": false },
    { "name": ..., "values": [...], "desc": ..., "role": "confounder",            "intervenable": false },"""
        extra_rules = (
            "`proposed_intervention` should be a manipulable policy/action a "
            "researcher could actually deploy.\n"
            "`alt_mediator` is a plausible downstream mechanism for that "
            "proposal. `intervenable_alt` is a separate manipulable action "
            "that should sound plausible but not obviously superior from the "
            "variable catalog alone."
        )
    else:
        raise ValueError(f"unknown invalid_premise sub_variant {sub!r}")

    return f"""{SYSTEM_JSON}

Design a CAUSAL STUDY PLAN for the topic "{topic}".
{_subdomain_clause(subdomain)}
Scenario archetype (invalid premise, sub-variant "{sub}"):
{scenario}

Output exactly this JSON structure:

{{
  "study_name": "<short descriptive name>",
  "roles": {{
{role_block.rstrip()}
  }},
  "variables": [
{role_lines}
    ... and {n_background} more "background" variables
  ]
}}

{extra_rules}
TARGET is ordinal with 2-3 states.

{_VARIABLE_SPEC_RULES}
"""


_PROMPT_DISPATCH: Dict[str, Callable[..., str]] = {
    "safety_constrained":   _prompt_safety_constrained,
    "confounding_reversal": _prompt_confounding_reversal,
    "satisficing":          _prompt_satisficing,
    "subgroup_robust":      _prompt_subgroup_robust,
}


# ===================================================================
# Variable spec sanitization
# ===================================================================

def _sanitize_pascal(name: str) -> str:
    out = "".join(ch for ch in str(name) if ch.isalnum())
    if not out:
        return "Var"
    if not out[0].isalpha():
        out = "V" + out
    return out[0].upper() + out[1:]


def _sanitize_value(v: str) -> str:
    s = str(v).strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in "-_")
    return s or "X"


def _validate_role_plan(
    archetype: str, sub_variant: Optional[str], plan_json: Dict[str, Any],
    expected_total: int,
) -> RolePlan:
    """Coerce LLM output into a RolePlan, raising on structural problems."""
    roles_raw = plan_json.get("roles") or {}
    variables_raw = plan_json.get("variables") or []
    if not isinstance(roles_raw, dict) or not isinstance(variables_raw, list):
        raise ValueError("role plan: missing 'roles' or 'variables'")

    roles = {str(k): _sanitize_pascal(v) for k, v in roles_raw.items() if v}
    if not roles:
        raise ValueError("role plan: empty 'roles'")

    seen_names: Set[str] = set()
    seen_descs: Set[str] = set()
    var_specs: List[VarSpec] = []
    for vd in variables_raw:
        if not isinstance(vd, dict):
            continue
        name = _sanitize_pascal(vd.get("name", ""))
        if not name or name in seen_names:
            continue
        values_raw = vd.get("values") or []
        values = [_sanitize_value(x) for x in values_raw if str(x).strip()]
        if not (2 <= len(values) <= 4):
            continue
        if len(set(v.lower() for v in values)) != len(values):
            continue
        desc = str(vd.get("desc") or vd.get("description") or "").strip()
        if not desc:
            desc = f"{name} variable"
        # avoid dup descs which the LLM occasionally emits
        if desc.lower() in seen_descs:
            desc = desc + f" ({name})"
        seen_descs.add(desc.lower())
        role = str(vd.get("role") or "background").strip() or "background"
        intervenable = bool(vd.get("intervenable", role == "background"))
        seen_names.add(name)
        var_specs.append(VarSpec(
            name=name, values=values, desc=desc, role=role,
            intervenable=intervenable,
        ))

    # Re-map roles -> sanitized names (LLM might have spelled the role-key
    # entry differently from how it appears in `variables`).  Replace each
    # role with the FIRST var spec carrying that role.
    role_to_var: Dict[str, str] = {}
    for r in roles.keys():
        match = next((v for v in var_specs if v.role == r), None)
        if match is not None:
            role_to_var[r] = match.name
        elif roles[r] in seen_names:
            role_to_var[r] = roles[r]
    missing = [r for r in roles if r not in role_to_var]
    if missing:
        raise ValueError(f"role plan: missing variables for roles {missing}")

    # Enforce that the LLM produced EVERY role this archetype needs.
    # Without this, an LLM that omits e.g. "confounder" sails past validation
    # and the build_* function later crashes with KeyError.
    expected_roles = _required_roles(archetype, sub_variant)
    not_provided = [r for r in expected_roles if r not in role_to_var]
    if not_provided:
        # Recovery: try to fill missing roles from variables tagged with that role
        # (in case LLM put the var in `variables` but forgot the `roles` entry).
        for r in not_provided:
            match = next((v for v in var_specs if v.role == r), None)
            if match is not None:
                role_to_var[r] = match.name
        still_missing = [r for r in expected_roles if r not in role_to_var]
        if still_missing:
            raise ValueError(
                f"role plan for {archetype}/{sub_variant or '-'}: "
                f"required roles missing: {still_missing}"
            )

    if len(var_specs) < expected_total - 2:
        raise ValueError(
            f"role plan: only {len(var_specs)} variables (expected ≈{expected_total})"
        )
    if len(var_specs) > expected_total + 4:
        # trim the tail of background variables to match target count
        roleful = [v for v in var_specs if v.role != "background"]
        background = [v for v in var_specs if v.role == "background"]
        target = max(0, expected_total - len(roleful))
        var_specs = roleful + background[:target]

    return RolePlan(
        archetype=archetype,
        topic="",   # filled in by caller
        sub_variant=sub_variant,
        study_name=str(plan_json.get("study_name") or f"{archetype} study"),
        variables=var_specs,
        roles=role_to_var,
    )


def _generate_role_plan(
    llm: _LLM, topic: str, archetype: str, sub_variant: Optional[str],
    n_nodes: int, max_tries: int = 3, subdomain: Optional[str] = None,
) -> RolePlan:
    n_role = _archetype_role_count(archetype, sub_variant)
    n_background = max(2, n_nodes - n_role)
    if archetype == "mediator_structure":
        prompt = _prompt_mediator_structure(
            topic, sub_variant or "mediated_only", n_background, subdomain=subdomain,
        )
    elif archetype == "invalid_premise":
        prompt = _prompt_invalid_premise(
            topic, sub_variant or "non_intervenable_proxy",
            n_background, subdomain=subdomain,
        )
    else:
        prompt = _PROMPT_DISPATCH[archetype](topic, n_background, subdomain=subdomain)

    last: Optional[Exception] = None
    for _ in range(max_tries):
        try:
            js = _llm_json(llm, SYSTEM_JSON, prompt)
            plan = _validate_role_plan(archetype, sub_variant, js, expected_total=n_nodes)
            plan.topic = topic
            # Set preferred_low=True for every outcome role.  The role-plan
            # prompt instructs the LLM to list states from BEST to WORST, so
            # state[0] is the preferred state by construction. The previous
            # token-based heuristic flipped this for variables like
            # FinalGrade=['High','Medium','Low'] (since "low" is in the
            # prefer-low-first set), producing CPDs that drive worlds in
            # the wrong direction.
            for v in plan.variables:
                if v.role in _OUTCOME_ROLES.get(archetype, set()):
                    v.preferred_low = True
            return plan
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"role plan generation failed for {archetype}/{topic}: {last}")


def _required_roles(archetype: str, sub_variant: Optional[str]) -> List[str]:
    """Roles the build/validate functions read out of `role_plan.roles`.
    Must match the roles enumerated in the prompts and used in builders.
    """
    if archetype == "safety_constrained":
        return ["target", "safety_outcome", "good_intervention",
                "risky_intervention", "weak_intervention",
                "good_mediator", "confounder"]
    if archetype == "confounding_reversal":
        return ["treatment", "outcome", "mediator", "confounder"]
    if archetype == "mediator_structure":
        if sub_variant in ("mediated_only", "direct_and_mediated"):
            return ["treatment", "outcome", "mediator", "confounder"]
        if sub_variant == "not_mediator":
            return ["treatment", "outcome", "proposed_M", "true_mediator", "confounder"]
        if sub_variant == "which_mediator":
            return ["treatment", "outcome", "true_mediator",
                    "decoy_M1", "decoy_M2", "confounder"]
    if archetype == "satisficing":
        return ["target", "intervention_A", "intervention_B",
                "intervention_C", "intervention_D", "confounder"]
    if archetype == "subgroup_robust":
        return ["target", "group", "intervention_avg",
                "intervention_rob", "intervention_bad", "confounder"]
    if archetype == "invalid_premise":
        if (sub_variant or "non_intervenable_proxy") in (
            "wrong_side_intervention",
            "valid_proposed_intervention",
        ):
            return ["target", "proposed_intervention", "intervenable_alt",
                    "alt_mediator", "confounder"]
        return ["target", "non_intervenable_x", "intervenable_alt",
                "alt_mediator", "confounder"]
    return []


def _archetype_role_count(archetype: str, sub_variant: Optional[str]) -> int:
    if archetype == "safety_constrained":
        return 7
    if archetype == "confounding_reversal":
        return 4
    if archetype == "mediator_structure":
        return {
            "mediated_only": 4, "direct_and_mediated": 4,
            "not_mediator": 5, "which_mediator": 6,
        }[sub_variant or "mediated_only"]
    if archetype == "satisficing":
        return 6
    if archetype == "subgroup_robust":
        return 6
    if archetype == "invalid_premise":
        return 5
    raise KeyError(archetype)


_OUTCOME_ROLES: Dict[str, Set[str]] = {
    "safety_constrained":   {"target", "safety_outcome", "good_mediator"},
    "confounding_reversal": {"outcome", "mediator"},
    "mediator_structure":   {"outcome", "mediator", "proposed_M",
                             "true_mediator", "decoy_M1", "decoy_M2"},
    "satisficing":          {"target"},
    "subgroup_robust":      {"target"},
    "invalid_premise":      {"target", "alt_mediator"},
}


# ===================================================================
# preferred_low heuristic (list-order based, no LLM call)
# ===================================================================

_PREFER_LOW_FIRST = {
    "none", "no", "absent", "low", "minimal", "healthy", "normal", "mild",
    "few", "zero", "0", "negative", "safe", "stable", "good",
}
_PREFER_HIGH_FIRST = {
    "high", "severe", "many", "yes", "positive", "bad", "poor",
}


def _heuristic_preferred_low(v: VarSpec) -> Optional[bool]:
    first = str(v.values[0]).lower()
    last = str(v.values[-1]).lower()
    if first in _PREFER_LOW_FIRST and last not in _PREFER_LOW_FIRST:
        return True
    if last in _PREFER_LOW_FIRST and first not in _PREFER_LOW_FIRST:
        return False
    if first in _PREFER_HIGH_FIRST:
        return False
    if last in _PREFER_HIGH_FIRST:
        return True
    # Heuristic on common name tokens
    name = v.name.lower()
    risk_tokens = ("risk", "complication", "failure", "dropout", "reoffense",
                   "readmission", "follow", "absence", "violation",
                   "missed", "error", "loss")
    good_tokens = ("recovery", "graduation", "satisfaction", "wellbeing",
                   "engagement", "retention", "improvement")
    if any(t in name for t in risk_tokens):
        return True
    if any(t in name for t in good_tokens):
        return False
    return None


# ===================================================================
# Graph construction
# ===================================================================

def _add_required_and_background(
    role_plan: RolePlan,
    required_edges: List[Tuple[str, str]],
    forbidden_edges: List[Tuple[str, str]],
    rng: random.Random,
    target_extra_edges: int = 4,
    max_in_degree: int = 3,
) -> nx.DiGraph:
    """Build a DAG with all required edges plus some plausible background edges.

    - Adds every (u, v) in required_edges.
    - Tries to inject extra edges among non-central nodes (or central → background)
      while respecting forbidden, max in-degree, and acyclicity.
    """
    g = nx.DiGraph()
    g.add_nodes_from([v.name for v in role_plan.variables])
    for u, v in required_edges:
        if u not in g.nodes or v not in g.nodes:
            raise ValueError(f"required edge references unknown node: {u}->{v}")
        g.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("required edges already form a cycle")

    central = {role_plan.roles[r] for r in role_plan.roles}
    background = [v.name for v in role_plan.variables if v.name not in central]
    forbidden_set = {tuple(e) for e in forbidden_edges}

    # Extra edges: prefer (background → central) and (background → background).
    # Skip edges into central nodes whose CPDs we plan to override to keep
    # validators stable.
    cpd_locked = _cpd_locked_nodes(role_plan)

    candidate_edges: List[Tuple[str, str]] = []
    for u in role_plan.by_name:
        for v in role_plan.by_name:
            if u == v:
                continue
            if (u, v) in forbidden_set:
                continue
            if g.has_edge(u, v):
                continue
            # Don't add edges into CPD-locked nodes (would invalidate
            # the controlled CPD or change number of parents).
            if v in cpd_locked:
                continue
            # Don't add edges INTO the group/confounder roots.
            if v in central and role_plan.by_name[v].role in (
                "group", "confounder",
            ):
                continue
            candidate_edges.append((u, v))

    rng.shuffle(candidate_edges)
    added = 0
    for (u, v) in candidate_edges:
        if added >= target_extra_edges:
            break
        if g.in_degree(v) >= max_in_degree:
            continue
        g.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(u, v)
            continue
        added += 1

    # Ensure the central component is weakly connected — required edges
    # already make it so for our archetypes (skeletons connect every role
    # to target/outcome).
    return g


def _cpd_locked_nodes(role_plan: RolePlan) -> Set[str]:
    """Nodes whose CPDs we set explicitly per archetype.

    Background edges INTO these nodes are forbidden because they would
    change the parent set and break the controlled CPD.
    """
    arch = role_plan.archetype
    sub = role_plan.sub_variant
    R = role_plan.roles
    if arch == "safety_constrained":
        return {R["target"], R["safety_outcome"], R["good_mediator"]}
    if arch == "confounding_reversal":
        return {R["treatment"], R["mediator"], R["outcome"]}
    if arch == "mediator_structure":
        if sub == "not_mediator":
            return {R["proposed_M"], R["true_mediator"], R["outcome"]}
        if sub == "which_mediator":
            return {R["true_mediator"], R["decoy_M1"], R["decoy_M2"], R["outcome"]}
        return {R["mediator"], R["outcome"]}
    if arch == "satisficing":
        return {R["target"]}
    if arch == "subgroup_robust":
        return {R["target"]}
    if arch == "invalid_premise":
        if sub == "wrong_side_intervention":
            return {R["proposed_intervention"], R["alt_mediator"], R["target"]}
        if sub == "valid_proposed_intervention":
            return {R["alt_mediator"], R["target"]}
        return {R["alt_mediator"], R["target"]}
    return set()


# ===================================================================
# CPD primitives
# ===================================================================

def _root_cpd(
    name: str, states: List[str], rng: random.Random,
    alpha: float = 1.5, min_prob: float = ROOT_MIN_PROB,
) -> TabularCPD:
    """Dirichlet root with floor on every state."""
    K = len(states)
    np_rng = np.random.default_rng(rng.randrange(1 << 31))
    probs = np_rng.dirichlet(np.ones(K) * alpha)
    floor = min(min_prob, 1.0 / K * 0.9)
    probs = np.maximum(probs, floor)
    probs = probs / probs.sum()
    return TabularCPD(
        variable=name, variable_card=K,
        values=[[float(p)] for p in probs],
        state_names={name: states},
    )


def _strong_logistic_cpd(
    child: str, parents: List[str], state_names: Dict[str, List[str]],
    rng: random.Random, weight_abs: float = 2.5, bias_abs: float = 0.4,
) -> TabularCPD:
    """Strong logistic CPD for background non-controlled nodes."""
    child_states = state_names[child]
    K = len(child_states)
    parent_cards = [len(state_names[p]) for p in parents]
    n_configs = int(np.prod(parent_cards))

    base = np.array([rng.uniform(-bias_abs, bias_abs) for _ in range(K)])
    weights = np.array([
        [rng.uniform(-weight_abs, weight_abs) for _ in parents]
        for _ in range(K)
    ])

    all_probs = []
    for cfg in range(n_configs):
        idx = []
        temp = cfg
        for c in reversed(parent_cards):
            idx.append(temp % c)
            temp //= c
        idx = list(reversed(idx))

        logits = base.copy()
        for k in range(K):
            for p_idx, p_val in enumerate(idx):
                logits[k] += weights[k][p_idx] * p_val
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        all_probs.append(probs)

    values = [[float(all_probs[j][i]) for j in range(n_configs)] for i in range(K)]
    return TabularCPD(
        variable=child, variable_card=K, values=values,
        evidence=parents, evidence_card=parent_cards,
        state_names={child: child_states, **{p: state_names[p] for p in parents}},
    )


def _directional_cpd(
    child: str, parents: List[str], state_names: Dict[str, List[str]],
    baseline_logits: np.ndarray, effects: Dict[str, Dict[str, np.ndarray]],
    noise: float = 0.05, rng: Optional[random.Random] = None,
) -> TabularCPD:
    """Build a CPD with explicit directional effects per parent state.

    `baseline_logits`: shape (K,)
    `effects[parent][parent_state]`: shape (K,) — additive shift on logit scale.
        Parent-states not in `effects[parent]` contribute zero.

    For each parent configuration:
        logits = baseline_logits + Σ effects[parent][state] + small_noise
        P(child | config) = softmax(logits)
    """
    rng = rng or random.Random()
    child_states = state_names[child]
    K = len(child_states)
    parent_cards = [len(state_names[p]) for p in parents]
    n_configs = int(np.prod(parent_cards))

    all_probs = []
    for cfg in range(n_configs):
        idx = []
        temp = cfg
        for c in reversed(parent_cards):
            idx.append(temp % c)
            temp //= c
        idx = list(reversed(idx))

        logits = np.array(baseline_logits, dtype=float).copy()
        for p_idx, p in enumerate(parents):
            p_state = state_names[p][idx[p_idx]]
            shift = effects.get(p, {}).get(p_state)
            if shift is not None:
                logits += np.asarray(shift, dtype=float)
        if noise > 0:
            logits += np.array([rng.uniform(-noise, noise) for _ in range(K)])
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        all_probs.append(probs)

    values = [[float(all_probs[j][i]) for j in range(n_configs)] for i in range(K)]
    return TabularCPD(
        variable=child, variable_card=K, values=values,
        evidence=parents, evidence_card=parent_cards,
        state_names={child: child_states, **{p: state_names[p] for p in parents}},
    )


def _direction_vector(K: int, sign: float, magnitude: float) -> np.ndarray:
    """Logit shift that biases the distribution toward early or late states.

    sign > 0: shift mass toward LATE states (state_index=K-1).
    sign < 0: shift mass toward EARLY states (state_index=0).
    """
    if K <= 1:
        return np.zeros(K)
    grid = np.linspace(-1.0, 1.0, K)
    return float(sign) * float(magnitude) * grid


def _build_bn_with_controlled(
    dag: nx.DiGraph, role_plan: RolePlan, controlled: Dict[str, TabularCPD],
    rng: random.Random,
) -> DiscreteBayesianNetwork:
    """Assemble BN: roots + controlled CPDs + strong-logistic for the rest."""
    state_names = {v.name: list(v.values) for v in role_plan.variables}
    model = DiscreteBayesianNetwork(list(dag.edges()))
    model.add_nodes_from(list(dag.nodes()))
    cpds: List[TabularCPD] = []
    topo = list(nx.topological_sort(dag))
    for node in topo:
        parents = list(dag.predecessors(node))
        if node in controlled:
            cpd = controlled[node]
            # Validate parent SET matches; column order inside the CPD is
            # the CPD's own concern — pgmpy uses the CPD's evidence ordering
            # for inference, not the DAG's.
            cpd_parents = set(cpd.variables[1:])
            if cpd_parents != set(parents):
                raise RuntimeError(
                    f"controlled CPD for {node} has parents {sorted(cpd_parents)} "
                    f"but DAG has {sorted(parents)}"
                )
        elif not parents:
            cpd = _root_cpd(node, state_names[node], rng)
        else:
            cpd = _strong_logistic_cpd(
                node, parents, state_names, rng,
            )
        cpds.append(cpd)
    model.add_cpds(*cpds)
    if not model.check_model():
        raise RuntimeError("model.check_model failed")
    return model


# ===================================================================
# Inference helpers
# ===================================================================

def _shift_toward(v: VarSpec) -> str:
    """Bare-infinitive phrase: 'shift {name} toward {best} (and away from {worst})'.

    For use after auxiliaries like 'wants to ___' or 'would ___'.
    """
    return f"shift {v.name} toward {v.values[0]} (and away from {v.values[-1]})"


def _shifts_toward(v: VarSpec) -> str:
    """Third-person singular: 'shifts {name} toward {best} (and away from {worst})'."""
    return f"shifts {v.name} toward {v.values[0]} (and away from {v.values[-1]})"


def _shift_toward_short(v: VarSpec) -> str:
    return f"shift {v.name} toward {v.values[0]}"


def _making_worse(v: VarSpec) -> str:
    """Gerund phrase: 'making {name} more likely to be {worst}'.

    For use after 'without ___'.
    """
    return f"making {v.name} more likely to be {v.values[-1]}"


def _make_more_likely(v: VarSpec) -> str:
    return f"make {v.name} more likely to be {v.values[0]}"


def _make_worse(v: VarSpec) -> str:
    return f"make {v.name} more likely to be {v.values[-1]}"


def _scoring_for(var: VarSpec) -> Callable[[str], float]:
    """Return scoring(state) -> float; lower = better under preferred_low.

    Falls back to index-as-score (lower index = better) if preferred_low None.
    """
    states = list(var.values)
    sign = 1.0
    if var.preferred_low is False:
        sign = -1.0
    idx = {s: i for i, s in enumerate(states)}
    def _s(state: str) -> float:
        return sign * float(idx[state])
    return _s


def _expected_under_do(
    model: DiscreteBayesianNetwork, target: str,
    do: Dict[str, str], scoring: Callable[[str], float],
    evidence: Optional[Dict[str, str]] = None,
) -> float:
    """Exact E[scoring(target) | do(...), evidence] via mutilation + VE."""
    mutilated = deepcopy(model)
    for var, state in do.items():
        for p in list(mutilated.get_parents(var)):
            mutilated.remove_edge(p, var)
        old = mutilated.get_cpds(var)
        states = list(old.state_names[var])
        if state not in states:
            raise ValueError(f"do({var}={state!r}): state not in {states}")
        ix = states.index(state)
        vals = [[0.0] for _ in range(len(states))]
        vals[ix] = [1.0]
        new_cpd = TabularCPD(
            variable=var, variable_card=len(states),
            values=vals, state_names={var: states},
        )
        mutilated.remove_cpds(old)
        mutilated.add_cpds(new_cpd)
    ve = VariableElimination(mutilated)
    f = ve.query(variables=[target], evidence=evidence, show_progress=False)
    return float(sum(float(p) * scoring(s) for p, s in zip(f.values, f.state_names[target])))


def _expected_observational(
    model: DiscreteBayesianNetwork, target: str,
    scoring: Callable[[str], float],
    evidence: Optional[Dict[str, str]] = None,
) -> float:
    ve = VariableElimination(model)
    f = ve.query(variables=[target], evidence=evidence, show_progress=False)
    return float(sum(float(p) * scoring(s) for p, s in zip(f.values, f.state_names[target])))


# ===================================================================
# Archetype: safety_constrained
# ===================================================================

def _build_safety_constrained(
    role_plan: RolePlan, rng: random.Random,
) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, Dict[str, Any]]:
    R = role_plan.roles
    target = R["target"]; safety = R["safety_outcome"]
    good = R["good_intervention"]; risky = R["risky_intervention"]; weak = R["weak_intervention"]
    gmed = R["good_mediator"]; conf = R["confounder"]

    required = [
        (conf, target),
        (conf, safety),
        (good, gmed),
        (gmed, target),
        (risky, target),
        (risky, safety),
        (weak, target),
    ]
    forbidden = [
        (good, safety),       # good intervention must be SAFE
        (good, target),       # good must work via mediator only
        (target, safety), (safety, target),
        (target, conf), (safety, conf),
    ]
    g = _add_required_and_background(role_plan, required, forbidden, rng)

    sn = {v.name: list(v.values) for v in role_plan.variables}
    K_t = len(sn[target]); K_s = len(sn[safety]); K_m = len(sn[gmed])

    # Effects (lower index = better for preferred_low targets/safety; we
    # apply directional shifts that favor early states for target/safety
    # under "best" intervention states.)
    target_var = role_plan.by_name[target]
    safety_var = role_plan.by_name[safety]
    sign_t = 1.0 if (target_var.preferred_low is not False) else -1.0
    sign_s = 1.0 if (safety_var.preferred_low is not False) else -1.0

    # Pick the "Optimal" state of each intervention — the LAST listed state
    # by convention (e.g. "On", "Weekly", "High"). For 2-state vars that's
    # state[1].
    good_opt = sn[good][-1]
    risky_opt = sn[risky][-1]
    weak_opt = sn[weak][-1]
    conf_low, conf_high = sn[conf][0], sn[conf][-1]

    # good_mediator CPD: strong response to good intervention
    eff_gmed = {
        good: {
            sn[good][0]: _direction_vector(K_m, +1.0, 0.6),  # off → bad mediator
            sn[good][-1]: _direction_vector(K_m, -1.0, 2.4), # on → good mediator
        },
    }
    # if good has 3 states, mid is in between
    if len(sn[good]) >= 3:
        eff_gmed[good][sn[good][1]] = _direction_vector(K_m, -1.0, 1.2)
    cpd_gmed = _directional_cpd(
        gmed, [good], sn, baseline_logits=np.zeros(K_m),
        effects=eff_gmed, rng=rng,
    )

    # target CPD: parents = [conf, gmed, risky, weak] (after _add_required edges
    # may add extras, but we forbade edges into target except above; the DAG
    # has exactly these four parents).
    target_parents = sorted(g.predecessors(target),
                            key=lambda p: [conf, gmed, risky, weak].index(p)
                            if p in (conf, gmed, risky, weak) else 99)
    # Convention: gmed states are listed BEST→WORST per the role-plan prompt
    # (it's an outcome-class role).  So state[0] is the GOOD mediator state and
    # state[-1] is the WORST.  When the mediator is GOOD, target should be
    # better; when WORST, target should be worse.  Earlier code had this
    # flipped, producing worlds where do(good_intv=On) made the target WORSE
    # because the mediator path was inverted.
    eff_target = {
        conf: {
            conf_high: _direction_vector(K_t,  +sign_t, 1.6),
            conf_low:  _direction_vector(K_t,  -sign_t, 0.6),
        },
        gmed: {
            sn[gmed][0]:  _direction_vector(K_t, -sign_t, 1.6),   # GOOD med → BETTER target
            sn[gmed][-1]: _direction_vector(K_t, +sign_t, 1.4),   # WORST med → WORSE target
        },
        risky: {
            sn[risky][0]:  _direction_vector(K_t, +sign_t, 0.4),
            sn[risky][-1]: _direction_vector(K_t, -sign_t, 2.2),  # strongest direct
        },
        weak: {
            sn[weak][0]:  _direction_vector(K_t, +sign_t, 0.1),
            sn[weak][-1]: _direction_vector(K_t, -sign_t, 0.4),
        },
    }
    if len(sn[gmed]) >= 3:
        eff_target[gmed][sn[gmed][1]] = _direction_vector(K_t, 0.0, 0.0)
    if len(sn[risky]) >= 3:
        eff_target[risky][sn[risky][1]] = _direction_vector(K_t, -sign_t, 1.0)
    cpd_target = _directional_cpd(
        target, target_parents, sn, baseline_logits=np.zeros(K_t),
        effects=eff_target, rng=rng,
    )

    # safety CPD: parents = [conf, risky]; risky_opt is harmful, conf_high also worsens.
    safety_parents = sorted(g.predecessors(safety),
                            key=lambda p: [conf, risky].index(p)
                            if p in (conf, risky) else 99)
    eff_safety = {
        conf: {
            conf_high: _direction_vector(K_s, +sign_s, 1.4),
            conf_low:  _direction_vector(K_s, -sign_s, 0.5),
        },
        risky: {
            sn[risky][0]:  _direction_vector(K_s, -sign_s, 0.2),
            sn[risky][-1]: _direction_vector(K_s, +sign_s, 1.8),  # risky harms safety
        },
    }
    if len(sn[risky]) >= 3:
        eff_safety[risky][sn[risky][1]] = _direction_vector(K_s, +sign_s, 0.7)
    cpd_safety = _directional_cpd(
        safety, safety_parents, sn, baseline_logits=np.zeros(K_s),
        effects=eff_safety, rng=rng,
    )

    controlled = {gmed: cpd_gmed, target: cpd_target, safety: cpd_safety}
    bn = _build_bn_with_controlled(g, role_plan, controlled, rng)
    return g, bn, {
        "good_opt": good_opt, "risky_opt": risky_opt, "weak_opt": weak_opt,
    }


def _validate_safety_constrained(
    bn: DiscreteBayesianNetwork, role_plan: RolePlan, build_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    R = role_plan.roles
    target = R["target"]; safety = R["safety_outcome"]
    good = R["good_intervention"]; risky = R["risky_intervention"]; weak = R["weak_intervention"]
    target_var = role_plan.by_name[target]
    safety_var = role_plan.by_name[safety]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    if safety_var.preferred_low is None:
        safety_var.preferred_low = True
    score_t = _scoring_for(target_var)
    score_s = _scoring_for(safety_var)

    base_t = _expected_observational(bn, target, score_t)
    base_s = _expected_observational(bn, safety, score_s)

    rows: List[Dict[str, Any]] = []
    for var in (good, risky, weak):
        for val in role_plan.by_name[var].values:
            et = _expected_under_do(bn, target, {var: val}, score_t)
            es = _expected_under_do(bn, safety, {var: val}, score_s)
            rows.append({
                "variable": var, "value": val,
                "expected_target": et, "expected_safety": es,
                "delta_target": et - base_t, "delta_safety": es - base_s,
                "safe": (es - base_s) <= SAFETY_TOLERANCE,
                "improves_target": (et - base_t) <= -MIN_EFFECT,
            })

    safe_improvers = [r for r in rows if r["safe"] and r["improves_target"]]
    if not safe_improvers:
        return False, "no safe-and-improving intervention exists", {}

    # Risky should worsen safety (otherwise there's no real tradeoff).  We
    # don't strictly require risky to also improve target more than the best
    # safe action — the test of "pick safe over the apparently-attractive
    # risky" works as long as risky is meaningfully tempting on either
    # dimension and it's clearly unsafe.
    best_safe = min(safe_improvers, key=lambda r: r["delta_target"])
    risky_rows = [r for r in rows if r["variable"] == risky
                  and r["value"] == role_plan.by_name[risky].values[-1]]
    if not risky_rows:
        return False, "risky intervention not enumerated", {}
    if risky_rows[0]["delta_safety"] <= SAFETY_TOLERANCE:
        return False, "risky intervention does not actually worsen safety", {}
    # Risky must at least improve target (else it's not tempting at all)
    if risky_rows[0]["delta_target"] >= -MIN_EFFECT * 0.5:
        return False, "risky intervention does not even tempt on target", {}

    # Best safe must beat runner-up safe by a margin
    safe_sorted = sorted(safe_improvers, key=lambda r: r["delta_target"])
    if len(safe_sorted) >= 2:
        gap = safe_sorted[1]["delta_target"] - safe_sorted[0]["delta_target"]
        if gap < MIN_TOP_GAP / 2:
            # OK: multiple acceptable safe options — store all as gold
            pass

    gold = {
        "best_safe_action": {
            "variable": best_safe["variable"], "value": best_safe["value"],
        },
        "all_safe_improvers": [
            {"variable": r["variable"], "value": r["value"],
             "delta_target": round(r["delta_target"], 4),
             "delta_safety": round(r["delta_safety"], 4)}
            for r in safe_improvers
        ],
        "rejected_risky": [
            {"variable": r["variable"], "value": r["value"],
             "delta_target": round(r["delta_target"], 4),
             "delta_safety": round(r["delta_safety"], 4),
             "reason": "worsens safety"}
            for r in rows
            if (not r["safe"]) and r["improves_target"]
        ],
        "baselines": {"E_target": round(base_t, 4), "E_safety": round(base_s, 4)},
        "safety_tolerance": SAFETY_TOLERANCE,
        "min_effect": MIN_EFFECT,
    }
    return True, "ok", gold


def _question_safety_constrained(role_plan: RolePlan, gold: Dict[str, Any]) -> str:
    R = role_plan.roles
    target = role_plan.by_name[R["target"]]
    safety = role_plan.by_name[R["safety_outcome"]]
    return (
        f"You want to {_shift_toward(target)}, but only if the same action "
        f"does not {_make_worse(safety)}. Investigate the system and "
        f"recommend a single intervention you would deploy."
    )


# ===================================================================
# Archetype: confounding_reversal
# ===================================================================

def _build_confounding_reversal(
    role_plan: RolePlan, rng: random.Random,
) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, Dict[str, Any]]:
    R = role_plan.roles
    T = R["treatment"]; M = R["mediator"]; O = R["outcome"]; C = R["confounder"]
    required = [
        (C, T), (C, O),
        (T, M), (M, O),
    ]
    # Forbid direct T→O so the only causal pathway is T→M→O (cleaner reversal)
    forbidden = [
        (T, O), (O, T), (M, T), (M, C), (T, C), (O, C),
    ]
    g = _add_required_and_background(role_plan, required, forbidden, rng)

    sn = {v.name: list(v.values) for v in role_plan.variables}
    K_T = len(sn[T]); K_M = len(sn[M]); K_O = len(sn[O])
    outcome_var = role_plan.by_name[O]
    if outcome_var.preferred_low is None:
        outcome_var.preferred_low = True
    sign_o = 1.0 if outcome_var.preferred_low else -1.0
    c_high = sn[C][-1]; c_low = sn[C][0]
    t_treated = sn[T][-1]; t_untreated = sn[T][0]

    # Treatment CPD: confounder strongly drives assignment toward "treated"
    eff_T = {
        C: {
            c_high: _direction_vector(K_T, +1.0, 2.4),
            c_low:  _direction_vector(K_T, -1.0, 1.6),
        },
    }
    cpd_T = _directional_cpd(
        T, [C], sn, baseline_logits=np.zeros(K_T), effects=eff_T, rng=rng,
    )

    # Mediator CPD: treatment improves mediator state (early index)
    eff_M = {
        T: {
            t_treated:   _direction_vector(K_M, -1.0, 2.0),  # treated → "good" early state
            t_untreated: _direction_vector(K_M, +1.0, 0.6),
        },
    }
    cpd_M = _directional_cpd(
        M, [T], sn, baseline_logits=np.zeros(K_M), effects=eff_M, rng=rng,
    )

    # Outcome CPD: confounder strongly worsens outcome; mediator improves it
    eff_O = {
        C: {
            c_high: _direction_vector(K_O, +sign_o, 2.4),  # confounder worsens
            c_low:  _direction_vector(K_O, -sign_o, 1.4),
        },
        M: {
            sn[M][0]:  _direction_vector(K_O, -sign_o, 1.5),  # good mediator helps
            sn[M][-1]: _direction_vector(K_O, +sign_o, 1.0),
        },
    }
    if K_M >= 3:
        eff_O[M][sn[M][1]] = _direction_vector(K_O, 0.0, 0.0)
    cpd_O = _directional_cpd(
        O, sorted(g.predecessors(O), key=lambda p: [C, M].index(p) if p in (C, M) else 99),
        sn, baseline_logits=np.zeros(K_O), effects=eff_O, rng=rng,
    )

    controlled = {T: cpd_T, M: cpd_M, O: cpd_O}
    bn = _build_bn_with_controlled(g, role_plan, controlled, rng)
    return g, bn, {"t_treated": t_treated, "t_untreated": t_untreated}


def _validate_confounding_reversal(
    bn: DiscreteBayesianNetwork, role_plan: RolePlan, build_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    R = role_plan.roles
    T = R["treatment"]; O = R["outcome"]; C = R["confounder"]
    outcome_var = role_plan.by_name[O]
    if outcome_var.preferred_low is None:
        outcome_var.preferred_low = True
    score = _scoring_for(outcome_var)

    t_treated = build_meta["t_treated"]; t_untreated = build_meta["t_untreated"]
    obs_treated = _expected_observational(bn, O, score, evidence={T: t_treated})
    obs_untreated = _expected_observational(bn, O, score, evidence={T: t_untreated})
    obs_diff = obs_treated - obs_untreated

    int_treated = _expected_under_do(bn, O, {T: t_treated}, score)
    int_untreated = _expected_under_do(bn, O, {T: t_untreated}, score)
    int_diff = int_treated - int_untreated

    if obs_diff * int_diff >= 0:
        return False, (f"no sign reversal: obs Δ={obs_diff:+.3f} int Δ={int_diff:+.3f}"), {}
    if abs(obs_diff) < REVERSAL_MARGIN:
        return False, f"observational gap too small ({obs_diff:+.3f})", {}
    if abs(int_diff) < REVERSAL_MARGIN:
        return False, f"interventional gap too small ({int_diff:+.3f})", {}

    direction = "harmful" if obs_diff > 0 else "beneficial"
    causal_direction = "beneficial" if int_diff < 0 else "harmful"
    gold = {
        "treatment": T, "outcome": O,
        "confounder": C, "confounder_name": C,
        "treated_value": t_treated, "untreated_value": t_untreated,
        "obs_E_outcome_treated": round(obs_treated, 4),
        "obs_E_outcome_untreated": round(obs_untreated, 4),
        "obs_diff": round(obs_diff, 4),  # >0 means treated looks worse
        "int_E_outcome_treated": round(int_treated, 4),
        "int_E_outcome_untreated": round(int_untreated, 4),
        "int_diff": round(int_diff, 4),  # <0 means treated is causally better
        "observational_appearance": direction,    # "harmful" / "beneficial"
        "causal_truth": causal_direction,         # "helpful" alias intended; preserve "beneficial"/"harmful"
        "is_confounded": True,
        "preferred_low": outcome_var.preferred_low,
    }
    return True, "ok", gold


def _question_confounding_reversal(role_plan: RolePlan, gold: Dict[str, Any]) -> str:
    R = role_plan.roles
    T = role_plan.by_name[R["treatment"]]
    O = role_plan.by_name[R["outcome"]]
    best_O = O.values[0]
    worst_O = O.values[-1]
    return (
        f"What is the actual causal relationship between {T.name} and "
        f"{O.name}? Specifically: if you set {T.name}={gold['treated_value']} "
        f"versus {T.name}={gold['untreated_value']}, does that make {O.name} "
        f"more likely to be {best_O}, more likely to be {worst_O}, or have "
        f"no real effect? Also explain whether what you observe in passive "
        f"data alone would mislead you about this relationship — and if so, "
        f"why."
    )


# ===================================================================
# Archetype: mediator_structure
# ===================================================================

def _build_mediator_structure(
    role_plan: RolePlan, rng: random.Random,
) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, Dict[str, Any]]:
    R = role_plan.roles
    sub = role_plan.sub_variant
    T = R["treatment"]; O = R["outcome"]; C = R["confounder"]

    if sub == "mediated_only":
        M = R["mediator"]
        required = [(T, M), (M, O), (C, T), (C, O)]
        forbidden = [(T, O), (O, T), (O, M), (M, T), (M, C), (T, C), (O, C)]
        controlled_outcome_parents = [C, M]
        meta = {"M": M}
    elif sub == "direct_and_mediated":
        M = R["mediator"]
        required = [(T, M), (M, O), (T, O), (C, T), (C, O)]
        forbidden = [(O, T), (O, M), (M, T), (M, C), (T, C), (O, C)]
        controlled_outcome_parents = [C, M, T]
        meta = {"M": M}
    elif sub == "not_mediator":
        M_proposed = R["proposed_M"]; M_true = R["true_mediator"]
        required = [(T, M_true), (M_true, O), (T, M_proposed), (C, T), (C, O)]
        # Critical: M_proposed must NOT lie on a path to O.
        forbidden = [
            (M_proposed, O), (T, O), (O, T),
            (M_true, T), (M_proposed, M_true), (M_true, M_proposed),
            (M_true, C), (M_proposed, C), (T, C), (O, C), (O, M_true), (O, M_proposed),
        ]
        controlled_outcome_parents = [C, M_true]
        meta = {"M_proposed": M_proposed, "M_true": M_true}
    elif sub == "which_mediator":
        M_true = R["true_mediator"]; D1 = R["decoy_M1"]; D2 = R["decoy_M2"]
        required = [(T, M_true), (M_true, O), (T, D1), (D2, O), (C, T), (C, O)]
        # D1 is reached by T but not on path to O.
        # D2 is on path to O but not reached by T.
        forbidden = [
            (D1, O), (T, D2), (T, O), (O, T),
            (M_true, T), (D1, M_true), (M_true, D1), (M_true, D2), (D2, M_true),
            (M_true, C), (D1, C), (D2, C), (T, C), (O, C),
        ]
        controlled_outcome_parents = [C, M_true, D2]
        meta = {"M_true": M_true, "D1": D1, "D2": D2}
    else:
        raise ValueError(f"unknown sub_variant {sub!r}")

    g = _add_required_and_background(role_plan, required, forbidden, rng)

    sn = {v.name: list(v.values) for v in role_plan.variables}
    outcome_var = role_plan.by_name[O]
    if outcome_var.preferred_low is None:
        outcome_var.preferred_low = True
    sign_o = 1.0 if outcome_var.preferred_low else -1.0
    K_O = len(sn[O])

    controlled: Dict[str, TabularCPD] = {}

    # Treatment CPD: not strictly required to be controlled; let strong-logistic
    # handle it via parent C.

    # Mediator(s) CPD
    if sub in ("mediated_only", "direct_and_mediated"):
        M = R["mediator"]; K_M = len(sn[M])
        eff_M = {
            T: {
                sn[T][-1]: _direction_vector(K_M, -1.0, 2.0),
                sn[T][0]:  _direction_vector(K_M, +1.0, 0.6),
            },
        }
        controlled[M] = _directional_cpd(
            M, [T], sn, baseline_logits=np.zeros(K_M), effects=eff_M, rng=rng,
        )
    elif sub == "not_mediator":
        M_t = meta["M_true"]; M_p = meta["M_proposed"]
        K_Mt = len(sn[M_t]); K_Mp = len(sn[M_p])
        controlled[M_t] = _directional_cpd(
            M_t, [T], sn, baseline_logits=np.zeros(K_Mt),
            effects={T: {
                sn[T][-1]: _direction_vector(K_Mt, -1.0, 2.0),
                sn[T][0]:  _direction_vector(K_Mt, +1.0, 0.6),
            }},
            rng=rng,
        )
        controlled[M_p] = _directional_cpd(
            M_p, [T], sn, baseline_logits=np.zeros(K_Mp),
            effects={T: {
                sn[T][-1]: _direction_vector(K_Mp, -1.0, 1.6),  # moves but is not on path
                sn[T][0]:  _direction_vector(K_Mp, +1.0, 0.4),
            }},
            rng=rng,
        )
    else:  # which_mediator
        M_t = meta["M_true"]; D1 = meta["D1"]; D2 = meta["D2"]
        K_Mt = len(sn[M_t]); K_D1 = len(sn[D1]); K_D2 = len(sn[D2])
        controlled[M_t] = _directional_cpd(
            M_t, [T], sn, baseline_logits=np.zeros(K_Mt),
            effects={T: {
                sn[T][-1]: _direction_vector(K_Mt, -1.0, 2.0),
                sn[T][0]:  _direction_vector(K_Mt, +1.0, 0.6),
            }},
            rng=rng,
        )
        controlled[D1] = _directional_cpd(
            D1, [T], sn, baseline_logits=np.zeros(K_D1),
            effects={T: {
                sn[T][-1]: _direction_vector(K_D1, -1.0, 1.4),
                sn[T][0]:  _direction_vector(K_D1, +1.0, 0.4),
            }},
            rng=rng,
        )
        # D2: parent of O via random root CPD (no parents)
        # (controlled handled by root cpd in builder)

    # Outcome CPD
    K_O = len(sn[O])
    parents_O = sorted(
        g.predecessors(O), key=lambda p: controlled_outcome_parents.index(p)
        if p in controlled_outcome_parents else 99,
    )
    c_high = sn[C][-1]; c_low = sn[C][0]
    eff_O: Dict[str, Dict[str, np.ndarray]] = {
        C: {
            c_high: _direction_vector(K_O, +sign_o, 1.6),
            c_low:  _direction_vector(K_O, -sign_o, 0.8),
        },
    }
    if sub in ("mediated_only", "direct_and_mediated"):
        M = R["mediator"]
        eff_O[M] = {
            sn[M][0]:  _direction_vector(K_O, -sign_o, 1.6),
            sn[M][-1]: _direction_vector(K_O, +sign_o, 1.0),
        }
        if sub == "direct_and_mediated":
            eff_O[T] = {
                sn[T][-1]: _direction_vector(K_O, -sign_o, 1.4),
                sn[T][0]:  _direction_vector(K_O, +sign_o, 0.4),
            }
    elif sub == "not_mediator":
        M_t = meta["M_true"]
        eff_O[M_t] = {
            sn[M_t][0]:  _direction_vector(K_O, -sign_o, 1.6),
            sn[M_t][-1]: _direction_vector(K_O, +sign_o, 1.0),
        }
    else:  # which_mediator
        M_t = meta["M_true"]; D2 = meta["D2"]
        eff_O[M_t] = {
            sn[M_t][0]:  _direction_vector(K_O, -sign_o, 1.6),
            sn[M_t][-1]: _direction_vector(K_O, +sign_o, 1.0),
        }
        eff_O[D2] = {
            sn[D2][0]:  _direction_vector(K_O, -sign_o, 1.0),
            sn[D2][-1]: _direction_vector(K_O, +sign_o, 0.6),
        }

    controlled[O] = _directional_cpd(
        O, parents_O, sn, baseline_logits=np.zeros(K_O), effects=eff_O, rng=rng,
    )
    bn = _build_bn_with_controlled(g, role_plan, controlled, rng)
    return g, bn, meta


def _validate_mediator_structure(
    bn: DiscreteBayesianNetwork, role_plan: RolePlan, build_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    sub = role_plan.sub_variant
    R = role_plan.roles
    T = R["treatment"]; O = R["outcome"]
    outcome_var = role_plan.by_name[O]
    if outcome_var.preferred_low is None:
        outcome_var.preferred_low = True
    score = _scoring_for(outcome_var)
    sn_T = list(role_plan.by_name[T].values)
    t_high = sn_T[-1]; t_low = sn_T[0]

    # (1) Total causal effect of T on O must be visible
    int_high = _expected_under_do(bn, O, {T: t_high}, score)
    int_low = _expected_under_do(bn, O, {T: t_low}, score)
    total_effect = int_high - int_low
    if abs(total_effect) < MIN_EFFECT:
        return False, f"total causal effect too small ({total_effect:+.3f})", {}

    # (2) Sub-variant-specific signature
    if sub == "mediated_only" or sub == "direct_and_mediated":
        M = R["mediator"]
        sn_M = list(role_plan.by_name[M].values)
        # Effect of T after fixing M to a single state
        residual_diffs = []
        for m_val in sn_M:
            r_high = _expected_under_do(bn, O, {T: t_high, M: m_val}, score)
            r_low = _expected_under_do(bn, O, {T: t_low, M: m_val}, score)
            residual_diffs.append(r_high - r_low)
        max_residual = max(abs(d) for d in residual_diffs)
        if sub == "mediated_only":
            if max_residual > 0.10:
                return False, f"mediated_only: residual T effect after fixing M too large ({max_residual:.3f})", {}
            label = "only_through_M"
        else:  # direct_and_mediated
            if max_residual < 0.18:
                return False, f"direct_and_mediated: residual too small ({max_residual:.3f})", {}
            label = "also_direct_or_other"
        gold = {
            "label": label, "T": T, "M": M, "O": O,
            "total_effect": round(total_effect, 4),
            "max_residual_effect": round(max_residual, 4),
        }
        return True, "ok", gold

    if sub == "not_mediator":
        M_p = build_meta["M_proposed"]; M_t = build_meta["M_true"]
        sn_Mp = list(role_plan.by_name[M_p].values)
        # Fixing the *proposed* mediator should not eliminate T effect
        residual_diffs = []
        for mp_val in sn_Mp:
            r_high = _expected_under_do(bn, O, {T: t_high, M_p: mp_val}, score)
            r_low = _expected_under_do(bn, O, {T: t_low, M_p: mp_val}, score)
            residual_diffs.append(r_high - r_low)
        max_residual = max(abs(d) for d in residual_diffs)
        if max_residual < 0.18:
            return False, f"not_mediator: T effect disappears when fixing proposed M ({max_residual:.3f})", {}
        # Also: do(M_p=val) should not move O much
        sn_O = list(role_plan.by_name[O].values)
        score = _scoring_for(outcome_var)
        e_high = _expected_under_do(bn, O, {M_p: sn_Mp[-1]}, score)
        e_low = _expected_under_do(bn, O, {M_p: sn_Mp[0]}, score)
        if abs(e_high - e_low) > 0.10:
            return False, f"not_mediator: do(M_p) still moves O ({abs(e_high - e_low):.3f})", {}
        gold = {
            "label": "not_mediator", "T": T, "M_proposed": M_p, "M_true": M_t, "O": O,
            "total_effect": round(total_effect, 4),
            "residual_after_fix_proposed": round(max_residual, 4),
            "do_proposed_effect_on_O": round(e_high - e_low, 4),
        }
        return True, "ok", gold

    if sub == "which_mediator":
        M_t = build_meta["M_true"]; D1 = build_meta["D1"]; D2 = build_meta["D2"]
        # The candidate names that the agent will be asked about.
        candidates = [M_t, D1, D2]
        # For each candidate, intervening on it should move O only if it's
        # truly on the T→O path.
        sn_O = list(role_plan.by_name[O].values)
        cand_effects = {}
        for cand in candidates:
            sn_c = list(role_plan.by_name[cand].values)
            e_high = _expected_under_do(bn, O, {cand: sn_c[-1]}, score)
            e_low = _expected_under_do(bn, O, {cand: sn_c[0]}, score)
            cand_effects[cand] = e_high - e_low
        # Validate: M_true has biggest absolute effect on O, and decoys are weaker
        best = max(candidates, key=lambda c: abs(cand_effects[c]))
        if best != M_t:
            return False, f"which_mediator: do(M_true) effect not strongest ({cand_effects})", {}
        rivals = [c for c in candidates if c != M_t]
        gap = abs(cand_effects[M_t]) - max(abs(cand_effects[c]) for c in rivals)
        if gap < MIN_TOP_GAP:
            return False, f"which_mediator: gap to runner-up too small ({gap:.3f})", {}
        gold = {
            "label": "which_mediator", "T": T, "O": O,
            "true_mediator": M_t, "candidates": candidates,
            "candidate_do_effects": {k: round(v, 4) for k, v in cand_effects.items()},
            "total_effect": round(total_effect, 4),
        }
        return True, "ok", gold

    return False, f"unknown sub_variant {sub!r}", {}


def _question_mediator_structure(role_plan: RolePlan, gold: Dict[str, Any]) -> str:
    R = role_plan.roles
    sub = role_plan.sub_variant
    T = role_plan.by_name[R["treatment"]]
    O = role_plan.by_name[R["outcome"]]
    if sub == "mediated_only" or sub == "direct_and_mediated":
        M = role_plan.by_name[R["mediator"]]
        return (
            f"How does {T.name} change {O.name}? Specifically, does the "
            f"effect of {T.name} on {O.name} flow entirely through "
            f"{M.name}, or is there also a separate pathway that does "
            f"not pass through {M.name}? Investigate using interventions "
            f"and explain your conclusion."
        )
    if sub == "not_mediator":
        M_p = role_plan.by_name[R["proposed_M"]]
        return (
            f"An analyst hypothesizes that {M_p.name} is the variable "
            f"through which {T.name} changes {O.name}. Investigate the "
            f"system and decide whether this hypothesis is supported, "
            f"and explain why."
        )
    if sub == "which_mediator":
        cands = ", ".join(role_plan.by_name[c].name for c in gold["candidates"])
        return (
            f"Among the variables [{cands}], which one is on the actual "
            f"causal pathway from {T.name} to {O.name}? Identify the "
            f"single variable and explain how you ruled out the others."
        )
    return "?"


# ===================================================================
# Archetype: satisficing
# ===================================================================

def _build_satisficing(
    role_plan: RolePlan, rng: random.Random,
) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, Dict[str, Any]]:
    R = role_plan.roles
    target = R["target"]; conf = R["confounder"]
    intvs = [R["intervention_A"], R["intervention_B"], R["intervention_C"], R["intervention_D"]]
    required = [(conf, target)] + [(x, target) for x in intvs]
    forbidden = [(target, x) for x in intvs] + [(target, conf)]
    g = _add_required_and_background(role_plan, required, forbidden, rng)

    sn = {v.name: list(v.values) for v in role_plan.variables}
    target_var = role_plan.by_name[target]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    sign_t = 1.0 if target_var.preferred_low else -1.0
    K_T = len(sn[target])

    # Want a spread of effect sizes among A, B, C, D — some clearly above
    # threshold, some clearly below.
    intv_strengths = {
        intvs[0]: 2.4,   # strong
        intvs[1]: 1.8,   # moderate-strong
        intvs[2]: 0.5,   # weak
        intvs[3]: 0.15,  # near-zero
    }
    rng.shuffle(intvs)  # randomize which role gets which strength so
                        # the answer doesn't track the role-letter
    role_to_strength = {x: s for x, s in zip(intvs, sorted(intv_strengths.values(), reverse=True))}

    eff_target = {
        conf: {
            sn[conf][-1]: _direction_vector(K_T, +sign_t, 1.4),
            sn[conf][0]:  _direction_vector(K_T, -sign_t, 0.6),
        },
    }
    for x in intvs:
        eff_target[x] = {
            sn[x][-1]: _direction_vector(K_T, -sign_t, role_to_strength[x]),
            sn[x][0]:  _direction_vector(K_T, +sign_t, role_to_strength[x] * 0.25),
        }
        if len(sn[x]) >= 3:
            eff_target[x][sn[x][1]] = _direction_vector(K_T, -sign_t, role_to_strength[x] * 0.5)

    parents_T = sorted(
        g.predecessors(target),
        key=lambda p: ([conf] + intvs).index(p) if p in [conf] + intvs else 99,
    )
    cpd_T = _directional_cpd(
        target, parents_T, sn, baseline_logits=np.zeros(K_T),
        effects=eff_target, rng=rng,
    )
    bn = _build_bn_with_controlled(g, role_plan, {target: cpd_T}, rng)
    return g, bn, {"intvs": intvs, "strengths": role_to_strength}


def _make_threshold_robust(values: List[float], guard_eps: float = 0.05) -> Optional[float]:
    """Pick a threshold that sits in the largest gap of `values` (sorted desc).

    Returns None if no acceptable gap exists.
    """
    if not values:
        return None
    sv = sorted(values, reverse=True)
    sv_pos = [v for v in sv if v > 0]
    if not sv_pos:
        return None
    # Use values + 0 as candidate boundaries
    boundaries = sv_pos + [0.0]
    gaps = [(boundaries[i] - boundaries[i + 1], (boundaries[i] + boundaries[i + 1]) / 2)
            for i in range(len(boundaries) - 1)]
    gaps = [(g, mid) for g, mid in gaps if g >= guard_eps * 2]
    if not gaps:
        return None
    best = max(gaps, key=lambda x: x[0])
    return float(best[1])


def _validate_satisficing(
    bn: DiscreteBayesianNetwork, role_plan: RolePlan, build_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    R = role_plan.roles
    target = R["target"]
    target_var = role_plan.by_name[target]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    score = _scoring_for(target_var)

    base = _expected_observational(bn, target, score)
    intvs = build_meta["intvs"]
    rows: List[Dict[str, Any]] = []
    for var in intvs:
        for val in role_plan.by_name[var].values:
            et = _expected_under_do(bn, target, {var: val}, score)
            rows.append({
                "variable": var, "value": val,
                "expected_target": et,
                "improvement": -(et - base),  # positive = improvement (lower is better)
            })

    # Choose threshold: sits in the largest gap among positive improvements
    improvements = [r["improvement"] for r in rows if r["improvement"] > 0]
    threshold = _make_threshold_robust(improvements, guard_eps=0.05)
    if threshold is None:
        return False, f"no satisficing threshold has a clean gap (improvements={improvements})", {}

    feasible = [r for r in rows if r["improvement"] >= threshold]
    if not feasible:
        return False, "no feasible intervention above threshold", {}
    if len(feasible) >= len(rows) - 1:
        return False, f"too many feasible interventions ({len(feasible)}/{len(rows)})", {}

    gold = {
        "feasible_actions": [
            {"variable": r["variable"], "value": r["value"],
             "improvement": round(r["improvement"], 4)}
            for r in feasible
        ],
        "all_actions": [
            {"variable": r["variable"], "value": r["value"],
             "improvement": round(r["improvement"], 4)}
            for r in rows
        ],
        "threshold": round(threshold, 4),
        "baseline_E_target": round(base, 4),
    }
    return True, "ok", gold


def _question_satisficing(role_plan: RolePlan, gold: Dict[str, Any]) -> str:
    R = role_plan.roles
    target = role_plan.by_name[R["target"]]
    return (
        f"Find an action that meaningfully {_shifts_toward(target)}. "
        f"Investigate the system and recommend one intervention you "
        f"would deploy, or report that nothing in the available actions "
        f"meaningfully changes {target.name}."
    )


# ===================================================================
# Archetype: subgroup_robust
# ===================================================================

def _build_subgroup_robust(
    role_plan: RolePlan, rng: random.Random,
) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, Dict[str, Any]]:
    R = role_plan.roles
    target = R["target"]; group = R["group"]; conf = R["confounder"]
    avg = R["intervention_avg"]; rob = R["intervention_rob"]; bad = R["intervention_bad"]
    required = [
        (group, target),
        (conf, target),
        (avg, target), (rob, target), (bad, target),
    ]
    forbidden = [
        (target, group), (target, conf), (target, avg),
        (target, rob), (target, bad),
        (avg, group), (rob, group), (bad, group),
    ]
    g = _add_required_and_background(role_plan, required, forbidden, rng)

    sn = {v.name: list(v.values) for v in role_plan.variables}
    target_var = role_plan.by_name[target]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    sign_t = 1.0 if target_var.preferred_low else -1.0
    K_T = len(sn[target])
    g_states = sn[group]
    if len(g_states) < 2:
        raise RuntimeError(f"subgroup_robust: group var {group} has <2 states")
    g_low = g_states[0]; g_high = g_states[-1]

    # We want target CPD to depend on (group × intervention) for a real
    # interaction.  Build effects accordingly.
    eff_target: Dict[str, Dict[str, np.ndarray]] = {
        conf: {
            sn[conf][-1]: _direction_vector(K_T, +sign_t, 1.4),
            sn[conf][0]:  _direction_vector(K_T, -sign_t, 0.6),
        },
        group: {
            g_high: _direction_vector(K_T, +sign_t, 0.6),
            g_low:  _direction_vector(K_T, -sign_t, 0.4),
        },
        avg: {
            # avg helps a lot in g_high (which dominates baseline) but does
            # nothing in g_low; we approximate by giving avg a big direct
            # logit shift independent of group, then add a counter-shift via
            # large negative shift conditional ON g_low (we can't directly
            # condition, but we encode interaction by tweaking the bias of
            # `avg=on`'s contribution and pairing with group's effect).
            #
            # Concrete trick: shift avg's contribution by -2.0 in g_low to
            # cancel improvement.  We do this by giving group an extra
            # "interaction" effect on target that is parameterized as if
            # group + avg jointly.  Since pgmpy CPDs are full joint over
            # all parents, this still yields exact behavior; the
            # _directional_cpd primitive sums per-parent shifts, so we use
            # a separate "joint" entry.
            sn[avg][-1]: _direction_vector(K_T, -sign_t, 1.4),
            sn[avg][0]:  _direction_vector(K_T, +sign_t, 0.4),
        },
        rob: {
            sn[rob][-1]: _direction_vector(K_T, -sign_t, 1.0),
            sn[rob][0]:  _direction_vector(K_T, +sign_t, 0.3),
        },
        bad: {
            # `bad` helps in one group, harms in the other — encoded by
            # shifting in opposite directions per group.
            sn[bad][-1]: _direction_vector(K_T, -sign_t, 0.6),
            sn[bad][0]:  _direction_vector(K_T, +sign_t, 0.2),
        },
    }
    # Interaction: we don't have a per-cell mechanism in _directional_cpd, so
    # we emulate it by enriching parents on target with `group_x_avg` style
    # logic via a hand-built CPD.  Build the target CPD from scratch here.
    parents_T = sorted(
        g.predecessors(target),
        key=lambda p: ([conf, group, avg, rob, bad]).index(p)
        if p in [conf, group, avg, rob, bad] else 99,
    )
    parent_cards = [len(sn[p]) for p in parents_T]
    n_configs = int(np.prod(parent_cards))
    all_probs = []
    for cfg in range(n_configs):
        idx = []
        temp = cfg
        for c in reversed(parent_cards):
            idx.append(temp % c)
            temp //= c
        idx = list(reversed(idx))
        logits = np.zeros(K_T)
        # add per-parent additive shifts
        config_states = {p: sn[p][idx[i]] for i, p in enumerate(parents_T)}
        for p, shifts in eff_target.items():
            if p in config_states:
                shift = shifts.get(config_states[p])
                if shift is not None:
                    logits += shift
        # interaction term: avg helps a lot in g_high but only weakly in g_low
        if group in config_states and avg in config_states:
            if config_states[group] == g_low and config_states[avg] == sn[avg][-1]:
                logits += _direction_vector(K_T, +sign_t, 1.0)  # cancel ~most of avg's help
        # bad: helps in g_high, harms in g_low (large flip)
        if group in config_states and bad in config_states:
            if config_states[group] == g_low and config_states[bad] == sn[bad][-1]:
                logits += _direction_vector(K_T, +sign_t, 1.4)
            if config_states[group] == g_high and config_states[bad] == sn[bad][-1]:
                logits += _direction_vector(K_T, -sign_t, 0.4)
        logits += np.array([rng.uniform(-0.05, 0.05) for _ in range(K_T)])
        logits -= logits.max()
        probs = np.exp(logits); probs /= probs.sum()
        all_probs.append(probs)

    values = [[float(all_probs[j][i]) for j in range(n_configs)] for i in range(K_T)]
    cpd_T = TabularCPD(
        variable=target, variable_card=K_T, values=values,
        evidence=parents_T, evidence_card=parent_cards,
        state_names={target: sn[target], **{p: sn[p] for p in parents_T}},
    )
    bn = _build_bn_with_controlled(g, role_plan, {target: cpd_T}, rng)
    return g, bn, {"avg": avg, "rob": rob, "bad": bad,
                    "g_low": g_low, "g_high": g_high}


def _validate_subgroup_robust(
    bn: DiscreteBayesianNetwork, role_plan: RolePlan, build_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    R = role_plan.roles
    target = R["target"]; group = R["group"]
    target_var = role_plan.by_name[target]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    score = _scoring_for(target_var)

    avg = build_meta["avg"]; rob = build_meta["rob"]; bad = build_meta["bad"]
    g_low = build_meta["g_low"]; g_high = build_meta["g_high"]

    base_low = _expected_observational(bn, target, score, evidence={group: g_low})
    base_high = _expected_observational(bn, target, score, evidence={group: g_high})

    def _imp(var: str, val: str, grp: str) -> float:
        e = _expected_under_do(bn, target, {var: val}, score, evidence={group: grp})
        base = base_low if grp == g_low else base_high
        return -(e - base)  # positive = improvement

    rows = []
    for var in (avg, rob, bad):
        opt = role_plan.by_name[var].values[-1]
        i_low = _imp(var, opt, g_low)
        i_high = _imp(var, opt, g_high)
        i_min = min(i_low, i_high)
        rows.append({
            "variable": var, "value": opt,
            "improvement_low": i_low, "improvement_high": i_high,
            "min_improvement": i_min,
        })

    rob_row = next(r for r in rows if r["variable"] == rob)
    avg_row = next(r for r in rows if r["variable"] == avg)
    bad_row = next(r for r in rows if r["variable"] == bad)

    if rob_row["min_improvement"] < MIN_EFFECT * 0.5:
        return False, f"robust intervention does not help every subgroup ({rob_row})", {}
    if avg_row["min_improvement"] >= rob_row["min_improvement"] - 0.05:
        return False, f"avg-best intervention also helps every subgroup ({avg_row}, {rob_row})", {}
    if bad_row["min_improvement"] >= 0:
        return False, f"bad intervention is not actually bad in any subgroup ({bad_row})", {}

    # Enumerate every (intervenable_var, value) and accept any pair that
    # improves EVERY subgroup by at least accept_threshold.  This lets the
    # scorer credit any genuinely-robust intervention, not just the gold one.
    accept_threshold = MIN_EFFECT * 0.5
    acceptable_robust_actions: List[Dict[str, Any]] = []
    for v in role_plan.variables:
        if not v.intervenable or v.name == target or v.name == group:
            continue
        for val in v.values:
            try:
                i_lo = _imp(v.name, val, g_low)
                i_hi = _imp(v.name, val, g_high)
            except Exception:
                continue
            i_min = min(i_lo, i_hi)
            if i_min >= accept_threshold:
                acceptable_robust_actions.append({
                    "variable": v.name, "value": val,
                    "improvement_low": round(i_lo, 4),
                    "improvement_high": round(i_hi, 4),
                    "min_improvement": round(i_min, 4),
                })

    gold = {
        "robust_action": {
            "variable": rob_row["variable"], "value": rob_row["value"],
            "min_improvement": round(rob_row["min_improvement"], 4),
            "improvement_low": round(rob_row["improvement_low"], 4),
            "improvement_high": round(rob_row["improvement_high"], 4),
        },
        "avg_best_but_uneven": {
            "variable": avg_row["variable"], "value": avg_row["value"],
            "min_improvement": round(avg_row["min_improvement"], 4),
            "improvement_low": round(avg_row["improvement_low"], 4),
            "improvement_high": round(avg_row["improvement_high"], 4),
        },
        "harmful_in_subgroup": {
            "variable": bad_row["variable"], "value": bad_row["value"],
            "min_improvement": round(bad_row["min_improvement"], 4),
            "improvement_low": round(bad_row["improvement_low"], 4),
            "improvement_high": round(bad_row["improvement_high"], 4),
        },
        "subgroups": [g_low, g_high],
        "group_variable": group,
        "min_effect": MIN_EFFECT,
        "acceptable_robust_actions": acceptable_robust_actions,
        "accept_threshold": accept_threshold,
    }
    return True, "ok", gold


def _question_subgroup_robust(role_plan: RolePlan, gold: Dict[str, Any]) -> str:
    R = role_plan.roles
    target = role_plan.by_name[R["target"]]
    group = role_plan.by_name[R["group"]]
    sg_low, sg_high = gold["subgroups"][0], gold["subgroups"][1]
    return (
        f"Recommend an intervention you would deploy to "
        f"{_shift_toward(target)}. The deployment will reach individuals "
        f"with {group.name}={sg_low} as well as {group.name}={sg_high}, so "
        f"explain how your choice performs for each."
    )


# ===================================================================
# Archetype: invalid_premise
# ===================================================================

def _build_invalid_premise(
    role_plan: RolePlan, rng: random.Random,
) -> Tuple[nx.DiGraph, DiscreteBayesianNetwork, Dict[str, Any]]:
    R = role_plan.roles
    sub = role_plan.sub_variant or "non_intervenable_proxy"
    target = R["target"]; conf = R["confounder"]
    alt = R["intervenable_alt"]; med = R["alt_mediator"]
    if sub == "wrong_side_intervention":
        proposed = R["proposed_intervention"]
        required = [
            (target, proposed),      # associated with target, but downstream
            (conf, target),
            (alt, med), (med, target),
        ]
        forbidden = [
            (proposed, target), (proposed, med), (proposed, alt),
            (proposed, conf), (target, alt), (target, med), (target, conf),
            (alt, conf), (med, conf),
        ]
    elif sub == "valid_proposed_intervention":
        proposed = R["proposed_intervention"]
        required = [
            (proposed, med),         # proposal moves a real mechanism
            (med, target),
            (conf, target),
            (alt, target),           # plausible decoy action, kept weak/harmful
        ]
        forbidden = [
            (target, proposed), (target, alt), (target, med), (target, conf),
            (alt, med), (alt, proposed), (proposed, alt),
            (proposed, conf), (alt, conf), (med, conf),
        ]
    else:
        nix = R["non_intervenable_x"]
        required = [
            (nix, target),
            (conf, target),
            (alt, med), (med, target),
        ]
        forbidden = [
            (alt, nix), (med, nix), (target, nix), (target, alt), (target, med),
            (target, conf), (nix, alt), (nix, med),
        ]
    g = _add_required_and_background(role_plan, required, forbidden, rng)

    sn = {v.name: list(v.values) for v in role_plan.variables}
    target_var = role_plan.by_name[target]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    sign_t = 1.0 if target_var.preferred_low else -1.0
    K_T = len(sn[target])
    K_M = len(sn[med])

    # Mediator CPD
    med_parent = R["proposed_intervention"] if sub == "valid_proposed_intervention" else alt
    eff_med = {
        med_parent: {
            sn[med_parent][-1]: _direction_vector(K_M, -1.0, 2.0),
            sn[med_parent][0]:  _direction_vector(K_M, +1.0, 0.6),
        },
    }
    cpd_med = _directional_cpd(
        med, [med_parent], sn, baseline_logits=np.zeros(K_M),
        effects=eff_med, rng=rng,
    )

    controlled: Dict[str, TabularCPD] = {med: cpd_med}

    if sub == "wrong_side_intervention":
        # The proposed variable is made predictive by placing it downstream of
        # the target. Intervening on it still cannot move the target.
        proposed = R["proposed_intervention"]
        K_P = len(sn[proposed])
        eff_proposed = {
            target: {
                sn[target][-1]: _direction_vector(K_P, +1.0, 2.0),
                sn[target][0]:  _direction_vector(K_P, -1.0, 0.8),
            },
        }
        controlled[proposed] = _directional_cpd(
            proposed, [target], sn, baseline_logits=np.zeros(K_P),
            effects=eff_proposed, rng=rng,
        )

    # Target CPD
    target_parent_order = [conf, med]
    eff_T: Dict[str, Dict[str, np.ndarray]] = {
        conf: {
            sn[conf][-1]: _direction_vector(K_T, +sign_t, 1.0),
            sn[conf][0]:  _direction_vector(K_T, -sign_t, 0.4),
        },
        med: {
            sn[med][0]:  _direction_vector(K_T, -sign_t, 1.6),
            sn[med][-1]: _direction_vector(K_T, +sign_t, 0.8),
        },
    }
    if sub != "wrong_side_intervention":
        if sub == "valid_proposed_intervention":
            target_parent_order = [conf, med, alt]
            eff_T[alt] = {
                sn[alt][-1]: _direction_vector(K_T, +sign_t, 0.4),
                sn[alt][0]:  _direction_vector(K_T, -sign_t, 0.1),
            }
        else:
            nix = R["non_intervenable_x"]
            target_parent_order = [nix, conf, med]
            eff_T[nix] = {
                sn[nix][-1]: _direction_vector(K_T, +sign_t, 1.6),  # high nix worsens target
                sn[nix][0]:  _direction_vector(K_T, -sign_t, 0.6),
            }
    parents_T = sorted(
        g.predecessors(target),
        key=lambda p: target_parent_order.index(p) if p in target_parent_order else 99,
    )
    cpd_T = _directional_cpd(
        target, parents_T, sn, baseline_logits=np.zeros(K_T),
        effects=eff_T, rng=rng,
    )
    controlled[target] = cpd_T
    bn = _build_bn_with_controlled(g, role_plan, controlled, rng)
    return g, bn, {}


def _validate_invalid_premise(
    bn: DiscreteBayesianNetwork, role_plan: RolePlan, build_meta: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    R = role_plan.roles
    sub = role_plan.sub_variant or "non_intervenable_proxy"
    target = R["target"]
    alt = R["intervenable_alt"]
    target_var = role_plan.by_name[target]
    if target_var.preferred_low is None:
        target_var.preferred_low = True
    score = _scoring_for(target_var)
    baseline_t = _expected_observational(bn, target, score)
    accept_threshold = MIN_EFFECT * 0.5

    def _acceptable_actions(exclude: Set[str]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for v in role_plan.variables:
            if not v.intervenable or v.name in exclude:
                continue
            for val in v.values:
                try:
                    e = _expected_under_do(bn, target, {v.name: val}, score)
                except Exception:
                    continue
                improvement = -(e - baseline_t)  # positive = improvement
                if improvement >= accept_threshold:
                    actions.append({
                        "variable": v.name, "value": val,
                        "improvement": round(improvement, 4),
                    })
        return actions

    premise_gold: Dict[str, Any]
    if sub == "valid_proposed_intervention":
        proposed = R["proposed_intervention"]
        sn_prop = list(role_plan.by_name[proposed].values)
        rows = []
        for val in sn_prop:
            e = _expected_under_do(bn, target, {proposed: val}, score)
            rows.append({"value": val, "expected_target": e})
        rows.sort(key=lambda r: r["expected_target"])  # lower is better
        best_prop = rows[0]
        if best_prop["value"] != sn_prop[-1]:
            return False, (
                f"valid proposal {proposed}: strongest state is not optimal "
                f"({best_prop['value']} vs expected {sn_prop[-1]})"
            ), {}
        proposed_effect = best_prop["expected_target"] - baseline_t
        if -proposed_effect < MIN_EFFECT:
            return False, (
                f"valid proposal {proposed} too weak "
                f"({-proposed_effect:+.3f} improvement)"
            ), {}
        low_effect = _expected_under_do(bn, target, {proposed: sn_prop[0]}, score)
        if abs(best_prop["expected_target"] - low_effect) < MIN_EFFECT:
            return False, (
                f"valid proposal {proposed}: high-vs-low do-effect too small "
                f"({best_prop['expected_target'] - low_effect:+.3f})"
            ), {}
        acceptable_alternatives = _acceptable_actions({target})
        gold = {
            "is_valid_intervention": True,
            "valid_reason": "proposed_intervention_meaningfully_improves_target",
            "proposed_intervention": {
                "variable": proposed,
                "value": best_prop["value"],
            },
            "proposed_effect": round(proposed_effect, 4),
            "acceptable_alternatives": acceptable_alternatives,
            "accept_threshold": accept_threshold,
        }
        return True, "ok", gold

    if sub == "wrong_side_intervention":
        proposed = R["proposed_intervention"]
        sn_prop = list(role_plan.by_name[proposed].values)
        p_high = _expected_observational(
            bn, target, score, evidence={proposed: sn_prop[-1]},
        )
        p_low = _expected_observational(
            bn, target, score, evidence={proposed: sn_prop[0]},
        )
        predictive_gap = abs(p_high - p_low)
        if predictive_gap < MIN_EFFECT:
            return False, (
                f"wrong-side proposed variable {proposed} is not predictive "
                f"enough ({predictive_gap:.3f})"
            ), {}
        e_prop_high = _expected_under_do(bn, target, {proposed: sn_prop[-1]}, score)
        e_prop_low = _expected_under_do(bn, target, {proposed: sn_prop[0]}, score)
        proposed_do_effect = e_prop_high - e_prop_low
        if abs(proposed_do_effect) > 0.08:
            return False, (
                f"wrong-side proposed intervention {proposed} moves target "
                f"too much ({proposed_do_effect:+.3f})"
            ), {}
        premise_gold = {
            "invalid_reason": "wrong_side_intervention",
            "proposed_intervention": {
                "variable": proposed,
                "value": sn_prop[-1],
            },
            "proposed_predictive_gap": round(predictive_gap, 4),
            "proposed_do_effect": round(proposed_do_effect, 4),
        }
    else:
        nix = R["non_intervenable_x"]
        # nix is genuinely predictive: |E[Y|nix=high] - E[Y|nix=low]| should be big
        sn_nix = list(role_plan.by_name[nix].values)
        p_high = _expected_observational(bn, target, score, evidence={nix: sn_nix[-1]})
        p_low = _expected_observational(bn, target, score, evidence={nix: sn_nix[0]})
        predictive_gap = abs(p_high - p_low)
        if predictive_gap < MIN_EFFECT:
            return False, (
                f"non-intervenable variable {nix} is not predictive enough "
                f"({predictive_gap:.3f})"
            ), {}
        premise_gold = {
            "invalid_reason": "non_intervenable_proxy",
            "non_intervenable_var": nix,
            "non_intervenable_predictive_gap": round(predictive_gap, 4),
        }

    # alt is intervenable and meaningfully moves target
    sn_alt = list(role_plan.by_name[alt].values)
    e_high = _expected_under_do(bn, target, {alt: sn_alt[-1]}, score)
    e_low = _expected_under_do(bn, target, {alt: sn_alt[0]}, score)
    if abs(e_high - e_low) < MIN_EFFECT:
        return False, f"alternative intervention {alt} too weak ({abs(e_high - e_low):.3f})", {}

    # alt's optimal value
    rows = []
    for v in sn_alt:
        e = _expected_under_do(bn, target, {alt: v}, score)
        rows.append({"value": v, "expected_target": e})
    rows.sort(key=lambda r: r["expected_target"])  # lower is better
    best_alt = rows[0]

    excluded_alternative_vars = {target}
    if sub != "wrong_side_intervention":
        excluded_alternative_vars.add(R["non_intervenable_x"])
    acceptable_alternatives = _acceptable_actions(excluded_alternative_vars)

    gold = {
        "is_valid_intervention": False,
        "alternative": {"variable": alt, "value": best_alt["value"]},
        "alternative_effect": round(best_alt["expected_target"]
                                    - baseline_t, 4),
        "acceptable_alternatives": acceptable_alternatives,
        "accept_threshold": accept_threshold,
    }
    gold.update(premise_gold)
    return True, "ok", gold


def _question_invalid_premise(role_plan: RolePlan, gold: Dict[str, Any]) -> str:
    R = role_plan.roles
    sub = role_plan.sub_variant or "non_intervenable_proxy"
    target = role_plan.by_name[R["target"]]
    if sub in ("wrong_side_intervention", "valid_proposed_intervention"):
        proposed = role_plan.by_name[R["proposed_intervention"]]
        proposed_text = f"setting {proposed.name}={proposed.values[-1]}"
    else:
        nix = role_plan.by_name[R["non_intervenable_x"]]
        proposed_text = f"setting {nix.name}={nix.values[-1]}"
    return (
        f"An analyst suggests {proposed_text} as a way to "
        f"{_shift_toward(target)}. Investigate whether this is a sensible "
        f"intervention to run in this study, and if not, recommend a "
        f"different intervention you would actually deploy to "
        f"{_shift_toward_short(target)}."
    )


# ===================================================================
# Archetype dispatch
# ===================================================================

_ARCHETYPE_BUILDERS: Dict[str, Callable] = {
    "safety_constrained":   _build_safety_constrained,
    "confounding_reversal": _build_confounding_reversal,
    "mediator_structure":   _build_mediator_structure,
    "satisficing":          _build_satisficing,
    "subgroup_robust":      _build_subgroup_robust,
    "invalid_premise":      _build_invalid_premise,
}
_ARCHETYPE_VALIDATORS: Dict[str, Callable] = {
    "safety_constrained":   _validate_safety_constrained,
    "confounding_reversal": _validate_confounding_reversal,
    "mediator_structure":   _validate_mediator_structure,
    "satisficing":          _validate_satisficing,
    "subgroup_robust":      _validate_subgroup_robust,
    "invalid_premise":      _validate_invalid_premise,
}
_ARCHETYPE_QUESTION: Dict[str, Callable] = {
    "safety_constrained":   _question_safety_constrained,
    "confounding_reversal": _question_confounding_reversal,
    "mediator_structure":   _question_mediator_structure,
    "satisficing":          _question_satisficing,
    "subgroup_robust":      _question_subgroup_robust,
    "invalid_premise":      _question_invalid_premise,
}
_ARCHETYPE_QTYPE: Dict[str, str] = {
    "safety_constrained":   "advanced_safety_constrained",
    "confounding_reversal": "advanced_confounding_reversal",
    "mediator_structure":   "advanced_mediator_structure",
    "satisficing":          "advanced_satisficing",
    "subgroup_robust":      "advanced_subgroup_robust",
    "invalid_premise":      "advanced_invalid_premise",
}


# ===================================================================
# Story generation
# ===================================================================

def _prompt_story(role_plan: RolePlan, edges: List[Tuple[str, str]]) -> str:
    # Names only — no roles, no descriptions, no edges. The story should
    # not be able to leak any causal structure.
    var_names = ", ".join(v.name for v in role_plan.variables)
    return f"""{SYSTEM_JSON}

Topic: "{role_plan.topic}"
Variable names in the dataset: {var_names}

Write a MINIMAL 1-2 sentence neutral setting description. State only:
  - the institutional setting (e.g. "a regional hospital network",
    "a university research lab", "a county court system");
  - that the researcher has access to records and can request
    observational or interventional samples.

STRICT BANS:
  - NO mention of any specific variable's meaning or measurement.
  - NO causal claims ("X causes Y", "X drives Y", "X affects Y",
    "X is associated with Y", "X may bias Y").
  - NO mention of confounding, mediation, subgroup effects, robustness,
    safety tradeoffs, or any research question / hypothesis.
  - NO hint at what variable the researcher cares about.
  - NO mention of "the goal is to" or "the researcher wants to determine".
  - Under 40 words total.

Output JSON exactly:
{{ "story": "<1-2 neutral sentences, under 40 words>" }}
"""


def _generate_story(llm: _LLM, role_plan: RolePlan, edges: List[Tuple[str, str]]) -> str:
    try:
        js = _llm_json(llm, SYSTEM_JSON, _prompt_story(role_plan, edges))
        s = str(js.get("story", "")).strip()
        if 20 <= len(s) <= 400:
            return s
    except Exception:
        pass
    return (
        f"A research team in the {role_plan.topic.lower()} domain has "
        f"access to observational records and can request additional "
        f"observational or interventional samples."
    )


# ===================================================================
# Save artifacts
# ===================================================================

def _serialize_cpds(model: DiscreteBayesianNetwork) -> List[Dict[str, Any]]:
    out = []
    for cpd in model.get_cpds():
        child = cpd.variable
        parents = list(cpd.variables[1:])
        child_card = int(cpd.cardinality[0])
        n_parent_configs = cpd.values.size // child_card
        values = np.array(cpd.values).reshape(child_card, n_parent_configs).tolist()
        out.append({
            "child": child, "parents": parents,
            "values": values, "cardinality": child_card,
        })
    return out


def _save_graph_png(
    edges: List[Tuple[str, str]], outpath: str, title: str, nodes: List[str],
) -> None:
    dg = nx.DiGraph()
    dg.add_nodes_from(nodes)
    dg.add_edges_from(edges)
    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(dg, seed=0, k=0.9)
    nx.draw_networkx(dg, pos=pos, with_labels=True, arrows=True,
                     node_size=900, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ===================================================================
# World generator
# ===================================================================

@dataclass
class WorldBuildResult:
    world: Dict[str, Any]
    json_path: str
    archetype: str
    sub_variant: Optional[str]
    topic: str
    seed: int


def _attempt_generate_world(
    llm: _LLM, topic: str, archetype: str, sub_variant: Optional[str],
    seed: int, n_nodes: int, outdir: str, attempt_seed_offset: int = 0,
    subdomain: Optional[str] = None,
) -> Tuple[Optional[WorldBuildResult], str]:
    rng = random.Random(seed + attempt_seed_offset)
    role_plan = _generate_role_plan(
        llm, topic, archetype, sub_variant, n_nodes, subdomain=subdomain,
    )

    builder = _ARCHETYPE_BUILDERS[archetype]
    validator = _ARCHETYPE_VALIDATORS[archetype]
    qwriter = _ARCHETYPE_QUESTION[archetype]
    qtype = _ARCHETYPE_QTYPE[archetype]

    try:
        g, bn, build_meta = builder(role_plan, rng)
    except Exception as e:
        return None, f"build error: {type(e).__name__}: {e}"

    ok, reason, gold = validator(bn, role_plan, build_meta)
    if not ok:
        return None, f"validator: {reason}"

    edges = list(g.edges())
    story = _generate_story(llm, role_plan, edges)

    # Compose lazy-gold metadata + question text
    question_text = qwriter(role_plan, gold)

    # Save graph PNG
    safe_topic = topic.replace(" ", "_").replace("&", "and")
    sub_tag = f"_{sub_variant}" if sub_variant else ""
    base = f"{archetype}{sub_tag}_{safe_topic}_n{n_nodes}_seed{seed}"
    png_path = os.path.join(outdir, f"graph_{base}.png")
    _save_graph_png(edges, png_path, title=f"{archetype} | {topic}",
                    nodes=[v.name for v in role_plan.variables])

    # Build the world JSON
    non_intervenable = [
        {"name": v.name, "reason": "marked non-intervenable in role plan"}
        for v in role_plan.variables
        if not v.intervenable
    ]
    answer = _gold_to_canonical_answer(archetype, sub_variant, gold)
    metadata = {
        "archetype": archetype,
        "sub_variant": sub_variant,
        "roles": role_plan.roles,
        "gold": gold,
    }
    world = {
        "meta": {
            "topic": topic,
            "n_nodes": n_nodes,
            "topology": archetype + (f":{sub_variant}" if sub_variant else ""),
            "seed": seed,
            "llm_model": llm.model_name,
            "graph_image_path": png_path,
            "n_questions": 1,
            "benchmark": "advanced_v2",
            "archetype": archetype,
            "sub_variant": sub_variant,
        },
        "story": story,
        "non_intervenable_variables": non_intervenable,
        "variables": [v.to_dict() for v in role_plan.variables],
        "edges": [[u, v] for (u, v) in edges],
        "cpds": _serialize_cpds(bn),
        "questions": [{
            "id": 0,
            "question_type": qtype,
            "difficulty": "hard",
            "question": question_text,
            "answer": answer,
            "metadata": metadata,
        }],
    }
    json_path = os.path.join(outdir, f"world_{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(world, f, ensure_ascii=False, indent=2)
    world["meta"]["json_path"] = json_path
    return WorldBuildResult(
        world=world, json_path=json_path,
        archetype=archetype, sub_variant=sub_variant, topic=topic, seed=seed,
    ), "ok"


def _gold_to_canonical_answer(
    archetype: str, sub_variant: Optional[str], gold: Dict[str, Any],
) -> Any:
    """Reduce the rich `gold` metadata dict to a compact `answer` field
    that downstream evaluators can sanity-check directly. The full lazy gold
    stays in question.metadata.gold; this canonical form is for humans /
    quick eyeballing.
    """
    if archetype == "safety_constrained":
        return gold["best_safe_action"]
    if archetype == "confounding_reversal":
        return {
            "causal_truth": gold["causal_truth"],
            "is_confounded": True,
            "confounder_name": gold.get("confounder_name") or gold.get("confounder"),
        }
    if archetype == "mediator_structure":
        return gold.get("label") or gold.get("true_mediator")
    if archetype == "satisficing":
        return gold["feasible_actions"]
    if archetype == "subgroup_robust":
        return gold["robust_action"]
    if archetype == "invalid_premise":
        answer = {
            "is_valid_intervention": bool(gold.get("is_valid_intervention")),
            "invalid_reason": gold.get("invalid_reason"),
        }
        if gold.get("is_valid_intervention"):
            answer["proposed_intervention"] = gold.get("proposed_intervention")
        else:
            answer["alternative"] = gold["alternative"]
        return answer
    return None


def generate_world(
    llm: _LLM, topic: str, archetype: str, sub_variant: Optional[str],
    seed: int, outdir: str, n_nodes: int = 10, max_attempts: int = 6,
    subdomain: Optional[str] = None,
) -> WorldBuildResult:
    last_err = "no attempt"
    for attempt in range(max_attempts):
        offset = attempt * 9173 + 7
        result, msg = _attempt_generate_world(
            llm, topic, archetype, sub_variant, seed,
            n_nodes=n_nodes, outdir=outdir,
            attempt_seed_offset=offset,
            subdomain=subdomain,
        )
        if result is not None:
            sd = f" sub={subdomain!r}" if subdomain else ""
            print(f"  [ok] {archetype}{'/' + sub_variant if sub_variant else ''} "
                  f"topic={topic!r}{sd} seed={seed} attempt={attempt + 1}")
            return result
        print(f"  [retry {attempt + 1}/{max_attempts}] {archetype}"
              f"{'/' + sub_variant if sub_variant else ''} "
              f"topic={topic!r} seed={seed}: {msg}")
        last_err = msg
    raise RuntimeError(f"world generation failed after {max_attempts} attempts: {last_err}")


# ===================================================================
# Dataset orchestrator
# ===================================================================

def _rotated(values: List[str], offset: int) -> List[str]:
    if not values:
        return []
    k = offset % len(values)
    return list(values[k:] + values[:k])


def _expand_subvariant_slots(
    arch: str, count: int, sub_variants: List[str], seed_base: int,
) -> List[Tuple[str, Optional[str]]]:
    """Balanced subvariant slots, with seed rotation for one-at-a-time runs."""
    ordered = _rotated(sub_variants, seed_base)
    per = count // len(ordered)
    extra = count - per * len(ordered)
    slots: List[Tuple[str, Optional[str]]] = []
    for i, sv in enumerate(ordered):
        k = per + (1 if i < extra else 0)
        slots.extend([(arch, sv)] * k)
    return slots


def _slot_distribution(
    distribution: Dict[str, int], seed_base: int = 0,
) -> List[Tuple[str, Optional[str]]]:
    """Expand {archetype: count} into a list of (archetype, sub_variant) slots.

    Multi-subvariant archetypes expand in a balanced way.  The seed-based
    rotation matters when a driver asks for one world per process: extras go
    to a different subvariant instead of always to the first label.
    """
    slots: List[Tuple[str, Optional[str]]] = []
    for arch, count in distribution.items():
        if count <= 0:
            continue
        if arch == "mediator_structure":
            slots.extend(_expand_subvariant_slots(
                arch, count, MEDIATOR_SUB_VARIANTS, seed_base,
            ))
        elif arch == "invalid_premise":
            slots.extend(_expand_subvariant_slots(
                arch, count, INVALID_PREMISE_SUB_VARIANTS, seed_base,
            ))
        else:
            slots.extend([(arch, None)] * count)
    return slots


def generate_dataset(
    llm: _LLM, distribution: Dict[str, int], n_nodes_list: List[int],
    outdir: str, seed_base: int, max_attempts: int = 6,
    only_archetype: Optional[str] = None,
    only_sub_variant: Optional[str] = None,
) -> List[WorldBuildResult]:
    os.makedirs(outdir, exist_ok=True)
    slots = _slot_distribution(distribution, seed_base=seed_base)
    if only_archetype and only_sub_variant:
        count = distribution.get(only_archetype, len(slots))
        slots = [(only_archetype, only_sub_variant)] * count
    elif only_archetype:
        slots = [(a, sv) for (a, sv) in slots if a == only_archetype]
    elif only_sub_variant:
        slots = [(a, sv) for (a, sv) in slots if sv == only_sub_variant]
    rng = random.Random(seed_base * 7919 + 11)
    rng.shuffle(slots)
    total = len(slots)
    results: List[WorldBuildResult] = []
    skipped: List[Tuple[int, str, Optional[str], str]] = []
    t_start = time.time()

    # Track how many times each (topic) has been used so each repeat gets a
    # DIFFERENT subdomain salt (drawn deterministically from SUBDOMAINS).
    topic_uses: Dict[str, int] = {}

    for i, (archetype, sub_variant) in enumerate(slots):
        compatible = ARCHETYPE_TOPICS.get(archetype, TOPICS)
        topic = compatible[(seed_base + i) % len(compatible)]
        seed = seed_base + i * 101
        # Round-robin n_nodes across the slot list so each value gets ~equal
        # representation regardless of slot ordering.
        n_nodes = n_nodes_list[i % len(n_nodes_list)]
        # Pick a subdomain for this slot. Iterate through SUBDOMAINS[topic]
        # so consecutive uses of the same topic get different sub-contexts.
        sub_pool = SUBDOMAINS.get(topic, [])
        if sub_pool:
            uses = topic_uses.get(topic, 0)
            subdomain = sub_pool[(seed_base + uses * 13) % len(sub_pool)]
            topic_uses[topic] = uses + 1
        else:
            subdomain = None
        elapsed = time.time() - t_start
        eta = (elapsed / max(i, 1)) * (total - i) if i > 0 else 0
        sd_tag = f" sub={subdomain!r}" if subdomain else ""
        print(f"\n[{i+1}/{total}] {archetype}"
              f"{'/' + sub_variant if sub_variant else ''} "
              f"n={n_nodes} topic={topic!r}{sd_tag} seed={seed} "
              f"(elapsed={elapsed:.0f}s, eta={eta:.0f}s)")
        try:
            result = generate_world(
                llm, topic, archetype, sub_variant, seed, outdir,
                n_nodes=n_nodes, max_attempts=max_attempts,
                subdomain=subdomain,
            )
            results.append(result)
        except Exception as e:
            print(f"  [skip] {archetype} topic={topic!r}: {e}")
            skipped.append((i, archetype, sub_variant, str(e)))

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} worlds:")
        for i, arch, sv, msg in skipped:
            print(f"  slot {i}: {arch}{'/' + sv if sv else ''} — {msg[:120]}")

    print(f"\nGenerated {len(results)}/{total} worlds in {outdir} "
          f"({time.time() - t_start:.0f}s)")
    return results


# ===================================================================
# CLI
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate advanced-benchmark v2 worlds (1 question per "
                    "world, 6 archetypes, code-controlled CPDs).",
    )
    ap.add_argument("--n-nodes", type=int, nargs="+", default=[10],
                    help="Number(s) of variables per world. Pass multiple "
                         "values to mix sizes round-robin, e.g. --n-nodes 10 15.")
    ap.add_argument("--outdir", type=str, default="./out_bn_advanced_v2")
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--backend", type=str, choices=["bedrock"], default="bedrock")
    ap.add_argument("--model", type=str, default="us.anthropic.claude-opus-4-7")
    ap.add_argument("--max-attempts-per-world", type=int, default=6)
    ap.add_argument("--only-archetype", type=str, default=None,
                    choices=list(_ARCHETYPE_BUILDERS.keys()) + [None],  # type: ignore
                    help="Filter to a single archetype (smoke testing).")
    ap.add_argument("--only-sub-variant", type=str, default=None,
                    help="Force/filter a specific sub-variant, e.g. "
                         "valid_proposed_intervention.")
    ap.add_argument("--distribution", type=str, default=None,
                    help="Override default distribution as JSON, e.g. "
                         '\'{"safety_constrained":2,"confounding_reversal":1}\'')
    args = ap.parse_args()

    if args.distribution:
        distribution = json.loads(args.distribution)
    else:
        distribution = dict(DEFAULT_DISTRIBUTION)

    llm = build_llm(args.backend, args.model)
    generate_dataset(
        llm=llm, distribution=distribution,
        n_nodes_list=list(args.n_nodes),
        outdir=args.outdir, seed_base=args.seed_base,
        max_attempts=args.max_attempts_per_world,
        only_archetype=args.only_archetype,
        only_sub_variant=args.only_sub_variant,
    )


if __name__ == "__main__":
    main()
