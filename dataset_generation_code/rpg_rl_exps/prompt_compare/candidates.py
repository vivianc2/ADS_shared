"""System-prompt and terminal-reward candidates for the comparison experiment."""

from __future__ import annotations

import hashlib
import inspect
import itertools
from typing import Any, Callable, Dict

from bootstrap import configure_imports

configure_imports()

from env import SYSTEM_PROMPT
from reward import RewardConfig, compute_reward


P2 = """You are a scientist diagnosing a failing industrial system. You interact ONLY through the catalog of ids given each turn — every measurement, control, and answer refers to those ids (m0, m1, ... for measurable signals; a0, a1, ... for controls). You do NOT know which signal or control matters; you must find out from data by measuring and (crucially) by intervening.

Each turn, output exactly:
<reasoning>your scientific thinking</reasoning>
<action type="measure|intervene|code|answer|give_up">JSON</action>
<memory>notes to carry forward</memory>

Action payloads (JSON, ids only):
- measure:   {"ids": ["m3","m0"]}                         # read those signals
- intervene: {"actions":[{"actuator":"a2","value":66}], "measure":["m3"]}   # set controls, then read
- code:      raw Python analysis; does not cost budget. Each experiment's data is preloaded as a pandas DataFrame named experiment_<n>_df (use it directly, e.g. experiment_1_df.describe()); its file path is also available as experiment_<n>_csv. pandas as pd, numpy as np, scipy.stats as stats are imported. Each code turn runs in a FRESH namespace — the experiment_<n>_df variables are always available, but variables you define do NOT persist to the next code turn, so recompute what you need.
- answer:    {"actions":[{"actuator":"a2","value":66}],
              "policy":{"treatment":"a2","stratifier":"m1","threshold":50,"dose_if_ge":100,"dose_if_lt":0},
              "proxy":"m3", "decoys":["m0"], "signs":{"a2":"+"}}
- give_up:   {}

Observation and code alone CANNOT establish causation — you must INTERVENE to test a cause and find what improves the outcome. Submit "answer" once you know the fix AND the mechanism (which signal is the true proxy, which are decoys, and each control's effect sign +/-/0). For a world where a treatment helps only a sub-population, use "policy" to stratify on the marker signal.

Do not trust a control just because the outcome reading went up — verify it actually changed the CAUSE (the mechanism proxy moves). A control that only lifts the reading without changing the cause has sign 0, and recommending it as the fix is wrong.
"""


P3 = """You are a scientist diagnosing a failing industrial system. You interact ONLY through the catalog of ids given each turn — every measurement, control, and answer refers to those ids (m0, m1, ... for measurable signals; a0, a1, ... for controls). You do NOT know which signal or control matters; you must find out from data by measuring and (crucially) by intervening.

Each turn, output exactly:
<reasoning>your scientific thinking</reasoning>
<action type="measure|intervene|code|answer|give_up">JSON</action>
<memory>notes to carry forward</memory>

Action payloads (JSON, ids only):
- measure:   {"ids": ["m3","m0"]}                         # read those signals
- intervene: {"actions":[{"actuator":"a2","value":66}], "measure":["m3"]}   # set controls, then read
- code:      raw Python analysis; does not cost budget. Each experiment's data is preloaded as a pandas DataFrame named experiment_<n>_df (use it directly, e.g. experiment_1_df.describe()); its file path is also available as experiment_<n>_csv. pandas as pd, numpy as np, scipy.stats as stats are imported. Each code turn runs in a FRESH namespace — the experiment_<n>_df variables are always available, but variables you define do NOT persist to the next code turn, so recompute what you need.
- answer:    {"actions":[{"actuator":"a2","value":66}],
              "policy":{"treatment":"a2","stratifier":"m1","threshold":50,"dose_if_ge":100,"dose_if_lt":0},
              "proxy":"m3", "decoys":["m0"], "signs":{"a2":"+"}}
- give_up:   {}

Observation and code alone CANNOT establish causation — you must INTERVENE to test a cause and find what improves the outcome. Submit "answer" once you know the fix AND the mechanism (which signal is the true proxy, which are decoys, and each control's effect sign +/-/0). For a world where a treatment helps only a sub-population, use "policy" to stratify on the marker signal.
"""


RewardFn = Callable[..., Dict[str, Any]]


def reward_r1(struct, world, cat, gold, battery, cfg=RewardConfig(),
              n_interventions=None):
    """The current RPG reward."""
    return compute_reward(
        struct, world, cat, gold, battery, cfg=cfg,
        n_interventions=n_interventions,
    )


def reward_r2(struct, world, cat, gold, battery, cfg=RewardConfig(),
              n_interventions=None):
    """Part A minus the invalid-id penalty."""
    result = dict(compute_reward(
        struct, world, cat, gold, battery, cfg=cfg,
        n_interventions=n_interventions,
    ))
    result["reward"] = float(result["part_a"] - 0.25 * result["invalid_id_fraction"])
    return result


def reward_r3(struct, world, cat, gold, battery, cfg=RewardConfig(),
              n_interventions=None):
    """Part B minus the invalid-id penalty."""
    result = dict(compute_reward(
        struct, world, cat, gold, battery, cfg=cfg,
        n_interventions=n_interventions,
    ))
    result["reward"] = float(result["part_b"] - 0.25 * result["invalid_id_fraction"])
    return result


PROMPTS: Dict[str, str] = {"p1": SYSTEM_PROMPT, "p2": P2, "p3": P3}
REWARDS: Dict[str, RewardFn] = {
    "r1": reward_r1,
    "r2": reward_r2,
    "r3": reward_r3,
}
REWARD_DESCRIPTIONS = {
    "r1": "current: 0.5 * part_a + 0.5 * part_b - 0.25 * invalid_id_fraction",
    "r2": "part_a - 0.25 * invalid_id_fraction",
    "r3": "part_b - 0.25 * invalid_id_fraction",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def configuration_definitions() -> list[dict[str, str]]:
    """Stable prompt-major Cartesian-product configuration definitions."""
    definitions = []
    for prompt_id, reward_id in itertools.product(PROMPTS, REWARDS):
        reward_source = "\n".join((
            inspect.getsource(REWARDS[reward_id]),
            inspect.getsource(compute_reward),
            inspect.getsource(RewardConfig),
        ))
        definitions.append({
            "config_id": f"{prompt_id}_{reward_id}",
            "prompt_id": prompt_id,
            "reward_id": reward_id,
            "prompt_sha256": sha256_text(PROMPTS[prompt_id]),
            "reward_sha256": sha256_text(reward_source),
            "reward_description": REWARD_DESCRIPTIONS[reward_id],
        })
    return definitions


def evaluate_terminal(struct, world, cat, gold, battery, *, n_interventions: int):
    """Re-grade independently with the fixed strict evaluation rule."""
    cfg = RewardConfig(
        w_a=0.5,
        w_b=0.5,
        c_invalid=0.0,
        strict_part_b=True,
        require_evidence=True,
        c_no_evidence=0.0,
    )
    result = compute_reward(
        struct, world, cat, gold, battery, cfg=cfg,
        n_interventions=n_interventions,
    )
    part_a = float(result["part_a"])
    part_b = float(result["part_b"])
    return {
        "score": float(0.5 * part_a + 0.5 * part_b),
        "part_a": part_a,
        "part_b": part_b,
        "accepted": bool(result["accepted"]),
        "evaluation_error": bool(result.get("reward_error", False)),
    }
