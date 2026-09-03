#!/usr/bin/env python3
"""Opaque-id catalog for the RL environment.

WHY THIS EXISTS (reward-integrity + anti-leakage, in one move):

The eval harness lets the agent answer in free text and RESOLVES it to canonical
variable/actuator names. That resolver is (a) a second model call / lexical guess in
the loop and (b) a source of reward noise — a correct-but-verbose answer can fail to
resolve, and a lucky phrasing can mis-resolve. For an RL REWARD that is unacceptable:
the reward must be a pure, deterministic function of what the policy chose, never of
how it phrased things (a hard reward-contract decision).

So for RL we present each world through a CATALOG of OPAQUE IDS:

    measurables: m0, m1, ...   actuators: a0, a1, ...

Each id carries a NEUTRAL one-line description (never the canonical variable name,
which could leak the causal role — e.g. "DissolvedMetal"). The agent selects ids;
the reward maps ids -> canonical names and calls the SAME grade() the eval uses. No
resolver, no LLM, in the reward path.

This is deterministic per world (ids are assigned in a fixed, seeded-shuffled order so
position carries no information), reversible (id<->name both directions), and it
tightens the benchmark's core property: the agent must learn WHICH signal matters
from data, not from the name.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Catalog:
    """Bidirectional id<->canonical-name maps for one world, plus neutral descriptions.

    measurable_ids / actuator_ids preserve a stable (seeded) order so the prompt can
    list them deterministically. id2name / name2id resolve both directions with no
    free-text matching.
    """
    world_id: str
    outcome: str
    outcome_direction: str
    m_id2name: Dict[str, str]
    a_id2name: Dict[str, str]
    m_desc: Dict[str, str]           # measurable id -> neutral description
    a_desc: Dict[str, str]           # actuator id   -> neutral description
    a_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # id -> {op,dtype,range,values}

    # ---- reverse maps (built in __post_init__) ----
    m_name2id: Dict[str, str] = field(default_factory=dict)
    a_name2id: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.m_name2id = {v: k for k, v in self.m_id2name.items()}
        self.a_name2id = {v: k for k, v in self.a_id2name.items()}

    # ---- id -> canonical name (reward path) ----
    def measurable_name(self, mid: str) -> Optional[str]:
        return self.m_id2name.get(mid)

    def actuator_name(self, aid: str) -> Optional[str]:
        return self.a_id2name.get(aid)

    # ---- name -> id (for building gold/eval answers in id-space) ----
    def measurable_id(self, name: str) -> Optional[str]:
        return self.m_name2id.get(name)

    def actuator_id(self, name: str) -> Optional[str]:
        return self.a_name2id.get(name)

    def measurable_ids(self) -> List[str]:
        return list(self.m_id2name)

    def actuator_ids(self) -> List[str]:
        return list(self.a_id2name)


def _neutral_measurable_desc(spec: Dict[str, Any]) -> str:
    """A NON-LEAKING one-line description of a measurable. We deliberately use the
    first alias (already a meaningful-but-neutral phrase in the skins) rather than the
    canonical variable name, and never mention causal role. The alias is what a domain
    operator would call the instrument reading; it does not reveal whether it matters."""
    aliases = spec.get("aliases") or []
    return aliases[0] if aliases else "an instrument reading"


def _neutral_actuator_desc(act: Dict[str, Any]) -> str:
    """Mirror the measurable path: use the first ALIAS — a neutral operator name for the
    control (e.g. 'set water temperature', 'corrosion inhibitor') — so the scientist knows
    WHAT each knob is and can form domain hypotheses, WITHOUT being told its causal ROLE.
    We deliberately do NOT use `description`, which encodes the answer: the true fix reads
    "dosing to reduce <true_root>" and the symptom trap reads "a control that adjusts the
    <outcome> readout". The alias drops both tells (fix -> 'corrosion inhibitor'; distractor
    'control for HardnessCaCO3' -> 'set hardness'), matching how measurables use a neutral
    alias rather than the canonical variable name.

    RESIDUAL LEAK — must be fixed at WORLD-GEN, not here: some skins give the symptom-trap
    actuator an alias that still reveals its role ('dye-masking additive', 'color-masking
    agent'), whereas others already give it a plausible legitimate alias (bioprocess: 'assay
    recalibration'). The catalog cannot invent a neutral name; the skins must give the trap a
    non-revealing alias so it is indistinguishable by label from a real control."""
    aliases = act.get("aliases") or []
    return aliases[0] if aliases else "an adjustable control"


def build_catalog(world: Dict[str, Any], scm, *, seed: int) -> Catalog:
    """Assign opaque ids to a world's measurables and actuators. Order is shuffled by
    a per-world seed so id POSITION carries no information (m0 is not 'the first/most
    important' signal). Deterministic in (world_id, seed)."""
    rng = random.Random(seed)

    measurables = [nm for nm, s in scm.variables.items()
                   if (s["kind"] in ("observable", "outcome") or s.get("measurable"))]
    rng.shuffle(measurables)
    m_id2name = {f"m{i}": nm for i, nm in enumerate(measurables)}
    m_desc = {f"m{i}": _neutral_measurable_desc(scm.variables[nm])
              for i, nm in enumerate(measurables)}

    actuators = list(scm.actuators)
    rng.shuffle(actuators)
    a_id2name = {f"a{i}": aid for i, aid in enumerate(actuators)}
    a_desc = {f"a{i}": _neutral_actuator_desc(scm.actuators[aid])
              for i, aid in enumerate(actuators)}
    a_meta = {}
    for i, aid in enumerate(actuators):
        act = scm.actuators[aid]
        a_meta[f"a{i}"] = {"op": act.get("op"), "dtype": act.get("dtype", "continuous"),
                           "range": act.get("range"), "values": act.get("values")}

    return Catalog(
        world_id=world["world_id"], outcome=scm.outcome,
        outcome_direction="higher_is_better" if scm.higher_is_better else "lower_is_better",
        m_id2name=m_id2name, a_id2name=a_id2name,
        m_desc=m_desc, a_desc=a_desc, a_meta=a_meta,
    )
