#!/usr/bin/env python3
"""Reward for RL on RPG v10, computed from the release grader's output (2026-10-23, RL session).

The episode is graded by the benchmark itself (`run_world` -> `Simulator.grade` -> release `oracle.grade`, n=30000,
seed 999, strict part B). This module only re-reads that grade:
- part A: `part_a_utility_ok` (benefit recovered >= 0.9) and `benefit_recovered` (can be negative or None);
- part B under the D1 `core` rule: proxy item, decoy item and the signs of the PRIMARY controls only (listed extra
  controls ignored in both directions), so padding the answer with correct "0" signs earns nothing. Ported from
  `personal_docs/camera_ready/analysis_rpg/d1_partb/d1_rescore.py::rescore` (rule "core"); `test_reward_v10.py`
  checks equality with that script's stored output on every episode it scored.
- accepted(core) = part A and part B(core).

Reward (RewardCfg, all weights env-overridable, RLV10_W_*, cap RLV10_SHAPING_CAP):
    r = w_acc * accepted_core + w_a * part_a_ok + w_benefit * clip(benefit, 0, 1) * min(1, cap / n_controls)
        + w_b * battery_core
Default (reward audit 2026-10-23, `reward_audit.py` / `reward_variants.py`): w_acc 1, w_benefit 0.5, cap 4. The cap
exists because "every control at mid-range, no experiment" recovers 0.48 of the benefit on validation and earned
0.242 under the uncapped shaping (base 9B: 0.144; informed script 0.309). With the cap it earns 0.039, while answers
naming <= 4 controls are unchanged (every gold answer names <= 3; scripts and the base 9B name 1-2 on average).
n_controls = number of controls in the graded recommended intervention (the policy's treatment included).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def d1_rescore(items, roles) -> Dict[str, float]:
    """Battery fraction under the D1 rules, from oracle._score_battery items [(name, ok)] and the world's v10
    roles (actuator -> role). Same arithmetic as d1_rescore.py::rescore ('stored' and 'core')."""
    proxy = float(items[0][1])
    decoys = float(items[1][1])
    signs = {n[len("sign:"):]: float(ok) for n, ok in items[2:]}
    out = {"stored": (proxy + decoys + sum(signs.values())) / (2 + len(signs))}
    core = {a: ok for a, ok in signs.items() if roles.get(a) == "primary"}
    out["core"] = (proxy + decoys + sum(core.values())) / (2 + len(core))
    return out


def reward_components(result: Optional[Dict[str, Any]], world: Dict[str, Any], pre: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the reward and the logs need, from run_world's result dict."""
    if result is None:                      # aborted episode (no grade)
        return {"graded": False, "answered": False, "part_a": 0, "benefit": None, "benefit_clip": 0.0,
                "battery_stored": 0.0, "battery_core": 0.0, "part_b_core": 0, "accepted_core": 0,
                "accepted_release": 0, "forced_no_answer": True, "n_controls": 0, "queries": 0, "interventions": 0}
    g = result["grade"]
    roles = (world.get("ground_truth") or {}).get("_v10_roles") or {}
    b = d1_rescore(g["battery_items"], roles)
    pa = int(bool(g["part_a_utility_ok"]))
    ben = g.get("benefit_recovered")
    pb_core = int(b["core"] >= 0.8 - 1e-9)
    return {"graded": True, "answered": not g.get("forced_no_answer", False), "part_a": pa, "benefit": ben,
            "benefit_clip": 0.0 if ben is None else max(0.0, min(1.0, float(ben))),
            "battery_stored": b["stored"], "battery_core": b["core"], "part_b_core": pb_core,
            "accepted_core": pa * pb_core, "accepted_release": int(bool(g["accepted"])),
            "forced_no_answer": bool(g.get("forced_no_answer", False)),
            "n_controls": len(g.get("recommended_intervention") or {}),
            "queries": result.get("queries_used", 0), "interventions": result.get("interventions_run", 0)}


def _w(name, default):
    return field(default_factory=lambda: float(os.environ.get(f"RLV10_W_{name}", default)))


@dataclass
class RewardCfg:
    w_acc: float = _w("ACC", "1.0")
    w_a: float = _w("A", "0.0")
    w_benefit: float = _w("BENEFIT", "0.5")
    w_b: float = _w("B", "0.0")
    cap: float = field(default_factory=lambda: float(os.environ.get("RLV10_SHAPING_CAP", "4")))

    def reward(self, c: Dict[str, Any]) -> float:
        n = max(1, int(c.get("n_controls", 0) or 0))
        shrink = min(1.0, self.cap / n) if self.cap > 0 else 1.0
        return float(self.w_acc * c["accepted_core"] + self.w_a * c["part_a"]
                     + self.w_benefit * c["benefit_clip"] * shrink + self.w_b * c["battery_core"])

    def as_dict(self):
        return {"w_acc": self.w_acc, "w_a": self.w_a, "w_benefit": self.w_benefit, "w_b": self.w_b, "cap": self.cap}
