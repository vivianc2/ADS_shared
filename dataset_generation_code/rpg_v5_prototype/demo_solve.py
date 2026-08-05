#!/usr/bin/env python3
"""Vertical-slice demo for RPG v5.

Runs the whole idea end to end on the bioreactor world (and prints the water
world's audit for cross-domain sanity):

1. builds the world (SCM ground truth + partial projection);
2. audits faithfulness + solvability;
3. plays a scripted "expert" trajectory through the query interface, printing
   each meaningful experiment and its result;
4. grades a *correct* answer and a *surface-proxy* answer, showing the computed
   grader accepts the former and rejects the latter.

Run:  python3 demo_solve.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from worlds import ALL_WORLDS
from scm import SCM
from oracle import (audit_world, calibrate_world, counterfactual_battery,
                    expected_utility, grade_answer, optimal_intervention)


# ---------------------------------------------------------------------------
# A tiny query interface = what the agent would call through the simulator.
# ---------------------------------------------------------------------------

class QueryInterface:
    def __init__(self, world, seed: int = 42):
        self.w = world
        self.scm: SCM = world.scm
        self.seed = seed
        self.cells = 0
        self.log: List[Dict[str, Any]] = []

    def observational(self, measurements: List[str], n: int = 300) -> Dict[str, float]:
        vals = self.scm.sample(n, seed=self.seed + len(self.log) + 1)
        obs = self.scm.observe(vals, measurements, seed=self.seed + 1000 + len(self.log))
        self.cells += n * (len(measurements) + 2)
        summ = {m: (round(float(np.mean(v)), 2), round(float(np.std(v)), 2)) for m, v in obs.items()}
        self.log.append({"mode": "observational", "measurements": measurements, "n": n, "summary": summ})
        return summ

    def sweep(self, knob: str, grid: List[Any], measurements: List[str], n: int = 300) -> List[Dict[str, Any]]:
        rows = []
        for lvl in grid:
            iv = {knob: lvl}
            vals = self.scm.sample(n, intervention=iv, seed=self.seed + hash((knob, str(lvl))) % 9999)
            obs = self.scm.observe(vals, measurements, seed=self.seed + 2000 + len(self.log), intervention=iv)
            self.cells += n * (len(measurements) + 3)
            rows.append({knob: lvl, **{m: round(float(np.mean(obs[m])), 2) for m in measurements}})
        self.log.append({"mode": "sweep", "knob": knob, "grid": grid, "measurements": measurements})
        return rows

    def clamp(self, node: str, levels: List[float], measurements: List[str], n: int = 300) -> List[Dict[str, Any]]:
        rows = []
        for lvl in levels:
            vals = self.scm.sample(n, clamp={node: lvl}, seed=self.seed + int(lvl) + len(self.log))
            obs = self.scm.observe(vals, measurements, seed=self.seed + 3000 + len(self.log))
            self.cells += n * (len(measurements) + 3)
            rows.append({f"clamp_{node}": lvl, **{m: round(float(np.mean(obs[m])), 2) for m in measurements}})
        self.log.append({"mode": "clamp", "node": node, "levels": levels, "measurements": measurements})
        return rows


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def expert_trajectory(world) -> None:
    q = QueryInterface(world)
    out = world.scm.outcome
    proxy = world.true_mechanism_proxy
    decoy = world.confounded_decoys[0]

    hr(f"EXPERT TRAJECTORY — {world.world_id}")
    print("\n[1] Observational baseline (form the naive hypothesis)")
    base = q.observational([out, decoy, proxy] + [m for m in world.observables if m not in (out, decoy, proxy)][:1])
    print("   ", base)
    print(f"    Naive read: {decoy} tracks {out} -> tempting 'oxygen/pressure' story.")
    print(f"    But {proxy} is elevated too, which does not fit that story.")

    print(f"\n[2] Break the confound: clamp {decoy} high vs low, watch {out}")
    rows = q.clamp(decoy, [20.0, 80.0], [out])
    print("   ", rows)
    d = rows[1][out] - rows[0][out]
    print(f"    d{out} when forcing {decoy}: {d:+.2f}  -> ~0 => {decoy} is NOT the cause (confounded).")

    tk = world.targeted_knob
    print(f"\n[3] Discriminating sweep of the targeted knob {tk} on {out} and {proxy}")
    spec = world.knobs[tk]
    grid = list(np.linspace(spec["range"][0], spec["range"][1], 6))
    rows = q.sweep(tk, [round(g, 1) for g in grid], [out, proxy])
    for r in rows:
        print("   ", r)
    print(f"    {out} recovers while {proxy} drops => targeted knob addresses the true mechanism.")

    print(f"\n[4] Interior-optimum check: best dose of {tk} is not the max")
    best_row = max(rows, key=lambda r: r[out]) if world.scm.higher_is_better else min(rows, key=lambda r: r[out])
    print(f"    best observed level in sweep: {best_row}")

    trap = world.symptom_trap_knob
    print(f"\n[5] Reject the trap: sweep {trap}, watch {out} vs {proxy}")
    tspec = world.knobs[trap]
    tgrid = list(np.linspace(tspec["range"][0], tspec["range"][1], 4))
    rows = q.sweep(trap, [round(g, 1) for g in tgrid], [out, proxy])
    for r in rows:
        print("   ", r)
    print(f"    {out} may shift but {proxy} does NOT improve => symptom masking, not a fix.")

    print(f"\n    cells spent by expert trajectory: {q.cells}")


def main() -> None:
    for wid, factory in ALL_WORLDS.items():
        world = factory()
        hr(f"AUDIT — {wid}")
        audit = audit_world(world)
        print(json.dumps({k: v for k, v in audit.items() if k != "battery"}, indent=2, default=str))
        print("\nBATTERY (ground truth for grading understanding):")
        print(json.dumps(audit["battery"], indent=2, default=str))

    # Detailed expert run + grading on the bioreactor world (calibrated).
    world = ALL_WORLDS["bioreactor_yield_collapse"]()
    calibrate_world(world)
    expert_trajectory(world)

    gold = optimal_intervention(world)
    battery = counterfactual_battery(world)

    hr("GRADING — correct answer vs surface-proxy answer")
    correct = {
        "recommended_intervention": {world.targeted_knob: gold["value"]},
        "structured": {
            "true_mechanism_proxy": world.true_mechanism_proxy,
            "confounded_decoys": world.confounded_decoys,
            "knob_sign_predictions": battery["knob_sign_predictions"],
        },
        "explanation": world.latent_plain_name,
    }
    surface = {
        # falls for the oxygen confound: recommends fiddling temperature, calls DO the proxy
        "recommended_intervention": {"TemperatureSetpoint": 39},
        "structured": {
            "true_mechanism_proxy": world.confounded_decoys[0],
            "confounded_decoys": [],
            "knob_sign_predictions": {k: "+" for k in world.knobs},
        },
        "explanation": "dissolved-oxygen control drift",
    }
    print("\nCORRECT answer grade:")
    print(json.dumps(grade_answer(world, battery, gold, correct), indent=2, default=str))
    print("\nSURFACE-PROXY answer grade:")
    print(json.dumps(grade_answer(world, battery, gold, surface), indent=2, default=str))


if __name__ == "__main__":
    main()
