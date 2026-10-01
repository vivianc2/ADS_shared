#!/usr/bin/env python3
"""Tests for rpg_rl/prefill.py (scripted screening prefill, 2026-10-01).

Checks on real v9 worlds: k=0 is a no-op; k=full runs measure-all + each control alone at 100 then at 0,
capped at budget-2; the prefill never ends the episode; env counters include the prefilled turns; an answer
after the prefill grades normally; 'rand' stays within [0, full]. The SkyRL wiring (skyrl_rpg/env.py::init)
was checked in the container against driver.py::_prefill_screen (identical transcripts on a4/validation).

Run:  PYTHONPATH=../rpg_v9 RPG_SYNERGY_SOFT=20 python test_prefill.py
"""
from __future__ import annotations
import random, sys, tempfile
from sampler import sample_world
from generate_v7 import audit
from env import RPGEnv
from prefill import prefill, resolve_k, screening_plan

failures = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: failures.append(m)

for seed, arch in ((52000001, "confounded_chain"), (52000002, "dose_window"), (52000003, "instrument_only"),
                   (52000004, "synergy_pair"), (52000005, "hidden_subtype")):
    w = sample_world(seed, skin="watertreatment", archetype=arch); w["ground_truth"]["_seed"] = seed
    res = audit(w)
    def fresh():
        e = RPGEnv(world=w, gold=res["gold"], battery=res["battery"], catalog_seed=seed, data_dir=tempfile.mkdtemp())
        e.reset(); return e
    print(f"{arch} (seed {seed})")
    e = fresh(); plan = screening_plan(e); n_a = len(e.cat.actuator_ids())
    check(len(plan) == min(1 + 2 * n_a, e.budget - 2), f"plan length {len(plan)} = min(1+2*{n_a}, budget-2)")
    check(plan[0][0] == "measure" and all(t == "intervene" for t, _ in plan[1:]), "measure first, then interventions")
    check(resolve_k("", e) == 0 and resolve_k("0", e) == 0, "k off -> 0")
    msgs, done = prefill(e, 0); check(msgs == [] and e._used == 0, "k=0 no-op")
    e = fresh(); k = resolve_k("full", e); msgs, done = prefill(e, k)
    check(not done and len(msgs) == 2 * k and e._used == k and e._turn == k, f"full: {k} experiments, 2k msgs, counters = k")
    check(e._n_interv == k - 1, "interventions counted (all but the baseline measure)")
    check(e.budget - e._used >= 2, "policy keeps >= 2 experiments")
    _o, _r, d, info = e.step('<reasoning>x</reasoning>\n<action type="answer">{"actions":[{"actuator":"a0","value":100}]}</action>')
    check(d and "part_a" in info, f"answer after prefill grades (part_a {info.get('part_a'):.3f})")
    e = fresh(); check(resolve_k("99", e) == len(plan), "int k capped at full")
    ks = [resolve_k("rand", fresh(), random.Random(i)) for i in range(20)]
    check(min(ks) >= 0 and max(ks) <= len(plan), f"rand within [0,{len(plan)}]: {sorted(set(ks))}")

print("\nALL PASS" if not failures else f"\n{len(failures)} FAILURES"); sys.exit(1 if failures else 0)
