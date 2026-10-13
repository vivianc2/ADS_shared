#!/usr/bin/env python3
"""v10 rebuild checks (2026-10-12): units, settable-only controls, marker polarity, per-world story, grading in units.

One world per archetype (seed 61000000 + i, rotating skins), each audited:
  1. every control has units; no control label starts with "set "; no control acts on a measurement-only quantity
     (a name from the skin's inert / decoy / proxy pools) -- heart rate, body temperature, ... are never settable;
  2. marker polarity: the mechanism proxy is a "higher = worse" marker unless the world is an AND-gate;
  3. the story names every control and every measurable signal (first alias) and no canonical id;
  4. free-text grading: the gold answer, converted to display units, scores benefit >= 0.99 through SimV6.grade;
  5. id-space reward: the gold answer in display units (via the catalog) scores part_a >= 0.99 in RPGEnv, and the
     internal-scale env (display_units=False, scripted policies) gives the same part_a for the internal gold;
  6. an intervention at the gold value given in units is applied internally at the gold value.
Usage: RPG_SYNERGY_SOFT=20 PYTHONPATH=../rpg_rl python test_units_v10.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rpg_rl"))
from sampler import sample_world, ARCHETYPES, _POSITIVE_PROXY            # noqa: E402
from skins import SKINS                                                    # noqa: E402
from skins_v10 import GOOD_MARKERS                                         # noqa: E402
from generate_v7 import audit                                              # noqa: E402
from sim_v6 import SimV6                                                   # noqa: E402
from env import RPGEnv                                                     # noqa: E402

SK = ["bioprocess", "datacenter", "watertreatment", "agronomy", "semiconductor", "aquaculture", "battery",
      "catalysis", "clinical", "fermentation"]


def check(i, arch):
    skin = SK[i % len(SK)]
    for k in range(6):                       # audit rejections are normal generation; take the first accepted seed
        seed = 61000000 + i + 100 * k
        w = sample_world(seed, skin=skin, archetype=arch, version="v10")
        w["ground_truth"]["_seed"] = seed
        res = audit(w)
        if res["ok"]:
            break
    else:
        return ["no accepted world in 6 seeds"], None
    scm, gt, gold = w["scm"], w["ground_truth"], res["gold"]
    bad = []
    S = SKINS[skin]
    meas_only = {e["name"] for k in ("inert_var_pool", "decoy_pool", "proxy_pool") for e in S[k]}
    for aid, act in scm.actuators.items():
        if "units" not in act:
            bad.append(f"{aid}: no units")
        if (act.get("aliases") or [""])[0].lower().startswith("set "):
            bad.append(f"{aid}: label starts with 'set '")
        if act["target"] in meas_only:
            bad.append(f"{aid}: sets a measurement-only quantity {act['target']}")
    from skins_v10 import LEVEL_UNITS, MEASURE_UNITS
    for n, v in scm.variables.items():
        if (v["kind"] in ("observable", "outcome") or v.get("measurable")):
            if "units" not in v:
                bad.append(f"measured signal {n}: no units")
                continue
            own = LEVEL_UNITS[skin].get(n) or MEASURE_UNITS[skin].get("__outcome__" if n == scm.outcome else n)
            if own is None or v["units"] != {"unit": own[0], "lo": own[1], "hi": own[2]}:
                bad.append(f"measured signal {n}: unit {v['units']} is not its own {own}")
    good = gt["true_mechanism_proxy"] in GOOD_MARKERS
    if good != ("two_cause" in gt["_features"]):
        bad.append(f"marker {gt['true_mechanism_proxy']} polarity good={good} vs two_cause={'two_cause' in gt['_features']}")
    story = w["scenario"]
    import re as _re
    norm = lambda t: _re.sub(r"[^a-z0-9]+", "", t.lower())     # the story capitalises acronyms (io -> I/O, ph -> pH)
    story_l = norm(story)
    for aid, act in scm.actuators.items():
        if norm(act["aliases"][0]) not in story_l:
            bad.append(f"story misses control '{act['aliases'][0]}'")
        if aid in story:
            bad.append(f"story shows canonical id {aid}")
    for n, v in scm.variables.items():
        if (v["kind"] == "observable" or v.get("measurable")) and n != scm.outcome and norm((v.get("aliases") or [n])[0]) not in story_l:
            bad.append(f"story misses signal '{(v.get('aliases') or [n])[0]}'")
    # free-text grading in display units
    sim = SimV6(w, precomputed={"gold": gold, "battery": res["battery"]})
    iv = gold["intervention"]
    ans = {"recommended_intervention": {k: sim.act_to_display(k, v) for k, v in iv.items() if not isinstance(v, dict)},
           "structured": {}}
    if gold.get("policy"):
        p = dict(gold["policy"])
        tr = p.get("treatment") or next(k for k, v in iv.items() if isinstance(v, dict))
        mu = scm.variables[p["marker"]].get("units")
        thr = (mu["lo"] + p["threshold"] / 100.0 * (mu["hi"] - mu["lo"])) if mu else p["threshold"]
        ans["recommended_policy"] = {"treatment": tr, "stratifier": p["marker"],
                                     "threshold": thr,
                                     "dose_if_ge": sim.act_to_display(tr, p["dose_if_ge"]),
                                     "dose_if_lt": sim.act_to_display(tr, p["dose_if_lt"])}
    g = sim.grade(ans)
    if (g["benefit_recovered"] or 0) < 0.99:
        bad.append(f"free-text gold in units: benefit {g['benefit_recovered']}")
    # intervention at the gold value given in units
    scal = {k: v for k, v in iv.items() if not isinstance(v, dict)}
    if scal:
        k0, v0 = next(iter(scal.items()))
        sim2 = SimV6(w, precomputed={"gold": gold, "battery": res["battery"]}, data_dir=tempfile.mkdtemp())
        r = sim2.intervene([{"request": k0, "value": sim2.act_to_display(k0, v0)}], [scm.outcome])
        if abs(r["applied_internal"][k0] - v0) > 0.05:
            bad.append(f"intervene in units applied {r['applied_internal'][k0]} internally, gold {v0}")
    # id-space reward, display and internal envs
    parts = {}
    for disp in (True, False):
        env = RPGEnv(world=w, gold=gold, battery=res["battery"], catalog_seed=seed, data_dir=tempfile.mkdtemp(),
                     display_units=disp)
        env.reset()
        cat = env.cat
        acts = [{"actuator": cat.a_name2id[k], "value": cat.value_to_display(cat.a_name2id[k], v)}
                for k, v in scal.items()]
        a_ = {"actions": acts, "proxy": cat.m_name2id.get(gt["true_mechanism_proxy"]), "decoys": [], "signs": {}}
        if gold.get("policy"):
            p = gold["policy"]
            tr = p.get("treatment") or next(k for k, v in iv.items() if isinstance(v, dict))
            tid = cat.a_name2id[tr]
            mid = cat.m_name2id[p["marker"]]
            mu = cat.m_units.get(mid)
            thr = (mu["lo"] + p["threshold"] / 100.0 * (mu["hi"] - mu["lo"])) if mu else p["threshold"]
            a_["policy"] = {"treatment": tid, "stratifier": mid, "threshold": thr,
                            "dose_if_ge": cat.value_to_display(tid, p["dose_if_ge"]),
                            "dose_if_lt": cat.value_to_display(tid, p["dose_if_lt"])}
        env.step(f'<action type="intervene">{json.dumps({"actions": acts[:1], "measure": []})}</action>')
        _o, _r, _d, info = env.step(f'<action type="answer">{json.dumps(a_)}</action>')
        parts[disp] = float(info.get("part_a") or 0)
    if parts[True] < 0.99 or abs(parts[True] - parts[False]) > 0.02:
        bad.append(f"id-space part_a display={parts[True]} internal={parts[False]}")
    return bad, (skin, len(scm.actuators), len(story.split()))


def main():
    n_bad = 0
    for i, arch in enumerate(ARCHETYPES):
        bad, info = check(i, arch)
        print(f"{'ok  ' if not bad else 'FAIL'} {arch:20s} {info} {bad[:4]}")
        n_bad += bool(bad)
    print("ALL PASS" if n_bad == 0 else f"{n_bad} FAIL")
    return n_bad


if __name__ == "__main__":
    raise SystemExit(main())
