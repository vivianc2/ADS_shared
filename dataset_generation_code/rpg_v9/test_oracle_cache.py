#!/usr/bin/env python3
"""The stored-oracle path (generate_v7.oracle_json / oracle_from_json, used by v10 dataset rows and the env / driver
caches) must give exactly the environment a fresh audit gives. Run from dataset_generation_code with
PYTHONPATH=rpg_v9:rpg_rl RPG_SYNERGY_SOFT=20:   python rpg_v9/test_oracle_cache.py"""
import json, tempfile

from sampler import sample_world
from generate_v7 import audit, oracle_json, oracle_from_json, _json_default
from env import RPGEnv

CASES = [(60000000, "clinical", "confounded_chain", "v10"), (60000005, "battery", "competing_causes", "v10"),
         (60000002, "fermentation", "hidden_subtype", "v10"), (52000000, "agronomy", "confounded_chain", "v9"),
         (52000032, "agronomy", "instrument_only", "v9.1")]


def env_from(seed, skin, arch, ver, stored=None):
    w = sample_world(seed, skin=skin, archetype=arch, version=ver); w["ground_truth"]["_seed"] = seed
    res = oracle_from_json(w, stored) if stored else audit(w)
    env = RPGEnv(world=w, gold=res["gold"], battery=res["battery"], catalog_seed=seed, data_dir=tempfile.mkdtemp())
    return w, res, env


def answer(env, cfg):
    payload = {"actions": [{"actuator": env.cat.a_name2id[k], "value": v} for k, v in cfg.items()
                           if not isinstance(v, dict)], "proxy": None, "decoys": [], "signs": {}}
    env.step('<action type="intervene">' + json.dumps({"actions": [{"actuator": env.cat.actuator_ids()[0], "value": 50}],
                                                        "measure": []}) + '</action>')
    _, r, _, info = env.step('<action type="answer">' + json.dumps(payload) + '</action>')
    return round(r, 6), round(float(info.get("part_a") or 0), 6)


def main():
    fails = 0
    for seed, skin, arch, ver in CASES:
        w1, r1, e1 = env_from(seed, skin, arch, ver)
        stored = oracle_json(w1, r1)
        w2, r2, e2 = env_from(seed, skin, arch, ver, stored)
        same_obs = e1.reset() == e2.reset()
        same_gold = json.dumps(r1["gold"], sort_keys=True, default=_json_default) == json.dumps(r2["gold"], sort_keys=True, default=_json_default)
        same_bat = json.dumps(r1["battery"], sort_keys=True, default=_json_default) == json.dumps(r2["battery"], sort_keys=True, default=_json_default)
        same_scm = json.dumps(w1["scm"].to_dict(), sort_keys=True, default=_json_default) == json.dumps(w2["scm"].to_dict(), sort_keys=True, default=_json_default)
        gold_cfg = r1["gold"]["intervention"]
        bad_cfg = {next(iter(w1["scm"].actuators)): 100.0}
        rew = []
        for cfg in (gold_cfg, bad_cfg):
            e1.reset(); e2.reset()
            rew.append((answer(e1, cfg), answer(e2, cfg)))
        same_rew = all(a == b for a, b in rew)
        ok = same_obs and same_gold and same_bat and same_scm and same_rew
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {ver:5s} {arch:18s} obs {same_obs} gold {same_gold} battery {same_bat} "
              f"calibrated-scm {same_scm} rewards {rew}")
    # a stored oracle for another world must be refused
    w1, r1, _ = env_from(*CASES[0])
    wx = sample_world(CASES[1][0], skin=CASES[1][1], archetype=CASES[1][2], version="v10")
    try:
        oracle_from_json(wx, oracle_json(w1, r1)); fails += 1; print("FAIL wrong-world oracle accepted")
    except RuntimeError:
        print("ok   wrong-world oracle refused")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
