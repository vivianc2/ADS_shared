#!/usr/bin/env python3
"""Reward audit (priority 0.6, 2026-10-23): what do cheap answers and the scripted baselines earn under the RL reward?

Synthetic answers on the 270 validation worlds, graded by the release grader exactly as run_world grades an answer
(`Simulator.grade`, internal units via `oracle.grade`, n=30000, seed 999, strict), then reward_v10:
  gold            gold intervention + the battery's proxy, decoys and signs           (should be the maximum)
  gold_iv_only    gold intervention, no mechanism answer                              (part A without part B)
  empty           no intervention, no mechanism                                       (should be 0)
  all0_signs      no intervention; right proxy + decoys; every control signed "0"     (the old padding hack)
  all_ctrl_0      every control at the bottom of its range
  all_ctrl_100    every control at the top of its range
  all_ctrl_50     every control at mid-range
  primaries_mid   the gold levers at mid-range (right controls, no dose search)
  passive_*       one observational batch only, top-1/3 level settings by |r| or by full partial correlation, pushed
                  to the range end or a quarter-range from their usual level (2026-10-24; see passive_answers)
Scripts (test split, from personal_docs/camera_ready/analysis_rpg/baselines/*.csv): reward from their part A /
benefit; their part B is undefined or heuristic, so accepted(core) is taken as 0 except where `accepted` is stored
(an upper bound: the release rule, not core).

  RPG_PROTO=rpg_v9 RPG_SYNERGY_SOFT=20 ~/miniconda3/envs/ADS/bin/python reward_audit.py --workers 32
"""
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("RLV10_HARDEN_SANDBOX", "0")
import stepenv  # noqa: E402,F401  (frozen bench on sys.path)
from agent_loop import load_world_file  # noqa: E402
from oracle import grade  # noqa: E402
from reward_v10 import reward_components, RewardCfg  # noqa: E402

VAL = os.path.expanduser("~/ADS_folder/ADS_shared/rpg_benchmark/data/validation")
BASE = os.path.expanduser("~/ADS_folder/personal_docs/camera_ready/analysis_rpg/baselines")


def answers(world, pre):
    scm = world["scm"]
    bat = pre["battery"]
    gold_iv = dict(pre["gold"]["intervention"])
    roles = world["ground_truth"].get("_v10_roles") or {}
    mech = {"true_mechanism_proxy": bat["true_mechanism_proxy"], "confounded_decoys": list(bat["confounded_decoys"]),
            "actuator_sign_predictions": dict(bat["actuator_sign_predictions"])}
    acts = list(scm.actuators)

    def rng(a, frac):
        lo, hi = scm.actuators[a].get("range") or [0, 100]
        return lo + frac * (hi - lo)
    prim = [a for a, r in roles.items() if r == "primary"]
    out = {
        "gold": {"recommended_intervention": gold_iv, "structured": mech},
        "gold_iv_only": {"recommended_intervention": gold_iv, "structured": {}},
        "empty": {"recommended_intervention": {}, "structured": {}},
        "all0_signs": {"recommended_intervention": {}, "structured": {
            "true_mechanism_proxy": bat["true_mechanism_proxy"], "confounded_decoys": list(bat["confounded_decoys"]),
            "actuator_sign_predictions": {a: "0" for a in acts}}},
        "all_ctrl_0": {"recommended_intervention": {a: rng(a, 0.0) for a in acts}, "structured": {}},
        "all_ctrl_100": {"recommended_intervention": {a: rng(a, 1.0) for a in acts}, "structured": {}},
        "all_ctrl_50": {"recommended_intervention": {a: rng(a, 0.5) for a in acts}, "structured": {}},
        "primaries_mid": {"recommended_intervention": {a: rng(a, 0.5) for a in prim}, "structured": {}},
    }
    import random as _r
    rr = _r.Random(int(world["world_id"].rsplit("_", 1)[1]))
    for k in (3, 6, 10):
        pick = rr.sample(acts, min(k, len(acts)))
        out[f"rand{k}_mid"] = {"recommended_intervention": {a: rng(a, 0.5) for a in pick}, "structured": {}}
    out["primaries_plus_all_mid"] = {"recommended_intervention": {**{a: rng(a, 0.5) for a in acts}, **gold_iv},
                                     "structured": mech}
    out.update(passive_answers(world, rng))
    return out


def _resid(y, X):
    X1 = np.column_stack([X, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def passive_answers(world, rng, n=400, seed=4242):
    """Passive-shortcut answers (2026-10-24): no experiment, only one observational batch of n units (selection
    applied, as the first measurement). Every level setting (a 'set' control whose level is measured) is ranked by
      r    |corr(level, utility)|                                  (the |r| cue)
      pA   |partial corr(level, utility | all other level settings)| (full linear adjustment)
    and the top k are pushed to the end of their range in the direction of the correlation (ends), or moved a quarter
    of the range from their usual level in that direction (q25). The reward must not pay for these."""
    scm = world["scm"]
    lev = [(a, act["target"]) for a, act in scm.actuators.items()
           if act.get("op") == "set" and scm.variables.get(act.get("target"), {}).get("kind") == "observable"
           and scm.variables.get(act.get("target"), {}).get("measurable")]
    if len(lev) < 2:
        return {}
    v = scm.sample(n, seed=seed, select=True)
    u = np.asarray(scm.utility(v), float)
    X = np.column_stack([np.asarray(v[t], float) for _, t in lev])
    sd = X.std(0)
    keep = sd > 1e-9
    Xz = np.where(keep, (X - X.mean(0)) / np.where(keep, sd, 1.0), 0.0)
    uz = (u - u.mean()) / (u.std() + 1e-12)
    r = np.array([float(np.mean(Xz[:, i] * uz)) if keep[i] else 0.0 for i in range(len(lev))])
    pa = np.zeros(len(lev))
    for i in range(len(lev)):
        if not keep[i]:
            continue
        others = [j for j in range(len(lev)) if j != i and keep[j]]
        rx, ru = _resid(Xz[:, i], Xz[:, others]), _resid(uz, Xz[:, others])
        pa[i] = float(np.corrcoef(rx, ru)[0, 1]) if rx.std() > 1e-9 else 0.0
    out = {}
    for name, score in (("r", r), ("pA", pa)):
        order = np.argsort(-np.abs(score))
        for k in (1, 3):
            top = [int(i) for i in order[:k]]
            ends = {lev[i][0]: rng(lev[i][0], 1.0 if score[i] > 0 else 0.0) for i in top}
            q25 = {}
            for i in top:
                lo, hi = scm.actuators[lev[i][0]].get("range") or [0, 100]
                q25[lev[i][0]] = float(np.clip(X[:, i].mean() + np.sign(score[i]) * 0.25 * (hi - lo), lo, hi))
            out[f"passive_{name}_top{k}_ends"] = {"recommended_intervention": ends, "structured": {}}
            out[f"passive_{name}_top{k}_q25"] = {"recommended_intervention": q25, "structured": {}}
    return out


def job(path):
    world, pre = load_world_file(path)
    rows = []
    for name, ans in answers(world, pre).items():
        g = grade(world, ans, pre["gold"], pre["battery"])
        c = reward_components({"grade": g, "queries_used": 0, "interventions_run": 0}, world, pre)
        rows.append({"world_id": world["world_id"], "archetype": world["ground_truth"]["_archetype"], "answer": name,
                     "n_rec": len(g["recommended_intervention"]),
                     "n_gold": len(pre["gold"]["intervention"]),
                     **{k: c[k] for k in ("part_a", "benefit", "benefit_clip", "battery_core", "accepted_core", "n_controls",
                                          "accepted_release")}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.expanduser("~/rpgrl/rl_v10/reward_audit"))
    ap.add_argument("--worlds", default=VAL, help="folder of release-format worlds (default: data/validation)")
    ap.add_argument("--tag", default="validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = RewardCfg()
    files = sorted(glob.glob(os.path.join(args.worlds, "world_*.json.gz")))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(job, files):
            rows += r
    for r in rows:
        r["reward"] = cfg.reward(r)
    with open(os.path.join(args.out, f"synthetic_{args.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"reward = {cfg.as_dict()}", "", f"Synthetic answers, {len(files)} worlds ({args.tag}: {args.worlds}):",
             f"{'answer':24s} {'reward':>7s} {'max':>6s} {'partA':>6s} {'acc_core':>8s} {'acc_rel':>7s} {'benefit':>8s}"]
    for name in dict.fromkeys(r["answer"] for r in rows):
        R = [r for r in rows if r["answer"] == name]
        lines.append(f"{name:24s} {np.mean([r['reward'] for r in R]):7.3f} {max(r['reward'] for r in R):6.3f} "
                     f"{np.mean([r['part_a'] for r in R]):6.3f} {np.mean([r['accepted_core'] for r in R]):8.3f} "
                     f"{np.mean([r['accepted_release'] for r in R]):7.3f} "
                     f"{np.mean([r['benefit'] for r in R if r['benefit'] is not None]):8.3f}")
    # scripts
    lines += ["", "Scripted baselines (test 540, reward from part A / benefit; accepted(core) = 0, upper bound uses stored "
              "release `accepted`):", f"{'policy':28s} {'reward':>7s} {'(ub)':>6s} {'partA':>6s} {'benefit':>8s}"]
    srcs = [("baselines_episodes.csv", lambda r: f"{r['policy']}@{r['budget']}"),
            ("cd1_episodes_nocd.csv", lambda r: r["policy"]), ("cd1_episodes.csv", lambda r: r["policy"])]
    for fn, key in srcs:
        by = {}
        for r in csv.DictReader(open(os.path.join(BASE, fn))):
            by.setdefault(key(r), []).append(r)
        for k, R in by.items():
            if fn == "baselines_episodes.csv" and not any(k.startswith(p) for p in ("random@15", "blind_single@15",
                                                                                     "blind_group@15", "informed@15",
                                                                                     "oracle_levers@15", "blind_grid@120")):
                continue
            ben = [max(0.0, min(1.0, float(r["benefit_recovered"]))) if r["benefit_recovered"] not in ("", "None") else 0.0 for r in R]
            acc = [float(r["accepted"]) if r.get("accepted") not in (None, "") else 0.0 for r in R]
            pa = [float(r["part_a"]) for r in R]
            rew = [cfg.w_benefit * b + cfg.w_a * p for b, p in zip(ben, pa)]
            ub = [x + cfg.w_acc * a for x, a in zip(rew, acc)]
            lines.append(f"{k:28s} {np.mean(rew):7.3f} {np.mean(ub):6.3f} {np.mean(pa):6.3f} {np.mean(ben):8.3f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, f"reward_audit_summary_{args.tag}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
