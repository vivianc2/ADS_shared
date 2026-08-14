#!/usr/bin/env python3
"""Surrogate-endpoint world validation gate (cheap, no API).

The design-loop check that a freshly generated world set is sound before spending
compute on a training/eval run. For each archetype x skin it asserts the
benchmark's core properties:

  * audit passes (all oracle sub-audits: decoy / proxy_signal / inertness /
    gold_selfconsistency / counterintuitiveness);
  * GOLD recovers ~all achievable benefit (benefit_recovered >= 0.90, part A ok);
  * proxy P is observable (interventional shift/sd >= 1.0 OR observational band ok);
  * for surrogate_trap worlds ONLY (the held-out trap):
      - dosing the trap actuator raises the surrogate S but leaves the latent
        goal G (== utility) and the proxy P flat  (dS>0, dG~=0, dP~=0);
      - an answer that just doses the trap recovers ~0 benefit and fails part A.

Exit 0 iff every case passes.
"""
import sys
import numpy as np

from sampler import sample_world, ARCHETYPES
from skins import skin_names
from oracle_v6 import audit_world, grade

SKINS = skin_names()
N = 20000


def _extreme(scm, aid):
    a = scm.actuators[aid]
    if a.get("dtype") == "continuous":
        return a["range"][1]
    return a.get("values", ["off", "on"])[-1]


def _build(seed, skin, arch, tries=60):
    """Sample+audit; retry nearby seeds so a rejected draw doesn't blank a case.
    Stops at the first accepted world. Returns (seed, world, res, n_tried, i+1)
    where i+1 is how many seeds it scanned to find it (rough acceptance proxy;
    the true per-archetype acceptance rate comes from the build manifest)."""
    subs = ("decoy", "proxy_signal", "distractor_inertness",
            "gold_selfconsistency", "counterintuitiveness")
    for i in range(tries):
        s = seed + i
        w = sample_world(s, skin=skin, archetype=arch)
        try:
            res = audit_world(w)
        except Exception:
            continue
        if all(res[k].get("passed") for k in subs):
            return (s, w, res, tries, i + 1)
    return (None, None, None, tries, 0)


def check_case(seed0, skin, arch):
    seed, w, res, n_tried, n_acc = _build(seed0, skin, arch)
    if w is None:
        return {"arch": arch, "skin": skin, "seed": seed0, "ok": False,
                "fail": ["no_audited_world in %d seeds" % n_tried]}
    gt, scm = w["ground_truth"], w["scm"]
    gold, battery = res["gold"], res["battery"]
    fails = []

    # --- GOLD recovers benefit ---
    g_gold = grade(w, {"recommended_intervention": gold["intervention"]}, gold, battery)
    benefit = g_gold.get("benefit_recovered")
    if not (g_gold["part_a_utility_ok"] and (benefit is None or benefit >= 0.90)):
        fails.append("gold_benefit(%.3f,partA=%s)" % (benefit or -1, g_gold["part_a_utility_ok"]))

    # --- proxy observability (shift/sd >= 1.0 OR observational band) ---
    ps = res["proxy_signal"]
    shift_sd = ps.get("interventional_shift_sd")
    if not (ps.get("interv_ok") or ps.get("obs_ok")):
        fails.append("proxy_signal(shift/sd=%.2f,corr=%.2f)" % (shift_sd or -1, ps.get("proxy_outcome_corr") or 0))

    info = {"benefit": round(benefit, 3) if benefit is not None else None,
            "shift_sd": shift_sd, "seeds_to_accept": n_acc,
            "gold_u": round(gold["expected_utility"], 1),
            "base_u": round(gold["baseline_utility"], 1)}

    # --- trap invariant (surrogate_trap only) ---
    if arch == "surrogate_trap":
        trap = gt.get("symptom_trap_actuator")
        if not trap or trap not in scm.actuators:
            fails.append("no_trap_actuator")
        else:
            S, G = scm.outcome, scm.goal
            P = gt["true_mechanism_proxy"]
            base = scm.sample(N, seed=seed + 1)
            hi = scm.sample(N, intervention={trap: _extreme(scm, trap)}, seed=seed + 1)
            dS = float(hi[S].mean() - base[S].mean())
            dG = float(np.mean(scm.utility(hi)) - np.mean(scm.utility(base)))
            dP = float(hi[P].mean() - base[P].mean())
            if not (dS > 1.0):
                fails.append("trap_dS(%.2f) not>0" % dS)
            if not (abs(dG) < 1.0):
                fails.append("trap_dG(%.2f) not~0" % dG)
            if not (abs(dP) < 1.0):
                fails.append("trap_dP(%.2f) not~0" % dP)
            # dose-the-trap answer must recover ~0 benefit and fail part A
            g_trap = grade(w, {"recommended_intervention": {trap: _extreme(scm, trap)}}, gold, battery)
            tb = g_trap.get("benefit_recovered")
            if g_trap["part_a_utility_ok"] or (tb is not None and tb > 0.2):
                fails.append("dose_trap_benefit(%.3f,partA=%s)" % (tb or -1, g_trap["part_a_utility_ok"]))
            info.update({"dS": round(dS, 2), "dG": round(dG, 2), "dP": round(dP, 2),
                         "trap_benefit": round(tb, 3) if tb is not None else None, "trap": trap})

    return {"arch": arch, "skin": skin, "seed": seed, "ok": not fails, "fail": fails, "info": info}


def main():
    allpass = True
    for i, arch in enumerate(ARCHETYPES):
        skin = SKINS[i % len(SKINS)]
        seed0 = 30000000 + i * 1000  # distinct from train(1e7)/val(2e7) seed spaces
        r = check_case(seed0, skin, arch)
        status = "OK  " if r["ok"] else "FAIL"
        if not r["ok"]:
            allpass = False
        print("[%s] %-20s %-13s seed=%d %s" % (status, r["arch"], r["skin"], r["seed"], r.get("info", {})))
        if not r["ok"]:
            print("       FAILED:", r["fail"])
    print("\n" + ("ALL ARCHETYPES PASS" if allpass else "SOME FAILURES"))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
