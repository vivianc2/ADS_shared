#!/usr/bin/env python3
"""Cheap design-loop check for the surrogate_trap archetype (no Bedrock).

Samples surrogate_trap worlds across skins, runs the full audit, and asserts the
archetype's intended properties:
  - generation succeeds; gold uses the real fix, NOT the surrogate handle;
  - the surrogate handle's true sign on the outcome is "0";
  - the surrogate is a confounded decoy and is NOT credited as a valid proxy (no leak);
  - decoy / proxy_signal / distractor_inertness / gold_selfconsistency / counterintuitiveness
    audits pass; the surrogate naive move captures ~none of the gold benefit;
  - interventionally: forcing the handle moves the surrogate a lot but the outcome ~0.
"""
import sys
import numpy as np
from sampler import sample_world
from skins import skin_names
from oracle_v6 import audit_world, expected_utility

SKINS = skin_names()


def check(seed, skin):
    w = sample_world(seed, skin=skin, archetype="surrogate_trap")
    gt = w["ground_truth"]
    scm = w["scm"]
    surr = gt.get("surrogate_node")
    surr_aid = f"set_{surr}" if surr else None
    res = audit_world(w)  # mutates scm via calibrate
    gold, batt = res["gold"], res["battery"]
    signs = batt["actuator_sign_predictions"]
    # interventional sanity: handle moves surrogate, not outcome
    base = scm.sample(8000, seed=seed + 1)
    hi = scm.sample(8000, intervention={surr_aid: 100.0}, seed=seed + 1)
    dsurr = float(hi[surr].mean() - base[surr].mean())
    dout = float(np.mean(scm.utility(hi)) - np.mean(scm.utility(base)))
    checks = {
        "gen_ok": surr is not None and surr_aid in scm.actuators,
        "gold_not_surrogate": surr_aid not in gold["intervention"],
        "surr_sign_is_0": signs.get(surr_aid) == "0",
        "surr_is_decoy": surr in batt["confounded_decoys"],
        "surr_not_valid_proxy": surr not in batt["valid_mechanism_proxies"],
        "surr_not_lenient_proxy": surr not in batt["lenient_mechanism_proxies"],
        "decoy_audit": res["decoy"]["passed"],
        "proxy_signal": res["proxy_signal"]["passed"],
        "distractor_inert": res["distractor_inertness"]["passed"],
        "gold_selfconsist": res["gold_selfconsistency"]["passed"],
        "counterintuitive": res["counterintuitiveness"]["passed"],
        "handle_moves_surr": abs(dsurr) > 5.0,
        "handle_not_outcome": abs(dout) < 1.0,
    }
    # surrogate-specific naive move fraction
    surr_naive = [r for r in res["counterintuitiveness"]["naive_results"]
                  if surr_aid in r["naive"]]
    frac = surr_naive[0]["fraction_of_gold_benefit"] if surr_naive else None
    return w, res, checks, dict(dsurr=round(dsurr, 2), dout=round(dout, 2),
                                surr_naive_frac=frac,
                                gold=gold["intervention"], gold_u=round(gold["expected_utility"], 2),
                                base_u=round(gold["baseline_utility"], 2),
                                targeted=gt["targeted_actuator"], surr=surr, surr_aid=surr_aid,
                                decoy_corr=res["calibration"].get("decoy_corr"))


def main():
    seeds_skins = [(700001 + i, SKINS[i % len(SKINS)]) for i in range(len(SKINS))]
    allpass = True
    for seed, skin in seeds_skins:
        try:
            w, res, checks, info = check(seed, skin)
        except Exception as e:
            print(f"[{skin}/{seed}] EXCEPTION {type(e).__name__}: {e}")
            allpass = False
            continue
        failed = [k for k, v in checks.items() if not v]
        status = "OK  " if not failed else "FAIL"
        if failed:
            allpass = False
        print(f"[{status}] {skin:12s} seed={seed} surr={info['surr']} aid={info['surr_aid']} "
              f"dSurr={info['dsurr']} dOut={info['dout']} naive_frac={info['surr_naive_frac']} "
              f"decoy_corr={info['decoy_corr']} gold={info['gold']}")
        if failed:
            print("        FAILED:", failed)
    print("\n", "ALL SKINS PASS" if allpass else "SOME FAILURES", sep="")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
