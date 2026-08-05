#!/usr/bin/env python3
"""
compute_cis.py — World-clustered bootstrap confidence intervals for paper results.

For PGM-Struct: 6 questions per world share the same graph/CPDs, so we resample
worlds (not questions) to respect within-world correlation.
For PGM-Decision: 1 question per world, so world-clustered = standard bootstrap.
For evidence-ledger metrics (CCA, ERR, EDV, PSW): bootstrap over trajectories.

Usage (from framework_code/):
    python compute_cis.py                      # full table, 95% CI, B=2000
    python compute_cis.py --B 5000             # more bootstrap samples
    python compute_cis.py --latex              # also print LaTeX macro strings
    python compute_cis.py --alpha 0.1          # 90% CI instead
    python compute_cis.py --ledger PATH        # path to per_trajectory_metrics.json
                                                # default: ../analysis/evidence_ledger/per_trajectory_metrics.json
    python compute_cis.py --latex --B 5000     # combine flags
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Config: eval files in evaluations/for_paper/
# ---------------------------------------------------------------------------

EVAL_DIR = "evaluations/for_paper"

STRUCTURAL_RUNS = {
    ("Claude Opus",  "Guess-shot"):   "eval_zero_shot_4_19_big_v2.json",
    ("Claude Opus",  "Reasoning"):    "eval_opus_agent_4_19_big_v2.json",
    ("Claude Opus",  "Coder"):        "eval_opus_coder_4_19_big_v2.json",
    ("Claude Opus",  "Modular"):      "eval_opus_coder_new_4_19_big.json",
    ("GPT-4o",       "Guess-shot"):   "eval_zero_shot_4_19_big_gpt4o.json",
    ("GPT-4o",       "Reasoning"):    "eval_gpt4o_agent_4_19_big.json",
    ("GPT-4o",       "Coder"):        "eval_gpt4o_coder_4_19_big.json",
    ("GPT-4o",       "Modular"):      "eval_gpt4o_coder_new_4_19_big.json",
    ("Llama-3-70B",  "Guess-shot"):   "eval_zero_shot_4_19_big_llama.json",
    ("Llama-3-70B",  "Reasoning"):    "eval_llama_agent_4_19_big.json",
    ("Llama-3-70B",  "Coder"):        "eval_llama_coder_4_19_big.json",
    ("Llama-3-70B",  "Modular"):      "eval_llama_coder_new_4_19_big.json",
}

ADVANCED_RUNS = {
    ("Claude Opus",  "Guess-shot"):   "eval_opus_zero_shot_adv_v3.json",
    ("Claude Opus",  "Reasoning"):    "eval_opus_agent_adv_v3.json",
    ("Claude Opus",  "Coder"):        "eval_opus_coder_adv_v3.json",
    ("Claude Opus",  "Modular"):      "eval_opus_coder_new_adv_v3.json",
    ("GPT-4o",       "Guess-shot"):   "eval_gpt4o_zero_shot_adv_v3.json",
    ("GPT-4o",       "Reasoning"):    "eval_gpt4o_agent_adv_v3.json",
    ("GPT-4o",       "Coder"):        "eval_gpt4o_coder_adv_v3.json",
    ("Llama-3-70B",  "Guess-shot"):   "eval_llama_zero_shot_adv_v3.json",
    ("Llama-3-70B",  "Reasoning"):    "eval_llama_agent_adv_v3.json",
    ("Llama-3-70B",  "Coder"):        "eval_llama_coder_adv_v3.json",
}

DEFAULT_LEDGER = "../analysis/evidence_ledger/per_trajectory_metrics.json"

# ---------------------------------------------------------------------------
# Bootstrap core
# ---------------------------------------------------------------------------

def _bootstrap_world_accuracy(items, B, rng):
    """
    World-clustered bootstrap. items is a list of dicts with keys
    'world_name' and 'correct' (bool).
    Returns array of B bootstrap accuracy values.
    """
    worlds = defaultdict(list)
    for item in items:
        worlds[item["world_name"]].append(item["correct"])
    world_keys = np.array(list(worlds.keys()))
    W = len(world_keys)

    boot = np.empty(B)
    for i in range(B):
        sampled = rng.choice(W, size=W, replace=True)
        all_correct = [c for idx in sampled for c in worlds[world_keys[idx]]]
        boot[i] = np.mean(all_correct)
    return boot


def ci_from_boot(boot, alpha):
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# Load eval JSON → list of {world_name, correct}
# ---------------------------------------------------------------------------

def load_items(path):
    with open(path) as f:
        d = json.load(f)
    out = []
    for item in d.get("evaluated", []):
        ev = item.get("eval", {})
        correct = bool(ev.get("correct", False))
        out.append({"world_name": item["world_name"], "correct": correct})
    return out


# ---------------------------------------------------------------------------
# Ledger metric bootstrap
# ---------------------------------------------------------------------------

def _compute_ledger_metrics(trajs):
    """
    trajs: list of per-trajectory dicts from per_trajectory_metrics.json.
    Returns dict of point-estimate metrics.
    """
    n = len(trajs)
    if n == 0:
        return {}

    cca_mask = np.array([t["had_strong_correct_evidence"] for t in trajs], dtype=bool)
    passed   = np.array([t["passed"] for t in trajs], dtype=bool)
    failed   = ~passed

    cca   = cca_mask.mean()
    # ERR: among trajectories that acquired strong correct evidence, fraction correct
    err   = passed[cca_mask].mean() if cca_mask.any() else float("nan")
    # EDV: among failed trajectories, fraction that had walked away from correct evidence
    edv   = np.array([t["walked_away_from_correct_evidence"] for t in trajs], dtype=bool)
    edv_rate = edv[failed].mean() if failed.any() else float("nan")
    # PSW: median turns after first strong correct turn (only for those with strong correct)
    psw_vals = [t["psw"] for t in trajs if t["psw"] is not None]
    med_psw  = float(np.median(psw_vals)) if psw_vals else float("nan")

    return {
        "n": n,
        "accuracy": passed.mean(),
        "CCA": cca,
        "ERR": err,
        "EDV": edv_rate,
        "Med_PSW": med_psw,
    }


def bootstrap_ledger(trajs, B, rng, alpha):
    """Bootstrap over individual trajectories for ledger metrics."""
    n = len(trajs)
    trajs = list(trajs)
    boot = {"accuracy": [], "CCA": [], "ERR": [], "EDV": []}
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        sample = [trajs[i] for i in idx]
        m = _compute_ledger_metrics(sample)
        for k in boot:
            boot[k].append(m[k])

    cis = {}
    for k, vals in boot.items():
        valid = [v for v in vals if not np.isnan(v)]
        if len(valid) < B * 0.5:
            cis[k] = (float("nan"), float("nan"))
        else:
            cis[k] = (np.percentile(valid, 100 * alpha / 2),
                      np.percentile(valid, 100 * (1 - alpha / 2)))
    return cis


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(v):
    return f"{100*v:.1f}%"

def fmt_ci(lo, hi):
    return f"[{100*lo:.1f}, {100*hi:.1f}]"

def latex_ci(v, lo, hi, pct=True):
    """Return string like 91.4 [89.0, 93.8] for LaTeX."""
    if pct:
        return f"{100*v:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

def model_short(m):
    return {"Claude Opus": "Opus", "GPT-4o": "GPT-4o", "Llama-3-70B": "Llama"}.get(m, m)

def method_short(m):
    return {"Guess-shot": "Guess", "Reasoning": "Agent", "Coder": "Coder", "Modular": "Modular"}.get(m, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(B, alpha, latex, ledger_path):
    rng = np.random.default_rng(42)
    ci_pct = int((1 - alpha) * 100)

    # ---- Structural --------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"PGM-Struct (structural, 60 worlds × 6 Q) — {ci_pct}% CI")
    print(f"{'='*72}")
    header = f"{'Model':<14} {'Method':<10} {'n':>5}  {'Acc':>6}  {ci_pct}% CI"
    print(header)
    print("-" * len(header))

    struct_results = {}
    for (model, method), fname in STRUCTURAL_RUNS.items():
        path = os.path.join(EVAL_DIR, fname)
        if not os.path.exists(path):
            print(f"{model_short(model):<14} {method_short(method):<10}  --  (file not found)")
            continue
        items = load_items(path)
        n = len(items)
        acc = np.mean([it["correct"] for it in items])
        boot = _bootstrap_world_accuracy(items, B, rng)
        lo, hi = ci_from_boot(boot, alpha)
        struct_results[(model, method)] = (n, acc, lo, hi)
        print(f"{model_short(model):<14} {method_short(method):<10} {n:>5}  {fmt_pct(acc):>6}  {fmt_ci(lo, hi)}")

    # ---- Advanced ----------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"PGM-Decision (advanced, 48 worlds × 1 Q) — {ci_pct}% CI")
    print(f"{'='*72}")
    print(f"Note: n=48 means CIs are wide (~±14pp). Report to signal uncertainty,")
    print(f"not to claim significance of small differences.")
    header2 = f"{'Model':<14} {'Method':<10} {'n':>5}  {'Acc':>6}  {ci_pct}% CI"
    print(header2)
    print("-" * len(header2))

    adv_results = {}
    for (model, method), fname in ADVANCED_RUNS.items():
        path = os.path.join(EVAL_DIR, fname)
        if not os.path.exists(path):
            print(f"{model_short(model):<14} {method_short(method):<10}  --  (file not found)")
            continue
        items = load_items(path)
        n = len(items)
        acc = np.mean([it["correct"] for it in items])
        boot = _bootstrap_world_accuracy(items, B, rng)
        lo, hi = ci_from_boot(boot, alpha)
        adv_results[(model, method)] = (n, acc, lo, hi)
        print(f"{model_short(model):<14} {method_short(method):<10} {n:>5}  {fmt_pct(acc):>6}  {fmt_ci(lo, hi)}")

    # ---- Key pairwise comparisons (paired bootstrap) -----------------------
    print(f"\n{'='*72}")
    print(f"Pairwise comparisons (paired world bootstrap) — {ci_pct}% CI on difference")
    print(f"{'='*72}")
    print(f"{'Comparison':<46} {'Diff':>7}  CI on diff       Sig?")
    print("-" * 80)

    struct_files = {k: os.path.join(EVAL_DIR, v) for k, v in STRUCTURAL_RUNS.items()
                    if os.path.exists(os.path.join(EVAL_DIR, v))}

    def paired_boot_diff(items_a, items_b, B, rng):
        """Resample worlds present in BOTH runs; compute acc_a - acc_b each iter."""
        worlds_a = defaultdict(list)
        for it in items_a:
            worlds_a[it["world_name"]].append(it["correct"])
        worlds_b = defaultdict(list)
        for it in items_b:
            worlds_b[it["world_name"]].append(it["correct"])
        shared = sorted(set(worlds_a) & set(worlds_b))
        if not shared:
            return None
        W = len(shared)
        shared = np.array(shared)
        diffs = np.empty(B)
        for i in range(B):
            idx = rng.choice(W, size=W, replace=True)
            sa = [c for k in shared[idx] for c in worlds_a[k]]
            sb = [c for k in shared[idx] for c in worlds_b[k]]
            diffs[i] = np.mean(sa) - np.mean(sb)
        return diffs

    comparisons = [
        # Label, (model, method_A), (model, method_B)
        ("Opus: Reasoning vs Guess-shot [struct]",
         ("Claude Opus","Reasoning"), ("Claude Opus","Guess-shot")),
        ("Opus: Coder vs Reasoning [struct]",
         ("Claude Opus","Coder"), ("Claude Opus","Reasoning")),
        ("GPT-4o: Reasoning vs Guess-shot [struct]",
         ("GPT-4o","Reasoning"), ("GPT-4o","Guess-shot")),
        ("GPT-4o: Modular vs Reasoning [struct]",
         ("GPT-4o","Modular"), ("GPT-4o","Reasoning")),
        ("Llama: Reasoning vs Guess-shot [struct]",
         ("Llama-3-70B","Reasoning"), ("Llama-3-70B","Guess-shot")),
        ("Llama: Modular vs Reasoning [struct]",
         ("Llama-3-70B","Modular"), ("Llama-3-70B","Reasoning")),
    ]

    for label, key_a, key_b in comparisons:
        path_a = struct_files.get(key_a)
        path_b = struct_files.get(key_b)
        if not path_a or not path_b:
            print(f"  {label:<44}  SKIP (missing file)")
            continue
        items_a = load_items(path_a)
        items_b = load_items(path_b)
        diffs = paired_boot_diff(items_a, items_b, B, rng)
        if diffs is None:
            print(f"  {label:<44}  SKIP (no shared worlds)")
            continue
        acc_a = np.mean([it["correct"] for it in items_a])
        acc_b = np.mean([it["correct"] for it in items_b])
        diff = acc_a - acc_b
        lo_d, hi_d = ci_from_boot(diffs, alpha)
        sig = "*" if lo_d > 0 else ("†" if hi_d < 0 else "")
        print(f"  {label:<44}  {fmt_pct(diff):>7}  {fmt_ci(lo_d, hi_d):<18} {sig}")

    print(f"  (* CI entirely above 0 = A significantly better than B at {ci_pct}% level)")

    # ---- Ledger metrics ----------------------------------------------------
    if not os.path.exists(ledger_path):
        print(f"\nLedger file not found: {ledger_path}")
        print("Re-run with --ledger PATH when evidence_ledger analysis is complete.")
    else:
        print(f"\n{'='*72}")
        print(f"Evidence-ledger metrics (bootstrap over trajectories) — {ci_pct}% CI")
        print(f"Note: trajectories are from the SUBSET used in ledger analysis,")
        print(f"not the full eval runs. Acc here may differ from main table.")
        print(f"{'='*72}")

        with open(ledger_path) as f:
            traj_all = json.load(f)

        # Use only the structural subset (the paper's Table 2 source)
        subset_keys = {
            "Claude Opus": ("Opus",  "out_bn_4_19_big_subset"),
            "GPT-4o":      ("GPT-4o","out_bn_4_19_big_subset"),
            "Llama-3-70B": ("Llama", "out_bn_4_19_big_subset"),
        }

        header3 = (f"{'Model':<14} {'n':>5}  {'Acc':>6}  {ci_pct}% CI"
                   f"    {'CCA':>6}  {ci_pct}% CI"
                   f"    {'ERR':>6}  {ci_pct}% CI"
                   f"    {'EDV':>6}  {ci_pct}% CI")
        print(header3)
        print("-" * len(header3))

        for model_name, (ml, dl) in subset_keys.items():
            trajs = [t for t in traj_all
                     if t["model_label"] == ml and t["dataset_label"] == dl]
            if not trajs:
                print(f"{model_short(model_name):<14}  (no trajectories in ledger file)")
                continue
            pt = _compute_ledger_metrics(trajs)
            cis = bootstrap_ledger(trajs, B, rng, alpha)

            def _fc(key):
                lo, hi = cis[key]
                v = pt[key]
                if np.isnan(v):
                    return "   N/A"
                return f"{fmt_pct(v):>6}  {fmt_ci(lo, hi)}"

            print(f"{model_short(model_name):<14} {pt['n']:>5}  {_fc('accuracy')}"
                  f"    {_fc('CCA')}"
                  f"    {_fc('ERR')}"
                  f"    {_fc('EDV')}")

        # Also advanced Opus
        adv_trajs = [t for t in traj_all
                     if t["model_label"] == "Opus" and t["dataset_label"] == "out_bn_adv_v3"]
        if adv_trajs:
            print(f"\nPGM-Decision (Opus modular, advanced subset):")
            pt = _compute_ledger_metrics(adv_trajs)
            cis = bootstrap_ledger(adv_trajs, B, rng, alpha)
            def _fc2(key):
                lo, hi = cis[key]
                v = pt[key]
                if np.isnan(v):
                    return "   N/A"
                return f"{fmt_pct(v):>6}  {fmt_ci(lo, hi)}"
            print(f"  n={pt['n']}, Acc={_fc2('accuracy')}, CCA={_fc2('CCA')}, "
                  f"ERR={_fc2('ERR')}, EDV={_fc2('EDV')}")

    # ---- LaTeX output ------------------------------------------------------
    if latex:
        print(f"\n{'='*72}")
        print(r"LaTeX macro strings (paste into paper or supplementary)")
        print(f"{'='*72}")
        print(r"% Structural accuracy with 95% CIs")
        for (model, method), (n, acc, lo, hi) in struct_results.items():
            tag = model_short(model).replace("-","").replace(".","") + method_short(method)
            print(f"  {tag}: {fmt_pct(acc)} {fmt_ci(lo, hi)}")

        print(r"% Advanced accuracy with 95% CIs")
        for (model, method), (n, acc, lo, hi) in adv_results.items():
            tag = model_short(model).replace("-","").replace(".","") + method_short(method) + "Adv"
            print(f"  {tag}: {fmt_pct(acc)} {fmt_ci(lo, hi)}")

        print()
        print(r"% For paper text: key claims with CIs")
        claim_keys = [
            ("Claude Opus", "Coder",    "Opus coder [struct]"),
            ("Claude Opus", "Reasoning","Opus reasoning [struct]"),
            ("GPT-4o",      "Modular",  "GPT-4o modular [struct]"),
            ("Llama-3-70B", "Modular",  "Llama modular [struct]"),
        ]
        for model, method, label in claim_keys:
            if (model, method) in struct_results:
                n, acc, lo, hi = struct_results[(model, method)]
                print(f"  {label}: {fmt_pct(acc)} (95\\% CI {fmt_ci(lo, hi)}, n={n})")

        adv_claim_keys = [
            ("Claude Opus", "Coder",     "Opus coder [advanced]"),
            ("GPT-4o",      "Coder",     "GPT-4o coder [advanced]"),
            ("Llama-3-70B", "Reasoning", "Llama reasoning [advanced]"),
        ]
        for model, method, label in adv_claim_keys:
            if (model, method) in adv_results:
                n, acc, lo, hi = adv_results[(model, method)]
                print(f"  {label}: {fmt_pct(acc)} (95\\% CI {fmt_ci(lo, hi)}, n={n})")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--B",      type=int,   default=2000,
                        help="Number of bootstrap samples (default: 2000)")
    parser.add_argument("--alpha",  type=float, default=0.05,
                        help="Significance level for CI (default: 0.05 → 95%% CI)")
    parser.add_argument("--latex",  action="store_true",
                        help="Print LaTeX-ready strings for paper")
    parser.add_argument("--ledger", type=str,   default=DEFAULT_LEDGER,
                        help=f"Path to per_trajectory_metrics.json "
                             f"(default: {DEFAULT_LEDGER})")
    args = parser.parse_args()

    run(B=args.B, alpha=args.alpha, latex=args.latex, ledger_path=args.ledger)
