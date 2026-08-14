#!/usr/bin/env python3
"""Summarize the Qwen3.6-27B behavioral-validation run on the v9 held-out set.

Answers the three questions the run exists for:
  1. INTEGRITY  — did any turn truncate? (finish_reason must be all 'stop'; else the
     ceiling is understated by output-cap truncation, per the max-length rule.)
  2. CEILING    — per-archetype Part A (utility recovered) and Part B (mechanism), so we
     can see the strong-model ceiling and the "acts right / explains wrong" gap.
  3. TRAP-RESISTANCE — on the held-out surrogate_trap worlds specifically, does the model
     recover real benefit (found the fix) rather than dosing the trap?

Usage: python analyze_v9_27b.py <results_dir>
"""
import glob
import json
import sys
from collections import defaultdict, Counter


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _load_arch_map(worlds_dir):
    """world_id -> archetype/skin, from the dumped worlds' manifest (results don't embed it)."""
    try:
        m = json.load(open(f"{worlds_dir}/manifest.json"))
        return {w["world_id"]: (w["archetype"], w["skin"]) for w in m["worlds"]}
    except Exception:
        return {}


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "/work/results_v9_validation_27b"
    worlds_dir = sys.argv[2] if len(sys.argv) > 2 else "/work/data/rpg_v9_val_worlds"
    arch_map = _load_arch_map(worlds_dir)
    files = [f for f in glob.glob(f"{rdir}/*.json") if "summary" not in f]
    by_arch = defaultdict(list)
    fr = Counter()
    hit_cap = 0
    n = 0
    for f in files:
        d = json.load(open(f))
        g = d.get("grade") or {}
        arche = arch_map.get(d.get("world_id"), ("?", "?"))[0]
        for t in d.get("turns", []):
            if "finish_reason" in t:
                fr[t["finish_reason"]] += 1
        if d.get("hit_turn_cap"):
            hit_cap += 1
        by_arch[arche].append(g)
        n += 1

    print(f"=== Qwen3.6-27B on v9 held-out: {n} worlds ===\n")
    print(f"INTEGRITY: finish_reason tally = {dict(fr)}  (want: only 'stop')")
    length_cut = fr.get("length", 0)
    print(f"           turns truncated at 'length' = {length_cut}   worlds hit turn cap = {hit_cap}\n")

    print(f"{'archetype':22s} {'n':>3} {'accept%':>8} {'partA%':>7} {'benefit':>8} {'partB':>7}")
    all_g = []
    for arche in sorted(by_arch):
        gs = by_arch[arche]
        all_g += gs
        acc = 100 * _mean([1.0 if x.get("accepted") else 0.0 for x in gs])
        pa = 100 * _mean([1.0 if x.get("part_a_utility_ok") else 0.0 for x in gs])
        ben = _mean([x.get("benefit_recovered") for x in gs])
        pb = _mean([x.get("battery_fraction") for x in gs])
        print(f"{arche:22s} {len(gs):>3} {acc:>7.0f}% {pa:>6.0f}% {ben:>8.3f} {pb:>7.3f}")
    acc = 100 * _mean([1.0 if x.get("accepted") else 0.0 for x in all_g])
    pa = 100 * _mean([1.0 if x.get("part_a_utility_ok") else 0.0 for x in all_g])
    print(f"{'OVERALL':22s} {len(all_g):>3} {acc:>7.0f}% {pa:>6.0f}% "
          f"{_mean([x.get('benefit_recovered') for x in all_g]):>8.3f} "
          f"{_mean([x.get('battery_fraction') for x in all_g]):>7.3f}")

    trap = by_arch.get("surrogate_trap", [])
    if trap:
        pa = 100 * _mean([1.0 if x.get("part_a_utility_ok") else 0.0 for x in trap])
        ben = _mean([x.get("benefit_recovered") for x in trap])
        fell = sum(1 for x in trap if (x.get("benefit_recovered") or 0) < 0.2)
        print(f"\nTRAP-RESISTANCE (held-out surrogate_trap, n={len(trap)}): "
              f"partA={pa:.0f}%  mean_benefit={ben:.3f}  worlds_near_zero_benefit(<0.2)={fell}")
        print("  (high benefit + partA = found the real fix; near-zero = dosed the trap / gave up)")


if __name__ == "__main__":
    sys.exit(main())
