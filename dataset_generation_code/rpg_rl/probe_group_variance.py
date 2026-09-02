#!/usr/bin/env python3
"""Pre-training GRPO-signal probe (Phase-1 design doc §6).

Runs G rollouts of the BASE model (via the live vLLM server) on N train worlds and
reports, per group: reward mean/std, part A/B, and — the key number — the FRACTION OF
NON-DEGENERATE GROUPS (std > 0). GRPO's gradient is the within-group reward spread, so:

  * many non-degenerate groups  -> there is a learning signal; proceed to train.
  * mostly all-equal groups     -> no gradient (all-0 = too hard, all-equal = saturated);
                                   fix reward/curriculum/difficulty BEFORE spending compute.

This is the cheapest, highest-leverage check and needs no training stack — just the
running server. Costs only inference.

Run (server must be up):
    PYTHONPATH=../rpg_v9 python probe_group_variance.py --n-worlds 8 --group 8
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter

from world_stream import WorldStream
from policy import VLLMPolicy
from rollout import rollout_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=8)
    ap.add_argument("--group", type=int, default=8, help="G rollouts per world")
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--seed0", type=int, default=7_000_000)
    ap.add_argument("--out", default=None, help="optional jsonl to write per-group stats")
    args = ap.parse_args()

    policy = VLLMPolicy(model=args.model, max_new_tokens=args.max_new_tokens,
                        enable_thinking=args.enable_thinking)
    stream = WorldStream(split=args.split, seed0=args.seed0)

    rows = []
    nondegenerate = 0
    all_rewards = []
    accepts = 0
    act_counter = Counter()
    print(f"probing {args.n_worlds} worlds x G={args.group}  (base={args.model}, "
          f"thinking={args.enable_thinking})\n")
    for wi in range(args.n_worlds):
        b = stream.next()
        trajs = rollout_group(b, policy, G=args.group, max_new_tokens=args.max_new_tokens)
        rewards = [t.reward for t in trajs]
        mean = st.mean(rewards)
        std = st.pstdev(rewards) if len(rewards) > 1 else 0.0
        pa = st.mean([t.part_a for t in trajs])
        pb = st.mean([t.part_b for t in trajs])
        turns = st.mean([t.n_turns for t in trajs])
        acc = sum(t.accepted for t in trajs)
        accepts += acc
        all_rewards += rewards
        for t in trajs:
            for tn in t.turns:
                act_counter[tn.action_type] += 1
        degen = std < 1e-9
        if not degen:
            nondegenerate += 1
        row = {"world_id": b.world["world_id"], "skin": b.skin, "archetype": b.archetype,
               "reward_mean": round(mean, 3), "reward_std": round(std, 3),
               "part_a": round(pa, 3), "part_b": round(pb, 3),
               "avg_turns": round(turns, 1), "n_accepted": acc,
               "reward_min": round(min(rewards), 3), "reward_max": round(max(rewards), 3),
               "degenerate": degen}
        rows.append(row)
        flag = "  <-- DEGENERATE (no gradient)" if degen else ""
        print(f"  [{wi+1}/{args.n_worlds}] {b.skin}/{b.archetype}: "
              f"r={mean:.2f}±{std:.2f} (min {min(rewards):.2f} max {max(rewards):.2f}) "
              f"A={pa:.2f} B={pb:.2f} turns~{turns:.0f} acc={acc}/{args.group}{flag}")

    frac = nondegenerate / max(1, args.n_worlds)
    print("\n=== GRPO-signal summary ===")
    print(f"non-degenerate groups (std>0, DAPO keeps): {nondegenerate}/{args.n_worlds} "
          f"({frac:.0%})")
    print(f"overall reward: mean {st.mean(all_rewards):.3f}  "
          f"std {st.pstdev(all_rewards):.3f}  "
          f"min {min(all_rewards):.3f}  max {max(all_rewards):.3f}")
    print(f"accepted rollouts: {accepts}/{args.n_worlds * args.group}")
    print(f"action-type mix across all turns: {dict(act_counter)}")
    verdict = ("SIGNAL PRESENT — proceed to train" if frac >= 0.5 else
               "WEAK SIGNAL — inspect reward/curriculum before training")
    print(f"VERDICT: {verdict}")

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote per-group stats to {args.out}")


if __name__ == "__main__":
    main()
