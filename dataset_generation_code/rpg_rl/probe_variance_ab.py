#!/usr/bin/env python3
"""A3 — extended GRPO-signal probe: per-group std of part_a AND part_b (not just r1),
plus ANSWER-DIVERSITY (distinct chosen causes / normalized entropy) on the SAME groups.

Purpose (week-2 meeting follow-up): confirm the offline finding (A1) in a live base-9B
generation setting, and show directly that within-group REWARD variance (the GRPO signal)
and ANSWER variance (what majority-voting exploits) are the same phenomenon, not opposed.

Run against an ALREADY-SERVED base endpoint (e.g. NewtonBench leaves base on :8001):
    cd ADS_shared/dataset_generation_code/rpg_rl
    PYTHONPATH=../rpg_v9 VLLM_BASE_URL=http://localhost:8001/v1 \
      python probe_variance_ab.py --n-worlds 60 --group 8 --model base \
      --enable-thinking --split train --out /home/ec2-user/SageMaker/vivian/box1_clean_eval/probe_ab_base.jsonl
"""
from __future__ import annotations
import argparse, json, re, math
import statistics as st
from collections import Counter, defaultdict

from world_stream import WorldStream
from policy import VLLMPolicy
from rollout import rollout_group

ANS_RE = re.compile(r'<action type="answer">(.*?)</action>', re.S)

def chosen_cause(traj):
    """Extract the group-comparable 'answer key' (the chosen causal target) from the last
    answer turn, so we can count distinct answers within a group. Falls back to None."""
    for tn in reversed(traj.turns):
        if tn.action_type == "answer":
            m = list(ANS_RE.finditer(tn.completion))
            if not m:
                return None
            try:
                a = json.loads(m[-1].group(1))
            except Exception:
                return None
            if not isinstance(a, dict):
                return None
            # the causal identity the task turns on: treatment (+ stratifier/subtype if present)
            key = (a.get("treatment"), a.get("stratifier"), a.get("subtype_of"))
            return json.dumps(key, sort_keys=True)
    return None

def pstd(xs):
    return st.pstdev(xs) if len(xs) > 1 else 0.0

def norm_entropy(labels):
    """Shannon entropy of the answer distribution, normalized to [0,1] by log(G)."""
    labs = [l for l in labels if l is not None]
    if len(labs) <= 1:
        return 0.0
    c = Counter(labs); n = len(labs)
    H = -sum((v/n) * math.log(v/n) for v in c.values())
    return H / math.log(len(labs)) if len(labs) > 1 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=60)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--model", default="base")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--seed0", type=int, default=7_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    policy = VLLMPolicy(model=args.model, max_new_tokens=args.max_new_tokens,
                        enable_thinking=args.enable_thinking)
    stream = WorldStream(split=args.split, seed0=args.seed0)

    rows = []
    agg = defaultdict(list)
    print(f"probing {args.n_worlds} worlds x G={args.group} (model={args.model}, "
          f"thinking={args.enable_thinking}, split={args.split})\n")
    for wi in range(args.n_worlds):
        b = stream.next()
        trajs = rollout_group(b, policy, G=args.group, max_new_tokens=args.max_new_tokens)
        rew = [t.reward for t in trajs]; pa = [t.part_a for t in trajs]; pb = [t.part_b for t in trajs]
        answers = [chosen_cause(t) for t in trajs]
        n_distinct = len(set(a for a in answers if a is not None))
        row = dict(world_id=b.world["world_id"], skin=b.skin, archetype=b.archetype,
                   r_mean=round(st.mean(rew), 3), r_std=round(pstd(rew), 3),
                   a_mean=round(st.mean(pa), 3), a_std=round(pstd(pa), 3),
                   b_mean=round(st.mean(pb), 3), b_std=round(pstd(pb), 3),
                   r_max=round(max(rew), 3), headroom=round(max(rew) - st.mean(rew), 3),
                   n_distinct_answers=n_distinct, answer_entropy=round(norm_entropy(answers), 3),
                   n=len(trajs))
        rows.append(row)
        for k in ("r_std", "a_std", "b_std", "headroom", "answer_entropy", "n_distinct_answers", "r_mean"):
            agg[k].append(row[k])
        print(f"  [{wi+1}/{args.n_worlds}] {b.skin}/{b.archetype}: r={row['r_mean']:.2f}±{row['r_std']:.2f} "
              f"A_std={row['a_std']:.2f} B_std={row['b_std']:.2f} headroom=+{row['headroom']:.2f} "
              f"distinct_ans={n_distinct}/{args.group} H={row['answer_entropy']:.2f}")

    print("\n=== SUMMARY (mean over worlds) ===")
    for k in ("r_mean", "r_std", "a_std", "b_std", "headroom", "n_distinct_answers", "answer_entropy"):
        print(f"  {k:20s} {st.mean(agg[k]):.3f}")
    dead = sum(1 for x in agg["r_std"] if x < 0.05)
    # correlation between reward-variance (a_std) and answer-variance (entropy)
    xs, ys = agg["a_std"], agg["answer_entropy"]
    if len(xs) > 2 and pstd(xs) > 0 and pstd(ys) > 0:
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / len(xs)
        corr = cov / (pstd(xs) * pstd(ys))
        print(f"  corr(part_a within-group std, answer entropy) = {corr:.2f}  "
              f"(positive => reward-variance and answer-variance are the SAME phenomenon)")
    print(f"  GRPO-dead groups (r_std<0.05): {dead}/{len(rows)} ({100*dead/len(rows):.0f}%)")

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote per-group stats to {args.out}")

if __name__ == "__main__":
    main()
