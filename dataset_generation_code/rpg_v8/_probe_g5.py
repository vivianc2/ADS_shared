#!/usr/bin/env python3
"""G5 — GRPO-signal probe on Qwen3-8B for v8 archetypes.

Samples N worlds of each archetype (rpg_v8 sampler), audits them, and runs G=8
rollouts/world with the base Qwen3-8B (via the vLLM OpenAI server). Reports, per
archetype, the numbers that decide "is there an RL training signal":
  - reward mean / mean within-group std
  - NON-DEGENERATE fraction (groups with std>0 -> GRPO has a gradient there)
  - partA / partB means, accepts, and a max-completion length (truncation proxy).

Run from rpg_v8/ with:  PYTHONPATH=../rpg_rl python _probe_g5.py ...
(so sampler/generate_v7 come from rpg_v8 = 9 archetypes, env/policy/rollout from rpg_rl).
Requires the vLLM server up (serve_qwen3_8b.sh) at $VLLM_BASE_URL or localhost:8000.
"""
import argparse, json, statistics as st
from sampler import sample_world, ARCHETYPES
from generate_v7 import audit
from world_stream import WorldBundle
from policy import VLLMPolicy
from rollout import rollout_group

SKINS = ["bioprocess", "clinical", "catalysis", "agronomy", "semiconductor",
         "aquaculture", "battery", "datacenter", "watertreatment", "fermentation"]


def make_bundle(seed0, skin, arche):
    for s in range(seed0, seed0 + 400):
        try:
            w = sample_world(s, skin=skin, archetype=arche)
            res = audit(w)
        except Exception:
            continue
        if res["ok"]:
            return WorldBundle(world=w, gold=res["gold"], battery=res["battery"],
                               seed=s, skin=w["domain"], archetype=arche, split="probe")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetypes", nargs="*", default=ARCHETYPES)
    ap.add_argument("--n-worlds", type=int, default=3)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=4096)   # match RL; no truncation
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=7_100_000)
    ap.add_argument("--out", default="g5_results.json")
    args = ap.parse_args()

    pol = VLLMPolicy(model="Qwen/Qwen3-8B", max_new_tokens=args.max_new_tokens,
                     enable_thinking=False, temperature=args.temperature)
    out = {}
    for ai, arche in enumerate(args.archetypes):
        groups = []
        for i in range(args.n_worlds):
            b = make_bundle(args.seed0 + ai * 20000 + i * 331, SKINS[i % len(SKINS)], arche)
            if b is None:
                print(f"[{arche}] world {i}: no audited world", flush=True)
                continue
            trajs = rollout_group(b, pol, G=args.group, max_new_tokens=args.max_new_tokens)
            rs = [t.reward for t in trajs]
            grp = dict(skin=b.skin, seed=b.seed,
                       mean=st.mean(rs), std=(st.pstdev(rs) if len(rs) > 1 else 0.0),
                       maxr=max(rs), minr=min(rs),
                       pa=st.mean([t.part_a for t in trajs]), pb=st.mean([t.part_b for t in trajs]),
                       accepts=sum(1 for t in trajs if t.accepted),
                       maxcomp=max((len(tt.completion) for t in trajs for tt in t.turns), default=0))
            groups.append(grp)
            print(f"[{arche}] {b.skin}/{b.seed}: mean={grp['mean']:.3f} std={grp['std']:.3f} "
                  f"max={grp['maxr']:.2f} A={grp['pa']:.2f} B={grp['pb']:.2f} acc={grp['accepts']}/{args.group} "
                  f"maxcomp~{grp['maxcomp']//4}tok", flush=True)
        if groups:
            nd = sum(1 for g in groups if g["std"] > 1e-6)
            out[arche] = dict(
                n=len(groups), reward_mean=st.mean([g["mean"] for g in groups]),
                grp_std_mean=st.mean([g["std"] for g in groups]),
                nondegenerate=f"{nd}/{len(groups)}",
                any_positive=sum(1 for g in groups if g["maxr"] > 0),
                partA_mean=st.mean([g["pa"] for g in groups]),
                partB_mean=st.mean([g["pb"] for g in groups]),
                accepts_total=sum(g["accepts"] for g in groups),
                groups=groups)
            o = out[arche]
            print(f"==> {arche}: reward_mean={o['reward_mean']:.3f} grp_std={o['grp_std_mean']:.3f} "
                  f"nondegen={o['nondegenerate']} A={o['partA_mean']:.2f} B={o['partB_mean']:.2f} "
                  f"accepts={o['accepts_total']}/{len(groups)*args.group}\n", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
