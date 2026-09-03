#!/usr/bin/env python3
"""v0 multi-turn GRPO + LoRA trainer for RPG (standalone HF-generate prototype; superseded by the SkyRL trainer in ../skyrl_rpg).

Correct-first debug loop, on a single training GPU (leave GPU 0 for the vLLM server):

  world_stream (train)  ->  B worlds x G on-policy rollouts (turn-synchronous, batched
  HF generate)  ->  terminal reward per episode (env, pure id-space)  ->  Dr.GRPO group
  advantage + DAPO dynamic sampling  ->  masked-completion policy-gradient LoRA update.

Design choices (see the design doc):
- Multi-turn -> per-turn (prompt, completion) examples, all sharing their episode's
  advantage; loss on COMPLETION tokens only.
- Advantage A_i = r_i - mean_group(r)  (Dr.GRPO; no std scaling by default).
- Dynamic sampling: groups with all-equal reward (std==0) contribute no gradient -> dropped.
- On-policy, single update/batch => plain policy gradient (no importance ratio / clip).
- beta=0 (no KL / no reference model) to start.
- Qwen3 thinking OFF by default (V15: thinking ~2x trace length -> too slow for HF-gen debug).

Run a tiny smoke:
    PYTHONPATH=../rpg_v9 CUDA_VISIBLE_DEVICES=1 python train_grpo.py \
        --worlds-per-step 2 --group 4 --steps 1 --gen-batch 8 --max-new-tokens 256
Run the debug loop:
    PYTHONPATH=../rpg_v9 CUDA_VISIBLE_DEVICES=1 python train_grpo.py \
        --worlds-per-step 8 --group 8 --steps 60
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from world_stream import WorldStream
from env import SYSTEM_PROMPT


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-8B"
    worlds_per_step: int = 8
    group: int = 8                    # G rollouts per world
    steps: int = 60
    max_new_tokens: int = 768
    gen_batch: int = 12               # max concurrent sequences per generate (KV memory cap)
    loss_micro_batch: int = 2
    lr: float = 1e-6
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    scale_rewards: bool = False       # False=Dr.GRPO (mean-subtract only); True=std-normalize
    enable_thinking: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    seed0: int = 8_000_000
    archetypes: str = ""              # comma-sep archetype restriction (e.g. "confounded_chain")
    max_turns: int = 32
    budget: int = 15
    grad_clip: float = 1.0
    save_dir: str = "rl_ckpt/qwen3_8b_grpo_v0"
    log_file: str = "rl_ckpt/qwen3_8b_grpo_v0/train_log.jsonl"


# --------------------------------------------------------------------------- setup
def setup_model(cfg: Cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda")
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def build_prompt_ids(tok, obs: str, cfg: Cfg) -> List[int]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
    res = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  enable_thinking=cfg.enable_thinking)
    if hasattr(res, "input_ids"):            # BatchEncoding (transformers 5.x default)
        res = res.input_ids
    if res and isinstance(res[0], list):     # batched -> take first row
        res = res[0]
    return list(res)


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def batched_generate(model, tok, prompts: List[List[int]], cfg: Cfg) -> List[List[int]]:
    """Generate completions for a list of prompt-id lists. Chunks by cfg.gen_batch to cap
    KV memory. Returns completion token-id lists (prompt stripped, trailing pad/eos removed)."""
    model.eval()
    out: List[List[int]] = []
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    for s in range(0, len(prompts), cfg.gen_batch):
        chunk = prompts[s:s + cfg.gen_batch]
        maxlen = max(len(p) for p in chunk)
        input_ids, attn = [], []
        for p in chunk:                                   # LEFT-pad
            npad = maxlen - len(p)
            input_ids.append([pad] * npad + p)
            attn.append([0] * npad + [1] * len(p))
        input_ids = torch.tensor(input_ids, device="cuda")
        attn = torch.tensor(attn, device="cuda")
        gen = model.generate(input_ids=input_ids, attention_mask=attn,
                             do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                             top_k=cfg.top_k, max_new_tokens=cfg.max_new_tokens,
                             pad_token_id=pad)
        comp = gen[:, maxlen:]                             # everything after the padded prompt
        for row in comp.tolist():
            ids = []
            for t in row:
                if t == eos:
                    break
                if t == pad:
                    continue
                ids.append(t)
            out.append(ids)
    return out


def rollout_batch(model, tok, bundles, cfg: Cfg):
    """Turn-synchronous on-policy rollouts: B*G episodes advanced together, one batched
    generate per turn over all still-active episodes. Returns per-episode (turns, reward,
    info) where turns = list of (prompt_ids, completion_ids)."""
    envs = []
    for b in bundles:
        for _ in range(cfg.group):
            envs.append(b.make_env(max_turns=cfg.max_turns, budget=cfg.budget))
    N = len(envs)
    obs = [e.reset() for e in envs]
    done = [False] * N
    turns: List[List[Tuple[List[int], List[int]]]] = [[] for _ in range(N)]
    rewards = [0.0] * N
    infos: List[Dict[str, Any]] = [{} for _ in range(N)]
    guard = 0
    while not all(done) and guard < cfg.max_turns + 2:
        guard += 1
        active = [i for i in range(N) if not done[i]]
        prompts = [build_prompt_ids(tok, obs[i], cfg) for i in active]
        comps = batched_generate(model, tok, prompts, cfg)
        for k, i in enumerate(active):
            comp_ids = comps[k]
            comp_text = tok.decode(comp_ids, skip_special_tokens=True)
            nobs, r, d, info = envs[i].step(comp_text)
            turns[i].append((prompts[k], comp_ids))
            obs[i] = nobs
            done[i] = d
            if d:
                rewards[i] = float(r)
                infos[i] = info
    return turns, rewards, infos


# --------------------------------------------------------------------------- advantages
def group_advantages(rewards: List[float], G: int, cfg: Cfg):
    """Dr.GRPO advantage per rollout + a keep-mask (DAPO dynamic sampling). rewards are laid
    out world-major: [w0g0..w0g(G-1), w1g0, ...]."""
    adv = [0.0] * len(rewards)
    keep = [False] * len(rewards)
    n_groups = len(rewards) // G
    dropped = 0
    for gi in range(n_groups):
        chunk = rewards[gi * G:(gi + 1) * G]
        mean = sum(chunk) / len(chunk)
        var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
        std = var ** 0.5
        if std < 1e-9:                                   # zero-variance group -> no gradient
            dropped += 1
            continue
        denom = std if cfg.scale_rewards else 1.0
        for j in range(G):
            adv[gi * G + j] = (chunk[j] - mean) / denom
            keep[gi * G + j] = True
    return adv, keep, dropped


# --------------------------------------------------------------------------- loss
def pg_update(model, tok, examples: List[Tuple[List[int], List[int], float]], cfg: Cfg, opt):
    """examples = list of (prompt_ids, completion_ids, advantage). Masked-completion policy
    gradient, micro-batched with grad accumulation, one optimizer step. Returns metrics."""
    model.train()
    opt.zero_grad(set_to_none=True)
    total_tokens = sum(len(c) for _, c, _ in examples) or 1
    n = len(examples)
    pad = tok.pad_token_id
    sum_entropy = 0.0
    sum_logp = 0.0
    ent_tokens = 0
    for s in range(0, n, cfg.loss_micro_batch):
        mb = examples[s:s + cfg.loss_micro_batch]
        seqs, comp_mask, advs = [], [], []
        for p, c, a in mb:
            seqs.append(p + c)
            comp_mask.append([0] * len(p) + [1] * len(c))
            advs.append(a)
        maxlen = max(len(x) for x in seqs)
        input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long, device="cuda")
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long, device="cuda")
        cmask = torch.zeros((len(seqs), maxlen), dtype=torch.float, device="cuda")
        for r, (seq, cm) in enumerate(zip(seqs, comp_mask)):
            input_ids[r, :len(seq)] = torch.tensor(seq, device="cuda")
            attn[r, :len(seq)] = 1
            cmask[r, :len(cm)] = torch.tensor(cm, dtype=torch.float, device="cuda")
        advs_t = torch.tensor(advs, dtype=torch.bfloat16, device="cuda")

        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        # logits[:, t] predicts token[:, t+1]; align completion targets
        logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = input_ids[:, 1:]
        tok_logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1)
        m = cmask[:, 1:]                                                # mask aligned to targets
        # per-example token-mean logprob (Dr.GRPO token-level), weighted by advantage
        ex_logp = (tok_logp * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        loss = -(advs_t * ex_logp).sum() / n               # sum over examples, /n outside accum
        loss.backward()

        with torch.no_grad():
            probs = logp_all.exp()
            ent = -(probs * logp_all).sum(-1)              # (B, L-1) token entropy
            sum_entropy += (ent * m).sum().item()
            sum_logp += (tok_logp * m).sum().item()
            ent_tokens += m.sum().item()

    if cfg.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    return {"n_examples": n, "total_comp_tokens": total_tokens,
            "mean_entropy": sum_entropy / max(1, ent_tokens),
            "mean_logp": sum_logp / max(1, ent_tokens)}


# --------------------------------------------------------------------------- train loop
def main():
    ap = argparse.ArgumentParser()
    for f in Cfg.__dataclass_fields__.values():
        dv = f.default                      # use the default's runtime type (future-annotations
        if isinstance(dv, bool):            # makes f.type a string, so don't rely on it)
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true")
        else:
            ap.add_argument("--" + f.name.replace("_", "-"),
                            type=(int if isinstance(dv, int) else float if isinstance(dv, float) else str),
                            default=None)
    a = ap.parse_args()
    cfg = Cfg()
    for f in Cfg.__dataclass_fields__:
        v = getattr(a, f)
        if v is not None and not (isinstance(v, bool) and v is False):
            setattr(cfg, f, v)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    print("CONFIG:", json.dumps(cfg.__dict__, indent=2))

    model, tok = setup_model(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    arches = [a.strip() for a in cfg.archetypes.split(",") if a.strip()] or None
    stream = WorldStream(split="train", seed0=cfg.seed0, archetypes=arches)
    print(f"train stream cells restricted to archetypes: {arches or 'ALL'}")

    logf = open(cfg.log_file, "a")
    for step in range(1, cfg.steps + 1):
        bundles = stream.take(cfg.worlds_per_step)
        turns, rewards, infos = rollout_batch(model, tok, bundles, cfg)
        adv, keep, dropped = group_advantages(rewards, cfg.group, cfg)

        examples: List[Tuple[List[int], List[int], float]] = []
        for i in range(len(turns)):
            if not keep[i]:
                continue
            for (p, c) in turns[i]:
                if c:                                     # skip empty completions
                    examples.append((p, c, adv[i]))

        metrics = pg_update(model, tok, examples, cfg, opt) if examples else \
            {"n_examples": 0, "mean_entropy": 0, "mean_logp": 0, "total_comp_tokens": 0}

        import statistics as st
        rmean = st.mean(rewards); rstd = st.pstdev(rewards)
        pa = st.mean([inf.get("part_a", 0.0) for inf in infos])
        pb = st.mean([inf.get("part_b", 0.0) for inf in infos])
        acc = sum(inf.get("accepted", False) for inf in infos)
        rec = {"step": step, "reward_mean": round(rmean, 4), "reward_std": round(rstd, 4),
               "part_a": round(pa, 4), "part_b": round(pb, 4),
               "accepted": acc, "n_rollouts": len(rewards),
               "groups_dropped": dropped, "n_train_examples": metrics["n_examples"],
               "mean_entropy": round(metrics["mean_entropy"], 4),
               "mean_logp": round(metrics["mean_logp"], 4)}
        print(f"[step {step}/{cfg.steps}] r={rmean:.3f}±{rstd:.3f} A={pa:.2f} B={pb:.2f} "
              f"acc={acc}/{len(rewards)} dropped={dropped}grp "
              f"ex={metrics['n_examples']} H={metrics['mean_entropy']:.3f}")
        logf.write(json.dumps(rec) + "\n"); logf.flush()

        if step % 20 == 0 or step == cfg.steps:
            model.save_pretrained(os.path.join(cfg.save_dir, f"step{step}"))
    logf.close()
    model.save_pretrained(os.path.join(cfg.save_dir, "final"))
    print("done. LoRA saved to", cfg.save_dir)


if __name__ == "__main__":
    main()
