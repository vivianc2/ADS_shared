#!/usr/bin/env python3
"""Unit-test the GRPO loss/backward path independent of whether rollouts produce reward
variance. Fabricates a few (prompt_ids, completion_ids, advantage) examples and confirms
pg_update runs backward+step, returns finite entropy/logp, and updates a LoRA param.

Run: PYTHONPATH=../rpg_v7_prototype CUDA_VISIBLE_DEVICES=1 python test_loss_path.py
"""
from __future__ import annotations
import torch
from train_grpo import Cfg, setup_model, build_prompt_ids, pg_update
from env import SYSTEM_PROMPT

def main():
    cfg = Cfg(max_new_tokens=32, loss_micro_batch=2)
    model, tok = setup_model(cfg)
    # a real-ish prompt + a couple short completions, with mixed advantages
    p = build_prompt_ids(tok, "SITUATION\nfoo\nMEASURABLE SIGNALS:\n m0: x\n", cfg)
    c1 = tok('<reasoning>a</reasoning>\n<action type="measure">{"ids":["m0"]}</action>', add_special_tokens=False).input_ids
    c2 = tok('<reasoning>b</reasoning>\n<action type="give_up">{}</action>', add_special_tokens=False).input_ids
    examples = [(p, c1, 1.0), (p, c2, -1.0), (p, c1, 0.5)]

    # snapshot a trainable LoRA param
    lora_p = next(q for n, q in model.named_parameters() if q.requires_grad)
    before = lora_p.detach().clone()

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-3)
    m = pg_update(model, tok, examples, cfg, opt)

    changed = (lora_p.detach() - before).abs().sum().item()
    import math
    ok = (m["n_examples"] == 3 and math.isfinite(m["mean_entropy"])
          and m["mean_entropy"] > 0 and math.isfinite(m["mean_logp"]) and changed > 0)
    print(f"metrics: {m}")
    print(f"LoRA param delta (sum abs): {changed:.3e}")
    print("LOSS-PATH TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    import sys; sys.exit(main())
