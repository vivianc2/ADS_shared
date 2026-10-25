#!/usr/bin/env python3
"""RayPPOTrainer for RL on RPG v10 (2026-10-25): drops overlong turn-sequences from the loss.

Why: smoke 7 (thinking off, LOG_2026-10-24.md 14:21) OOM'd in the policy forward on one episode whose answer looped to
the 32k output cap twice, so its history reached 97,672 tokens and its turn sequences ~100-130k. In eval no turn
sequence comes close (base 9B thinking off, 2,372 episodes: max 56,020 tokens, none > 64k). SkyRL's
`apply_overlong_filtering` only masks the loss; the forward still runs on the full sequence and still OOMs.

What it does (step-wise trajectories): every turn-sequence (prompt + response tokens) longer than
RLV10_MAX_TRAIN_SEQ (default 65536; 0 = off) is shrunk to one prompt token and one response token, with loss mask 0
and its reward kept (summed into that one position). The row stays, so `is_last_step`, the GRPO group and the
trajectory's advantage are unchanged: the episode's other turns still train, the overlong turn contributes no loss
and costs no memory. Generation and the eval protocol are untouched.

Also logs per step: rlv10/overlong_dropped (rows shrunk), rlv10/max_seq_tokens (before shrinking),
rlv10/length_stop_frac (share of turns that stopped at the output cap).

Note: dropping removes the runaway turn's own negative signal; the episode's low reward still reaches its other turns.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from skyrl.train.trainer import RayPPOTrainer


def shrink_overlong(go: Dict[str, Any], cap: int) -> Tuple[int, int]:
    """In place. Returns (rows shrunk, longest prompt+response before shrinking). Rewards must be per-token lists
    (as after RayPPOTrainer.postprocess_generator_output)."""
    lens = [len(p) + len(r) for p, r in zip(go["prompt_token_ids"], go["response_ids"])]
    longest = max(lens) if lens else 0
    if cap <= 0:
        return 0, longest
    n = 0
    for i, L in enumerate(lens):
        if L <= cap:
            continue
        n += 1
        rw = go["rewards"][i]
        total = float(sum(rw)) if isinstance(rw, list) else float(rw)
        go["prompt_token_ids"][i] = list(go["prompt_token_ids"][i][:1])
        go["response_ids"][i] = list(go["response_ids"][i][-1:])
        go["rewards"][i] = [total] if isinstance(rw, list) else total
        go["loss_masks"][i] = [0]
        lp = go.get("rollout_logprobs")
        if lp is not None:
            lp[i] = list(lp[i][-1:])
        ex = go.get("rollout_expert_indices")
        if ex is not None:
            raise NotImplementedError("shrink_overlong: rollout_expert_indices (MoE routing replay) not supported")
    return n, longest


class RLv10Trainer(RayPPOTrainer):
    def postprocess_generator_output(self, generator_output, uids, metrics_generator_output=None,
                                     metrics_uids=None):
        generator_output, uids = super().postprocess_generator_output(
            generator_output, uids, metrics_generator_output, metrics_uids)
        cap = int(os.environ.get("RLV10_MAX_TRAIN_SEQ", "65536"))
        n, longest = shrink_overlong(generator_output, cap)
        stops: Optional[List[str]] = generator_output.get("stop_reasons")
        m = {"rlv10/overlong_dropped": n, "rlv10/max_seq_tokens": longest}
        if stops:
            m["rlv10/length_stop_frac"] = sum(1 for s in stops if s == "length") / len(stops)
        self.all_metrics.update(m)
        if n:
            print(f"[rlv10] step {self.global_step}: {n} turn-sequence(s) > {cap} tokens dropped from the loss "
                  f"(longest {longest})", flush=True)
        return generator_output, uids
