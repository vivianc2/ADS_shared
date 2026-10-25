#!/usr/bin/env python3
"""CPU test of trainer_rlv10.shrink_overlong inside SkyRL's own advantage path (2026-10-25).

Builds a step-wise batch (2 GRPO groups x 2 episodes x 3 turns), shrinks one middle turn and one last turn, then runs
the real RayPPOTrainer.compute_advantages_and_returns on both versions. Checks: every row keeps its place and its
episode's advantage; a shrunk row has 2 tokens and loss mask 0; the last-step reward survives the shrink.
Run in the container:
  docker exec -u root -w /work/SkyRL skyrl /work/SkyRL/.venv/bin/python -m examples.train.rpg_v10.test_trainer_rlv10
"""
import copy

import torch

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

from examples.train.rpg_v10.trainer_rlv10 import RLv10Trainer, shrink_overlong


def _batch():
    go = {"prompt_token_ids": [], "response_ids": [], "rewards": [], "loss_masks": [], "rollout_logprobs": [],
          "is_last_step": [], "stop_reasons": []}
    uids, final = [], {("w0", 0): 1.0, ("w0", 1): 0.0, ("w1", 0): 0.3, ("w1", 1): 0.7}
    for (w, e), r in final.items():
        for t in range(3):
            plen = 50 + 40 * t + (500 if (w, e, t) in (("w0", 0, 1), ("w1", 1, 2)) else 0)   # two overlong rows
            go["prompt_token_ids"].append(list(range(1, plen + 1)))
            go["response_ids"].append(list(range(7, 17)))
            go["rewards"].append([0.0] * 9 + [r if t == 2 else 0.0])
            go["loss_masks"].append([1] * 10)
            go["rollout_logprobs"].append([-0.5] * 10)
            go["is_last_step"].append(t == 2)
            go["stop_reasons"].append("stop")
            uids.append(w)
    return go, uids


def _advantages(go, uids):
    cfg = SkyRLTrainConfig()
    cfg.generator.step_wise_trajectories = True
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.algorithm.grpo_norm_by_std = False
    seq, att, resp, rew, lm, lp, _ = convert_prompts_responses_to_batch_tensors(
        0, go["prompt_token_ids"], go["response_ids"], go["rewards"], go["loss_masks"], go["rollout_logprobs"])
    b = TrainingInputBatch({"sequences": seq, "attention_mask": att, "response_mask": resp, "rewards": rew,
                            "loss_mask": lm, "values": None})
    b.metadata = {"uids": uids, "is_last_step": go["is_last_step"], "avg_response_length": 10.0}
    t = RLv10Trainer.__new__(RLv10Trainer)
    t.cfg, t.all_metrics = cfg, {}
    out = t.compute_advantages_and_returns(b)
    a = out["advantages"]
    m = out["response_mask"].bool()
    return [float(a[i][m[i]][0]) for i in range(a.shape[0])], out["loss_mask"], seq


def main():
    go, uids = _batch()
    ref, _, _ = _advantages(copy.deepcopy(go), uids)
    n, longest = shrink_overlong(go, 550)
    assert n == 2 and longest == 50 + 80 + 500 + 10, (n, longest)
    got, lm, seq = _advantages(go, uids)
    for i in (1, 11):                           # the shrunk rows: (w0, ep0, turn1), (w1, ep1, turn2)
        assert len(go["prompt_token_ids"][i]) == 1 and len(go["response_ids"][i]) == 1
        assert int(lm[i].sum()) == 0, "shrunk row must carry no loss"
    assert go["rewards"][11] == [0.7], go["rewards"][11]
    assert all(abs(x - y) < 1e-6 for x, y in zip(ref, got)), (ref, got)
    assert int(lm.sum()) == 10 * (len(go["loss_masks"]) - 2)
    print(f"OK: 2/{len(uids)} rows shrunk, advantages unchanged {sorted(set(round(x, 3) for x in got))}, "
          f"padded length {seq.shape[1]} (was {50 + 80 + 500 + 10})")


if __name__ == "__main__":
    main()
