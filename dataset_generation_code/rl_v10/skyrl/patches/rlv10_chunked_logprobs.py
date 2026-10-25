"""LOCAL PATCH (ygg 2026-10-23, RL v10): log-probs of target tokens through a FROZEN lm_head without materialising the
(B, K, V) logits or their gradient. Used by model_wrapper.py when RLV10_CHUNKED_LOGPROBS=1.

forward:  for each chunk of positions: logits = h_c @ W^T (fp32), lse = logsumexp, logp = logits[y] - lse,
          entropy (no grad) = lse - sum(softmax * logits)
backward: recompute logits per chunk; d logp_y / d h = W_y - softmax @ W  (W is frozen: no weight gradient).
Exact up to fp32 summation order vs log_softmax over the full logits (tested in rl_v10 scratch tests).
"""
import torch

CHUNK = 2048


class _ChunkedTargetLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature):
        B, K, H = hidden.shape
        h2 = hidden.reshape(-1, H)
        y = labels.reshape(-1)
        logp = torch.empty(h2.shape[0], device=hidden.device, dtype=torch.float32)
        ent = torch.empty_like(logp)
        lse_all = torch.empty_like(logp)
        for s in range(0, h2.shape[0], CHUNK):
            lg = (h2[s:s + CHUNK] @ weight.t()).float() / temperature
            lse = torch.logsumexp(lg, dim=-1)
            logp[s:s + CHUNK] = lg.gather(1, y[s:s + CHUNK, None]).squeeze(1) - lse
            p = torch.softmax(lg, dim=-1)
            ent[s:s + CHUNK] = lse - (p * lg).sum(-1)
            lse_all[s:s + CHUNK] = lse
            del lg, p
        ctx.save_for_backward(hidden, weight, labels, lse_all)
        ctx.temperature = temperature
        ctx.mark_non_differentiable(ent)
        return logp.view(B, K), ent.view(B, K)

    @staticmethod
    def backward(ctx, g_logp, g_ent):
        hidden, weight, labels, lse_all = ctx.saved_tensors
        B, K, H = hidden.shape
        h2 = hidden.reshape(-1, H)
        y = labels.reshape(-1)
        g = g_logp.reshape(-1).float()
        grad_h = torch.empty_like(h2)
        Wf = weight
        for s in range(0, h2.shape[0], CHUNK):
            lg = (h2[s:s + CHUNK] @ Wf.t()).float() / ctx.temperature
            p = torch.exp(lg - lse_all[s:s + CHUNK, None])          # softmax
            gc = g[s:s + CHUNK, None]
            # d logp_y/d logits = onehot(y) - p ; through logits = h W^T / T
            dlg = -p * gc
            dlg.scatter_add_(1, y[s:s + CHUNK, None], gc)
            grad_h[s:s + CHUNK] = ((dlg / ctx.temperature).to(Wf.dtype) @ Wf).to(grad_h.dtype)
            del lg, p, dlg
        return grad_h.view(B, K, H), None, None, None


def chunked_target_logprobs(hidden, weight, labels, temperature=1.0):
    """hidden (B,K,H), weight (V,H) frozen, labels (B,K) -> (logp (B,K) fp32 with grad wrt hidden, entropy (B,K))."""
    return _ChunkedTargetLogprob.apply(hidden, weight, labels, float(temperature))
