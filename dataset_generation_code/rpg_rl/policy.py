#!/usr/bin/env python3
"""Generation policies for RPG RL rollouts.

A "policy" here is just a callable ``gen_fn(system, user, max_new_tokens) -> str`` that
returns the model's raw turn text (which `RPGEnv.step` then parses). Keeping this as a
plain callable lets the same rollout code drive either:

- `VLLMPolicy`  — hits the running vLLM OpenAI server (fast, batched). Used for the
  group-variance probe (§6 of the Phase-1 design doc) and for eval. Base or LoRA model.
- (the trainer supplies its own on-policy HF generation; see train_grpo.py)

Qwen3 thinking is TOGGLEABLE: `enable_thinking=False` (the debug-loop default) is passed
through the chat template so episodes stay short (V15: thinking ~doubles trace length).
"""

from __future__ import annotations

import os
from typing import Optional

from openai_llm import OpenAILLM, resolve_preset


class VLLMPolicy:
    """Policy backed by the vLLM OpenAI-compatible server."""

    def __init__(self, model: str = "Qwen/Qwen3-8B", base_url: Optional[str] = None,
                 max_new_tokens: int = 768, enable_thinking: bool = False,
                 temperature: Optional[float] = None, top_p: Optional[float] = None):
        base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        preset = resolve_preset(model) or {}
        extra_body = dict(preset.get("extra_body", {}))          # top_k / min_p sampling
        # Qwen3 thinking switch (chat-template kwarg understood by vLLM's Qwen3 template)
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        self.enable_thinking = enable_thinking
        self.llm = OpenAILLM(model_name=model, base_url=base_url, api_key="EMPTY",
                             max_new_tokens=max_new_tokens, temperature=temperature,
                             top_p=top_p, extra_body=extra_body, capture_reasoning=True)

    def __call__(self, system: str, user: str,
                 max_new_tokens: Optional[int] = None) -> str:
        # /no_think is a Qwen3 soft-switch fallback if the template kwarg is ignored.
        if not self.enable_thinking and "/no_think" not in user:
            user = user + "\n/no_think"
        return self.llm.generate(system, user, max_new_tokens)
