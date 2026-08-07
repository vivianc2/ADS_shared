#!/usr/bin/env python3
"""OpenAI-compatible LLM client for RPG v6 (Nautilus / vLLM / OpenAI).

Self-contained copy of the framework's OpenAILLM so the v6 prototype does not
import world_model_causal.py (which pulls in torch at module load). Same
generate(system, user, max_new_tokens) interface used by the runners, plus
per-model sampling presets (Qwen3.6, gpt-oss, DeepSeek) and reasoning_content
capture.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openai_llm")

# Per-model recommended sampling + max output tokens. 'match' is a case-insensitive
# substring of the model name. Explicit kwargs to OpenAILLM override these.
# Qwen3.6 is served on NRP Nautilus as "qwen3-small"; match both strings.
# max_output_tokens = the model's true generation ceiling; we always request the
# max so a verbose/thinking model is never truncated mid-answer (costs nothing
# unless the model actually generates that many tokens).
_MODEL_PRESETS: List[Dict[str, Any]] = [
    {"matches": ["qwen3.6", "qwen3-small", "qwen3"], "temperature": 1.0, "top_p": 0.95,
     "extra_body": {"top_k": 20, "min_p": 0.0}, "max_output_tokens": 32768},
    {"matches": ["gpt-oss"], "temperature": 1.0, "top_p": 1.0,
     "extra_body": {"top_k": 0, "min_p": 0.0}, "max_output_tokens": 32768},
    {"matches": ["deepseek-v4-pro"], "temperature": 1.0, "top_p": 0.95,
     "extra_body": {"reasoning_effort": "high", "thinking": {"type": "enabled"}},
     "max_output_tokens": 65536},
]


def resolve_preset(model_name: str) -> Optional[Dict[str, Any]]:
    name = (model_name or "").lower()
    for preset in _MODEL_PRESETS:
        if any(m.lower() in name for m in preset["matches"]):
            return preset
    return None


@dataclass
class OpenAILLM:
    model_name: str = "qwen3.6"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # max_new_tokens=None -> auto-set to the model's max output (preset, else a
    # safe large default). An explicit int caps it lower. We always request the
    # model's true ceiling so a verbose/thinking model is never truncated.
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extra_body: Optional[Dict[str, Any]] = None
    use_preset: bool = True
    capture_reasoning: bool = True

    DEFAULT_MAX_OUTPUT: int = 32768   # fallback ceiling for unknown models

    client: Any = field(default=None, init=False, repr=False)
    last_reasoning: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from openai import OpenAI

        resolved_base = self.base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        preset = resolve_preset(self.model_name) if self.use_preset else None
        if self.temperature is None:
            self.temperature = preset["temperature"] if preset else 0.3
        if self.top_p is None:
            self.top_p = preset["top_p"] if preset else 0.9
        if self.extra_body is None and preset and "extra_body" in preset:
            self.extra_body = dict(preset["extra_body"])
        # resolve the output ceiling once: explicit value wins, else preset max,
        # else the safe large default.
        if self.max_new_tokens is None:
            self.max_new_tokens = (preset.get("max_output_tokens") if preset else None) \
                or self.DEFAULT_MAX_OUTPUT

        self.client = OpenAI(base_url=resolved_base, api_key=resolved_key)
        logger.info(
            "OpenAILLM ready — model=%s base_url=%s temp=%s top_p=%s max_out=%s preset=%s",
            self.model_name, resolved_base, self.temperature, self.top_p,
            self.max_new_tokens, "yes" if preset else "no",
        )

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: Optional[int] = None) -> str:
        return self.generate_messages(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            max_new_tokens=max_new_tokens,
        )

    def generate_messages(self, messages: List[Dict[str, Any]],
                          max_new_tokens: Optional[int] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if self.capture_reasoning:
            self.last_reasoning = reasoning
        content = msg.content or ""
        if not content.strip() and reasoning:
            logger.warning("empty content, using reasoning_content (%d chars)", len(reasoning))
            content = reasoning
        return content.strip()
