"""
bedrock_llm.py

AWS Bedrock LLM wrapper using the Converse API.
Drop-in replacement for OpenAILLM / ScientistLLM — same generate() interface.

Usage:
    from bedrock_llm import BedrockLLM

    llm = BedrockLLM(model_id="us.meta.llama3-3-70b-instruct-v1:0")
    response = llm.generate("You are helpful.", "What is 2+2?")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# 2026-10-20: Bedrock's input filter returns stopReason=content_filtered (no output at all) for Claude Opus 5.5 and
# Sonnet 5.5 on some ACED prompts. Probes (personal_docs/camera_ready/results_aced/ACED_RERUN_2026-10-20.md) found two
# triggers, and one format-only adaptation for each, applied to these two models only and to every question of the
# affected agent (not as a per-question fallback), so that each cell uses one prompt:
#   (1) the request to write a <reasoning> block before the action, in the Active agent and guess-only prompts (100% of
#       Opus prompts filtered; renaming the tag or rewording the placeholder does not help). The block request is
#       dropped; the action / memory blocks and everything else are unchanged, and the model may still reason in prose.
#   (2) the Modular agent's INIT call (Opus: 36/60 decision and 18/60 structural questions filtered). The INIT
#       instructions are sent at the top of the user turn instead of as the system prompt; the text is unchanged.
_FILTERED_FORMAT_MODELS = ("claude-opus-5", "claude-sonnet-5")
_REASONING_BLOCK_EDITS = (
    ("<reasoning>[Analysis, updated understanding, hypothesis, decision rationale]</reasoning>\n", ""),
    ("Output: <reasoning>, <action>, and <scientist_memory> blocks (in that order).",
     "Output: <action> and <scientist_memory> blocks (in that order)."),
    ('Output: <reasoning> and <action type="answer"> blocks (in that order).', 'Output: an <action type="answer"> block.'),
)
_MODULAR_INIT_PREFIX = "You are a domain-knowledgeable scientist about to start an investigation."


def _adapt_for_filter(system_blocks: List[Dict[str, Any]], converse_msgs: List[Dict[str, Any]]) -> tuple:
    """Apply adaptations (1) and (2) above; returns (system_blocks, converse_msgs, names of the edits applied)."""
    applied = []
    def edit(t: str) -> str:
        for a, b in _REASONING_BLOCK_EDITS:
            if a in t:
                t = t.replace(a, b)
                applied.append("drop_reasoning_block")
        return t
    system_blocks = [{"text": edit(b["text"])} for b in system_blocks]
    converse_msgs = [{"role": m["role"], "content": [{"text": edit(c["text"])} for c in m["content"]]}
                     for m in converse_msgs]
    if system_blocks and system_blocks[0]["text"].startswith(_MODULAR_INIT_PREFIX) and converse_msgs \
            and converse_msgs[0]["role"] == "user":
        sys_text = "\n\n".join(b["text"] for b in system_blocks)
        first = converse_msgs[0]
        converse_msgs = [{"role": "user", "content": [{"text": sys_text + "\n\n" + first["content"][0]["text"]}]}] \
            + converse_msgs[1:]
        system_blocks = []
        applied.append("init_instructions_in_user_turn")
    return system_blocks, converse_msgs, sorted(set(applied))


def _log_usage(model_id: str, response: Dict[str, Any], adapted: List[str]) -> None:
    """Append one line of token usage per call to $ACED_USAGE_LOG (for cost accounting); no-op when unset."""
    path = os.environ.get("ACED_USAGE_LOG")
    if not path:
        return
    import json
    u = response.get("usage") or {}
    blocks = (response.get("output") or {}).get("message", {}).get("content", []) or []
    n_text = sum(1 for b in blocks if "text" in b and b["text"].strip())
    with open(path, "a") as f:
        f.write(json.dumps({"model": model_id, "stop": response.get("stopReason"), "in": u.get("inputTokens"),
                            "out": u.get("outputTokens"), "text_blocks": n_text, "adapted": adapted}) + "\n")


@dataclass
class BedrockLLM:
    """
    LLM wrapper using AWS Bedrock's Converse API.

    Args:
        model_id: Bedrock model identifier
            (e.g. "us.meta.llama3-3-70b-instruct-v1:0",
             "us.anthropic.claude-3-5-sonnet-20241022-v2:0").
        region_name: AWS region. Falls back to env var ``AWS_DEFAULT_REGION``
            or "us-west-2".
        max_new_tokens: Default max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
    """
    model_id: str = "us.meta.llama3-3-70b-instruct-v1:0"
    region_name: Optional[str] = None
    max_new_tokens: int = 1536
    temperature: float = 0.3
    top_p: float = 0.9

    client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        region = self.region_name or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        # Most local runs authenticate via an AWS profile, env credentials, or
        # a Bedrock API key. If none are present, botocore otherwise waits on
        # the EC2 metadata endpoint before failing, which is confusing on a
        # workstation.
        if os.environ.get("BEDROCK_ALLOW_EC2_METADATA", "").lower() not in {"1", "true", "yes"}:
            os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        connect_timeout = float(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "10"))
        read_timeout = float(os.environ.get("BEDROCK_READ_TIMEOUT", "120"))
        max_attempts = int(os.environ.get("BEDROCK_MAX_ATTEMPTS", "3"))
        config = Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": max_attempts, "mode": "standard"},
        )
        self.client = boto3.client(
            "bedrock-runtime", region_name=region, config=config,
        )
        session = boto3.Session()
        credentials = session.get_credentials()
        credential_method = getattr(credentials, "method", None) if credentials is not None else None
        logger.info(
            f"BedrockLLM ready — model={self.model_id}, region={region}, "
            f"read_timeout={read_timeout}s, credential_method={credential_method}, "
            f"AWS_PROFILE_set={bool(os.environ.get('AWS_PROFILE'))}, "
            f"AWS_BEARER_TOKEN_BEDROCK_set={bool(os.environ.get('AWS_BEARER_TOKEN_BEDROCK'))}, "
            f"AWS_EC2_METADATA_DISABLED={os.environ.get('AWS_EC2_METADATA_DISABLED')}"
        )

    def _to_bedrock_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple:
        """Convert OpenAI-style messages to Bedrock converse format.

        Returns (system_blocks, converse_messages).
        """
        system_blocks: List[Dict[str, Any]] = []
        converse_msgs: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_blocks.append({"text": text})
            else:
                converse_msgs.append({
                    "role": role,
                    "content": [{"text": text}],
                })

        return system_blocks, converse_msgs

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response via AWS Bedrock Converse API."""
        return self.generate_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=max_new_tokens,
        )

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from a full message list (supports multi-turn)."""
        system_blocks, converse_msgs = self._to_bedrock_messages(messages)
        inference_config: Dict[str, Any] = {
            "maxTokens": max_new_tokens or self.max_new_tokens,
        }
        if self._supports_temperature():
            inference_config["temperature"] = self.temperature

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": converse_msgs,
            "inferenceConfig": inference_config,
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        adapted = []
        if any(m in self.model_id.lower() for m in _FILTERED_FORMAT_MODELS):
            sb, cm, adapted = _adapt_for_filter(kwargs.get("system", []), converse_msgs)
            kwargs["messages"] = cm
            if sb:
                kwargs["system"] = sb
            else:
                kwargs.pop("system", None)

        response = self.client.converse(**kwargs)
        _log_usage(self.model_id, response, adapted)
        # Newer models (Opus 5.x, GPT-6 on Bedrock) can put a reasoningContent block before the text block,
        # so content[0] is not always the answer. 2026-10-20: GPT-6 Astra/Sol return several text blocks in 11-18%
        # of calls. The blocks are NOT draft + final: the first block is the model's action for this turn; after it
        # the model continues the episode on its own (more hidden reasoning, sometimes a fabricated "tool execution
        # output", further actions, and a final answer or give-up based on results it never received). Use the
        # FIRST non-empty text block. (The old code joined all blocks: the agents executed the first action too, but
        # the code-loop agents kept the fabricated continuation in their message history. Taking the LAST block,
        # as rpg_benchmark v10.1 does, executes the fabricated continuation: ACED_RERUN_2026-10-20.md.)
        blocks = response["output"]["message"].get("content", []) or []
        texts = [b["text"] for b in blocks if "text" in b and b["text"].strip()]
        text = (texts[0] if texts else "").strip()
        if not text:
            raise RuntimeError(f"no text in Bedrock reply (stopReason={response.get('stopReason')})")
        return text

    def _supports_temperature(self) -> bool:
        """Some current Bedrock inference profiles reject temperature."""
        model_id = self.model_id.lower()
        # Opus 4.7+ inference profiles reject an explicit temperature; omit it
        # for the whole 4.x Opus line to stay safe as new profiles ship.
        # Opus 5.x / Sonnet 5.x ("`temperature` is deprecated") and GPT-6 on Bedrock reject it too (checked 2026-10-13).
        # Haiku 5.x too (checked 2026-10-23, A7 leakage audit).
        blocked = ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-4-9", "claude-opus-5", "claude-sonnet-5",
                   "claude-haiku-5", "openai.gpt-6")
        return not any(b in model_id for b in blocked)
