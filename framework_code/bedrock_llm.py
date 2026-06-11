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

        response = self.client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"].strip()

    def _supports_temperature(self) -> bool:
        """Some current Bedrock inference profiles reject temperature."""
        model_id = self.model_id.lower()
        return "claude-opus-4-7" not in model_id
