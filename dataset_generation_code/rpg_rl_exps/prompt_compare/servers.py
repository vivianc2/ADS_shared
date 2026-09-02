"""Owned vLLM worker lifecycle and a seed-aware HTTP client."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


class NonRetryableRequestError(RuntimeError):
    """The server rejected a request that cannot succeed unchanged."""


class ContextLengthExceededError(NonRetryableRequestError):
    """The accumulated chat plus requested output exceeded the model context."""

    def __init__(self, message: str, *, prompt_tokens: int | None = None):
        self.prompt_tokens = prompt_tokens
        super().__init__(message)


class InputLengthExceededError(ContextLengthExceededError):
    """The rendered chat prompt exceeded the client-side SkyRL input limit."""

    def __init__(self, prompt_tokens: int, max_input_tokens: int):
        self.prompt_tokens = int(prompt_tokens)
        self.max_input_tokens = int(max_input_tokens)
        super().__init__(
            f"client-side prompt length {self.prompt_tokens} exceeds "
            f"max_input_tokens={self.max_input_tokens}",
            prompt_tokens=self.prompt_tokens,
        )


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        payload = ""
    return payload.strip() or str(exc.reason)


def _is_context_length_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in (
        "maximum context length",
        "max_model_len",
        "context length",
        "prompt is too long",
        "too many tokens",
        "prompt tokens",
    ))


@dataclass(frozen=True)
class ServerSettings:
    model: str
    served_model_name: str
    host: str
    ports: tuple[int, ...]
    gpus: tuple[str, ...]
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    health_timeout_s: int
    executable: str
    disable_multimodal: bool


@dataclass
class OwnedServer:
    gpu: str
    port: int
    process: subprocess.Popen
    log_path: Path
    log_handle: Any


def inspect_gpus(gpus: tuple[str, ...]) -> list[dict[str, Any]]:
    """Confirm requested physical GPU indices exist and capture memory preflight data."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"GPU preflight could not run nvidia-smi: {exc}") from exc
    inventory = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        inventory.append({
            "index": fields[0],
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_free_mib": int(fields[3]),
        })
    by_index = {entry["index"]: entry for entry in inventory}
    missing = [gpu for gpu in gpus if gpu not in by_index]
    if missing:
        raise RuntimeError(f"requested GPU indices not reported by nvidia-smi: {missing}")
    return [by_index[gpu] for gpu in gpus]


def _assert_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"configured port {host}:{port} is unavailable: {exc}") from exc


def _tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(server log unavailable)"


class ServerManager:
    """Start and stop only the vLLM processes owned by this invocation."""

    def __init__(self, settings: ServerSettings, logs_dir: Path):
        self.settings = settings
        self.logs_dir = logs_dir
        self.owned: list[OwnedServer] = []

    @property
    def base_urls(self) -> tuple[str, ...]:
        return tuple(
            f"http://{self.settings.host}:{server.port}/v1" for server in self.owned
        )

    def _launch_one(self, gpu: str, port: int) -> OwnedServer:
        executable = shutil.which(self.settings.executable)
        if executable is None:
            raise RuntimeError(
                f"vLLM executable {self.settings.executable!r} is not on PATH"
            )
        _assert_port_available(self.settings.host, port)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"vllm_gpu_{gpu}_port_{port}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        command = [
            executable,
            "serve",
            self.settings.model,
            "--served-model-name",
            self.settings.served_model_name,
            "--host",
            self.settings.host,
            "--port",
            str(port),
            "--dtype",
            self.settings.dtype,
            "--max-model-len",
            str(self.settings.max_model_len),
            "--gpu-memory-utilization",
            str(self.settings.gpu_memory_utilization),
        ]
        if self.settings.disable_multimodal:
            command.extend(["--limit-mm-per-prompt", '{"image":0,"video":0}'])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["RPG_PROTO"] = "rpg_v9"
        env["RPG_SYNERGY_SOFT"] = "20"
        try:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise
        server = OwnedServer(gpu, port, process, log_path, log_handle)
        self.owned.append(server)
        return server

    def _wait_healthy(self, server: OwnedServer) -> None:
        deadline = time.monotonic() + self.settings.health_timeout_s
        url = f"http://{self.settings.host}:{server.port}/health"
        last_error = "not contacted"
        while time.monotonic() < deadline:
            code = server.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"vLLM on GPU {server.gpu} exited with code {code} during startup.\n"
                    f"Last log lines:\n{_tail(server.log_path)}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if 200 <= response.status < 300:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(2)
        raise RuntimeError(
            f"vLLM on GPU {server.gpu} did not become healthy within "
            f"{self.settings.health_timeout_s}s ({last_error}).\n"
            f"Last log lines:\n{_tail(server.log_path)}"
        )

    def start(self) -> tuple[str, ...]:
        if len(self.settings.gpus) != 3 or len(self.settings.ports) != 3:
            raise ValueError("exactly three GPUs and three ports are required")
        if (
            len(set(self.settings.gpus)) != len(self.settings.gpus)
            or len(set(self.settings.ports)) != len(self.settings.ports)
        ):
            raise ValueError("GPU ids and ports must each be unique")
        inspect_gpus(self.settings.gpus)
        try:
            # Starting and health-checking worker zero is the required empirical
            # one-GPU fit preflight. Only after it succeeds are the other workers started.
            first = self._launch_one(self.settings.gpus[0], self.settings.ports[0])
            self._wait_healthy(first)
            remaining = [
                self._launch_one(gpu, port)
                for gpu, port in zip(self.settings.gpus[1:], self.settings.ports[1:])
            ]
            for server in remaining:
                self._wait_healthy(server)
            return self.base_urls
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for server in reversed(self.owned):
            if server.process.poll() is None:
                try:
                    os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    server.process.terminate()
        deadline = time.monotonic() + 30
        for server in reversed(self.owned):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                server.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    server.process.kill()
                server.process.wait(timeout=10)
            finally:
                server.log_handle.close()
        self.owned.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()


@dataclass(frozen=True)
class SamplingSettings:
    max_input_tokens: int = 18432
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    enable_thinking: bool = True
    request_timeout_s: int = 900
    transport_retries: int = 3

    def request_payload(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def request_record(self) -> dict[str, Any]:
        return {
            "max_input_tokens": self.max_input_tokens,
            **self.request_payload(),
        }


class ChatTemplateTokenCounter:
    """Count the exact rendered chat prompt tokens before sending a request."""

    def __init__(self, model: str, *, enable_thinking: bool, tokenizer=None):
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"could not load tokenizer for client-side input-length checks: {model}"
                ) from exc
        self.tokenizer = tokenizer
        self.enable_thinking = bool(enable_thinking)
        # One counter is shared by all rollout workers. Tokenizer calls are short
        # compared with generation, and the lock avoids relying on tokenizer/Jinja
        # thread-safety while concurrent G=8 requests are prepared.
        self._lock = Lock()

    def __call__(self, messages: list[dict[str, str]]) -> int:
        with self._lock:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=self.enable_thinking,
            )
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise ValueError("expected one chat conversation while counting tokens")
            input_ids = input_ids[0]
        return len(input_ids)


@dataclass(frozen=True)
class Generation:
    raw_text: str
    action_text: str
    reasoning_content: str | None
    finish_reason: str | None
    usage: dict[str, Any]
    attempts: int
    latency_s: float
    prompt_tokens: int | None = None


class VLLMClient:
    def __init__(self, base_url: str, model: str, settings: SamplingSettings,
                 api_key: str = "EMPTY", token_counter=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.settings = settings
        self.api_key = api_key
        self.token_counter = token_counter

    def generate(self, messages: list[dict[str, str]], seed: int) -> Generation:
        if len(messages) < 2 or messages[0].get("role") != "system":
            raise ValueError("messages must start with a system message and include a user turn")
        if any(
            message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("every message must have a supported role and string content")
        expected_roles = [
            "system",
            *("user" if index % 2 else "assistant" for index in range(1, len(messages))),
        ]
        if (
            [message["role"] for message in messages] != expected_roles
            or messages[-1]["role"] != "user"
        ):
            raise ValueError("messages must alternate user and assistant and end with user")
        prompt_tokens = None
        if self.token_counter is not None:
            prompt_tokens = int(self.token_counter(messages))
            if prompt_tokens > self.settings.max_input_tokens:
                raise InputLengthExceededError(
                    prompt_tokens,
                    self.settings.max_input_tokens,
                )
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "seed": seed,
            **self.settings.request_payload(),
        }
        body = json.dumps(payload).encode("utf-8")
        start = time.perf_counter()
        errors = []
        total_attempts = self.settings.transport_retries + 1
        for attempt in range(1, total_attempts + 1):
            request = urllib.request.Request(
                self.url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.settings.request_timeout_s
                ) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                choice = decoded["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                if not isinstance(content, str):
                    raise ValueError("response message.content is not a string")
                reasoning = message.get("reasoning_content")
                if reasoning is not None and not isinstance(reasoning, str):
                    reasoning = str(reasoning)
                action_text = content
                if reasoning and "<reasoning>" not in content:
                    action_text = f"<reasoning>{reasoning}</reasoning>\n{content}"
                usage = decoded.get("usage") or {}
                if prompt_tokens is None:
                    reported_prompt_tokens = usage.get("prompt_tokens")
                    if isinstance(reported_prompt_tokens, int):
                        prompt_tokens = reported_prompt_tokens
                return Generation(
                    raw_text=content,
                    action_text=action_text,
                    reasoning_content=reasoning,
                    finish_reason=choice.get("finish_reason"),
                    usage=usage,
                    attempts=attempt,
                    latency_s=float(time.perf_counter() - start),
                    prompt_tokens=prompt_tokens,
                )
            except urllib.error.HTTPError as exc:
                detail = _http_error_detail(exc)
                message = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and _is_context_length_error(detail):
                    raise ContextLengthExceededError(
                        message,
                        prompt_tokens=prompt_tokens,
                    ) from exc
                if 400 <= exc.code < 500 and exc.code not in {408, 409, 425, 429}:
                    raise NonRetryableRequestError(message) from exc
                errors.append(message)
                if attempt == total_attempts:
                    break
                time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt == total_attempts:
                    break
                time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
        raise RuntimeError(
            f"model request failed after {total_attempts} attempts with seed {seed}: "
            + " | ".join(errors)
        )
