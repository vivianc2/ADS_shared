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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    """Start and stop only the three vLLM processes owned by this invocation."""

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
            "--disable-log-requests",
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
            raise ValueError("the full topology requires exactly three GPUs and three ports")
        if len(set(self.settings.gpus)) != 3 or len(set(self.settings.ports)) != 3:
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
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    enable_thinking: bool = True
    request_timeout_s: int = 900
    transport_retries: int = 3

    def request_record(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }


@dataclass(frozen=True)
class Generation:
    raw_text: str
    action_text: str
    reasoning_content: str | None
    finish_reason: str | None
    usage: dict[str, Any]
    attempts: int
    latency_s: float


class VLLMClient:
    def __init__(self, base_url: str, model: str, settings: SamplingSettings,
                 api_key: str = "EMPTY"):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.settings = settings
        self.api_key = api_key

    def generate(self, system_prompt: str, observation: str, seed: int) -> Generation:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation},
            ],
            "seed": seed,
            **self.settings.request_record(),
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
                return Generation(
                    raw_text=content,
                    action_text=action_text,
                    reasoning_content=reasoning,
                    finish_reason=choice.get("finish_reason"),
                    usage=decoded.get("usage") or {},
                    attempts=attempt,
                    latency_s=float(time.perf_counter() - start),
                )
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt == total_attempts:
                    break
                time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
        raise RuntimeError(
            f"model request failed after {total_attempts} attempts with seed {seed}: "
            + " | ".join(errors)
        )
