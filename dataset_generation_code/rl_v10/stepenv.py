#!/usr/bin/env python3
"""RL environment for RPG v10 that IS the release benchmark loop (2026-10-23, RL session).

Why this design: v9 RL trained in `rpg_rl/env.py`, a separate re-implementation of the task (id catalog, its own
prompt, parser, budget rule and grader), and the protocol gaps cost us results. Here the episode is run by the
release harness's own `agent_loop.run_world` (from the frozen copy in `rl_v10/bench/`), unchanged. The policy is
plugged in through a stand-in LLM object: `run_world` asks it for each reply, the stand-in hands the request out to
the trainer and blocks until `step()` delivers the policy's text. So the prompt, the reply parser, the budget
refusal, the directives, the resolver (deterministic, no LLM), the code sandbox and the grader are byte-for-byte the
benchmark's. `run_world` runs in a worker thread; `reset()` and `step()` exchange one message each way.

    env = BenchStepEnv(world_file, data_dir)
    msgs = env.reset()                      # [system, user] = what run_world sends on turn 0
    while True:
        out = env.step(policy_text)         # policy_text = the model's full turn output (may contain thinking)
        if out.done: break
        next_user = out.observation         # the new user message run_world appends (full history)
    out.reward, out.info["components"], env.result (= run_world's result dict, as in result_*.json)

Thinking: the eval path serves the model with vLLM's qwen3 reasoning parser, so `run_world` receives the reply
WITHOUT the thinking (the content after `</think>`), and the conversation history it keeps holds content only.
`step()` does the same split (`split_thinking`, same rule as the parser) so run_world sees what it sees in eval.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.environ.get("RL_V10_BENCH", os.path.join(_HERE, "bench"))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import agent_loop                       # noqa: E402  (frozen release harness)
from agent_loop import run_world, load_world_file   # noqa: E402
from simulator import Simulator         # noqa: E402

from reward_v10 import reward_components, RewardCfg   # noqa: E402
import sandbox_hardened                                 # noqa: E402

# Training always runs agent code in the hardened sandbox (sandbox_hardened.py: no ctypes route to files, no
# environment, no UDP); identical output to the release sandbox on legitimate code (replay_parity with
# RLV10_PARITY_B_HARDENED=1). RLV10_HARDEN_SANDBOX=0 keeps the release sandbox (parity tests only).
if os.environ.get("RLV10_HARDEN_SANDBOX", "1") != "0":
    sandbox_hardened.install()

BENCH_HARNESS_VERSION = agent_loop.HARNESS_VERSION


_EOS_STRINGS = tuple(x for x in os.environ.get("RLV10_EOS_STRINGS", "<|im_end|>,<|endoftext|>").split(",") if x)


class _Abort(BaseException):
    """Raised inside run_world's thread to unwind an abandoned episode. BaseException, so none of run_world's
    `except Exception/RuntimeError/ValueError` clauses catch it."""


def split_thinking(text: str, thinking: Optional[str] = None):
    """(reasoning, content) exactly as the eval path produces them for a Qwen3.5 reply.
    Thinking off (`thinking="off"`, default from RLV10_THINKING, else "on"): the eval server runs without a reasoning
    parser, so the whole reply is content and OpenAILLM returns content.strip(), reasoning None. A thinking-off reply
    can still carry inline reasoning closed by a stray '</think>' (34 of 42,117 base-9B turns, LOG_2026-10-24.md);
    eval hands that whole reply to run_world, so we must not split it.
    Thinking on:
    1. vLLM 0.19.1 `Qwen3ReasoningParser.extract_reasoning`: drop everything up to a first '<think>' if present;
       no '</think>' -> (all reasoning, content None) (truncated thinking); else split at the first '</think>',
       empty content -> None.
    2. `llm_openai.OpenAILLM.generate_messages`: content = content or ""; if it is blank and there is reasoning,
       content = reasoning; return content.strip().
    Known difference: OpenAILLM first RETRIES a blank reply up to twice (new samples) before step 2; a trainer
    cannot resample inside a turn, so here the blank reply falls through to step 2 at once. Count such turns."""
    text = text or ""
    thinking = thinking or os.environ.get("RLV10_THINKING", "on")
    if thinking == "off":
        return None, text.strip()
    if thinking != "on":
        raise ValueError(f"RLV10_THINKING must be on or off, got {thinking!r}")
    head, sep, tail = text.partition("<think>")
    body = tail if sep else head
    if "</think>" not in body:
        reasoning, content = body, None
    else:
        reasoning, _, content = body.partition("</think>")
        content = content or None
    c = content or ""
    if not c.strip() and reasoning:
        c = reasoning
    return reasoning, c.strip()


class StepLLM:
    """Stand-in for the agent LLM. run_world calls generate_messages(msgs); we publish msgs and wait for the reply."""

    def __init__(self):
        self.req_q: "queue.Queue" = queue.Queue(maxsize=1)
        self.rep_q: "queue.Queue" = queue.Queue(maxsize=1)
        self.last_finish_reason: Optional[str] = None
        self.last_reasoning: Optional[str] = None
        self.last_usage = None
        self.n_calls = 0

    def generate_messages(self, messages: List[Dict[str, Any]], max_new_tokens: Optional[int] = None) -> str:
        self.n_calls += 1
        self.req_q.put(("prompt", [dict(m) for m in messages]))
        kind, payload = self.rep_q.get()
        if kind == "abort":
            raise _Abort()
        content, reasoning, finish = payload
        self.last_reasoning, self.last_finish_reason = reasoning, finish
        return content

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        return self.generate_messages([{"role": "system", "content": system_prompt},
                                       {"role": "user", "content": user_prompt}])


@dataclass
class StepOut:
    observation: Optional[str]          # the next user message (None when done)
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class BenchStepEnv:
    """One episode of the release benchmark on one world, driven turn by turn."""

    def __init__(self, world_file: str, data_dir: str, *, history: str = "full", budget: int = 15,
                 max_turns: int = 32, reward_cfg: Optional[RewardCfg] = None, timeout_s: float = 600.0):
        self.world_file = world_file
        self.data_dir = data_dir
        self.history = history
        self.budget, self.max_turns = budget, max_turns
        self.reward_cfg = reward_cfg or RewardCfg()
        self.timeout_s = timeout_s
        self.world, self.pre = load_world_file(world_file)
        self.llm = StepLLM()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._msgs: List[Dict[str, Any]] = []     # the last request run_world made
        self._assistant: List[str] = []           # contents we delivered, in order (consistency check)
        self.raw_turns: List[str] = []            # full policy outputs, thinking included
        self.done = False

    # -- worker thread ------------------------------------------------------------------------------------------
    def _run(self):
        try:
            sim = Simulator(self.world, resolver_llm=None, data_dir=self.data_dir, precomputed=self.pre)
            self.result = run_world(sim, self.llm, self.max_turns, self.budget, False, history=self.history)
        except _Abort:
            pass
        except BaseException as e:             # surfaced to the caller of step()
            self.error = e
        finally:
            self.llm.req_q.put(("done", None))

    def _next(self):
        kind, msgs = self.llm.req_q.get(timeout=self.timeout_s)
        if kind == "done":
            if self.error is not None:
                raise RuntimeError(f"run_world failed: {type(self.error).__name__}: {self.error}") from self.error
            return None
        return msgs

    # -- API ----------------------------------------------------------------------------------------------------
    def reset(self) -> List[Dict[str, Any]]:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        msgs = self._next()
        assert msgs is not None and len(msgs) == 2, "run_world must open with [system, user]"
        self._msgs = msgs
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    def step(self, text: str, finish_reason: Optional[str] = "stop") -> StepOut:
        assert not self.done, "step() after the episode ended"
        self.raw_turns.append(text)
        # SkyRL's decoded output keeps the end-of-turn token ('<|im_end|>' for Qwen); vLLM's chat content (the eval
        # path) never contains it, and SkyRL strips it from its own chat history too (_update_chat_history).
        for eos in _EOS_STRINGS:
            if text.endswith(eos):
                text = text[: -len(eos)]
        reasoning, content = split_thinking(text)
        self._assistant.append(content)
        self.llm.rep_q.put(("reply", (content, reasoning, finish_reason)))
        msgs = self._next()
        if msgs is None:
            self.done = True
            self._thread.join(timeout=60)
            comps = reward_components(self.result, self.world, self.pre)
            r = self.reward_cfg.reward(comps)
            return StepOut(None, r, True, {"components": comps, "turns": len(self.raw_turns)})
        if self.history == "full":
            self._check_history(msgs)
        self._msgs = msgs
        return StepOut(msgs[-1]["content"], 0.0, False, {"turns": len(self.raw_turns)})

    def _check_history(self, msgs):
        """Full history: run_world's request must be [system, user0, asst0, user1, ..., asst_k, user_k+1] with the
        assistant contents exactly what we delivered. Fails loudly on any drift between what the trainer thinks the
        conversation is and what the benchmark loop sent."""
        roles = [m["role"] for m in msgs]
        n = len(self._assistant)
        want = ["system"] + ["user", "assistant"] * n + ["user"]
        if roles != want:
            raise AssertionError(f"history roles {roles} != {want}")
        got = [m["content"] for m in msgs if m["role"] == "assistant"]
        exp = [c or "(empty reply)" for c in self._assistant]
        if got != exp:
            raise AssertionError("assistant history differs from the delivered replies")

    def close(self):
        """Abandon a running episode (e.g. the trainer hit its context limit). Safe to call twice."""
        if self._thread is not None and self._thread.is_alive() and not self.done:
            try:
                self.llm.rep_q.put_nowait(("abort", None))
            except queue.Full:
                pass
            self._thread.join(timeout=30)
        self.done = True
