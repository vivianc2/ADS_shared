#!/usr/bin/env python3
"""SkyRL-Gym wrapper over BenchStepEnv (RL on RPG v10, 2026-10-23).

One sample = one world file. The dataset row's prompt is [system, user0] exactly as run_world opens the episode;
`init()` checks it against a fresh reset of the env (fails loudly on drift, as the v9 guard did). Each `step(output)`
passes the model's full turn output (thinking included; `split_thinking` gives run_world the content the eval path
gives it) and returns run_world's next user message. The reward is terminal (reward_v10.RewardCfg on the release
grade). Use with `generator.step_wise_trajectories=true` so every turn is trained on the prompt the benchmark shows
(see personal_docs/rl_v10/LOG_2026-10-23.md, "Token parity").

Env vars:
  RLV10_DATA_ROOT     root that extra_info.world_file is relative to (container default /work/rl_v10/data)
  RLV10_EPISODE_ROOT  where per-episode experiment folders go (default: system temp)
  RLV10_TRACE_DIR     if set, every finished episode's run_world result (+ raw outputs) is written there as .json.gz
  RLV10_SANDBOX_PY    sandbox interpreter (training: /work/envs/rpg_sandbox/bin/python)
  RLV10_W_*           reward weights (reward_v10.RewardCfg)
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(os.environ.get("RLV10_SRC_FILE", __file__)))
_SRC = os.environ.get("RLV10_SRC", "/work/ADS_shared/dataset_generation_code/rl_v10")
for _p in (_SRC, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput  # noqa: E402

from stepenv import BenchStepEnv, BENCH_HARNESS_VERSION, _EOS_STRINGS  # noqa: E402


# Sandbox monitor: code turns that reach for files, the OS, the environment, ctypes or sockets. Flag only (the hardened
# sandbox already refuses them); a rising count during training means the policy is probing the sandbox: read those
# traces.
_SUSPICIOUS = re.compile(r"\bopen\s*\(|\bos\.|\bsys\.|subprocess|ctypes|\bglob\.|\bglob\s+import|pathlib|__import__|"
                         r"importlib|\bsocket\b|\benviron\b|/proc/|/home/|/work/|\bshutil\b|pickle\.load")


def _episode_metrics(result: Dict[str, Any], comps: Dict[str, Any], n_turns: int, raw_turns=()) -> Dict[str, Any]:
    turns = [t for t in (result or {}).get("turns", []) if t.get("turn") != "post"]
    errs = [t.get("error") or "" for t in turns]
    # raw policy outputs (before stepenv strips the end-of-turn token): no EOS = the turn stopped at the output cap;
    # thinking on, no '</think>' = the thinking was cut and run_world received the reasoning as the reply (2026-10-25)
    cut = [not any(r.endswith(e) for e in _EOS_STRINGS) for r in raw_turns]
    think_on = os.environ.get("RLV10_THINKING", "on") == "on"
    return {
        "cap_cut_turns": sum(cut),
        "thinking_cut_turns": sum(1 for r in raw_turns if think_on and "</think>" not in r),
        "part_a": comps.get("part_a", 0), "accepted_core": comps.get("accepted_core", 0),
        "accepted_release": comps.get("accepted_release", 0), "benefit_clip": comps.get("benefit_clip", 0.0),
        "battery_core": comps.get("battery_core", 0.0), "answered": int(bool(comps.get("answered"))),
        "forced_no_answer": int(bool(comps.get("forced_no_answer"))), "queries": comps.get("queries", 0),
        "interventions": comps.get("interventions", 0), "turns": n_turns,
        "code_turns": sum(1 for t in turns if t.get("action_type") == "code"),
        "unparsed_turns": sum(1 for e in errs if e == "no action parsed"),
        "json_error_turns": sum(1 for e in errs if e.startswith("json parse")),
        "budget_refused_turns": sum(1 for e in errs if e == "budget spent"),
        "suspicious_code_turns": sum(1 for t in turns if t.get("action_type") == "code"
                                     and _SUSPICIOUS.search(t.get("code") or "")),
        "sandbox_refusals": sum(1 for t in turns if "PermissionError" in (t.get("code_output") or "")),
    }


class RPGv10Env(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        info = extras.get("extra_info", extras) or {}
        assert "world_file" in info, "extra_info must carry world_file"
        root = os.environ.get("RLV10_DATA_ROOT", "/work/rl_v10/data")
        wf = info["world_file"]
        self._world_file = wf if os.path.isabs(wf) else os.path.join(root, wf)
        self._info = info
        self._data_dir = tempfile.mkdtemp(prefix="rlv10_ep_", dir=os.environ.get("RLV10_EPISODE_ROOT") or None)
        self._env = BenchStepEnv(self._world_file, self._data_dir, history="full")
        self.max_turns = int(info.get("max_turns", 32))
        self._first = self._env.reset()
        self._metrics: Dict[str, Any] = {}
        self._t0 = time.time()

    def init(self, prompt):
        stored = [{"role": m.get("role"), "content": m.get("content")} for m in (prompt or [])]
        if stored != self._first:
            raise RuntimeError(f"RPG v10 dataset prompt != run_world's opening for {self._world_file} "
                               "(stale parquet or different harness; rebuild with rl_v10/build_dataset.py)")
        return list(self._first), {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        out = self._env.step(action)
        meta: Dict[str, Any] = {"turn": out.info.get("turns")}
        if out.done:
            comps = out.info["components"]
            self._metrics = _episode_metrics(self._env.result, comps, out.info.get("turns", 0),
                                            self._env.raw_turns)
            meta.update(self._metrics)
            self._dump(comps, out.reward)
            self._cleanup()
        return BaseTextEnvStepOutput(
            observations=[] if out.done else [{"role": "user", "content": out.observation}],
            reward=float(out.reward), done=bool(out.done), metadata=meta)

    def _dump(self, comps, reward):
        d = os.environ.get("RLV10_TRACE_DIR")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            rec = {"world_file": self._info.get("world_file"), "world_id": self._env.world["world_id"],
                   "harness_version": BENCH_HARNESS_VERSION, "reward": reward, "components": comps,
                   "metrics": self._metrics, "wall_s": round(time.time() - self._t0, 1),
                   "raw_outputs": self._env.raw_turns, "result": self._env.result}
            fn = os.path.join(d, f"{int(time.time())}_{self._env.world['world_id']}_{uuid.uuid4().hex[:8]}.json.gz")
            with gzip.open(fn, "wt") as f:
                json.dump(rec, f, default=str)
        except Exception as e:  # never kill a rollout over logging
            print(f"[rlv10] trace dump failed: {e}", file=sys.stderr)

    def _cleanup(self):
        shutil.rmtree(self._data_dir, ignore_errors=True)

    def close(self):
        if not self._env.done:              # abandoned (e.g. the trainer's context limit): unwind run_world
            self._env.close()
            self._metrics = {"abandoned": 1}
        self._cleanup()

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)
