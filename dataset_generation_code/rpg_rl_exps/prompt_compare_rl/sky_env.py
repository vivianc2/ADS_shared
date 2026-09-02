#!/usr/bin/env python3
"""SkyRL-Gym environment for the prompt comparison.

This is a THIN subclass of the pipeline's existing ``skyrl_rpg.env.RPGSkyEnv`` (the
class the RPG SkyRL runs already use). It adds exactly two things:

1. **A prompt-arrival assertion.** ``init()`` verifies that the system message SkyRL is
   about to send really is the prompt this run selected (digest match against
   ``PC_PROMPT_SHA256``). The system prompt lives in the dataset parquet, so a stale
   parquet or a mixed-up ``data.train_data`` would otherwise train the wrong prompt
   silently; here it fails on the first episode instead. Requirement (2).

2. **The evaluation metrics.** ``get_metrics()`` reports the five numbers the experiment
   is graded on -- score, part_a, part_b, truncation, turns -- plus a small amount of
   diagnostic context. SkyRL averages these across episodes into
   ``eval/all/environment/<key>``.

Nothing about the world, the action grammar, or the reward is re-implemented: those come
from the verified ``rpg_rl`` / ``rpg_v9`` modules through the base class.

Why the module is named ``sky_env`` and not ``env``: ``rpg_rl/env.py`` is imported as the
top-level module ``env`` by the RPG stack, and a second top-level ``env`` on the path
would be an import-shadowing hazard inside Ray workers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
from typing import Any, Dict, List

_RPG_SRC = os.environ.get("RPG_SRC", "/work/ADS_shared/dataset_generation_code")
_PROTO = os.environ.get("RPG_PROTO", "rpg_v9")

# Same bootstrap the shipped adapter uses: make the verified science + RL-env code
# importable in any context, including Ray workers running from a copied working dir.
for _p in (os.path.join(_RPG_SRC, "rpg_rl"), os.path.join(_RPG_SRC, _PROTO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_base_module():
    """Import ``skyrl_rpg/env.py`` from disk under a private module name.

    Loading by file path (rather than putting ``dataset_generation_code`` on sys.path)
    keeps the many loose modules in that directory out of the import namespace.
    """
    path = os.path.join(_RPG_SRC, "skyrl_rpg", "env.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"expected the shipped SkyRL RPG adapter at {path}; set RPG_SRC to the "
            "dataset_generation_code directory"
        )
    name = "prompt_compare_rl._skyrl_rpg_env"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_module()
RPGSkyEnv = _base.RPGSkyEnv

from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput  # noqa: E402

_prompt_check_lock = threading.Lock()
_prompt_check_state: Dict[str, Any] = {"verified": False}


def _verify_system_prompt(prompt: List[Dict[str, str]]) -> None:
    """Fail loudly unless the dataset's system message is the run's selected prompt."""
    expected = os.environ.get("PC_PROMPT_SHA256")
    if not expected:
        return  # not running under the prompt-comparison launcher; stay a plain RPG env
    if not prompt or prompt[0].get("role") != "system":
        raise RuntimeError(
            "prompt-compare run expected the dataset prompt to start with a system "
            f"message, got roles={[m.get('role') for m in (prompt or [])]}"
        )
    actual = hashlib.sha256(prompt[0]["content"].encode("utf-8")).hexdigest()
    if actual != expected:
        raise RuntimeError(
            "SYSTEM PROMPT MISMATCH: this run is configured for "
            f"PC_PROMPT_ID={os.environ.get('PC_PROMPT_ID')} (sha256={expected}) but the "
            f"dataset row carries sha256={actual}. The parquet was built for a different "
            "prompt -- re-run build_dataset.py and check data.train_data/data.val_data."
        )
    with _prompt_check_lock:
        if not _prompt_check_state["verified"]:
            _prompt_check_state["verified"] = True
            print(
                f"[prompt_compare_rl] verified system prompt {os.environ.get('PC_PROMPT_ID')} "
                f"sha256={actual} reached the model",
                flush=True,
            )


class PromptCompareRPGEnv(RPGSkyEnv):
    """One RPG episode, with prompt verification and comparison metrics."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__(env_config=env_config, extras=extras)
        info = extras.get("extra_info", extras) or {}
        self._archetype = str(info.get("archetype", "unknown"))
        self._terminal: Dict[str, Any] = {}

    # -- prompt arrival ----------------------------------------------------------------

    def init(self, prompt):
        _verify_system_prompt(prompt)
        return super().init(prompt)

    # -- terminal capture --------------------------------------------------------------

    def step(self, action: str) -> BaseTextEnvStepOutput:
        out = super().step(action)
        if out["done"]:
            # The base class already surfaces the graded parts in `metadata`; the
            # remaining fields come off the env's own terminal turn record.
            meta = out.get("metadata") or {}
            last = self._rpg.turns[-1] if self._rpg.turns else {}
            breakdown = last.get("reward_breakdown", {}) or {}
            self._terminal = {
                "score": float(out["reward"]),
                "part_a": float(meta.get("part_a") or 0.0),
                "part_b": float(meta.get("part_b") or 0.0),
                "accepted": float(bool(meta.get("accepted"))),
                "reward_error": float(bool(meta.get("reward_error"))),
                "invalid_id_fraction": float(breakdown.get("invalid_id_fraction", 0.0)),
                "forced_no_answer": float(bool(last.get("forced_no_answer", False))),
            }
        return out

    # -- metrics -----------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Per-episode metrics; SkyRL averages them into ``environment/<key>``.

        ``get_metrics`` is also called when the generator abandons a trajectory (input
        length exceeded, generation truncated), in which case the episode never reached
        a terminal reward. Those episodes count as ``truncated=1`` with score/part_a/
        part_b of 0.0 -- the same 0.0 SkyRL already attributes to them in ``avg_score``,
        so the reported means stay consistent with the reward the policy actually got.
        """
        rpg = self._rpg
        completed = bool(rpg._done)
        metrics: Dict[str, Any] = {
            "score": 0.0,
            "part_a": 0.0,
            "part_b": 0.0,
            "accepted": 0.0,
            "reward_error": 0.0,
            "invalid_id_fraction": 0.0,
            "forced_no_answer": 0.0,
            "truncated": 0.0 if completed else 1.0,
            "completed": 1.0 if completed else 0.0,
            "turns": float(rpg._turn),
            "interventions": float(rpg._n_interv),
            "queries_used": float(rpg._used),
            "parse_failures": float(sum(1 for t in rpg.turns if t.get("action_type") is None)),
        }
        metrics.update(self._terminal)
        # Non-numeric, used only by the custom aggregator below.
        metrics["archetype"] = self._archetype
        return metrics

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean over episodes, plus a per-archetype breakdown of the headline numbers.

        The validation set holds 5 worlds per archetype, so the breakdown is what makes
        an 8-step comparison interpretable; the flat keys stay exactly where SkyRL's
        default aggregator would have put them.
        """
        from skyrl_gym.metrics import default_aggregate_metrics

        aggregated = default_aggregate_metrics(metrics)
        by_archetype: Dict[str, List[Dict[str, Any]]] = {}
        for m in metrics:
            by_archetype.setdefault(str(m.get("archetype", "unknown")), []).append(m)
        for archetype, subset in sorted(by_archetype.items()):
            for key in ("score", "part_a", "part_b", "truncated", "turns"):
                values = [float(m[key]) for m in subset if isinstance(m.get(key), (int, float))]
                if values:
                    aggregated[f"archetype/{archetype}/{key}"] = sum(values) / len(values)
            aggregated[f"archetype/{archetype}/episodes"] = float(len(subset))
        return aggregated
