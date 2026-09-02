#!/usr/bin/env python3
"""The three system-prompt candidates for the RL comparison.

The prompt TEXT is not redefined here. ``p2`` and ``p3`` are imported from the existing
inference-side experiment (``rpg_rl_exps/prompt_compare/candidates.py``) and ``p1`` is
``rpg_rl/env.py``'s prompt, so the RL comparison and the inference comparison are
provably testing the same three strings (``tests/test_prompts.py`` asserts the digests).

Why the import is done in a SUBPROCESS
--------------------------------------
``prompt_compare.candidates`` executes ``bootstrap.configure_imports()`` at import time,
which mutates ``os.environ`` (``RPG_PROTO``, ``RPG_SYNERGY_SOFT``) and rewrites
``sys.path``. That is correct for that experiment but it is a side effect we do not want
inside a training worker or a dataset builder. So the text is pulled once, in a
throwaway subprocess, and cached as plain files under the experiment's prompt directory.
Everything downstream (dataset build, ``RPG_SYSTEM_PROMPT_FILE``, the runtime assertion)
reads those files, so exactly one byte-string per prompt id exists in the experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROMPT_IDS: List[str] = ["p1", "p2", "p3"]

# ``prompt_compare`` lives next to this package.
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_PROMPT_COMPARE_DIR = _EXPERIMENTS_DIR / "prompt_compare"

# Digests of the three prompts as they exist today. They are asserted on every load so a
# silent edit to `env.py` or `candidates.py` cannot quietly change a running experiment.
# p1 is also the prompt baked into rpg_v9/data_v9_deleaked/*.parquet.
EXPECTED_SHA256: Dict[str, str] = {
    "p1": "27ad9561715a46f1d3cf56d1e8b5ccb1d66cc43769798dbab2659d80aecd2a2e",
    "p2": "bd3fb7d0da614f9bbf048e6d4a5d73c2cd48d6ed575a9a92496e508f78e3f745",
    "p3": "fd140d7779cc9b1d3397059c771b87b017e664c9811989851ec714d79f47ef83",
}

_EXTRACT_SNIPPET = r"""
import json, sys
sys.path.insert(0, {compare_dir!r})
from candidates import PROMPTS
sys.stdout.write("<<<PROMPTS>>>" + json.dumps({{k: PROMPTS[k] for k in PROMPTS}}))
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompts_from_source() -> Dict[str, str]:
    """Import ``prompt_compare.candidates`` in a subprocess and return {id: text}.

    Runs with a clean ``RPG_SYSTEM_PROMPT_FILE`` so that p1 is the *module default* of
    ``rpg_rl/env.py`` and never an override we ourselves installed earlier.
    """
    env = dict(os.environ)
    env.pop("RPG_SYSTEM_PROMPT_FILE", None)
    env.pop("RPG_SYSTEM_PROMPT_SHA256", None)
    code = _EXTRACT_SNIPPET.format(compare_dir=str(_PROMPT_COMPARE_DIR))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_PROMPT_COMPARE_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to import prompt_compare.candidates:\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-4000:]}"
        )
    marker = "<<<PROMPTS>>>"
    if marker not in proc.stdout:
        raise RuntimeError(f"prompt extraction produced no payload:\n{proc.stdout[-2000:]}")
    prompts = json.loads(proc.stdout.split(marker, 1)[1])
    _validate(prompts)
    return prompts


def _validate(prompts: Dict[str, str]) -> None:
    missing = [pid for pid in PROMPT_IDS if pid not in prompts]
    if missing:
        raise ValueError(f"prompt source is missing ids: {missing}")
    for pid in PROMPT_IDS:
        actual = sha256_text(prompts[pid])
        expected = EXPECTED_SHA256[pid]
        if actual != expected:
            raise ValueError(
                f"prompt {pid} changed: sha256={actual} but this experiment pins {expected}. "
                "If the change is intended, update EXPECTED_SHA256 and rebuild the datasets "
                "(a running comparison must never silently switch prompts)."
            )
    if len({prompts[pid] for pid in PROMPT_IDS}) != len(PROMPT_IDS):
        raise ValueError("the three prompts are not pairwise distinct")


def materialize(prompt_dir: os.PathLike | str) -> Dict[str, Path]:
    """Write ``<prompt_dir>/<pid>.txt`` plus a manifest; return {id: path}.

    Idempotent: rewriting with identical content is a no-op for downstream consumers,
    and the digest check makes a stale file impossible to use by accident.
    """
    prompt_dir = Path(prompt_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts_from_source()
    paths: Dict[str, Path] = {}
    for pid in PROMPT_IDS:
        path = prompt_dir / f"{pid}.txt"
        path.write_text(prompts[pid], encoding="utf-8")
        paths[pid] = path
    manifest = {
        "prompt_ids": PROMPT_IDS,
        "sha256": {pid: sha256_text(prompts[pid]) for pid in PROMPT_IDS},
        "chars": {pid: len(prompts[pid]) for pid in PROMPT_IDS},
        "source": str(_PROMPT_COMPARE_DIR / "candidates.py"),
        "files": {pid: str(paths[pid]) for pid in PROMPT_IDS},
    }
    (prompt_dir / "prompts_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths


def read_materialized(prompt_dir: os.PathLike | str) -> Dict[str, str]:
    """Read back the materialized prompt files, verifying the pinned digests."""
    prompt_dir = Path(prompt_dir)
    prompts = {}
    for pid in PROMPT_IDS:
        path = prompt_dir / f"{pid}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing prompt file {path}; run build_dataset.py first")
        prompts[pid] = path.read_text(encoding="utf-8")
    _validate(prompts)
    return prompts


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    for _pid, _text in load_prompts_from_source().items():
        print(f"{_pid} {sha256_text(_text)} {len(_text)} chars")
