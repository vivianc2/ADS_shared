#!/usr/bin/env python3
"""Build the three per-prompt SkyRL datasets from the v9 de-leaked parquet files.

Requirement (2): the system prompt is baked into every parquet row, so selecting a
prompt means REWRITING the dataset, not just changing ``env.py``. This script takes the
required sources --

    rpg_v9/data_v9_deleaked/train.parquet             (training set)
    rpg_v9/data_v9_deleaked/validation_small.parquet  (validation set)

-- and writes ``<out>/datasets/<pid>/{train,validation_small}.parquet`` for pid in
p1/p2/p3. The three copies share every row, in the same order, with the same
``extra_info`` (seed / skin / archetype), so all three runs see the SAME training worlds
in the SAME order and the SAME validation worlds. Only ``prompt[0].content`` differs.

Observation refresh (``--regen-obs``, ON by default)
---------------------------------------------------
``prompt[1].content`` is the world's first observation, including the id catalog the
policy must act through. It was rendered when the parquet was created (2026-08-18). The
v9 world generator has changed since (``sampler.py`` / ``skins.py`` / ``engine.py`` were
edited 2026-08-31), and ``RPGSkyEnv`` REBUILDS each world from ``extra_info`` at episode
time -- so for ~57-60% of rows the catalog in the stored prompt no longer matches the
world the policy actually interacts with, which would make the id-space task incoherent
and the reward meaningless.

This script therefore re-renders each row's first observation with the CURRENT code
(``RPGEnv.reset()``, exactly what the env will produce), which restores the invariant
``dataset prompt == env.reset()`` that the adapter documents. Pass ``--no-regen-obs`` to
keep the stored text verbatim and reproduce the mismatch instead; the build report always
records how many rows changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from prompt_compare_rl import prompts as prompt_lib  # noqa: E402
from prompt_compare_rl.config import (  # noqa: E402
    RPG_PROTO,
    ExperimentConfig,
    PROMPT_IDS,
)

_WORKER_STATE: Dict[str, Any] = {}


def _bootstrap_rpg_imports(rpg_src: str) -> None:
    """Put the verified RPG modules on sys.path with the pinned protocol."""
    os.environ.setdefault("RPG_PROTO", RPG_PROTO)
    for path in (os.path.join(rpg_src, "rpg_rl"), os.path.join(rpg_src, RPG_PROTO)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _render_first_observation(row_extra: Dict[str, Any], rpg_src: str) -> Tuple[str, bool]:
    """Rebuild one world with the current code and return (observation, audit_ok)."""
    if "env" not in _WORKER_STATE:
        _bootstrap_rpg_imports(rpg_src)
        from env import RPGEnv  # noqa: WPS433 - deliberate late import
        from generate_v7 import audit
        from sampler import sample_world

        _WORKER_STATE.update({"env": RPGEnv, "audit": audit, "sample_world": sample_world})
    RPGEnv = _WORKER_STATE["env"]
    audit = _WORKER_STATE["audit"]
    sample_world = _WORKER_STATE["sample_world"]

    seed = int(row_extra["seed"])
    world = sample_world(seed, skin=row_extra["skin"], archetype=row_extra["archetype"])
    world["ground_truth"]["_seed"] = seed
    result = audit(world)
    scratch = tempfile.mkdtemp(prefix="pc_build_")
    env = RPGEnv(
        world=world,
        gold=result["gold"],
        battery=result["battery"],
        catalog_seed=seed,
        max_turns=int(row_extra.get("max_turns", 32)),
        budget=int(row_extra.get("budget", 15)),
        data_dir=scratch,
    )
    observation = env.reset()
    return observation, bool(result.get("ok"))


def _worker(args: Tuple[Dict[str, Any], str]) -> Tuple[str, bool]:
    return _render_first_observation(*args)


def _load_source(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"source dataset not found: {path}")
    return pd.read_parquet(path)


def _row_to_record(row, system_prompt: str, user_content: str) -> Dict[str, Any]:
    """Rebuild a SkyRL row, preserving every field except the two message contents."""
    return {
        "data_source": row["data_source"],
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "env_class": row["env_class"],
        "reward_spec": {
            "method": row["reward_spec"]["method"],
            "ground_truth": row["reward_spec"]["ground_truth"],
        },
        "extra_info": {k: v for k, v in dict(row["extra_info"]).items()},
    }


def _observations_for(df, rpg_src: str, jobs: int) -> Tuple[List[str], Dict[str, int]]:
    """Re-render every row's first observation; return the texts and a change report."""
    extras = [dict(r) for r in df["extra_info"]]
    payload = [(e, rpg_src) for e in extras]
    started = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_worker, payload, chunksize=4))
    else:
        results = [_worker(p) for p in payload]
    observations = [obs for obs, _ok in results]
    stats = {
        "rows": len(observations),
        "audit_ok": sum(1 for _obs, ok in results if ok),
        "changed_vs_source": sum(
            1 for obs, stored in zip(observations, df["prompt"]) if obs != stored[1]["content"]
        ),
        "seconds": round(time.time() - started, 1),
    }
    return observations, stats


def _write_split(
    df,
    observations: Optional[List[str]],
    system_prompt: str,
    destination: Path,
) -> int:
    from datasets import Dataset

    records = []
    for index in range(len(df)):
        row = df.iloc[index]
        user_content = observations[index] if observations is not None else row["prompt"][1]["content"]
        records.append(_row_to_record(row, system_prompt, user_content))
    destination.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(destination))
    return len(records)


def build(cfg: ExperimentConfig, regen_obs: bool, jobs: int, prompt_ids: List[str]) -> Dict[str, Any]:
    prompt_paths = prompt_lib.materialize(cfg.prompt_dir)
    prompt_text = prompt_lib.read_materialized(cfg.prompt_dir)

    report: Dict[str, Any] = {
        "exp_tag": cfg.exp_tag,
        "rpg_proto": RPG_PROTO,
        "regen_obs": regen_obs,
        "prompt_files": {pid: str(path) for pid, path in prompt_paths.items()},
        "prompt_sha256": {pid: prompt_lib.sha256_text(prompt_text[pid]) for pid in prompt_ids},
        "splits": {},
    }

    for split_name, source in (("train", cfg.source_train), ("validation_small", cfg.source_val)):
        df = _load_source(source)
        observations = None
        stats: Dict[str, Any] = {"rows": len(df), "source": str(source)}
        if regen_obs:
            observations, regen_stats = _observations_for(df, cfg.rpg_src, jobs)
            stats.update(regen_stats)
        stored_sha = {
            prompt_lib.sha256_text(row[0]["content"]) for row in df["prompt"]
        }
        stats["source_system_prompt_sha256"] = sorted(stored_sha)
        written = {}
        for pid in prompt_ids:
            destination = cfg.dataset_dir(pid) / f"{split_name}.parquet"
            count = _write_split(df, observations, prompt_text[pid], destination)
            written[pid] = {"path": str(destination), "rows": count}
        stats["written"] = written
        report["splits"][split_name] = stats

    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    report_path = cfg.dataset_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                        help="processes used to re-render observations")
    parser.add_argument("--prompt-ids", default=",".join(PROMPT_IDS))
    regen = parser.add_mutually_exclusive_group()
    regen.add_argument("--regen-obs", dest="regen_obs", action="store_true", default=True,
                       help="re-render the first observation with the current v9 code (default)")
    regen.add_argument("--no-regen-obs", dest="regen_obs", action="store_false",
                       help="keep the stored observation text verbatim")
    args = parser.parse_args(argv)

    prompt_ids = [p.strip() for p in args.prompt_ids.split(",") if p.strip()]
    unknown = [p for p in prompt_ids if p not in PROMPT_IDS]
    if unknown:
        raise SystemExit(f"unknown prompt ids: {unknown}")

    cfg = ExperimentConfig()
    report = build(cfg, regen_obs=args.regen_obs, jobs=max(1, args.jobs), prompt_ids=prompt_ids)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
