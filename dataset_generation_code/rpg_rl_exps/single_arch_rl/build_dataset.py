#!/usr/bin/env python3
"""Build the two single-archetype training sets and the shared validation set.

Requirement (2): unlike the earlier RPG runs, whose training set is a uniform mixture of
the six train archetypes, each run here sees ONE archetype:

    easy -> 96 `dose_window` worlds
    hard -> 96 `confounded_reversal` worlds

Both are drawn from the committed de-leaked v9 train split
(``rpg_v9/data_v9_deleaked/train.parquet``, 1536 audited worlds on the 8 train skins), so
the two sets differ in archetype and in nothing else about how they were produced. Both
runs evaluate on the same ``rpg_v9/data_v9_deleaked/validation_small.parquet`` (45
held-out worlds, 9 archetypes x 5), which is copied here unchanged except for the two
edits described below.

Selection (deterministic, no RNG)
---------------------------------
The source pool has 292 `dose_window` and 242 `confounded_reversal` worlds, unevenly
spread over the 8 train skins. Taking the first 96 of each would let the two runs differ
in their skin mix as well as their archetype. Instead we take **12 worlds per skin x 8
skins**, in ascending seed order, preferring worlds that still pass ``audit()`` under the
current generator. The skin histogram is then identical for `easy` and `hard` by
construction. If some skin cannot supply 12 accepted worlds the shortfall is filled from
the remaining pool in ascending seed order, and the build report records it.

Observation refresh (``--regen-obs``, ON by default)
----------------------------------------------------
``prompt[1].content`` is the world's first observation, including the id catalog the
policy must act through. It was rendered when the parquet was written (2026-08-18) and the
v9 generator has changed since; ``RPGSkyEnv`` REBUILDS each world from ``extra_info`` at
episode time. Measured on the current sources, **45% of dose_window rows, 53% of
confounded_reversal rows and 60% of the validation rows** now render a different catalog
than the one stored in the prompt -- the policy would be shown one id->signal mapping and
graded against another. Every row is therefore re-rendered with the current code
(``RPGEnv.reset()``, exactly what the env will produce), restoring the
``dataset prompt == env.reset()`` invariant. Pass ``--no-regen-obs`` to keep the stored
text and reproduce the mismatch instead.

Validation ``data_source``
--------------------------
The 45 validation rows keep their worlds, order, prompt and ``extra_info``; only
``data_source`` is relabelled from ``rpg_v9`` to ``rpg_v9_eval_<archetype>``. SkyRL groups
its evaluation metrics by ``data_source``, so this is what produces
``eval/rpg_v9_eval_<archetype>/avg_score`` per archetype and one eval dump per archetype
(log requirement 2) alongside the ``eval/all/environment/archetype/...`` breakdown that
``sky_env.py`` contributes.

Run (inside the container):
    bash scripts/in_container.sh python -m single_arch_rl.build_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from single_arch_rl.config import (  # noqa: E402
    ENV_ID,
    RPG_PROTO,
    RUN_IDS,
    TRAIN_ARCHETYPE,
    TRAIN_WORLDS_PER_RUN,
    WORLDS_PER_SKIN,
    ExperimentConfig,
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
    scratch = tempfile.mkdtemp(prefix="sa_build_")
    env = RPGEnv(
        world=world,
        gold=result["gold"],
        battery=result["battery"],
        catalog_seed=seed,
        max_turns=int(row_extra.get("max_turns", 32)),
        budget=int(row_extra.get("budget", 15)),
        data_dir=scratch,
    )
    return env.reset(), bool(result.get("ok"))


def _worker(args: Tuple[Dict[str, Any], str]) -> Tuple[str, bool]:
    return _render_first_observation(*args)


def _render_all(extras: List[Dict[str, Any]], rpg_src: str, jobs: int) -> List[Tuple[str, bool]]:
    payload = [(e, rpg_src) for e in extras]
    if jobs > 1 and len(payload) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(_worker, payload, chunksize=4))
    return [_worker(p) for p in payload]


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def select_worlds(
    candidates: List[Dict[str, Any]],
    *,
    n_total: int = TRAIN_WORLDS_PER_RUN,
    per_skin: int = WORLDS_PER_SKIN,
) -> Tuple[List[int], Dict[str, Any]]:
    """Pick ``n_total`` rows, ``per_skin`` from each skin, deterministically.

    ``candidates`` is a list of ``{"index", "skin", "seed", "audit_ok"}`` dicts. Within a
    skin, worlds that still pass ``audit()`` come first, then ascending seed -- so the
    choice is a pure function of the source parquet and this code. Returns the chosen
    source indices (in the candidates' original order) plus a report.
    """
    by_skin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_skin[row["skin"]].append(row)
    order = lambda r: (not r["audit_ok"], int(r["seed"]))  # noqa: E731 - accepted first, then seed

    chosen: List[Dict[str, Any]] = []
    shortfalls: Dict[str, int] = {}
    for skin in sorted(by_skin):
        pool = sorted(by_skin[skin], key=order)
        take = pool[:per_skin]
        if len(take) < per_skin:
            shortfalls[skin] = per_skin - len(take)
        chosen.extend(take)

    # Fill any shortfall from what is left, still deterministically.
    if len(chosen) < n_total:
        taken = {r["index"] for r in chosen}
        leftovers = sorted((r for r in candidates if r["index"] not in taken), key=order)
        chosen.extend(leftovers[: n_total - len(chosen)])
    chosen = chosen[:n_total]

    indices = sorted(r["index"] for r in chosen)
    report = {
        "selected": len(indices),
        "per_skin_target": per_skin,
        "skins": dict(sorted(Counter(r["skin"] for r in chosen).items())),
        "audit_ok": sum(1 for r in chosen if r["audit_ok"]),
        "skin_shortfalls": shortfalls,
    }
    return indices, report


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def _record(row, user_content: str, data_source: Optional[str] = None) -> Dict[str, Any]:
    """Rebuild a SkyRL row, preserving every field except the ones we deliberately set."""
    return {
        "data_source": data_source if data_source is not None else row["data_source"],
        "prompt": [
            # The system prompt is copied verbatim: both runs use the pipeline's default
            # RPG system prompt, which tests/test_dataset.py pins to rpg_rl/env.py.
            {"role": "system", "content": row["prompt"][0]["content"]},
            {"role": "user", "content": user_content},
        ],
        # NOT copied from the source, which says "rpg". SkyRL builds each episode's env
        # from the ROW's env_class, not from environment.env_class, so a copied value
        # sends every episode to the shipped `rpg` env -- which this process never
        # registers -- and the run dies at the first evaluation with
        # "No registered env with id: rpg".
        "env_class": ENV_ID,
        "reward_spec": {
            "method": row["reward_spec"]["method"],
            "ground_truth": row["reward_spec"]["ground_truth"],
        },
        "extra_info": {k: v for k, v in dict(row["extra_info"]).items()},
    }


def _write(records: List[Dict[str, Any]], destination: Path) -> int:
    from datasets import Dataset

    destination.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(destination))
    return len(records)


def _load(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"source dataset not found: {path}")
    return pd.read_parquet(path)


def build(cfg: ExperimentConfig, *, regen_obs: bool, jobs: int, run_ids: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "exp_tag": cfg.exp_tag,
        "rpg_proto": RPG_PROTO,
        "regen_obs": regen_obs,
        "source_train": str(cfg.source_train),
        "source_val": str(cfg.source_val),
        "train_worlds_per_run": TRAIN_WORLDS_PER_RUN,
        "runs": {},
    }

    # ---- training sets, one archetype each --------------------------------------------
    train_df = _load(cfg.source_train)
    train_extras = [dict(r) for r in train_df["extra_info"]]
    for run_id in run_ids:
        archetype = TRAIN_ARCHETYPE[run_id]
        pool_idx = [i for i, e in enumerate(train_extras) if e["archetype"] == archetype]
        if not pool_idx:
            raise SystemExit(f"no {archetype!r} worlds in {cfg.source_train}")
        started = time.time()
        rendered = _render_all([train_extras[i] for i in pool_idx], cfg.rpg_src, jobs)
        candidates = [
            {"index": i, "skin": train_extras[i]["skin"], "seed": train_extras[i]["seed"],
             "audit_ok": ok}
            for i, (_obs, ok) in zip(pool_idx, rendered)
        ]
        indices, selection = select_worlds(candidates)
        obs_by_index = {i: obs for i, (obs, _ok) in zip(pool_idx, rendered)}

        records, changed = [], 0
        for i in indices:
            row = train_df.iloc[i]
            content = obs_by_index[i] if regen_obs else row["prompt"][1]["content"]
            changed += content != row["prompt"][1]["content"]
            records.append(_record(row, content))
        destination = cfg.train_parquet(run_id)
        written = _write(records, destination)

        report["runs"][run_id] = {
            "archetype": archetype,
            "pool_size": len(pool_idx),
            "pool_audit_ok": sum(1 for _o, ok in rendered if ok),
            "selection": selection,
            "rows_written": written,
            "observations_changed_vs_source": changed,
            "path": str(destination),
            "seeds": [int(train_extras[i]["seed"]) for i in indices],
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"[{run_id}] {archetype}: pool={len(pool_idx)} -> selected {written} "
            f"(skins {selection['skins']}, audit_ok {selection['audit_ok']}/{written}, "
            f"{changed} observations refreshed)",
            flush=True,
        )

    # ---- shared validation set ----------------------------------------------------------
    val_df = _load(cfg.source_val)
    val_extras = [dict(r) for r in val_df["extra_info"]]
    started = time.time()
    rendered = _render_all(val_extras, cfg.rpg_src, jobs) if regen_obs else None
    records, changed, audit_ok = [], 0, 0
    for i in range(len(val_df)):
        row = val_df.iloc[i]
        if rendered is not None:
            content, ok = rendered[i]
            audit_ok += ok
        else:
            content, ok = row["prompt"][1]["content"], True
            audit_ok += 1
        changed += content != row["prompt"][1]["content"]
        records.append(_record(row, content, data_source=f"rpg_v9_eval_{val_extras[i]['archetype']}"))
    written = _write(records, cfg.val_parquet)
    report["validation"] = {
        "rows_written": written,
        "audit_ok": audit_ok,
        "observations_changed_vs_source": changed,
        "archetypes": dict(sorted(Counter(e["archetype"] for e in val_extras).items())),
        "path": str(cfg.val_parquet),
        "seconds": round(time.time() - started, 1),
    }
    print(
        f"[validation] {written} worlds, {report['validation']['archetypes']}, "
        f"audit_ok {audit_ok}/{written}, {changed} observations refreshed",
        flush=True,
    )

    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    report_path = cfg.dataset_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)),
                        help="processes used to re-render observations")
    parser.add_argument("--runs", default=",".join(RUN_IDS))
    regen = parser.add_mutually_exclusive_group()
    regen.add_argument("--regen-obs", dest="regen_obs", action="store_true", default=True,
                       help="re-render the first observation with the current v9 code (default)")
    regen.add_argument("--no-regen-obs", dest="regen_obs", action="store_false",
                       help="keep the stored observation text verbatim")
    args = parser.parse_args(argv)

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    unknown = [r for r in run_ids if r not in RUN_IDS]
    if unknown:
        raise SystemExit(f"unknown run ids: {unknown}")

    report = build(ExperimentConfig(), regen_obs=args.regen_obs, jobs=max(1, args.jobs),
                   run_ids=run_ids)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    print(f"build report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
