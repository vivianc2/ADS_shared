"""The built datasets are what the experiment claims: 96 worlds of ONE archetype each.

The heavy checks (which need pandas + the v9 generator) skip automatically when the
parquet files have not been built yet, so the file is still useful before a build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from single_arch_rl.build_dataset import select_worlds
from single_arch_rl.config import (
    RUN_IDS,
    TRAIN_ARCHETYPE,
    TRAIN_WORLDS_PER_RUN,
    WORLDS_PER_SKIN,
    ExperimentConfig,
)

pd = pytest.importorskip("pandas")


# ---- pure selection logic (no data needed) --------------------------------------------


def _candidates(skins, per_skin, audit_ok=True):
    rows, index = [], 0
    for skin in skins:
        for k in range(per_skin):
            rows.append({"index": index, "skin": skin, "seed": 1000 + index,
                         "audit_ok": audit_ok or k > 0})
            index += 1
    return rows


def test_selection_is_balanced_over_skins():
    chosen, report = select_worlds(_candidates([f"s{i}" for i in range(8)], 30))
    assert len(chosen) == TRAIN_WORLDS_PER_RUN
    assert set(report["skins"].values()) == {WORLDS_PER_SKIN}
    assert report["skin_shortfalls"] == {}


def test_selection_is_deterministic():
    pool = _candidates([f"s{i}" for i in range(8)], 30)
    assert select_worlds(pool)[0] == select_worlds(list(reversed(pool)))[0]


def test_selection_prefers_worlds_that_still_pass_audit():
    pool = _candidates(["s0"], 20)
    pool[0]["audit_ok"] = False            # lowest seed, but rejected by the current audit
    chosen, _ = select_worlds(pool, n_total=4, per_skin=4)
    assert pool[0]["index"] not in chosen


def test_selection_fills_a_short_skin_from_the_rest():
    pool = _candidates([f"s{i}" for i in range(7)], 30) + _candidates(["s7"], 3)
    # renumber the tail so indices stay unique
    for offset, row in enumerate(pool[-3:]):
        row["index"] = 10_000 + offset
    chosen, report = select_worlds(pool)
    assert len(chosen) == TRAIN_WORLDS_PER_RUN
    assert report["skin_shortfalls"] == {"s7": WORLDS_PER_SKIN - 3}


# ---- the built artifacts ----------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return ExperimentConfig()


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run python -m single_arch_rl.build_dataset")
    return pd.read_parquet(path)


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_training_set_is_96_worlds_of_one_archetype(cfg, run_id):
    frame = _require(cfg.train_parquet(run_id))
    extras = [dict(r) for r in frame["extra_info"]]
    assert len(frame) == TRAIN_WORLDS_PER_RUN
    assert {e["archetype"] for e in extras} == {TRAIN_ARCHETYPE[run_id]}
    assert {e["split"] for e in extras} == {"train"}
    assert len({e["seed"] for e in extras}) == TRAIN_WORLDS_PER_RUN


def test_the_two_training_sets_share_no_world(cfg):
    frames = {r: _require(cfg.train_parquet(r)) for r in RUN_IDS}
    seeds = {r: {dict(e)["seed"] for e in f["extra_info"]} for r, f in frames.items()}
    assert seeds["easy"].isdisjoint(seeds["hard"])


def test_the_two_training_sets_have_the_same_skin_histogram(cfg):
    """Archetype is the only thing that differs; the skin mix is held equal."""
    histograms = {}
    for run_id in RUN_IDS:
        extras = [dict(r) for r in _require(cfg.train_parquet(run_id))["extra_info"]]
        histograms[run_id] = {s: sum(1 for e in extras if e["skin"] == s)
                              for s in {e["skin"] for e in extras}}
    assert histograms["easy"] == histograms["hard"]
    assert set(histograms["easy"].values()) == {WORLDS_PER_SKIN}


def test_validation_set_is_the_shared_45_world_heldout_set(cfg):
    frame = _require(cfg.val_parquet)
    extras = [dict(r) for r in frame["extra_info"]]
    assert len(frame) == 45
    assert {e["split"] for e in extras} == {"heldout"}
    counts = {a: sum(1 for e in extras if e["archetype"] == a) for a in {e["archetype"] for e in extras}}
    assert len(counts) == 9 and set(counts.values()) == {5}


def test_validation_rows_and_worlds_match_the_committed_source(cfg):
    """Only data_source (and the refreshed observation) may differ from the source."""
    source = _require(cfg.source_val)
    built = _require(cfg.val_parquet)
    assert len(source) == len(built)
    for i in range(len(source)):
        assert dict(source["extra_info"].iloc[i]) == dict(built["extra_info"].iloc[i])
        assert source["prompt"].iloc[i][0]["content"] == built["prompt"].iloc[i][0]["content"]


def test_every_row_names_this_experiments_env(cfg):
    """SkyRL builds each episode's env from the ROW's env_class, not the config's.

    Copying the source's "rpg" would send every episode to the shipped env, which this
    process never registers -- the run dies at the first evaluation.
    """
    from single_arch_rl.config import ENV_ID

    for path in [cfg.train_parquet(r) for r in RUN_IDS] + [cfg.val_parquet]:
        assert set(_require(path)["env_class"]) == {ENV_ID}, path


def test_validation_data_source_is_per_archetype(cfg):
    """This is what produces eval/rpg_v9_eval_<archetype>/avg_score in W&B."""
    frame = _require(cfg.val_parquet)
    for i in range(len(frame)):
        archetype = dict(frame["extra_info"].iloc[i])["archetype"]
        assert frame["data_source"].iloc[i] == f"rpg_v9_eval_{archetype}"


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_system_prompt_is_the_pipeline_default(cfg, run_id):
    """Both runs use the stock RPG system prompt; nothing here rewrites it."""
    import hashlib

    frame = _require(cfg.train_parquet(run_id))
    digests = {hashlib.sha256(p[0]["content"].encode("utf-8")).hexdigest() for p in frame["prompt"]}
    assert digests == {"27ad9561715a46f1d3cf56d1e8b5ccb1d66cc43769798dbab2659d80aecd2a2e"}


def test_stored_observation_matches_what_the_live_environment_renders(cfg):
    """The invariant the adapter documents: dataset prompt == RPGEnv.reset().

    RPGSkyEnv rebuilds each world from extra_info at episode time, so a stale observation
    would show the policy one id catalog and grade it against another.
    """
    import os
    import sys
    import tempfile

    for path in (os.path.join(cfg.rpg_src, "rpg_rl"), os.path.join(cfg.rpg_src, "rpg_v9")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from env import RPGEnv
    from generate_v7 import audit
    from sampler import sample_world

    checked = 0
    for path in [cfg.train_parquet(r) for r in RUN_IDS] + [cfg.val_parquet]:
        frame = _require(path)
        for i in range(0, len(frame), 16):          # a sample; rebuilding all is minutes
            extra = dict(frame["extra_info"].iloc[i])
            world = sample_world(int(extra["seed"]), skin=extra["skin"], archetype=extra["archetype"])
            world["ground_truth"]["_seed"] = int(extra["seed"])
            result = audit(world)
            env = RPGEnv(world=world, gold=result["gold"], battery=result["battery"],
                         catalog_seed=int(extra["seed"]), max_turns=int(extra.get("max_turns", 32)),
                         budget=int(extra.get("budget", 15)), data_dir=tempfile.mkdtemp())
            assert env.reset() == frame["prompt"].iloc[i][1]["content"], f"{path} row {i}"
            checked += 1
    assert checked > 0


def test_build_report_is_written(cfg):
    path = cfg.dataset_root / "build_report.json"
    if not path.exists():
        pytest.skip("datasets not built yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["regen_obs"] is True
    for run_id in RUN_IDS:
        assert report["runs"][run_id]["rows_written"] == TRAIN_WORLDS_PER_RUN
        assert report["runs"][run_id]["archetype"] == TRAIN_ARCHETYPE[run_id]
