"""The built datasets inject the right prompt and hold the worlds fixed across runs."""

from __future__ import annotations

import hashlib
import json

import pytest

from prompt_compare_rl import prompts as prompt_lib
from prompt_compare_rl.config import PROMPT_IDS

pd = pytest.importorskip("pandas")

SPLITS = ("train.parquet", "validation_small.parquet")


def _frames(cfg, filename):
    return {pid: pd.read_parquet(cfg.dataset_dir(pid) / filename) for pid in PROMPT_IDS}


@pytest.mark.parametrize("filename", SPLITS)
def test_every_row_carries_its_run_prompt(datasets_built, filename):
    texts = prompt_lib.read_materialized(datasets_built.prompt_dir)
    for pid, frame in _frames(datasets_built, filename).items():
        expected = prompt_lib.sha256_text(texts[pid])
        digests = {hashlib.sha256(r[0]["content"].encode()).hexdigest() for r in frame["prompt"]}
        assert digests == {expected}


@pytest.mark.parametrize("filename", SPLITS)
def test_worlds_and_order_identical_across_runs(datasets_built, filename):
    frames = _frames(datasets_built, filename)
    reference = frames[PROMPT_IDS[0]]
    keys = [
        (dict(e)["seed"], dict(e)["skin"], dict(e)["archetype"]) for e in reference["extra_info"]
    ]
    for pid in PROMPT_IDS[1:]:
        other = frames[pid]
        assert len(other) == len(reference)
        assert [
            (dict(e)["seed"], dict(e)["skin"], dict(e)["archetype"]) for e in other["extra_info"]
        ] == keys
        for index in range(len(reference)):
            assert reference.iloc[index]["prompt"][1]["content"] == other.iloc[index]["prompt"][1]["content"]
            assert dict(reference.iloc[index]["extra_info"]) == dict(other.iloc[index]["extra_info"])


@pytest.mark.parametrize("filename", SPLITS)
def test_rows_preserved_from_source(datasets_built, filename):
    source = datasets_built.source_train if filename == "train.parquet" else datasets_built.source_val
    if not source.exists():
        pytest.skip(f"source missing: {source}")
    original = pd.read_parquet(source)
    built = pd.read_parquet(datasets_built.dataset_dir("p1") / filename)
    assert len(built) == len(original)
    for index in range(len(original)):
        assert dict(built.iloc[index]["extra_info"]) == dict(original.iloc[index]["extra_info"])
        assert built.iloc[index]["data_source"] == original.iloc[index]["data_source"]
        assert built.iloc[index]["env_class"] == original.iloc[index]["env_class"]


def test_validation_set_is_validation_small(datasets_built):
    frame = pd.read_parquet(datasets_built.dataset_dir("p1") / "validation_small.parquet")
    assert len(frame) == 45
    assert {dict(e)["split"] for e in frame["extra_info"]} == {"heldout"}
    archetypes = {}
    for extra in frame["extra_info"]:
        archetypes[dict(extra)["archetype"]] = archetypes.get(dict(extra)["archetype"], 0) + 1
    assert len(archetypes) == 9 and set(archetypes.values()) == {5}


def test_train_set_is_the_deleaked_v9_train_split(datasets_built):
    frame = pd.read_parquet(datasets_built.dataset_dir("p1") / "train.parquet")
    assert len(frame) == 1536
    assert {dict(e)["split"] for e in frame["extra_info"]} == {"train"}
    assert "data_v9_deleaked/train.parquet" in str(datasets_built.source_train)


def test_regenerated_observation_matches_the_live_environment(datasets_built, rpg_src):
    """The catalog in the dataset prompt is the one the env will actually simulate.

    This is the invariant `RPGSkyEnv` relies on and the reason the builder re-renders
    observations: the stored 2026-08-18 text no longer matched the current v9 generator
    for most rows.
    """
    import os
    import sys
    import tempfile

    for path in (os.path.join(rpg_src, "rpg_rl"), os.path.join(rpg_src, "rpg_v9")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from env import RPGEnv
    from generate_v7 import audit
    from sampler import sample_world

    frame = pd.read_parquet(datasets_built.dataset_dir("p1") / "validation_small.parquet")
    for index in range(0, len(frame), 9):  # a spread across archetypes, kept fast
        row = frame.iloc[index]
        extra = dict(row["extra_info"])
        world = sample_world(int(extra["seed"]), skin=extra["skin"], archetype=extra["archetype"])
        world["ground_truth"]["_seed"] = int(extra["seed"])
        graded = audit(world)
        env = RPGEnv(
            world=world, gold=graded["gold"], battery=graded["battery"],
            catalog_seed=int(extra["seed"]), max_turns=int(extra["max_turns"]),
            budget=int(extra["budget"]), data_dir=tempfile.mkdtemp(prefix="pc_test_"),
        )
        assert env.reset() == row["prompt"][1]["content"], (
            f"row {index} ({extra['archetype']}/{extra['skin']}) prompt does not match env.reset()"
        )


def test_build_report_records_the_refresh(datasets_built):
    report_path = datasets_built.dataset_root / "build_report.json"
    if not report_path.exists():
        pytest.skip("no build report")
    report = json.loads(report_path.read_text())
    assert report["rpg_proto"] == "rpg_v9"
    assert report["regen_obs"] is True
    for split in ("train", "validation_small"):
        assert report["splits"][split]["source_system_prompt_sha256"] == [
            prompt_lib.EXPECTED_SHA256["p1"]
        ]
        assert "changed_vs_source" in report["splits"][split]
