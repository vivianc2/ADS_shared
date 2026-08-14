#!/usr/bin/env python3
"""On-demand audited world stream for RL training (V12) + split routing (V13).

RL wants a FRESH seeded world (almost) every step, not a fixed file set — that is
what makes the task reasoning rather than lookup. This module yields audited worlds
as ready-to-use env bundles. Each world is generated IN MEMORY (sample_world ->
audit); the audit already computes gold + battery, so we attach them and the env
never recomputes the oracle per episode (decision V9).

Determinism / no collision: seeds are monotonic (seed0, seed0+1, ...) and every
yielded world_id is tracked, so no two yielded worlds ever share a world_id — the
stream is reproducible given (split, seed0, rng_seed) and collision-free.

Split routing (splits.py, decision V13):
- a TRAIN stream only draws (train_skin x train_archetype) cells;
- a HELDOUT stream only draws cells whose skin OR archetype is reserved.
Every yielded world is re-checked with split_of() as a defensive leakage guard, so a
world can NEVER leak into the wrong split even if a cell list were mis-built.

Usage (as a library):
    from world_stream import WorldStream
    stream = WorldStream(split="train", seed0=1_000_000)
    for bundle in stream:            # infinite
        env = bundle.make_env()
        ...

Usage (CLI, prints acceptance + distribution for a split):
    PYTHONPATH=../rpg_v7_prototype python world_stream.py --split train --n 60
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from sampler import sample_world, ARCHETYPES, FEATURES
from skins import skin_names
from generate_v7 import audit
from splits import split_of, train_skins, train_archetypes
from env import RPGEnv
from reward import RewardConfig


@dataclass
class WorldBundle:
    """One audited world plus its precomputed oracle, ready to become an env."""
    world: Dict[str, Any]
    gold: Dict[str, Any]
    battery: Dict[str, Any]
    seed: int
    skin: str
    archetype: str
    split: str

    def make_env(self, *, max_turns: int = 32, budget: int = 15,
                 reward_cfg: Optional[RewardConfig] = None,
                 data_dir: Optional[str] = None) -> RPGEnv:
        """Construct an RPGEnv over this world. catalog_seed is tied to the world seed
        so the id<->name map is reproducible for this world."""
        return RPGEnv(world=self.world, gold=self.gold, battery=self.battery,
                      catalog_seed=self.seed, max_turns=max_turns, budget=budget,
                      reward_cfg=reward_cfg or RewardConfig(), data_dir=data_dir)


def _cells_for(split: str) -> List[Tuple[str, str]]:
    """All (skin, archetype) cells that route to `split`."""
    if split == "train":
        return [(s, a) for s in train_skins() for a in train_archetypes()]
    # heldout: any cell whose skin OR archetype is reserved
    return [(s, a) for s in skin_names() for a in ARCHETYPES
            if split_of(s, a) == "heldout"]


@dataclass
class WorldStream:
    """Infinite (until seed space exhausts) generator of audited world bundles for a
    given split. Deterministic in (split, seed0, rng_seed)."""
    split: str = "train"                 # "train" | "heldout"
    seed0: int = 1_000_000
    require_feature: Optional[str] = None
    archetypes: Optional[List[str]] = None  # restrict to these archetypes (e.g. curriculum start)
    rng_seed: int = 0                    # controls cell-selection order only

    # runtime counters (useful for the acceptance-rate health check)
    attempts: int = field(default=0, init=False)
    accepted: int = field(default=0, init=False)
    rejected_by_gate: Counter = field(default_factory=Counter, init=False)
    _seed: int = field(default=0, init=False)
    _seen_wids: Set[str] = field(default_factory=set, init=False)
    _cells: List[Tuple[str, str]] = field(default_factory=list, init=False)
    _rng: random.Random = field(default=None, init=False)

    def __post_init__(self):
        assert self.split in ("train", "heldout"), self.split
        if self.require_feature is not None:
            assert self.require_feature in FEATURES, self.require_feature
        self._seed = self.seed0
        self._cells = _cells_for(self.split)
        if self.archetypes:                       # optional archetype restriction
            self._cells = [(s, a) for (s, a) in self._cells if a in self.archetypes]
        assert self._cells, f"no cells for split={self.split} archetypes={self.archetypes}"
        self._rng = random.Random(self.rng_seed)

    def acceptance(self) -> float:
        return (self.accepted / self.attempts) if self.attempts else 0.0

    def next(self, max_attempts: int = 100_000) -> WorldBundle:
        """Return the next audited, split-correct, non-duplicate world bundle."""
        tried = 0
        while tried < max_attempts:
            tried += 1
            self.attempts += 1
            seed = self._seed
            self._seed += 1
            skin, arche = self._rng.choice(self._cells)
            try:
                world = sample_world(seed, skin=skin, archetype=arche)
            except Exception as e:                       # noqa: BLE001
                self.rejected_by_gate[f"sample_exc:{type(e).__name__}"] += 1
                continue
            world["ground_truth"]["_seed"] = seed
            wid = world["world_id"]
            if wid in self._seen_wids:                   # collision guard
                self.rejected_by_gate["dup_world_id"] += 1
                continue
            if self.require_feature and \
               self.require_feature not in world["ground_truth"]["_features"]:
                self.rejected_by_gate["missing_feature"] += 1
                continue
            # defensive leakage guard: NEVER yield a world into the wrong split
            if split_of(world["domain"], world["ground_truth"]["_archetype"]) != self.split:
                self.rejected_by_gate["wrong_split"] += 1
                continue
            try:
                res = audit(world)
            except Exception as e:                       # noqa: BLE001
                self.rejected_by_gate[f"audit_exc:{type(e).__name__}"] += 1
                continue
            if not res["ok"]:
                for f in res["fails"]:
                    self.rejected_by_gate[f] += 1
                continue
            self._seen_wids.add(wid)
            self.accepted += 1
            return WorldBundle(world=world, gold=res["gold"], battery=res["battery"],
                               seed=seed, skin=world["domain"],
                               archetype=world["ground_truth"]["_archetype"],
                               split=self.split)
        raise RuntimeError(f"world_stream: {max_attempts} attempts without an audited "
                           f"world (split={self.split}, seed~{self._seed})")

    def take(self, n: int) -> List[WorldBundle]:
        return [self.next() for _ in range(n)]

    def __iter__(self) -> Iterator[WorldBundle]:
        while True:
            yield self.next()


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--require-feature", default=None, choices=FEATURES)
    args = ap.parse_args()

    stream = WorldStream(split=args.split, seed0=args.seed0,
                         require_feature=args.require_feature)
    skins_seen, arch_seen = Counter(), Counter()
    for b in stream.take(args.n):
        skins_seen[b.skin] += 1
        arch_seen[b.archetype] += 1

    print(f"split={args.split}  accepted={stream.accepted}/{stream.attempts} "
          f"(acceptance {stream.acceptance():.0%})")
    print(f"skins:      {dict(skins_seen)}")
    print(f"archetypes: {dict(arch_seen)}")
    print(f"rejected by gate: {dict(stream.rejected_by_gate)}")


if __name__ == "__main__":
    _main()
