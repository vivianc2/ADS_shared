#!/usr/bin/env python3
"""ACED -> RPG easy-first curriculum (INITIAL implementation).

Self-contained: imports the existing RPG generator/oracle/split READ-ONLY and never
modifies them. Nothing here is wired into training yet — it's the scheduler + tier
definitions you plug into the RL dataset builder / trainer.

Design (matches rpg_v7_rl_research.md "Curriculum" + rpg_v7_reward_contract_decisions V3):
  Difficulty is parameterized by  archetype x features x depth.  Train easy-first and make
  it ADAPTIVE to batch pass-rate (RLVE-style) so GRPO groups stay in the informative band
  (neither all-solved nor all-failed -> that's what gives advantage/gradient).

Two hard constraints this honors:
  1. TRAIN SPLIT ONLY. The curriculum samples only train families (splits.py):
     the held-out archetypes {hidden_subtype, surrogate_trap, competing_causes} and skins
     {clinical, fermentation} are NEVER produced here — otherwise we'd leak the transfer set.
  2. `symptom_trap` is never a training feature (it only exists in the held-out surrogate_trap).

Tiers:
  T0  = ACED warm-up (structured, explicit-graph causal tasks) -- see AcedTier (stub / open design).
  T1..T5 = RPG, easy -> hard, over the 6 TRAIN archetypes x features x depth.
The RPG tier ordering is informed by both the design docs AND the measured frontier ceiling
(collider/instrument easy; synergy hard; see rpg_personal_docs/v9_dataset.md).

Run `python curriculum.py` for a no-LLM demo (prints the ladder + samples a few worlds/tier).
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- import the existing RPG code read-only (no modification) ---
_BASE = os.environ.get("RPG_SRC", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_BASE, "rpg_v9"), os.path.join(_BASE, "rpg_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sampler import sample_world                     # noqa: E402
from generate_v7 import audit                        # noqa: E402
from splits import train_archetypes, train_skins, split_of  # noqa: E402


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------
# Only TRAIN archetypes appear (the 6 non-reserved). feature_options is a list of
# feature-sets; a world at this tier draws ONE option. [] means "no difficulty feature".
# `two_cause` and `sign_flip` are mutually exclusive (sampler.py), so we never combine them.

@dataclass
class TierSpec:
    name: str
    archetypes: List[str]
    depths: Tuple[int, ...]
    feature_options: List[List[str]]
    note: str = ""


def rpg_ladder() -> List[TierSpec]:
    """Easy -> hard RPG tiers, restricted to the train split. Built at call time so it always
    reflects the current split (raises if a hoped-for train archetype was reserved)."""
    tr = set(train_archetypes())
    def keep(*a):  # only archetypes that are actually in the train split
        return [x for x in a if x in tr]
    return [
        TierSpec("T1_intro", keep("confounded_chain"), (2,), [[]],
                 "single confounded chain, shallow, no difficulty feature"),
        TierSpec("T2_single_lever", keep("confounded_chain", "collider_selection", "instrument_only"),
                 (2, 3), [[], ["sign_flip"]],
                 "add selection-bias + upstream-only lever; allow a sign flip"),
        TierSpec("T3_dose_reversal", keep("confounded_chain", "collider_selection", "instrument_only",
                                          "confounded_reversal", "dose_window"),
                 (3,), [[], ["sign_flip"], ["interior_dose"]],
                 "add Simpson's reversal + therapeutic-window dosing (interior optimum)"),
        TierSpec("T4_conjunction", keep("confounded_chain", "collider_selection", "instrument_only",
                                        "confounded_reversal", "dose_window", "synergy_pair"),
                 (3, 4), [[], ["interior_dose"], ["two_cause"]],
                 "add AND-gate synergy (must set both levers)"),
        TierSpec("T5_full", keep("confounded_chain", "collider_selection", "instrument_only",
                                 "confounded_reversal", "dose_window", "synergy_pair"),
                 (4,), [["two_cause"], ["sign_flip", "interior_dose"], ["interior_dose"]],
                 "deepest chains + layered features"),
    ]


# ---------------------------------------------------------------------------
# ACED warm-up tier (T0) -- interface + stub (open design decision)
# ---------------------------------------------------------------------------
class AcedTier:
    """Tier 0: warm up on ACED-Bench structured causal tasks (explicit PGM graphs, do-calculus
    queries) before the open-scenario RPG worlds. ACED lives in vivian/ADS (separate benchmark).

    OPEN DESIGN DECISION (documented, not yet implemented): to sit in ONE RL curriculum, an ACED
    task must expose the same contract as an RPG episode -- an id-space observation, measure/
    intervene/answer actions, and a verifiable reward. Two candidate routes:
      (a) adapt ACED worlds into an RPGEnv-compatible bundle (write an ADS->rpg_rl shim); or
      (b) run a short ACED-native phase first, then switch datasets (phase curriculum, not unified).
    Until one is chosen this raises, so the ladder cleanly no-ops T0 for now."""
    def __init__(self, aced_worlds_dir: Optional[str] = None):
        self.aced_worlds_dir = aced_worlds_dir

    def iter_tasks(self):
        raise NotImplementedError(
            "ACED tier-0 not wired yet — decide (a) ADS->rpg_rl env shim vs (b) separate ACED phase. "
            "See AcedTier docstring + rpg_curriculum/README.md.")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@dataclass
class CurriculumScheduler:
    """Holds the current tier and advances/retreats it. `adaptive` keeps the batch pass-rate in
    [retreat_lo, advance_hi] (RLVE-style); `fixed` advances every `advance_every` updates."""
    tiers: List[TierSpec] = field(default_factory=rpg_ladder)
    mode: str = "adaptive"            # "adaptive" | "fixed"
    advance_hi: float = 0.70          # pass-rate above this -> tier too easy -> advance
    retreat_lo: float = 0.20          # pass-rate below this -> tier too hard -> retreat
    advance_every: int = 20           # (fixed mode) updates per tier
    idx: int = 0
    _n_updates: int = 0

    def current(self) -> TierSpec:
        return self.tiers[self.idx]

    def update(self, batch_pass_rate: Optional[float] = None) -> TierSpec:
        """Call once per training batch with the fraction of the batch that 'passed'
        (e.g. reward>0, or accepted). Returns the (possibly new) current tier."""
        self._n_updates += 1
        if self.mode == "fixed":
            if self._n_updates % self.advance_every == 0 and self.idx < len(self.tiers) - 1:
                self.idx += 1
        elif self.mode == "adaptive" and batch_pass_rate is not None:
            if batch_pass_rate >= self.advance_hi and self.idx < len(self.tiers) - 1:
                self.idx += 1
            elif batch_pass_rate <= self.retreat_lo and self.idx > 0:
                self.idx -= 1
        return self.current()


# ---------------------------------------------------------------------------
# Tier sampling (wraps sample_world + audit; respects split + tier constraints)
# ---------------------------------------------------------------------------
def sample_tier_world(spec: TierSpec, seed: int, rng: random.Random,
                      max_tries: int = 60) -> Optional[Dict[str, Any]]:
    """Return one audited world matching `spec` (archetype/features/depth), or None if no
    audited world was found within `max_tries` seeds. Reuses the production audit gate."""
    skins = train_skins()
    for i in range(max_tries):
        s = seed + i
        r = random.Random(s)
        skin = r.choice(skins)
        arche = r.choice(spec.archetypes)
        feats = list(r.choice(spec.feature_options))
        if split_of(skin, arche) != "train":            # defensive: never leak the held-out set
            continue
        try:
            w = sample_world(s, skin=skin, features=feats, archetype=arche)
        except Exception:
            continue
        if w["ground_truth"].get("_depth") not in spec.depths:
            continue
        try:
            res = audit(w)
        except Exception:
            continue
        if res.get("ok"):
            w["ground_truth"]["_seed"] = s
            w["_curriculum_tier"] = spec.name
            return {"world": w, "gold": res["gold"], "battery": res["battery"], "tier": spec.name,
                    "seed": s, "skin": skin, "archetype": arche, "features": feats}
    return None


def curriculum_worlds(scheduler: CurriculumScheduler, n: int, seed0: int = 40_000_000):
    """Yield `n` audited worlds at the scheduler's CURRENT tier. The trainer is expected to call
    scheduler.update(pass_rate) between batches; this just samples the current tier."""
    rng = random.Random(seed0)
    got, seed = 0, seed0
    while got < n:
        b = sample_tier_world(scheduler.current(), seed, rng)
        seed += 100
        if b is None:
            continue
        got += 1
        yield b


# ---------------------------------------------------------------------------
# Demo (no LLM): print the ladder + sample a couple worlds per tier to prove it runs
# ---------------------------------------------------------------------------
def _demo():
    ladder = rpg_ladder()
    print("Train archetypes:", train_archetypes())
    print("Train skins:", train_skins())
    print("\n=== RPG difficulty ladder (train-split only) ===")
    for t in ladder:
        print(f"  {t.name:16s} arch={t.archetypes} depths={t.depths} feats={t.feature_options}\n"
              f"      {t.note}")
    print("\n=== sample 2 audited worlds per tier (verifies split + audit gate) ===")
    for t in ladder:
        rng = random.Random(0)
        for k in range(2):
            b = sample_tier_world(t, 40_000_000 + hash(t.name) % 100000 + k * 137, rng)
            if b:
                print(f"  {t.name:16s} -> {b['archetype']:18s} skin={b['skin']:12s} "
                      f"depth={b['world']['ground_truth']['_depth']} feats={b['features']} "
                      f"split={split_of(b['skin'], b['archetype'])}")
            else:
                print(f"  {t.name:16s} -> (no audited world found)")


if __name__ == "__main__":
    _demo()
