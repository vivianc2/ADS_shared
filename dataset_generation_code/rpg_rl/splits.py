#!/usr/bin/env python3
"""Train / held-out split for RL — with ZERO leakage (decision V13).

The headline result of the whole project is TRANSFER: does the RL-tuned model reason
about hidden causes on world *families it never trained on*, or did it memorize a
motif? That number is only meaningful if the eval families are held out entirely. So
we reserve, by construction:

    - >= 2 whole SKINS (domains) — e.g. clinical + fermentation — never seen in training
    - >= 1 whole ARCHETYPE (structural family) — e.g. hidden_subtype — never trained on

A world is TRAIN only if its skin is a train-skin AND its archetype is a train-archetype.
A world is HELD-OUT if its skin OR its archetype is reserved. This is deliberately strict
(a held-out skin in a trained archetype is still held out) so the transfer claim is clean.

Deterministic: the reserved sets are fixed here (not random) so train/eval are stable
across machines and reruns.
"""

from __future__ import annotations

from typing import Dict, List

from skins import skin_names
from sampler import ARCHETYPES

# --- reserved (held-out) families. Edit here to change the split; keep >=2 skins + >=1 arch. ---
# run-6 (v8, 9 archetypes): held out = 3 distinct + near-flat@base families (transfer set),
# plus 2 reserved skins for domain transfer. Train = the 6 signal archetypes on the 8 train skins.
HELDOUT_SKINS = ["clinical", "fermentation"]        # 2 of 10 skins reserved
HELDOUT_ARCHETYPES = ["hidden_subtype", "surrogate_trap", "competing_causes"]  # 3 of 9 reserved


def train_skins() -> List[str]:
    return [s for s in skin_names() if s not in HELDOUT_SKINS]


def train_archetypes() -> List[str]:
    return [a for a in ARCHETYPES if a not in HELDOUT_ARCHETYPES]


def split_of(skin: str, archetype: str) -> str:
    """"train" iff skin AND archetype are both non-reserved; else "heldout"."""
    if skin in HELDOUT_SKINS or archetype in HELDOUT_ARCHETYPES:
        return "heldout"
    return "train"


def describe() -> Dict[str, object]:
    return {
        "heldout_skins": HELDOUT_SKINS,
        "heldout_archetypes": HELDOUT_ARCHETYPES,
        "train_skins": train_skins(),
        "train_archetypes": train_archetypes(),
        "note": "a world is held out if its skin OR archetype is reserved (strict).",
    }
