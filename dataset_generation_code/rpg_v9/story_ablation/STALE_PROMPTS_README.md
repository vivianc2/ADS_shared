# ⚠️ STALE PROMPTS — do not train on these parquets with SkyRL as-is (2026-09-24)

The parquets here (except `rl_train/rl_hard_lever_ds*`, built 2026-09-21/22) were rendered by generator code
**before commit 745c110 (2026-08-30, synergy redesign)**. For ~half the worlds the stored first prompt — the only
copy of the id→variable catalog the policy ever sees — no longer matches the world the current `rpg_v9` code
rebuilds from (seed, skin, archetype). `skyrl_rpg/env.py` now refuses such rows (override: RPG_ALLOW_STALE_PROMPT=1).

Use instead: `rpg_v9/experiment_datasets/v9_rebuilt_2026-09-24/` (same seeds, re-rendered; see its MANIFEST.md),
or rebuild any file with `skyrl_rpg/rebuild_v9_sets.py rebuild <in> <out>`. Kept here unchanged for provenance
(they reproduce the historical runs). Full story: personal_docs `RPG_VERSIONS_READ_ME_FIRST.md`.
