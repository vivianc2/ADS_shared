# v9_rebuilt_2026-09-24 — parquets rendered by the CURRENT generator (ADS_shared rpg @ f0345b9, RPG_SYNERGY_SOFT=20)

Built by `skyrl_rpg/rebuild_v9_sets.py`; every file verifies 0 stale prompts. Why: personal_docs
`results/RPG_V9_DATASET_DRIFT_2026-09-23.md`, `results/RPG_EVAL_HARNESS_V8_SHADOWING_2026-09-24.md`.

| path | rows | what |
|---|---|---|
| rl_a4_base_ds/{train,validation}.parquet | 180 / 128 | same seeds as rl_train/rl_a4_base_ds (= rl_a5_partb_ds), prompts re-rendered (86 / 65 changed) |
| data_v9_deleaked/validation.parquet | 128 | same seeds as rpg_v9/data_v9_deleaked/validation, re-rendered (65 changed) — the old "128 held-out" set |
| heldout_balanced_30x9/test.parquet | 270 | NEW: 30 worlds x 9 archetypes, held-out skins clinical/fermentation, seeds 43,000,000+ |
| a4_archetypes_heldout_skins_90/test.parquet | 90 | subset of the above: confounded_chain, dose_window, instrument_only (the archetypes rl_a4 trains on) |
| heldout_balanced_10x9/test.parquet | 90 | first 10 per archetype of heldout_balanced_30x9 (probe subset) |
