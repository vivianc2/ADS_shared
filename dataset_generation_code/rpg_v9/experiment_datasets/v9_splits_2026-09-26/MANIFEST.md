# v9_splits_2026-09-26 — train / dev / test for RPG v9 (built at ADS_shared rpg @ 8c3453a, RPG_SYNERGY_SOFT=20)

Full rationale, history of earlier datasets and usage rules: personal_docs `benchmark/V9_DATASETS_2026-09-26.md`.

| file (all9/) | worlds | skins (domains) | seeds | role |
|---|---|---|---|---|
| train.parquet | 1800 (200 × 9 archetypes) | 8 training skins | 50,000,000+ | training |
| dev_heldout_domain.parquet | 270 (30 × 9) | clinical, fermentation | 51,000,000+ | in-run eval / checkpoint selection |
| dev_in_domain.parquet | 270 (30 × 9) | 8 training skins | 52,000,000+ | in-run "is it learning at all" |
| test_in_domain.parquet | 270 (30 × 9) | 8 training skins | 53,000,000+ | final test, tier 1 (in-distribution) |
| test_heldout_domain.parquet | 270 (30 × 9) | clinical, fermentation | 43,000,000+ | final test, tier 2 (= v9_rebuilt heldout_balanced_30x9, byte-identical) |

`a4/` and `hard3/` = the same files filtered to {confounded_chain, dose_window, instrument_only} and
{competing_causes, synergy_pair, hidden_subtype}: train 600, validation (= dev_heldout_domain) 90, dev_in_domain 90,
test_heldout_domain 90, test_in_domain 90. Drop-in `DATA_DIR` (launcher reads train.parquet + validation.parquet).
`a4/test_heldout_domain` is byte-identical to `v9_rebuilt_2026-09-24/a4_archetypes_heldout_skins_90`.
Tier 3 (held-out archetype) = the other 6 archetypes' rows of all9/test_heldout_domain.

Verified: 0 stale prompts (all files); no duplicate worlds/prompts within files; zero world or prompt overlap between any
two splits and between train and every older parquet; skins balanced within each archetype (train 23–27 per cell).
Rebuild: `PYTHONPATH=rpg_rl:rpg_v9 RPG_SYNERGY_SOFT=20 python skyrl_rpg/rebuild_v9_sets.py balanced <out> --per_arch N
--seed0 S --skins {train,heldout} --label L` (seeds / N / skins above).
