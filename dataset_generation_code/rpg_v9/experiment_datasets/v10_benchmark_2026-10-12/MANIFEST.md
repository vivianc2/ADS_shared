# v10_benchmark_2026-10-12: RPG v10 evaluation splits (ygg, RPG_WORLD_VERSION=v10, RPG_SYNERGY_SOFT=20)

Built after the v10 rebuild (units on every control and signal, real controls only, per-world stories that list everything
and state the true recoverable loss, role-agnostic names, marker polarity, interior random gold values in every
archetype, v10 sign definition). Details and results: personal_docs `results/v10/V10_BENCHMARK_REBUILD_2026-10-12.md`.

| file | worlds | skins | seed base | role |
|---|---|---|---|---|
| dev_heldout_domain.parquet | 270 (30 x 9) | clinical, fermentation | 51,000,000 | validation |
| test_heldout_domain.parquet | 270 | clinical, fermentation | 43,000,000 | final test (unseen skins) |
| test_in_domain.parquet | 270 | the 8 other skins | 53,000,000 | final test |

- Same recipe as v9_splits_2026-09-26: the first 30 audited worlds per archetype in seed order.
- No train split is built.
- Every row carries `extra_info.world_version = "v10"` and `oracle_json` (gold + battery, checked against the world id).
- **Code from before the 2026-10-12 rebuild builds different v10 worlds:** the env's stale-prompt guard fails loudly.
  Superseded v10 builds are outside the repo (ygg `~/rpgrl/w10/OLD_*`).
- Rebuild: ygg `~/rpgrl/w10/build_v10_benchmark.sh` (~10 min per split at 32 workers).
- World JSON for the free-text harness: `python dump_worlds_from_parquet.py <split>.parquet <dir>` (~1 min per split;
  uses the stored oracle).
- Verified 2026-10-13: 0/270 stale prompts in each split (third build: interior golds).
