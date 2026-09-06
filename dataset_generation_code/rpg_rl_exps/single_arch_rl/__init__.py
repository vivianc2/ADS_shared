"""single_arch_rl -- two GRPO runs that differ only in the training archetype.

    easy -> 96 `dose_window` worlds
    hard -> 96 `confounded_reversal` worlds

Both evaluate on the same held-out `validation_small.parquet` (45 worlds, 9 archetypes
x 5). See README.md; every setting lives in `config.py`.
"""

__all__ = ["config", "metrics", "build_dataset", "sky_env", "main", "report_eval"]
