"""prompt_compare_rl -- system-prompt comparison under the SkyRL RPG v9 GRPO pipeline.

Three independent GRPO runs (p1/p2/p3) that differ ONLY in the system prompt. Every
other input -- model init, training worlds and their order, validation worlds, seeds,
reward, and all hyperparameters -- is held byte-identical across the three runs.

Modules
-------
- ``prompts``       : the three prompt candidates, sourced from ``prompt_compare``.
- ``config``        : the single source of truth for paths, env vars and SkyRL overrides.
- ``build_dataset`` : writes the per-prompt train/validation parquet files.
- ``sky_env``       : the SkyRL-Gym environment (thin subclass of the shipped ``RPGSkyEnv``)
                      that adds the evaluation metrics and the prompt-arrival assertion.
- ``main``          : the SkyRL training entrypoint.
- ``report_eval``   : the step-0 / step-4 / step-8 comparison report.
- ``smoke_test``    : the required pre-launch smoke test.
"""

__all__ = ["prompts", "config", "build_dataset", "sky_env", "main", "report_eval"]
