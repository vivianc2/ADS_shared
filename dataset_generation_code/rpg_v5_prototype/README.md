# RPG v5 prototype — declarative SCM worlds

Vertical slice for the design in
`docs/rpg/worldgen_rpg_plan_v5_scm_chain.md`. Standalone (numpy for gen; boto3 +
`../../framework_code/bedrock_llm.py` for the live agent). Does **not** touch the
shared RPG engine yet. Progress + worked example:
`docs/rpg/rpg_v5_progress_and_worked_example.md`.

## Environment

Use the `ADS-rpg` conda env (has numpy, boto3, pandas):

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ADS-rpg
cd dataset_generation_code/rpg_v5_prototype
```

## End-to-end trial

```bash
# audit + demo (no API)
python demo_solve.py

# 1) generate a trial batch of worlds (audited; only passing worlds emitted)
python generate.py --outdir out_v5_trial --n 6 --seed 20260804

# 2) smoke-test the whole loop with NO API access
python run_agent_v5.py --worlds-dir out_v5_trial --backend mock --out results_v5/mock_smoke.json

# 3) LIVE: set your Bedrock credential first, then smoke ONE world before the batch
export AWS_BEARER_TOKEN_BEDROCK=...   # your token
export AWS_DEFAULT_REGION=us-west-2
python run_agent_v5.py \
  --world-json out_v5_trial/world_bioreactor_yield_collapse_20260905.json \
  --backend bedrock --model us.anthropic.claude-opus-4-8 --out results_v5/live_smoke.json -v

# 4) full trial (Opus 4.8), ~10–25 min for 6 worlds
python run_agent_v5.py --worlds-dir out_v5_trial \
  --backend bedrock --model us.anthropic.claude-opus-4-8 --out results_v5/opus48_trial.json -v
```

Default model is Opus 4.8. Each result JSON carries per-turn traces (reasoning,
query, stats, memory) and a per-world grade breakdown (part A utility, part B
battery, which predictions failed, queries used) for gap analysis.

## Files

- `scm.py` — generic SCM evaluator + closed mechanism library
  (`linear/saturating/hill/soft_threshold/interaction/sign_flip`). One
  topological interpreter is the whole world engine. `knob_effects` are
  structural interventions (scale/add/set on target nodes, propagated through
  the chain); `obs_effects` bias a *measured* reading without changing the true
  state or utility — this is how symptom-masking traps are modeled faithfully.
- `worlds.py` — two domains (bioreactor yield collapse; municipal water
  discoloration) built on the same structural family: 3-hop hidden chain,
  confounded decoy observable, sign-flip knob, interior-optimum dose, symptom
  trap.
- `oracle.py` — computed golden answer (MC + golden-section on continuous
  knobs), counterfactual battery (ground truth for grading *understanding*),
  auto-calibration to difficulty bands, faithfulness audits (name leakage,
  decoy inertness, proxy-signal band), and a solvability certificate.
- `demo_solve.py` — audits both worlds, plays a scripted expert trajectory, and
  grades a correct vs. surface-proxy answer.

## What the slice proves

1. The SCM is faithful (the evaluator *is* the world definition).
2. The world is solvable by meaningful experiments (clamp to break a confound,
   dose-sweep to find the interior optimum, trap-sweep to reject symptom
   masking) within ~3 decisive queries.
3. The **computed** grader accepts a correct answer (utility gap ≈ 0, battery
   1.0) and rejects a surface-proxy answer (utility gap ~44, battery 0.25) —
   the thing the v4 string-matcher could not do.

## Next (port into shared engine)

Per doc §9: move `scm.py`/`oracle.py` logic into `world_gen_rpg_old.py` as a new
`scm_mechanism_chain` archetype (additive — do not delete existing functions),
add `sweep`/`clamp` query modes to `simulator_rpg.py`, extend the answer schema
and grader, and wire efficiency/convergence logging.
