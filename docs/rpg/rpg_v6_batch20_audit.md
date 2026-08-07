# RPG v6 — Batch-20 Evaluation Audit (Opus-4.8 vs Qwen3-small)

Proactive fairness audit of the full 20-world runs for both models. Question
driving it: are we truly measuring reasoning, or tripping models on trivial /
harness issues? Snapshot 2026-08-05. Numbers from `results_v6/batch20_*`.

## Headline (after all grading fixes, fresh re-grade)

| Model | n | Accept | Accuracy [95% CI] | Part-A rate | Mean Part-B |
|---|---|---|---|---|---|
| Opus-4.8 | 20 | 4 | 0.20 [0.08, 0.42] | 0.50 | 0.65 |
| Qwen3-small (3.6-27B) | 20 | 1 | 0.05 [0.01, 0.24] | 0.35 | 0.46 |

**Do not present the Qwen number as-is.** The audit finds it is materially
depressed by harness/format effects, not reasoning. Details below.

## What is SOLID (verified)

1. **Every world is solvable and fairly gradeable.** Playing the ground-truth
   answer scores accept (A∧B) on all 20 worlds, both models' world set. The
   worlds are not broken and the gold is not mis-specified.
2. **Grading is now internally consistent** after three fixes:
   - Part A = fraction-of-achievable-benefit ≥ 0.90 (not an absolute tolerance
     that was 5% of range on wide worlds but 35% on the narrow clinic).
   - Part B credits ANY genuine mechanism proxy (not one hardcoded string), and
     strips verbose parentheticals before resolving proxy/decoy names.
   - Actuator signs judged in the increasing direction; **non-monotone
     (interior-optimum) actuators are not scored on sign** (no single sign is
     correct), so a titrated fix is never penalized on the sign item.
3. **Opus is clean:** 0/538 turns failed to parse; 0/20 hit the turn cap. Its
   numbers reflect reasoning.

## What is UNFAIR to Qwen (must fix before quoting its score)

1. **Output-format / verbosity trips (Qwen 8% of turns wasted; Opus 0%).**
   - **26 empty responses** across the run — the API returned no content on that
     turn (thinking model spent the budget on hidden reasoning, or the 2500-token
     cap cut it off before any answer). Wasted turn, not a wrong answer.
   - **6 truncated** turns — `<reasoning>` present but the `<action>` block was cut
     off (verbose thinking blew the 2500-token output budget).
2. **Turn-cap starvation (Qwen 9/20 hit the 32-turn cap; Opus 0/20).** Qwen uses
   ~18 code turns (like Opus) but its wasted turns + extra experiments push it
   over 32 before it submits. The 4 worlds that never answered were **actively
   solving**, e.g. greenhouse_40303's last reasoning:
   > "clear synergistic interaction between Iron and Acid... yield increased from
   > ~26 to ~66"
   It had essentially found the two-cause fix and ran out of turns before
   submitting. That is a harness limit, not a reasoning failure.

**Net:** an unknown but non-trivial share of Qwen's 19 non-accepts are
format/turn-budget artifacts. Its true reasoning accuracy is ≥ the measured 0.05
and the gap to Opus is overstated.

## Required fixes before the Qwen numbers are quotable — IMPLEMENTED

1. **Always request the model's MAX output tokens.** `openai_llm.py` presets now
   carry `max_output_tokens` (Qwen/gpt-oss 32768, deepseek 65536); `OpenAILLM`
   auto-applies it when `max_new_tokens` is unset. Runner `--max-new-tokens`
   default is now `None` (= use the model max). Bedrock/Opus gets 8192 (was an
   implicit 1536). Removes truncation as a failure mode. ✅
2. **Unproductive turns no longer consume the turn budget.** The agent loop is
   now `while productive < max_turns` with a separate hard iteration cap: a turn
   with no parsable action (empty/truncated) is logged (`unproductive: True`),
   the model is told to reply with a valid action, and the turn is retried
   without cost. Verified: 3 injected empty responses did not prevent answering.
   Removes the turn-cap starvation. ✅
3. **Re-run Qwen** with these in effect, then compare to Opus. (Opus can be
   re-run too for symmetry, though it was already clean; a re-run under identical
   settings is cleanest for the paper.)

Interim (no re-run): report Qwen answered-only / non-truncated as a labeled lower
bound. Prefer the re-run now that the fixes are in.

## What is GENUINE (holds for Opus, and likely for Qwen once fixed)

- **Bioreactor is easiest by construction** (single targeted knob, clean interior
  optimum, strong proxy, natural confound-break): Opus 4/5.
- **The other three fail for real, structure-specific reasons:** datacenter needs
  a second causal hop (over-cooling → condensation → dehumidify; both models stop
  at "cool less"); greenhouse needs a conjunction (iron AND pH); clinic needs an
  interior dose under effect heterogeneity.
- **Recurring genuine pattern:** models often get part-A (a reasonable
  intervention, 95–99% of benefit) while failing part-B (mislabel the mechanism)
  — "acts right, explains wrong."

## Known construction caveat (state in any writeup)

- **Clinic's ideal answer is a conditional policy** (treat the responsive
  subgroup, spare the other); the grader scores only a single dose. So clinic's
  0/5 partly reflects a skill/grader mismatch, not pure reasoning difficulty —
  not equivalent evidence to datacenter's 0/5.

## Residual (verdict-neutral, log it)

- One proxy string ("Interface/network error count (IEC)", slash+abbreviation)
  still doesn't resolve; the world fails part-A anyway so no verdict changes.
- Grading Monte-Carlo uses a fixed seed but not the world's seed, so a ±1-world
  wobble is possible at the 90%-benefit boundary. Pin per-world seed before final
  numbers if exactness matters.
