# RPG v6 — Results

## Overall (all topologies pooled)

| Model | n | Accepted | Accuracy [95% CI] | Part-A rate | Mean Part-B | Artifact-flagged |
|---|---|---|---|---|---|---|
| opus4.8 | 20 | 4 | 0.20 [0.08, 0.42] | 0.50 | 0.65 | 8 |
| qwen3-small | 20 | 2 | 0.10 [0.03, 0.30] | 0.40 | 0.61 | 9 |

## By topology (accept rate [95% CI], n)

| Topology | opus4.8 | qwen3-small |
|---|---|---|
| bioreactor_titer_loss | 4/5 (0.80) [0.38,0.96] | 2/5 (0.40) [0.12,0.77] |
| datacenter_throughput | 0/5 (0.00) [0.00,0.43] | 0/5 (0.00) [0.00,0.43] |
| greenhouse_yield | 0/5 (0.00) [0.00,0.43] | 0/5 (0.00) [0.00,0.43] |
| clinic_readmission | 0/5 (0.00) [0.00,0.43] | 0/5 (0.00) [0.00,0.43] |

## Decomposition: part-A (found the fix) vs part-B (understood structure)

### opus4.8

| Topology | n | Accept | Part-A rate | Mean Part-B |
|---|---|---|---|---|
| bioreactor_titer_loss | 5 | 4/5 | 1.00 | 0.92 |
| datacenter_throughput | 5 | 0/5 | 0.00 | 0.65 |
| greenhouse_yield | 5 | 0/5 | 0.40 | 0.66 |
| clinic_readmission | 5 | 0/5 | 0.60 | 0.37 |

### qwen3-small

| Topology | n | Accept | Part-A rate | Mean Part-B |
|---|---|---|---|---|
| bioreactor_titer_loss | 5 | 2/5 | 0.60 | 0.87 |
| datacenter_throughput | 5 | 0/5 | 0.00 | 0.67 |
| greenhouse_yield | 5 | 0/5 | 0.20 | 0.55 |
| clinic_readmission | 5 | 0/5 | 0.80 | 0.35 |

## Notes

- Accept = part-A (utility within tolerance of computed optimum) AND part-B (≥0.8 of structure items correct).
- CIs are Wilson 95% intervals; wide at small n.
- Artifact-flagged results across all runs: **17** — these were auto-flagged as possible harness/resolution issues and require inspection before being read as reasoning failures.