# ThinkRouter Final Official Report

Protocol version: `official_v1`

This report contains the only official benchmark results tracked by the repository.

| benchmark   | official_learned_policy   | official_router_name   |   accuracy |    avg_cost |   p95_latency |   utility | strongest_fixed_policy                        |   strongest_fixed_utility | aggregate_policy                           |   aggregate_utility |   cost_reduction_vs_strongest_fixed |   accuracy_drop_vs_strongest_fixed | beats_strongest_fixed   | beats_aggregate_baseline   |
|:------------|:--------------------------|:-----------------------|-----------:|------------:|--------------:|----------:|:----------------------------------------------|--------------------------:|:-------------------------------------------|--------------------:|------------------------------------:|-----------------------------------:|:------------------------|:---------------------------|
| gsm8k       | phase2_logreg_joint       | logreg_joint           |       0.95 | 0.00018638  |       7.40148 |  0.801039 | fixed_model_budget_qwen3.5-flash-2026-02-23_0 |                 0.757455  | aggregate_utility_qwen3-max-2026-01-23_256 |           0.755108  |                            0.255309 |                               0.05 | True                    | True                       |
| math500     | phase2_logreg_joint       | logreg_joint           |       0.25 | 0.000154564 |      18.2515  | -0.115803 | fixed_model_budget_qwen3-max-2026-01-23_0     |                -0.0905211 | aggregate_utility_qwen3-max-2026-01-23_0   |          -0.0905211 |                            0.736239 |                               0.05 | False                   | False                      |
| humaneval   | phase2_logreg_joint       | logreg_joint           |       0.6  | 0.000377477 |      19.0264  |  0.217585 | fixed_model_budget_qwen3-max-2026-01-23_0     |                 0.881456  | aggregate_utility_qwen3-max-2026-01-23_0   |           0.881456  |                            0.615604 |                               0.4  | False                   | False                      |

## Win Condition

- Benchmarks where the learned router beats both strongest fixed and aggregate baselines: `gsm8k`

## Failure Summary

| benchmark   | error_type       |   count |    avg_cost |   avg_latency |
|:------------|:-----------------|--------:|------------:|--------------:|
| gsm8k       | wrong_answer     |       1 | 0.000436    |       1.28016 |
| math500     | no_final_answer  |      15 | 0           |       0       |
| humaneval   | malformed_answer |       8 | 0.000355055 |       9.18923 |

Historical smoke/dev slices are appendix-only artifacts and are not part of this official report.
