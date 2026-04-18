# Qwen Multi-Benchmark Policy Report

This report aggregates committed Qwen policy summaries across official GSM8K and MATH subsets.

| benchmark   | split   | policy                                  |   accuracy |   avg_cost |   p95_latency |   cost_vs_fixed_1024 |   n |
|:------------|:--------|:----------------------------------------|-----------:|-----------:|--------------:|---------------------:|----:|
| gsm8k       | test20  | fixed_budget_0                          |       0.9  |   0.000242 |       19.6518 |             0.446136 |  20 |
| gsm8k       | test20  | fixed_budget_256                        |       0.95 |   0.000344 |       16.7214 |             0.634985 |  20 |
| gsm8k       | test20  | fixed_budget_1024                       |       0.95 |   0.000542 |       23.3898 |             1        |  20 |
| gsm8k       | test20  | oracle_lowest_cost_correct              |       1    |   0.000288 |       10.3245 |             0.531784 |  20 |
| gsm8k       | test20  | aggregate_utility_budget_256            |       0.95 |   0.000344 |       16.7214 |             0.634985 |  20 |
| gsm8k       | test20  | safe_learned_policy_fallback_budget_256 |       0.95 |   0.000344 |       16.7214 |             0.634985 |  20 |
| math        | test20  | fixed_budget_0                          |       0.5  |   0.000604 |       32.3479 |             0.59066  |  20 |
| math        | test20  | fixed_budget_256                        |       0.55 |   0.000876 |       54.6521 |             0.857057 |  20 |
| math        | test20  | fixed_budget_1024                       |       0.25 |   0.001022 |       49.1511 |             1        |  20 |
| math        | test20  | oracle_lowest_cost_correct              |       0.6  |   0.000611 |       36.2456 |             0.59827  |  20 |
| math        | test20  | aggregate_utility_budget_0              |       0.5  |   0.000604 |       32.3479 |             0.59066  |  20 |

Notes:

- GSM8K rows use held-out `test20` results.
- MATH rows use held-out `test20` results.
- `oracle_lowest_cost_correct` is an offline upper bound and is not deployable.
