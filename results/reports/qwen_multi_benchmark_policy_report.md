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
| math        | dev20   | fixed_budget_0                          |       0.8  |   0.001702 |       58.9353 |             2.18563  |  20 |
| math        | dev20   | fixed_budget_256                        |       0.7  |   0.000649 |       42.5375 |             0.833717 |  20 |
| math        | dev20   | fixed_budget_1024                       |       0.35 |   0.000779 |       32.6618 |             1        |  20 |
| math        | dev20   | oracle_lowest_cost_correct              |       0.8  |   0.000486 |       29.6516 |             0.624164 |  20 |
| math        | dev20   | aggregate_utility_budget_256            |       0.7  |   0.000649 |       42.5375 |             0.833717 |  20 |

Notes:

- GSM8K rows use held-out `test20` results.
- MATH rows currently use `dev20` results; this is not yet a held-out MATH test result.
- `oracle_lowest_cost_correct` is an offline upper bound and is not deployable.
