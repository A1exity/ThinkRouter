# GSM8K Qwen Report

This report aggregates the committed Qwen GSM8K dev/test policy results.

| split   | source          | policy                                  |   accuracy |   avg_cost |   p95_latency |   cost_vs_fixed_1024 |   accuracy_delta_vs_fixed_1024 |   n |
|:--------|:----------------|:----------------------------------------|-----------:|-----------:|--------------:|---------------------:|-------------------------------:|----:|
| dev20   | fixed_or_oracle | fixed_budget_0                          |       0.95 |   0.000297 |      14.3337  |             0.53698  |                           0.05 |  20 |
| dev20   | fixed_or_oracle | fixed_budget_256                        |       0.95 |   0.000319 |      11.37    |             0.577308 |                           0.05 |  20 |
| dev20   | fixed_or_oracle | fixed_budget_1024                       |       0.9  |   0.000553 |      28.8787  |             1        |                           0    |  20 |
| dev20   | fixed_or_oracle | oracle_lowest_cost_correct              |       0.95 |   0.000233 |       8.39621 |             0.42225  |                           0.05 |  20 |
| dev20   | fixed_or_oracle | aggregate_utility_budget_256            |       0.95 |   0.000319 |      11.37    |             0.577308 |                           0.05 |  20 |
| test20  | fixed_or_oracle | fixed_budget_0                          |       0.9  |   0.000242 |      19.6518  |             0.446136 |                          -0.05 |  20 |
| test20  | fixed_or_oracle | fixed_budget_256                        |       0.95 |   0.000344 |      16.7214  |             0.634985 |                           0    |  20 |
| test20  | fixed_or_oracle | fixed_budget_1024                       |       0.95 |   0.000542 |      23.3898  |             1        |                           0    |  20 |
| test20  | fixed_or_oracle | oracle_lowest_cost_correct              |       1    |   0.000288 |      10.3245  |             0.531784 |                           0.05 |  20 |
| test20  | fixed_or_oracle | aggregate_utility_budget_256            |       0.95 |   0.000344 |      16.7214  |             0.634985 |                           0    |  20 |
| dev20   | raw_learned     | learned_policy                          |       0.95 |   0.000373 |      14.4546  |             0.67504  |                           0.05 |  20 |
| dev20   | safe_train_only | safe_learned_policy_fallback_budget_0   |       0.95 |   0.000297 |      14.3337  |             0.53698  |                           0.05 |  20 |
| test20  | raw_learned     | learned_policy                          |       0.9  |   0.000284 |      19.6518  |             0.523656 |                          -0.05 |  20 |
| test20  | safe_train_only | safe_learned_policy_fallback_budget_0   |       0.9  |   0.000242 |      19.6518  |             0.446136 |                          -0.05 |  20 |
| test20  | dev_calibrated  | safe_learned_policy_fallback_budget_256 |       0.95 |   0.000344 |      16.7214  |             0.634985 |                           0    |  20 |

Interpretation:

- `fixed_budget_1024` is the high-budget reference for cost and accuracy deltas.
- `oracle_lowest_cost_correct` is an offline upper bound and is not deployable.
- `safe_learned_policy_fallback_budget_256` is trained on train60, calibrated on dev20, and evaluated on held-out test20.
