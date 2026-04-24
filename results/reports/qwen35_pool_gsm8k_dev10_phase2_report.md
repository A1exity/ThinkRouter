# ThinkRouter Phase 2 Report

This report ranks the integrated baseline and Phase 2 router summaries by utility.

- Best overall policy: `aggregate_utility_qwen3-max-2026-01-23_0`
- Best overall utility: `0.916026`
- Best Phase 2 policy: `phase2_logreg_joint`
- Best Phase 2 utility: `0.876122`

| policy                                                        | policy_family            | router_name       | selected_model           | selected_model_tier   | selected_budget   |   accuracy |   avg_cost |   avg_latency |   avg_route_confidence |   fallback_rate |   utility |
|:--------------------------------------------------------------|:-------------------------|:------------------|:-------------------------|:----------------------|:------------------|-----------:|-----------:|--------------:|-----------------------:|----------------:|----------:|
| aggregate_utility_qwen3-max-2026-01-23_0                      | joint_aggregate_utility  | nan               | qwen3-max-2026-01-23     | strong                | 0                 |        1   |   0.000798 |       3.9992  |             nan        |           nan   |  0.916026 |
| aggregate_utility_safe_fallback_qwen3-max-2026-01-23_0_to_256 | joint_safe_fallback      | nan               | qwen3-max-2026-01-23     | strong                | 0                 |        1   |   0.000798 |       3.9992  |             nan        |           nan   |  0.916026 |
| fixed_model_budget_qwen3-max-2026-01-23_0                     | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                | 0                 |        1   |   0.000798 |       3.9992  |             nan        |           nan   |  0.916026 |
| phase2_logreg_joint                                           | phase2_logreg_joint      | logreg_joint      | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |        1   |   0.000234 |       6.1353  |               0.911575 |             0   |  0.876122 |
| phase2_mlp_factorized                                         | phase2_mlp_factorized    | mlp_factorized    | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |        1   |   0.000234 |       6.1353  |               0.947759 |             0   |  0.876122 |
| phase2_threshold                                              | phase2_threshold         | threshold         | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |        1   |   0.000234 |       6.1353  |               0.675    |             0   |  0.876122 |
| phase2_uncertainty_aware                                      | phase2_uncertainty_aware | uncertainty_aware | qwen3.5-flash-2026-02-23 | cheap                 | mixed             |        1   |   0.000234 |       6.1353  |               0.947759 |             0.1 |  0.876122 |
| model_only_best_qwen3-max-2026-01-23                          | model_only               | nan               | qwen3-max-2026-01-23     | strong                | nan               |        1   |   0.001093 |       6.13911 |             nan        |           nan   |  0.871754 |
| fixed_model_budget_qwen3-max-2026-01-23_256                   | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                | 256               |        1   |   0.001094 |       6.15997 |             nan        |           nan   |  0.871331 |
| fixed_model_budget_qwen3.5-flash-2026-02-23_256               | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 | 256               |        1   |   0.000312 |       7.90627 |             nan        |           nan   |  0.840314 |
| fixed_model_budget_qwen3-max-2026-01-23_1024                  | fixed_model_budget       | nan               | qwen3-max-2026-01-23     | strong                | 1024              |        1   |   0.001386 |       8.25816 |             nan        |           nan   |  0.827905 |
| fixed_model_budget_qwen3.5-flash-2026-02-23_0                 | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 | 0                 |        1   |   0.000331 |       8.52217 |             nan        |           nan   |  0.8279   |
| budget_only_best_256                                          | budget_only              | nan               | nan                      | nan                   | 256               |        1   |   0.001018 |      12.6118  |             nan        |           nan   |  0.742675 |
| fixed_model_budget_qwen3.5-flash-2026-02-23_1024              | fixed_model_budget       | nan               | qwen3.5-flash-2026-02-23 | cheap                 | 1024              |        1   |   0.000491 |      12.9922  |             nan        |           nan   |  0.737702 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_256                | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   | 256               |        1   |   0.001648 |      23.7691  |             nan        |           nan   |  0.516379 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_1024               | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   | 1024              |        0.9 |   0.0016   |      23.1742  |             nan        |           nan   |  0.428517 |
| fixed_model_budget_qwen3.5-plus-2026-02-15_0                  | fixed_model_budget       | nan               | qwen3.5-plus-2026-02-15  | mid                   | 0                 |        0.9 |   0.001828 |      26.5434  |             nan        |           nan   |  0.359991 |

